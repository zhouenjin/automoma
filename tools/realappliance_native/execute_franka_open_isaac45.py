#!/usr/bin/env python3
"""Physically open one RealAppliance asset with an Isaac Sim Franka.

This is the compatibility executor for the 4090 host's installed Isaac Sim 4.5.
It consumes articulation/contact geometry on the command line and never branches
on an asset id.  Only Franka joints are commanded; the appliance articulation is
read back from PhysX.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def csv_vector(value: str, size: int) -> np.ndarray:
    result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if result.shape != (size,):
        raise argparse.ArgumentTypeError(f"expected {size} comma-separated values")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--asset-usd", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--target-dof-name", required=True)
    result.add_argument(
        "--moving-body-path",
        required=True,
        help="Source path below /World, e.g. /World/part_03",
    )
    result.add_argument(
        "--joint-type", choices=("revolute", "prismatic"), required=True
    )
    result.add_argument("--joint-axis", required=True)
    result.add_argument("--joint-pivot", required=True)
    result.add_argument("--contact-center", required=True)
    result.add_argument("--approach-direction", required=True)
    result.add_argument("--closing-direction", required=True)
    result.add_argument("--asset-translation", default="0.55,0.0,0.55")
    result.add_argument("--robot-translation", default="0.10,-0.35,0.0")
    result.add_argument("--precontact-distance-m", type=float, default=0.12)
    result.add_argument("--contact-offset-m", type=float, default=-0.002)
    result.add_argument("--target-fraction", type=float, default=0.80)
    result.add_argument("--acceptance-fraction", type=float, default=0.70)
    result.add_argument("--friction-multiplier", type=float, default=5.0)
    result.add_argument("--finger-effort-multiplier", type=float, default=2.0)
    result.add_argument("--render-stride", type=int, default=3)
    result.add_argument("--camera-distance-multiplier", type=float, default=2.0)
    result.add_argument("--passive-baseline-steps", type=int, default=120)
    result.add_argument("--max-passive-drift-fraction", type=float, default=0.025)
    result.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    return result


ARGS = parser().parse_args()
ARGS.output_dir.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402


SIMULATION_APP = SimulationApp(
    {
        "headless": bool(ARGS.headless),
        "width": 960,
        "height": 720,
        "renderer": "RayTracedLighting",
    }
)

import cv2  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.sensors import RigidContactView  # noqa: E402
from isaacsim.core.prims import Articulation, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.robot.manipulators.examples.franka import Franka  # noqa: E402
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (  # noqa: E402
    RMPFlowController,
)
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import UsdPhysics, UsdShade  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-9:
        raise ValueError("zero vector")
    return value / norm


def parse_inputs() -> dict[str, np.ndarray]:
    values = {
        "joint_axis": csv_vector(ARGS.joint_axis, 3),
        "joint_pivot": csv_vector(ARGS.joint_pivot, 3),
        "contact_center": csv_vector(ARGS.contact_center, 3),
        "approach_direction": csv_vector(ARGS.approach_direction, 3),
        "closing_direction": csv_vector(ARGS.closing_direction, 3),
        "asset_translation": csv_vector(ARGS.asset_translation, 3),
        "robot_translation": csv_vector(ARGS.robot_translation, 3),
    }
    values["joint_axis"] = unit(values["joint_axis"])
    values["approach_direction"] = unit(values["approach_direction"])
    closing = values["closing_direction"]
    closing = closing - values["approach_direction"] * float(
        np.dot(closing, values["approach_direction"])
    )
    values["closing_direction"] = unit(closing)
    return values


def contact_orientation_wxyz(
    approach_world: np.ndarray, closing_world: np.ndarray
) -> np.ndarray:
    """Construct a Panda hand frame: +Z approaches, +Y is the closing axis."""

    z_axis = unit(approach_world)
    y_axis = unit(closing_world - z_axis * float(np.dot(closing_world, z_axis)))
    x_axis = unit(np.cross(y_axis, z_axis))
    y_axis = unit(np.cross(z_axis, x_axis))
    matrix = np.column_stack((x_axis, y_axis, z_axis))
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.roll(xyzw, 1)


def rotate_about_axis(
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec(axis * delta)
    moved = pivot + rotation.apply(position - pivot)
    current = Rotation.from_quat(np.roll(orientation_wxyz, -1))
    moved_orientation = np.roll((rotation * current).as_quat(), 1)
    return moved, moved_orientation


def runtime_path(source_path: str) -> str:
    if not source_path.startswith("/World/"):
        raise ValueError(f"source path must start with /World/: {source_path}")
    return "/World/Appliance/" + source_path[len("/World/") :]


def apply_contact_material(
    stage: Any, target_path: str, multiplier: float
) -> dict[str, Any]:
    material = UsdShade.Material.Define(stage, "/World/NativeOpenContactMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    static_friction = 0.8 * float(multiplier)
    dynamic_friction = 0.6 * float(multiplier)
    physics.CreateStaticFrictionAttr().Set(static_friction)
    physics.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics.CreateRestitutionAttr().Set(0.0)
    candidates = [
        "/World/Franka/panda_leftfinger",
        "/World/Franka/panda_rightfinger",
        target_path,
    ]
    bound: list[str] = []
    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics",
        )
        bound.append(path)
    return {
        "material_path": str(material.GetPath()),
        "bound_prim_paths": bound,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
    }


def rgba_rgb(camera: Camera) -> np.ndarray:
    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError(f"camera returned no image: {camera.prim_path}")
    image = np.asarray(rgba)
    if image.size == 0 or image.ndim != 3 or image.shape[-1] < 3:
        raise RuntimeError(
            f"camera returned an uninitialized image {image.shape}: {camera.prim_path}"
        )
    if image.dtype != np.uint8:
        scale = 255.0 if float(np.max(image, initial=0.0)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image[..., :3]


def compose_frame(
    images: dict[str, np.ndarray], phase: str, progress: float
) -> np.ndarray:
    labelled: list[np.ndarray] = []
    for name in ("overview", "side", "contact"):
        image = cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)
        cv2.putText(
            image,
            name.upper(),
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        labelled.append(image)
    frame = np.concatenate(labelled, axis=1)
    cv2.putText(
        frame,
        f"PHASE: {phase}   PHYSICAL OPEN: {100.0 * progress:.1f}%",
        (18, frame.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 230, 255),
        2,
    )
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    if not frames:
        raise RuntimeError("cannot write an empty video")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    values = parse_inputs()
    world = World(
        stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0
    )
    world.scene.add_default_ground_plane()

    add_reference_to_stage(str(ARGS.asset_usd.resolve()), "/World/Appliance")
    appliance_xform = SingleXFormPrim(
        "/World/Appliance", name="native_open_appliance_xform"
    )
    appliance_xform.set_world_pose(position=values["asset_translation"])
    franka = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="native_open_franka",
            position=values["robot_translation"],
        )
    )

    target_body_path = runtime_path(ARGS.moving_body_path)
    cameras = {
        "overview": Camera(
            "/World/NativeOverviewCamera", frequency=30, resolution=(640, 480)
        ),
        "side": Camera("/World/NativeSideCamera", frequency=30, resolution=(640, 480)),
        "contact": Camera(
            "/World/NativeContactCamera", frequency=30, resolution=(640, 480)
        ),
    }
    target_contacts = RigidContactView(
        prim_paths_expr="/World/Franka/*",
        filter_paths_expr=[target_body_path],
        name="native_franka_target_contacts",
        prepare_contact_sensors=True,
        max_contact_count=8192,
    )

    world.reset()
    appliance = Articulation("/World/Appliance")
    appliance.initialize()
    target_contacts.initialize()
    target_matches = [
        index
        for index, name in enumerate(appliance.dof_names)
        if name == ARGS.target_dof_name
    ]
    if len(target_matches) != 1:
        raise RuntimeError(
            f"expected one target DOF {ARGS.target_dof_name!r}; got {appliance.dof_names}"
        )
    target_index = target_matches[0]
    limits = np.asarray(appliance.get_dof_limits(), dtype=np.float64).reshape(-1, 2)
    lower, upper = map(float, limits[target_index])
    initial_joint = float(
        np.asarray(appliance.get_joint_positions(), dtype=np.float64).reshape(-1)[
            target_index
        ]
    )
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise RuntimeError(f"invalid target joint limits: {(lower, upper)}")
    if abs(upper - initial_joint) >= abs(lower - initial_joint):
        goal_joint = initial_joint + float(ARGS.target_fraction) * (
            upper - initial_joint
        )
        opening_sign = 1.0
    else:
        goal_joint = initial_joint + float(ARGS.target_fraction) * (
            lower - initial_joint
        )
        opening_sign = -1.0
    joint_range = upper - lower

    friction = apply_contact_material(
        world.stage, target_body_path, ARGS.friction_multiplier
    )
    controller = RMPFlowController("native_open_rmpflow", robot_articulation=franka)
    robot_controller = franka.get_articulation_controller()
    finger_indices = [
        index for index, name in enumerate(franka.dof_names) if "finger_joint" in name
    ]
    if len(finger_indices) == 2:
        baseline_effort = np.asarray(
            robot_controller.get_max_efforts(), dtype=np.float32
        ).reshape(-1)[finger_indices]
        robot_controller.set_max_efforts(
            baseline_effort * float(ARGS.finger_effort_multiplier),
            joint_indices=finger_indices,
        )

    contact_source = values["contact_center"] + values["approach_direction"] * float(
        ARGS.contact_offset_m
    )
    contact_world = values["asset_translation"] + contact_source
    pivot_world = values["asset_translation"] + values["joint_pivot"]
    orientation = contact_orientation_wxyz(
        values["approach_direction"], values["closing_direction"]
    )
    precontact_world = contact_world - values["approach_direction"] * float(
        ARGS.precontact_distance_m
    )

    camera_multiplier = float(ARGS.camera_distance_multiplier)
    camera_specs = {
        "overview": (np.array([-1.2, -1.8, 1.8]), np.array([0.35, -0.05, 0.65])),
        "side": (np.array([1.4, -1.2, 1.15]), contact_world),
        "contact": (contact_world + np.array([-0.65, -0.75, 0.45]), contact_world),
    }
    for name, camera in cameras.items():
        camera.initialize()
        eye, target = camera_specs[name]
        eye = target + (eye - target) * camera_multiplier
        set_camera_view(eye=eye, target=target, camera_prim_path=camera.prim_path)

    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    step_count = 0

    def measured_progress() -> tuple[float, float]:
        position = float(
            np.asarray(appliance.get_joint_positions(), dtype=np.float64).reshape(-1)[
                target_index
            ]
        )
        fraction = max(
            0.0, min(1.0, opening_sign * (position - initial_joint) / joint_range)
        )
        return position, fraction

    def target_contact_force() -> float:
        matrix = target_contacts.get_contact_force_matrix(dt=1.0 / 120.0)
        if matrix is None:
            return 0.0
        array = np.asarray(matrix, dtype=np.float64)
        return 0.0 if not array.size else float(np.max(np.linalg.norm(array, axis=-1)))

    def record_step(phase: str, render: bool) -> None:
        nonlocal step_count
        measured_joint, progress = measured_progress()
        force = target_contact_force()
        trace.append(
            {
                "step": step_count,
                "phase": phase,
                "target_joint_position": measured_joint,
                "range_fraction": progress,
                "target_contact_force_n": force,
                "object_target_was_written": False,
                "attachment_active": False,
            }
        )
        if render:
            try:
                frames.append(
                    compose_frame(
                        {name: rgba_rgb(camera) for name, camera in cameras.items()},
                        phase,
                        progress,
                    )
                )
            except RuntimeError:
                # RTX cameras may need several rendered frames after initialize().
                pass
        step_count += 1

    def step_pose(
        phase: str,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        steps: int,
        gripper_width: float | None = None,
    ) -> None:
        nonlocal step_count
        for _ in range(steps):
            action = controller.forward(position, quaternion_wxyz)
            robot_controller.apply_action(action)
            if gripper_width is not None:
                franka.gripper.apply_action(
                    ArticulationAction(joint_positions=[gripper_width, gripper_width])
                )
            render = step_count % int(ARGS.render_stride) == 0
            world.step(render=render)
            record_step(phase, render)

    for _ in range(int(ARGS.passive_baseline_steps)):
        render = step_count % int(ARGS.render_stride) == 0
        world.step(render=render)
        record_step("PASSIVE_BASELINE", render)
    _, passive_baseline_drift = measured_progress()

    step_pose("PRECONTACT", precontact_world, orientation, 300, 0.04)
    step_pose("CONTACT", contact_world, orientation, 180, 0.04)
    step_pose("CLOSE", contact_world, orientation, 180, 0.0)

    reference_joint = initial_joint
    no_progress_steps = 0
    best_progress = 0.0
    for _ in range(1600):
        measured_joint, progress = measured_progress()
        if progress > best_progress + 1.0e-4:
            best_progress = progress
            no_progress_steps = 0
        else:
            no_progress_steps += 1
        if progress >= float(ARGS.target_fraction):
            break
        maximum_lead = 0.03 * joint_range
        measured_directed = opening_sign * (measured_joint - initial_joint)
        reference_directed = opening_sign * (reference_joint - initial_joint)
        if reference_directed - measured_directed <= maximum_lead:
            reference_joint += opening_sign * 0.0015 * joint_range
        reference_joint = min(max(reference_joint, lower), upper)
        if ARGS.joint_type == "revolute":
            target_position, target_orientation = rotate_about_axis(
                contact_world,
                orientation,
                pivot_world,
                values["joint_axis"],
                reference_joint - initial_joint,
            )
        else:
            target_position = contact_world + values["joint_axis"] * (
                reference_joint - initial_joint
            )
            target_orientation = orientation
        step_pose("OPEN", target_position, target_orientation, 1, 0.0)
        if no_progress_steps > 420:
            break

    step_pose(
        "HOLD",
        target_position if "target_position" in locals() else contact_world,
        target_orientation if "target_orientation" in locals() else orientation,
        90,
        0.0,
    )
    final_joint, final_fraction = measured_progress()
    maximum_fraction = max(row["range_fraction"] for row in trace)
    contact_steps = sum(row["target_contact_force_n"] > 0.1 for row in trace)
    open_contact_steps = sum(
        row["phase"] == "OPEN" and row["target_contact_force_n"] > 0.1 for row in trace
    )
    passive_baseline_failed = passive_baseline_drift > float(
        ARGS.max_passive_drift_fraction
    )
    result = {
        "schema_version": 1,
        "executor": "automoma_native_franka_isaac45",
        "asset_usd": str(ARGS.asset_usd.resolve()),
        "asset_specific_parameters": False,
        "target_dof_name": ARGS.target_dof_name,
        "joint_type": ARGS.joint_type,
        "joint_limits": [lower, upper],
        "initial_joint_position": initial_joint,
        "requested_goal_joint_position": goal_joint,
        "final_joint_position": final_joint,
        "final_range_fraction": final_fraction,
        "maximum_range_fraction": maximum_fraction,
        "planning_target_fraction": float(ARGS.target_fraction),
        "acceptance_fraction": float(ARGS.acceptance_fraction),
        "functional_open_success": (
            maximum_fraction >= float(ARGS.acceptance_fraction)
            and not passive_baseline_failed
        ),
        "contact_steps": contact_steps,
        "contact_evidence": contact_steps > 0,
        "open_contact_steps": open_contact_steps,
        "passive_baseline_drift_fraction": passive_baseline_drift,
        "max_passive_drift_fraction": float(ARGS.max_passive_drift_fraction),
        "passive_baseline_failed": passive_baseline_failed,
        "object_joint_commands_issued": 0,
        "attachment_used": False,
        "penetration_audit_complete": False,
        "dataset_ready": False,
        "friction": friction,
        "trace_steps": len(trace),
    }
    if result["functional_open_success"] and open_contact_steps > 0:
        result["provisional_physical_success"] = True
        result["failure_reason"] = "penetration_audit_pending"
    elif passive_baseline_failed:
        result["provisional_physical_success"] = False
        result["failure_reason"] = "passive_joint_drift_before_robot_contact"
    else:
        result["provisional_physical_success"] = False
        result["failure_reason"] = "insufficient_opening_or_contact"

    (ARGS.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (ARGS.output_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    write_video(ARGS.output_dir / "multiview.mp4", frames)
    print(json.dumps(result, indent=2), flush=True)


try:
    main()
except BaseException:  # Isaac Sim close may otherwise hide the original traceback.
    error = traceback.format_exc()
    print(error, flush=True)
    (ARGS.output_dir / "error.txt").write_text(error, encoding="utf-8")
finally:
    SIMULATION_APP.close()
