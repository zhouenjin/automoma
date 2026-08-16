#!/usr/bin/env python3
"""Physically execute one automatically selected G2 AKR opening path.

The target articulation remains passive for the complete episode.  Only G2
joint drives and the four physical swerve modules receive commands.  This
first executor gate starts from the planned grasp pose; transit from home and
release/retreat are intentionally left for the next gate and therefore every
result remains ``dataset_ready=false``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--g2-usd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--render-stride", type=int, default=4)
    parser.add_argument("--friction-multiplier", type=float, default=5.0)
    parser.add_argument("--gripper-effort-multiplier", type=float, default=2.0)
    parser.add_argument("--planner-effort-multiplier", type=float, default=1.5)
    parser.add_argument("--base-effort-multiplier", type=float, default=1.5)
    parser.add_argument("--base-position-gain", type=float, default=1.5)
    parser.add_argument("--base-yaw-gain", type=float, default=1.5)
    parser.add_argument("--maximum-penetration-m", type=float, default=0.003)
    parser.add_argument("--maximum-contact-force-n", type=float, default=250.0)
    parser.add_argument("--camera-distance-multiplier", type=float, default=2.0)
    parser.add_argument("--maximum-opening-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-interaction-lead-m", type=float, default=0.003)
    parser.add_argument("--maximum-probe-interaction-lead-m", type=float, default=0.015)
    parser.add_argument("--progress-probe-after-seconds", type=float, default=0.5)
    parser.add_argument("--probe-lead-ramp-seconds", type=float, default=1.0)
    parser.add_argument("--progress-epsilon-fraction", type=float, default=0.0005)
    parser.add_argument("--initial-contact-acquisition-seconds", type=float, default=1.0)
    parser.add_argument("--maximum-regrasp-attempts", type=int, default=2)
    parser.add_argument("--contact-loss-seconds", type=float, default=0.20)
    parser.add_argument("--regrasp-freeze-seconds", type=float, default=0.20)
    parser.add_argument("--regrasp-open-fraction", type=float, default=0.90)
    parser.add_argument("--regrasp-open-seconds", type=float, default=0.10)
    parser.add_argument("--regrasp-reposition-seconds", type=float, default=0.20)
    parser.add_argument("--regrasp-close-seconds", type=float, default=0.40)
    parser.add_argument("--regrasp-stable-contact-seconds", type=float, default=0.10)
    parser.add_argument("--regrasp-maximum-progress-rollback-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-base-tracking-error-m", type=float, default=0.008)
    parser.add_argument("--maximum-contact-base-tracking-error-m", type=float, default=0.015)
    parser.add_argument("--maximum-base-yaw-error-rad", type=float, default=0.02)
    parser.add_argument("--maximum-robot-joint-tracking-error-rad", type=float, default=0.04)
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.physics_hz <= 0:
    raise ValueError("--physics-hz must be positive")
if ARGS.render_stride <= 0:
    raise ValueError("--render-stride must be positive")
if ARGS.maximum_opening_seconds <= 0.0:
    raise ValueError("--maximum-opening-seconds must be positive")
if ARGS.maximum_interaction_lead_m < 0.0:
    raise ValueError("--maximum-interaction-lead-m must be non-negative")
if ARGS.maximum_probe_interaction_lead_m < ARGS.maximum_interaction_lead_m:
    raise ValueError("--maximum-probe-interaction-lead-m must not be smaller than the ordinary lead limit")
if ARGS.progress_probe_after_seconds < 0.0 or ARGS.probe_lead_ramp_seconds <= 0.0:
    raise ValueError("probe timing must be non-negative with a positive ramp duration")
if ARGS.progress_epsilon_fraction <= 0.0:
    raise ValueError("--progress-epsilon-fraction must be positive")
if ARGS.initial_contact_acquisition_seconds <= 0.0:
    raise ValueError("--initial-contact-acquisition-seconds must be positive")
if ARGS.maximum_regrasp_attempts < 0:
    raise ValueError("--maximum-regrasp-attempts must be non-negative")
if ARGS.contact_loss_seconds <= 0.0 or ARGS.regrasp_stable_contact_seconds <= 0.0:
    raise ValueError("contact loss and stable regrasp durations must be positive")
if not 0.0 <= ARGS.regrasp_open_fraction < 1.0:
    raise ValueError("--regrasp-open-fraction must be in [0, 1)")
if min(
    ARGS.regrasp_freeze_seconds,
    ARGS.regrasp_open_seconds,
    ARGS.regrasp_reposition_seconds,
    ARGS.regrasp_close_seconds,
) <= 0.0:
    raise ValueError("regrasp phase durations must be positive")
if not 0.0 <= ARGS.regrasp_maximum_progress_rollback_fraction <= 1.0:
    raise ValueError("--regrasp-maximum-progress-rollback-fraction must be in [0, 1]")
if not math.isfinite(ARGS.planner_effort_multiplier) or ARGS.planner_effort_multiplier <= 0.0:
    raise ValueError("--planner-effort-multiplier must be finite and positive")
if not math.isfinite(ARGS.base_effort_multiplier) or ARGS.base_effort_multiplier <= 0.0:
    raise ValueError("--base-effort-multiplier must be finite and positive")
if ARGS.base_position_gain <= 0.0 or ARGS.base_yaw_gain <= 0.0:
    raise ValueError("base tracking gains must be positive")
if ARGS.maximum_contact_base_tracking_error_m < ARGS.maximum_base_tracking_error_m:
    raise ValueError("contact-loaded base tolerance must not be smaller than the free-space tolerance")
APP = SimulationApp({"headless": True, "width": 640, "height": 480, "renderer": "RayTracedLighting"})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.sensors import RigidContactView  # noqa: E402
from isaacsim.core.prims import Articulation, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automoma.integrations.realappliance.contracts import (  # noqa: E402
    PhysicsRunPolicy,
    physical_open_failure_reasons,
)
from automoma.integrations.realappliance.g2_adapter import Hand  # noqa: E402
from automoma.integrations.realappliance.g2_runtime import (  # noqa: E402
    GRIPPER_JOINT_NAMES,
    GRIPPER_LINK_GROUPS,
    STEERING_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    actuator_groups,
    akr_parameter_for_articulation_progress,
    adaptive_interaction_lead_limit_m,
    gripper_targets,
    named_indices,
    normalize_angle,
    planar_tracking_twist,
    planner_robot_joint_names,
    positive_interaction_lead_m,
    swerve_inverse_kinematics,
    yaw_from_quaternion_wxyz,
)
from automoma.integrations.realappliance.transform_math import (  # noqa: E402
    invert_rigid,
    matrix_to_pose_wxyz,
    quaternion_transform,
    transform_point,
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _map_source_path(source_path: str) -> str:
    if source_path == "/World":
        return "/World/Appliance"
    if not source_path.startswith("/World/"):
        raise ValueError(f"source path is outside /World: {source_path}")
    return "/World/Appliance/" + source_path[len("/World/"):]


def _anchor_body_to_world(stage: Any, body_path: str, anchor_path: str) -> None:
    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"appliance anchor body is not rigid: {body_path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    anchor = UsdPhysics.FixedJoint.Define(stage, anchor_path)
    anchor.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*translation))
    anchor.CreateLocalRot0Attr(Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary())))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


def _scale_authored_joint_efforts(
    stage: Any, joint_names: Sequence[str], multiplier: float,
) -> list[dict[str, Any]]:
    """Scale the supplied robot's authored drive limits without hardcoded torques."""

    wanted = set(joint_names)
    records = []
    for prim in stage.Traverse():
        if prim.GetName() not in wanted:
            continue
        for drive_kind in ("angular", "linear"):
            attribute = prim.GetAttribute(f"drive:{drive_kind}:physics:maxForce")
            if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
                continue
            authored = attribute.Get()
            if authored is None or not math.isfinite(float(authored)):
                continue
            scaled = float(authored) * float(multiplier)
            attribute.Set(scaled)
            records.append(
                {
                    "joint_name": prim.GetName(),
                    "drive_kind": drive_kind,
                    "authored_max_force": float(authored),
                    "scaled_max_force": scaled,
                }
            )
    missing = wanted - {record["joint_name"] for record in records}
    if missing:
        raise RuntimeError(f"G2 planner joints are missing authored drive effort limits: {sorted(missing)}")
    return records


def _configure_appliance_collision(stage: Any) -> dict[str, Any]:
    changed = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Appliance/") or not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        source = mesh_api.GetApproximationAttr().Get()
        if not mesh_api:
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set("convexDecomposition")
        decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        decomposition.CreateMaxConvexHullsAttr().Set(64)
        decomposition.CreateVoxelResolutionAttr().Set(500000)
        decomposition.CreateErrorPercentageAttr().Set(1.0)
        decomposition.CreateHullVertexLimitAttr().Set(64)
        decomposition.CreateShrinkWrapAttr().Set(True)
        changed.append({"path": path, "source": source})
    if not changed:
        raise RuntimeError("no appliance collision mesh received the global convex-decomposition policy")
    return {"policy": "global_convex_decomposition", "collision_disabled": False, "meshes": changed}


def _bind_contact_friction(
    stage: Any,
    finger_paths: Sequence[str],
    handle_path: str,
    multiplier: float,
    *,
    bind_handle_body: bool,
) -> dict:
    material = UsdShade.Material.Define(stage, "/World/ExternalFirstContactMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    coefficient = 0.5 * float(multiplier)
    physics.CreateStaticFrictionAttr().Set(coefficient)
    physics.CreateDynamicFrictionAttr().Set(coefficient)
    physics.CreateRestitutionAttr().Set(0.0)
    bound = []
    requested_paths = [*finger_paths]
    if bind_handle_body:
        requested_paths.append(handle_path)
    for path in sorted(set(requested_paths)):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"friction target prim does not exist: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics"
        )
        bound.append(path)
    return {
        "multiplier": float(multiplier),
        "coefficient": coefficient,
        "bound_paths": bound,
        "handle_body_bound": bool(bind_handle_body),
        "scope_reason": (
            "separate_handle_rigid_body"
            if bind_handle_body
            else "handle_geometry_shares_door_body_so_whole_door_material_override_is_forbidden"
        ),
    }


def _contact_audit(view: RigidContactView | None, dt: float) -> dict[str, Any]:
    if view is None:
        return {"maximum_force_n": 0.0, "minimum_separation_m": None, "force_by_link_n": {}}
    matrix_raw = view.get_contact_force_matrix(dt=dt)
    matrix = np.asarray(matrix_raw, dtype=np.float64) if matrix_raw is not None else np.empty((0,))
    maximum_force = 0.0 if matrix.size == 0 else float(np.max(np.linalg.norm(matrix, axis=-1)))
    paths = list(getattr(view, "_prim_paths", []) or [])
    force_by_link = {}
    if matrix.size:
        magnitudes = np.linalg.norm(matrix, axis=-1)
        if magnitudes.ndim == 1:
            magnitudes = magnitudes[:, None]
        for index in range(magnitudes.shape[0]):
            name = paths[index] if index < len(paths) else f"sensor_{index}"
            force_by_link[name] = float(np.max(magnitudes[index]))
    raw = view.get_contact_force_data(dt=dt)
    minimum_separation = None
    if raw is not None and len(raw) >= 4:
        separation = np.asarray(raw[3], dtype=np.float64).reshape(-1)
        finite = separation[np.isfinite(separation)]
        if finite.size:
            minimum_separation = float(np.min(finite))
    return {
        "maximum_force_n": maximum_force,
        "minimum_separation_m": minimum_separation,
        "force_by_link_n": force_by_link,
    }


def _group_force(audit: dict[str, Any], names: Sequence[str]) -> float:
    leaves = set(names)
    return max(
        (float(force) for path, force in audit["force_by_link_n"].items() if Path(path).name in leaves),
        default=0.0,
    )


def _spatial_contact_audit(
    view: RigidContactView,
    dt: float,
    selected_points_world_m: np.ndarray,
    *,
    radius_m: float = 0.035,
) -> dict[str, Any]:
    raw = view.get_contact_force_data(dt=dt)
    if raw is None:
        return {"force_by_link_n": {}, "maximum_off_surface_force_n": 0.0, "contact_records": []}
    forces_raw, points_raw, _, _, counts_raw, starts_raw = raw
    forces = np.abs(np.asarray(forces_raw, dtype=np.float64).reshape(-1))
    points = np.asarray(points_raw, dtype=np.float64).reshape(-1, 3)
    counts = np.asarray(counts_raw, dtype=np.int64)
    starts = np.asarray(starts_raw, dtype=np.int64)
    if counts.ndim == 1:
        counts = counts[:, None]
        starts = starts[:, None]
    paths = list(getattr(view, "_prim_paths", []) or [])
    selected = np.asarray(selected_points_world_m, dtype=np.float64).reshape(-1, 3)
    near: dict[str, float] = {}
    maximum_off_surface = 0.0
    records = []
    for robot_index in range(counts.shape[0]):
        path = paths[robot_index] if robot_index < len(paths) else f"sensor_{robot_index}"
        for filter_index in range(counts.shape[1]):
            start = int(starts[robot_index, filter_index])
            count = int(counts[robot_index, filter_index])
            for contact_index in range(start, start + count):
                force = float(forces[contact_index])
                point = points[contact_index]
                distance = float(np.min(np.linalg.norm(selected - point[None, :], axis=1)))
                if distance <= radius_m:
                    near[path] = max(near.get(path, 0.0), force)
                else:
                    maximum_off_surface = max(maximum_off_surface, force)
                records.append(
                    {"robot_link": path, "force_n": force, "distance_to_selected_surface_m": distance}
                )
    records.sort(key=lambda item: -float(item["force_n"]))
    return {
        "force_by_link_n": near,
        "maximum_off_surface_force_n": maximum_off_surface,
        "contact_records": records[:12],
    }


def _sample_trajectory(trajectory: np.ndarray, progress: float) -> np.ndarray:
    coordinate = float(np.clip(progress, 0.0, 1.0)) * (len(trajectory) - 1)
    lower = int(math.floor(coordinate))
    upper = min(lower + 1, len(trajectory) - 1)
    fraction = coordinate - lower
    return (1.0 - fraction) * trajectory[lower] + fraction * trajectory[upper]


def _aligned_base_target(plan_start: np.ndarray, measured_start: np.ndarray, reference: np.ndarray) -> np.ndarray:
    yaw_offset = normalize_angle(float(measured_start[2] - plan_start[2]))
    cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    xy = measured_start[:2] + rotation @ (reference[:2] - plan_start[:2])
    return np.asarray((xy[0], xy[1], normalize_angle(measured_start[2] + reference[2] - plan_start[2])))


def _camera_poses(base: np.ndarray, contact: np.ndarray, multiplier: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    direction = contact[:2] - base[:2]
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1.0e-8 else np.asarray((1.0, 0.0))
    lateral = np.asarray((-direction[1], direction[0]))
    midpoint = 0.5 * (base + contact)
    midpoint[2] = max(midpoint[2], contact[2] - 0.25)
    definitions = {
        "overview": (midpoint + multiplier * np.asarray((*(-1.4 * direction + 1.2 * lateral), 1.4)), midpoint),
        "contact": (contact + multiplier * np.asarray((*(-0.45 * direction + 0.70 * lateral), 0.35)), contact),
        "side": (contact + multiplier * np.asarray((*(-0.25 * direction - 0.85 * lateral), 0.55)), contact),
    }
    return {name: (np.asarray(eye), np.asarray(target)) for name, (eye, target) in definitions.items()}


class Recorder:
    def __init__(self, root: Path, cameras: dict[str, Camera], fps: int):
        self.root = root
        self.cameras = cameras
        self.fps = fps
        self.frame_count = 0
        self.skipped_empty_frames = 0
        for name in (*cameras, "multiview"):
            (root / "frames" / name).mkdir(parents=True, exist_ok=True)

    def capture(self, telemetry: dict[str, Any]) -> bool:
        arrays = {}
        for name, camera in self.cameras.items():
            rgba = camera.get_rgba()
            if rgba is None:
                self.skipped_empty_frames += 1
                return False
            array = np.asarray(rgba)
            if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] < 3:
                self.skipped_empty_frames += 1
                return False
            if array.dtype != np.uint8:
                scale = 255.0 if float(np.max(array, initial=0.0)) <= 1.0 else 1.0
                array = np.clip(array * scale, 0, 255).astype(np.uint8)
            arrays[name] = array[..., :3]
        images = {}
        for name, array in arrays.items():
            image = Image.fromarray(array[..., :3], mode="RGB")
            image.save(self.root / "frames" / name / f"frame_{self.frame_count:06d}.png")
            images[name] = image
        canvas = Image.new("RGB", (1280, 960), (15, 15, 18))
        positions = {"overview": (0, 0), "contact": (640, 0), "side": (0, 480)}
        for name, position in positions.items():
            canvas.paste(images[name].resize((640, 480)), position)
        draw = ImageDraw.Draw(canvas)
        draw.text((660, 500), "G2 EXTERNAL-FIRST PHYSX AUDIT", fill=(100, 210, 255))
        y = 535
        for key in sorted(telemetry):
            draw.text((660, y), f"{key}: {telemetry[key]}"[:92], fill=(235, 235, 240))
            y += 22
        canvas.save(self.root / "frames" / "multiview" / f"frame_{self.frame_count:06d}.png")
        with (self.root / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"frame": self.frame_count, **telemetry}, sort_keys=True) + "\n")
        self.frame_count += 1
        return True

    def encode(self) -> dict[str, str]:
        outputs = {}
        for name in (*self.cameras, "multiview"):
            output = self.root / f"{name}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(self.fps),
                    "-i", str(self.root / "frames" / name / "frame_%06d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
                ],
                check=True,
            )
            outputs[name] = str(output)
        return outputs


def _open_limit(task: dict[str, Any]) -> float:
    initial = float(task["initial_position"])
    lower = float(task["lower_limit"])
    upper = float(task["upper_limit"])
    return upper if abs(upper - initial) >= abs(lower - initial) else lower


def _progress(task: dict[str, Any], joint_position: float) -> float:
    initial = float(task["initial_position"])
    denominator = _open_limit(task) - initial
    return float(np.clip((float(joint_position) - initial) / denominator, 0.0, 1.0))


def main() -> int:
    PhysicsRunPolicy().validate()
    if ARGS.physics_hz <= 0 or ARGS.render_stride <= 0:
        raise ValueError("physics_hz and render_stride must be positive")
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = json.loads((ARGS.attempt_dir / "attempt.json").read_text(encoding="utf-8"))
    tensors = torch.load(ARGS.attempt_dir / "trajectory.pt", map_location="cpu", weights_only=True)
    selected = ARGS.trajectory_index
    if selected is None:
        selected = attempt["planning"]["automatic_trajectory_selection"]["selected_trajectory_index"]
    if selected is None or not bool(tensors["success"][int(selected)]):
        raise ValueError("requested trajectory is not a valid automatically selected AKR path")
    trajectory = _as_numpy(tensors["trajectories"][int(selected)]).astype(np.float64)
    hand = Hand(attempt["hand"])
    task_record = attempt["target_articulation"]
    task = task_record["task"]
    dt = 1.0 / ARGS.physics_hz
    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": "physx_drive_from_planned_contact_pose",
        "dataset_ready": False,
        "dataset_ready_blockers": ["home_to_precontact_transit_pending", "release_and_retreat_pending"],
        "attempt_dir": str(ARGS.attempt_dir.resolve()),
        "trajectory_index": int(selected),
        "hand": hand.value,
        "policy": {
            "attachment_used": False,
            "target_collision_disabled": False,
            "target_joint_written_during_episode": False,
            "object_set_state_during_episode": False,
        },
    }

    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=1.0 / 30.0)
    world.scene.add_default_ground_plane(z_position=-0.04)
    dome = UsdLux.DomeLight.Define(world.stage, "/World/ExternalFirstDome")
    dome.CreateIntensityAttr(900.0)
    light = UsdLux.DistantLight.Define(world.stage, "/World/ExternalFirstKey")
    light.CreateIntensityAttr(2500.0)

    report = json.loads((ARGS.attempt_dir.parent / "report.json").read_text(encoding="utf-8"))
    appliance_usd = Path(report["source_usd"])
    add_reference_to_stage(str(ARGS.g2_usd.resolve()), "/World/G2")
    add_reference_to_stage(str(appliance_usd), "/World/Appliance")
    world_from_source = np.asarray(attempt["world_from_source"], dtype=np.float64)
    appliance_pose = matrix_to_pose_wxyz(world_from_source)
    SingleXFormPrim("/World/Appliance", name="external_first_appliance").set_world_pose(
        position=np.asarray(appliance_pose[:3]), orientation=np.asarray(appliance_pose[3:])
    )
    base_start = trajectory[0, :3]
    base_quaternion = np.asarray((math.cos(base_start[2] / 2.0), 0.0, 0.0, math.sin(base_start[2] / 2.0)))
    SingleXFormPrim("/World/G2", name="external_first_g2").set_world_pose(
        position=np.asarray((base_start[0], base_start[1], 0.0)), orientation=base_quaternion
    )
    collision_policy = _configure_appliance_collision(world.stage)
    anchor_body = _map_source_path(task_record["body0_path"])
    _anchor_body_to_world(world.stage, anchor_body, "/World/Appliance/ExternalFirstWorldAnchor")

    handle_path = _map_source_path(attempt["handle_body_source_path"])
    moving_paths = [_map_source_path(path) for path in attempt["collision_world"]["moving_cluster_paths"]]
    all_appliance_rigid_paths = sorted(
        str(prim.GetPath())
        for prim in world.stage.Traverse()
        if str(prim.GetPath()).startswith("/World/Appliance/") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    static_paths = [path for path in all_appliance_rigid_paths if path not in moving_paths]
    finger_names = GRIPPER_LINK_GROUPS[hand]["finger_a"] + GRIPPER_LINK_GROUPS[hand]["finger_b"]
    finger_paths = [f"/World/G2/{name}" for name in finger_names]
    target_moving_path = _map_source_path(attempt["target_moving_body_source_path"])
    friction = _bind_contact_friction(
        world.stage,
        finger_paths,
        handle_path,
        ARGS.friction_multiplier,
        bind_handle_body=handle_path != target_moving_path,
    )
    target_contact = RigidContactView(
        prim_paths_expr="/World/G2/*", filter_paths_expr=[handle_path], name="external_first_target_contact",
        prepare_contact_sensors=True, max_contact_count=8192,
    )
    static_contact = RigidContactView(
        prim_paths_expr="/World/G2/*", filter_paths_expr=static_paths, name="external_first_static_contact",
        prepare_contact_sensors=True, max_contact_count=16384,
    ) if static_paths else None

    contact_world = np.asarray(attempt["contact_center_world_m"], dtype=np.float64)
    camera_definitions = _camera_poses(base_start, contact_world, ARGS.camera_distance_multiplier)
    cameras = {
        name: Camera(
            prim_path=f"/World/ExternalFirst{name.title()}Camera", position=eye, frequency=30, resolution=(640, 480)
        )
        for name, (eye, _) in camera_definitions.items()
    }

    planner_names = planner_robot_joint_names(hand)
    planner_effort = _scale_authored_joint_efforts(
        world.stage, planner_names, ARGS.planner_effort_multiplier,
    )
    base_effort = _scale_authored_joint_efforts(
        world.stage,
        STEERING_JOINT_NAMES + WHEEL_JOINT_NAMES,
        ARGS.base_effort_multiplier,
    )

    world.reset()
    robot = Articulation("/World/G2/base_link")
    robot.initialize()
    appliance = Articulation("/World/Appliance")
    appliance.initialize()
    robot_names = list(robot.dof_names)
    planner_indices = named_indices(planner_names, robot_names)
    gripper_indices = named_indices(GRIPPER_JOINT_NAMES[hand], robot_names)
    steering_indices = named_indices(STEERING_JOINT_NAMES, robot_names)
    wheel_indices = named_indices(WHEEL_JOINT_NAMES, robot_names)
    for group in actuator_groups(hand, gripper_effort_multiplier=ARGS.gripper_effort_multiplier):
        indices = named_indices(group["names"], robot_names)
        robot.set_gains(
            kps=np.asarray(group["kp"], dtype=np.float32).reshape(1, -1),
            kds=np.asarray(group["kd"], dtype=np.float32).reshape(1, -1),
            joint_indices=indices,
        )
        if "max_effort" in group:
            robot.set_max_efforts(
                np.asarray(group["max_effort"], dtype=np.float32).reshape(1, -1), joint_indices=indices
            )
    robot.set_joint_positions(trajectory[0, 3:-1].reshape(1, -1), joint_indices=planner_indices)
    robot.set_joint_positions(gripper_targets(0.0).reshape(1, -1), joint_indices=gripper_indices)
    robot.set_joint_position_targets(trajectory[0, 3:-1].reshape(1, -1), joint_indices=planner_indices)
    robot.set_joint_position_targets(gripper_targets(0.0).reshape(1, -1), joint_indices=gripper_indices)
    robot.set_joint_velocity_targets(np.zeros((1, 4), dtype=np.float32), joint_indices=wheel_indices)

    target_leaf = Path(task_record["joint_path"]).name
    matching_target_indices = [index for index, name in enumerate(appliance.dof_names) if name == target_leaf]
    if len(matching_target_indices) != 1:
        raise RuntimeError(f"could not uniquely resolve target joint {target_leaf!r}: {appliance.dof_names}")
    target_index = matching_target_indices[0]
    initial_appliance = _as_numpy(appliance.get_joint_positions()).reshape(-1)
    initial_appliance[target_index] = float(task["initial_position"])
    appliance.set_joint_positions(initial_appliance.reshape(1, -1))
    target_contact.initialize()
    if static_contact is not None:
        static_contact.initialize()
    for name, camera in cameras.items():
        camera.initialize()
        camera.set_focal_length(8.0 if name == "overview" else 12.0)
        eye, target = camera_definitions[name]
        set_camera_view(eye=eye, target=target, camera_prim_path=camera.prim_path)
    recorder = Recorder(ARGS.output_dir, cameras, fps=max(1, ARGS.physics_hz // ARGS.render_stride))
    base_probe = SingleXFormPrim("/World/G2/base_link", name="external_first_base_probe")
    handle_probe = SingleXFormPrim(handle_path, name="external_first_handle_probe")
    initial_handle_position, initial_handle_quaternion = handle_probe.get_world_pose()
    world_from_handle_initial = quaternion_transform(initial_handle_position, initial_handle_quaternion)
    handle_from_selected_points = [
        transform_point(invert_rigid(world_from_handle_initial), point)
        for point in attempt["contact_points_world_m"]
    ]

    maximum_force = 0.0
    minimum_separation = 0.0
    measured_finger_contact = False
    contact_during_opening = False
    maximum_progress = 0.0
    maximum_off_surface_force = 0.0
    maximum_interaction_lead = 0.0
    maximum_contact_interaction_lead = 0.0
    maximum_probe_fraction = 0.0
    maximum_opening_finger_a_force = 0.0
    maximum_opening_finger_b_force = 0.0
    opening_selected_contact_steps = 0
    opening_bilateral_contact_steps = 0
    phase = "settle"
    step_count = 0
    control_telemetry: dict[str, Any] = {}

    def step_and_record() -> tuple[float, dict[str, Any]]:
        nonlocal step_count, maximum_force, minimum_separation, measured_finger_contact, maximum_progress
        nonlocal maximum_off_surface_force
        world.step(render=(step_count % ARGS.render_stride == 0))
        target_audit = _contact_audit(target_contact, dt)
        static_audit = _contact_audit(static_contact, dt)
        handle_position, handle_quaternion = handle_probe.get_world_pose()
        world_from_handle = quaternion_transform(handle_position, handle_quaternion)
        selected_points_world = np.asarray(
            [transform_point(world_from_handle, point) for point in handle_from_selected_points]
        )
        spatial_audit = _spatial_contact_audit(target_contact, dt, selected_points_world)
        finger_a_force = _group_force(spatial_audit, GRIPPER_LINK_GROUPS[hand]["finger_a"])
        finger_b_force = _group_force(spatial_audit, GRIPPER_LINK_GROUPS[hand]["finger_b"])
        measured_finger_contact = measured_finger_contact or max(finger_a_force, finger_b_force) >= 0.05
        maximum_off_surface_force = max(
            maximum_off_surface_force, float(spatial_audit["maximum_off_surface_force_n"])
        )
        force = max(target_audit["maximum_force_n"], static_audit["maximum_force_n"])
        maximum_force = max(maximum_force, force)
        separations = [
            value for value in (target_audit["minimum_separation_m"], static_audit["minimum_separation_m"])
            if value is not None
        ]
        if separations:
            minimum_separation = min(minimum_separation, min(separations))
        joint_position = float(_as_numpy(appliance.get_joint_positions()).reshape(-1)[target_index])
        progress = _progress(task, joint_position)
        maximum_progress = max(maximum_progress, progress)
        if step_count % ARGS.render_stride == 0:
            recorder.capture(
                {
                    "phase": phase,
                    "progress": round(progress, 4),
                    "finger_a_n": round(finger_a_force, 3),
                    "finger_b_n": round(finger_b_force, 3),
                    "maximum_force_n": round(maximum_force, 2),
                    "minimum_separation_m": round(minimum_separation, 5),
                    "off_surface_force_n": round(maximum_off_surface_force, 3),
                    **control_telemetry,
                }
            )
        step_count += 1
        return progress, {"target": target_audit, "spatial_target": spatial_audit, "static": static_audit}

    for _ in range(120):
        robot.set_joint_position_targets(trajectory[0, 3:-1].reshape(1, -1), joint_indices=planner_indices)
        robot.set_joint_velocity_targets(np.zeros((1, 4), dtype=np.float32), joint_indices=wheel_indices)
        step_and_record()

    phase = "close"
    for close_step in range(180):
        fraction = (close_step + 1) / 180.0
        robot.set_joint_position_targets(gripper_targets(fraction).reshape(1, -1), joint_indices=gripper_indices)
        step_and_record()
    for _ in range(60):
        robot.set_joint_position_targets(gripper_targets(1.0).reshape(1, -1), joint_indices=gripper_indices)
        step_and_record()

    phase = "opening"
    initial_position_raw, initial_quaternion_raw = base_probe.get_world_pose()
    measured_base_start = np.asarray(
        (*_as_numpy(initial_position_raw).reshape(3)[:2], yaw_from_quaternion_wxyz(initial_quaternion_raw))
    )
    base_plan_start = trajectory[0, :3].copy()
    base_path = float(np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)))
    yaw_path = float(np.sum(np.abs(np.diff(trajectory[:, 2]))))
    robot_path = float(np.max(np.sum(np.abs(np.diff(trajectory[:, 3:-1], axis=0)), axis=0)))
    duration = max(4.0, 1.5 * base_path / 0.15, 1.5 * yaw_path / 0.2, 1.5 * robot_path / 0.5)
    duration = min(duration, float(ARGS.maximum_opening_seconds))
    nominal_steps = max(1, int(math.ceil(duration * ARGS.physics_hz)))
    maximum_total_opening_steps = max(
        nominal_steps,
        int(math.ceil(float(ARGS.maximum_opening_seconds) * ARGS.physics_hz)),
    )
    reference_progress = 0.0
    stalled_steps = 0
    opening_start_joint = float(_as_numpy(appliance.get_joint_positions()).reshape(-1)[target_index])
    opening_start_progress = _progress(task, opening_start_joint)
    maximum_opening_progress = opening_start_progress
    best_opening_progress = opening_start_progress
    no_object_progress_steps = 0
    latest_audit: dict[str, Any] = {}
    last_blocking_gates: list[str] = []
    opening_termination = "time_budget_exhausted"
    regrasp_phase: str | None = None
    regrasp_phase_step = 0
    regrasp_attempts = 0
    regrasp_recoveries = 0
    selected_contact_seen = False
    selected_contact_missing_steps = 0
    regrasp_stable_contact_steps = 0
    regrasp_trigger_progress_fractions: list[float] = []
    regrasp_recovery_progress_fractions: list[float] = []
    regrasp_trigger_progress: float | None = None
    regrasp_hold_robot_joints: np.ndarray | None = None
    regrasp_hold_base: np.ndarray | None = None
    contact_loss_steps = max(1, int(math.ceil(ARGS.contact_loss_seconds * ARGS.physics_hz)))
    regrasp_freeze_steps = max(1, int(math.ceil(ARGS.regrasp_freeze_seconds * ARGS.physics_hz)))
    regrasp_open_steps = max(1, int(math.ceil(ARGS.regrasp_open_seconds * ARGS.physics_hz)))
    regrasp_reposition_steps = max(1, int(math.ceil(ARGS.regrasp_reposition_seconds * ARGS.physics_hz)))
    regrasp_close_steps = max(1, int(math.ceil(ARGS.regrasp_close_seconds * ARGS.physics_hz)))
    stable_regrasp_steps = max(1, int(math.ceil(ARGS.regrasp_stable_contact_seconds * ARGS.physics_hz)))
    for opening_step in range(maximum_total_opening_steps):
        reference = _sample_trajectory(trajectory, reference_progress)
        position_raw, quaternion_raw = base_probe.get_world_pose()
        measured_base = np.asarray(
            (*_as_numpy(position_raw).reshape(3)[:2], yaw_from_quaternion_wxyz(quaternion_raw))
        )
        if regrasp_phase is not None:
            if regrasp_hold_base is None or regrasp_hold_robot_joints is None:
                raise RuntimeError("regrasp hold state was not initialized")
            base_target = regrasp_hold_base
            planner_target = regrasp_hold_robot_joints
        else:
            base_target = _aligned_base_target(base_plan_start, measured_base_start, reference[:3])
            planner_target = reference[3:-1]
        twist, base_error, yaw_error = planar_tracking_twist(
            measured_base,
            base_target,
            position_gain=ARGS.base_position_gain,
            yaw_gain=ARGS.base_yaw_gain,
        )
        current_steering = _as_numpy(robot.get_joint_positions(joint_indices=steering_indices)).reshape(-1)
        steering, wheel = swerve_inverse_kinematics(twist, current_steering_angles_rad=current_steering)
        steering_error = max(
            abs(normalize_angle(target - current)) for target, current in zip(steering, current_steering)
        )
        wheel *= max(0.0, math.cos(min(math.pi / 2.0, steering_error)))
        robot.set_joint_position_targets(steering.reshape(1, -1), joint_indices=steering_indices)
        robot.set_joint_velocity_targets(wheel.reshape(1, -1), joint_indices=wheel_indices)
        robot.set_joint_position_targets(planner_target.reshape(1, -1), joint_indices=planner_indices)
        gripper_fraction = 1.0
        if regrasp_phase == "freeze":
            phase = "regrasp_freeze"
        elif regrasp_phase == "open":
            local_fraction = min(1.0, (regrasp_phase_step + 1) / regrasp_open_steps)
            gripper_fraction = 1.0 + local_fraction * (ARGS.regrasp_open_fraction - 1.0)
            phase = "regrasp_open"
        elif regrasp_phase == "reposition":
            gripper_fraction = ARGS.regrasp_open_fraction
            phase = "regrasp_reposition"
        elif regrasp_phase == "close":
            local_fraction = min(1.0, (regrasp_phase_step + 1) / regrasp_close_steps)
            gripper_fraction = ARGS.regrasp_open_fraction + local_fraction * (1.0 - ARGS.regrasp_open_fraction)
            phase = "regrasp_close"
        else:
            phase = "opening"
        robot.set_joint_position_targets(
            gripper_targets(gripper_fraction).reshape(1, -1), joint_indices=gripper_indices
        )
        measured_progress, latest_audit = step_and_record()
        maximum_opening_progress = max(maximum_opening_progress, measured_progress)
        target_audit = latest_audit["spatial_target"]
        finger_a_force = _group_force(target_audit, GRIPPER_LINK_GROUPS[hand]["finger_a"])
        finger_b_force = _group_force(target_audit, GRIPPER_LINK_GROUPS[hand]["finger_b"])
        maximum_opening_finger_a_force = max(maximum_opening_finger_a_force, finger_a_force)
        maximum_opening_finger_b_force = max(maximum_opening_finger_b_force, finger_b_force)
        if max(finger_a_force, finger_b_force) >= 0.05:
            contact_during_opening = True
        selected_contact_now = max(finger_a_force, finger_b_force) >= 0.05
        if regrasp_phase is None:
            if selected_contact_now:
                selected_contact_seen = True
                selected_contact_missing_steps = 0
            elif selected_contact_seen:
                selected_contact_missing_steps += 1
        opening_selected_contact_steps += int(selected_contact_now)
        opening_bilateral_contact_steps += int(min(finger_a_force, finger_b_force) >= 0.05)
        if maximum_opening_progress >= float(task["acceptance_fraction"]):
            opening_termination = "acceptance_reached"
            break
        akr_start = trajectory[0, -1]
        akr_goal = trajectory[-1, -1]
        expected_progress = float(task["planning_fraction"]) * (reference[-1] - akr_start) / (akr_goal - akr_start)
        reference_joint_position = float(task["initial_position"]) + expected_progress * (
            _open_limit(task) - float(task["initial_position"])
        )
        measured_joint_position = float(task["initial_position"]) + measured_progress * (
            _open_limit(task) - float(task["initial_position"])
        )
        interaction_lead_m = positive_interaction_lead_m(
            joint_kind=task["joint_kind"],
            joint_axis_world=task["axis"],
            joint_pivot_world_m=task["pivot"],
            contact_world_m=contact_world,
            opening_delta=_open_limit(task) - float(task["initial_position"]),
            reference_joint_position=reference_joint_position,
            measured_joint_position=measured_joint_position,
        )
        maximum_interaction_lead = max(maximum_interaction_lead, interaction_lead_m)
        if selected_contact_now:
            maximum_contact_interaction_lead = max(maximum_contact_interaction_lead, interaction_lead_m)
        measured_robot_joints = _as_numpy(robot.get_joint_positions(joint_indices=planner_indices)).reshape(-1)
        robot_joint_errors = np.abs(measured_robot_joints - planner_target)
        worst_robot_joint_index = int(np.argmax(robot_joint_errors))
        robot_joint_error = float(robot_joint_errors[worst_robot_joint_index])
        worst_robot_joint_name = planner_names[worst_robot_joint_index]
        active_base_error_limit_m = (
            ARGS.maximum_contact_base_tracking_error_m
            if selected_contact_now
            else ARGS.maximum_base_tracking_error_m
        )
        tracking_ready = bool(
            base_error <= active_base_error_limit_m
            and yaw_error <= ARGS.maximum_base_yaw_error_rad
            and robot_joint_error <= ARGS.maximum_robot_joint_tracking_error_rad
        )
        if measured_progress >= best_opening_progress + ARGS.progress_epsilon_fraction:
            best_opening_progress = measured_progress
            no_object_progress_steps = 0
        elif selected_contact_now and tracking_ready:
            no_object_progress_steps += 1
        else:
            no_object_progress_steps = 0
        active_lead_limit_m, probe_fraction = adaptive_interaction_lead_limit_m(
            no_progress_steps=no_object_progress_steps,
            physics_hz=ARGS.physics_hz,
            ordinary_limit_m=ARGS.maximum_interaction_lead_m,
            probe_limit_m=ARGS.maximum_probe_interaction_lead_m,
            probe_after_seconds=ARGS.progress_probe_after_seconds,
            ramp_seconds=ARGS.probe_lead_ramp_seconds,
        )
        maximum_probe_fraction = max(maximum_probe_fraction, probe_fraction)
        probe_enabled = bool(selected_contact_now and probe_fraction > 0.0)
        last_blocking_gates = []
        if base_error > active_base_error_limit_m:
            last_blocking_gates.append("base_tracking_error")
        if yaw_error > ARGS.maximum_base_yaw_error_rad:
            last_blocking_gates.append("base_yaw_tracking_error")
        if robot_joint_error > ARGS.maximum_robot_joint_tracking_error_rad:
            last_blocking_gates.append("robot_joint_tracking_error")
        if interaction_lead_m > active_lead_limit_m:
            last_blocking_gates.append("interaction_lead")
        if not selected_contact_now:
            last_blocking_gates.append("selected_surface_contact_missing")
        control_telemetry.update(
            {
                "reference_progress": round(reference_progress, 4),
                "expected_object_progress": round(expected_progress, 4),
                "interaction_lead_m": round(interaction_lead_m, 5),
                "base_error_m": round(base_error, 5),
                "active_base_error_limit_m": round(active_base_error_limit_m, 5),
                "base_target_xy_yaw": [round(float(value), 5) for value in base_target],
                "base_measured_xy_yaw": [round(float(value), 5) for value in measured_base],
                "base_command_body": [round(float(value), 5) for value in twist],
                "base_yaw_error_rad": round(yaw_error, 5),
                "robot_joint_error_rad": round(robot_joint_error, 5),
                "worst_robot_joint_name": worst_robot_joint_name,
                "tracking_ready": tracking_ready,
                "blocking_gates": last_blocking_gates,
                "probe_enabled": probe_enabled,
                "probe_fraction": round(probe_fraction, 4),
                "active_lead_limit_m": round(active_lead_limit_m, 5),
                "no_object_progress_steps": no_object_progress_steps,
                "gripper_command_fraction": round(gripper_fraction, 4),
                "regrasp_phase": regrasp_phase,
                "regrasp_attempt": regrasp_attempts,
                "regrasp_recoveries": regrasp_recoveries,
                "selected_contact_missing_steps": selected_contact_missing_steps,
            }
        )
        if regrasp_phase is not None:
            regrasp_phase_step += 1
            if (
                regrasp_trigger_progress is not None
                and measured_progress
                < regrasp_trigger_progress - ARGS.regrasp_maximum_progress_rollback_fraction
            ):
                opening_termination = "regrasp_progress_rollback_exceeded"
                last_blocking_gates = ["articulation_progress_rollback"]
                break
            if regrasp_phase in {"freeze", "close"}:
                if selected_contact_now:
                    regrasp_stable_contact_steps += 1
                else:
                    regrasp_stable_contact_steps = 0
            if regrasp_stable_contact_steps >= stable_regrasp_steps:
                regrasp_recoveries += 1
                regrasp_recovery_progress_fractions.append(measured_progress)
                reference_progress = akr_parameter_for_articulation_progress(
                    trajectory[:, -1],
                    planning_fraction=float(task["planning_fraction"]),
                    measured_progress_fraction=measured_progress,
                )
                regrasp_phase = None
                regrasp_phase_step = 0
                regrasp_stable_contact_steps = 0
                regrasp_trigger_progress = None
                regrasp_hold_robot_joints = None
                regrasp_hold_base = None
                selected_contact_seen = True
                selected_contact_missing_steps = 0
                stalled_steps = 0
                no_object_progress_steps = 0
                best_opening_progress = max(best_opening_progress, measured_progress)
                continue
            if regrasp_phase == "freeze" and regrasp_phase_step >= regrasp_freeze_steps:
                regrasp_phase = "open"
                regrasp_phase_step = 0
                regrasp_stable_contact_steps = 0
            elif regrasp_phase == "open" and regrasp_phase_step >= regrasp_open_steps:
                regrasp_phase = "reposition"
                regrasp_phase_step = 0
                regrasp_stable_contact_steps = 0
            elif regrasp_phase == "reposition" and regrasp_phase_step >= regrasp_reposition_steps:
                regrasp_phase = "close"
                regrasp_phase_step = 0
                regrasp_stable_contact_steps = 0
            elif regrasp_phase == "close" and regrasp_phase_step >= regrasp_close_steps:
                if regrasp_stable_contact_steps < stable_regrasp_steps:
                    if regrasp_attempts < ARGS.maximum_regrasp_attempts:
                        regrasp_attempts += 1
                        regrasp_trigger_progress_fractions.append(measured_progress)
                        regrasp_phase = "freeze"
                        regrasp_phase_step = 0
                        regrasp_stable_contact_steps = 0
                    else:
                        opening_termination = "regrasp_contact_not_recovered"
                        last_blocking_gates = ["selected_surface_contact_missing"]
                        break
            continue

        if selected_contact_missing_steps >= contact_loss_steps:
            if regrasp_attempts < ARGS.maximum_regrasp_attempts:
                regrasp_attempts += 1
                regrasp_trigger_progress_fractions.append(measured_progress)
                regrasp_trigger_progress = measured_progress
                regrasp_hold_robot_joints = measured_robot_joints.copy()
                regrasp_hold_base = measured_base.copy()
                regrasp_phase = "freeze"
                regrasp_phase_step = 0
                regrasp_stable_contact_steps = 0
                stalled_steps = 0
                no_object_progress_steps = 0
                continue
            opening_termination = "regrasp_budget_exhausted"
            last_blocking_gates = ["selected_surface_contact_missing"]
            break
        can_advance = (
            tracking_ready
            and interaction_lead_m <= active_lead_limit_m
            and selected_contact_now
        )
        if can_advance and reference_progress < 1.0:
            reference_progress = min(1.0, reference_progress + 1.0 / nominal_steps)
            stalled_steps = 0
        else:
            stalled_steps += 1
        if stalled_steps > ARGS.physics_hz * 8:
            opening_termination = "control_stalled"
            break
        if (
            not contact_during_opening
            and reference_progress <= 1.0e-12
            and opening_step + 1 >= int(math.ceil(ARGS.initial_contact_acquisition_seconds * ARGS.physics_hz))
        ):
            opening_termination = "initial_selected_contact_not_acquired"
            break

    robot.set_joint_velocity_targets(np.zeros((1, 4), dtype=np.float32), joint_indices=wheel_indices)
    phase = "final_hold"
    for _ in range(60):
        hold_progress, _ = step_and_record()
        maximum_opening_progress = max(maximum_opening_progress, hold_progress)
    videos = recorder.encode()
    penetration = max(0.0, -minimum_separation)
    failure_reasons = physical_open_failure_reasons(
        maximum_progress_fraction=maximum_opening_progress,
        acceptance_fraction=float(task["acceptance_fraction"]),
        contact_during_opening=contact_during_opening,
        maximum_penetration_m=penetration,
        allowed_penetration_m=ARGS.maximum_penetration_m,
        maximum_contact_force_n=maximum_force,
        allowed_contact_force_n=ARGS.maximum_contact_force_n,
    )
    strict_success = not failure_reasons
    result.update(
        {
            "strict_physical_open_success": strict_success,
            "maximum_progress_fraction": maximum_opening_progress,
            "maximum_episode_progress_fraction": maximum_progress,
            "opening_start_progress_fraction": opening_start_progress,
            "acceptance_fraction": float(task["acceptance_fraction"]),
            "measured_finger_contact": measured_finger_contact,
            "contact_during_opening": contact_during_opening,
            "maximum_contact_force_n": maximum_force,
            "minimum_contact_separation_m": minimum_separation,
            "maximum_penetration_m": penetration,
            "maximum_off_selected_surface_force_n": maximum_off_surface_force,
            "maximum_interaction_lead_m": maximum_interaction_lead,
            "maximum_interaction_lead_while_selected_contact_m": maximum_contact_interaction_lead,
            "maximum_probe_fraction": maximum_probe_fraction,
            "maximum_opening_finger_a_force_n": maximum_opening_finger_a_force,
            "maximum_opening_finger_b_force_n": maximum_opening_finger_b_force,
            "opening_selected_contact_steps": opening_selected_contact_steps,
            "opening_bilateral_contact_steps": opening_bilateral_contact_steps,
            "regrasp": {
                "maximum_attempts": ARGS.maximum_regrasp_attempts,
                "attempts": regrasp_attempts,
                "recoveries": regrasp_recoveries,
                "contact_loss_seconds": ARGS.contact_loss_seconds,
                "freeze_seconds": ARGS.regrasp_freeze_seconds,
                "open_fraction": ARGS.regrasp_open_fraction,
                "open_seconds": ARGS.regrasp_open_seconds,
                "reposition_seconds": ARGS.regrasp_reposition_seconds,
                "close_seconds": ARGS.regrasp_close_seconds,
                "stable_contact_seconds": ARGS.regrasp_stable_contact_seconds,
                "maximum_progress_rollback_fraction": ARGS.regrasp_maximum_progress_rollback_fraction,
                "trigger_progress_fractions": regrasp_trigger_progress_fractions,
                "recovery_progress_fractions": regrasp_recovery_progress_fractions,
            },
            "final_control_telemetry": control_telemetry,
            "interaction_lead_limit_m": ARGS.maximum_interaction_lead_m,
            "probe_interaction_lead_limit_m": ARGS.maximum_probe_interaction_lead_m,
            "tracking_limits": {
                "free_space_base_error_m": ARGS.maximum_base_tracking_error_m,
                "contact_loaded_base_error_m": ARGS.maximum_contact_base_tracking_error_m,
                "base_yaw_error_rad": ARGS.maximum_base_yaw_error_rad,
                "robot_joint_error_rad": ARGS.maximum_robot_joint_tracking_error_rad,
            },
            "friction": friction,
            "planner_effort": {
                "multiplier": ARGS.planner_effort_multiplier,
                "drives": planner_effort,
            },
            "base_effort": {
                "multiplier": ARGS.base_effort_multiplier,
                "drives": base_effort,
                "position_gain": ARGS.base_position_gain,
                "yaw_gain": ARGS.base_yaw_gain,
            },
            "collision_policy": collision_policy,
            "videos": videos,
            "step_count": step_count,
            "nominal_opening_duration_seconds": duration,
            "maximum_total_opening_seconds": ARGS.maximum_opening_seconds,
            "initial_contact_acquisition_seconds": ARGS.initial_contact_acquisition_seconds,
            "recorded_frame_count": recorder.frame_count,
            "skipped_empty_camera_frames": recorder.skipped_empty_frames,
            "opening_termination": opening_termination,
            "blocking_gates_at_termination": last_blocking_gates,
            "failure_reasons": list(failure_reasons),
            "failure_reason": None if strict_success else "+".join(failure_reasons),
        }
    )
    (ARGS.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if strict_success else 1


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except Exception as error:
        ARGS.output_dir.mkdir(parents=True, exist_ok=True)
        (ARGS.output_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (ARGS.output_dir / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": "physx_drive_from_planned_contact_pose",
                    "dataset_ready": False,
                    "strict_physical_open_success": False,
                    "failure_reason": type(error).__name__,
                    "failure_message": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        APP.close()
    raise SystemExit(exit_code)
