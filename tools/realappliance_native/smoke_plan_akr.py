#!/usr/bin/env python3
"""Run a native GraspGen -> IK -> AutoMoMa AKR trajectory smoke test."""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import pi
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo.geom.types import Cuboid, WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from automoma.integrations.realappliance_native import manifest_from_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--joint-path", required=True)
    parser.add_argument("--component-cloud", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--base-robot-config", type=Path, required=True)
    parser.add_argument("--akr-config", type=Path, required=True)
    parser.add_argument("--target-fraction", type=float, default=0.8)
    parser.add_argument("--pair-count", type=int, default=64)
    parser.add_argument("--ik-seeds", type=int, default=2048)
    parser.add_argument("--staging-center", type=float, nargs=3, default=(0.75, 0.0, 0.8))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pose_matrix(pose) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(np.roll(np.asarray(pose[3:]), -1)).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def pose_list(matrix: np.ndarray) -> list[float]:
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [*matrix[:3, 3].tolist(), *np.roll(quaternion, 1).tolist()]


def clean_base_config(config: dict) -> dict:
    kinematics = config["kinematics"]
    kinematics["extra_links"] = {}
    kinematics["collision_link_names"] = [
        value for value in kinematics["collision_link_names"] if value != "attached_object"
    ]
    kinematics["extra_collision_spheres"] = {}
    for values in kinematics["self_collision_ignore"].values():
        if "attached_object" in values:
            values.remove("attached_object")
    return config


def motion_gen(robot_cfg: dict, tensor_args: TensorDeviceType, *, seeds: int) -> MotionGen:
    sentinel = Cuboid(
        name="empty_world_sentinel",
        pose=[100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
        dims=[0.1, 0.1, 0.1],
    )
    config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        WorldConfig(cuboid=[sentinel]),
        tensor_args,
        num_ik_seeds=seeds,
        num_trajopt_seeds=12,
        num_graph_seeds=4,
        trajopt_tsteps=32,
        use_cuda_graph=False,
        self_collision_check=True,
    )
    return MotionGen(config)


def solve_ik(mg: MotionGen, pose: list[float], retract: torch.Tensor, seeds: int) -> torch.Tensor:
    return mg.ik_solver.solve_single(
        goal_pose=Pose.from_list(pose),
        retract_config=retract.unsqueeze(0),
        return_seeds=min(256, seeds),
        num_seeds=seeds,
    ).get_unique_solution()


def tensor_to_numpy(value):
    if value is None:
        return None
    return value.detach().cpu().numpy()


def endpoint_diagnostics(mg: MotionGen, positions: torch.Tensor) -> tuple[list[bool], list[str]]:
    feasible = []
    statuses = []
    for position in positions:
        valid, status = mg.check_start_state(JointState.from_position(position))
        feasible.append(bool(valid))
        statuses.append("valid" if status is None else str(status))
    return feasible, statuses


def metric_summary(metrics) -> dict:
    result = {}
    for name in ("feasible", "constraint", "cost"):
        value = getattr(metrics, name, None) if metrics is not None else None
        if value is None:
            result[name] = None
            continue
        array = tensor_to_numpy(value)
        result[name] = {
            "shape": list(array.shape),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
        }
        if name == "feasible":
            result[name]["true_count"] = int(np.count_nonzero(array))
            result[name]["total_count"] = int(array.size)
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_fraction <= 1.0:
        raise ValueError("target fraction must be in (0, 1]")
    manifest = manifest_from_mapping(json.loads(args.manifest.read_text(encoding="utf-8")))
    joint_matches = [value for value in manifest.joints if value.path == args.joint_path]
    if len(joint_matches) != 1:
        raise RuntimeError("joint path is missing or ambiguous")
    joint = joint_matches[0]
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["candidates"]
    candidate = next(value for value in candidates if int(value["rank"]) == args.candidate_rank)
    cloud = np.load(args.component_cloud)

    component_to_ee = pose_matrix(candidate["component_to_gripper_base_pose"])
    component_to_world = np.asarray(cloud["component_to_world"], dtype=np.float64)
    component_center = np.asarray(cloud["points_component_m"], dtype=np.float64).mean(axis=0)
    source_translation = component_to_world[:3, 3].copy()
    component_to_world[:3, 3] = np.asarray(args.staging_center) - (
        component_to_world[:3, :3] @ component_center
    )
    staging_shift = component_to_world[:3, 3] - source_translation
    start_ee = component_to_world @ component_to_ee
    axis = np.asarray(cloud["joint_axis_world"], dtype=np.float64)
    pivot = np.asarray(cloud["joint_pivot_world_m"], dtype=np.float64) + staging_shift

    lower = float(joint.lower_limit if joint.lower_limit is not None else 0.0)
    upper = float(joint.upper_limit if joint.upper_limit is not None else 0.0)
    initial = min(max(0.0, lower), upper)
    positive_room = upper - initial
    negative_room = initial - lower
    source_goal = initial + args.target_fraction * positive_room
    if negative_room > positive_room:
        source_goal = initial - args.target_fraction * negative_room
    delta_source = source_goal - initial
    goal_ee = start_ee.copy()
    if joint.joint_type == "revolute":
        rotation = Rotation.from_rotvec(axis * delta_source * pi / 180.0).as_matrix()
        goal_ee[:3, 3] = pivot + rotation @ (start_ee[:3, 3] - pivot)
        goal_ee[:3, :3] = rotation @ start_ee[:3, :3]
        akr_goal = -source_goal * pi / 180.0
        akr_initial = -initial * pi / 180.0
    else:
        delta_m = delta_source * manifest.meters_per_unit
        goal_ee[:3, 3] = start_ee[:3, 3] + axis * delta_m
        akr_goal = -source_goal * manifest.meters_per_unit
        akr_initial = -initial * manifest.meters_per_unit

    tensor_args = TensorDeviceType()
    base_cfg = clean_base_config(load_yaml(str(args.base_robot_config))["robot_cfg"])
    base_mg = motion_gen(base_cfg, tensor_args, seeds=args.ik_seeds)
    retract = tensor_args.to_device(base_cfg["kinematics"]["cspace"]["retract_config"])
    start_iks = solve_ik(base_mg, pose_list(start_ee), retract, args.ik_seeds)
    goal_iks = solve_ik(base_mg, pose_list(goal_ee), retract, args.ik_seeds)
    if start_iks is None or goal_iks is None or len(start_iks) == 0 or len(goal_iks) == 0:
        raise RuntimeError(f"IK failed: start={len(start_iks)} goal={len(goal_iks)}")

    pairs = list(product(range(min(len(start_iks), 16)), range(min(len(goal_iks), 16))))
    pairs = pairs[: args.pair_count]
    start = torch.stack([start_iks[i] for i, _ in pairs])
    goal = torch.stack([goal_iks[j] for _, j in pairs])
    start = torch.cat(
        [start, torch.full((len(start), 1), akr_initial, device=start.device)], dim=1
    )
    goal = torch.cat([goal, torch.full((len(goal), 1), akr_goal, device=goal.device)], dim=1)

    akr_raw = load_yaml(str(args.akr_config))
    akr_cfg = akr_raw["robot_cfg"]
    akr_mg = motion_gen(akr_cfg, tensor_args, seeds=min(args.ik_seeds, 512))
    start_state = JointState.from_position(start)
    goal_state = JointState.from_position(goal)
    start_endpoint_feasible, start_endpoint_status = endpoint_diagnostics(akr_mg, start)
    goal_endpoint_feasible, goal_endpoint_status = endpoint_diagnostics(akr_mg, goal)
    start_constraint_metrics = akr_mg.check_constraints(start_state)
    goal_constraint_metrics = akr_mg.check_constraints(goal_state)
    anchor_goal = akr_mg.ik_solver.fk(goal_state.position).ee_pose
    result = akr_mg.trajopt_solver.solve_batch(
        Goal(goal_pose=anchor_goal, goal_state=goal_state, current_state=start_state)
    )
    trajectories = result.solution.position.detach().cpu()
    success = result.success.detach().cpu().reshape(-1)
    valid = []
    anchor_position_error = []
    anchor_rotation_error = []
    for index in range(len(trajectories)):
        fk = akr_mg.ik_solver.fk(trajectories[index].to(start.device)).ee_pose
        position = fk.position.detach().cpu().numpy()
        quaternion = fk.quaternion.detach().cpu().numpy()
        reference_position = position[-1]
        reference_quaternion = quaternion[-1]
        pos_error = float(np.linalg.norm(position - reference_position, axis=1).max())
        dots = np.abs((quaternion * reference_quaternion).sum(axis=1)).clip(0.0, 1.0)
        rot_error = float((2.0 * np.arccos(dots)).max())
        anchor_position_error.append(pos_error)
        anchor_rotation_error.append(rot_error)
        valid.append(bool(success[index]) and pos_error <= 0.01 and rot_error <= 0.05)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        start_states=start.detach().cpu().numpy(),
        goal_states=goal.detach().cpu().numpy(),
        start_endpoint_feasible=np.asarray(start_endpoint_feasible),
        goal_endpoint_feasible=np.asarray(goal_endpoint_feasible),
        start_constraint_feasible=tensor_to_numpy(start_constraint_metrics.feasible),
        start_constraint=tensor_to_numpy(start_constraint_metrics.constraint),
        goal_constraint_feasible=tensor_to_numpy(goal_constraint_metrics.feasible),
        goal_constraint=tensor_to_numpy(goal_constraint_metrics.constraint),
        trajectory_metric_feasible=tensor_to_numpy(
            result.metrics.feasible if result.metrics is not None else None
        ),
        trajectory_metric_constraint=tensor_to_numpy(
            result.metrics.constraint if result.metrics is not None else None
        ),
        trajectories=trajectories.numpy(),
        trajopt_success=success.numpy(),
        anchor_valid=np.asarray(valid),
        anchor_position_error_m=np.asarray(anchor_position_error),
        anchor_rotation_error_rad=np.asarray(anchor_rotation_error),
        start_ee_pose=np.asarray(pose_list(start_ee)),
        goal_ee_pose=np.asarray(pose_list(goal_ee)),
    )
    report = {
        "schema_version": "automoma.realappliance.akr_smoke.v1",
        "provenance": {"pipeline": "automoma_native", "g2_inputs_used": False},
        "asset_id": manifest.asset_id,
        "joint_path": joint.path,
        "candidate_rank": args.candidate_rank,
        "candidate_source": candidate["source_component"],
        "graspgen_confidence": candidate["graspgen_confidence"],
        "source_initial_position": initial,
        "source_goal_position": source_goal,
        "target_fraction": args.target_fraction,
        "start_ik_count": len(start_iks),
        "goal_ik_count": len(goal_iks),
        "trajectory_count": len(trajectories),
        "trajopt_success_count": int(success.sum()),
        "start_endpoint_feasible_count": int(np.count_nonzero(start_endpoint_feasible)),
        "goal_endpoint_feasible_count": int(np.count_nonzero(goal_endpoint_feasible)),
        "start_endpoint_status_counts": {
            value: start_endpoint_status.count(value) for value in sorted(set(start_endpoint_status))
        },
        "goal_endpoint_status_counts": {
            value: goal_endpoint_status.count(value) for value in sorted(set(goal_endpoint_status))
        },
        "start_constraint_metrics": metric_summary(start_constraint_metrics),
        "goal_constraint_metrics": metric_summary(goal_constraint_metrics),
        "trajectory_metrics": metric_summary(result.metrics),
        "anchor_valid_count": int(np.asarray(valid).sum()),
        "strict_physical_success": False,
        "strict_physical_pending": True,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
