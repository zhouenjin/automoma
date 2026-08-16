#!/usr/bin/env python3
"""Search generic suspended-object poses reachable by a fixed Franka.

The search uses only a fresh grasp candidate, articulation geometry, and IK.
It never branches on a RealAppliance asset id and does not consume G2 data.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import pi
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml

from automoma.integrations.realappliance_native import manifest_from_mapping
from tools.realappliance_native.smoke_plan_akr import (
    articulated_ee_pose,
    clean_base_config,
    motion_gen,
    pose_list,
    pose_matrix,
    solve_ik,
)


def float_values(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--joint-path", required=True)
    parser.add_argument("--component-cloud", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--base-robot-config", type=Path, required=True)
    parser.add_argument("--target-fraction", type=float, default=0.8)
    parser.add_argument("--ik-seeds", type=int, default=128)
    parser.add_argument("--x-values", type=float_values, default=(0.35, 0.45, 0.55, 0.65))
    parser.add_argument("--y-values", type=float_values, default=(-0.3, -0.15, 0.0, 0.15, 0.3))
    parser.add_argument("--z-values", type=float_values, default=(0.55, 0.7, 0.85))
    parser.add_argument(
        "--yaw-values",
        type=float_values,
        default=(-pi, -3 * pi / 4, -pi / 2, -pi / 4, 0.0, pi / 4, pi / 2, 3 * pi / 4),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = manifest_from_mapping(json.loads(args.manifest.read_text(encoding="utf-8")))
    joint = next(value for value in manifest.joints if value.path == args.joint_path)
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["candidates"]
    candidates = candidates[: args.candidate_count]
    cloud = np.load(args.component_cloud)
    points_component = np.asarray(cloud["points_component_m"], dtype=np.float64)
    component_center = points_component.mean(axis=0)
    source_component_to_world = np.asarray(cloud["component_to_world"], dtype=np.float64)
    source_axis_world = np.asarray(cloud["joint_axis_world"], dtype=np.float64)
    source_pivot_world = np.asarray(cloud["joint_pivot_world_m"], dtype=np.float64)
    axis_component = source_component_to_world[:3, :3].T @ source_axis_world
    pivot_component = np.linalg.inv(source_component_to_world) @ np.asarray(
        [*source_pivot_world.tolist(), 1.0]
    )

    lower = float(joint.lower_limit if joint.lower_limit is not None else 0.0)
    upper = float(joint.upper_limit if joint.upper_limit is not None else 0.0)
    initial = min(max(0.0, lower), upper)
    positive_room = upper - initial
    negative_room = initial - lower
    goal = initial + args.target_fraction * positive_room
    if negative_room > positive_room:
        goal = initial - args.target_fraction * negative_room
    delta_source = goal - initial

    tensor_args = TensorDeviceType()
    base_cfg = clean_base_config(load_yaml(str(args.base_robot_config))["robot_cfg"])
    mg = motion_gen(base_cfg, tensor_args, seeds=args.ik_seeds)
    retract = tensor_args.to_device(base_cfg["kinematics"]["cspace"]["retract_config"])

    records = []
    search_index = 0
    for candidate, x, y, z, yaw in product(
        candidates,
        args.x_values,
        args.y_values,
        args.z_values,
        args.yaw_values,
    ):
        rotation = Rotation.from_rotvec(np.asarray([0.0, 0.0, yaw])).as_matrix()
        component_to_world = source_component_to_world.copy()
        component_to_world[:3, :3] = rotation @ source_component_to_world[:3, :3]
        center = np.asarray([x, y, z], dtype=np.float64)
        component_to_world[:3, 3] = center - component_to_world[:3, :3] @ component_center
        axis = component_to_world[:3, :3] @ axis_component
        axis /= np.linalg.norm(axis)
        pivot = (component_to_world @ pivot_component)[:3]
        start_ee = component_to_world @ pose_matrix(candidate["component_to_gripper_base_pose"])
        goal_ee = articulated_ee_pose(
            start_ee,
            joint_type=joint.joint_type,
            axis=axis,
            pivot=pivot,
            delta_source=delta_source,
            meters_per_unit=manifest.meters_per_unit,
            fraction=1.0,
        )
        start_iks = solve_ik(mg, pose_list(start_ee), retract, args.ik_seeds)
        goal_iks = solve_ik(mg, pose_list(goal_ee), retract, args.ik_seeds)
        start_count = 0 if start_iks is None else len(start_iks)
        goal_count = 0 if goal_iks is None else len(goal_iks)
        if start_count and goal_count:
            records.append(
                {
                    "search_index": search_index,
                    "candidate_rank": int(candidate["rank"]),
                    "graspgen_confidence": float(candidate["graspgen_confidence"]),
                    "staging_center": center.tolist(),
                    "staging_yaw_rad": float(yaw),
                    "start_ik_count": start_count,
                    "goal_ik_count": goal_count,
                    "endpoint_score": int(min(start_count, goal_count)),
                }
            )
        search_index += 1

    records.sort(
        key=lambda value: (
            value["endpoint_score"],
            value["graspgen_confidence"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": "automoma.realappliance.fixed_staging_search.v1",
        "provenance": {"pipeline": "automoma_native", "g2_inputs_used": False},
        "asset_id": manifest.asset_id,
        "joint_path": joint.path,
        "tested_pose_count": search_index,
        "endpoint_reachable_count": len(records),
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "results": records[:10]}))


if __name__ == "__main__":
    main()
