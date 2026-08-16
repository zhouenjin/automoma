#!/usr/bin/env python3
"""Plan grasp-conditioned G2 AKR trajectories for a RealAppliance asset.

This is a planning diagnostic, not a physical-success evaluator.  It consumes
automatically ranked contact candidates, tries both hands, jointly optimizes
the planar base/body/arm/object chain, and records every failure.  Dataset-ready
success remains the responsibility of the later PhysX drive-mode executor.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automoma.integrations.realappliance.akr_adapter import build_g2_akr_urdf, make_g2_akr_config  # noqa: E402
from automoma.integrations.realappliance.contact_candidates import (  # noqa: E402
    ContactCandidate,
    load_contact_candidates,
)
from automoma.integrations.realappliance.contracts import JointKind  # noqa: E402
from automoma.integrations.realappliance.g2_adapter import Hand  # noqa: E402
from automoma.integrations.realappliance.transform_math import (  # noqa: E402
    axis_motion_transform,
    invert_rigid,
    rotation_matrix_to_quaternion_wxyz,
    transform_point,
)
from automoma.integrations.realappliance.usd_task import ExtractedUsdTask, extract_usd_tasks  # noqa: E402
from automoma.integrations.realappliance.usd_collision import (  # noqa: E402
    fixed_body_cluster,
    load_placed_collision_world,
)
from automoma.integrations.realappliance.trajectory_selection import weighted_cspace_path_scores  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _pose_list(matrix: Sequence[Sequence[float]]) -> list[float]:
    transform = np.asarray(matrix, dtype=np.float64)
    return [*transform[:3, 3].tolist(), *rotation_matrix_to_quaternion_wxyz(transform)]


def _goal_ee_pose(task: ExtractedUsdTask, world_from_ee_start: Sequence[Sequence[float]]) -> np.ndarray:
    goal_position = task.task.position_at_fraction(task.task.planning_fraction)
    displacement = goal_position - task.task.initial_position
    motion = axis_motion_transform(
        task.joint_axis_local,
        displacement,
        revolute=task.task.joint_kind is JointKind.REVOLUTE,
    )
    world_from_joint = np.asarray(task.world_from_joint_at_initial, dtype=np.float64)
    world_from_target_start = np.asarray(task.world_from_target_link_at_initial, dtype=np.float64)
    world_from_target_goal = world_from_joint @ motion @ invert_rigid(world_from_joint) @ world_from_target_start
    target_from_ee = invert_rigid(world_from_target_start) @ np.asarray(world_from_ee_start, dtype=np.float64)
    return world_from_target_goal @ target_from_ee


class HandPlanner:
    """Cache one kinematic IK solver per hand for the diagnostic gate."""

    def __init__(self, hand: Hand, config: Dict[str, Any], ik_seeds: int):
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        self.hand = hand
        self.config = config
        self.tensor_args = TensorDeviceType()
        self.ik_seeds = int(ik_seeds)
        robot = RobotConfig.from_dict(copy.deepcopy(config["robot_cfg"]), self.tensor_args)
        solver_config = IKSolverConfig.load_from_robot_config(
            robot,
            None,
            self.tensor_args,
            num_seeds=self.ik_seeds,
            use_cuda_graph=False,
        )
        self.solver = IKSolver(solver_config)
        retract_values = config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        self.retract = self.tensor_args.to_device(retract_values).unsqueeze(0)

    def solve(self, world_from_ee: Sequence[Sequence[float]], max_solutions: int) -> torch.Tensor:
        from curobo.types.math import Pose

        goal = Pose.from_list(_pose_list(world_from_ee))
        result = self.solver.solve_single(
            goal_pose=goal,
            retract_config=self.retract,
            return_seeds=max_solutions,
            num_seeds=self.ik_seeds,
        ).get_unique_solution()
        return result[:max_solutions]


def _nearest_pairs(start: torch.Tensor, goal: torch.Tensor, maximum: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if start.shape[0] == 0 or goal.shape[0] == 0:
        return start[:0], goal[:0]
    distances = torch.linalg.vector_norm(start[:, None, :] - goal[None, :, :], dim=-1)
    flat_indices = torch.argsort(distances.flatten())[:maximum]
    start_indices = torch.div(flat_indices, goal.shape[0], rounding_mode="floor")
    goal_indices = flat_indices % goal.shape[0]
    return start[start_indices], goal[goal_indices]


def _terminal_fk_audit(motion_gen: Any, trajectories: torch.Tensor) -> Dict[str, Any]:
    batch, steps, dof = trajectories.shape
    positions = motion_gen.tensor_args.to_device(trajectories.reshape(batch * steps, dof))
    poses = motion_gen.ik_solver.fk(positions).ee_pose
    xyz = poses.position.reshape(batch, steps, 3)
    quaternion = poses.quaternion.reshape(batch, steps, 4)
    xyz_reference = xyz[:, -1:, :]
    quaternion_reference = quaternion[:, -1:, :]
    position_drift = torch.linalg.vector_norm(xyz - xyz_reference, dim=-1)
    quaternion_dot = torch.abs(torch.sum(quaternion * quaternion_reference, dim=-1)).clamp(max=1.0)
    rotation_drift = 2.0 * torch.arccos(quaternion_dot)
    position_per_trajectory = torch.max(position_drift, dim=1).values.detach().cpu()
    rotation_per_trajectory = torch.max(rotation_drift, dim=1).values.detach().cpu()
    return {
        "max_position_drift_m": float(torch.max(position_per_trajectory).item()),
        "max_rotation_drift_rad": float(torch.max(rotation_per_trajectory).item()),
        "position_drift_per_trajectory_m": position_per_trajectory.tolist(),
        "rotation_drift_per_trajectory_rad": rotation_per_trajectory.tolist(),
    }


def _base_motion_audit(trajectories: torch.Tensor, valid: torch.Tensor) -> Dict[str, Any]:
    """Measure optimized planar-base motion without imposing a fixed sharing ratio."""

    base = trajectories[:, :, :3]
    delta_xy = base[:, 1:, :2] - base[:, :-1, :2]
    delta_yaw = base[:, 1:, 2] - base[:, :-1, 2]
    net_xy = torch.linalg.vector_norm(base[:, -1, :2] - base[:, 0, :2], dim=-1)
    path_xy = torch.sum(torch.linalg.vector_norm(delta_xy, dim=-1), dim=-1)
    net_yaw = torch.abs(base[:, -1, 2] - base[:, 0, 2])
    path_yaw = torch.sum(torch.abs(delta_yaw), dim=-1)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    return {
        "net_xy_m_per_trajectory": net_xy.tolist(),
        "path_xy_m_per_trajectory": path_xy.tolist(),
        "net_yaw_rad_per_trajectory": net_yaw.tolist(),
        "path_yaw_rad_per_trajectory": path_yaw.tolist(),
        "valid_trajectory_indices": valid_indices.tolist(),
        "valid_net_xy_m": net_xy[valid_indices].tolist(),
        "valid_path_xy_m": path_xy[valid_indices].tolist(),
        "valid_net_yaw_rad": net_yaw[valid_indices].tolist(),
        "valid_path_yaw_rad": path_yaw[valid_indices].tolist(),
    }


def _plan_augmented_trajectory(
    start_robot: torch.Tensor,
    goal_robot: torch.Tensor,
    akr_config: Dict[str, Any],
    akr_start: float,
    akr_goal: float,
    maximum_pairs: int,
    planning_fraction: float,
    acceptance_fraction: float,
    world_model: Any,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.rollout.rollout_base import Goal
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    start_robot, goal_robot = _nearest_pairs(start_robot, goal_robot, maximum_pairs)
    if start_robot.shape[0] == 0:
        raise RuntimeError("no IK pair is available for trajectory optimization")
    start = torch.cat((start_robot, torch.full((start_robot.shape[0], 1), akr_start, device=start_robot.device)), dim=1)
    goal = torch.cat((goal_robot, torch.full((goal_robot.shape[0], 1), akr_goal, device=goal_robot.device)), dim=1)
    tensor_args = TensorDeviceType()
    motion_config = MotionGenConfig.load_from_robot_config(
        copy.deepcopy(akr_config["robot_cfg"]),
        world_model,
        tensor_args,
        num_trajopt_seeds=8,
        num_graph_seeds=4,
        trajopt_tsteps=32,
        interpolation_dt=0.05,
        optimize_dt=True,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.MESH,
    )
    motion_gen = MotionGen(motion_config)
    start_state = JointState.from_position(tensor_args.to_device(start))
    goal_state = JointState.from_position(tensor_args.to_device(goal))
    terminal_pose = motion_gen.ik_solver.fk(goal_state.position).ee_pose
    pose_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        offset_tstep_fraction=-1.0,
    )
    metric_enabled = motion_gen.update_pose_cost_metric(
        pose_metric,
        start_state=start_state,
        goal_pose=terminal_pose,
    )
    if not metric_enabled:
        raise RuntimeError("cuRobo rejected the full-pose AKR path constraint")
    result = motion_gen.trajopt_solver.solve_batch(
        Goal(goal_pose=terminal_pose, goal_state=goal_state, current_state=start_state)
    )
    trajectories = result.solution.position.detach().clone().cpu()
    trajopt_success = result.success.detach().clone().cpu().reshape(-1)
    if trajectories.ndim == 2:
        trajectories = trajectories.unsqueeze(0)
    audit = _terminal_fk_audit(motion_gen, trajectories)
    final_akr = trajectories[:, -1, -1]
    denominator = akr_goal - akr_start
    if abs(denominator) < 1e-9:
        raise ValueError("AKR start and goal positions are identical")
    achieved_open_fraction = planning_fraction * (final_akr - akr_start) / denominator
    position_ok = torch.as_tensor(audit["position_drift_per_trajectory_m"]) <= 0.005
    rotation_ok = torch.as_tensor(audit["rotation_drift_per_trajectory_rad"]) <= 0.05
    progress_ok = achieved_open_fraction >= acceptance_fraction
    valid = trajopt_success & position_ok & rotation_ok & progress_ok
    base_motion_audit = _base_motion_audit(trajectories, valid)
    selection = weighted_cspace_path_scores(
        trajectories.numpy(),
        valid.numpy(),
        akr_config["robot_cfg"]["kinematics"]["cspace"]["cspace_distance_weight"],
    )
    result_report = {
        "pair_count": int(start.shape[0]),
        "trajopt_successful_plans": int(trajopt_success.sum().item()),
        "valid_akr_plans": int(valid.sum().item()),
        "achieved_open_fraction_per_trajectory": achieved_open_fraction.tolist(),
        "planning_fraction": planning_fraction,
        "acceptance_fraction": acceptance_fraction,
        "status": str(getattr(result, "status", "not_exposed_by_trajopt_result")),
        "terminal_fk_audit": audit,
        "base_motion_audit": base_motion_audit,
        "automatic_trajectory_selection": selection,
    }
    tensors = {
        "start_states": start.detach().cpu(),
        "goal_states": goal.detach().cpu(),
        "trajectories": trajectories,
        "trajopt_success": trajopt_success,
        "success": valid,
        "selected_trajectory_index": selection["selected_trajectory_index"],
    }
    return result_report, tensors


def _attempt_id(candidate: ContactCandidate, hand: Hand, depth_fraction: float) -> str:
    return f"rank{candidate.automatic_rank:04d}_{candidate.hypothesis_id}_{hand.value}_depth{depth_fraction:.3f}"


def _iter_attempts(
    candidates: Sequence[ContactCandidate], hands: Iterable[Hand], depth_fractions: Sequence[float]
) -> Iterable[Tuple[ContactCandidate, Hand, float]]:
    for candidate in candidates:
        for depth_fraction in depth_fractions:
            for hand in hands:
                yield candidate, hand, float(depth_fraction)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--capture-depth-fraction", type=float, action="append")
    parser.add_argument("--handle-anchor", type=float, nargs=3, default=(0.8, 0.0, 1.34))
    parser.add_argument("--desired-approach", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--hand", action="append", choices=["left", "right"])
    parser.add_argument("--ik-seeds", type=int, default=96)
    parser.add_argument("--max-ik-solutions", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=16)
    args = parser.parse_args()

    depth_fractions = tuple(args.capture_depth_fraction or (0.20, 0.35, 0.50))
    hands = tuple(Hand(value) for value in (args.hand or ("left", "right")))
    source_usd = args.asset_dir / "Aligned.usd"
    tasks = extract_usd_tasks(source_usd, args.asset_dir / "gt_part.json", "open")
    base_configs = {
        hand: _load_yaml(args.generated_dir / f"g2_automoma_{hand.value}.yml") for hand in hands
    }
    hand_planners = {hand: HandPlanner(hand, base_configs[hand], args.ik_seeds) for hand in hands}
    report: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "akr_diagnostic_fixed_appliance_mesh_collision",
        "dataset_ready": False,
        "source_usd": str(source_usd.resolve()),
        "candidate_file": str(args.candidate_file.resolve()),
        "attempts": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        moving_cluster = fixed_body_cluster(source_usd, task.body1_path)
        candidates = load_contact_candidates(
            args.candidate_file,
            source_usd=source_usd,
            target_joint_path=task.joint_path,
        )[: args.max_candidates]
        for candidate, hand, depth_fraction in _iter_attempts(candidates, hands, depth_fractions):
            attempt_id = _attempt_id(candidate, hand, depth_fraction)
            attempt_dir = args.output_dir / attempt_id
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt: Dict[str, Any] = {
                "attempt_id": attempt_id,
                "hypothesis_id": candidate.hypothesis_id,
                "automatic_rank": candidate.automatic_rank,
                "pre_ik_score": candidate.pre_ik_score,
                "handle_body_source_path": candidate.handle_body_path,
                "target_moving_body_source_path": candidate.target_moving_body_path,
                "asset_id_for_reporting_only": candidate.asset_id_for_reporting_only,
                "hand": hand.value,
                "capture_depth_fraction": depth_fraction,
                "success": False,
            }
            try:
                placed = candidate.place(
                    handle_anchor_world=args.handle_anchor,
                    desired_approach_world=args.desired_approach,
                    capture_depth_fraction=depth_fraction,
                )
                placed_task = task.placed(placed.world_from_source)
                attempt["target_articulation"] = placed_task.to_dict()
                attempt["target_articulation"]["task"].update(
                    {
                        "planning_fraction": placed_task.task.planning_fraction,
                        "acceptance_fraction": placed_task.task.acceptance_fraction,
                        "open_limit": placed_task.task.open_limit,
                    }
                )
                attempt["contact_center_world_m"] = transform_point(
                    placed.world_from_source, candidate.contact_center_source
                )
                attempt["contact_points_world_m"] = [
                    transform_point(placed.world_from_source, point) for point in candidate.contact_points_source
                ]
                collision_world = load_placed_collision_world(
                    source_usd,
                    placed.world_from_source,
                    exclude_paths=moving_cluster,
                    moving_cluster_paths=moving_cluster,
                )
                attempt["collision_world"] = {
                    "planning_proxy_only": True,
                    "included_mesh_paths": collision_world.included_mesh_paths,
                    "excluded_mesh_paths": collision_world.excluded_mesh_paths,
                    "moving_cluster_paths": collision_world.moving_cluster_paths,
                    "physx_collision_policy": "all original collisions remain enabled during physical execution",
                }
                world_from_ee_start = np.asarray(placed.world_from_ee_contact)
                world_from_ee_goal = _goal_ee_pose(placed_task, world_from_ee_start)
                start_ik = hand_planners[hand].solve(world_from_ee_start, args.max_ik_solutions)
                goal_ik = hand_planners[hand].solve(world_from_ee_goal, args.max_ik_solutions)
                attempt["start_ik_solutions"] = int(start_ik.shape[0])
                attempt["goal_ik_solutions"] = int(goal_ik.shape[0])
                attempt["world_from_source"] = np.asarray(placed.world_from_source).tolist()
                attempt["start_ee_pose_xyz_wxyz"] = _pose_list(world_from_ee_start)
                attempt["goal_ee_pose_xyz_wxyz"] = _pose_list(world_from_ee_goal)
                if start_ik.shape[0] == 0 or goal_ik.shape[0] == 0:
                    raise RuntimeError("start or goal IK has no solution")

                attachment = placed_task.make_attachment_spec(hand, world_from_ee_start)
                akr_urdf = attempt_dir / "g2_akr.urdf"
                build_g2_akr_urdf(
                    args.generated_dir / "g2_crsB_swiftpicker_automoma_planar.urdf",
                    akr_urdf,
                    attachment,
                )
                akr_config = make_g2_akr_config(base_configs[hand], akr_urdf, attachment)
                with (attempt_dir / "g2_akr.yml").open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(akr_config, stream, sort_keys=False)
                goal_position = placed_task.task.position_at_fraction(placed_task.task.planning_fraction)
                planning_report, tensors = _plan_augmented_trajectory(
                    start_ik,
                    goal_ik,
                    akr_config,
                    attachment.akr_position(placed_task.task.initial_position),
                    attachment.akr_position(goal_position),
                    args.max_pairs,
                    placed_task.task.planning_fraction,
                    placed_task.task.acceptance_fraction,
                    collision_world.world,
                )
                torch.save(tensors, attempt_dir / "trajectory.pt")
                attempt["planning"] = planning_report
                attempt["success"] = planning_report["valid_akr_plans"] > 0
                attempt["failure_reason"] = None if attempt["success"] else "trajopt_no_success"
            except Exception as error:  # retain the exact failed hypothesis and traceback
                attempt["failure_reason"] = type(error).__name__
                attempt["failure_message"] = str(error)
                (attempt_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            (attempt_dir / "attempt.json").write_text(json.dumps(attempt, indent=2, sort_keys=True), encoding="utf-8")
            report["attempts"].append(attempt)
            print(
                attempt_id,
                "success" if attempt["success"] else f"failed:{attempt.get('failure_reason')}",
                flush=True,
            )

    report["successful_attempts"] = sum(bool(attempt["success"]) for attempt in report["attempts"])
    report["attempt_count"] = len(report["attempts"])
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)
    return 0 if report["successful_attempts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
