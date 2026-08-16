#!/usr/bin/env python3
"""Reproducible CUDA smoke test for generated G2 and synthetic AKR configs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automoma.integrations.realappliance.akr_adapter import (  # noqa: E402
    AkrAttachmentSpec,
    TransformRPY,
    build_g2_akr_urdf,
    make_g2_akr_config,
)
from automoma.integrations.realappliance.contracts import JointKind  # noqa: E402
from automoma.integrations.realappliance.g2_adapter import Hand  # noqa: E402


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _load_config(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _solve_ik(config, target_offset: float):
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    robot_config = RobotConfig.from_dict(config["robot_cfg"], tensor_args)
    solver_config = IKSolverConfig.load_from_robot_config(
        robot_config,
        None,
        tensor_args,
        num_seeds=64,
        use_cuda_graph=False,
    )
    solver = IKSolver(solver_config)
    retract_values = config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
    retract = tensor_args.to_device(retract_values).unsqueeze(0)
    start_pose = solver.fk(retract).ee_pose
    offset = torch.zeros_like(start_pose.position)
    offset[..., 0] = target_offset
    goal = Pose(position=start_pose.position + offset, quaternion=start_pose.quaternion)
    solutions = solver.solve_single(
        goal_pose=goal,
        retract_config=retract,
        return_seeds=8,
        num_seeds=64,
    ).get_unique_solution()
    max_base_motion = 0.0
    if solutions.shape[0]:
        max_base_motion = float(torch.max(torch.abs(solutions[:, :3] - retract[:, :3])).item())
    return {
        "dof": int(retract.shape[-1]),
        "solutions": int(solutions.shape[0]),
        "max_abs_base_delta": max_base_motion,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-offset", type=float, default=0.08)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this smoke test requires a CUDA device")
    planar_urdf = args.generated_dir / "g2_crsB_swiftpicker_automoma_planar.urdf"
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "numpy": _package_version("numpy"),
            "warp_lang": _package_version("warp-lang"),
            "curobo": _package_version("nvidia-curobo"),
        },
        "hands": {},
    }
    for hand in (Hand.LEFT, Hand.RIGHT):
        config_path = args.generated_dir / f"g2_automoma_{hand.value}.yml"
        base_config = _load_config(config_path)
        hand_report = {"ik": _solve_ik(base_config, args.target_offset)}
        spec = AkrAttachmentSpec(
            hand=hand,
            source_joint_name="smoke_hinge",
            joint_kind=JointKind.REVOLUTE,
            joint_axis=(0.0, 0.0, 1.0),
            lower_limit=-1.2,
            upper_limit=0.0,
            initial_position=0.0,
            ee_to_handle=TransformRPY(),
            handle_to_joint_at_initial=TransformRPY(xyz=(0.0, 0.2, 0.0)),
            joint_at_initial_to_object_root=TransformRPY(xyz=(0.0, -0.2, 0.0)),
        )
        akr_urdf = args.generated_dir / f"g2_automoma_{hand.value}_smoke_akr.urdf"
        build_g2_akr_urdf(planar_urdf, akr_urdf, spec)
        akr_config = make_g2_akr_config(base_config, akr_urdf, spec)
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig

        loaded_akr = RobotConfig.from_dict(akr_config["robot_cfg"], TensorDeviceType())
        hand_report["akr"] = {
            "dof": len(akr_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]),
            "loaded_dof": int(loaded_akr.kinematics.kinematics_config.n_dof),
            "target_joint": spec.akr_joint_name,
        }
        report["hands"][hand.value] = hand_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0 if all(hand["ik"]["solutions"] > 0 for hand in report["hands"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
