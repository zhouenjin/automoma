#!/usr/bin/env python3
"""Build a grasp-specific AutoMoMa AKR config from a native USD manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from automoma.integrations.realappliance_native import (
    build_akr_robot_config,
    build_open_joint_components,
    manifest_from_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-robot-config", type=Path, required=True)
    parser.add_argument("--base-spheres", type=Path, required=True)
    parser.add_argument("--joint-path")
    parser.add_argument(
        "--component-to-ee-pose",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        required=True,
    )
    parser.add_argument("--source-initial-position", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = manifest_from_mapping(json.loads(args.manifest.read_text(encoding="utf-8")))
    components = build_open_joint_components(manifest)
    if args.joint_path:
        components = tuple(
            component for component in components if component.joint.path == args.joint_path
        )
    if not components:
        raise RuntimeError("no matching open-joint component")
    component = components[0]

    base = yaml.safe_load(args.base_robot_config.read_text(encoding="utf-8"))
    sphere_config = yaml.safe_load(args.base_spheres.read_text(encoding="utf-8"))
    base["robot_cfg"]["kinematics"]["collision_spheres"] = sphere_config["collision_spheres"]
    akr = build_akr_robot_config(
        base,
        manifest,
        component,
        component_to_ee_pose=args.component_to_ee_pose,
        source_initial_position=args.source_initial_position,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(akr, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "asset_id": manifest.asset_id,
                "joint_path": component.joint.path,
                "component_bodies": component.rigid_bodies,
                "collision_sphere_count": len(
                    akr["realappliance_native"]["collision_spheres"][
                        "realappliance_grasped_component"
                    ]
                ),
                "g2_inputs_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
