#!/usr/bin/env python3
"""Build left/right AutoMoMa planner configs from the supplied G2 asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automoma.integrations.realappliance.g2_adapter import (  # noqa: E402
    Hand,
    build_planar_g2_urdf,
    fit_bounds_with_spheres,
    infer_link_visual_bounds,
    make_g2_curobo_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, required=True, help="Root of the supplied g2_task_ready package")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-distance-weight",
        type=float,
        nargs=3,
        default=(4.0, 4.0, 2.0),
        metavar=("X", "Y", "YAW"),
        help="Global soft path cost for planar-base motion; this is not a fixed base-motion share.",
    )
    parser.add_argument(
        "--base-null-space-weight",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 1.0),
        metavar=("X", "Y", "YAW"),
        help="Global soft attraction of the planar base toward its retract pose.",
    )
    args = parser.parse_args()
    for name, values in (
        ("base-distance-weight", args.base_distance_weight),
        ("base-null-space-weight", args.base_null_space_weight),
    ):
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"{name} values must be finite and positive")

    source_urdf = args.g2_root / "assets" / "urdf" / "g2_crsB_swiftpicker_curobo.urdf"
    generated_urdf = args.output_dir / "g2_crsB_swiftpicker_automoma_planar.urdf"
    build_planar_g2_urdf(source_urdf, generated_urdf)
    base_bounds = infer_link_visual_bounds(source_urdf, "base_link")
    base_collision_spheres = fit_bounds_with_spheres(base_bounds)

    outputs = {
        "urdf": str(generated_urdf),
        "urdf_sha256": _sha256(generated_urdf),
        "mesh_root": str(source_urdf.parent),
        "base_bounds": {"minimum": base_bounds.minimum, "maximum": base_bounds.maximum},
        "base_collision_sphere_count": len(base_collision_spheres),
        "base_regularization": {
            "distance_weight": args.base_distance_weight,
            "null_space_weight": args.base_null_space_weight,
            "semantics": "global_soft_cost_not_fixed_motion_share",
        },
        "hands": {},
    }
    for hand in (Hand.LEFT, Hand.RIGHT):
        source_yaml = args.g2_root / "curobo" / f"whole_body_{hand.value}.yml"
        with source_yaml.open("r", encoding="utf-8") as stream:
            source_config = yaml.safe_load(stream)
        generated_config = make_g2_curobo_config(
            source_config,
            generated_urdf,
            hand,
            asset_root_path=source_urdf.parent,
            base_collision_spheres=base_collision_spheres,
            base_distance_weight=args.base_distance_weight,
            base_null_space_weight=args.base_null_space_weight,
        )
        output_yaml = args.output_dir / f"g2_automoma_{hand.value}.yml"
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        with output_yaml.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(generated_config, stream, sort_keys=False)
        outputs["hands"][hand.value] = {
            "config": str(output_yaml),
            "config_sha256": _sha256(output_yaml),
            "source": str(source_yaml),
        }

    manifest_path = args.output_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
