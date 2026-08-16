#!/usr/bin/env python3
"""Extract a RealAppliance USD articulation manifest under Isaac Sim.

Run this with Isaac Sim's ``python.sh``.  The output is an AutoMoMa-native
planner input inventory; it does not import or consume the separate G2 project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from automoma.integrations.realappliance_native.usd_articulation import (  # noqa: E402
    RealApplianceUsdManifest,
    UsdJointDescriptor,
    choose_open_joint_candidates,
)


def relationship_target(relationship) -> str | None:
    targets = relationship.GetTargets()
    return str(targets[0]) if targets else None


def local_position(joint: UsdPhysics.Joint, body_index: int) -> tuple[float, float, float]:
    attribute = joint.GetLocalPos0Attr() if body_index == 0 else joint.GetLocalPos1Attr()
    value = attribute.Get()
    return tuple(float(component) for component in value) if value is not None else (0.0, 0.0, 0.0)


def optional_float(attribute) -> float | None:
    value = attribute.Get()
    return float(value) if value is not None else None


def inspect() -> RealApplianceUsdManifest:
    stage = Usd.Stage.Open(str(ARGS.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {ARGS.usd}")

    joints = []
    meshes = []
    rigid_bodies = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))

        joint_type = None
        axis = None
        lower_limit = None
        upper_limit = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed_joint = UsdPhysics.RevoluteJoint(prim)
            joint_type = "revolute"
            axis = str(typed_joint.GetAxisAttr().Get())
            lower_limit = optional_float(typed_joint.GetLowerLimitAttr())
            upper_limit = optional_float(typed_joint.GetUpperLimitAttr())
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed_joint = UsdPhysics.PrismaticJoint(prim)
            joint_type = "prismatic"
            axis = str(typed_joint.GetAxisAttr().Get())
            lower_limit = optional_float(typed_joint.GetLowerLimitAttr())
            upper_limit = optional_float(typed_joint.GetUpperLimitAttr())
        elif prim.IsA(UsdPhysics.FixedJoint):
            joint_type = "fixed"

        if joint_type is None:
            continue
        joint = UsdPhysics.Joint(prim)
        joints.append(
            UsdJointDescriptor(
                path=str(prim.GetPath()),
                joint_type=joint_type,
                parent_body=relationship_target(joint.GetBody0Rel()),
                child_body=relationship_target(joint.GetBody1Rel()),
                axis=axis,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                local_position_parent=local_position(joint, 0),
                local_position_child=local_position(joint, 1),
            )
        )

    manifest = RealApplianceUsdManifest(
        asset_id=ARGS.asset_id,
        source_usd=str(ARGS.usd.resolve()),
        joints=tuple(joints),
        mesh_paths=tuple(meshes),
        rigid_body_paths=tuple(rigid_bodies),
    )
    if not choose_open_joint_candidates(manifest.joints):
        raise RuntimeError("USD contains no mechanically meaningful open candidates")
    return manifest


try:
    manifest = inspect()
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(ARGS.output), "openable_joint_paths": manifest.to_dict()["openable_joint_paths"]}))
finally:
    APP.close()
