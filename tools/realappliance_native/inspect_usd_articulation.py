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
    UsdMeshGeometry,
    choose_open_joint_candidates,
)


def relationship_target(relationship) -> str | None:
    targets = relationship.GetTargets()
    return str(targets[0]) if targets else None


def local_position(joint: UsdPhysics.Joint, body_index: int) -> tuple[float, float, float]:
    attribute = joint.GetLocalPos0Attr() if body_index == 0 else joint.GetLocalPos1Attr()
    value = attribute.Get()
    return tuple(float(component) for component in value) if value is not None else (0.0, 0.0, 0.0)


def local_rotation(joint: UsdPhysics.Joint, body_index: int) -> tuple[float, float, float, float]:
    attribute = joint.GetLocalRot0Attr() if body_index == 0 else joint.GetLocalRot1Attr()
    value = attribute.Get()
    if value is None:
        return (1.0, 0.0, 0.0, 0.0)
    imaginary = value.GetImaginary()
    return (float(value.GetReal()), *(float(component) for component in imaginary))


def transformed_bounds(points, transform) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    transformed = [transform.Transform(point) for point in points]
    return (
        tuple(min(float(point[index]) for point in transformed) for index in range(3)),
        tuple(max(float(point[index]) for point in transformed) for index in range(3)),
    )


def optional_float(attribute) -> float | None:
    value = attribute.Get()
    return float(value) if value is not None else None


def inspect() -> RealApplianceUsdManifest:
    stage = Usd.Stage.Open(str(ARGS.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {ARGS.usd}")

    joints = []
    meshes = []
    mesh_geometries = []
    rigid_bodies = []
    prims = tuple(stage.Traverse())
    rigid_body_set = {
        str(prim.GetPath())
        for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            mesh_path = str(prim.GetPath())
            meshes.append(mesh_path)
            owner = prim
            while owner and str(owner.GetPath()) not in rigid_body_set:
                owner = owner.GetParent()
            rigid_body = str(owner.GetPath()) if owner else None
            aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = aligned_range.GetMin()
            upper = aligned_range.GetMax()
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or ()
            owner_bounds_min = None
            owner_bounds_max = None
            if owner and points:
                mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
                world_to_owner = xform_cache.GetLocalToWorldTransform(owner).GetInverse()
                mesh_to_owner = mesh_to_world * world_to_owner
                owner_bounds_min, owner_bounds_max = transformed_bounds(points, mesh_to_owner)
            mesh_geometries.append(
                UsdMeshGeometry(
                    path=mesh_path,
                    rigid_body=rigid_body,
                    world_bounds_min=tuple(float(value) for value in lower),
                    world_bounds_max=tuple(float(value) for value in upper),
                    point_count=len(points),
                    rigid_body_bounds_min=owner_bounds_min,
                    rigid_body_bounds_max=owner_bounds_max,
                )
            )
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
                local_rotation_parent_wxyz=local_rotation(joint, 0),
                local_rotation_child_wxyz=local_rotation(joint, 1),
            )
        )

    manifest = RealApplianceUsdManifest(
        asset_id=ARGS.asset_id,
        source_usd=str(ARGS.usd.resolve()),
        joints=tuple(joints),
        mesh_paths=tuple(meshes),
        rigid_body_paths=tuple(rigid_bodies),
        mesh_geometries=tuple(mesh_geometries),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
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
