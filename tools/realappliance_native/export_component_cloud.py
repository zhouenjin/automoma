#!/usr/bin/env python3
"""Export fresh component-frame point clouds for native GraspGen inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--joint-path")
    parser.add_argument("--points-per-body", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": True, "width": 320, "height": 240})

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from automoma.integrations.realappliance_native import (  # noqa: E402
    build_open_joint_components,
    manifest_from_mapping,
)


def rigid_owner(prim, rigid_bodies: set[str]):
    cursor = prim
    while cursor and str(cursor.GetPath()) not in rigid_bodies:
        cursor = cursor.GetParent()
    return cursor if cursor else None


def mesh_triangles_in_component(stage, prim, component_prim, scale: float) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or ()
    counts = mesh.GetFaceVertexCountsAttr().Get() or ()
    indices = mesh.GetFaceVertexIndicesAttr().Get() or ()
    if not points or not counts or not indices:
        return np.empty((0, 3, 3), dtype=np.float64)
    mesh_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_to_component = UsdGeom.Xformable(component_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).GetInverse()
    vertices = np.asarray(
        [
            tuple(
                float(value)
                for value in world_to_component.Transform(
                    mesh_to_world.Transform(Gf.Vec3d(*(float(item) for item in point)))
                )
            )
            for point in points
        ],
        dtype=np.float64,
    ) * scale
    triangles = []
    cursor = 0
    for raw_count in counts:
        count = int(raw_count)
        face = [int(value) for value in indices[cursor : cursor + count]]
        cursor += count
        for offset in range(1, count - 1):
            triangles.append(vertices[[face[0], face[offset], face[offset + 1]]])
    return np.stack(triangles) if triangles else np.empty((0, 3, 3), dtype=np.float64)


def sample_triangles(triangles: np.ndarray, count: int, rng) -> tuple[np.ndarray, np.ndarray]:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    crosses = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(crosses, axis=1)
    valid = lengths > 1.0e-12
    triangles = triangles[valid]
    crosses = crosses[valid]
    lengths = lengths[valid]
    probabilities = lengths / lengths.sum()
    selected = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
    chosen = triangles[selected]
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    points = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (
        chosen[:, 2] - chosen[:, 0]
    )
    normals = crosses[selected] / lengths[selected, None]
    return points.astype(np.float32), normals.astype(np.float32)


def export() -> dict[str, object]:
    raw_manifest = json.loads(ARGS.manifest.read_text(encoding="utf-8"))
    manifest = manifest_from_mapping(raw_manifest)
    components = build_open_joint_components(manifest)
    if ARGS.joint_path:
        components = tuple(value for value in components if value.joint.path == ARGS.joint_path)
    if not components:
        raise RuntimeError("no matching open component")
    component = components[0]
    assert component.joint.child_body is not None
    stage = Usd.Stage.Open(manifest.source_usd)
    if stage is None:
        raise RuntimeError(f"failed to open {manifest.source_usd}")
    component_prim = stage.GetPrimAtPath(component.joint.child_body)
    component_to_world = np.asarray(
        UsdGeom.Xformable(component_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ),
        dtype=np.float64,
    ).T
    component_to_world[:3, 3] *= manifest.meters_per_unit
    axis_lookup = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    parent_prim = stage.GetPrimAtPath(component.joint.parent_body)
    parent_to_world_raw = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    parent_joint_rotation = Gf.Rotation(
        Gf.Quatd(
            component.joint.local_rotation_parent_wxyz[0],
            Gf.Vec3d(*component.joint.local_rotation_parent_wxyz[1:]),
        )
    )
    axis_parent = parent_joint_rotation.TransformDir(axis_lookup[component.joint.axis])
    axis_world = np.asarray(
        tuple(float(value) for value in parent_to_world_raw.TransformDir(axis_parent)),
        dtype=np.float64,
    )
    axis_world /= np.linalg.norm(axis_world)
    pivot_world = np.asarray(
        tuple(
            float(value)
            for value in parent_to_world_raw.Transform(
                Gf.Vec3d(*component.joint.local_position_parent)
            )
        ),
        dtype=np.float64,
    ) * manifest.meters_per_unit
    rigid_body_set = set(manifest.rigid_body_paths)
    # Sample every rigid body in the appliance in the movable-component frame.
    # The target component is used for GraspGen inference; the remaining bodies
    # are retained as collision context for grasp filtering, IK, and transit.
    body_to_triangles: dict[str, list[np.ndarray]] = {
        body: [] for body in manifest.rigid_body_paths
    }
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        owner = rigid_owner(prim, rigid_body_set)
        owner_path = str(owner.GetPath()) if owner else None
        if owner_path not in body_to_triangles:
            continue
        triangles = mesh_triangles_in_component(
            stage, prim, component_prim, manifest.meters_per_unit
        )
        if len(triangles):
            body_to_triangles[owner_path].append(triangles)

    rng = np.random.default_rng(ARGS.seed)
    def sample_bodies(body_paths: tuple[str, ...] | list[str]):
        point_blocks = []
        normal_blocks = []
        body_indices = []
        exported_bodies = []
        for body_path in body_paths:
            blocks = body_to_triangles.get(body_path, [])
            if not blocks:
                continue
            points, normals = sample_triangles(
                np.concatenate(blocks, axis=0), ARGS.points_per_body, rng
            )
            body_index = len(exported_bodies)
            exported_bodies.append(body_path)
            point_blocks.append(points)
            normal_blocks.append(normals)
            body_indices.append(np.full(len(points), body_index, dtype=np.int32))
        return point_blocks, normal_blocks, body_indices, exported_bodies

    point_blocks, normal_blocks, body_indices, exported_bodies = sample_bodies(
        component.rigid_bodies
    )
    if not point_blocks:
        raise RuntimeError("selected component contains no triangle geometry")
    static_body_paths = [
        path for path in manifest.rigid_body_paths if path not in component.rigid_bodies
    ]
    (
        scene_point_blocks,
        scene_normal_blocks,
        scene_body_indices,
        scene_exported_bodies,
    ) = sample_bodies(static_body_paths)
    scene_points = (
        np.concatenate(scene_point_blocks)
        if scene_point_blocks
        else np.empty((0, 3), dtype=np.float32)
    )
    scene_normals = (
        np.concatenate(scene_normal_blocks)
        if scene_normal_blocks
        else np.empty((0, 3), dtype=np.float32)
    )
    scene_indices = (
        np.concatenate(scene_body_indices)
        if scene_body_indices
        else np.empty((0,), dtype=np.int32)
    )
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARGS.output,
        points_component_m=np.concatenate(point_blocks),
        normals_component=np.concatenate(normal_blocks),
        body_index=np.concatenate(body_indices),
        body_paths=np.asarray(exported_bodies),
        scene_points_component_m=scene_points,
        scene_normals_component=scene_normals,
        scene_body_index=scene_indices,
        scene_body_paths=np.asarray(scene_exported_bodies),
        joint_path=np.asarray(component.joint.path),
        component_body_path=np.asarray(component.joint.child_body),
        component_to_world=component_to_world.astype(np.float32),
        joint_axis_world=axis_world.astype(np.float32),
        joint_pivot_world_m=pivot_world.astype(np.float32),
    )
    metadata = {
        "schema_version": "automoma.realappliance.component_cloud.v1",
        "provenance": {"pipeline": "automoma_native", "g2_inputs_used": False},
        "asset_id": manifest.asset_id,
        "source_usd": manifest.source_usd,
        "joint_path": component.joint.path,
        "component_body_path": component.joint.child_body,
        "body_paths": exported_bodies,
        "scene_body_paths": scene_exported_bodies,
        "point_count": int(sum(len(value) for value in point_blocks)),
        "scene_point_count": int(len(scene_points)),
        "component_to_world": component_to_world.tolist(),
        "joint_axis_world": axis_world.tolist(),
        "joint_pivot_world_m": pivot_world.tolist(),
        "output": str(ARGS.output),
    }
    ARGS.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


try:
    print(json.dumps(export()))
finally:
    APP.close()
