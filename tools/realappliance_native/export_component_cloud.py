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
    rigid_body_set = set(manifest.rigid_body_paths)
    body_to_triangles: dict[str, list[np.ndarray]] = {body: [] for body in component.rigid_bodies}
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
    point_blocks = []
    normal_blocks = []
    body_indices = []
    exported_bodies = []
    for body_path, blocks in body_to_triangles.items():
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
    if not point_blocks:
        raise RuntimeError("selected component contains no triangle geometry")
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARGS.output,
        points_component_m=np.concatenate(point_blocks),
        normals_component=np.concatenate(normal_blocks),
        body_index=np.concatenate(body_indices),
        body_paths=np.asarray(exported_bodies),
        joint_path=np.asarray(component.joint.path),
        component_body_path=np.asarray(component.joint.child_body),
    )
    metadata = {
        "schema_version": "automoma.realappliance.component_cloud.v1",
        "provenance": {"pipeline": "automoma_native", "g2_inputs_used": False},
        "asset_id": manifest.asset_id,
        "source_usd": manifest.source_usd,
        "joint_path": component.joint.path,
        "component_body_path": component.joint.child_body,
        "body_paths": exported_bodies,
        "point_count": int(sum(len(value) for value in point_blocks)),
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
