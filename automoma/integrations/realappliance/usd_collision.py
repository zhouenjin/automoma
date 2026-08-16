"""Build placed cuRobo collision worlds from RealAppliance USD geometry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Sequence, Tuple

import numpy as np

from .transform_math import matrix_to_pose_wxyz, pose_wxyz_to_matrix


@dataclass(frozen=True)
class PlacedCollisionWorld:
    """A cuRobo world plus provenance needed for failure reports."""

    world: Any
    included_mesh_paths: Tuple[str, ...]
    excluded_mesh_paths: Tuple[str, ...]
    moving_cluster_paths: Tuple[str, ...]


def fixed_body_cluster(usd_path: Path, seed_body_path: str) -> FrozenSet[str]:
    """Find bodies rigidly connected to ``seed_body_path`` through fixed joints."""

    try:
        from pxr import Usd
    except ImportError as error:  # pragma: no cover - remote compatibility environment owns pxr
        raise RuntimeError("fixed-body extraction requires the pxr Python package") from error

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"could not open USD stage {usd_path}")
    adjacency: dict[str, set[str]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        body0 = [str(path) for path in prim.GetRelationship("physics:body0").GetTargets()]
        body1 = [str(path) for path in prim.GetRelationship("physics:body1").GetTargets()]
        if len(body0) != 1 or len(body1) != 1:
            continue
        adjacency.setdefault(body0[0], set()).add(body1[0])
        adjacency.setdefault(body1[0], set()).add(body0[0])
    visited = {seed_body_path}
    frontier = [seed_body_path]
    while frontier:
        body = frontier.pop()
        for neighbour in adjacency.get(body, ()):
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return frozenset(visited)


def _is_excluded(path: str, excluded_paths: Iterable[str]) -> bool:
    return any(path == excluded or path.startswith(excluded.rstrip("/") + "/") for excluded in excluded_paths)


def load_placed_collision_world(
    usd_path: Path,
    world_from_source: Sequence[Sequence[float]],
    *,
    exclude_paths: Iterable[str] = (),
    moving_cluster_paths: Iterable[str] = (),
) -> PlacedCollisionWorld:
    """Load USD meshes, apply an asset placement, and remove requested bodies."""

    try:
        from curobo.geom.types import WorldConfig
        from curobo.util.usd_helper import UsdHelper
    except ImportError as error:  # pragma: no cover - remote compatibility environment owns cuRobo
        raise RuntimeError("collision-world extraction requires cuRobo") from error

    placement = np.asarray(world_from_source, dtype=np.float64)
    helper = UsdHelper()
    helper.load_stage_from_file(str(usd_path))
    source_world = helper.get_obstacles_from_stage().get_collision_check_world()
    exclusions = tuple(str(path) for path in exclude_paths)
    included = []
    included_paths = []
    excluded_mesh_paths = []
    for source_mesh in source_world.mesh:
        path = str(source_mesh.name)
        if _is_excluded(path, exclusions):
            excluded_mesh_paths.append(path)
            continue
        mesh = copy.deepcopy(source_mesh)
        mesh.pose = matrix_to_pose_wxyz(placement @ pose_wxyz_to_matrix(mesh.pose))
        included.append(mesh)
        included_paths.append(path)
    if not included:
        raise ValueError("collision filtering removed every USD mesh")
    return PlacedCollisionWorld(
        world=WorldConfig(mesh=included),
        included_mesh_paths=tuple(included_paths),
        excluded_mesh_paths=tuple(excluded_mesh_paths),
        moving_cluster_paths=tuple(sorted(str(path) for path in moving_cluster_paths)),
    )
