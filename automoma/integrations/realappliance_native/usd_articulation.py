"""USD articulation descriptors used by the AutoMoMa RealAppliance adapter.

The module intentionally contains no G2- or asset-id-specific logic.  The USD
reader lives in the Isaac-only tool; these dependency-free records can be
validated and ranked in the planner environment and in unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


_MOVABLE_TYPES = {"revolute", "prismatic"}


@dataclass(frozen=True)
class UsdJointDescriptor:
    """One physics joint extracted from an appliance USD."""

    path: str
    joint_type: str
    parent_body: str | None
    child_body: str | None
    axis: str | None
    lower_limit: float | None
    upper_limit: float | None
    local_position_parent: tuple[float, float, float]
    local_position_child: tuple[float, float, float]
    local_rotation_parent_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    local_rotation_child_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def movable(self) -> bool:
        return self.joint_type in _MOVABLE_TYPES

    @property
    def range(self) -> float | None:
        if self.lower_limit is None or self.upper_limit is None:
            return None
        return float(self.upper_limit - self.lower_limit)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UsdMeshGeometry:
    """World-aligned geometry summary for one USD mesh."""

    path: str
    rigid_body: str | None
    world_bounds_min: tuple[float, float, float]
    world_bounds_max: tuple[float, float, float]
    point_count: int
    rigid_body_bounds_min: tuple[float, float, float] | None = None
    rigid_body_bounds_max: tuple[float, float, float] | None = None

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(
            max(0.0, upper - lower)
            for lower, upper in zip(self.world_bounds_min, self.world_bounds_max)
        )

    @property
    def volume(self) -> float:
        x, y, z = self.extent
        return x * y * z

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["extent"] = self.extent
        return value

    @property
    def rigid_body_extent(self) -> tuple[float, float, float] | None:
        if self.rigid_body_bounds_min is None or self.rigid_body_bounds_max is None:
            return None
        return tuple(
            max(0.0, upper - lower)
            for lower, upper in zip(self.rigid_body_bounds_min, self.rigid_body_bounds_max)
        )


@dataclass(frozen=True)
class OpenJointComponent:
    """A movable child link plus every rigid body fixed beneath it."""

    joint: UsdJointDescriptor
    rigid_bodies: tuple[str, ...]
    mesh_paths: tuple[str, ...]
    world_bounds_min: tuple[float, float, float] | None
    world_bounds_max: tuple[float, float, float] | None
    point_count: int

    @property
    def extent(self) -> tuple[float, float, float]:
        if self.world_bounds_min is None or self.world_bounds_max is None:
            return (0.0, 0.0, 0.0)
        return tuple(
            max(0.0, upper - lower)
            for lower, upper in zip(self.world_bounds_min, self.world_bounds_max)
        )

    @property
    def geometry_score(self) -> float:
        """Coarse, asset-agnostic priority before learned grasp scoring.

        Large articulated panels and drawers are evaluated before tiny knobs or
        buttons.  This score never decides success or suppresses a mechanically
        valid joint; the native grasp generator still evaluates every component.
        """

        x, y, z = self.extent
        surface_proxy = x * y + y * z + z * x
        return surface_proxy + 1.0e-7 * float(self.point_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_path": self.joint.path,
            "joint_type": self.joint.joint_type,
            "rigid_bodies": list(self.rigid_bodies),
            "mesh_paths": list(self.mesh_paths),
            "world_bounds_min": self.world_bounds_min,
            "world_bounds_max": self.world_bounds_max,
            "extent": self.extent,
            "point_count": self.point_count,
            "geometry_score": self.geometry_score,
        }


@dataclass(frozen=True)
class RealApplianceUsdManifest:
    """Planner-facing inventory extracted from one RealAppliance USD."""

    asset_id: str
    source_usd: str
    joints: tuple[UsdJointDescriptor, ...]
    mesh_paths: tuple[str, ...]
    rigid_body_paths: tuple[str, ...]
    mesh_geometries: tuple[UsdMeshGeometry, ...] = ()
    meters_per_unit: float = 1.0

    @property
    def openable_joints(self) -> tuple[UsdJointDescriptor, ...]:
        return tuple(joint for joint in self.joints if joint.movable)

    def to_dict(self) -> dict[str, object]:
        components = build_open_joint_components(self)
        return {
            "schema_version": "automoma.realappliance.usd_manifest.v2",
            "provenance": {
                "pipeline": "automoma_native",
                "g2_inputs_used": False,
            },
            "asset_id": self.asset_id,
            "source_usd": self.source_usd,
            "meters_per_unit": self.meters_per_unit,
            "joints": [joint.to_dict() for joint in self.joints],
            "openable_joint_paths": [joint.path for joint in self.openable_joints],
            "mesh_paths": list(self.mesh_paths),
            "rigid_body_paths": list(self.rigid_body_paths),
            "mesh_geometries": [geometry.to_dict() for geometry in self.mesh_geometries],
            "open_joint_components": [component.to_dict() for component in components],
        }


def _fixed_descendants(
    root_body: str, joints: Iterable[UsdJointDescriptor]
) -> tuple[str, ...]:
    children: dict[str, list[str]] = {}
    for joint in joints:
        if joint.joint_type != "fixed" or not joint.parent_body or not joint.child_body:
            continue
        children.setdefault(joint.parent_body, []).append(joint.child_body)

    ordered = []
    stack = [root_body]
    seen = set()
    while stack:
        body = stack.pop()
        if body in seen:
            continue
        seen.add(body)
        ordered.append(body)
        stack.extend(reversed(children.get(body, ())))
    return tuple(ordered)


def build_open_joint_components(
    manifest: RealApplianceUsdManifest,
) -> tuple[OpenJointComponent, ...]:
    """Build geometry components for every openable joint without ID rules."""

    components = []
    for joint in choose_open_joint_candidates(manifest.joints):
        assert joint.child_body is not None
        bodies = _fixed_descendants(joint.child_body, manifest.joints)
        body_set = set(bodies)
        geometries = tuple(
            geometry
            for geometry in manifest.mesh_geometries
            if geometry.rigid_body in body_set
        )
        bounds_min = None
        bounds_max = None
        if geometries:
            bounds_min = tuple(
                min(geometry.world_bounds_min[index] for geometry in geometries)
                for index in range(3)
            )
            bounds_max = tuple(
                max(geometry.world_bounds_max[index] for geometry in geometries)
                for index in range(3)
            )
        components.append(
            OpenJointComponent(
                joint=joint,
                rigid_bodies=bodies,
                mesh_paths=tuple(geometry.path for geometry in geometries),
                world_bounds_min=bounds_min,
                world_bounds_max=bounds_max,
                point_count=sum(geometry.point_count for geometry in geometries),
            )
        )
    return tuple(sorted(components, key=lambda value: value.geometry_score, reverse=True))


def choose_open_joint_candidates(
    joints: Iterable[UsdJointDescriptor],
    *,
    minimum_range: float = 1.0e-5,
) -> tuple[UsdJointDescriptor, ...]:
    """Return all mechanically meaningful open candidates without ID branches.

    Selection of a final target is deliberately deferred until fresh grasp
    hypotheses are generated on each child link.  This function only rejects
    fixed joints, missing child links, reversed/degenerate limits, and therefore
    cannot silently encode per-asset target choices.
    """

    candidates = []
    for joint in joints:
        if not joint.movable or not joint.child_body:
            continue
        joint_range = joint.range
        if joint_range is not None and joint_range <= minimum_range:
            continue
        candidates.append(joint)
    return tuple(candidates)


def descriptor_from_mapping(value: Mapping[str, object]) -> UsdJointDescriptor:
    """Create a descriptor from an Isaac-side JSON-compatible record."""

    def vector(name: str) -> tuple[float, float, float]:
        raw = value.get(name, (0.0, 0.0, 0.0))
        if not isinstance(raw, Sequence) or len(raw) != 3:
            raise ValueError(f"{name} must contain three values")
        return tuple(float(component) for component in raw)

    def quaternion(name: str) -> tuple[float, float, float, float]:
        raw = value.get(name, (1.0, 0.0, 0.0, 0.0))
        if not isinstance(raw, Sequence) or len(raw) != 4:
            raise ValueError(f"{name} must contain four values")
        return tuple(float(component) for component in raw)

    return UsdJointDescriptor(
        path=str(value["path"]),
        joint_type=str(value["joint_type"]),
        parent_body=(str(value["parent_body"]) if value.get("parent_body") else None),
        child_body=(str(value["child_body"]) if value.get("child_body") else None),
        axis=(str(value["axis"]) if value.get("axis") else None),
        lower_limit=(
            float(value["lower_limit"]) if value.get("lower_limit") is not None else None
        ),
        upper_limit=(
            float(value["upper_limit"]) if value.get("upper_limit") is not None else None
        ),
        local_position_parent=vector("local_position_parent"),
        local_position_child=vector("local_position_child"),
        local_rotation_parent_wxyz=quaternion("local_rotation_parent_wxyz"),
        local_rotation_child_wxyz=quaternion("local_rotation_child_wxyz"),
    )


def manifest_from_mapping(value: Mapping[str, object]) -> RealApplianceUsdManifest:
    """Restore a manifest written by the Isaac-side inventory tool."""

    geometries = []
    for raw in value.get("mesh_geometries", ()):  # type: ignore[union-attr]
        geometry = dict(raw)

        def optional_vector(name: str):
            data = geometry.get(name)
            return tuple(float(item) for item in data) if data is not None else None

        geometries.append(
            UsdMeshGeometry(
                path=str(geometry["path"]),
                rigid_body=(str(geometry["rigid_body"]) if geometry.get("rigid_body") else None),
                world_bounds_min=tuple(float(item) for item in geometry["world_bounds_min"]),
                world_bounds_max=tuple(float(item) for item in geometry["world_bounds_max"]),
                point_count=int(geometry["point_count"]),
                rigid_body_bounds_min=optional_vector("rigid_body_bounds_min"),
                rigid_body_bounds_max=optional_vector("rigid_body_bounds_max"),
            )
        )
    return RealApplianceUsdManifest(
        asset_id=str(value["asset_id"]),
        source_usd=str(value["source_usd"]),
        joints=tuple(descriptor_from_mapping(raw) for raw in value.get("joints", ())),  # type: ignore[arg-type]
        mesh_paths=tuple(str(item) for item in value.get("mesh_paths", ())),
        rigid_body_paths=tuple(str(item) for item in value.get("rigid_body_paths", ())),
        mesh_geometries=tuple(geometries),
        meters_per_unit=float(value.get("meters_per_unit", 1.0)),
    )
