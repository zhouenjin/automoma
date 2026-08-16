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
class RealApplianceUsdManifest:
    """Planner-facing inventory extracted from one RealAppliance USD."""

    asset_id: str
    source_usd: str
    joints: tuple[UsdJointDescriptor, ...]
    mesh_paths: tuple[str, ...]
    rigid_body_paths: tuple[str, ...]

    @property
    def openable_joints(self) -> tuple[UsdJointDescriptor, ...]:
        return tuple(joint for joint in self.joints if joint.movable)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "automoma.realappliance.usd_manifest.v1",
            "provenance": {
                "pipeline": "automoma_native",
                "g2_inputs_used": False,
            },
            "asset_id": self.asset_id,
            "source_usd": self.source_usd,
            "joints": [joint.to_dict() for joint in self.joints],
            "openable_joint_paths": [joint.path for joint in self.openable_joints],
            "mesh_paths": list(self.mesh_paths),
            "rigid_body_paths": list(self.rigid_body_paths),
        }


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
    )
