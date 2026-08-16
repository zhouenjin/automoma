"""Convert generic contact candidates into grasp-conditioned EE poses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .transform_math import Matrix4, freeze_matrix, transform_point, transform_vector


@dataclass(frozen=True)
class ContactCandidate:
    """Minimal candidate interface consumed by the external-first planner."""

    hypothesis_id: str
    source_usd: str
    target_joint_path: str
    contact_center_source: Tuple[float, float, float]
    contact_points_source: Tuple[Tuple[float, float, float], ...]
    approach_source: Tuple[float, float, float]
    closing_source: Tuple[float, float, float]
    gripper_y_source: Tuple[float, float, float]
    handle_approach_depth: float
    gripper_width: float
    pre_ik_score: float
    automatic_rank: int

    def validate(self) -> None:
        basis = np.column_stack((self.closing_source, self.gripper_y_source, self.approach_source))
        if not np.allclose(basis.T @ basis, np.eye(3), atol=5e-3):
            raise ValueError(f"{self.hypothesis_id}: candidate frame is not orthonormal")
        if np.linalg.det(basis) < 0.99:
            raise ValueError(f"{self.hypothesis_id}: candidate frame is not right-handed")
        if self.handle_approach_depth <= 0.0:
            raise ValueError(f"{self.hypothesis_id}: handle depth must be positive")
        if self.gripper_width <= 0.0:
            raise ValueError(f"{self.hypothesis_id}: gripper width must be positive")

    def place(
        self,
        *,
        handle_anchor_world: Sequence[float],
        desired_approach_world: Sequence[float],
        capture_depth_fraction: float,
        precontact_distance: float = 0.12,
    ) -> "PlacedContactCandidate":
        """Place an asset by aligning its approach with a global workspace ray."""

        self.validate()
        if not 0.0 <= capture_depth_fraction <= 1.0:
            raise ValueError("capture_depth_fraction must be in [0, 1]")
        source_approach = np.asarray(self.approach_source, dtype=np.float64)
        desired_approach = np.asarray(desired_approach_world, dtype=np.float64)
        source_yaw = math.atan2(source_approach[1], source_approach[0])
        desired_yaw = math.atan2(desired_approach[1], desired_approach[0])
        yaw = desired_yaw - source_yaw
        cosine, sine = math.cos(yaw), math.sin(yaw)
        world_from_source = np.eye(4, dtype=np.float64)
        world_from_source[:3, :3] = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rotated_contact = world_from_source[:3, :3] @ np.asarray(self.contact_center_source, dtype=np.float64)
        world_from_source[:3, 3] = np.asarray(handle_anchor_world, dtype=np.float64) - rotated_contact

        closing = np.asarray(transform_vector(world_from_source, self.closing_source), dtype=np.float64)
        gripper_y = np.asarray(transform_vector(world_from_source, self.gripper_y_source), dtype=np.float64)
        approach = np.asarray(transform_vector(world_from_source, self.approach_source), dtype=np.float64)
        rotation = np.column_stack((closing, gripper_y, approach))
        contact_center = np.asarray(transform_point(world_from_source, self.contact_center_source), dtype=np.float64)
        ee_position = contact_center - approach * self.handle_approach_depth * capture_depth_fraction
        world_from_ee = np.eye(4, dtype=np.float64)
        world_from_ee[:3, :3] = rotation
        world_from_ee[:3, 3] = ee_position
        world_from_precontact = world_from_ee.copy()
        world_from_precontact[:3, 3] -= approach * float(precontact_distance)
        return PlacedContactCandidate(
            candidate=self,
            capture_depth_fraction=float(capture_depth_fraction),
            world_from_source=freeze_matrix(world_from_source),
            world_from_ee_contact=freeze_matrix(world_from_ee),
            world_from_ee_precontact=freeze_matrix(world_from_precontact),
        )


@dataclass(frozen=True)
class PlacedContactCandidate:
    candidate: ContactCandidate
    capture_depth_fraction: float
    world_from_source: Matrix4
    world_from_ee_contact: Matrix4
    world_from_ee_precontact: Matrix4


def _vector(values: Sequence[Any]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"expected a 3-vector, got {values!r}")
    return tuple(float(value) for value in values)


def _candidate_from_mapping(data: Mapping[str, Any]) -> ContactCandidate:
    candidate = ContactCandidate(
        hypothesis_id=str(data["hypothesis_id"]),
        source_usd=str(data["source_usd"]),
        target_joint_path=str(data["target_joint_path"]),
        contact_center_source=_vector(data["contact_center_source_m"]),
        contact_points_source=tuple(_vector(point) for point in data["contact_points_source_m"]),
        approach_source=_vector(data["approach_direction_source"]),
        closing_source=_vector(data["closing_direction_source"]),
        gripper_y_source=_vector(data["gripper_y_direction_source"]),
        handle_approach_depth=float(data["handle_approach_depth_m"]),
        gripper_width=float(data["gripper_width_m"]),
        pre_ik_score=float(data.get("pre_ik_score", 0.0)),
        automatic_rank=int(data.get("automatic_pre_ik_rank", 2**31 - 1)),
    )
    candidate.validate()
    return candidate


def load_contact_candidates(
    path: Path, *, source_usd: Path, target_joint_path: str
) -> Tuple[ContactCandidate, ...]:
    """Load and deterministically sort candidates matching a task source."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("candidate file must contain a list or a {'candidates': [...]} object")
    expected_source = str(source_usd.resolve())
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(Path(str(row.get("source_usd", ""))).resolve()) != expected_source:
            continue
        if str(row.get("target_joint_path")) != target_joint_path:
            continue
        candidates.append(_candidate_from_mapping(row))
    if not candidates:
        raise ValueError(f"no candidates match source={expected_source!r}, joint={target_joint_path!r}")
    return tuple(sorted(candidates, key=lambda item: (item.automatic_rank, -item.pre_ik_score, item.hypothesis_id)))
