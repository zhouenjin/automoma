"""Project-level contracts for physically valid RealAppliance episodes.

These contracts deliberately live outside the legacy G2 implementation.  They
describe the result that an AutoMoMa-backed planner must produce, rather than
encoding per-asset controller rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class JointKind(str, Enum):
    """Supported target articulation types for the current ``open`` stage."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


@dataclass(frozen=True)
class ArticulationTaskSpec:
    """Normalized articulation metadata, independent of an asset identifier."""

    joint_name: str
    target_link: str
    handle_link: str
    joint_kind: JointKind
    axis: Tuple[float, float, float]
    pivot: Tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    initial_position: float

    @property
    def planning_fraction(self) -> float:
        """Target used by trajectory synthesis, intentionally above acceptance."""

        return 0.80 if self.joint_kind is JointKind.REVOLUTE else 0.90

    @property
    def acceptance_fraction(self) -> float:
        """Strict dataset-ready threshold agreed for the G2 line."""

        return 0.70 if self.joint_kind is JointKind.REVOLUTE else 0.80

    @property
    def open_limit(self) -> float:
        """Infer the opening endpoint as the joint limit farthest from reset state."""

        lower_distance = abs(self.initial_position - self.lower_limit)
        upper_distance = abs(self.upper_limit - self.initial_position)
        return self.upper_limit if upper_distance >= lower_distance else self.lower_limit

    def position_at_fraction(self, fraction: float) -> float:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        return self.initial_position + fraction * (self.open_limit - self.initial_position)

    def progress_fraction(self, joint_position: float) -> float:
        denominator = self.open_limit - self.initial_position
        if abs(denominator) < 1e-9:
            raise ValueError("initial position and inferred open limit are identical")
        return max(0.0, min(1.0, (joint_position - self.initial_position) / denominator))


@dataclass(frozen=True)
class PhysicsRunPolicy:
    """Guardrails that prevent kinematic replay from being called a success."""

    set_object_state_during_episode: bool = False
    disable_target_collision: bool = False
    create_attachment: bool = False
    write_target_joint: bool = False
    require_measured_contact: bool = True
    require_penetration_audit: bool = True

    def validate(self) -> None:
        forbidden = []
        if self.set_object_state_during_episode:
            forbidden.append("set_object_state_during_episode")
        if self.disable_target_collision:
            forbidden.append("disable_target_collision")
        if self.create_attachment:
            forbidden.append("create_attachment")
        if self.write_target_joint:
            forbidden.append("write_target_joint")
        if forbidden:
            raise ValueError("non-physical execution options are forbidden: " + ", ".join(forbidden))
        if not self.require_measured_contact:
            raise ValueError("measured robot-handle contact is required")
        if not self.require_penetration_audit:
            raise ValueError("penetration auditing is required")
