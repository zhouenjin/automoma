"""Strict physical-success contract for the native RealAppliance branch.

The contract deliberately audits *how* an object opened.  A large final joint
value alone is insufficient because a replay may write the target joint, use an
attachment, or pass through collision geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class OpenEpisodeSample:
    """One runtime sample used by :class:`OpenEpisodeContract`."""

    target_joint_position: float
    robot_target_was_written: bool = True
    object_target_was_written: bool = False
    attachment_active: bool = False
    robot_target_contact: bool = False
    maximum_penetration_m: float = 0.0


@dataclass(frozen=True)
class OpenEpisodeAudit:
    """Machine-readable verdict for one attempted episode."""

    success: bool
    range_fraction: float
    maximum_range_fraction: float
    contact_steps: int
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OpenEpisodeContract:
    """Acceptance rules shared by every handled-open asset.

    ``target_fraction`` is the controller target.  ``acceptance_fraction`` is
    the lower dataset-ready threshold.  Revolute doors default to 80%/70%; a
    prismatic task should construct the contract with 90%/80%.
    """

    lower_limit: float
    upper_limit: float
    opening_sign: int
    initial_position: float
    target_fraction: float = 0.80
    acceptance_fraction: float = 0.70
    maximum_allowed_penetration_m: float = 0.002
    minimum_contact_steps: int = 1

    def __post_init__(self) -> None:
        if self.upper_limit <= self.lower_limit:
            raise ValueError("upper_limit must exceed lower_limit")
        if self.opening_sign not in (-1, 1):
            raise ValueError("opening_sign must be -1 or +1")
        if not 0.0 < self.acceptance_fraction <= self.target_fraction <= 1.0:
            raise ValueError("require 0 < acceptance <= target <= 1")
        if self.maximum_allowed_penetration_m < 0.0:
            raise ValueError("maximum_allowed_penetration_m must be non-negative")
        if self.minimum_contact_steps < 1:
            raise ValueError("minimum_contact_steps must be positive")

    @property
    def joint_range(self) -> float:
        return self.upper_limit - self.lower_limit

    def range_fraction(self, position: float) -> float:
        directed_progress = self.opening_sign * (position - self.initial_position)
        return max(0.0, min(1.0, directed_progress / self.joint_range))

    def evaluate(self, samples: Iterable[OpenEpisodeSample]) -> OpenEpisodeAudit:
        trace = tuple(samples)
        if not trace:
            return OpenEpisodeAudit(False, 0.0, 0.0, 0, ("empty_trace",))

        fractions = tuple(
            self.range_fraction(sample.target_joint_position) for sample in trace
        )
        contact_steps = sum(sample.robot_target_contact for sample in trace)
        failure_reasons: list[str] = []

        if any(sample.object_target_was_written for sample in trace):
            failure_reasons.append("target_joint_commanded")
        if any(sample.attachment_active for sample in trace):
            failure_reasons.append("attachment_used")
        if any(
            sample.maximum_penetration_m > self.maximum_allowed_penetration_m
            for sample in trace
        ):
            failure_reasons.append("penetration_limit_exceeded")
        if contact_steps < self.minimum_contact_steps:
            failure_reasons.append("insufficient_robot_target_contact")
        if max(fractions) < self.acceptance_fraction:
            failure_reasons.append("insufficient_opening")

        return OpenEpisodeAudit(
            success=not failure_reasons,
            range_fraction=fractions[-1],
            maximum_range_fraction=max(fractions),
            contact_steps=contact_steps,
            failure_reasons=tuple(failure_reasons),
        )


def require_robot_only_action(action_width: int, robot_dof: int) -> None:
    """Reject trajectories that append object DOFs to the executable action."""

    if action_width != robot_dof:
        raise ValueError(
            f"native drive requires robot-only actions: action_width={action_width}, robot_dof={robot_dof}"
        )
