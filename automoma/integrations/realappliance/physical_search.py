"""Automatic ordering of learned-contact AKR trials for physical validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PhysicalExecutionTrial:
    attempt_id: str
    attempt_dir: str
    automatic_rank: int
    pre_ik_score: float
    hand: str
    capture_depth_fraction: float
    trajectory_index: int
    trajectory_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def enumerate_physical_trials(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> tuple[PhysicalExecutionTrial, ...]:
    """Expand every valid planned path without requiring a hand or rank input."""

    trials = []
    for attempt_order, attempt in enumerate(report.get("attempts", ())):
        if not attempt.get("success"):
            continue
        planning = attempt.get("planning", {})
        selection = planning.get("automatic_trajectory_selection", {})
        trajectory_indices: Sequence[int] = selection.get("ranked_valid_trajectory_indices", ())
        scores: Sequence[float] = selection.get("scores_per_trajectory", ())
        attempt_id = str(attempt["attempt_id"])
        attempt_dir = report_path.parent / attempt_id
        for path_order, trajectory_index_raw in enumerate(trajectory_indices):
            trajectory_index = int(trajectory_index_raw)
            if trajectory_index < 0 or trajectory_index >= len(scores):
                raise ValueError(f"invalid trajectory index {trajectory_index} for {attempt_id}")
            trial = PhysicalExecutionTrial(
                attempt_id=attempt_id,
                attempt_dir=str(attempt_dir.resolve()),
                automatic_rank=int(attempt.get("automatic_rank", attempt_order)),
                pre_ik_score=float(attempt.get("pre_ik_score", 0.0)),
                hand=str(attempt["hand"]),
                capture_depth_fraction=float(attempt["capture_depth_fraction"]),
                trajectory_index=trajectory_index,
                trajectory_score=float(scores[trajectory_index]),
            )
            trials.append((attempt_order, path_order, trial))
    # Search breadth before exhausting trajectory variants of one hypothesis.
    # Round zero executes the best valid AKR path for every learned-contact and
    # hand hypothesis, round one executes each hypothesis' second-best path,
    # and so on.  This preserves the learned candidate rank within each round
    # while preventing a large path pool for one grasp from consuming the
    # complete physical budget.
    trials.sort(
        key=lambda item: (
            item[1],
            item[2].automatic_rank,
            item[2].trajectory_score,
            -item[2].pre_ik_score,
            item[0],
        )
    )
    return tuple(item[2] for item in trials)
