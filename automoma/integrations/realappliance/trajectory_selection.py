"""Automatic selection of valid whole-body AKR trajectories."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def weighted_cspace_path_scores(
    trajectories: Any,
    valid_mask: Sequence[bool],
    cspace_weights: Sequence[float],
) -> dict[str, Any]:
    """Rank valid paths by globally configured weighted cspace arc length.

    The score does not impose a base/manipulator motion share and contains no
    asset feature.  It simply applies the same planner cspace metric to every
    consecutive trajectory step.
    """

    paths = np.asarray(trajectories, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    weights = np.asarray(cspace_weights, dtype=np.float64).reshape(-1)
    if paths.ndim != 3:
        raise ValueError(f"trajectories must have shape [N, T, D], got {paths.shape}")
    if paths.shape[0] != len(valid):
        raise ValueError("valid_mask length does not match trajectory count")
    if paths.shape[2] != len(weights):
        raise ValueError("cspace weight count does not match trajectory DOF")
    if not np.all(np.isfinite(paths)) or not np.all(np.isfinite(weights)):
        raise ValueError("trajectory values and cspace weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("cspace weights must be positive")
    increments = np.diff(paths, axis=1) * weights[None, None, :]
    scores = np.sum(np.linalg.norm(increments, axis=-1), axis=1)
    ranked = [int(index) for index in np.argsort(scores) if valid[index]]
    selected = ranked[0] if ranked else None
    return {
        "metric": "globally_weighted_cspace_arc_length",
        "scores_per_trajectory": scores.tolist(),
        "ranked_valid_trajectory_indices": ranked,
        "selected_trajectory_index": selected,
        "selected_score": None if selected is None else float(scores[selected]),
    }
