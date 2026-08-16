"""Hand/base hypothesis generation used only as optimizer initialization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from .g2_adapter import Hand


@dataclass(frozen=True)
class BasePoseSeed:
    x: float
    y: float
    yaw: float
    standoff: float
    lateral_offset: float


@dataclass(frozen=True)
class InteractionHypothesis:
    hand: Hand
    base_seed: BasePoseSeed
    grasp_index: int


def _normalize_xy(vector: Sequence[float]) -> Tuple[float, float]:
    if len(vector) < 2:
        raise ValueError("an XY direction requires at least two values")
    norm = math.hypot(float(vector[0]), float(vector[1]))
    if norm < 1e-9:
        raise ValueError("outward normal has zero XY magnitude")
    return float(vector[0]) / norm, float(vector[1]) / norm


def generate_base_pose_seeds(
    handle_position: Sequence[float],
    outward_normal: Sequence[float],
    *,
    standoffs: Iterable[float] = (0.45, 0.60, 0.75),
    lateral_offsets: Iterable[float] = (-0.20, 0.0, 0.20),
    yaw_offsets: Iterable[float] = (-0.20, 0.0, 0.20),
) -> list[BasePoseSeed]:
    """Generate geometry-relative seeds without imposing a hard front fan."""

    nx, ny = _normalize_xy(outward_normal)
    tx, ty = -ny, nx
    hx, hy = float(handle_position[0]), float(handle_position[1])
    seeds: list[BasePoseSeed] = []
    for standoff in standoffs:
        if standoff <= 0:
            raise ValueError("standoffs must be positive")
        for lateral in lateral_offsets:
            x = hx + nx * standoff + tx * lateral
            y = hy + ny * standoff + ty * lateral
            facing = math.atan2(hy - y, hx - x)
            for yaw_offset in yaw_offsets:
                seeds.append(
                    BasePoseSeed(
                        x=x, y=y, yaw=facing + yaw_offset, standoff=float(standoff), lateral_offset=float(lateral),
                    )
                )
    return seeds


def generate_interaction_hypotheses(
    handle_position: Sequence[float], outward_normal: Sequence[float], grasp_indices: Iterable[int],
) -> list[InteractionHypothesis]:
    """Cross product of both hands, all base seeds, and all grasp candidates."""

    base_seeds = generate_base_pose_seeds(handle_position, outward_normal)
    grasps = list(grasp_indices)
    if not grasps:
        raise ValueError("at least one grasp candidate is required")
    return [
        InteractionHypothesis(hand=hand, base_seed=seed, grasp_index=grasp_index)
        for hand in (Hand.LEFT, Hand.RIGHT)
        for seed in base_seeds
        for grasp_index in grasps
    ]
