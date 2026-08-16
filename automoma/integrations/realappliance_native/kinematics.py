"""Joint-conditioned contact motion shared across handled assets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(
    value: np.ndarray | list[float] | tuple[float, ...], *, size: int
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"expected shape {(size,)}, got {array.shape}")
    return array


def graspgen_franka_to_panda_hand_matrix(matrix: np.ndarray) -> np.ndarray:
    """Map a GraspGen Franka pose onto the URDF ``panda_hand`` frame.

    GraspGen's canonical antipodal closing axis is local +X; Franka's fingers
    translate along local +/-Y. Both use local +Z for approach. Therefore the
    robot-frame pose is the GraspGen pose followed by a -90 degree local-Z
    rotation.
    """

    source = np.asarray(matrix, dtype=np.float64)
    if source.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose matrix, got {source.shape}")
    graspgen_from_panda_hand = np.eye(4, dtype=np.float64)
    graspgen_from_panda_hand[:3, :3] = Rotation.from_rotvec(
        np.asarray([0.0, 0.0, -np.pi / 2.0])
    ).as_matrix()
    return source @ graspgen_from_panda_hand


@dataclass(frozen=True)
class ArticulationSpec:
    joint_type: str
    axis_world: np.ndarray
    pivot_world_m: np.ndarray
    lower_limit: float
    upper_limit: float
    initial_position: float
    goal_position: float

    def __post_init__(self) -> None:
        if self.joint_type not in {"revolute", "prismatic"}:
            raise ValueError(f"unsupported joint_type: {self.joint_type}")
        axis = _vector(self.axis_world, size=3)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-9:
            raise ValueError("joint axis must be non-zero")
        object.__setattr__(self, "axis_world", axis / norm)
        object.__setattr__(self, "pivot_world_m", _vector(self.pivot_world_m, size=3))
        if not self.lower_limit <= self.initial_position <= self.upper_limit:
            raise ValueError("initial_position is outside joint limits")
        if not self.lower_limit <= self.goal_position <= self.upper_limit:
            raise ValueError("goal_position is outside joint limits")


@dataclass(frozen=True)
class JointConditionedContactPath:
    """Keep the same end-effector-to-handle transform while the joint moves."""

    articulation: ArticulationSpec
    contact_position_world_m: np.ndarray
    contact_orientation_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contact_position_world_m",
            _vector(self.contact_position_world_m, size=3),
        )
        quaternion = _vector(self.contact_orientation_wxyz, size=4)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-9:
            raise ValueError("contact quaternion must be non-zero")
        object.__setattr__(self, "contact_orientation_wxyz", quaternion / norm)

    def pose(self, joint_position: float) -> tuple[np.ndarray, np.ndarray]:
        delta = float(joint_position - self.articulation.initial_position)
        if self.articulation.joint_type == "prismatic":
            position = (
                self.contact_position_world_m + self.articulation.axis_world * delta
            )
            return position, self.contact_orientation_wxyz.copy()

        rotation = Rotation.from_rotvec(self.articulation.axis_world * delta)
        relative = self.contact_position_world_m - self.articulation.pivot_world_m
        position = self.articulation.pivot_world_m + rotation.apply(relative)
        initial_xyzw = np.roll(self.contact_orientation_wxyz, -1)
        orientation_xyzw = (rotation * Rotation.from_quat(initial_xyzw)).as_quat()
        return position, np.roll(orientation_xyzw, 1)

    def sample(self, steps: int) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
        if steps < 2:
            raise ValueError("steps must be at least two")
        values = np.linspace(
            self.articulation.initial_position, self.articulation.goal_position, steps,
        )
        return tuple((float(value), *self.pose(float(value))) for value in values)
