"""Small rigid-transform helpers shared by the USD and AKR adapters.

Matrices in this module use the conventional column-vector representation:
``world_point = world_from_local @ local_point``.  USD's ``Gf.Matrix4d`` is
transposed when it enters this module because USD stores translation in the
last row and applies matrices to row vectors.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .akr_adapter import TransformRPY


Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


def as_matrix4(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and return a floating-point 4x4 matrix."""

    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {result.shape}")
    return result


def freeze_matrix(matrix: Sequence[Sequence[float]]) -> Matrix4:
    """Convert a matrix into an immutable, JSON-friendly representation."""

    result = as_matrix4(matrix)
    return tuple(tuple(float(value) for value in row) for row in result)  # type: ignore[return-value]


def invert_rigid(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Invert a rigid transform while rejecting non-rigid scale/shear."""

    transform = as_matrix4(matrix)
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("transform rotation is not orthonormal")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ transform[:3, 3])
    return inverse


def transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> Tuple[float, float, float]:
    """Transform a 3-D point."""

    vector = np.asarray(tuple(point) + (1.0,), dtype=np.float64)
    result = as_matrix4(matrix) @ vector
    return tuple(float(value) for value in result[:3])


def transform_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate a 3-D direction without applying translation."""

    result = as_matrix4(matrix)[:3, :3] @ np.asarray(vector, dtype=np.float64)
    return tuple(float(value) for value in result)


def axis_motion_transform(axis: Sequence[float], displacement: float, *, revolute: bool) -> np.ndarray:
    """Create a revolute or prismatic joint transform in the joint frame."""

    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("joint axis has zero length")
    direction /= norm
    result = np.eye(4, dtype=np.float64)
    if not revolute:
        result[:3, 3] = direction * float(displacement)
        return result
    x, y, z = direction
    cosine = math.cos(displacement)
    sine = math.sin(displacement)
    one_minus_cosine = 1.0 - cosine
    result[:3, :3] = np.asarray(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )
    return result


def rotation_matrix_to_quaternion_wxyz(matrix: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Convert a rotation or homogeneous matrix to a normalized WXYZ quaternion."""

    array = np.asarray(matrix, dtype=np.float64)
    rotation = array[:3, :3] if array.shape == (4, 4) else array
    if rotation.shape != (3, 3):
        raise ValueError(f"expected a 3x3 or 4x4 matrix, got {array.shape}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("rotation is not orthonormal")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion_array = np.asarray(quaternion, dtype=np.float64)
    quaternion_array /= np.linalg.norm(quaternion_array)
    if quaternion_array[0] < 0.0:
        quaternion_array *= -1.0
    return tuple(float(value) for value in quaternion_array)


def quaternion_transform(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Build a homogeneous transform from translation and a WXYZ quaternion."""

    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def pose_wxyz_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` to a homogeneous matrix."""

    if len(pose) != 7:
        raise ValueError(f"expected a 7-D pose, got {pose!r}")
    return quaternion_transform(pose[:3], pose[3:])


def matrix_to_pose_wxyz(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Convert a homogeneous matrix to ``[x, y, z, qw, qx, qy, qz]``."""

    transform = as_matrix4(matrix)
    return [*transform[:3, 3].tolist(), *rotation_matrix_to_quaternion_wxyz(transform)]


def transform_rpy_to_matrix(transform: TransformRPY) -> np.ndarray:
    """Convert a URDF fixed-axis RPY transform into a homogeneous matrix."""

    roll, pitch, yaw = transform.rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(transform.xyz, dtype=np.float64)
    return result


def matrix_to_transform_rpy(matrix: Sequence[Sequence[float]]) -> TransformRPY:
    """Convert a rigid transform to URDF fixed-axis RPY coordinates."""

    transform = as_matrix4(matrix)
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("transform rotation is not orthonormal")
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return TransformRPY(
        xyz=tuple(float(value) for value in transform[:3, 3]),
        rpy=(float(roll), float(pitch), float(yaw)),
    )
