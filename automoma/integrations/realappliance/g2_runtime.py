"""Robot-specific G2/SwiftPicker drive definitions for Isaac Sim execution.

This module contains no appliance IDs and no task-selection policy.  It only
maps AutoMoMa planner coordinates to the supplied G2 robot's named actuators.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .g2_adapter import Hand


BODY_JOINT_NAMES = tuple(f"idx0{index}_body_joint{index}" for index in range(1, 6))
ARM_JOINT_NAMES = {
    Hand.LEFT: tuple(f"idx2{index}_arm_l_joint{index}" for index in range(1, 8)),
    Hand.RIGHT: tuple(f"idx6{index}_arm_r_joint{index}" for index in range(1, 8)),
}
GRIPPER_JOINT_NAMES = {
    Hand.LEFT: (
        "idx31_gripper_l_inner_joint1",
        "idx32_gripper_l_outer_joint1",
        "idx33_gripper_l_left_support_joint",
        "idx34_gripper_l_inner_joint2",
        "idx35_gripper_l_outer_joint2",
        "idx36_gripper_l_right_support_joint",
    ),
    Hand.RIGHT: (
        "idx71_gripper_r_inner_joint1",
        "idx72_gripper_r_outer_joint1",
        "idx73_gripper_r_left_support_joint",
        "idx74_gripper_r_inner_joint2",
        "idx75_gripper_r_outer_joint2",
        "idx76_gripper_r_right_support_joint",
    ),
}
GRIPPER_LINK_GROUPS = {
    Hand.LEFT: {
        "finger_a": ("gripper_l_left_inner_link", "gripper_l_left_outer_link", "gripper_l_left_support_link"),
        "finger_b": ("gripper_l_right_inner_link", "gripper_l_right_outer_link", "gripper_l_right_support_link"),
        "palm_wrist": ("gripper_l_base_link", "gripper_l_camera_link", "arm_l_end_link", "arm_l_link7"),
    },
    Hand.RIGHT: {
        "finger_a": ("gripper_r_left_inner_link", "gripper_r_left_outer_link", "gripper_r_left_support_link"),
        "finger_b": ("gripper_r_right_inner_link", "gripper_r_right_outer_link", "gripper_r_right_support_link"),
        "palm_wrist": ("gripper_r_base_link", "gripper_r_camera_link", "arm_r_end_link", "arm_r_link7"),
    },
}
GRIPPER_OPEN = np.zeros(6, dtype=np.float64)
GRIPPER_CLOSED = np.asarray((-0.85, -0.85, 0.85, 0.85, 0.85, 0.85), dtype=np.float64)
STEERING_JOINT_NAMES = (
    "idx111_chassis_lwheel_front_joint1",
    "idx131_chassis_rwheel_front_joint1",
    "idx141_chassis_rwheel_rear_joint1",
    "idx121_chassis_lwheel_rear_joint1",
)
WHEEL_JOINT_NAMES = (
    "idx112_chassis_lwheel_front_joint2",
    "idx132_chassis_rwheel_front_joint2",
    "idx142_chassis_rwheel_rear_joint2",
    "idx122_chassis_lwheel_rear_joint2",
)
MODULE_POSITIONS_BODY_M = np.asarray(
    ((0.23, 0.218), (0.23, -0.218), (-0.23, -0.218), (-0.23, 0.218)), dtype=np.float64
)
WHEEL_RADIUS_M = 0.07
MAXIMUM_WHEEL_SPEED_RAD_S = 20.9


def named_indices(expected: Sequence[str], available: Sequence[str]) -> np.ndarray:
    missing = [name for name in expected if name not in available]
    if missing:
        raise ValueError(f"G2 articulation is missing joints: {missing}")
    return np.asarray([available.index(name) for name in expected], dtype=np.int64)


def planner_robot_joint_names(hand: Hand) -> tuple[str, ...]:
    return BODY_JOINT_NAMES + ARM_JOINT_NAMES[hand]


def gripper_targets(close_fraction: float) -> np.ndarray:
    fraction = float(close_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("close_fraction must be in [0, 1]")
    return GRIPPER_OPEN + fraction * (GRIPPER_CLOSED - GRIPPER_OPEN)


def actuator_groups(hand: Hand, *, gripper_effort_multiplier: float = 2.0) -> tuple[dict[str, Any], ...]:
    effort_scale = float(gripper_effort_multiplier)
    if not math.isfinite(effort_scale) or effort_scale <= 0.0:
        raise ValueError("gripper effort multiplier must be finite and positive")
    arm = ARM_JOINT_NAMES[hand]
    gripper = GRIPPER_JOINT_NAMES[hand]
    return (
        {"names": BODY_JOINT_NAMES, "kp": (100000.0,) * 5, "kd": (1000.0,) * 5},
        {
            "names": arm,
            "kp": (40000.0, 40000.0, 15000.0, 15000.0, 15000.0, 8000.0, 8000.0),
            "kd": (600.0, 600.0, 220.0, 220.0, 220.0, 100.0, 100.0),
        },
        {
            "names": gripper,
            "kp": (1000.0,) * 6,
            "kd": (100.0,) * 6,
            "max_effort": tuple(effort_scale * value for value in (50.0, 10.0, 10.0, 10.0, 10.0, 10.0)),
        },
        {"names": STEERING_JOINT_NAMES, "kp": (1500.0,) * 4, "kd": (100.0,) * 4},
        {"names": WHEEL_JOINT_NAMES, "kp": (0.0,) * 4, "kd": (10.0,) * 4},
    )


def normalize_angle(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def positive_interaction_lead_m(
    *,
    joint_kind: str,
    joint_axis_world: Sequence[float],
    joint_pivot_world_m: Sequence[float],
    contact_world_m: Sequence[float],
    opening_delta: float,
    reference_joint_position: float,
    measured_joint_position: float,
) -> float:
    """Convert positive articulation-reference lead into contact-point metres."""

    direction = 1.0 if float(opening_delta) >= 0.0 else -1.0
    lead = max(0.0, direction * (float(reference_joint_position) - float(measured_joint_position)))
    if joint_kind == "prismatic":
        return lead
    if joint_kind != "revolute":
        raise ValueError(f"unsupported joint kind: {joint_kind}")
    axis = np.asarray(joint_axis_world, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        raise ValueError("joint axis must be nonzero")
    axis /= axis_norm
    radial = np.asarray(contact_world_m, dtype=np.float64).reshape(3) - np.asarray(
        joint_pivot_world_m, dtype=np.float64
    ).reshape(3)
    radial -= axis * float(np.dot(axis, radial))
    radius = float(np.linalg.norm(radial))
    return float(2.0 * radius * math.sin(min(math.pi, lead) / 2.0))


def yaw_from_quaternion_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def planar_tracking_twist(
    measured_xy_yaw: Sequence[float],
    target_xy_yaw: Sequence[float],
    *,
    position_gain: float = 1.5,
    yaw_gain: float = 1.5,
    maximum_linear_speed_m_s: float = 0.15,
    maximum_yaw_rate_rad_s: float = 0.2,
) -> tuple[np.ndarray, float, float]:
    measured = np.asarray(measured_xy_yaw, dtype=np.float64).reshape(3)
    target = np.asarray(target_xy_yaw, dtype=np.float64).reshape(3)
    error_world = target[:2] - measured[:2]
    cosine, sine = math.cos(measured[2]), math.sin(measured[2])
    error_body = np.asarray(
        (cosine * error_world[0] + sine * error_world[1], -sine * error_world[0] + cosine * error_world[1])
    )
    velocity = float(position_gain) * error_body
    speed = float(np.linalg.norm(velocity))
    if speed > maximum_linear_speed_m_s:
        velocity *= float(maximum_linear_speed_m_s) / speed
    yaw_error = normalize_angle(target[2] - measured[2])
    omega = float(np.clip(float(yaw_gain) * yaw_error, -maximum_yaw_rate_rad_s, maximum_yaw_rate_rad_s))
    return np.asarray((velocity[0], velocity[1], omega)), float(np.linalg.norm(error_world)), abs(yaw_error)


def swerve_inverse_kinematics(
    body_twist: Sequence[float],
    *,
    current_steering_angles_rad: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    vx, vy, omega = np.asarray(body_twist, dtype=np.float64).reshape(3)
    current = np.asarray(current_steering_angles_rad, dtype=np.float64).reshape(4)
    steering = np.empty(4, dtype=np.float64)
    wheel = np.empty(4, dtype=np.float64)
    for index, (x_position, y_position) in enumerate(MODULE_POSITIONS_BODY_M):
        module_vx = vx - omega * y_position
        module_vy = vy + omega * x_position
        speed = math.hypot(module_vx, module_vy)
        angle = math.atan2(module_vy, module_vx) if speed > 1.0e-9 else current[index]
        direct_error = normalize_angle(angle - current[index])
        flipped_angle = normalize_angle(angle + math.pi)
        flipped_error = normalize_angle(flipped_angle - current[index])
        if abs(flipped_error) < abs(direct_error):
            angle = flipped_angle
            speed = -speed
        steering[index] = normalize_angle(angle)
        wheel[index] = speed / WHEEL_RADIUS_M
    maximum = float(np.max(np.abs(wheel), initial=0.0))
    if maximum > MAXIMUM_WHEEL_SPEED_RAD_S:
        wheel *= MAXIMUM_WHEEL_SPEED_RAD_S / maximum
    return steering, wheel
