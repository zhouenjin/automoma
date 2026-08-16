"""Construct an AutoMoMa articulated kinematic representation from USD data."""

from __future__ import annotations

from copy import deepcopy
from math import ceil, pi, sqrt
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .usd_articulation import (
    OpenJointComponent,
    RealApplianceUsdManifest,
    UsdJointDescriptor,
)


def _pose_matrix(
    position: Sequence[float], quaternion_wxyz: Sequence[float]
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_quat(np.roll(np.asarray(quaternion_wxyz), -1)).as_matrix()
    value[:3, 3] = np.asarray(position, dtype=np.float64)
    return value


def _matrix_pose(matrix: np.ndarray) -> list[float]:
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [*matrix[:3, 3].tolist(), *np.roll(quaternion_xyzw, 1).tolist()]


def parent_to_child_zero(joint: UsdJointDescriptor, scale: float = 1.0) -> np.ndarray:
    """Return the child pose in the parent frame at zero joint displacement."""

    parent_to_joint = _pose_matrix(
        np.asarray(joint.local_position_parent) * scale,
        joint.local_rotation_parent_wxyz,
    )
    child_to_joint = _pose_matrix(
        np.asarray(joint.local_position_child) * scale,
        joint.local_rotation_child_wxyz,
    )
    return parent_to_joint @ np.linalg.inv(child_to_joint)


def fixed_body_transforms(
    root_body: str,
    joints: Sequence[UsdJointDescriptor],
    *,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Map every fixed descendant body into ``root_body`` coordinates."""

    result = {root_body: np.eye(4, dtype=np.float64)}
    pending = list(joints)
    changed = True
    while changed:
        changed = False
        for joint in pending:
            if (
                joint.joint_type != "fixed"
                or not joint.parent_body
                or not joint.child_body
                or joint.parent_body not in result
                or joint.child_body in result
            ):
                continue
            result[joint.child_body] = (
                result[joint.parent_body] @ parent_to_child_zero(joint, scale)
            )
            changed = True
    return result


def fit_aabb_spheres(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    maximum_radius_m: float = 0.03,
    minimum_radius_m: float = 0.006,
    maximum_spheres: int = 96,
) -> tuple[tuple[float, float, float, float], ...]:
    """Cover an axis-aligned link-local box with overlapping spheres."""

    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    extent = np.maximum(upper_array - lower_array, 0.0)
    positive = np.sort(extent[extent > 1.0e-8])
    natural_radius = 0.5 * positive[0] if positive.size else minimum_radius_m
    radius = min(maximum_radius_m, max(minimum_radius_m, natural_radius))

    def counts_for(value: float) -> np.ndarray:
        spacing = max(2.0 * value / sqrt(3.0), 1.0e-6)
        return np.maximum(1, np.ceil(extent / spacing).astype(int))

    counts = counts_for(radius)
    while int(np.prod(counts)) > maximum_spheres:
        radius *= 1.15
        counts = counts_for(radius)

    axes = []
    for axis in range(3):
        if counts[axis] == 1:
            axes.append(np.asarray([(lower_array[axis] + upper_array[axis]) * 0.5]))
        else:
            axes.append(np.linspace(lower_array[axis], upper_array[axis], counts[axis]))
    return tuple(
        (float(x), float(y), float(z), float(radius))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def component_collision_spheres(
    manifest: RealApplianceUsdManifest,
    component: OpenJointComponent,
) -> tuple[tuple[float, float, float, float], ...]:
    """Create collision proxies in the movable child-link frame."""

    if component.joint.child_body is None:
        raise ValueError("open component has no child body")
    transforms = fixed_body_transforms(
        component.joint.child_body,
        manifest.joints,
        scale=manifest.meters_per_unit,
    )
    spheres = []
    body_set = set(component.rigid_bodies)
    for geometry in manifest.mesh_geometries:
        if (
            geometry.rigid_body not in body_set
            or geometry.rigid_body_bounds_min is None
            or geometry.rigid_body_bounds_max is None
            or geometry.rigid_body not in transforms
        ):
            continue
        local_spheres = fit_aabb_spheres(
            np.asarray(geometry.rigid_body_bounds_min) * manifest.meters_per_unit,
            np.asarray(geometry.rigid_body_bounds_max) * manifest.meters_per_unit,
        )
        body_to_component = transforms[geometry.rigid_body]
        for x, y, z, radius in local_spheres:
            center = body_to_component @ np.asarray([x, y, z, 1.0])
            spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
    return tuple(spheres)


def _joint_type(joint: UsdJointDescriptor) -> str:
    axis = (joint.axis or "").upper()
    suffix = "ROT" if joint.joint_type == "revolute" else "PRISM"
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"unsupported USD joint axis: {joint.axis}")
    return f"{axis}_{suffix}"


def _source_to_akr_position(joint: UsdJointDescriptor, value: float) -> float:
    converted = value * pi / 180.0 if joint.joint_type == "revolute" else value
    return -converted


def build_akr_robot_config(
    base_robot_config: Mapping[str, object],
    manifest: RealApplianceUsdManifest,
    component: OpenJointComponent,
    *,
    component_to_ee_pose: Sequence[float],
    source_initial_position: float = 0.0,
) -> dict[str, object]:
    """Attach a selected component to the EE and invert its source joint.

    ``component_to_ee_pose`` is ``T_component_ee`` in xyz+wxyz form and must come
    from a freshly generated grasp hypothesis, never from a G2 trajectory.
    """

    if len(component_to_ee_pose) != 7:
        raise ValueError("component_to_ee_pose must contain xyz+wxyz")
    joint = component.joint
    if joint.parent_body is None or joint.child_body is None:
        raise ValueError("target joint must connect parent and child rigid bodies")

    result = deepcopy(dict(base_robot_config))
    robot_cfg = result["robot_cfg"] if "robot_cfg" in result else result
    kinematics = robot_cfg["kinematics"]
    ee_link = kinematics["ee_link"]

    # Remove the stock demonstration payload.  This branch creates a new
    # component from the selected RealAppliance joint and fresh grasp pose.
    extras = dict(kinematics.get("extra_links", {}))
    extras.pop("attached_object", None)
    component_link = "realappliance_grasped_component"
    anchor_link = "realappliance_object_anchor"
    target_joint_name = "realappliance_target_joint"

    component_to_ee = _pose_matrix(component_to_ee_pose[:3], component_to_ee_pose[3:])
    ee_to_component = np.linalg.inv(component_to_ee)
    child_to_parent_zero = np.linalg.inv(
        parent_to_child_zero(joint, manifest.meters_per_unit)
    )
    lower = joint.lower_limit if joint.lower_limit is not None else -1.0e3
    upper = joint.upper_limit if joint.upper_limit is not None else 1.0e3
    akr_limits = sorted(
        (_source_to_akr_position(joint, lower), _source_to_akr_position(joint, upper))
    )

    extras[component_link] = {
        "parent_link_name": ee_link,
        "link_name": component_link,
        "fixed_transform": _matrix_pose(ee_to_component),
        "joint_type": "FIXED",
        "joint_name": "realappliance_grasp_attachment",
    }
    extras[anchor_link] = {
        "parent_link_name": component_link,
        "link_name": anchor_link,
        "fixed_transform": _matrix_pose(child_to_parent_zero),
        "joint_type": _joint_type(joint),
        "joint_name": target_joint_name,
        "joint_limits": akr_limits,
    }
    kinematics["extra_links"] = extras
    # The reversed chain terminates at the stationary appliance body.  Making
    # that anchor the planning EE is what couples robot motion to object motion:
    # trajectories must keep the appliance body pose fixed while q_object moves.
    kinematics["ee_link"] = anchor_link

    collision_spheres = kinematics.get("collision_spheres")
    spheres = component_collision_spheres(manifest, component)
    counts = dict(kinematics.get("extra_collision_spheres", {}))
    counts.pop("attached_object", None)
    if isinstance(collision_spheres, dict):
        collision_spheres[component_link] = [
            {"center": [x, y, z], "radius": radius} for x, y, z, radius in spheres
        ]
        kinematics["extra_collision_spheres"] = counts
    else:
        # cuRobo allocates placeholders which the runtime adapter replaces after
        # resolving the stock Franka sphere file.
        counts[component_link] = len(spheres)
        kinematics["extra_collision_spheres"] = counts

    collision_links = [
        value for value in kinematics.get("collision_link_names", [])
        if value != "attached_object"
    ]
    if component_link not in collision_links:
        collision_links.append(component_link)
    kinematics["collision_link_names"] = collision_links
    buffers = dict(kinematics.get("self_collision_buffer", {}))
    buffers.pop("attached_object", None)
    buffers[component_link] = 0.0
    kinematics["self_collision_buffer"] = buffers
    ignores = deepcopy(kinematics.get("self_collision_ignore", {}))
    for values in ignores.values():
        if "attached_object" in values:
            values.remove("attached_object")
    for gripper_link in (ee_link, "panda_leftfinger", "panda_rightfinger"):
        ignores.setdefault(gripper_link, [])
        if component_link not in ignores[gripper_link]:
            ignores[gripper_link].append(component_link)
    kinematics["self_collision_ignore"] = ignores

    cspace = kinematics["cspace"]
    if target_joint_name not in cspace["joint_names"]:
        cspace["joint_names"].append(target_joint_name)
        cspace["retract_config"].append(
            _source_to_akr_position(joint, source_initial_position)
        )
        cspace["null_space_weight"].append(1.0)
        cspace["cspace_distance_weight"].append(1.0)
    result.setdefault("realappliance_native", {})
    result["realappliance_native"].update(
        {
            "pipeline": "automoma_native",
            "g2_inputs_used": False,
            "asset_id": manifest.asset_id,
            "source_joint_path": joint.path,
            "component_bodies": list(component.rigid_bodies),
            "component_to_ee_pose": list(component_to_ee_pose),
            "akr_target_joint": target_joint_name,
            "source_to_akr_position_sign": -1.0,
            "collision_spheres": {
                component_link: [
                    {"center": [x, y, z], "radius": radius}
                    for x, y, z, radius in spheres
                ]
            },
        }
    )
    return result
