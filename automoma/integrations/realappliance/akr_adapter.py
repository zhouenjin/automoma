"""Construct the missing G2/object AKR chain used by AutoMoMa planning.

The public AutoMoMa repository consumes grasp-specific AKR robot files but does
not publish the generator. This module recreates only that missing mechanical
interface. It does not select an asset, grasp, hand, or motion primitive.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .contracts import JointKind
from .g2_adapter import Hand


@dataclass(frozen=True)
class TransformRPY:
    """A fixed transform represented with URDF xyz and fixed-axis RPY."""

    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AkrAttachmentSpec:
    """Geometry needed to invert one articulated target joint from a grasp."""

    hand: Hand
    source_joint_name: str
    joint_kind: JointKind
    joint_axis: Tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    initial_position: float
    ee_to_handle: TransformRPY
    handle_to_joint_at_initial: TransformRPY
    joint_at_initial_to_object_root: TransformRPY
    effort: float = 1000.0
    velocity: float = 1.0

    @property
    def robot_ee_link(self) -> str:
        return "left_gripper_center" if self.hand is Hand.LEFT else "right_gripper_center"

    @property
    def akr_joint_name(self) -> str:
        return "automoma_target_joint"

    @property
    def akr_goal_link(self) -> str:
        return "automoma_object_root"

    @property
    def akr_limits(self) -> Tuple[float, float]:
        """Limits for ``q_akr = -(q_object - q_initial)``."""

        return self.initial_position - self.upper_limit, self.initial_position - self.lower_limit

    def akr_position(self, object_position: float) -> float:
        return self.initial_position - object_position

    def validate(self) -> None:
        if self.lower_limit >= self.upper_limit:
            raise ValueError("target joint lower limit must be smaller than upper limit")
        if not self.lower_limit <= self.initial_position <= self.upper_limit:
            raise ValueError("target joint initial position is outside its limits")
        norm = math.sqrt(sum(component * component for component in self.joint_axis))
        if norm < 1e-9:
            raise ValueError("target joint axis has zero length")


_AKR_LINKS = (
    "automoma_handle_frame",
    "automoma_joint_frame",
    "automoma_joint_moving",
    "automoma_object_root",
)


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _add_origin(joint: ET.Element, transform: TransformRPY) -> None:
    ET.SubElement(
        joint, "origin", {"xyz": _format_vector(transform.xyz), "rpy": _format_vector(transform.rpy)},
    )


def _fixed_joint(name: str, parent: str, child: str, transform: TransformRPY,) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    _add_origin(joint, transform)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return joint


def build_g2_akr_urdf(planar_g2_urdf: Path, output_urdf: Path, spec: AkrAttachmentSpec,) -> Path:
    """Append a grasp-conditioned inverse object chain to the planar G2 URDF."""

    spec.validate()
    tree = ET.parse(planar_g2_urdf)
    robot = tree.getroot()
    link_names = {element.attrib["name"] for element in robot.findall("link")}
    joint_names = {element.attrib["name"] for element in robot.findall("joint")}
    if spec.robot_ee_link not in link_names:
        raise ValueError(f"robot end-effector link {spec.robot_ee_link!r} is absent")
    collisions = (link_names | joint_names) & set(_AKR_LINKS + (spec.akr_joint_name,))
    if collisions:
        raise ValueError(f"AKR names already exist: {sorted(collisions)}")

    for link_name in _AKR_LINKS:
        robot.append(ET.Element("link", {"name": link_name}))
    robot.append(_fixed_joint("automoma_ee_to_handle", spec.robot_ee_link, "automoma_handle_frame", spec.ee_to_handle,))
    robot.append(
        _fixed_joint(
            "automoma_handle_to_joint",
            "automoma_handle_frame",
            "automoma_joint_frame",
            spec.handle_to_joint_at_initial,
        )
    )

    target_joint = ET.Element("joint", {"name": spec.akr_joint_name, "type": spec.joint_kind.value},)
    _add_origin(target_joint, TransformRPY())
    ET.SubElement(target_joint, "parent", {"link": "automoma_joint_frame"})
    ET.SubElement(target_joint, "child", {"link": "automoma_joint_moving"})
    norm = math.sqrt(sum(component * component for component in spec.joint_axis))
    normalized_axis = tuple(component / norm for component in spec.joint_axis)
    ET.SubElement(target_joint, "axis", {"xyz": _format_vector(normalized_axis)})
    lower, upper = spec.akr_limits
    ET.SubElement(
        target_joint,
        "limit",
        {
            "effort": f"{spec.effort:.12g}",
            "lower": f"{lower:.12g}",
            "upper": f"{upper:.12g}",
            "velocity": f"{spec.velocity:.12g}",
        },
    )
    robot.append(target_joint)
    robot.append(
        _fixed_joint(
            "automoma_joint_to_object_root",
            "automoma_joint_moving",
            spec.akr_goal_link,
            spec.joint_at_initial_to_object_root,
        )
    )

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


def make_g2_akr_config(
    base_robot_config: Mapping[str, Any],
    akr_urdf: Path,
    spec: AkrAttachmentSpec,
    *,
    object_distance_weight: float = 1.0,
    object_null_space_weight: float = 1.0,
    object_max_acceleration: float = 1.0,
    object_max_jerk: float = 10.0,
) -> Dict[str, Any]:
    """Extend a generated G2 planner config with the inverse object joint."""

    spec.validate()
    config = copy.deepcopy(dict(base_robot_config))
    robot_cfg = config.get("robot_cfg")
    if not isinstance(robot_cfg, dict):
        raise ValueError("cuRobo config must contain a robot_cfg mapping")
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError("robot_cfg.kinematics must be a mapping")
    cspace = kinematics.get("cspace")
    if not isinstance(cspace, dict):
        raise ValueError("robot_cfg.kinematics.cspace must be a mapping")
    if spec.akr_joint_name in cspace.get("joint_names", []):
        raise ValueError("AKR target joint already exists in cspace")

    kinematics["urdf_path"] = str(akr_urdf)
    kinematics["ee_link"] = spec.akr_goal_link
    additions = {
        "joint_names": spec.akr_joint_name,
        "retract_config": 0.0,
        "null_space_weight": object_null_space_weight,
        "cspace_distance_weight": object_distance_weight,
        "max_acceleration": object_max_acceleration,
        "max_jerk": object_max_jerk,
    }
    for key, value in additions.items():
        values = cspace.get(key)
        if not isinstance(values, list):
            raise ValueError(f"cspace.{key} must be a list")
        values.append(value)

    expected = len(cspace["joint_names"])
    for key in additions:
        if len(cspace[key]) != expected:
            raise ValueError(f"cspace.{key} has {len(cspace[key])} values; expected {expected}")
    return config
