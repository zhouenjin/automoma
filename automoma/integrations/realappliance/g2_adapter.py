"""Generate AutoMoMa-compatible G2 planner assets without asset-specific rules."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PlanarBaseLimits:
    x: tuple[float, float] = (-2.0, 2.0)
    y: tuple[float, float] = (-2.0, 2.0)
    yaw: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    linear_velocity: float = 0.6
    angular_velocity: float = 0.8


_WORLD_LINK = "automoma_world"
_BASE_JOINTS = ("base_x", "base_y", "base_yaw")


def _joint_xml(
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    axis: Sequence[float],
    limits: Sequence[float],
    velocity: float,
) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": " ".join(str(v) for v in axis)})
    ET.SubElement(
        joint,
        "limit",
        {
            "effort": "1000",
            "lower": str(limits[0]),
            "upper": str(limits[1]),
            "velocity": str(velocity),
        },
    )
    return joint


def build_planar_g2_urdf(
    source_urdf: Path,
    output_urdf: Path,
    *,
    robot_root_link: str = "base_link",
    limits: PlanarBaseLimits = PlanarBaseLimits(),
) -> Path:
    """Prepend virtual planar joints to the supplied G2 URDF.

    The virtual joints are planner coordinates only.  The Isaac Sim executor
    remains responsible for converting them to the physical mobile-base action.
    """

    tree = ET.parse(source_urdf)
    robot = tree.getroot()
    link_names = {element.attrib["name"] for element in robot.findall("link")}
    joint_names = {element.attrib["name"] for element in robot.findall("joint")}

    if robot_root_link not in link_names:
        raise ValueError(f"G2 root link {robot_root_link!r} is absent from {source_urdf}")
    collisions = (link_names | joint_names) & {
        _WORLD_LINK,
        "automoma_base_x_link",
        "automoma_base_y_link",
        "automoma_base_yaw_link",
        *_BASE_JOINTS,
        "automoma_g2_mount",
    }
    if collisions:
        raise ValueError(f"virtual-base names already exist: {sorted(collisions)}")

    virtual_elements: List[ET.Element] = [
        ET.Element("link", {"name": _WORLD_LINK}),
        ET.Element("link", {"name": "automoma_base_x_link"}),
        ET.Element("link", {"name": "automoma_base_y_link"}),
        ET.Element("link", {"name": "automoma_base_yaw_link"}),
        _joint_xml(
            "base_x", "prismatic", _WORLD_LINK, "automoma_base_x_link", (1, 0, 0), limits.x, limits.linear_velocity,
        ),
        _joint_xml(
            "base_y",
            "prismatic",
            "automoma_base_x_link",
            "automoma_base_y_link",
            (0, 1, 0),
            limits.y,
            limits.linear_velocity,
        ),
        _joint_xml(
            "base_yaw",
            "revolute",
            "automoma_base_y_link",
            "automoma_base_yaw_link",
            (0, 0, 1),
            limits.yaw,
            limits.angular_velocity,
        ),
    ]
    mount = ET.Element("joint", {"name": "automoma_g2_mount", "type": "fixed"})
    ET.SubElement(mount, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": "automoma_base_yaw_link"})
    ET.SubElement(mount, "child", {"link": robot_root_link})
    virtual_elements.append(mount)

    for offset, element in enumerate(virtual_elements):
        robot.insert(offset, element)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


def _prepend(values: Any, prefix: Iterable[float]) -> Any:
    if isinstance(values, list):
        return list(prefix) + values
    return values


def make_g2_curobo_config(
    source_config: Mapping[str, Any],
    generated_urdf: Path,
    hand: Hand,
    *,
    base_null_space_weight: Sequence[float] = (1.0, 1.0, 1.0),
    base_distance_weight: Sequence[float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Return a G2 config whose cspace jointly optimizes planar base and body/arm."""

    config = copy.deepcopy(dict(source_config))
    robot_cfg = config.get("robot_cfg")
    if not isinstance(robot_cfg, dict):
        raise ValueError("cuRobo config must contain a robot_cfg mapping")
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError("robot_cfg.kinematics must be a mapping")
    cspace = kinematics.get("cspace")
    if not isinstance(cspace, dict):
        raise ValueError("robot_cfg.kinematics.cspace must be a mapping")

    joint_names = list(cspace.get("joint_names", []))
    if any(name in joint_names for name in _BASE_JOINTS):
        raise ValueError("source cspace already contains AutoMoMa planar-base joints")

    kinematics["urdf_path"] = str(generated_urdf)
    kinematics["asset_root_path"] = str(generated_urdf.parent)
    kinematics["base_link"] = _WORLD_LINK
    kinematics["ee_link"] = "left_gripper_center" if hand is Hand.LEFT else "right_gripper_center"

    cspace["joint_names"] = list(_BASE_JOINTS) + joint_names
    cspace["retract_config"] = _prepend(cspace.get("retract_config", []), (0.0, 0.0, 0.0))
    cspace["null_space_weight"] = _prepend(cspace.get("null_space_weight", []), base_null_space_weight)
    cspace["cspace_distance_weight"] = _prepend(cspace.get("cspace_distance_weight", []), base_distance_weight)
    cspace["max_acceleration"] = _prepend(cspace.get("max_acceleration"), (1.0, 1.0, 1.0))
    cspace["max_jerk"] = _prepend(cspace.get("max_jerk"), (10.0, 10.0, 10.0))

    expected = len(cspace["joint_names"])
    for key in ("retract_config", "null_space_weight", "cspace_distance_weight"):
        values = cspace.get(key)
        if not isinstance(values, list) or len(values) != expected:
            actual = len(values) if isinstance(values, list) else "non-list"
            raise ValueError(f"cspace.{key} has {actual} values; expected {expected}")
    return config
