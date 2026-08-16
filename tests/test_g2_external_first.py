from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from automoma.integrations.realappliance.akr_adapter import (
    AkrAttachmentSpec,
    TransformRPY,
    build_g2_akr_urdf,
    make_g2_akr_config,
)
from automoma.integrations.realappliance.contracts import ArticulationTaskSpec, JointKind, PhysicsRunPolicy
from automoma.integrations.realappliance.g2_adapter import (
    Bounds3D,
    Hand,
    build_planar_g2_urdf,
    fit_bounds_with_spheres,
    make_g2_curobo_config,
)
from automoma.integrations.realappliance.hypotheses import generate_interaction_hypotheses


MINIMAL_G2 = """<?xml version="1.0"?>
<robot name="g2">
  <link name="base_link"/>
  <link name="body_link"/>
  <link name="left_gripper_center"/>
  <link name="right_gripper_center"/>
  <joint name="body_joint" type="revolute">
    <parent link="base_link"/><child link="body_link"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="left_ee" type="fixed"><parent link="body_link"/><child link="left_gripper_center"/></joint>
  <joint name="right_ee" type="fixed"><parent link="body_link"/><child link="right_gripper_center"/></joint>
</robot>
"""


def test_planar_g2_urdf_adds_three_base_dofs(tmp_path):
    source = tmp_path / "g2.urdf"
    output = tmp_path / "g2_planar.urdf"
    source.write_text(MINIMAL_G2, encoding="utf-8")

    build_planar_g2_urdf(source, output)
    root = ET.parse(output).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    assert {"base_x", "base_y", "base_yaw", "automoma_g2_mount"}.issubset(joints)
    assert joints["automoma_g2_mount"].find("child").attrib["link"] == "base_link"


def test_curobo_config_enumerates_base_and_selected_hand(tmp_path):
    source = {
        "robot_cfg": {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "right_gripper_center",
                "cspace": {
                    "joint_names": ["body_joint"],
                    "retract_config": [0.0],
                    "null_space_weight": [1.0],
                    "cspace_distance_weight": [1.0],
                    "max_acceleration": [2.0],
                    "max_jerk": [20.0],
                },
            }
        }
    }
    spheres = fit_bounds_with_spheres(Bounds3D((-0.4, -0.3, 0.0), (0.4, 0.3, 0.4)))
    mesh_root = tmp_path / "source_mesh_root"
    result = make_g2_curobo_config(
        source, tmp_path / "g2_planar.urdf", Hand.LEFT, asset_root_path=mesh_root, base_collision_spheres=spheres,
    )
    kin = result["robot_cfg"]["kinematics"]

    assert kin["base_link"] == "automoma_world"
    assert kin["ee_link"] == "left_gripper_center"
    assert kin["asset_root_path"] == str(mesh_root)
    assert kin["collision_spheres"]["base_link"] == spheres
    assert kin["cspace"]["joint_names"] == ["base_x", "base_y", "base_yaw", "body_joint"]
    assert len(kin["cspace"]["retract_config"]) == 4


def test_base_collision_lattice_covers_bounds():
    bounds = Bounds3D((-0.4, -0.3, 0.0), (0.4, 0.3, 0.4))
    spheres = fit_bounds_with_spheres(bounds, cells=(4, 3, 2), margin=0.0)
    assert len(spheres) == 24
    radius = spheres[0]["radius"]
    assert radius > 0.0
    for corner in (
        (x, y, z)
        for x in (bounds.minimum[0], bounds.maximum[0])
        for y in (bounds.minimum[1], bounds.maximum[1])
        for z in (bounds.minimum[2], bounds.maximum[2])
    ):
        assert (
            min(sum((corner[index] - sphere["center"][index]) ** 2 for index in range(3)) ** 0.5 for sphere in spheres)
            <= radius + 1e-12
        )


def test_akr_builder_inverts_target_joint_and_extends_cspace(tmp_path):
    source = tmp_path / "g2.urdf"
    planar = tmp_path / "g2_planar.urdf"
    akr = tmp_path / "g2_akr.urdf"
    source.write_text(MINIMAL_G2, encoding="utf-8")
    build_planar_g2_urdf(source, planar)
    spec = AkrAttachmentSpec(
        hand=Hand.RIGHT,
        source_joint_name="door_hinge",
        joint_kind=JointKind.REVOLUTE,
        joint_axis=(0.0, 0.0, 2.0),
        lower_limit=0.0,
        upper_limit=1.5,
        initial_position=0.2,
        ee_to_handle=TransformRPY(),
        handle_to_joint_at_initial=TransformRPY(xyz=(-0.4, 0.0, 0.0)),
        joint_at_initial_to_object_root=TransformRPY(),
    )
    build_g2_akr_urdf(planar, akr, spec)
    root = ET.parse(akr).getroot()
    target = root.find("./joint[@name='automoma_target_joint']")
    assert target is not None
    assert target.attrib["type"] == "revolute"
    assert target.find("axis").attrib["xyz"] == "0 0 1"
    assert float(target.find("limit").attrib["lower"]) == pytest.approx(-1.3)
    assert float(target.find("limit").attrib["upper"]) == pytest.approx(0.2)
    assert spec.akr_position(1.0) == pytest.approx(-0.8)

    base_config = {
        "robot_cfg": {
            "kinematics": {
                "ee_link": "right_gripper_center",
                "cspace": {
                    "joint_names": ["base_x", "body_joint"],
                    "retract_config": [0.0, 0.0],
                    "null_space_weight": [1.0, 1.0],
                    "cspace_distance_weight": [1.0, 1.0],
                    "max_acceleration": [1.0, 1.0],
                    "max_jerk": [10.0, 10.0],
                },
            }
        }
    }
    result = make_g2_akr_config(base_config, akr, spec)
    kinematics = result["robot_cfg"]["kinematics"]
    assert kinematics["ee_link"] == "automoma_object_root"
    assert kinematics["cspace"]["joint_names"][-1] == "automoma_target_joint"
    assert len(kinematics["cspace"]["joint_names"]) == 3


def test_hypothesis_pool_contains_both_hands_without_asset_rules():
    hypotheses = generate_interaction_hypotheses((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), [2, 7])
    assert {hypothesis.hand for hypothesis in hypotheses} == {Hand.LEFT, Hand.RIGHT}
    assert {hypothesis.grasp_index for hypothesis in hypotheses} == {2, 7}
    assert len(hypotheses) == 2 * 27 * 2


def test_physics_policy_rejects_object_joint_writes():
    with pytest.raises(ValueError, match="write_target_joint"):
        PhysicsRunPolicy(write_target_joint=True).validate()
    PhysicsRunPolicy().validate()


@pytest.mark.parametrize(
    ("kind", "planning", "acceptance"), [(JointKind.REVOLUTE, 0.80, 0.70), (JointKind.PRISMATIC, 0.90, 0.80)],
)
def test_open_thresholds_are_task_conditioned(kind, planning, acceptance):
    spec = ArticulationTaskSpec(
        joint_name="joint",
        target_link="door",
        handle_link="handle",
        joint_kind=kind,
        axis=(0.0, 0.0, 1.0),
        pivot=(0.0, 0.0, 0.0),
        lower_limit=0.0,
        upper_limit=1.0,
        initial_position=0.0,
    )
    assert spec.planning_fraction == planning
    assert spec.acceptance_fraction == acceptance
    assert spec.position_at_fraction(planning) == pytest.approx(planning)
