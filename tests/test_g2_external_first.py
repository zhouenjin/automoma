from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from automoma.integrations.realappliance.akr_adapter import (
    AkrAttachmentSpec,
    TransformRPY,
    build_g2_akr_urdf,
    make_g2_akr_config,
)
from automoma.integrations.realappliance.contracts import (
    ArticulationTaskSpec,
    JointKind,
    PhysicsRunPolicy,
    physical_open_failure_reasons,
)
from automoma.integrations.realappliance.contact_candidates import ContactCandidate
from automoma.integrations.realappliance.g2_adapter import (
    Bounds3D,
    Hand,
    build_planar_g2_urdf,
    fit_bounds_with_spheres,
    make_g2_curobo_config,
)
from automoma.integrations.realappliance.g2_runtime import (
    adaptive_interaction_lead_limit_m,
    gripper_targets,
    planar_tracking_twist,
    positive_interaction_lead_m,
    swerve_inverse_kinematics,
)
from automoma.integrations.realappliance.hypotheses import generate_interaction_hypotheses
from automoma.integrations.realappliance.physical_search import enumerate_physical_trials
from automoma.integrations.realappliance.transform_math import (
    axis_motion_transform,
    matrix_to_pose_wxyz,
    matrix_to_transform_rpy,
    pose_wxyz_to_matrix,
    quaternion_transform,
    rotation_matrix_to_quaternion_wxyz,
    transform_rpy_to_matrix,
)
from automoma.integrations.realappliance.usd_task import ExtractedUsdTask, resolve_annotated_parts
from automoma.integrations.realappliance.trajectory_selection import weighted_cspace_path_scores


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


def test_open_semantics_resolve_all_door_like_parts_without_asset_ids():
    annotations = {
        "start button": "part_01",
        "left glass door": "part_02",
        "drawer": "part_03",
        "right glass door": "part_04",
    }
    assert resolve_annotated_parts(annotations, "open") == (
        ("left glass door", "part_02"),
        ("drawer", "part_03"),
        ("right glass door", "part_04"),
    )


def test_rpy_round_trip_preserves_rigid_transform():
    source = quaternion_transform((0.2, -0.4, 0.7), (0.91, 0.1, -0.25, 0.3))
    encoded = matrix_to_transform_rpy(source)
    assert np.allclose(transform_rpy_to_matrix(encoded), source, atol=1e-8)


def test_curobo_pose_round_trip_preserves_rigid_transform():
    source = quaternion_transform((0.2, -0.4, 0.7), (0.91, 0.1, -0.25, 0.3))
    encoded = matrix_to_pose_wxyz(source)
    assert np.allclose(pose_wxyz_to_matrix(encoded), source, atol=1e-8)


def test_contact_candidate_placement_aligns_geometry_without_asset_rules():
    candidate = ContactCandidate(
        hypothesis_id="candidate",
        source_usd="Aligned.usd",
        target_joint_path="/World/part/joint",
        contact_center_source=(0.1, -0.2, 0.3),
        contact_points_source=((0.09, -0.2, 0.3), (0.11, -0.2, 0.3)),
        approach_source=(0.0, 1.0, 0.0),
        closing_source=(1.0, 0.0, 0.0),
        gripper_y_source=(0.0, 0.0, -1.0),
        handle_approach_depth=0.04,
        gripper_width=0.02,
        pre_ik_score=1.0,
        automatic_rank=0,
    )
    placed = candidate.place(
        handle_anchor_world=(0.8, 0.0, 1.2),
        desired_approach_world=(1.0, 0.0, 0.0),
        capture_depth_fraction=0.25,
    )
    source = np.asarray(placed.world_from_source)
    ee = np.asarray(placed.world_from_ee_contact)
    precontact = np.asarray(placed.world_from_ee_precontact)
    assert np.allclose(source @ np.asarray([0.1, -0.2, 0.3, 1.0]), (0.8, 0.0, 1.2, 1.0))
    assert np.allclose(ee[:3, 2], (1.0, 0.0, 0.0), atol=1e-8)
    assert np.allclose(ee[:3, 3], (0.79, 0.0, 1.2), atol=1e-8)
    assert np.allclose(precontact[:3, 3], (0.67, 0.0, 1.2), atol=1e-8)


def test_joint_motion_and_quaternion_helpers_agree():
    angle = 0.7
    motion = axis_motion_transform((0.0, 0.0, 1.0), angle, revolute=True)
    quaternion = rotation_matrix_to_quaternion_wxyz(motion)
    reconstructed = quaternion_transform((0.0, 0.0, 0.0), quaternion)
    assert np.allclose(reconstructed, motion, atol=1e-8)
    translation = axis_motion_transform((0.0, 2.0, 0.0), 0.4, revolute=False)
    assert np.allclose(translation[:3, 3], (0.0, 0.4, 0.0))


def test_extracted_task_builds_an_akr_chain_that_cancels_target_motion():
    world_from_joint = quaternion_transform((0.4, -0.2, 0.8), (0.9238795, 0.0, 0.0, 0.3826834))
    joint_from_handle = quaternion_transform((0.0, 0.35, 0.0), (1.0, 0.0, 0.0, 0.0))
    world_from_handle = world_from_joint @ joint_from_handle
    handle_from_ee = quaternion_transform((0.0, 0.0, -0.1), (1.0, 0.0, 0.0, 0.0))
    world_from_ee = world_from_handle @ handle_from_ee
    world_from_root = quaternion_transform((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    task = ArticulationTaskSpec(
        joint_name="/World/door/joint",
        target_link="/World/door",
        handle_link="/World/door",
        joint_kind=JointKind.REVOLUTE,
        axis=(0.0, 0.0, 1.0),
        pivot=(0.4, -0.2, 0.8),
        lower_limit=0.0,
        upper_limit=1.5,
        initial_position=0.0,
    )
    extracted = ExtractedUsdTask(
        task_name="open",
        semantic_label="door",
        annotated_part="part_00",
        joint_path=task.joint_name,
        body0_path="/World/body",
        body1_path=task.target_link,
        task=task,
        joint_axis_local=(0.0, 0.0, 1.0),
        world_from_joint_at_initial=tuple(tuple(v for v in row) for row in world_from_joint),
        world_from_target_link_at_initial=tuple(tuple(v for v in row) for row in world_from_handle),
        world_from_object_root=tuple(tuple(v for v in row) for row in world_from_root),
    )
    attachment = extracted.make_attachment_spec(Hand.LEFT, world_from_ee)
    ee_to_handle = transform_rpy_to_matrix(attachment.ee_to_handle)
    handle_to_joint = transform_rpy_to_matrix(attachment.handle_to_joint_at_initial)
    joint_to_root = transform_rpy_to_matrix(attachment.joint_at_initial_to_object_root)

    for delta in (0.0, 0.4, 1.2):
        joint_motion = quaternion_transform((0.0, 0.0, 0.0), (np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)))
        moved_handle = world_from_joint @ joint_motion @ joint_from_handle
        moved_ee = moved_handle @ handle_from_ee
        inverse_joint_motion = quaternion_transform(
            (0.0, 0.0, 0.0), (np.cos(delta / 2), 0.0, 0.0, -np.sin(delta / 2))
        )
        terminal = moved_ee @ ee_to_handle @ handle_to_joint @ inverse_joint_motion @ joint_to_root
        assert np.allclose(terminal, world_from_root, atol=1e-8)


def test_physics_policy_rejects_object_joint_writes():
    with pytest.raises(ValueError, match="write_target_joint"):
        PhysicsRunPolicy(write_target_joint=True).validate()
    PhysicsRunPolicy().validate()


def test_physical_failure_contract_reports_every_violated_clause():
    assert physical_open_failure_reasons(
        maximum_progress_fraction=0.2,
        acceptance_fraction=0.7,
        contact_during_opening=False,
        maximum_penetration_m=0.004,
        allowed_penetration_m=0.003,
        maximum_contact_force_n=260.0,
        allowed_contact_force_n=250.0,
    ) == (
        "insufficient_open_progress",
        "no_selected_surface_contact_during_opening",
        "excessive_penetration",
        "excessive_contact_force",
    )
    assert not physical_open_failure_reasons(
        maximum_progress_fraction=0.7,
        acceptance_fraction=0.7,
        contact_during_opening=True,
        maximum_penetration_m=0.003,
        allowed_penetration_m=0.003,
        maximum_contact_force_n=250.0,
        allowed_contact_force_n=250.0,
    )


def test_automatic_trajectory_selection_uses_global_cspace_metric():
    trajectories = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.5]],
            [[0.0, 0.0], [0.1, 0.1]],
        ]
    )
    result = weighted_cspace_path_scores(trajectories, [True, True, False], [2.0, 1.0])
    assert result["ranked_valid_trajectory_indices"] == [1, 0]
    assert result["selected_trajectory_index"] == 1
    assert result["scores_per_trajectory"] == pytest.approx([4.0, np.hypot(1.0, 0.5), np.hypot(0.2, 0.1)])


def test_physical_search_expands_automatic_candidate_and_path_order(tmp_path):
    report_path = tmp_path / "report.json"
    report = {
        "attempts": [
            {
                "attempt_id": "rank1",
                "automatic_rank": 1,
                "pre_ik_score": 9.0,
                "hand": "left",
                "capture_depth_fraction": 0.35,
                "success": True,
                "planning": {
                    "automatic_trajectory_selection": {
                        "ranked_valid_trajectory_indices": [2, 0],
                        "scores_per_trajectory": [2.0, 99.0, 1.0],
                    }
                },
            },
            {
                "attempt_id": "rank0",
                "automatic_rank": 0,
                "pre_ik_score": 5.0,
                "hand": "right",
                "capture_depth_fraction": 0.20,
                "success": True,
                "planning": {
                    "automatic_trajectory_selection": {
                        "ranked_valid_trajectory_indices": [1],
                        "scores_per_trajectory": [99.0, 3.0],
                    }
                },
            },
            {"attempt_id": "failed", "success": False},
        ]
    }
    trials = enumerate_physical_trials(report, report_path=report_path)
    assert [(trial.attempt_id, trial.trajectory_index) for trial in trials] == [
        ("rank0", 1),
        ("rank1", 2),
        ("rank1", 0),
    ]


def test_g2_runtime_maps_planar_twist_to_four_swerve_modules():
    steering, wheel = swerve_inverse_kinematics((0.14, 0.0, 0.0), current_steering_angles_rad=(0.0,) * 4)
    assert steering == pytest.approx((0.0,) * 4)
    assert wheel == pytest.approx((2.0,) * 4)
    twist, position_error, yaw_error = planar_tracking_twist((0.0, 0.0, 0.0), (1.0, 0.0, 0.5))
    assert twist == pytest.approx((0.15, 0.0, 0.2))
    assert position_error == pytest.approx(1.0)
    assert yaw_error == pytest.approx(0.5)
    assert gripper_targets(0.0) == pytest.approx((0.0,) * 6)
    assert gripper_targets(1.0) == pytest.approx((-0.85, -0.85, 0.85, 0.85, 0.85, 0.85))


def test_joint_reference_lead_is_measured_at_the_contact_point():
    revolute = positive_interaction_lead_m(
        joint_kind="revolute",
        joint_axis_world=(0.0, 0.0, 1.0),
        joint_pivot_world_m=(0.0, 0.0, 0.0),
        contact_world_m=(0.5, 0.0, 0.0),
        opening_delta=1.0,
        reference_joint_position=0.1,
        measured_joint_position=0.0,
    )
    assert revolute == pytest.approx(2.0 * 0.5 * np.sin(0.05))
    assert positive_interaction_lead_m(
        joint_kind="prismatic",
        joint_axis_world=(1.0, 0.0, 0.0),
        joint_pivot_world_m=(0.0, 0.0, 0.0),
        contact_world_m=(0.0, 0.0, 0.0),
        opening_delta=-1.0,
        reference_joint_position=-0.02,
        measured_joint_position=0.0,
    ) == pytest.approx(0.02)


def test_static_friction_probe_ramps_metric_lead_instead_of_jumping():
    arguments = {
        "physics_hz": 100,
        "ordinary_limit_m": 0.003,
        "probe_limit_m": 0.015,
        "probe_after_seconds": 0.5,
        "ramp_seconds": 1.0,
    }
    assert adaptive_interaction_lead_limit_m(no_progress_steps=50, **arguments) == pytest.approx((0.003, 0.0))
    assert adaptive_interaction_lead_limit_m(no_progress_steps=100, **arguments) == pytest.approx((0.009, 0.5))
    assert adaptive_interaction_lead_limit_m(no_progress_steps=150, **arguments) == pytest.approx((0.015, 1.0))
    assert adaptive_interaction_lead_limit_m(no_progress_steps=500, **arguments) == pytest.approx((0.015, 1.0))


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
