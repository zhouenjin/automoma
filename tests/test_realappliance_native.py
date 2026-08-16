import numpy as np

from automoma.integrations.realappliance_native import (
    ArticulationSpec,
    JointConditionedContactPath,
    OpenEpisodeContract,
    OpenEpisodeSample,
    RealApplianceUsdManifest,
    UsdJointDescriptor,
    UsdMeshGeometry,
    build_akr_robot_config,
    build_open_joint_components,
    choose_open_joint_candidates,
)
from automoma.integrations.realappliance_native.contracts import (
    require_robot_only_action,
)


def test_revolute_contact_follows_the_door_arc() -> None:
    articulation = ArticulationSpec(
        joint_type="revolute",
        axis_world=np.array([1.0, 0.0, 0.0]),
        pivot_world_m=np.zeros(3),
        lower_limit=0.0,
        upper_limit=np.pi / 2,
        initial_position=0.0,
        goal_position=np.pi / 2,
    )
    path = JointConditionedContactPath(
        articulation=articulation,
        contact_position_world_m=np.array([0.0, 0.0, 1.0]),
        contact_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    position, _ = path.pose(np.pi / 2)

    assert np.allclose(position, [0.0, -1.0, 0.0], atol=1.0e-7)


def test_strict_contract_rejects_object_commands_even_when_open() -> None:
    contract = OpenEpisodeContract(0.0, 1.0, 1, 0.0)
    audit = contract.evaluate(
        [
            OpenEpisodeSample(0.0, robot_target_contact=True),
            OpenEpisodeSample(
                0.8, robot_target_contact=True, object_target_was_written=True
            ),
        ]
    )

    assert not audit.success
    assert "target_joint_commanded" in audit.failure_reasons


def test_strict_contract_accepts_contact_driven_opening() -> None:
    contract = OpenEpisodeContract(0.0, 1.0, 1, 0.0)
    audit = contract.evaluate(
        [
            OpenEpisodeSample(0.0),
            OpenEpisodeSample(0.2, robot_target_contact=True),
            OpenEpisodeSample(0.71, robot_target_contact=True),
        ]
    )

    assert audit.success
    assert audit.maximum_range_fraction == 0.71


def test_robot_only_action_width_is_enforced() -> None:
    require_robot_only_action(action_width=12, robot_dof=12)

    try:
        require_robot_only_action(action_width=13, robot_dof=12)
    except ValueError as exc:
        assert "robot-only" in str(exc)
    else:
        raise AssertionError("object-appended action must be rejected")


def test_usd_inventory_selects_all_openable_joints_without_asset_id_rules() -> None:
    fixed = UsdJointDescriptor(
        path="/World/body/fixed",
        joint_type="fixed",
        parent_body=None,
        child_body="/World/body",
        axis=None,
        lower_limit=None,
        upper_limit=None,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )
    door = UsdJointDescriptor(
        path="/World/door/hinge",
        joint_type="revolute",
        parent_body="/World/body",
        child_body="/World/door",
        axis="X",
        lower_limit=0.0,
        upper_limit=90.0,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )
    button = UsdJointDescriptor(
        path="/World/button/slide",
        joint_type="prismatic",
        parent_body="/World/body",
        child_body="/World/button",
        axis="Z",
        lower_limit=0.0,
        upper_limit=0.0,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )

    candidates = choose_open_joint_candidates((fixed, button, door))

    assert candidates == (door,)
    manifest = RealApplianceUsdManifest(
        asset_id="unseen",
        source_usd="/tmp/Aligned.usd",
        joints=(fixed, button, door),
        mesh_paths=("/World/door",),
        rigid_body_paths=("/World/body", "/World/door"),
    )
    assert manifest.to_dict()["provenance"]["g2_inputs_used"] is False


def test_open_component_includes_fixed_handle_and_ranks_large_door_first() -> None:
    door = UsdJointDescriptor(
        path="/World/door/hinge",
        joint_type="revolute",
        parent_body="/World/body",
        child_body="/World/door",
        axis="Y",
        lower_limit=0.0,
        upper_limit=90.0,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )
    handle = UsdJointDescriptor(
        path="/World/handle/fixed",
        joint_type="fixed",
        parent_body="/World/door",
        child_body="/World/handle",
        axis=None,
        lower_limit=None,
        upper_limit=None,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )
    knob = UsdJointDescriptor(
        path="/World/knob/hinge",
        joint_type="revolute",
        parent_body="/World/body",
        child_body="/World/knob",
        axis="Y",
        lower_limit=-90.0,
        upper_limit=90.0,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )
    geometries = (
        UsdMeshGeometry("/World/door/mesh", "/World/door", (0, 0, 0), (1, 1, 0.1), 100),
        UsdMeshGeometry("/World/handle/mesh", "/World/handle", (0, 0, 0), (0.5, 0.1, 0.1), 50),
        UsdMeshGeometry("/World/knob/mesh", "/World/knob", (0, 0, 0), (0.05, 0.05, 0.05), 20),
    )
    manifest = RealApplianceUsdManifest(
        asset_id="unseen",
        source_usd="/tmp/Aligned.usd",
        joints=(door, handle, knob),
        mesh_paths=tuple(value.path for value in geometries),
        rigid_body_paths=("/World/body", "/World/door", "/World/handle", "/World/knob"),
        mesh_geometries=geometries,
    )

    components = build_open_joint_components(manifest)

    assert components[0].joint.path == door.path
    assert components[0].rigid_bodies == ("/World/door", "/World/handle")
    assert "/World/handle/mesh" in components[0].mesh_paths


def test_akr_inverts_target_joint_and_keeps_g2_out_of_provenance() -> None:
    joint = UsdJointDescriptor(
        path="/World/door/hinge",
        joint_type="revolute",
        parent_body="/World/body",
        child_body="/World/door",
        axis="Y",
        lower_limit=0.0,
        upper_limit=90.0,
        local_position_parent=(0.0, 0.0, 0.0),
        local_position_child=(0.0, 0.0, 0.0),
    )
    geometry = UsdMeshGeometry(
        "/World/door/mesh",
        "/World/door",
        (0.0, 0.0, 0.0),
        (0.4, 0.05, 0.3),
        100,
        (0.0, 0.0, 0.0),
        (0.4, 0.05, 0.3),
    )
    manifest = RealApplianceUsdManifest(
        asset_id="unseen",
        source_usd="/tmp/Aligned.usd",
        joints=(joint,),
        mesh_paths=(geometry.path,),
        rigid_body_paths=("/World/body", "/World/door"),
        mesh_geometries=(geometry,),
    )
    component = build_open_joint_components(manifest)[0]
    base = {
        "robot_cfg": {
            "kinematics": {
                "ee_link": "panda_hand",
                "extra_links": {
                    "attached_object": {
                        "parent_link_name": "panda_hand",
                        "link_name": "attached_object",
                    }
                },
                "collision_link_names": ["panda_hand", "attached_object"],
                "collision_spheres": {},
                "self_collision_buffer": {"panda_hand": 0.0},
                "self_collision_ignore": {
                    "panda_link7": ["attached_object"],
                    "panda_hand": [],
                },
                "cspace": {
                    "joint_names": ["panda_joint1"],
                    "retract_config": [0.0],
                    "null_space_weight": [1.0],
                    "cspace_distance_weight": [1.0],
                },
            }
        }
    }

    akr = build_akr_robot_config(
        base,
        manifest,
        component,
        component_to_ee_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )

    kinematics = akr["robot_cfg"]["kinematics"]
    target = kinematics["extra_links"]["realappliance_target_joint_link"]
    assert target["joint_type"] == "Y_ROT"
    assert np.allclose(target["joint_limits"], [-np.pi / 2.0, 0.0])
    assert kinematics["ee_link"] == "realappliance_object_anchor"
    assert (
        kinematics["extra_links"]["realappliance_object_anchor"]["joint_type"]
        == "FIXED"
    )
    assert kinematics["cspace"]["joint_names"][-1] == "realappliance_target_joint"
    assert akr["realappliance_native"]["g2_inputs_used"] is False
    assert "attached_object" not in kinematics["extra_links"]
    assert "realappliance_grasped_component" in kinematics["self_collision_ignore"][
        "panda_link7"
    ]
    assert akr["realappliance_native"]["payload_self_collision_ignore_links"] == [
        "panda_link7"
    ]
