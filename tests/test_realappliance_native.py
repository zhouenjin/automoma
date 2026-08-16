import numpy as np

from automoma.integrations.realappliance_native import (
    ArticulationSpec,
    JointConditionedContactPath,
    OpenEpisodeContract,
    OpenEpisodeSample,
)
from automoma.integrations.realappliance_native.contracts import require_robot_only_action


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
            OpenEpisodeSample(0.8, robot_target_contact=True, object_target_was_written=True),
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
