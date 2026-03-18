#!/usr/bin/env python

from __future__ import annotations

from unittest.mock import patch

import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.rby1 import RBY1, RBY1Config
from lerobot.robots.rby1.schema import (
    DEFAULT_READY_LEFT_ARM_POSITIONS,
    DEFAULT_READY_RIGHT_ARM_POSITIONS,
    FULL_ACTION_KEYS,
)
from tests.fixtures.fake_rby1_sdk import FakeControlManagerState, FakeRBY1SDK


@pytest.fixture
def fake_rby1_sdk():
    sdk = FakeRBY1SDK()
    with patch("lerobot.robots.rby1.rby1.rby", sdk):
        yield sdk


@pytest.fixture
def rby1_robot(fake_rby1_sdk):
    robot = RBY1(RBY1Config(gripper_home_on_connect=False))
    yield robot
    if robot.is_connected:
        robot.disconnect()


def test_make_robot_from_config_returns_rby1(fake_rby1_sdk):
    robot = make_robot_from_config(RBY1Config())
    assert isinstance(robot, RBY1)


def test_connect_disconnect(rby1_robot, fake_rby1_sdk):
    assert not rby1_robot.is_connected

    rby1_robot.connect()
    assert rby1_robot.is_connected

    sdk_robot = fake_rby1_sdk.created_robots[-1]
    sdk_gripper = fake_rby1_sdk.created_buses[-1]
    assert sdk_robot.power_on_calls == [rby1_robot.config.power_device_pattern]
    assert sdk_robot.servo_on_calls == [rby1_robot.config.servo_device_pattern]
    assert sdk_robot.create_command_stream_calls == [rby1_robot.config.command_priority]
    assert sdk_gripper.port_open is True

    rby1_robot.disconnect()
    assert not rby1_robot.is_connected
    assert sdk_robot.stream.cancelled is True
    assert sdk_robot.disable_control_manager_calls == 1
    assert sdk_robot.servo_off_calls == []


def test_disconnect_can_servo_off_when_enabled(fake_rby1_sdk):
    robot = RBY1(RBY1Config(disable_torque_on_disconnect=True))
    robot.connect()

    sdk_robot = fake_rby1_sdk.created_robots[-1]
    robot.disconnect()

    assert sdk_robot.servo_off_calls == [robot.config.servo_device_pattern]


def test_connect_resets_faulted_control_manager(rby1_robot, fake_rby1_sdk):
    sdk_robot = fake_rby1_sdk.create_robot(rby1_robot.config.address, rby1_robot.config.model)
    sdk_robot.control_manager_state.state = FakeControlManagerState.State.MajorFault

    with patch("lerobot.robots.rby1.rby1.rby.create_robot", return_value=sdk_robot):
        rby1_robot.connect()

    assert sdk_robot.reset_fault_calls == 1


def test_get_observation_returns_expected_joint_keys(rby1_robot):
    rby1_robot.connect()
    obs = rby1_robot.get_observation()

    assert tuple(obs) == FULL_ACTION_KEYS
    for key in rby1_robot.action_features:
        assert isinstance(obs[key], float)


def test_send_action_uses_command_stream(rby1_robot, fake_rby1_sdk):
    rby1_robot.connect()

    action = {key: float(idx) * 0.1 for idx, key in enumerate(rby1_robot.action_features)}
    action["right_gripper"] = 0.25
    action["left_gripper"] = 0.75
    returned = rby1_robot.send_action(action)

    sdk_robot = fake_rby1_sdk.created_robots[-1]
    sdk_gripper = fake_rby1_sdk.created_buses[-1]
    assert returned == action
    assert len(sdk_robot.stream.sent_commands) == 1
    # position_writes includes an initial write from gripper connect + the send_action write
    assert len(sdk_gripper.position_writes) >= 1

    command, timeout_ms = sdk_robot.stream.sent_commands[0]
    assert timeout_ms == rby1_robot.config.command_timeout_ms
    assert command.command.body_command.torso_command.position == [action[f"torso_{idx}"] for idx in range(6)]
    assert command.command.body_command.right_arm_command.position == [
        action[f"right_arm_{idx}"] for idx in range(7)
    ]
    assert command.command.body_command.left_arm_command.position == [
        action[f"left_arm_{idx}"] for idx in range(7)
    ]
    assert command.command.head_command.position == [action["head_0"], action["head_1"]]
    assert sdk_gripper.position_writes[-1] == [(0, 1000.5), (1, 999.5)]


def test_send_action_impedance_mode(fake_rby1_sdk):
    robot = RBY1(RBY1Config(gripper_home_on_connect=False, use_impedance=True))
    robot.connect()

    action = {key: float(idx) * 0.1 for idx, key in enumerate(robot.action_features)}
    action["right_gripper"] = 0.25
    action["left_gripper"] = 0.75
    returned = robot.send_action(action)

    sdk_robot = fake_rby1_sdk.created_robots[-1]
    assert len(sdk_robot.stream.sent_commands) == 1

    command, _ = sdk_robot.stream.sent_commands[0]
    # In impedance mode, body_command should be a JointImpedanceControlCommandBuilder
    impedance_builder = command.command.body_command
    assert impedance_builder.position is not None
    assert impedance_builder.stiffness is not None
    assert impedance_builder.torque_limit is not None
    assert impedance_builder.damping_ratio is not None
    assert len(impedance_builder.position) == 22  # torso(6) + right(7) + left(7) + head(2)
    assert returned == action

    robot.disconnect()


def test_gripper_usb_latency_optimization(fake_rby1_sdk):
    robot = RBY1(RBY1Config(gripper_home_on_connect=False))
    robot.connect()

    assert fake_rby1_sdk.upc.initialize_device_calls == ["/dev/rby1_gripper"]

    robot.disconnect()


def test_send_action_requires_exact_key_match(rby1_robot):
    rby1_robot.connect()

    bad_action = {key: 0.0 for key in rby1_robot.action_features}
    bad_action.pop("head_1")

    with pytest.raises(KeyError):
        rby1_robot.send_action(bad_action)


def test_send_action_clips_to_absolute_joint_limits(fake_rby1_sdk):
    robot = RBY1(RBY1Config(gripper_home_on_connect=False))
    robot.connect()

    sdk_robot = fake_rby1_sdk.created_robots[-1]
    sdk_robot.dynamics.q_lower[10] = -0.5
    sdk_robot.dynamics.q_upper[10] = 0.5
    robot._joint_lower_limits["right_arm_0"] = -0.5
    robot._joint_upper_limits["right_arm_0"] = 0.5

    action = {key: 0.0 for key in robot.action_features}
    action["right_arm_0"] = 5.0
    action["right_gripper"] = -1.0
    action["left_gripper"] = 2.0

    returned = robot.send_action(action)

    assert returned["right_arm_0"] == pytest.approx(0.5)
    assert returned["right_gripper"] == pytest.approx(0.0)
    assert returned["left_gripper"] == pytest.approx(1.0)

    robot.disconnect()


def test_move_to_initial_pose_uses_fixed_torso_head_and_default_arm_ready_pose(fake_rby1_sdk):
    robot = RBY1(
        RBY1Config(
            gripper_home_on_connect=False,
            fixed_torso_positions=[0.3] * 6,
            fixed_head_positions=[-0.2, 0.4],
        )
    )
    robot.connect()

    initial_action = robot.move_to_initial_pose()
    sdk_robot = fake_rby1_sdk.created_robots[-1]

    assert len(sdk_robot.send_command_calls) == 1
    assert [initial_action[f"torso_{idx}"] for idx in range(6)] == [0.3] * 6
    assert [initial_action[f"right_arm_{idx}"] for idx in range(7)] == list(DEFAULT_READY_RIGHT_ARM_POSITIONS)
    assert [initial_action[f"left_arm_{idx}"] for idx in range(7)] == list(DEFAULT_READY_LEFT_ARM_POSITIONS)
    assert [initial_action["head_0"], initial_action["head_1"]] == [-0.2, 0.4]
    assert sdk_robot.position[4:10] == [0.3] * 6

    robot.disconnect()
