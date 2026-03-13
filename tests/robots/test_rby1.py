#!/usr/bin/env python

from __future__ import annotations

from unittest.mock import patch

import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.rby1 import RBY1, RBY1Config
from tests.fixtures.fake_rby1_sdk import FakeControlManagerState, FakeRBY1SDK


@pytest.fixture
def fake_rby1_sdk():
    sdk = FakeRBY1SDK()
    with patch("lerobot.robots.rby1.rby1.rby", sdk):
        yield sdk


@pytest.fixture
def rby1_robot(fake_rby1_sdk):
    robot = RBY1(RBY1Config())
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
    assert sdk_robot.power_on_calls == [rby1_robot.config.power_device_pattern]
    assert sdk_robot.servo_on_calls == [rby1_robot.config.servo_device_pattern]
    assert sdk_robot.create_command_stream_calls == [rby1_robot.config.command_priority]

    rby1_robot.disconnect()
    assert not rby1_robot.is_connected
    assert sdk_robot.stream.cancelled is True
    assert sdk_robot.disable_control_manager_calls == 1
    assert sdk_robot.servo_off_calls == [rby1_robot.config.servo_device_pattern]


def test_connect_resets_faulted_control_manager(rby1_robot, fake_rby1_sdk):
    sdk_robot = fake_rby1_sdk.create_robot(rby1_robot.config.address, rby1_robot.config.model)
    sdk_robot.control_manager_state.state = FakeControlManagerState.State.MajorFault

    with patch("lerobot.robots.rby1.rby1.rby.create_robot", return_value=sdk_robot):
        rby1_robot.connect()

    assert sdk_robot.reset_fault_calls == 1


def test_get_observation_returns_expected_joint_keys(rby1_robot):
    rby1_robot.connect()
    obs = rby1_robot.get_observation()

    assert set(obs) == set(rby1_robot.observation_features)
    for key in rby1_robot.action_features:
        assert isinstance(obs[key], float)


def test_send_action_uses_command_stream(rby1_robot, fake_rby1_sdk):
    rby1_robot.connect()

    action = {key: float(idx) for idx, key in enumerate(rby1_robot.action_features)}
    returned = rby1_robot.send_action(action)

    sdk_robot = fake_rby1_sdk.created_robots[-1]
    assert returned == action
    assert len(sdk_robot.stream.sent_commands) == 1

    command, timeout_ms = sdk_robot.stream.sent_commands[0]
    assert timeout_ms == rby1_robot.config.command_timeout_ms
    assert command.command.body_command.torso_command.position == [action[f"torso_{idx}.pos"] for idx in range(6)]
    assert command.command.body_command.right_arm_command.position == [
        action[f"right_arm_{idx}.pos"] for idx in range(7)
    ]
    assert command.command.body_command.left_arm_command.position == [
        action[f"left_arm_{idx}.pos"] for idx in range(7)
    ]
    assert command.command.head_command.position == [action["head_0.pos"], action["head_1.pos"]]


def test_send_action_requires_exact_key_match(rby1_robot):
    rby1_robot.connect()

    bad_action = {key: 0.0 for key in rby1_robot.action_features}
    bad_action.pop("head_1.pos")

    with pytest.raises(KeyError):
        rby1_robot.send_action(bad_action)
