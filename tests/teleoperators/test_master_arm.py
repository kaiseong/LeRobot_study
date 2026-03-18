#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.robots.rby1.schema import FULL_ACTION_KEYS
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.master_arm import MasterArm, MasterArmConfig
from tests.fixtures.fake_rby1_sdk import FakeRBY1SDK

_FIXED_TORSO = [0.1 * idx for idx in range(6)]
_FIXED_HEAD = [0.7, -0.2]


def _make_master_arm_config(model_path: Path, **overrides) -> MasterArmConfig:
    values = {
        "robot_address": "127.0.0.1:50051",
        "master_arm_model_path": model_path,
        "fixed_torso_positions": list(_FIXED_TORSO),
        "fixed_head_positions": list(_FIXED_HEAD),
    }
    values.update(overrides)
    return MasterArmConfig(
        **values,
    )


@pytest.fixture
def master_arm_model_path(tmp_path: Path) -> Path:
    model_path = tmp_path / "master_arm.urdf"
    model_path.write_text("<robot name='master_arm'/>")
    return model_path


@pytest.fixture
def fake_rby1_sdk():
    sdk = FakeRBY1SDK()
    with patch("lerobot.teleoperators.master_arm.master_arm.rby", sdk):
        yield sdk


@pytest.fixture
def master_arm(fake_rby1_sdk, master_arm_model_path: Path):
    teleop = MasterArm(_make_master_arm_config(master_arm_model_path))
    yield teleop
    if teleop.is_connected:
        teleop.disconnect()


def test_make_teleoperator_from_config_returns_master_arm(fake_rby1_sdk, master_arm_model_path: Path):
    teleop = make_teleoperator_from_config(_make_master_arm_config(master_arm_model_path))
    assert isinstance(teleop, MasterArm)


def test_connect_disconnect_manages_power_and_control(master_arm, fake_rby1_sdk):
    master_arm.connect()

    power_robot = fake_rby1_sdk.created_robots[-1]
    sdk_master_arm = fake_rby1_sdk.upc.created_master_arms[-1]

    assert master_arm.is_connected
    assert power_robot.power_on_calls == ["12v"]
    assert sdk_master_arm.model_path is not None
    assert sdk_master_arm.start_control_calls == 1

    master_arm.disconnect()

    assert not master_arm.is_connected
    assert sdk_master_arm.stop_control_calls == [False]
    assert power_robot.power_off_calls == []
    assert power_robot.disconnect_calls == 1


def test_disconnect_can_power_off_when_enabled(fake_rby1_sdk, master_arm_model_path: Path):
    teleop = MasterArm(
        _make_master_arm_config(
            master_arm_model_path,
            power_off_12v_on_disconnect=True,
        )
    )
    teleop.connect()

    power_robot = fake_rby1_sdk.created_robots[-1]
    teleop.disconnect()

    assert power_robot.power_off_calls == ["12v"]


def test_get_action_returns_full_rby1_schema(master_arm, fake_rby1_sdk):
    master_arm.connect()
    sdk_master_arm = fake_rby1_sdk.upc.created_master_arms[-1]

    q_joint = np.linspace(-0.7, 0.6, 14)
    sdk_master_arm.emit_state(
        q_joint=q_joint,
        gravity_term=np.ones(14),
        right_button=1,
        left_button=1,
        right_trigger=250,
        left_trigger=750,
    )

    action = master_arm.get_action()

    assert tuple(action) == FULL_ACTION_KEYS
    assert [action[f"torso_{idx}"] for idx in range(6)] == _FIXED_TORSO
    assert [action[f"right_arm_{idx}"] for idx in range(7)] == list(q_joint[:7])
    assert [action[f"left_arm_{idx}"] for idx in range(7)] == list(q_joint[7:14])
    assert [action["head_0"], action["head_1"]] == _FIXED_HEAD
    assert action["right_gripper"] == pytest.approx(0.25)
    assert action["left_gripper"] == pytest.approx(0.75)


def test_get_action_holds_last_arm_targets_when_clutch_released(master_arm, fake_rby1_sdk):
    master_arm.connect()
    sdk_master_arm = fake_rby1_sdk.upc.created_master_arms[-1]

    first_q = np.linspace(0.0, 1.3, 14)
    sdk_master_arm.emit_state(q_joint=first_q, gravity_term=np.ones(14), right_button=1, left_button=1)

    released_q = np.linspace(2.0, 3.3, 14)
    control_input = sdk_master_arm.emit_state(
        q_joint=released_q,
        gravity_term=np.ones(14) * 2.0,
        right_button=0,
        left_button=0,
    )
    action = master_arm.get_action()

    assert [action[f"right_arm_{idx}"] for idx in range(7)] == list(first_q[:7])
    assert [action[f"left_arm_{idx}"] for idx in range(7)] == list(first_q[7:14])
    assert np.all(control_input.target_position[:7] == first_q[:7])
    assert np.all(control_input.target_position[7:14] == first_q[7:14])


def test_align_hold_targets_updates_master_arm_pose(master_arm, fake_rby1_sdk):
    master_arm.connect()
    target_q = np.linspace(-0.4, 0.9, 14)

    action = {key: 0.0 for key in FULL_ACTION_KEYS}
    for idx in range(7):
        action[f"right_arm_{idx}"] = float(target_q[idx])
        action[f"left_arm_{idx}"] = float(target_q[7 + idx])
    action["right_gripper"] = 0.2
    action["left_gripper"] = 0.8

    master_arm.align_hold_targets(action, wait_timeout_s=0.1, position_tolerance_rad=1e-6)
    aligned_action = master_arm.get_action()
    sdk_master_arm = fake_rby1_sdk.upc.created_master_arms[-1]

    assert [aligned_action[f"right_arm_{idx}"] for idx in range(7)] == list(target_q[:7])
    assert [aligned_action[f"left_arm_{idx}"] for idx in range(7)] == list(target_q[7:14])
    assert aligned_action["right_gripper"] == pytest.approx(0.2)
    assert aligned_action["left_gripper"] == pytest.approx(0.8)
    assert np.allclose(sdk_master_arm.current_state.q_joint, target_q)


def test_config_validates_fixed_pose_lengths(master_arm_model_path: Path):
    with pytest.raises(ValueError):
        _make_master_arm_config(master_arm_model_path, fixed_torso_positions=[0.0] * 5)

    with pytest.raises(ValueError):
        _make_master_arm_config(master_arm_model_path, fixed_head_positions=[0.0] * 3)
