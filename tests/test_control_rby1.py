#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
from unittest.mock import patch

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.teleoperators.master_arm import MasterArmConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.robots.rby1 import RBY1Config
from lerobot.robots.rby1.schema import FULL_ACTION_KEYS
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.fixtures.fake_rby1_sdk import FakeRBY1SDK


def _make_master_arm_config(tmp_path: Path) -> MasterArmConfig:
    model_path = tmp_path / "master_arm.urdf"
    model_path.write_text("<robot name='master_arm'/>")
    return MasterArmConfig(
        robot_address="127.0.0.1:50051",
        master_arm_model_path=model_path,
        fixed_torso_positions=[0.0] * 6,
        fixed_head_positions=[0.0, 0.0],
    )


@TeleoperatorConfig.register_subclass("rby1_test_teleop")
@dataclass
class RBY1TestTeleopConfig(TeleoperatorConfig):
    action_value: float = 0.0


class RBY1TestTeleop(Teleoperator):
    config_class = RBY1TestTeleopConfig
    name = "rby1_test_teleop"

    def __init__(self, config: RBY1TestTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(FULL_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(FULL_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_action(self) -> RobotAction:
        return {key: self.config.action_value for key in self.action_features}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    def disconnect(self) -> None:
        self._is_connected = False


def test_calibrate_with_rby1_config():
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config(gripper_home_on_connect=False)

    with patch("lerobot.robots.rby1.rby1.rby", fake_sdk):
        calibrate(CalibrateConfig(robot=robot_cfg))


def test_calibrate_with_master_arm_config(tmp_path):
    fake_sdk = FakeRBY1SDK()
    teleop_cfg = _make_master_arm_config(tmp_path)

    with patch("lerobot.teleoperators.master_arm.master_arm.rby", fake_sdk):
        calibrate(CalibrateConfig(teleop=teleop_cfg))


def test_teleoperate_with_rby1_config():
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config(gripper_home_on_connect=False)
    teleop_cfg = RBY1TestTeleopConfig()

    with patch("lerobot.robots.rby1.rby1.rby", fake_sdk):
        teleoperate(
            TeleoperateConfig(
                robot=robot_cfg,
                teleop=teleop_cfg,
                teleop_time_s=0.1,
            )
        )


def test_teleoperate_with_master_arm_and_rby1_config(tmp_path):
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config(gripper_home_on_connect=False)
    teleop_cfg = _make_master_arm_config(tmp_path)

    with (
        patch("lerobot.robots.rby1.rby1.rby", fake_sdk),
        patch("lerobot.teleoperators.master_arm.master_arm.rby", fake_sdk),
    ):
        teleoperate(
            TeleoperateConfig(
                robot=robot_cfg,
                teleop=teleop_cfg,
                teleop_time_s=0.1,
            )
        )


def test_record_and_replay_with_rby1_config(tmp_path):
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config(gripper_home_on_connect=False)
    teleop_cfg = RBY1TestTeleopConfig()

    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy RBY1 task",
            root=tmp_path / "rby1_record",
            num_episodes=1,
            episode_time_s=0.1,
            push_to_hub=False,
        ),
        teleop=teleop_cfg,
        play_sounds=False,
    )

    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=DatasetReplayConfig(
            repo_id=DUMMY_REPO_ID,
            episode=0,
            root=tmp_path / "rby1_record",
        ),
        play_sounds=False,
    )

    with patch("lerobot.robots.rby1.rby1.rby", fake_sdk):
        dataset = record(record_cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert len(fake_sdk.created_robots) == 1
    assert len(fake_sdk.created_robots[0].send_command_calls) == 1

    with (
        patch("lerobot.robots.rby1.rby1.rby", fake_sdk),
        patch("lerobot.datasets.lerobot_dataset.get_safe_version", return_value="v3.0"),
        patch("lerobot.datasets.lerobot_dataset.snapshot_download", return_value=str(tmp_path / "rby1_record")),
    ):
        replay(replay_cfg)


def test_record_with_master_arm_and_rby1_config(tmp_path):
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config(gripper_home_on_connect=False)
    teleop_cfg = _make_master_arm_config(tmp_path)

    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy RBY1 master-arm task",
            root=tmp_path / "rby1_master_arm_record",
            num_episodes=1,
            episode_time_s=0.1,
            push_to_hub=False,
        ),
        teleop=teleop_cfg,
        play_sounds=False,
    )

    with (
        patch("lerobot.robots.rby1.rby1.rby", fake_sdk),
        patch("lerobot.teleoperators.master_arm.master_arm.rby", fake_sdk),
    ):
        dataset = record(record_cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    follower_robot = fake_sdk.created_robots[0]
    assert len(follower_robot.send_command_calls) == 1
