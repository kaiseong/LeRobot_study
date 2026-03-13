#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any
from unittest.mock import patch

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.robots.rby1 import RBY1Config
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.fixtures.fake_rby1_sdk import FakeRBY1SDK

_RBY1_ACTION_KEYS = (
    *(f"torso_{idx}.pos" for idx in range(6)),
    *(f"right_arm_{idx}.pos" for idx in range(7)),
    *(f"left_arm_{idx}.pos" for idx in range(7)),
    "head_0.pos",
    "head_1.pos",
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
        return dict.fromkeys(_RBY1_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(_RBY1_ACTION_KEYS, float)

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
    robot_cfg = RBY1Config()

    with patch("lerobot.robots.rby1.rby1.rby", fake_sdk):
        calibrate(CalibrateConfig(robot=robot_cfg))


def test_teleoperate_with_rby1_config():
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config()
    teleop_cfg = RBY1TestTeleopConfig()

    with patch("lerobot.robots.rby1.rby1.rby", fake_sdk):
        teleoperate(
            TeleoperateConfig(
                robot=robot_cfg,
                teleop=teleop_cfg,
                teleop_time_s=0.1,
            )
        )


def test_record_and_replay_with_rby1_config(tmp_path):
    fake_sdk = FakeRBY1SDK()
    robot_cfg = RBY1Config()
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

    with (
        patch("lerobot.robots.rby1.rby1.rby", fake_sdk),
        patch("lerobot.datasets.lerobot_dataset.get_safe_version", return_value="v3.0"),
        patch("lerobot.datasets.lerobot_dataset.snapshot_download", return_value=str(tmp_path / "rby1_record")),
    ):
        replay(replay_cfg)
