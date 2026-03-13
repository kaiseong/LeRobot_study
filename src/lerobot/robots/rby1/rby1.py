#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _rby1_sdk_available

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rby1 import RBY1Config

if TYPE_CHECKING or _rby1_sdk_available:
    import rby1_sdk as rby
else:
    rby = None

logger = logging.getLogger(__name__)

_RBY1_M_JOINT_ORDER = (
    "wheel_fr",
    "wheel_fl",
    "wheel_rr",
    "wheel_rl",
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "head_0",
    "head_1",
)
_TORSO_JOINTS = tuple(f"torso_{idx}" for idx in range(6))
_RIGHT_ARM_JOINTS = tuple(f"right_arm_{idx}" for idx in range(7))
_LEFT_ARM_JOINTS = tuple(f"left_arm_{idx}" for idx in range(7))
_HEAD_JOINTS = ("head_0", "head_1")
_ACTIVE_JOINT_NAMES = _TORSO_JOINTS + _RIGHT_ARM_JOINTS + _LEFT_ARM_JOINTS + _HEAD_JOINTS
_ACTIVE_JOINT_KEYS = tuple(f"{joint}.pos" for joint in _ACTIVE_JOINT_NAMES)


class RBY1(Robot):
    config_class = RBY1Config
    name = "rby1"

    def __init__(self, config: RBY1Config):
        super().__init__(config)
        self.config = config
        if config.model.lower() != "m":
            raise NotImplementedError("RBY1 v1 currently supports only model='m'.")

        self._robot: Any | None = None
        self._command_stream: Any | None = None
        self._joint_index_by_key = self._build_joint_index_by_key(_RBY1_M_JOINT_ORDER)
        self._last_joint_positions: dict[str, float] = {key: 0.0 for key in _ACTIVE_JOINT_KEYS}
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int | None, int | None, int]]:
        return {**self._joint_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_features

    @property
    def _joint_features(self) -> dict[str, type]:
        return dict.fromkeys(_ACTIVE_JOINT_KEYS, float)

    @property
    def _camera_features(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {
            cam_name: (self.config.cameras[cam_name].height, self.config.cameras[cam_name].width, 3)
            for cam_name in self.cameras
        }

    @property
    def is_connected(self) -> bool:
        return bool(self._robot is not None and self._robot.is_connected())

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # The SDK robot is already calibrated.
        if rby is None:
            raise ImportError("rby1_sdk is required to use the RBY1 robot integration.")

        robot = rby.create_robot(self.config.address, self.config.model.lower())
        if not robot.connect():
            raise ConnectionError(f"Failed to connect to RBY1 at {self.config.address}.")

        try:
            if not robot.is_power_on(self.config.power_device_pattern):
                if not robot.power_on(self.config.power_device_pattern):
                    raise ConnectionError(
                        f"Failed to power on RBY1 devices matching '{self.config.power_device_pattern}'."
                    )

            if not robot.is_servo_on(self.config.servo_device_pattern):
                if not robot.servo_on(self.config.servo_device_pattern):
                    raise ConnectionError(
                        f"Failed to servo on RBY1 joints matching '{self.config.servo_device_pattern}'."
                    )

            control_manager_state = robot.get_control_manager_state()
            if self._is_control_manager_fault(control_manager_state):
                if self.config.reset_fault_control_manager_on_connect:
                    if not robot.reset_fault_control_manager():
                        raise ConnectionError("Failed to reset the RBY1 control manager fault.")
                else:
                    raise ConnectionError("RBY1 control manager is in a fault state.")

            if not robot.enable_control_manager(self.config.unlimited_mode_enabled):
                raise ConnectionError("Failed to enable the RBY1 control manager.")

            if self.config.wait_for_control_ready_timeout_ms > 0:
                try:
                    if not robot.wait_for_control_ready(self.config.wait_for_control_ready_timeout_ms):
                        logger.warning("RBY1 control did not report ready before timeout.")
                except AttributeError:
                    logger.debug("wait_for_control_ready is not available in this SDK build.")

            self._robot = robot
            self._joint_index_by_key = self._resolve_joint_index_by_key(robot)
            self.configure()
            self._command_stream = robot.create_command_stream(self.config.command_priority)

            for camera in self.cameras.values():
                if not camera.is_connected:
                    camera.connect()

            self._last_joint_positions = self._read_joint_positions()
            logger.info(f"{self} connected")
        except Exception:
            if self._command_stream is not None:
                try:
                    self._command_stream.cancel()
                except Exception:  # nosec B110
                    logger.debug("Ignoring command-stream cancellation failure during connect cleanup.")
            for camera in self.cameras.values():
                if camera.is_connected:
                    try:
                        camera.disconnect()
                    except Exception:  # nosec B110
                        logger.debug("Ignoring camera disconnect failure during connect cleanup.")
            try:
                robot.disconnect()
            except Exception:  # nosec B110
                pass
            self._robot = None
            self._command_stream = None
            raise

    def calibrate(self) -> None:
        return

    @check_if_not_connected
    def configure(self) -> None:
        if self._robot is None:
            raise DeviceNotConnectedError()

        if self.config.joint_position_command_cutoff_frequency is None:
            return

        success = self._robot.set_parameter(
            "joint_position_command.cutoff_frequency",
            str(self.config.joint_position_command_cutoff_frequency),
        )
        if not success:
            logger.warning("Failed to set joint_position_command.cutoff_frequency on RBY1.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation = self._read_joint_positions()

        obs_dict: RobotObservation = dict(observation)
        for cam_name, camera in self.cameras.items():
            obs_dict[cam_name] = camera.read_latest()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._robot is None or self._command_stream is None:
            raise DeviceNotConnectedError()

        sanitized_action = self._sanitize_action(action)
        command = self._build_joint_position_command(sanitized_action)
        self._command_stream.send_command(command, timeout_ms=self.config.command_timeout_ms)
        self._last_joint_positions = dict(sanitized_action)
        return sanitized_action

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._robot is None:
            raise DeviceNotConnectedError()

        robot = self._robot
        stream = self._command_stream

        try:
            if stream is not None:
                try:
                    stream.cancel()
                except Exception:  # nosec B110
                    logger.debug("Ignoring command-stream cancellation failure during disconnect.")

            if self.config.disable_control_manager_on_disconnect:
                try:
                    robot.disable_control_manager()
                except Exception:  # nosec B110
                    logger.debug("Ignoring control-manager shutdown failure during disconnect.")

            if self.config.disable_torque_on_disconnect:
                try:
                    robot.servo_off(self.config.servo_device_pattern)
                except Exception:  # nosec B110
                    logger.debug("Ignoring servo_off failure during disconnect.")
        finally:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            try:
                robot.disconnect()
            finally:
                self._command_stream = None
                self._robot = None

    def _read_joint_positions(self) -> dict[str, float]:
        if self._robot is None:
            raise DeviceNotConnectedError()

        state = self._robot.get_state()
        positions = state.position
        observation = {key: float(positions[idx]) for key, idx in self._joint_index_by_key.items()}
        self._last_joint_positions = observation
        return observation

    def _sanitize_action(self, action: RobotAction) -> dict[str, float]:
        action_keys = set(action)
        expected_keys = set(self.action_features)

        missing_keys = expected_keys - action_keys
        extra_keys = action_keys - expected_keys
        if missing_keys or extra_keys:
            raise KeyError(
                "RBY1 action keys must exactly match the configured action space. "
                f"Missing={sorted(missing_keys)} Extra={sorted(extra_keys)}"
            )

        sanitized_action = {key: float(action[key]) for key in self.action_features}
        if self.config.max_relative_target is not None:
            goal_present_pos = {
                key: (sanitized_action[key], self._last_joint_positions[key]) for key in sanitized_action
            }
            sanitized_action = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        return sanitized_action

    def _build_joint_position_command(self, action: dict[str, float]) -> Any:
        if rby is None:
            raise ImportError("rby1_sdk is required to construct RBY1 commands.")

        def build_joint_command(joint_names: tuple[str, ...]) -> Any:
            return (
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.config.control_hold_time))
                .set_position([action[f"{joint}.pos"] for joint in joint_names])
                .set_minimum_time(self.config.command_minimum_time)
            )

        body_command = (
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(build_joint_command(_TORSO_JOINTS))
            .set_right_arm_command(build_joint_command(_RIGHT_ARM_JOINTS))
            .set_left_arm_command(build_joint_command(_LEFT_ARM_JOINTS))
        )

        component_command = (
            rby.ComponentBasedCommandBuilder()
            .set_body_command(body_command)
            .set_head_command(build_joint_command(_HEAD_JOINTS))
        )
        return rby.RobotCommandBuilder().set_command(component_command)

    @staticmethod
    def _is_control_manager_fault(control_manager_state: Any) -> bool:
        if rby is None:
            return False

        fault_states = {
            rby.ControlManagerState.State.MinorFault,
            rby.ControlManagerState.State.MajorFault,
        }
        return getattr(control_manager_state, "state", None) in fault_states

    @staticmethod
    def _build_joint_index_by_key(joint_names: tuple[str, ...] | list[str]) -> dict[str, int]:
        joint_index_by_name = {joint_name: idx for idx, joint_name in enumerate(joint_names)}
        missing_joints = [joint_name for joint_name in _ACTIVE_JOINT_NAMES if joint_name not in joint_index_by_name]
        if missing_joints:
            raise RuntimeError(f"RBY1 model is missing required joints for LeRobot v1: {missing_joints}")

        return {f"{joint_name}.pos": joint_index_by_name[joint_name] for joint_name in _ACTIVE_JOINT_NAMES}

    def _resolve_joint_index_by_key(self, robot: Any) -> dict[str, int]:
        try:
            return self._build_joint_index_by_key(tuple(robot.model().robot_joint_names))
        except Exception:
            logger.debug("Falling back to the built-in RBY1-M joint order.", exc_info=True)
            return self._build_joint_index_by_key(_RBY1_M_JOINT_ORDER)
