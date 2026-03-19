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
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _rby1_sdk_available

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rby1 import RBY1Config
from .gripper import RBY1GripperController
from .schema import (
    BODY_ACTION_KEYS,
    BODY_JOINT_NAMES,
    DEFAULT_READY_HEAD_POSITIONS,
    DEFAULT_READY_LEFT_ARM_POSITIONS,
    DEFAULT_READY_RIGHT_ARM_POSITIONS,
    DEFAULT_READY_TORSO_POSITIONS,
    FULL_ACTION_KEYS,
    GRIPPER_ACTION_KEYS,
    HEAD_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RBY1_M_JOINT_ORDER,
    RIGHT_ARM_JOINT_NAMES,
    TORSO_JOINT_NAMES,
    build_action_features,
)

if TYPE_CHECKING or _rby1_sdk_available:
    import rby1_sdk as rby
else:
    rby = None

logger = logging.getLogger(__name__)


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
        self._joint_index_by_key = self._build_joint_index_by_key(RBY1_M_JOINT_ORDER)
        self._joint_lower_limits: dict[str, float] = {}
        self._joint_upper_limits: dict[str, float] = {}
        self._last_action: dict[str, float] = {key: 0.0 for key in FULL_ACTION_KEYS}
        self._fixed_action_values = self._build_fixed_action_values()
        self._dyn_model: Any | None = None
        self._dyn_state: Any | None = None
        self._cached_position: np.ndarray | None = None
        self._gripper_tool_flanges_powered = False
        self._gripper = self._build_gripper_controller()
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int | None, int | None, int]]:
        return {**self._joint_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_features

    @property
    def _joint_features(self) -> dict[str, type]:
        return build_action_features(FULL_ACTION_KEYS)

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

    @property
    def fixed_action_values(self) -> dict[str, float]:
        return dict(self._fixed_action_values)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
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
            self._joint_lower_limits, self._joint_upper_limits = self._resolve_joint_limits(robot)
            self.configure()
            self._command_stream = robot.create_command_stream(self.config.command_priority)

            self._set_gripper_tool_flange_voltage(robot, 12)
            self._gripper.connect()
            for camera in self.cameras.values():
                if not camera.is_connected:
                    camera.connect()

            self._last_action = self._read_full_state()
            logger.info("%s connected", self)
        except Exception:
            self._cleanup_failed_connect(robot)
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
        observation = self._read_full_state()
        obs_dict: RobotObservation = dict(observation)
        for cam_name, camera in self.cameras.items():
            obs_dict[cam_name] = camera.read_latest()
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._robot is None or self._command_stream is None:
            raise DeviceNotConnectedError()

        sanitized_action = self._sanitize_action(action)
        if self._check_collision(sanitized_action):
            logger.warning("Self-collision detected for the proposed action. Skipping to maintain safety.")
            return dict(self._last_action)
        command = self._build_body_command(sanitized_action)
        self._command_stream.send_command(command, timeout_ms=self.config.command_timeout_ms)

        if self.config.enable_grippers:
            gripper_targets = [sanitized_action[key] for key in GRIPPER_ACTION_KEYS]
            self._gripper.set_targets(gripper_targets)

        self._last_action = dict(sanitized_action)
        return sanitized_action

    @check_if_not_connected
    def move_to_initial_pose(self, teleop_fixed_action_values: dict[str, float] | None = None) -> RobotAction:
        if self._robot is None:
            raise DeviceNotConnectedError()

        initial_action = self.resolve_initial_action(teleop_fixed_action_values=teleop_fixed_action_values)
        body_command = self._build_joint_position_command(initial_action, minimum_time=self.config.initial_move_time_s)
        handler = self._robot.send_command(body_command, self.config.command_priority)
        finish_code = handler.get()
        ok_finish_code = getattr(rby.RobotCommandFeedback.FinishCode, "Ok", None) if rby is not None else None
        if ok_finish_code is not None and finish_code != ok_finish_code:
            raise RuntimeError(f"Failed to move RBY1 to the initial pose. finish_code={finish_code!r}")

        self._wait_until_body_action_reached(
            initial_action,
            timeout_s=self.config.initial_wait_timeout_s,
            tolerance_rad=self.config.initial_position_tolerance_rad,
        )
        self._last_action = self._read_full_state()
        return initial_action

    @check_if_not_connected
    def resolve_initial_action(self, teleop_fixed_action_values: dict[str, float] | None = None) -> RobotAction:
        if self._robot is None:
            raise DeviceNotConnectedError()

        observation = self._read_full_state()
        initial_action = dict(observation)
        effective_fixed_action_values = dict(teleop_fixed_action_values or {})
        effective_fixed_action_values.update(self.fixed_action_values)

        torso_positions = tuple(
            effective_fixed_action_values.get(joint_name, default_value)
            for joint_name, default_value in zip(
                TORSO_JOINT_NAMES,
                DEFAULT_READY_TORSO_POSITIONS,
                strict=True,
            )
        )
        head_positions = tuple(
            effective_fixed_action_values.get(joint_name, default_value)
            for joint_name, default_value in zip(
                HEAD_JOINT_NAMES,
                DEFAULT_READY_HEAD_POSITIONS,
                strict=True,
            )
        )
        right_arm_positions = tuple(
            float(value)
            for value in (
                self.config.initial_right_arm_positions
                if self.config.initial_right_arm_positions is not None
                else DEFAULT_READY_RIGHT_ARM_POSITIONS
            )
        )
        left_arm_positions = tuple(
            float(value)
            for value in (
                self.config.initial_left_arm_positions
                if self.config.initial_left_arm_positions is not None
                else DEFAULT_READY_LEFT_ARM_POSITIONS
            )
        )

        for joint_name, value in zip(TORSO_JOINT_NAMES, torso_positions, strict=True):
            initial_action[joint_name] = float(value)
        for joint_name, value in zip(RIGHT_ARM_JOINT_NAMES, right_arm_positions, strict=True):
            initial_action[joint_name] = float(value)
        for joint_name, value in zip(LEFT_ARM_JOINT_NAMES, left_arm_positions, strict=True):
            initial_action[joint_name] = float(value)
        for joint_name, value in zip(HEAD_JOINT_NAMES, head_positions, strict=True):
            initial_action[joint_name] = float(value)

        self._apply_absolute_joint_limits(initial_action)
        self._clamp_gripper_targets(initial_action)
        return initial_action

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
                self._gripper.disconnect()
            finally:
                try:
                    self._set_gripper_tool_flange_voltage(robot, 0)
                except Exception:  # nosec B110
                    logger.debug("Ignoring tool-flange power-off failure during disconnect.")
                try:
                    robot.disconnect()
                finally:
                    self._command_stream = None
                    self._robot = None
                    self._joint_lower_limits = {}
                    self._joint_upper_limits = {}
                    self._dyn_model = None
                    self._dyn_state = None
                    self._cached_position = None
                    self._gripper_tool_flanges_powered = False

    def _read_full_state(self) -> dict[str, float]:
        if self._robot is None:
            raise DeviceNotConnectedError()

        robot_state = self._robot.get_state()
        positions = robot_state.position
        self._cached_position = np.asarray(positions, dtype=float)
        observation = {key: float(positions[idx]) for key, idx in self._joint_index_by_key.items()}
        if self.config.enable_grippers:
            observation.update(self._gripper.get_positions())
        else:
            observation.update({key: self._last_action[key] for key in GRIPPER_ACTION_KEYS})
        self._last_action.update(observation)
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
        self._apply_absolute_joint_limits(sanitized_action)
        if self.config.max_relative_target is not None:
            safe_subset = {
                key: (sanitized_action[key], self._last_action[key]) for key in BODY_ACTION_KEYS
            }
            bounded_subset = ensure_safe_goal_position(safe_subset, self.config.max_relative_target)
            sanitized_action.update(bounded_subset)

        self._clamp_gripper_targets(sanitized_action)
        return sanitized_action

    def _apply_absolute_joint_limits(self, action: dict[str, float]) -> None:
        for key in BODY_ACTION_KEYS:
            lower_limit = self._joint_lower_limits.get(key)
            upper_limit = self._joint_upper_limits.get(key)
            if lower_limit is None or upper_limit is None:
                continue
            action[key] = float(min(max(action[key], lower_limit), upper_limit))

    @staticmethod
    def _clamp_gripper_targets(action: dict[str, float]) -> None:
        for key in GRIPPER_ACTION_KEYS:
            if key in action:
                action[key] = max(0.0, min(1.0, float(action[key])))

    def _build_body_command(self, action: dict[str, float]) -> Any:
        if rby is None:
            raise ImportError("rby1_sdk is required to construct RBY1 commands.")

        if self.config.use_impedance:
            return self._build_joint_impedance_command(action)
        return self._build_joint_position_command(action)

    def _build_joint_position_command(
        self,
        action: dict[str, float],
        *,
        minimum_time: float | None = None,
        control_hold_time: float | None = None,
    ) -> Any:
        minimum_time = self.config.command_minimum_time if minimum_time is None else float(minimum_time)
        control_hold_time = self.config.control_hold_time if control_hold_time is None else float(control_hold_time)

        def build_joint_command(joint_names: tuple[str, ...]) -> Any:
            return (
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time))
                .set_position([action[joint] for joint in joint_names])
                .set_minimum_time(minimum_time)
            )

        body_command = (
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(build_joint_command(TORSO_JOINT_NAMES))
            .set_right_arm_command(build_joint_command(RIGHT_ARM_JOINT_NAMES))
            .set_left_arm_command(build_joint_command(LEFT_ARM_JOINT_NAMES))
        )

        component_command = (
            rby.ComponentBasedCommandBuilder()
            .set_body_command(body_command)
            .set_head_command(build_joint_command(HEAD_JOINT_NAMES))
        )
        return rby.RobotCommandBuilder().set_command(component_command)

    def _build_joint_impedance_command(self, action: dict[str, float]) -> Any:
        body_positions = [action[joint] for joint in BODY_JOINT_NAMES]

        body_builder = (
            rby.JointImpedanceControlCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.config.control_hold_time))
            .set_position(body_positions)
            .set_minimum_time(self.config.command_minimum_time)
            .set_stiffness(self.config.impedance_stiffness)
            .set_torque_limit(self.config.impedance_torque_limit)
            .set_damping_ratio([self.config.impedance_damping_ratio] * len(body_positions))
        )

        component_command = (
            rby.ComponentBasedCommandBuilder()
            .set_body_command(body_builder)
        )
        return rby.RobotCommandBuilder().set_command(component_command)

    def _build_gripper_controller(self) -> RBY1GripperController:
        return RBY1GripperController(
            rby_module=rby,
            enabled=self.config.enable_grippers,
            device_ids=self.config.gripper_device_ids,
            baud_rate=self.config.gripper_baud_rate,
            torque_constants=self.config.gripper_torque_constants,
            homing_torque=self.config.gripper_homing_torque,
            hold_torque=self.config.gripper_hold_torque,
            homing_sleep_s=self.config.gripper_homing_sleep_s,
            homing_stall_cycles=self.config.gripper_homing_stall_cycles,
            direction_reversed=self.config.gripper_direction_reversed,
            home_on_connect=self.config.gripper_home_on_connect,
        )

    def _set_gripper_tool_flange_voltage(self, robot: Any, voltage: int) -> None:
        if not self.config.enable_grippers:
            return

        target_voltage = int(voltage)
        if target_voltage == 12 and self._gripper_tool_flanges_powered:
            return
        if target_voltage == 0 and not self._gripper_tool_flanges_powered:
            return

        for arm in ("right", "left"):
            if not robot.set_tool_flange_output_voltage(arm, target_voltage):
                raise ConnectionError(
                    f"Failed to set RBY1 tool flange voltage for {arm} arm to {target_voltage}V."
                )

        if target_voltage > 0:
            time.sleep(0.5)
            self._gripper_tool_flanges_powered = True
        else:
            self._gripper_tool_flanges_powered = False

    def _build_fixed_action_values(self) -> dict[str, float]:
        fixed_values: dict[str, float] = {}
        if self.config.fixed_torso_positions is not None:
            for key, value in zip(
                TORSO_JOINT_NAMES,
                self.config.fixed_torso_positions,
                strict=True,
            ):
                fixed_values[key] = float(value)
        if self.config.fixed_head_positions is not None:
            for key, value in zip(
                HEAD_JOINT_NAMES,
                self.config.fixed_head_positions,
                strict=True,
            ):
                fixed_values[key] = float(value)
        return fixed_values

    def _cleanup_failed_connect(self, robot: Any) -> None:
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
            self._gripper.disconnect()
        except Exception:  # nosec B110
            logger.debug("Ignoring gripper disconnect failure during connect cleanup.")

        try:
            self._set_gripper_tool_flange_voltage(robot, 0)
        except Exception:  # nosec B110
            logger.debug("Ignoring tool-flange power-off failure during connect cleanup.")

        try:
            robot.disconnect()
        except Exception:  # nosec B110
            logger.debug("Ignoring robot disconnect failure during connect cleanup.")

        self._robot = None
        self._command_stream = None
        self._joint_lower_limits = {}
        self._joint_upper_limits = {}
        self._dyn_model = None
        self._dyn_state = None
        self._cached_position = None
        self._gripper_tool_flanges_powered = False

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
        missing_joints = [joint_name for joint_name in BODY_JOINT_NAMES if joint_name not in joint_index_by_name]
        if missing_joints:
            raise RuntimeError(f"RBY1 model is missing required joints for LeRobot v1: {missing_joints}")

        return {joint_name: joint_index_by_name[joint_name] for joint_name in BODY_JOINT_NAMES}

    def _resolve_joint_index_by_key(self, robot: Any) -> dict[str, int]:
        try:
            return self._build_joint_index_by_key(tuple(robot.model().robot_joint_names))
        except Exception:
            logger.debug("Falling back to the built-in RBY1-M joint order.", exc_info=True)
            return self._build_joint_index_by_key(RBY1_M_JOINT_ORDER)

    def _resolve_joint_limits(self, robot: Any) -> tuple[dict[str, float], dict[str, float]]:
        try:
            dynamics = robot.get_dynamics()
            model = robot.model()
            state = dynamics.make_state([], model.robot_joint_names)
            self._dyn_model = dynamics
            self._dyn_state = state
            lower_limits = dynamics.get_limit_q_lower(state)
            upper_limits = dynamics.get_limit_q_upper(state)
            return (
                {key: float(lower_limits[idx]) for key, idx in self._joint_index_by_key.items()},
                {key: float(upper_limits[idx]) for key, idx in self._joint_index_by_key.items()},
            )
        except Exception:
            logger.warning("Failed to resolve RBY1 joint limits from the SDK. Absolute joint clipping is disabled.")
            logger.debug("Joint-limit resolution failure details:", exc_info=True)
            return {}, {}

    def _check_collision(self, action: dict[str, float]) -> bool:
        if not self.config.collision_check_enabled:
            return False
        if self._dyn_model is None or self._dyn_state is None:
            return False
        try:
            if self._cached_position is not None:
                q = self._cached_position.copy()
            elif self._robot is not None:
                q = np.asarray(self._robot.get_state().position, dtype=float)
            else:
                return False
            for key, idx in self._joint_index_by_key.items():
                if key in action:
                    q[idx] = float(action[key])
            self._dyn_state.set_q(q)
            self._dyn_model.compute_forward_kinematics(self._dyn_state)
            collisions = self._dyn_model.detect_collisions_or_nearest_links(self._dyn_state, 1)
            return collisions[0].distance < self.config.collision_threshold
        except Exception:
            logger.debug("Collision check failed.", exc_info=True)
            return False

    def _wait_until_body_action_reached(self, action: dict[str, float], *, timeout_s: float, tolerance_rad: float) -> None:
        if timeout_s <= 0:
            return

        deadline = time.monotonic() + timeout_s
        target_positions = np.asarray([action[key] for key in BODY_ACTION_KEYS], dtype=float)
        while True:
            observation = self._read_full_state()
            current_positions = np.asarray([observation[key] for key in BODY_ACTION_KEYS], dtype=float)
            max_error = float(np.max(np.abs(current_positions - target_positions)))
            if max_error <= tolerance_rad:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out while waiting for the RBY1 to reach the initial pose. "
                    f"Maximum joint error was {max_error:.4f} rad."
                )
            time.sleep(min(self.config.command_minimum_time, 0.05))
