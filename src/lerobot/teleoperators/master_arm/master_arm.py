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
import threading
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _rby1_sdk_available

from ...robots.rby1.schema import (
    FULL_ACTION_KEYS,
    GRIPPER_ACTION_KEYS,
    HEAD_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    TORSO_JOINT_NAMES,
    build_action_features,
    normalize_trigger_value,
)
from ..teleoperator import Teleoperator
from .config_master_arm import MasterArmConfig

if TYPE_CHECKING or _rby1_sdk_available:
    import rby1_sdk as rby
else:
    rby = None

logger = logging.getLogger(__name__)

_POWER_DEVICE = "12v"
_INITIAL_STATE_TIMEOUT_S = 1.0


class MasterArm(Teleoperator):
    config_class = MasterArmConfig
    name = "master_arm"

    def __init__(self, config: MasterArmConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self._power_robot: Any | None = None
        self._master_arm: Any | None = None
        self._is_connected = False
        self._powered_12v_on_connect = False
        self._right_arm_target: np.ndarray | None = None
        self._left_arm_target: np.ndarray | None = None
        self._latest_q_joint: np.ndarray | None = None
        self._gripper_targets = np.zeros(len(GRIPPER_ACTION_KEYS), dtype=float)
        self._state_ready = threading.Event()
        self._target_lock = threading.Lock()
        self._fixed_torso = tuple(float(value) for value in config.fixed_torso_positions)
        self._fixed_head = tuple(float(value) for value in config.fixed_head_positions)
        self._ma_min_q = np.asarray(config.ma_min_q, dtype=float)
        self._ma_max_q = np.asarray(config.ma_max_q, dtype=float)
        self._ma_torque_limit = np.asarray(config.ma_torque_limit, dtype=float)
        self._ma_viscous_gain = np.asarray(config.ma_viscous_gain, dtype=float)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return build_action_features(FULL_ACTION_KEYS)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def fixed_action_values(self) -> dict[str, float]:
        fixed_values: dict[str, float] = {}
        fixed_values.update(
            {joint_name: value for joint_name, value in zip(TORSO_JOINT_NAMES, self._fixed_torso, strict=True)}
        )
        fixed_values.update(
            {joint_name: value for joint_name, value in zip(HEAD_JOINT_NAMES, self._fixed_head, strict=True)}
        )
        return fixed_values

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if rby is None:
            raise ImportError("rby1_sdk is required to use the MasterArm teleoperator.")

        model_path = self.config.master_arm_model_path.expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Master arm URDF was not found at {model_path}.")

        self._state_ready.clear()
        self._powered_12v_on_connect = False
        self._right_arm_target = None
        self._left_arm_target = None
        self._latest_q_joint = None
        self._gripper_targets = np.zeros(len(GRIPPER_ACTION_KEYS), dtype=float)

        power_robot = None
        master_arm = None

        try:
            if self.config.manage_power_12v:
                power_robot = rby.create_robot(self.config.robot_address, self.config.robot_model.lower())
                if not power_robot.connect():
                    raise ConnectionError(
                        f"Failed to connect to RBY1 at {self.config.robot_address} to manage master arm power."
                    )

                if not power_robot.is_power_on(_POWER_DEVICE):
                    if not power_robot.power_on(_POWER_DEVICE):
                        raise ConnectionError("Failed to power on UPC 12V for the master arm.")
                    self._powered_12v_on_connect = True

            rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
            master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
            master_arm.set_model_path(str(model_path))
            master_arm.set_control_period(self.config.control_period)

            active_ids = master_arm.initialize(verbose=self.config.initialize_verbose)
            if len(active_ids) != rby.upc.MasterArm.DeviceCount:
                raise ConnectionError(
                    "Mismatch in the number of devices detected for the RBY1 master arm: "
                    f"expected {rby.upc.MasterArm.DeviceCount}, got {len(active_ids)} ({active_ids})."
                )

            if not master_arm.start_control(self._control_loop):
                raise ConnectionError("Failed to start the RBY1 master arm control loop.")

            if not self._state_ready.wait(timeout=max(_INITIAL_STATE_TIMEOUT_S, self.config.control_period * 10)):
                raise TimeoutError("Timed out while waiting for the first RBY1 master arm state.")

            self._power_robot = power_robot
            self._master_arm = master_arm
            self._is_connected = True
            logger.info("%s connected.", self)
        except Exception:
            self._cleanup_partial_connect(master_arm, power_robot)
            raise

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._target_lock:
            if self._right_arm_target is None or self._left_arm_target is None:
                raise DeviceNotConnectedError()
            right_arm_target = self._right_arm_target.copy()
            left_arm_target = self._left_arm_target.copy()
            gripper_targets = self._gripper_targets.copy()

        action: RobotAction = {}
        action.update(
            {
                joint_name: value
                for joint_name, value in zip(TORSO_JOINT_NAMES, self._fixed_torso, strict=True)
            }
        )
        action.update(
            {
                joint_name: float(value)
                for joint_name, value in zip(RIGHT_ARM_JOINT_NAMES, right_arm_target, strict=True)
            }
        )
        action.update(
            {
                joint_name: float(value)
                for joint_name, value in zip(LEFT_ARM_JOINT_NAMES, left_arm_target, strict=True)
            }
        )
        action.update(
            {
                joint_name: value
                for joint_name, value in zip(HEAD_JOINT_NAMES, self._fixed_head, strict=True)
            }
        )
        action.update({key: float(value) for key, value in zip(GRIPPER_ACTION_KEYS, gripper_targets, strict=True)})
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    @check_if_not_connected
    def align_hold_targets(
        self,
        action: RobotAction,
        *,
        wait_timeout_s: float = 0.0,
        position_tolerance_rad: float = 0.05,
    ) -> None:
        target_q = np.asarray(
            [action[joint_name] for joint_name in RIGHT_ARM_JOINT_NAMES + LEFT_ARM_JOINT_NAMES],
            dtype=float,
        )
        gripper_targets = np.asarray(
            [float(action[key]) for key in GRIPPER_ACTION_KEYS if key in action],
            dtype=float,
        )

        with self._target_lock:
            self._right_arm_target = target_q[: len(RIGHT_ARM_JOINT_NAMES)].copy()
            self._left_arm_target = target_q[len(RIGHT_ARM_JOINT_NAMES) :].copy()
            if len(gripper_targets) == len(GRIPPER_ACTION_KEYS):
                self._gripper_targets = gripper_targets.copy()

        master_arm = self._master_arm
        if master_arm is not None and hasattr(master_arm, "emit_state"):
            right_trigger = 0
            left_trigger = 0
            if len(gripper_targets) == len(GRIPPER_ACTION_KEYS):
                right_trigger = int(round(float(gripper_targets[0]) * self.config.trigger_max_value))
                left_trigger = int(round(float(gripper_targets[1]) * self.config.trigger_max_value))
            master_arm.emit_state(
                q_joint=target_q,
                right_button=0,
                left_button=0,
                right_trigger=right_trigger,
                left_trigger=left_trigger,
            )

        if wait_timeout_s <= 0:
            return

        deadline = time.monotonic() + wait_timeout_s
        while True:
            with self._target_lock:
                latest_q_joint = None if self._latest_q_joint is None else self._latest_q_joint.copy()

            if latest_q_joint is not None:
                max_error = float(np.max(np.abs(latest_q_joint[: len(target_q)] - target_q)))
                if max_error <= position_tolerance_rad:
                    return
            else:
                max_error = float("inf")

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out while waiting for the master arm to reach the initial pose. "
                    f"Maximum joint error was {max_error:.4f} rad."
                )
            time.sleep(min(self.config.control_period, 0.05))

    @check_if_not_connected
    def disconnect(self) -> None:
        self._disconnect_impl(power_off_12v=self.config.power_off_12v_on_disconnect and self._powered_12v_on_connect)

    def _disconnect_impl(self, power_off_12v: bool) -> None:
        master_arm = self._master_arm
        power_robot = self._power_robot

        try:
            if master_arm is not None:
                try:
                    master_arm.stop_control(torque_disable=False)
                except Exception:  # nosec B110
                    logger.debug("Ignoring RBY1 master arm stop_control failure during disconnect.")

            if power_robot is not None:
                if power_off_12v:
                    try:
                        power_robot.power_off(_POWER_DEVICE)
                    except Exception:  # nosec B110
                        logger.debug("Ignoring UPC 12V power-off failure during disconnect.")
                try:
                    power_robot.disconnect()
                except Exception:  # nosec B110
                    logger.debug("Ignoring power robot disconnect failure during disconnect.")
        finally:
            self._power_robot = None
            self._master_arm = None
            self._is_connected = False
            self._powered_12v_on_connect = False
            self._state_ready.clear()
            self._right_arm_target = None
            self._left_arm_target = None
            self._latest_q_joint = None
            self._gripper_targets = np.zeros(len(GRIPPER_ACTION_KEYS), dtype=float)

    def _cleanup_partial_connect(self, master_arm: Any | None, power_robot: Any | None) -> None:
        try:
            if master_arm is not None:
                try:
                    master_arm.stop_control(torque_disable=False)
                except Exception:  # nosec B110
                    logger.debug("Ignoring RBY1 master arm stop_control failure during connect cleanup.")

            if power_robot is not None:
                if self._powered_12v_on_connect:
                    try:
                        power_robot.power_off(_POWER_DEVICE)
                    except Exception:  # nosec B110
                        logger.debug("Ignoring UPC 12V power-off failure during connect cleanup.")
                try:
                    power_robot.disconnect()
                except Exception:  # nosec B110
                    logger.debug("Ignoring power robot disconnect failure during connect cleanup.")
        finally:
            self._power_robot = None
            self._master_arm = None
            self._is_connected = False
            self._powered_12v_on_connect = False
            self._state_ready.clear()
            self._right_arm_target = None
            self._left_arm_target = None
            self._latest_q_joint = None
            self._gripper_targets = np.zeros(len(GRIPPER_ACTION_KEYS), dtype=float)

    def _control_loop(self, state: Any) -> Any:
        if rby is None:
            raise ImportError("rby1_sdk is required to use the MasterArm teleoperator.")

        q_joint = np.asarray(state.q_joint, dtype=float)
        gravity_term = np.asarray(state.gravity_term, dtype=float)
        right_arm_q = q_joint[: len(RIGHT_ARM_JOINT_NAMES)].copy()
        left_arm_q = q_joint[len(RIGHT_ARM_JOINT_NAMES) : len(RIGHT_ARM_JOINT_NAMES) + len(LEFT_ARM_JOINT_NAMES)].copy()
        qvel_joint = np.asarray(getattr(state, "qvel_joint", np.zeros_like(q_joint)), dtype=float)
        barrier_torque = self.config.q_limit_barrier_gain * (
            np.maximum(self._ma_min_q - q_joint, 0.0)
            + np.minimum(self._ma_max_q - q_joint, 0.0)
        )
        viscous_torque = self._ma_viscous_gain * qvel_joint
        current_torque = np.clip(
            gravity_term + barrier_torque + viscous_torque,
            -self._ma_torque_limit,
            self._ma_torque_limit,
        )

        right_button = bool(getattr(state.button_right, "button", 0))
        left_button = bool(getattr(state.button_left, "button", 0))
        right_gripper = normalize_trigger_value(
            getattr(state.button_right, "trigger", 0),
            maximum_value=self.config.trigger_max_value,
        )
        left_gripper = normalize_trigger_value(
            getattr(state.button_left, "trigger", 0),
            maximum_value=self.config.trigger_max_value,
        )

        with self._target_lock:
            self._latest_q_joint = q_joint.copy()
            if self._right_arm_target is None:
                self._right_arm_target = right_arm_q
            if self._left_arm_target is None:
                self._left_arm_target = left_arm_q

            if right_button:
                self._right_arm_target = right_arm_q
            if left_button:
                self._left_arm_target = left_arm_q

            self._gripper_targets = np.asarray([right_gripper, left_gripper], dtype=float)
            right_arm_target = self._right_arm_target.copy()
            left_arm_target = self._left_arm_target.copy()

        control_input = rby.upc.MasterArm.ControlInput()

        if right_button:
            control_input.target_operating_mode[: len(RIGHT_ARM_JOINT_NAMES)] = rby.DynamixelBus.CurrentControlMode
            control_input.target_torque[: len(RIGHT_ARM_JOINT_NAMES)] = current_torque[: len(RIGHT_ARM_JOINT_NAMES)]
        else:
            control_input.target_operating_mode[: len(RIGHT_ARM_JOINT_NAMES)] = (
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            control_input.target_position[: len(RIGHT_ARM_JOINT_NAMES)] = right_arm_target
            control_input.target_torque[: len(RIGHT_ARM_JOINT_NAMES)] = self._ma_torque_limit[: len(RIGHT_ARM_JOINT_NAMES)]

        left_start = len(RIGHT_ARM_JOINT_NAMES)
        left_end = left_start + len(LEFT_ARM_JOINT_NAMES)
        if left_button:
            control_input.target_operating_mode[left_start:left_end] = rby.DynamixelBus.CurrentControlMode
            control_input.target_torque[left_start:left_end] = current_torque[left_start:left_end]
        else:
            control_input.target_operating_mode[left_start:left_end] = rby.DynamixelBus.CurrentBasedPositionControlMode
            control_input.target_position[left_start:left_end] = left_arm_target
            control_input.target_torque[left_start:left_end] = self._ma_torque_limit[left_start:left_end]

        self._state_ready.set()
        return control_input
