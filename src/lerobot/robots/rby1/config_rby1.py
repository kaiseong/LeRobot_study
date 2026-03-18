#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import List

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .schema import HEAD_JOINT_NAMES, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, TORSO_JOINT_NAMES


@RobotConfig.register_subclass("rby1")
@dataclass
class RBY1Config(RobotConfig):
    # Robot network address.
    address: str = "192.168.30.1:50051"
    # V1 targets the mecanum-base model and excludes mobility from the action space.
    model: str = "m"
    # Regex pattern passed to the SDK to power required devices on connect.
    power_device_pattern: str = ".*"
    # Regex pattern passed to the SDK to enable the controlled joints on connect.
    servo_device_pattern: str = "torso_.*|right_arm_.*|left_arm_.*|head_.*"
    # Command-stream priority used for continuous joint position commands.
    command_priority: int = 1
    # Minimum execution time for each streamed joint position command.
    command_minimum_time: float = 0.1
    # Control hold time attached to each streamed joint position command.
    control_hold_time: float = 1.0
    # Timeout used when sending a command through the command stream.
    command_timeout_ms: int = 1000
    # Optional wait after enabling the control manager before streaming commands.
    wait_for_control_ready_timeout_ms: int = 5000
    # Optional SDK parameter tuning for smoother position control.
    joint_position_command_cutoff_frequency: float | None = 5.0

    # Optional fixed values used when a policy does not predict torso/head commands.
    fixed_torso_positions: list[float] | None = None
    fixed_head_positions: list[float] | None = None
    # Record-start preparation pose applied once per lerobot-record session.
    prepare_initial_pose: bool = True
    initial_right_arm_positions: list[float] | None = None
    initial_left_arm_positions: list[float] | None = None
    initial_move_time_s: float = 5.0
    initial_wait_timeout_s: float = 8.0
    initial_position_tolerance_rad: float = 0.05

    # UPC gripper control.
    enable_grippers: bool = True
    gripper_device_ids: list[int] = field(default_factory=lambda: [0, 1])
    gripper_baud_rate: int = 2_000_000
    gripper_torque_constants: list[float] = field(default_factory=lambda: [1.0, 1.0])
    gripper_home_on_connect: bool = True
    gripper_homing_torque: float = 0.5
    gripper_hold_torque: float = 5.0
    gripper_homing_sleep_s: float = 0.1
    gripper_homing_stall_cycles: int = 30
    gripper_direction_reversed: bool = False

    # Impedance control: when True, send_action uses JointImpedanceControlCommandBuilder
    # instead of JointPositionCommandBuilder, yielding compliant behaviour.
    use_impedance: bool = False

    # Joint stiffness (Nm/rad) for the 22 body joints in order:
    #   torso_0…5 (6), right_arm_0…6 (7), left_arm_0…6 (7), head_0…1 (2)
    # Higher values → stiffer, lower → more compliant.
    impedance_stiffness: List[float] = field(
        default_factory=lambda: [400.0] * 6 + [100.0] * 7 + [100.0] * 7 + [400.0] * 2
    )

    # Torque limits (Nm) for the 22 body joints (same ordering as above).
    impedance_torque_limit: List[float] = field(
        default_factory=lambda: [500.0] * 6 + [40.0] * 7 + [40.0] * 7 + [500.0] * 2
    )

    # Damping ratio applied to all joints [0.0, 1.0].
    # 0.7 gives critical damping; lower values allow faster motion.
    impedance_damping_ratio: float = 0.7

    disable_torque_on_disconnect: bool = False
    disable_control_manager_on_disconnect: bool = True
    reset_fault_control_manager_on_connect: bool = True
    unlimited_mode_enabled: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # Self-collision safety check applied in send_action().
    collision_check_enabled: bool = True
    collision_threshold: float = 0.02

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.fixed_torso_positions is not None and len(self.fixed_torso_positions) != len(TORSO_JOINT_NAMES):
            raise ValueError("fixed_torso_positions must contain exactly 6 values.")
        if self.fixed_head_positions is not None and len(self.fixed_head_positions) != len(HEAD_JOINT_NAMES):
            raise ValueError("fixed_head_positions must contain exactly 2 values.")
        if self.initial_right_arm_positions is not None and len(self.initial_right_arm_positions) != len(RIGHT_ARM_JOINT_NAMES):
            raise ValueError("initial_right_arm_positions must contain exactly 7 values.")
        if self.initial_left_arm_positions is not None and len(self.initial_left_arm_positions) != len(LEFT_ARM_JOINT_NAMES):
            raise ValueError("initial_left_arm_positions must contain exactly 7 values.")
        if self.initial_move_time_s <= 0:
            raise ValueError("initial_move_time_s must be positive.")
        if self.initial_wait_timeout_s < 0:
            raise ValueError("initial_wait_timeout_s must be non-negative.")
        if self.initial_position_tolerance_rad <= 0:
            raise ValueError("initial_position_tolerance_rad must be positive.")
        if len(self.gripper_device_ids) != 2:
            raise ValueError("gripper_device_ids must contain exactly two device ids.")
        if len(self.gripper_torque_constants) != len(self.gripper_device_ids):
            raise ValueError("gripper_torque_constants must match gripper_device_ids length.")
