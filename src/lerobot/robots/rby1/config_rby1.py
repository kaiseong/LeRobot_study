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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


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

    disable_torque_on_disconnect: bool = False
    disable_control_manager_on_disconnect: bool = True
    reset_fault_control_manager_on_connect: bool = True
    unlimited_mode_enabled: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
