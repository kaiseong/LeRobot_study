#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

import numpy as np

from ..config import TeleoperatorConfig
from ...robots.rby1.schema import HEAD_JOINT_NAMES, TORSO_JOINT_NAMES, TRIGGER_MAX_VALUE


@TeleoperatorConfig.register_subclass("master_arm")
@dataclass(kw_only=True)
class MasterArmConfig(TeleoperatorConfig):
    # Address/model used to manage UPC 12V power for the master arm.
    robot_address: str
    robot_model: str = "m"
    manage_power_12v: bool = True
    power_off_12v_on_disconnect: bool = False

    # UPC master arm model/configuration.
    master_arm_model_path: Path
    control_period: float = 0.01
    initialize_verbose: bool = False

    # Torso/head values are fixed in v1 and arm values come from the master arm.
    fixed_torso_positions: list[float]
    fixed_head_positions: list[float]
    trigger_max_value: float = TRIGGER_MAX_VALUE

    # Master arm joint limit barrier: pushes back when approaching limits.
    q_limit_barrier_gain: float = 0.5
    ma_min_q: list[float] = field(
        default_factory=lambda: np.deg2rad(
            [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
        ).tolist()
    )
    ma_max_q: list[float] = field(
        default_factory=lambda: np.deg2rad(
            [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
        ).tolist()
    )
    # Per-joint torque limits for the master arm (7 right + 7 left).
    ma_torque_limit: list[float] = field(
        default_factory=lambda: [3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2
    )
    # Velocity-proportional damping gains for the master arm.
    ma_viscous_gain: list[float] = field(
        default_factory=lambda: [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2
    )

    def __post_init__(self):
        raw_model_path = self.master_arm_model_path
        if not str(raw_model_path).strip():
            raise ValueError("master_arm_model_path must be provided.")
        self.master_arm_model_path = Path(raw_model_path)
        if len(self.fixed_torso_positions) != len(TORSO_JOINT_NAMES):
            raise ValueError("fixed_torso_positions must contain exactly 6 values.")
        if len(self.fixed_head_positions) != len(HEAD_JOINT_NAMES):
            raise ValueError("fixed_head_positions must contain exactly 2 values.")
        if self.control_period <= 0:
            raise ValueError("control_period must be positive.")
        if self.trigger_max_value <= 0:
            raise ValueError("trigger_max_value must be positive.")
        if len(self.ma_min_q) != len(self.ma_max_q):
            raise ValueError("ma_min_q and ma_max_q must have the same length.")
        if len(self.ma_torque_limit) != len(self.ma_min_q):
            raise ValueError("ma_torque_limit must have the same length as ma_min_q.")
        if len(self.ma_viscous_gain) != len(self.ma_min_q):
            raise ValueError("ma_viscous_gain must have the same length as ma_min_q.")
