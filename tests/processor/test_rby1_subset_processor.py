#!/usr/bin/env python

from __future__ import annotations

import torch

from lerobot.processor.converters import create_transition
from lerobot.processor.rby1_subset_processor import (
    RBY1ExpandRobotActionSubsetProcessorStep,
    RBY1JointSubsetProcessorStep,
)
from lerobot.robots.rby1.schema import FULL_ACTION_KEYS
from lerobot.utils.constants import OBS_STATE


def test_rby1_joint_subset_processor_slices_state_and_action():
    selected_names = [f"right_arm_{idx}" for idx in range(7)]
    step = RBY1JointSubsetProcessorStep(
        observation_state_names=list(FULL_ACTION_KEYS),
        action_names=list(FULL_ACTION_KEYS),
        selected_state_names=selected_names,
        selected_action_names=selected_names,
    )

    transition = create_transition(
        observation={OBS_STATE: torch.arange(len(FULL_ACTION_KEYS), dtype=torch.float32)},
        action=torch.arange(len(FULL_ACTION_KEYS), dtype=torch.float32),
    )
    processed = step(transition)

    assert processed["observation"][OBS_STATE].shape[-1] == len(selected_names)
    assert processed["action"].shape[-1] == len(selected_names)
    assert processed["action"].tolist() == list(range(6, 13))


def test_rby1_expand_robot_action_subset_processor_uses_fixed_and_observation_values():
    fixed_action_values = {
        f"torso_{idx}": float(idx) for idx in range(6)
    } | {"head_0": 0.3, "head_1": -0.4}
    step = RBY1ExpandRobotActionSubsetProcessorStep(
        full_action_names=list(FULL_ACTION_KEYS),
        fixed_action_values=fixed_action_values,
    )

    observation = {key: float(index) * 0.1 for index, key in enumerate(FULL_ACTION_KEYS)}
    subset_action = {f"right_arm_{idx}": float(idx) for idx in range(7)}
    subset_action.update({f"left_arm_{idx}": float(idx) + 10.0 for idx in range(7)})

    processed = step(create_transition(observation=observation, action=subset_action))

    assert processed["action"]["torso_0"] == 0.0
    assert processed["action"]["head_0"] == 0.3
    assert processed["action"]["right_arm_6"] == 6.0
    assert processed["action"]["left_arm_6"] == 16.0
    assert processed["action"]["right_gripper"] == observation["right_gripper"]
    assert processed["action"]["left_gripper"] == observation["left_gripper"]
