#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

TORSO_JOINT_NAMES = tuple(f"torso_{idx}" for idx in range(6))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_{idx}" for idx in range(7))
LEFT_ARM_JOINT_NAMES = tuple(f"left_arm_{idx}" for idx in range(7))
HEAD_JOINT_NAMES = ("head_0", "head_1")

BODY_JOINT_NAMES = TORSO_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES + LEFT_ARM_JOINT_NAMES + HEAD_JOINT_NAMES
BODY_ACTION_KEYS = BODY_JOINT_NAMES

RIGHT_GRIPPER_ACTION_KEY = "right_gripper"
LEFT_GRIPPER_ACTION_KEY = "left_gripper"
GRIPPER_ACTION_KEYS = (RIGHT_GRIPPER_ACTION_KEY, LEFT_GRIPPER_ACTION_KEY)

FULL_ACTION_KEYS = BODY_ACTION_KEYS + GRIPPER_ACTION_KEYS
FULL_OBSERVATION_STATE_KEYS = FULL_ACTION_KEYS

RBY1_M_JOINT_ORDER = (
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

TRIGGER_MAX_VALUE = 1000.0

DEFAULT_READY_TORSO_POSITIONS = tuple(np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]).tolist())
DEFAULT_READY_RIGHT_ARM_POSITIONS = tuple(np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]).tolist())
DEFAULT_READY_LEFT_ARM_POSITIONS = tuple(np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]).tolist())
DEFAULT_READY_HEAD_POSITIONS = (0.0, 0.0)


def validate_rby1_action_keys(action_keys: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(action_keys) - set(FULL_ACTION_KEYS))
    if unknown:
        raise ValueError(f"Unsupported RBY1 action keys: {unknown}")
    if len(set(action_keys)) != len(tuple(action_keys)):
        raise ValueError("RBY1 action key selections must not contain duplicates.")
    return tuple(action_keys)


def coerce_fixed_action_values(
    torso_positions: Sequence[float] | None = None,
    head_positions: Sequence[float] | None = None,
) -> dict[str, float]:
    fixed_action_values: dict[str, float] = {}
    if torso_positions is not None:
        if len(torso_positions) != len(TORSO_JOINT_NAMES):
            raise ValueError("RBY1 torso fixed positions must contain exactly 6 values.")
        for key, value in zip(TORSO_JOINT_NAMES, torso_positions, strict=True):
            fixed_action_values[key] = float(value)
    if head_positions is not None:
        if len(head_positions) != len(HEAD_JOINT_NAMES):
            raise ValueError("RBY1 head fixed positions must contain exactly 2 values.")
        for key, value in zip(HEAD_JOINT_NAMES, head_positions, strict=True):
            fixed_action_values[key] = float(value)
    return fixed_action_values


def build_action_features(action_keys: Iterable[str]) -> dict[str, type]:
    return dict.fromkeys(tuple(action_keys), float)


def normalize_trigger_value(trigger_value: int | float, maximum_value: float = TRIGGER_MAX_VALUE) -> float:
    if maximum_value <= 0:
        raise ValueError("maximum_value must be positive.")
    normalized = float(trigger_value) / float(maximum_value)
    return max(0.0, min(1.0, normalized))
