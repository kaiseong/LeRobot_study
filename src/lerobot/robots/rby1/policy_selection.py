#!/usr/bin/env python

from __future__ import annotations

from copy import deepcopy
from typing import Any

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .schema import FULL_ACTION_KEYS, validate_rby1_action_keys


def resolve_joint_subset(selected_joint_names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not selected_joint_names:
        return FULL_ACTION_KEYS
    return validate_rby1_action_keys(tuple(selected_joint_names))


def build_policy_features_for_joint_subset(
    *,
    all_features: dict[str, PolicyFeature],
    selected_joint_names: tuple[str, ...],
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    input_features = {key: feature for key, feature in all_features.items() if feature.type is not FeatureType.ACTION}
    output_features = {key: feature for key, feature in all_features.items() if feature.type is FeatureType.ACTION}

    if OBS_STATE in input_features:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(len(selected_joint_names),))
    if ACTION in output_features:
        output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(len(selected_joint_names),))
    return input_features, output_features


def slice_stats_for_joint_subset(
    stats: dict[str, dict[str, Any]] | None,
    *,
    selected_joint_names: tuple[str, ...],
    observation_state_names: list[str] | tuple[str, ...],
    action_names: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if not stats:
        return {}

    selected_stats = deepcopy(stats)
    state_indices = _resolve_indices(observation_state_names, selected_joint_names, label="observation.state")
    action_indices = _resolve_indices(action_names, selected_joint_names, label="action")

    if OBS_STATE in selected_stats:
        selected_stats[OBS_STATE] = {
            stat_name: _slice_stat_value(stat_value, state_indices)
            for stat_name, stat_value in selected_stats[OBS_STATE].items()
        }
    if ACTION in selected_stats:
        selected_stats[ACTION] = {
            stat_name: _slice_stat_value(stat_value, action_indices)
            for stat_name, stat_value in selected_stats[ACTION].items()
        }
    return selected_stats


def _resolve_indices(
    available_names: list[str] | tuple[str, ...],
    selected_names: tuple[str, ...],
    *,
    label: str,
) -> list[int]:
    name_to_index = {name: index for index, name in enumerate(available_names)}
    missing = [name for name in selected_names if name not in name_to_index]
    if missing:
        raise ValueError(f"Selected {label} keys are not available: {missing}")
    return [name_to_index[name] for name in selected_names]


def _slice_stat_value(value: Any, indices: list[int]) -> Any:
    if isinstance(value, list):
        return [value[index] for index in indices]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    if hasattr(value, "__getitem__"):
        return value[indices]
    return value
