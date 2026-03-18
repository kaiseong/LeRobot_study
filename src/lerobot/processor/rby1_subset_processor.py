#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .core import PolicyAction, RobotAction, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@dataclass
@ProcessorStepRegistry.register("rby1_joint_subset_processor")
class RBY1JointSubsetProcessorStep(ProcessorStep):
    observation_state_names: list[str]
    action_names: list[str]
    selected_state_names: list[str]
    selected_action_names: list[str]

    _state_indices: list[int] = field(init=False, repr=False)
    _action_indices: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._state_indices = self._resolve_indices(
            self.observation_state_names,
            self.selected_state_names,
            label="observation.state",
        )
        self._action_indices = self._resolve_indices(
            self.action_names,
            self.selected_action_names,
            label="action",
        )

    def __call__(self, transition):
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            observation = dict(observation)
            observation[OBS_STATE] = self._slice_tensor(observation[OBS_STATE], self._state_indices)
            new_transition[TransitionKey.OBSERVATION] = observation

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, torch.Tensor):
            new_transition[TransitionKey.ACTION] = self._slice_tensor(action, self._action_indices)

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "observation_state_names": list(self.observation_state_names),
            "action_names": list(self.action_names),
            "selected_state_names": list(self.selected_state_names),
            "selected_action_names": list(self.selected_action_names),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {feature_type: dict(values) for feature_type, values in features.items()}
        if OBS_STATE in transformed[PipelineFeatureType.OBSERVATION]:
            transformed[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(len(self.selected_state_names),),
            )
        if ACTION in transformed[PipelineFeatureType.ACTION]:
            transformed[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(len(self.selected_action_names),),
            )
        return transformed

    @staticmethod
    def _slice_tensor(value: Any, indices: list[int]) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        index_tensor = torch.tensor(indices, device=value.device)
        return torch.index_select(value, dim=-1, index=index_tensor)

    @staticmethod
    def _resolve_indices(
        available_names: list[str],
        selected_names: list[str],
        *,
        label: str,
    ) -> list[int]:
        name_to_index = {name: index for index, name in enumerate(available_names)}
        missing = [name for name in selected_names if name not in name_to_index]
        if missing:
            raise ValueError(f"Selected {label} names are not available: {missing}")
        return [name_to_index[name] for name in selected_names]


@dataclass
@ProcessorStepRegistry.register("rby1_expand_robot_action_subset_processor")
class RBY1ExpandRobotActionSubsetProcessorStep(RobotActionProcessorStep):
    full_action_names: list[str]
    fixed_action_values: dict[str, float] = field(default_factory=dict)
    hold_from_observation: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        unknown_keys = sorted(set(action) - set(self.full_action_names))
        if unknown_keys:
            raise KeyError(f"Unknown RBY1 action keys received by the robot action processor: {unknown_keys}")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        observation = observation if isinstance(observation, dict) else {}

        expanded_action: RobotAction = {}
        for key in self.full_action_names:
            if key in action:
                expanded_action[key] = float(action[key])
            elif key in self.fixed_action_values:
                expanded_action[key] = float(self.fixed_action_values[key])
            elif self.hold_from_observation and key in observation:
                expanded_action[key] = float(observation[key])
            else:
                raise KeyError(
                    "RBY1 subset expansion needs either an explicit action, a fixed value, or an observation "
                    f"hold value for '{key}'."
                )
        return expanded_action

    def get_config(self) -> dict[str, Any]:
        return asdict(self)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {feature_type: dict(values) for feature_type, values in features.items()}
        transformed[PipelineFeatureType.ACTION] = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for key in self.full_action_names
        }
        return transformed


def insert_rby1_joint_subset_step(processor, step: RBY1JointSubsetProcessorStep):
    for processor_step in getattr(processor, "steps", []):
        if isinstance(processor_step, RBY1JointSubsetProcessorStep):
            return processor
    insertion_index = len(processor.steps)
    for index, processor_step in enumerate(processor.steps):
        registry_name = getattr(processor_step, "_registry_name", "")
        if registry_name in {"normalizer_processor", "device_processor"}:
            insertion_index = index
            break
    processor.steps = list(processor.steps[:insertion_index]) + [step] + list(processor.steps[insertion_index:])
    return processor


def get_rby1_selected_action_names(processor, fallback_action_names: list[str] | tuple[str, ...]) -> list[str]:
    for step in getattr(processor, "steps", []):
        if isinstance(step, RBY1JointSubsetProcessorStep):
            return list(step.selected_action_names)
    return list(fallback_action_names)
