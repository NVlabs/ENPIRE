# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image
from enpire.env.forge.experimental._types import VLAStepData
from enpire.env.forge.experimental.robot_interface import RobotInterface


@dataclass
class PolicyAdapters:
    """Adapters for mapping between env <-> policy and URDF joint orders.

    Attributes:
        map_observation: Optional function to map environment observation to
            (images, proprio) used by the policy.
        map_action: Optional function to map policy-step action to environment action.
    """

    map_observation: None | Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]] = None
    map_action: None | Callable[[dict[str, Any]], dict[str, Any]] = None


def chunk_to_action_list(chunk: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    actions = []
    for key, value in chunk.items():
        # Skip non-action keys (e.g., future_image_predictions)
        if key == "future_image_predictions" or isinstance(value, dict):
            continue
        seq = np.asarray(value)
        # Handle both (H, D) and (B, H, D) shapes
        if seq.ndim == 3:
            assert seq.shape[0] == 1, "Should only have batch size 1"
            seq = seq[0]

        for t in range(seq.shape[0]):
            while t >= len(actions):
                actions.append(dict())
            actions[t][key] = seq[t]
    return actions


class GetActionPolicy:
    def __init__(
        self,
        policy: RobotInterface,
        adapters: PolicyAdapters,
    ):
        self.policy = policy
        # If the policy doesn't have a step method, you probably want a different wrapper
        assert hasattr(
            self.policy, "step"
        ), "Policy must have a step method to be wrapped by GetActionPolicy"
        self.adapters = adapters

    # Public API
    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Takes in an observation and produces an action and info dictionary."""

        if self.adapters.map_observation is None:
            raise RuntimeError("PolicyAdapters.map_observation must be provided for get_action().")
        images, proprio = self.adapters.map_observation(observation)

        task_description = observation["annotation.task"]
        vla_step_data = self._prepare_vla_step_data(images, proprio, task_description)
        response = self.policy.step(vla_step_data=vla_step_data)[0]
        
        # Extract attention data if present
        attention_data = response.get("attention", None)
        if attention_data is None:
            action_chunk = response
        else:
            action_chunk = response.copy()
            action_chunk.pop("attention")
            

        # Map to environment action space
        if self.adapters.map_action is not None:
            action_chunk = self.adapters.map_action(action_chunk)

        action = chunk_to_action_list(action_chunk)[0]

        info = {
            "action_chunk": action_chunk,
        }
        if attention_data is not None:
            info["attention"] = attention_data
            
        return action, info

    # Private API only from here on out
    def _prepare_vla_step_data(
        self, image: dict[str, Image.Image], proprio: dict[str, Any], task_description: str
    ) -> VLAStepData:
        reformatted_images = {k.replace("observation.images.", ""): v for k, v in image.items()}
        return VLAStepData(
            images=reformatted_images,
            states=proprio,
            actions={},
            text=task_description,
            embodiment=self.policy.embodiment_tag,
        )

