# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight RobotInterface wrapper for ACT policies.

This simplified version loads a .pt checkpoint (TorchScript or pickled Module)
and exposes a compatible `step(vla_step_data=...)` API used by
`experimental/get_action_policy.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental._types import VLAStepData

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is optional at runtime
    Image = None
JointArray = np.ndarray
ActionDict = Dict[str, JointArray]


class RobotInterface:
    """Minimal ACT policy wrapper.

    `checkpoint_dir` should point to a .pt file (TorchScript or pickled Module).
    The loaded object is expected to be callable or to expose `step`/`predict`.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        device: Optional[Union[str, torch.device]] = None,
        action_chunk_size: Optional[int] = None,
        action_joint_groups: Optional[list[str]] = None,
        server_address: str = "localhost:8964",
        startup_timeout: float = 120.0,
        adapters: Any | None = None,
        **_: Any,
    ) -> None:
        self.embodiment_tag = embodiment_tag
        self.device = torch.device(device) if device is not None else None
        self._client: RobotInterfaceClient | None = None
        self.model = None
        self._client = RobotInterfaceClient(
            server_address=server_address, startup_timeout=startup_timeout
        )

        self.action_chunk_size = action_chunk_size
        self.action_joint_groups = action_joint_groups
        self.adapters = adapters

    def reset(self) -> None:
        if self._client is not None:
            self._client.reset()

    @torch.no_grad()
    def step(
        self,
        vla_step_data: VLAStepData,
        max_new_tokens: int = 300,
        temperature: float = 0.0,
    ) -> tuple[ActionDict, bool, float | None]:
        if self._client is not None:
            return self._client.step(
                vla_step_data=vla_step_data,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        raise RuntimeError("RobotInterface is configured for server inference only.")

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.adapters is None or self.adapters.map_observation is None:
            raise RuntimeError(
                "RobotInterface requires adapters.map_observation for get_action()."
            )
        timings: dict[str, float] = {}
        overall_start = time.perf_counter() if _profile_enabled() else 0.0

        map_start = time.perf_counter() if _profile_enabled() else 0.0
        images, proprio = self.adapters.map_observation(observation)
        prev_obs = observation.get("__prev_observation")
        if isinstance(prev_obs, dict):
            # Thread previous-step proprio through to the server so UMI obs_horizon=2
            # can use consecutive frames at chunk boundaries under chunked replanning.
            _, prev_proprio = self.adapters.map_observation(prev_obs)
            for key, value in prev_proprio.items():
                proprio[f"__prev_{key}"] = value
        if _profile_enabled():
            timings["map_observation_ms"] = _ms(time.perf_counter() - map_start)

        prepare_start = time.perf_counter() if _profile_enabled() else 0.0
        task_description = observation.get("annotation.task")
        metadata: dict[str, Any] = {}
        rtc_metadata = observation.get("__rtc")
        if isinstance(rtc_metadata, dict):
            metadata["rtc"] = _make_arrays_contiguous(rtc_metadata)
        vla_step_data = self._prepare_vla_step_data(
            images,
            proprio,
            task_description,
            metadata=metadata,
        )
        if _profile_enabled():
            timings["prepare_vla_ms"] = _ms(time.perf_counter() - prepare_start)

        step_start = time.perf_counter() if _profile_enabled() else 0.0
        action_chunk = self.step(vla_step_data=vla_step_data)[0]
        if _profile_enabled():
            timings["policy_step_ms(model forward + network communication)"] = _ms(time.perf_counter() - step_start)

        if self.adapters.map_action is not None:
            map_action_start = time.perf_counter() if _profile_enabled() else 0.0
            action_chunk = self.adapters.map_action(action_chunk)
            if _profile_enabled():
                timings["map_action_ms"] = _ms(time.perf_counter() - map_action_start)

        chunk_start = time.perf_counter() if _profile_enabled() else 0.0
        action = _chunk_to_action_list(action_chunk)[0]
        if _profile_enabled():
            timings["chunk_to_action_ms"] = _ms(time.perf_counter() - chunk_start)

        info = {"action_chunk": action_chunk}
        if _profile_enabled():
            timings["total_ms"] = _ms(time.perf_counter() - overall_start)
            if self._client is not None and self._client.last_profile:
                timings["client_serialize_ms"] = self._client.last_profile.get(
                    "serialize_ms", 0.0
                )
                timings["client_rpc_ms"] = self._client.last_profile.get("rpc_ms", 0.0)
                timings["client_total_ms"] = self._client.last_profile.get(
                    "total_ms", 0.0
                )
            info["profiling_ms"] = timings
            print(f"[RobotInterface] get_action timings (ms): {timings}")
        return action, info

    def _prepare_vla_step_data(
        self,
        image: dict[str, Any],
        proprio: dict[str, Any],
        task_description: str | None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VLAStepData:
        reformatted_images = {
            k.replace("observation.images.", ""): v for k, v in image.items()
        }
        return VLAStepData(
            images=reformatted_images,
            states=proprio,
            actions={},
            text=task_description,
            embodiment=self.embodiment_tag,
            metadata=metadata or {},
        )


def _make_arrays_contiguous(obj: Any, memo: dict[int, Any] | None = None) -> Any:
    """Recursively ensure numpy arrays are contiguous for serialization."""
    if memo is None:
        memo = {}
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    if Image is not None and isinstance(obj, Image.Image):
        return np.ascontiguousarray(np.asarray(obj))
    if isinstance(obj, dict):
        new_obj: dict[Any, Any] = {}
        memo[obj_id] = new_obj
        for key, value in obj.items():
            new_obj[key] = _make_arrays_contiguous(value, memo)
        return new_obj
    if isinstance(obj, (list, tuple)):
        new_list: list[Any] = []
        memo[obj_id] = new_list
        for item in obj:
            new_list.append(_make_arrays_contiguous(item, memo))
        return new_list
    return obj


def _chunk_to_action_list(chunk: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    actions: list[dict[str, np.ndarray]] = []
    for key, value in chunk.items():
        # Skip non-action keys (e.g., future_image_predictions)
        if key == "future_image_predictions" or isinstance(value, dict):
            continue
        seq = np.asarray(value)
        if seq.ndim == 3:
            assert seq.shape[0] == 1, "Should only have batch size 1"
            seq = seq[0]
        # print("key", key, "shape", seq.shape)
        for t in range(seq.shape[0]):
            while t >= len(actions):
                actions.append(dict())
            actions[t][key] = seq[t]
    return actions


def _serialize_vla_step_data(vla_step_data: VLAStepData) -> dict[str, Any]:
    embodiment = vla_step_data.embodiment
    if isinstance(embodiment, EmbodimentTag):
        embodiment = embodiment.value
    return {
        "images": _make_arrays_contiguous(vla_step_data.images),
        "states": _make_arrays_contiguous(vla_step_data.states),
        "actions": _make_arrays_contiguous(vla_step_data.actions),
        "text": vla_step_data.text,
        "rl_info": _make_arrays_contiguous(vla_step_data.rl_info),
        "embodiment": embodiment,
        "is_demonstration": vla_step_data.is_demonstration,
        "metadata": vla_step_data.metadata,
    }


class RobotInterfaceClient:
    """Portal-based client that forwards RobotInterface.step to a server."""

    def __init__(
        self, server_address: str = "localhost:8964", startup_timeout: float = 120.0
    ):
        import portal

        self.server_address = server_address
        self._client: portal.Client | None = None
        self.last_profile: dict[str, float] = {}
        self._connect(startup_timeout)

    def _connect(self, timeout: float) -> None:
        import portal

        start = time.time()
        while time.time() - start < timeout:
            try:
                self._client = portal.Client(self.server_address)
                if self._client.health_check().result(timeout=5.0):
                    return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(
            f"RobotInterfaceClient could not connect to {self.server_address} after {timeout}s"
        )

    def reset(self) -> dict[str, Any] | None:
        future = self._client.reset()
        return future.result()

    def step(
        self,
        vla_step_data: VLAStepData,
        max_new_tokens: int = 300,
        temperature: float = 0.0,
    ) -> tuple[ActionDict, bool, float | None]:
        _ = max_new_tokens, temperature  # kept for API compatibility
        serialize_start = time.perf_counter() if _profile_enabled() else 0.0
        payload = _serialize_vla_step_data(vla_step_data)
        if _profile_enabled():
            serialize_ms = _ms(time.perf_counter() - serialize_start)

        rpc_start = time.perf_counter() if _profile_enabled() else 0.0
        future = self._client.step(payload)
        action_dict = future.result()
        if _profile_enabled():
            rpc_ms = _ms(time.perf_counter() - rpc_start)
            total_ms = serialize_ms + rpc_ms
            self.last_profile = {
                "serialize_ms": serialize_ms,
                "rpc_ms": rpc_ms,
                "total_ms": total_ms,
            }
            print(
                "[RobotInterfaceClient] step timings (ms): "
                f"serialize={serialize_ms:.2f}, rpc={rpc_ms:.2f}, total={total_ms:.2f}"
            )
        return action_dict, False, None

    def health_check(self) -> bool:
        return bool(self._client.health_check().result(timeout=5.0))


def _profile_enabled() -> bool:
    return os.getenv("ROBOT_INTERFACE_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ms(seconds: float) -> float:
    return seconds * 1000.0

