# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import hydra
import numpy as np
import portal
import torch
import yaml
from data4robotics.transforms import get_transform_by_name
from normalizer import MeanStdNormalize


def _make_arrays_contiguous(obj: Any, memo: dict[int, Any] | None = None) -> Any:
    """Recursively ensure numpy arrays are contiguous for Portal serialization."""
    if memo is None:
        memo = {}
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
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


@dataclass
class ServerConfig:
    ckpt_path: str
    ckpt_name: str
    port: int
    device: str
    left_joint_dim: int
    right_joint_dim: int
    gripper_dim: int


class PolicyServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._load_policy()
        self._setup_server()

    def _load_policy(self) -> None:
        print(f"[PolicyServer] Loading policy from {self.config.ckpt_path}")
        ckpt_path = self.config.ckpt_path
        with open(os.path.join(ckpt_path, "agent_config.yaml"), "r") as f:
            agent_config = yaml.safe_load(f.read())
        with open(os.path.join(ckpt_path, "obs_config.yaml"), "r") as f:
            obs_config = yaml.safe_load(f.read())
        self.image_keys = obs_config.get("img", [])
        transform_cfg = obs_config.get("transform")
        if isinstance(transform_cfg, dict) and "_target_" in transform_cfg:
            self.transform = hydra.utils.instantiate(transform_cfg)
        else:
            self.transform = get_transform_by_name(transform_cfg or "preproc")

        self.agent = hydra.utils.instantiate(agent_config)
        ckpt_name = self.config.ckpt_name
        if not ckpt_name:
            checkpoint_files = [
                name for name in os.listdir(ckpt_path) if name.endswith(".ckpt")
            ]
            if not checkpoint_files:
                raise FileNotFoundError(f"No .ckpt files found in {ckpt_path}")
            ckpt_name = checkpoint_files[0]
        self._ckpt_name = ckpt_name
        checkpoint_path = os.path.join(ckpt_path, ckpt_name)
        save_dict = torch.load(checkpoint_path, map_location="cpu")
        self.agent.load_state_dict(save_dict["model"])
        self.agent = self.agent.eval().to(self.config.device)
        self._load_action_normalizer()

    def _load_action_normalizer(self) -> None:
        norm_path = os.path.join(self.config.ckpt_path, "norm_stats.json")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Missing norm stats at {norm_path}")
        self.action_normalizer = MeanStdNormalize()
        self.action_normalizer.load(self.config.ckpt_path)

    def _setup_server(self) -> None:
        self._server = portal.Server(self.config.port)
        self._server.bind("step", self._handle_step)
        self._server.bind("reset", self._handle_reset)
        self._server.bind("health_check", self._handle_health_check)
        self._server.bind("get_info", self._handle_get_info)

    def _handle_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        images = payload.get("images", {})
        states = payload.get("states", {})
        imgs = self._prepare_images(images)
        obs = self._prepare_state(states)
        with torch.no_grad():
            if str(self.config.device).startswith("cuda"):
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            actions = self.agent.get_actions(imgs, obs)
            if str(self.config.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"[PolicyServer] Inference time: {elapsed_ms:.2f} ms")
            actions = self._unnormalize_actions(actions)
        action_dict = self._split_actions(actions)
        return _make_arrays_contiguous(action_dict)

    def _handle_reset(self) -> dict[str, Any] | None:
        if hasattr(self.agent, "reset"):
            return self.agent.reset()
        return None

    def _handle_health_check(self) -> bool:
        return True

    def _handle_get_info(self) -> dict[str, Any]:
        """Return metadata about the loaded policy checkpoint."""
        import re

        ckpt_path = self.config.ckpt_path
        ckpt_name = getattr(self, "_ckpt_name", "")
        policy_name = os.path.basename(ckpt_path.rstrip("/"))
        # Extract step number from ckpt filename (e.g. "..._step024000.ckpt")
        step = 0
        m = re.search(r"step(\d+)", ckpt_name)
        if m:
            step = int(m.group(1))
        return {
            "ckpt_path": ckpt_path,
            "ckpt_name": ckpt_name,
            "policy_name": policy_name,
            "step": step,
        }

    def serve(self) -> None:
        print(f"[PolicyServer] Starting server on port {self.config.port}")
        self._server.start()

    def _prepare_images(self, images: dict[str, Any]) -> dict[str, torch.Tensor]:

        # print("image keys", images.keys())
        if not self.image_keys:
            image_keys = list(images.keys())
        else:
            image_keys = self.image_keys
        device = torch.device(self.config.device)
        output: dict[str, torch.Tensor] = {}

        print("image keys", image_keys)
        for idx, key in enumerate(image_keys):
            value = self._get_image_value(images, key)
            if value is None:
                raise KeyError(f"Missing image key {key} in payload")
            tensor = self._image_to_tensor(value)
            if tensor.ndim == 3:
                tensor = self.transform(tensor)
                tensor = tensor.unsqueeze(0)
            else:
                frames = [self.transform(frame) for frame in tensor]
                tensor = torch.stack(frames, dim=0).unsqueeze(0)
            output[f"cam{idx}"] = tensor.to(device)
        return output

    def _get_image_value(self, images: dict[str, Any], key: str) -> Any | None:
        candidates = [key]
        if "-" in key:
            candidates.extend(
                [
                    key.replace("-", "_"),
                    key.replace("-", "/"),
                    key.replace("-", "."),
                ]
            )
        if "_" in key:
            candidates.append(key.replace("_", "-"))
        for candidate in candidates:
            if candidate in images:
                return images[candidate]
            dotted = f"observation.images.{candidate}"
            if dotted in images:
                return images[dotted]
        observation = images.get("observation")
        if isinstance(observation, dict):
            obs_images = observation.get("images")
            if isinstance(obs_images, dict):
                for candidate in candidates:
                    if candidate in obs_images:
                        return obs_images[candidate]
        return None

    def _image_to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, list):
            frames = [self._image_to_tensor(frame) for frame in value]
            return torch.stack(frames, dim=0)
        if hasattr(value, "convert"):
            value = np.asarray(value)
        array = np.asarray(value)
        if array.ndim == 4:
            frames = [self._image_to_tensor(frame) for frame in array]
            return torch.stack(frames, dim=0)
        if array.ndim != 3:
            raise ValueError(f"Expected image with shape (H, W, C), got {array.shape}")
        if array.shape[2] < 3:
            raise ValueError(f"Expected image with 3 channels, got {array.shape}")
        array = array[..., :3]
        array = self._convert_to_rgb(array)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return torch.from_numpy(array).float().permute((2, 0, 1)) / 255.0

    def _convert_to_rgb(self, bgr_image: np.ndarray) -> np.ndarray:
        """Match training-time conversion: BGR -> RGB."""
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def _prepare_state(self, states: Any) -> torch.Tensor:
        if isinstance(states, np.ndarray):
            flat = states.reshape(-1)
        elif isinstance(states, list):
            flat = np.asarray(states).reshape(-1)
        elif isinstance(states, dict):
            if "state" in states:
                flat = np.asarray(states["state"]).reshape(-1)
            elif "observation.state" in states:
                flat = np.asarray(states["observation.state"]).reshape(-1)
            elif (
                "observation" in states
                and isinstance(states["observation"], dict)
                and "state" in states["observation"]
            ):
                flat = np.asarray(states["observation"]["state"]).reshape(-1)
            else:
                preferred_orders = [
                    [
                        "joint_pos_obs_left",
                        "gripper_pos_obs_left",
                        "joint_pos_obs_right",
                        "gripper_pos_obs_right",
                    ],
                    [
                        "left_joint_pos",
                        "left_gripper_pos",
                        "right_joint_pos",
                        "right_gripper_pos",
                    ],
                ]
                flat = None
                for order in preferred_orders:
                    if all(key in states for key in order):
                        parts = [np.asarray(states[key]).reshape(-1) for key in order]
                        flat = np.concatenate(parts, axis=0)
                        break
                if flat is None:
                    parts = [np.asarray(value).reshape(-1) for value in states.values()]
                    flat = np.concatenate(parts, axis=0)
        else:
            raise TypeError("Unsupported state format")
        return torch.from_numpy(flat[None].astype(np.float32)).to(self.config.device)

    def _unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_np = actions.detach().cpu().numpy()
        if action_np.ndim == 1:
            action_np = action_np[None]
        unnorm = self.action_normalizer.unnormalize(action_np)
        return torch.from_numpy(unnorm).to(device=actions.device, dtype=actions.dtype)

    def _split_actions(self, actions: torch.Tensor) -> dict[str, np.ndarray]:
        action_np = actions.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[None]
        left_dim = self.config.left_joint_dim
        right_dim = self.config.right_joint_dim
        grip_dim = self.config.gripper_dim
        total = left_dim + grip_dim + right_dim + grip_dim
        if action_np.shape[-1] < total:
            raise ValueError(
                f"Action dim {action_np.shape[-1]} smaller than expected {total}"
            )
        if action_np.shape[-1] > total:
            action_np = action_np[..., :total]
        left = action_np[..., :left_dim].squeeze(0)
        left_grip = action_np[..., left_dim : left_dim + grip_dim].squeeze(0)
        right = action_np[
            ..., left_dim + grip_dim : left_dim + grip_dim + right_dim
        ].squeeze(0)
        right_grip = action_np[..., left_dim + grip_dim + right_dim : total].squeeze(0)
        # print("left joint shape", left.shape)
        return {
            "joint_pos_action_left": left,
            "gripper_pos_action_left": left_grip,
            "joint_pos_action_right": right,
            "gripper_pos_action_right": right_grip,
        }


def _run_server(config: ServerConfig) -> None:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["HF_HUB_OFFLINE"] = "1"
    server = PolicyServer(config)
    server.serve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch policy server.")
    parser.add_argument("--ckpt-path", required=True, help="Path to ACT checkpoint directory.")
    parser.add_argument("--ckpt-name", default="", help="Checkpoint filename (e.g. 'policy_step024000.ckpt'). Auto-detected if omitted.")
    parser.add_argument("--port", type=int, default=8964, help="Portal server port.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for policy.")
    parser.add_argument("--left-joint-dim", type=int, default=6, help="Left arm DOF.")
    parser.add_argument("--right-joint-dim", type=int, default=6, help="Right arm DOF.")
    parser.add_argument("--gripper-dim", type=int, default=1, help="Gripper DOF.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ServerConfig(
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        port=args.port,
        device=args.device,
        left_joint_dim=args.left_joint_dim,
        right_joint_dim=args.right_joint_dim,
        gripper_dim=args.gripper_dim,
    )

    _run_server(config)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
