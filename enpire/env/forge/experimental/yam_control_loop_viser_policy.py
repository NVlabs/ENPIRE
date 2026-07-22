# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MuJoCo simulation environment for YAM bimanual robot station.

Mostly adapted from `yam_env.py` in the `xdof_samples` starter code.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_HUB_CACHE"] = "/mnt/amlfs-02/shared/ckpts"
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import socket
from typing import Any, Dict, List, Literal

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from PIL import Image
import tyro

from groot.control.envs.yam.cloud_upload_wrapper import CloudUploadWrapper
from groot.control.envs.yam.experimental.viser_policy import (
    ActionType,
    PolicyAdapters,
    ViserPolicy,
)
from groot.control.envs.yam.record_episode_wrapper import RecordEpisodeWrapper
from groot.vla.data.schema import EmbodimentTag
from groot.vla.omni.inference.robot_interface import RobotInterface
from groot.vla.omni.model.base.sim_policy import OmniDiffusionPolicy

PROPRIO_KEY_MAP_xdof_oss = {
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
}

ACTION_KEY_MAP_xdof_oss = {
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
}

CAMERA_KEY_MAP_xdof_oss = {
    "top_camera_image": "top",
    "left_camera_image": "left",
    "right_camera_image": "right",
}


PROPRIO_KEY_MAP_xdof = {
    "left_joint_pos": "joint_pos_obs_left",
    "left_gripper_pos": "gripper_pos_obs_left",
    "right_joint_pos": "joint_pos_obs_right",
    "right_gripper_pos": "gripper_pos_obs_right",
}

ACTION_KEY_MAP_xdof = {
    "joint_pos_action_left": "left_joint_pos",
    "gripper_pos_action_left": "left_gripper_pos",
    "joint_pos_action_right": "right_joint_pos",
    "gripper_pos_action_right": "right_gripper_pos",
}

CAMERA_KEY_MAP_xdof_240 = {
    "top_camera_image": "observation.images.top_camera-images-rgb_320_240",
    "left_camera_image": "observation.images.left_camera-images-rgb_320_240",
    "right_camera_image": "observation.images.right_camera-images-rgb_320_240",
}

CAMERA_KEY_MAP_xdof_480 = {
    "top_camera_image": "observation.images.top_camera-images-rgb",
    "left_camera_image": "observation.images.left_camera-images-rgb",
    "right_camera_image": "observation.images.right_camera-images-rgb",
}


@dataclass
class EvalConfig:
    ckpt_path: str
    use_real_robot: bool = False
    task_description: str = "Do something useful"
    use_vllm: bool = True
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF
    # We data collect at 30 but for now it is generally safer to eval if the policy is slower
    policy_control_freq: int = 10
    action_horizon: int = 50
    use_robot_interface: bool = False
    record_episode: bool = False
    cloud_upload: bool = False
    station: int | None = None
    operator: str | None = None
    relative_action: bool = False
    resolution: Literal[240, 480] = 240


class KeyReset:
    """Helper class to handle keyboard input for environment reset."""

    def __init__(self) -> None:
        """Initialize the key reset handler."""
        self.reset = False

    def key_callback(self, keycode: int) -> None:
        """Handle key press events.

        Args:
            keycode: The pressed key code.
        """
        if keycode == 32:  # <Space>
            self.reset = True


def map_observation(
    observation: Dict[str, Any],
    embodiment_tag: EmbodimentTag,
    resolution: Literal[240, 480],
):
    """
    mapping observation to image observation and
    """
    proprio = {}
    if embodiment_tag == EmbodimentTag.XDOF_OSS_DATA:
        PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof_oss
        CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_oss
    elif embodiment_tag == EmbodimentTag.XDOF:
        PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof
        if resolution == 240:
            CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_240
        elif resolution == 480:
            CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_480
        else:
            raise ValueError(f"Resolution {resolution} not supported")
    else:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported")

    for key, value in PROPRIO_KEY_MAP.items():
        proprio[value] = observation[key]
    images = {}
    for key, value in CAMERA_KEY_MAP.items():
        if resolution == 240:
            images[value] = Image.fromarray(observation[key]).resize((320, 240))
        elif resolution == 480:
            images[value] = Image.fromarray(observation[key])
        else:
            raise ValueError(f"Resolution {resolution} not supported")
    return images, proprio


def map_action(action: Dict[str, Any], embodiment_tag: EmbodimentTag):
    """
    mapping action to action_dict
    """
    action_dict = {}
    if embodiment_tag == EmbodimentTag.XDOF_OSS_DATA:
        ACTION_KEY_MAP = ACTION_KEY_MAP_xdof_oss
    elif embodiment_tag == EmbodimentTag.XDOF:
        ACTION_KEY_MAP = ACTION_KEY_MAP_xdof
    else:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported")
    for key, value in ACTION_KEY_MAP.items():
        action_dict[value] = action[key]
    return action_dict


def prompt_missing_fields(cfg: EvalConfig) -> EvalConfig:
    # Operator username
    if cfg.operator is None:
        cfg.operator = input("Enter operator username: ")

    # Station number
    if cfg.station is None:
        hostname = socket.gethostname()
        if hostname == "maxf-desktop":
            cfg.station = 1
        else:
            assert hostname.startswith("gear-yam-desktop-"), (
                "If not using gear-yam-desktop-*, station number must be specified."
            )
            cfg.station = int(hostname.split("-")[-1])

    return cfg


class WrappedPolicy:
    # Deprecated: kept for reference; superseded by ViserPolicy which wraps the policy and provides UI.
    def __init__(self, policy, action_exec_horizon=40):
        self.policy = policy
        self.action_exec_horizon = min(
            action_exec_horizon, self.policy.action_chunk_size
        )
        print("action_exec_horizon: ", self.action_exec_horizon)
        self.action_keys = self.policy.action_joint_groups
        self.action_queue = {
            key: deque([], maxlen=self.action_exec_horizon) for key in self.action_keys
        }
        self.last_action_chunk: Dict[str, Any] | None = None

    def step(
        self,
        image: Image.Image | List[Image.Image] | Dict[str, Image.Image],
        proprio: Dict[str, np.ndarray],
        task_description: str | None = None,
    ):
        action_dict = self.policy.step(
            image=image, proprio=proprio, task_description=task_description
        )[0]
        self.last_action_chunk = action_dict
        step_action = {}
        for key in self.action_keys:
            action_dim = action_dict[key].shape[-1]
            new_actions = deque(action_dict[key][0, : self.action_exec_horizon])
            self.action_queue[key].append(new_actions)
            actions_current_timestep = np.empty(
                (len(self.action_queue[key]), action_dim)
            )

            k = 0.05
            for i, q in enumerate(self.action_queue[key]):
                actions_current_timestep[i] = q.popleft()
            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()
            action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            step_action[key] = action
        return map_action(step_action)


def main(cfg: EvalConfig) -> None:
    """Run the environment with random actions in the MuJoCo viewer.

    This function demonstrates the environment by running it with the policy
    and allowing the user to reset with the spacebar.

    Args:
        ckpt_path: Path to the model checkpoint.
        use_real_robot: Whether to use the real robot (vs simulation).
        task_description: Task description for the policy.
        use_vllm: Whether to use VLLM for inference.
        policy_control_freq: Control frequency for the policy.
        action_horizon: Action horizon for the policy.
        use_robot_interface: Whether to use RobotInterface (vs OmniDiffusionPolicy).
        relative_action: Whether to use relative actions.
        embodiment_tag: Embodiment tag for the policy.
    """
    cfg = prompt_missing_fields(cfg)

    if cfg.use_real_robot:
        register(
            id="YamReal-v0",
            entry_point="groot.control.envs.yam.yam_real_env:YamRealEnv",
        )
        env = gym.make("YamReal-v0", policy_control_freq=cfg.policy_control_freq)
    else:
        register(
            id="YamSim-v0", entry_point="groot.control.envs.yam.yam_sim_env:YamSimEnv"
        )
        env = gym.make("YamSim-v0")
    embodiment_tag = cfg.embodiment_tag
    if cfg.record_episode:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path("data/eval") / f"{timestamp}-YAM-{cfg.station:02d}-eval"
        operator = (
            cfg.operator
            if cfg.operator is not None
            else input("Enter operator username: ")
        )
        env = RecordEpisodeWrapper(env, output_dir=str(output_dir), operator=operator)
        if cfg.cloud_upload:
            env = CloudUploadWrapper(env, output_dir, "GearYAMEvalDataV1")
    if cfg.use_robot_interface:
        robot_interface = RobotInterface(
            checkpoint_dir=cfg.ckpt_path,
            embodiment_tag=embodiment_tag,
            device="cuda:0",
            freq_bins=20,
            use_vllm=cfg.use_vllm,
        )
    else:
        robot_interface = OmniDiffusionPolicy(
            model_path=cfg.ckpt_path,
            embodiment_tag=embodiment_tag,
            device="cuda:0",
        )

    initial_state = {
        "left_joint_pos": np.zeros(6),
        "left_gripper_pos": np.ones(1),
        "right_joint_pos": np.zeros(6),
        "right_gripper_pos": np.ones(1),
    }
    observation, info = env.reset(
        options=dict(initial_state=initial_state, start_new_episode=False)
    )
    # MuJoCo-specific Viser removed; ViserPolicy will own the UI and overlays
    # Shared Viser UI + policy/IK controller (URDF is the single source of truth)
    from enpire.env.forge.robot.models.station.paths import get_station_urdf

    urdf_path = get_station_urdf()
    adapters = PolicyAdapters(
        # If env and URDF joint orders differ, provide explicit ordering here
        map_observation=lambda _observation: map_observation(
            _observation, embodiment_tag, cfg.resolution
        ),
        map_action=lambda _action: map_action(_action, embodiment_tag),
    )
    if cfg.relative_action:
        action_type = ActionType.RELATIVE
    else:
        action_type = ActionType.ABSOLUTE

    viser_policy = ViserPolicy(
        policy=robot_interface,
        urdf_path=urdf_path,
        action_exec_horizon=cfg.action_horizon,
        adapters=adapters,
        action_type=action_type,
        record_episode=cfg.record_episode,
        video_enabled=True,
        video_fps=cfg.policy_control_freq,
        video_realtime=True,
    )
    viser_policy.cmd_input.value = cfg.task_description
    viser_policy.run_eval = False

    reset = KeyReset()  # press <Space> to reset the environment

    # with mujoco.viewer.launch_passive(
    #     env._model, env._data, key_callback=reset.key_callback
    # ) as viewer:
    #     while viewer.is_running():
    while True:
        # if task description is changed, update the task description
        task_description = viser_policy.task_command
        # Active step: get action from policy (handles IK/policy/exec mode/step_once)
        action = viser_policy.control_step_from_observation(
            observation=observation,
            default_task=task_description,
        )
        if action == "Homing":
            print("[INFO] Homing...")
            observation, info = env.reset(
                options=dict(
                    initial_state=initial_state,
                    start_new_episode=False,
                    force_reset=True,
                )
            )
            continue
        elif action == "Homing and Discard Recording":
            print("[INFO] Homing and discarding recording...")
            observation, info = env.reset(
                options=dict(
                    initial_state=initial_state,
                    start_new_episode=False,
                    force_reset=True,
                    discard_episode=True,
                )
            )
            continue
        elif action == "Stop Recording":
            observation, info = env.reset(options={"start_new_episode": False})
            continue
        elif action == "Start Recording":
            print(f"[INFO] Recording started for task: {task_description}")
            observation, info = env.reset(options={"task_name": task_description})
            continue
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated or reset.reset:
            observation, info = env.reset(options=dict(initial_state=initial_state))
            if hasattr(robot_interface, "reset"):
                robot_interface.reset()
            reset.reset = False

    env.close()


if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
