# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim environment for YAM bimanual robot station."""

import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from enpire.env.forge.robot.yam._base_yam_env import _BaseYamEnv
from enpire.env.forge.tools.data_collection.async_video_compression import (
    compress_image,
)


class YamSimEnv(_BaseYamEnv):
    GRIPPER_CTRL_SCALE = 0.041
    GRIPPER_QPOS_SCALE = 0.0376

    def __init__(
        self,
        control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        ] = "joint_position",
        policy_control_freq: float = 30.0,
        enable_cameras: bool = True,
        enabled_camera_names: tuple[str, ...] = ("top", "left", "right"),
        enabled_sides: (Literal["left", "right", "both"] | tuple[str, ...] | None) = None,
        crop_camera_names: tuple[str, ...] = (),
        crop_region: tuple[str, ...] = ("center",),
        delta_ee_translation_xyz_max: tuple[float, float, float] = (
            0.0003,
            0.0003,
            0.0006,
        ),
        real_time: bool = True,
    ):
        if control_mode is None:
            raise ValueError("Control mode must be specified")
        self.enabled_camera_names = tuple(str(x) for x in (enabled_camera_names))
        super().__init__(
            control_mode=control_mode,
            enable_cameras=enable_cameras,
            enabled_camera_names=self.enabled_camera_names,
            enabled_sides=enabled_sides,
            crop_camera_names=crop_camera_names,
            crop_region=crop_region,
            delta_ee_translation_xyz_max=delta_ee_translation_xyz_max,
        )
        self.policy_control_freq = policy_control_freq
        # Offline learning can omit wall-clock pacing; physics substeps and
        # actuator integration remain identical to the default paced mode.
        self.real_time = real_time
        self.control_period = 1.0 / policy_control_freq
        self.last_step_time = time.time()

        self._data = mujoco.MjData(self._model)
        self._n_substeps = int(self.control_period // self._model.opt.timestep)
        self._renderer: Optional[mujoco.Renderer] = None

        # Joint mappings
        self.left_joint_names = [x.name for x in self._spec.joints if x.name.startswith("left_")]
        self.right_joint_names = [x.name for x in self._spec.joints if x.name.startswith("right_")]
        self.left_joint_ids = np.array(
            [self._model.joint(name).id for name in self.left_joint_names]
        )
        self.right_joint_ids = np.array(
            [self._model.joint(name).id for name in self.right_joint_names]
        )

        # Actuator mappings
        self.actuator_ids = np.concatenate([self.left_actuator_ids, self.right_actuator_ids])

        # Camera mappings
        self.camera_ids = {}
        if self.enable_cameras:
            for camera_name in self.enabled_camera_names:
                self.camera_ids[camera_name] = self._resolve_model_camera_name(camera_name)

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Convert from delta EE pose to absolute cartesian, then to joint
        if self._is_delta_ee_mode:
            action = self._convert_from_delta_ee_to_cartesian(action)
            action = self._convert_from_cartesian_to_joint(action)
        # Convert from Cartesian space to absolute joint positions
        elif self.control_mode == "cartesian_position":
            action = self._convert_from_cartesian_to_joint(action)
        # Convert from delta joint space to absolute joint positions
        elif self.control_mode == "delta_joint_position":
            action = self._convert_from_delta_joint_to_absolute(action)
        elif self.control_mode == "joint_position":
            pass
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        if self._is_delta_mode:
            self._record_gripper_cmd(
                action
            )  # Record initial and each steps gripper commands as internal state, so in delta control mode we can hold grippers

        # Set control inputs
        ctrl = np.concatenate(
            [
                action["left_joint_pos"],
                action["left_gripper_pos"] * self.GRIPPER_CTRL_SCALE,
                action["right_joint_pos"],
                action["right_gripper_pos"] * self.GRIPPER_CTRL_SCALE,
            ]
        )
        self._data.ctrl[self.actuator_ids] = ctrl

        # Advance the simulation
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # Maintain desired control freq
        sleep_end_time = self.last_step_time + self.control_period
        if self.real_time and time.time() > sleep_end_time:  # TODO: Resolve latency issues
            print(
                f"\rWarning: Control loop timing overrun (budget {self.control_period * 1000:.1f} ms, actual {1000 * (time.time() - self.last_step_time):.1f} ms)    ",
                end="",
                flush=True,
            )
        while self.real_time and time.time() < sleep_end_time:
            time.sleep(0.0001)
        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)

        # Update current joint positions for delta control modes
        if self._is_delta_mode:
            self._update_current_joint_pos_from_observation(obs)

        reward = self._compute_reward()
        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        # Expose the post-IK joint action for delta_ee_pose debugging.
        # Format: 14D [left_jp(6), left_grip(1), right_jp(6), right_grip(1)]
        if self._is_delta_ee_mode:
            info["_debug_joint_action"] = np.concatenate(
                [
                    action["left_joint_pos"],
                    action["left_gripper_pos"],
                    action["right_joint_pos"],
                    action["right_gripper_pos"],
                ]
            ).astype(np.float64)

        return obs, reward, False, False, info

    def reset(
        self, seed: int | None = None, options: str | Mapping[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        print("Resetting the environment...")
        target_state = self._resolve_reset_target_state(options)

        # Reset the simulation
        mujoco.mj_resetData(self._model, self._data)

        for side, joint_ids, actuator_ids in [
            ("left", self.left_joint_ids, self.left_actuator_ids),
            ("right", self.right_joint_ids, self.right_actuator_ids),
        ]:
            # Joint positions
            self._data.qpos[joint_ids[:6]] = target_state[f"{side}_joint_pos"]
            self._data.qpos[joint_ids[6:8]] = target_state[f"{side}_gripper_pos"] * np.array(
                [self.GRIPPER_QPOS_SCALE, -self.GRIPPER_QPOS_SCALE]
            )
            # Control inputs
            self._data.ctrl[actuator_ids[:6]] = target_state[f"{side}_joint_pos"]
            self._data.ctrl[actuator_ids[6:7]] = (
                target_state[f"{side}_gripper_pos"] * self.GRIPPER_CTRL_SCALE
            )

        self._record_gripper_cmd(target_state)

        # Advance the simulation
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)
        # Initialize current joint positions for delta control tracking
        if self._is_delta_mode:
            self._update_current_joint_pos_from_observation(obs)
            self._seed_delta_ee_command(obs)

        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        return obs, info

    def _read_current_joint_state(self) -> dict[str, np.ndarray]:
        current: dict[str, np.ndarray] = {}
        for side, joint_ids in (
            ("left", self.left_joint_ids),
            ("right", self.right_joint_ids),
        ):
            qpos = np.asarray(self._data.qpos[joint_ids], dtype=np.float32)
            current[f"{side}_joint_pos"] = qpos[:6].copy()
            current[f"{side}_gripper_pos"] = (
                np.abs(qpos[6:]).mean(keepdims=True) / self.GRIPPER_QPOS_SCALE
            ).copy()
        return current

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {}
        _ts: dict[str, float] = {}

        # Joint and gripper states
        for side, joint_ids in [
            ("left", self.left_joint_ids),
            ("right", self.right_joint_ids),
        ]:
            qpos = self._data.qpos[joint_ids]
            qpos_arm = qpos[:6]
            qpos_gripper = np.abs(qpos[6:]).mean(keepdims=True) / self.GRIPPER_QPOS_SCALE

            obs[f"{side}_joint_pos"] = qpos_arm
            obs[f"{side}_gripper_pos"] = qpos_gripper
            _ts[f"{side}_state"] = time.time()

        # Update current joint positions for delta control tracking
        self._update_current_joint_pos_from_observation(obs)

        # Convert from joint space to Cartesian space when policy proprioception is EE pose.
        if self._use_cartesian_proprioception:
            obs = self._convert_from_joint_to_cartesian(obs)

        # Camera images
        for camera_name in self.enabled_camera_names:
            if self.enable_cameras:
                image = self._render(camera_name)
                image = self._crop_camera_image(camera_name, image)
                obs[f"{camera_name}_camera_image"] = compress_image(image)
            else:
                # Preserve the declared observation-space dimensions when no
                # renderer is created. Downstream wrappers may resize later.
                obs[f"{camera_name}_camera_image"] = self._blank_camera_image()
            _ts[f"{camera_name}_camera"] = time.time()

        # Convert float64 to float32
        obs = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in obs.items()}
        # Attach per-component creation timestamps
        obs["__timestamps"] = _ts

        return obs

    def _render(self, camera_name: str) -> np.ndarray:
        if not self.enable_cameras:
            return self._blank_camera_image()
        camera_id = self.camera_ids[camera_name]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, self.CAMERA_HEIGHT, self.CAMERA_WIDTH)
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self._data, camera=camera_id)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    def _compute_reward(self) -> float:
        return 0.0

    def _get_info(self) -> dict[str, Any]:
        return {}


class MujocoViewerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.viewer = mujoco.viewer.launch_passive(env.unwrapped._model, env.unwrapped._data)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update viewer
        if not self.viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed")
        self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()


class SimDummyPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self) -> None:
        return None

    def get_action(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Simple sinusoidal action for testing
        t = time.time()
        frequency = 0.5
        osc = np.sin(2 * np.pi * frequency * t)

        def scale(key, low=None, high=None, factor=1.0, fallback=None):
            if low is None:
                low = self.action_space[key].low
            if high is None:
                high = self.action_space[key].high
            low = np.asarray(low, dtype=np.float32)
            high = np.asarray(high, dtype=np.float32)
            if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
                if fallback is None:
                    return np.zeros(self.action_space[key].shape, dtype=np.float32)
                return np.asarray(fallback, dtype=np.float32).copy()
            mid = (low + high) / 2
            amp = factor * (high - low) / 2
            return (mid + amp * osc).astype(np.float32)

        action = {
            "left_gripper_pos": scale("left_gripper_pos"),
            "right_gripper_pos": scale("right_gripper_pos"),
        }
        action_spaces = self.action_space.spaces

        if "left_joint_pos" in action_spaces:
            action["left_joint_pos"] = scale(
                "left_joint_pos", factor=0.1, fallback=np.zeros(6, dtype=np.float32)
            )
            action["right_joint_pos"] = scale(
                "right_joint_pos", factor=0.1, fallback=np.zeros(6, dtype=np.float32)
            )

        elif "left_ee_pos" in action_spaces:
            identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            for side in ("left", "right"):
                pos_key = f"{side}_ee_pos"
                rot6d_key = f"{side}_ee_rot6d"
                if pos_key in obs and rot6d_key in obs:
                    action[pos_key] = np.asarray(obs[pos_key], dtype=np.float32).copy()
                    action[rot6d_key] = np.asarray(obs[rot6d_key], dtype=np.float32).copy()
                else:
                    action[pos_key] = np.zeros(3, dtype=np.float32)
                    action[rot6d_key] = identity_rot6d.copy()

        return action, {}


@dataclass
class SimConfig:
    """Configuration for YamSim environment."""

    control_mode: Literal[
        "joint_position",
        "cartesian_position",
        "delta_joint_position",
        "delta_ee_pose",
        "delta_ee_pose_translation",
    ] = "joint_position"
    """Control mode for the robot."""
    policy_control_freq: float = 30.0
    """Policy control frequency in Hz."""
    num_steps: int = 1000
    """Number of steps to run the simulation."""
    headless: bool = False
    """Run without visualization (for headless servers)."""


def main(config: SimConfig):
    from gymnasium.envs.registration import register

    # Environment
    register(id="YamSim-v0", entry_point="enpire.env.forge.robot.yam.yam_sim_env:YamSimEnv")
    env = gym.make(
        "YamSim-v0",
        control_mode=config.control_mode,
        policy_control_freq=config.policy_control_freq,
    )
    if not config.headless:
        env = MujocoViewerWrapper(env)  # Visualize env with Mujoco viewer
    else:
        print("Running in headless mode (no visualization)")

    # Policy
    policy = SimDummyPolicy(env.action_space)

    # Main loop
    obs, info = env.reset()
    print("DEBUG: Info:", info)
    for _ in range(config.num_steps):
        action, _ = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
