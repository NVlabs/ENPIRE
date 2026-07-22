# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real environment for YAM bimanual robot station."""

import logging
import time
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np

from enpire.env.forge.robot.constants import (
    LEFT_FOLLOWER_PORT,
    RIGHT_FOLLOWER_PORT,
    YAM_ARM_KD,
    YAM_ARM_KP,
    YAM_GRIPPER_GRAVCOMP_TORQUE_LIMIT_NM,
    YAM_GRIPPER_KD,
    YAM_GRIPPER_KP,
)
from enpire.env.forge.robot.yam._base_yam_env import _BaseYamEnv
from enpire.env.forge.robot.yam.arm_client import FollowerRobotClient
from enpire.env.forge.robot.yam.non_blocking_camera import NonBlockingCamera
from enpire.env.forge.tools.data_collection.async_video_compression import (
    blank_compressed_image,
    compress_image,
)

logger = logging.getLogger(__name__)


class YamRealEnv(_BaseYamEnv):
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
        enabled_depth_camera_names: tuple[str, ...] = (),
        enable_eef_force_observation: bool = False,
        delta_ee_translation_xyz_max: tuple[float, float, float] = (
            0.0003,
            0.0003,
            0.0006,
        ),
        enable_reset_telemetry: bool = False,
        reset_max_joint_velocity: float = 0.5,
    ):
        if control_mode is None:
            raise ValueError("Control mode must be specified")
        self.enable_eef_force_observation = bool(enable_eef_force_observation)
        self.enable_reset_telemetry = bool(enable_reset_telemetry)
        self.reset_max_joint_velocity = float(reset_max_joint_velocity)
        if self.reset_max_joint_velocity <= 0:
            raise ValueError("reset_max_joint_velocity must be positive")
        self.enabled_camera_names = tuple(str(x) for x in (enabled_camera_names))
        self.enabled_depth_camera_names = tuple(str(x) for x in enabled_depth_camera_names)
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
        self.control_period = 1.0 / policy_control_freq
        self.last_step_time = time.time()
        self.command_enabled_sides = {"left", "right"}
        if self.enable_eef_force_observation:
            for side in ("left", "right"):
                self.observation_space.spaces[f"{side}_eef_force"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                )

        # Follower arms
        self.follower_arms = {}
        for side, port in [
            ("left", LEFT_FOLLOWER_PORT),
            ("right", RIGHT_FOLLOWER_PORT),
        ]:
            print(f"[YAM] Connecting {side} follower arm to localhost:{port}")
            follower_arm = FollowerRobotClient(host="localhost", port=port)

            # Set arm to gravcomp mode
            follower_arm.command_joint_state(
                {
                    "pos": np.zeros(7),
                    "vel": np.zeros(7),
                    "kp": np.zeros(7),
                    "kd": np.zeros(7),
                    "gripper_torque_limit_nm": YAM_GRIPPER_GRAVCOMP_TORQUE_LIMIT_NM,
                }
            )

            self.follower_arms[side] = follower_arm

        # Cameras
        self.cameras = (
            {
                camera_name: NonBlockingCamera(
                    camera_name,
                    image_transform=lambda image, camera_name=camera_name: self._crop_camera_image(
                        camera_name, image
                    ),
                    enable_depth=camera_name in self.enabled_depth_camera_names,
                )
                for camera_name in self.enabled_camera_names
            }
            if self.enable_cameras
            else {}
        )

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Convert from delta EE pose to absolute cartesian, then to joint
        ## Debug control loop timing overrun###############
        ################################################
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

        # Command follower arms to desired joint positions
        for side, follower_arm in self.follower_arms.items():
            if side not in self.command_enabled_sides:
                continue
            follower_arm.command_joint_pos(
                np.concatenate([action[f"{side}_joint_pos"], action[f"{side}_gripper_pos"]])
            )

        # Maintain desired control freq
        sleep_end_time = self.last_step_time + self.control_period
        if time.time() > sleep_end_time:  # TODO: Resolve latency issues
            print(
                f"\r\033[91mWarning: Control loop timing overrun (budget {self.control_period * 1000:.1f} ms, actual {1000 * (time.time() - self.last_step_time):.1f} ms)\033[0m    ",
                end="",
                flush=True,
            )
        while time.time() < sleep_end_time:
            time.sleep(0.0001)
        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)

        reward = self._compute_reward()
        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        return obs, reward, False, False, info

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:

        if self.enable_reset_telemetry:
            print("Resetting the environment...")
        target_state = self._resolve_reset_target_state(options)
        self._interpolate_to_target(target_state)
        self._record_gripper_cmd(target_state)
        if self.enable_reset_telemetry:
            print("Done resetting the environment")

        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)
        # Initialize current joint positions for delta control tracking
        if self._is_delta_mode:
            self._seed_delta_ee_command(obs)
        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        return obs, info

    def set_command_enabled_sides(self, sides) -> None:
        enabled = set(sides)
        invalid = enabled - {"left", "right"}
        if invalid:
            raise ValueError(f"Invalid command side(s): {sorted(invalid)!r}")
        self.command_enabled_sides = enabled

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        obs_l = self.follower_arms["left"].get_observations()
        obs_r = self.follower_arms["right"].get_observations()
        left_pos, left_quat, right_pos, right_quat = self._kinematics.forward_kinematics(
            np.asarray(obs_l["joint_pos"], dtype=np.float64).reshape(6),
            np.asarray(obs_r["joint_pos"], dtype=np.float64).reshape(6),
        )
        side_obs = obs_l if side == "left" else obs_r
        return {
            "joint_pos": np.asarray(side_obs["joint_pos"], dtype=np.float32).reshape(6),
            "gripper_pos": np.asarray(side_obs["gripper_pos"], dtype=np.float32).reshape(1),
            "ee_pos": np.asarray(
                left_pos if side == "left" else right_pos, dtype=np.float32
            ).reshape(3),
            "ee_quat": np.asarray(
                left_quat if side == "left" else right_quat, dtype=np.float32
            ).reshape(4),
        }

    def render_rgb(self, camera: str) -> np.ndarray:
        return self.cameras[camera].get_image()

    def render_depth(self, camera: str) -> np.ndarray | None:
        return self.cameras[camera].get_depth()

    def get_camera_intrinsics(self, camera: str) -> list[float]:
        intr = self.cameras[camera].get_intrinsics()
        return [
            float(intr["fx"]),
            float(intr["fy"]),
            float(intr["cx"]),
            float(intr["cy"]),
        ]

    def move_bimanual_joint_keypoints(
        self,
        timestamps,
        left_joint_positions,
        right_joint_positions,
        left_gripper_positions=None,
        right_gripper_positions=None,
        playback_speed: float = 1.0,
        command_hz: float = 60.0,
        start_interp_s: float = 0.0,
    ) -> dict[str, Any]:
        ts_original = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts_original.size < 1:
            return {"success": False, "reason": "empty timestamps"}
        if not np.all(np.isfinite(ts_original)):
            return {"success": False, "reason": "timestamps contain non-finite values"}
        ts = ts_original - float(ts_original[0])
        if np.any(np.diff(ts) < -1e-9):
            return {"success": False, "reason": "timestamps must be monotonic"}
        keep = np.ones(ts.shape[0], dtype=bool)
        keep[1:] = np.diff(ts) > 1e-9
        ts = ts[keep]
        left7 = self._prepare_joint7_waypoints(
            "left",
            left_joint_positions,
            left_gripper_positions,
            len(ts_original),
        )[keep]
        right7 = self._prepare_joint7_waypoints(
            "right",
            right_joint_positions,
            right_gripper_positions,
            len(ts_original),
        )[keep]
        speed = max(0.05, float(playback_speed))
        ts = ts / speed
        duration_s = float(ts[-1]) if ts.size else 0.0
        dt = 1.0 / max(1.0, float(command_hz))

        cur_left7 = np.asarray(
            self.follower_arms["left"].get_joint_pos(),
            dtype=np.float64,
        ).reshape(7)
        cur_right7 = np.asarray(
            self.follower_arms["right"].get_joint_pos(),
            dtype=np.float64,
        ).reshape(7)
        interp_steps = int(np.ceil(max(0.0, float(start_interp_s)) / dt))
        for step in range(1, interp_steps + 1):
            alpha = float(step) / float(max(interp_steps, 1))
            self._command_bimanual_joint7(
                (1.0 - alpha) * cur_left7 + alpha * left7[0],
                (1.0 - alpha) * cur_right7 + alpha * right7[0],
            )
            time.sleep(dt)

        t0 = time.time()
        command_count = 0
        while True:
            t_now = time.time() - t0
            self._command_bimanual_joint7(
                self._sample_keypoints(ts, left7, t_now),
                self._sample_keypoints(ts, right7, t_now),
            )
            command_count += 1
            if t_now >= duration_s:
                break
            time.sleep(dt)

        settle_steps = max(1, int(round(0.2 / dt)))
        for _ in range(settle_steps):
            self._command_bimanual_joint7(left7[-1], right7[-1])
            time.sleep(dt)
        return {
            "success": True,
            "reason": "ok",
            "waypoints": int(ts.size),
            "duration_s": round(duration_s, 4),
            "command_count": int(command_count),
        }

    def _prepare_joint7_waypoints(
        self,
        side: str,
        joint_positions,
        gripper_positions,
        n: int,
    ) -> np.ndarray:
        arr = np.asarray(joint_positions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 6:
            raise ValueError(
                f"{side}_joint_positions must have shape (N,6) or (N,7); got {arr.shape}"
            )
        joints = arr[:, :6]
        if gripper_positions is None:
            if arr.shape[1] >= 7:
                gripper = arr[:, 6]
            else:
                cur = self.follower_arms[side].get_observations()["gripper_pos"]
                gripper = np.full(
                    n,
                    float(np.asarray(cur).reshape(-1)[0]),
                    dtype=np.float64,
                )
        else:
            gripper = np.asarray(gripper_positions, dtype=np.float64).reshape(n, -1)[:, 0]
        return np.column_stack([joints, np.clip(gripper, 0.0, 1.0)]).astype(np.float64)

    @staticmethod
    def _sample_keypoints(ts: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
        if t <= ts[0]:
            return values[0]
        if t >= ts[-1]:
            return values[-1]
        hi = int(np.searchsorted(ts, t, side="right"))
        lo = hi - 1
        denom = max(float(ts[hi] - ts[lo]), 1e-9)
        alpha = (float(t) - float(ts[lo])) / denom
        return (1.0 - alpha) * values[lo] + alpha * values[hi]

    def _command_bimanual_joint7(
        self,
        left7: np.ndarray,
        right7: np.ndarray,
    ) -> None:
        kp = np.asarray(YAM_ARM_KP + [YAM_GRIPPER_KP], dtype=np.float32)
        kd = np.asarray(YAM_ARM_KD + [YAM_GRIPPER_KD], dtype=np.float32)
        for side, joint_pos in (
            ("left", np.asarray(left7, dtype=np.float32).reshape(7)),
            ("right", np.asarray(right7, dtype=np.float32).reshape(7)),
        ):
            self.follower_arms[side].command_joint_state(
                {
                    "pos": joint_pos,
                    "vel": np.zeros(7),
                    "kp": kp,
                    "kd": kd,
                }
            )

    def _read_current_joint_state(self) -> dict[str, np.ndarray]:
        current: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            qpos = np.asarray(self.follower_arms[side].get_joint_pos(), dtype=np.float32).reshape(7)
            current[f"{side}_joint_pos"] = qpos[:6].copy()
            current[f"{side}_gripper_pos"] = qpos[6:7].copy()
        return current

    def _interpolate_to_target(
        self,
        joint_state: dict[str, np.ndarray],
        duration: float | None = None,
    ) -> None:
        """Interpolate to target joint state.

        ``duration`` defaults to a distance-adaptive value capped at ~0.5 rad/s
        with a 1 s floor. Callers can still override.
        """
        initial_pos = np.concatenate(
            [
                self.follower_arms["left"].get_joint_pos(),
                self.follower_arms["right"].get_joint_pos(),
            ]
        )

        target_pos = np.concatenate(
            [
                joint_state["left_joint_pos"],
                joint_state["left_gripper_pos"],
                joint_state["right_joint_pos"],
                joint_state["right_gripper_pos"],
            ]
        )
        assert initial_pos.shape == (14,) and target_pos.shape == (14,)

        max_err = float(np.max(np.abs(target_pos - initial_pos)))
        if max_err < 0.005:
            if self.enable_reset_telemetry:
                print(
                    f"[yam_real_env] _interpolate_to_target: already at target "
                    f"(max_joint_err={max_err:.3f} rad); skipping"
                )
            return
        if duration is None:
            duration = max(1.0, max_err / self.reset_max_joint_velocity)
        if self.enable_reset_telemetry:
            print(
                f"[yam_real_env] _interpolate_to_target: duration={duration:.1f}s, "
                f"max_joint_err={max_err:.3f} rad, "
                f"max_joint_velocity={self.reset_max_joint_velocity:.3f} rad/s"
            )

        kp = np.asarray(YAM_ARM_KP + [YAM_GRIPPER_KP], dtype=np.float32)
        kd = np.asarray(YAM_ARM_KD + [YAM_GRIPPER_KD], dtype=np.float32)

        start_time = time.time()
        while time.time() - start_time < duration:
            # Linearly interpolate between initial and target joint state
            alpha = (time.time() - start_time) / duration
            interp_pos = (1 - alpha) * initial_pos + alpha * target_pos

            # Command follower arms to interpolated joint state
            for side, joint_pos in [
                ("left", interp_pos[:7]),
                ("right", interp_pos[7:]),
            ]:
                self.follower_arms[side].command_joint_state(
                    {
                        "pos": joint_pos,
                        "vel": np.zeros(7),
                        "kp": kp,
                        "kd": kd,
                    }
                )

            time.sleep(0.02)

        # Keep commanding the final target until we are close enough.
        settle_start = time.time()
        settle_timeout = 1.0
        settle_tol = 0.01
        while time.time() - settle_start < settle_timeout:
            current_pos = np.concatenate(
                [
                    self.follower_arms["left"].get_joint_pos(),
                    self.follower_arms["right"].get_joint_pos(),
                ]
            )
            max_err = float(np.max(np.abs(current_pos - target_pos)))
            for side, joint_pos in [
                ("left", target_pos[:7]),
                ("right", target_pos[7:]),
            ]:
                self.follower_arms[side].command_joint_state(
                    {
                        "pos": joint_pos,
                        "vel": np.zeros(7),
                        "kp": kp,
                        "kd": kd,
                    }
                )
            if max_err < settle_tol:
                break
            time.sleep(0.02)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {}
        _ts: dict[str, float] = {}

        # Joint and gripper states
        for side, follower_arm in self.follower_arms.items():
            follower_obs = follower_arm.get_observations()
            _ts[f"{side}_state"] = time.time()

            obs[f"{side}_joint_pos"] = follower_obs["joint_pos"]
            obs[f"{side}_gripper_pos"] = follower_obs["gripper_pos"]
            if self.enable_eef_force_observation and "eef_force" in follower_obs:
                obs[f"{side}_eef_force"] = np.asarray(
                    follower_obs["eef_force"], dtype=np.float32
                ).reshape(3)
        # Update current joint positions for delta control tracking
        if self._is_delta_mode:
            self._update_current_joint_pos_from_observation(obs)

        # Convert from joint space to Cartesian space if we use ee control mode.
        if self._use_cartesian_proprioception:
            obs = self._convert_from_joint_to_cartesian(obs)

        # Camera images
        for camera_name in self.enabled_camera_names:
            if self.enable_cameras:
                image = self.cameras[camera_name].get_image()
                obs[f"{camera_name}_camera_image"] = compress_image(image)
            else:
                obs[f"{camera_name}_camera_image"] = blank_compressed_image()
            _ts[f"{camera_name}_camera"] = time.time()

        # Convert float64 to float32
        obs = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in obs.items()}
        # Attach per-component creation timestamps
        obs["__timestamps"] = _ts

        return obs

    def _compute_reward(self) -> float:
        return 0.0

    def _get_info(self) -> dict[str, Any]:
        return {}

    def close(self):
        # Close cameras then cleanup gymnasium env
        for camera in getattr(self, "cameras", {}).values():
            camera.close()
        self.cameras = {}
        super().close()


class RealDummyPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self._started_at = time.monotonic()

    def reset(self) -> None:
        return None

    def get_action(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Simple sinusoidal action for testing
        frequency = 0.5
        t = time.monotonic() - self._started_at
        osc = np.sin(2 * np.pi * frequency * t)

        def scale(key):
            low = self.action_space[key].low
            high = self.action_space[key].high
            mid = (low + high) / 2
            amp = (high - low) / 2 * 0.5
            return mid + amp * osc

        action = {
            "left_joint_pos": scale("left_joint_pos"),
            "left_gripper_pos": scale("left_gripper_pos"),
            "right_joint_pos": scale("right_joint_pos"),
            "right_gripper_pos": scale("right_gripper_pos"),
        }

        # For real robot testing, only move last arm joint and gripper
        action["left_joint_pos"][:5] = 0.0
        action["right_joint_pos"][:5] = 0.0

        return action, {}


def main():
    from gymnasium.envs.registration import register

    # Environment
    register(id="YamReal-v0", entry_point="enpire.env.forge.robot.yam.yam_real_env:YamRealEnv")
    env = gym.make("YamReal-v0")

    # Policy
    policy = RealDummyPolicy(env.action_space)

    # Main loop
    obs, info = env.reset()

    print("DEBUG: Info:", info)
    for _ in range(1000):
        action, _ = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    main()
