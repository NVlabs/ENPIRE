# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side eval loop for RoboCasa365 benchmark tasks.

Uses robocasa365 gym envs (robocasa/{TaskName}) with split support.
Supports both N1.5 (native obs keys) and N1.6 (obs key remap).

Launched by run_eval_365.py as a subprocess — not run directly.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym  # noqa: E402
import robocasa.wrappers.gym_wrapper  # noqa: E402, F401  — registers robocasa/* envs

from enpire.env.forge.cap.policy import (  # noqa: E402
    InferencePolicyConfig,
    ZMQPolicyBackend,
    inference_policy,
)
from enpire.env.forge.cap.policy.backend import PolicyBackend  # noqa: E402


class N15PolicyBackend(PolicyBackend):
    """Adapter: wraps RobotInferenceClient (torch serialization) as a PolicyBackend."""

    def __init__(self, host: str, port: int):
        from gr00t.eval.robot import RobotInferenceClient

        self._client = RobotInferenceClient(host=host, port=port)

    def predict(self, obs: dict) -> dict:
        # Add batch dim for single obs.  Strings (e.g. annotation.human.task_description)
        # must become np.array(["str"]) (1-D), NOT np.array("str") (0-D) — the N1.5
        # server transforms index into dim-0 and crash on 0-D arrays.
        batched = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                batched[k] = v[None]
            elif isinstance(v, str):
                batched[k] = np.array([v])
            else:
                batched[k] = v
        action = self._client.get_action(batched)
        return {
            k: v[0] if hasattr(v, "ndim") and v.ndim > 0 else v
            for k, v in action.items()
        }

    def predict_batch(self, obs: dict) -> dict:
        return self._client.get_action(obs)

    def reset(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Obs key remap: robocasa365 → N1.6 expected keys
# ---------------------------------------------------------------------------

_REMAP_365_TO_N16 = {
    "video.robot0_agentview_left": "video.res256_image_side_0",
    "video.robot0_agentview_right": "video.res256_image_side_1",
    "video.robot0_eye_in_hand": "video.res256_image_wrist_0",
    "annotation.human.task_description": "annotation.human.action.task_description",
}

# Camera maps for video recording
_CAMERA_MAP_N15 = {
    "video.robot0_agentview_left": "side_left",
    "video.robot0_agentview_right": "side_right",
    "video.robot0_eye_in_hand": "wrist",
}

_CAMERA_MAP_N16 = {
    "video.res256_image_side_0": "side_left",
    "video.res256_image_side_1": "side_right",
    "video.res256_image_wrist_0": "wrist",
}


# ---------------------------------------------------------------------------
# Env factory (must be picklable for AsyncVectorEnv spawn)
# ---------------------------------------------------------------------------


def _make_env(env_name, split):
    """Module-level factory — picklable by AsyncVectorEnv(context='spawn')."""
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return gym.make(env_name, enable_render=True, split=split)


def _make_env_n16(env_name, split):
    """Factory that creates a robocasa365 env producing N1.6-compatible observations.

    Subclasses RoboCasaGymEnv to override get_observation(), extracting extra
    state keys (joint_position, ee_absolute, etc.) from the SAME raw robosuite
    obs dict — no second _get_observations() call, no state corruption in
    multiprocessing. Must be fully self-contained for AsyncVectorEnv pickling.
    """
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as _np
    from gymnasium import spaces as _spaces
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    # Extra raw robosuite keys that N1.6 expects but robocasa365 doesn't expose
    _EXTRA_RAW_KEYS = {
        "robot0_eef_pos": ("state.end_effector_position_absolute", (3,)),
        "robot0_eef_quat": ("state.end_effector_rotation_absolute", (4,)),
        "robot0_joint_pos": ("state.joint_position", (7,)),
        "robot0_joint_pos_cos": ("state.joint_position_cos", (7,)),
        "robot0_joint_pos_sin": ("state.joint_position_sin", (7,)),
        "robot0_joint_vel": ("state.joint_velocity", (7,)),
        "robot0_gripper_qvel": ("state.gripper_qvel", (2,)),
    }
    _OBS_RENAME = {
        "video.robot0_agentview_left": "video.res256_image_side_0",
        "video.robot0_agentview_right": "video.res256_image_side_1",
        "video.robot0_eye_in_hand": "video.res256_image_wrist_0",
        "annotation.human.task_description": "annotation.human.action.task_description",
    }

    class N16RoboCasaEnv(RoboCasaGymEnv):
        """RoboCasaGymEnv subclass that produces N1.6-format observations."""

        def _create_obs_and_action_space(self):
            super()._create_obs_and_action_space()
            # Rename obs space keys
            new_obs = {}
            for k, v in self.observation_space.items():
                new_obs[_OBS_RENAME.get(k, k)] = v
            # Add extra state spaces
            for raw_key, (state_key, shape) in _EXTRA_RAW_KEYS.items():
                new_obs[state_key] = _spaces.Box(-10, 10, shape, dtype=_np.float32)
            self.observation_space = _spaces.Dict(new_obs)
            # Fix action space: N1.6 expects Discrete for gripper/control_mode
            new_act = {}
            for k, v in self.action_space.items():
                if k in ("action.gripper_close", "action.control_mode"):
                    new_act[k] = _spaces.Discrete(2)
                else:
                    new_act[k] = v
            self.action_space = _spaces.Dict(new_act)

        def get_observation(self, raw_obs):
            """Override: produce N1.6-format obs from the same raw_obs dict.

            Called once per step/reset by the parent class. No extra
            _get_observations() call needed — raw_obs already has everything.
            """
            obs = super().get_observation(raw_obs)
            out = {}
            # Rename keys
            for k, v in obs.items():
                out[_OBS_RENAME.get(k, k)] = v
            # Extract extra states from the raw robosuite obs
            for raw_key, (state_key, _shape) in _EXTRA_RAW_KEYS.items():
                if raw_key in raw_obs:
                    out[state_key] = raw_obs[raw_key].astype(_np.float32)
            return out

    # Extract bare task name from "robocasa/TaskName"
    task_name = env_name.split("/")[-1]
    env = N16RoboCasaEnv(env_name=task_name, enable_render=True, split=split)
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # N1.5 uses torch serialization; N1.6 uses msgpack (our ZMQPolicyBackend)
    if args.model_version == "n15":
        backend = N15PolicyBackend(host=args.server_host, port=args.server_port)
    else:
        backend = ZMQPolicyBackend(host=args.server_host, port=args.server_port)

    # Official benchmark: n_action_steps=16 means execute all 16 predicted steps.
    # In our InferencePolicyConfig, replan_horizon controls how many steps are
    # executed per policy query. Set both to 16 to match official full-chunk execution.
    # For receding horizon (our default for PandaOmron), use --n-action-steps 8.
    config = InferencePolicyConfig(
        action_horizon=16,
        replan_horizon=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )

    # Select env factory and camera map based on model version
    if args.model_version == "n16":
        env_fn = partial(_make_env_n16, args.env_name, args.split)
        cam_map = _CAMERA_MAP_N16
    else:
        env_fn = partial(_make_env, args.env_name, args.split)
        cam_map = _CAMERA_MAP_N15

    vid_dir = str(log_dir / "videos") if args.record_video else None
    if not args.record_video:
        cam_map = None

    task_short = args.env_name.split("/")[-1]

    if args.n_parallel_envs <= 1:
        env = env_fn()
        results = inference_policy(
            env,
            backend,
            config,
            n_episodes=args.n_eps_per_task,
            video_dir=vid_dir,
            camera_map=cam_map,
            task_name=task_short,
            model_name=args.model_name,
            model_path=args.model_path,
        )
        env.close()
    else:
        results = inference_policy(
            env_fn,
            backend,
            config,
            n_episodes=args.n_eps_per_task,
            n_envs=args.n_parallel_envs,
            video_dir=vid_dir,
            camera_map=cam_map,
            task_name=task_short,
            model_name=args.model_name,
            model_path=args.model_path,
        )

    backend.close()

    successes = [r.success for r in results]
    rate = float(np.mean(successes) * 100)
    # Extract task_description from the first episode that has one.
    task_description = ""
    for r in results:
        td = r.info.get("task_description", "")
        if td:
            task_description = td
            break
    out = {
        "env_name": args.env_name,
        "model_path": args.model_path,
        "task_description": task_description,
        "n_episodes": len(successes),
        "success_rate": rate,
        "successes": successes,
    }
    with open(log_dir / "task_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"success_rate": rate, "n_episodes": len(successes)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=5555)
    p.add_argument("--env-name", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--model-version", default="n15", choices=["n15", "n16"])
    p.add_argument("--model-name", default="GR00T-N1.5-3B")
    p.add_argument("--model-path", default="")
    p.add_argument("--n-eps-per-task", type=int, default=50)
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=16,
        help="Steps to execute per policy query (16=official full chunk, 8=receding horizon)",
    )
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--n-parallel-envs", type=int, default=1)
    p.add_argument("--record-video", action="store_true")
    run(p.parse_args())
