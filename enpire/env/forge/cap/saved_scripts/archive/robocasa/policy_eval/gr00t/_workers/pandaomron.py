# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side eval loop using inference_policy() interface.

Runs in the robocasa_uv venv (for GrootRoboCasaEnv) with
PYTHONPATH pointing to lecar-tbd (for cap.policy).
Isaac-GR00T code is untouched.

Launched by benchmark.py, not run directly.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from functools import partial  # noqa: E402

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402, F401
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: E402, F401

from enpire.env.forge.cap.policy import (  # noqa: E402
    InferencePolicyConfig,
    ZMQPolicyBackend,
    inference_policy,
)


def _make_env(env_name):
    """Module-level factory — picklable by AsyncVectorEnv(context='spawn').

    Must re-import robocasa in spawned child to register gymnasium env IDs.
    """
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    return gym.make(env_name, enable_render=True)


def run(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    backend = ZMQPolicyBackend(host=args.server_host, port=args.server_port)
    config = InferencePolicyConfig(
        action_horizon=args.action_horizon,
        replan_horizon=args.replan_horizon,
        max_episode_steps=args.max_episode_steps,
    )

    env_fn = partial(_make_env, args.env_name)

    vid_dir = str(log_dir / "videos") if args.record_video else None
    cam_map = None
    if vid_dir:
        from enpire.env.forge.cap.policy.video import CAMERA_MAPS

        cam_map = CAMERA_MAPS.get("robocasa_panda_omron")

    task_short = args.env_name.split("/")[-1].replace("_PandaOmron_Env", "")

    model_name = args.model_name
    model_path = args.model_path

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
            model_name=model_name,
            model_path=model_path,
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
            model_name=model_name,
            model_path=model_path,
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
    p.add_argument("--n-eps-per-task", type=int, default=10)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--replan-horizon", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=720)
    p.add_argument("--n-parallel-envs", type=int, default=1)
    p.add_argument("--record-video", action="store_true", help="Save per-camera videos")
    p.add_argument("--model-name", default="GR00T-N1.6-3B")
    p.add_argument("--model-path", default="")
    run(p.parse_args())
