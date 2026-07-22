# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAP Environment Layer — robot and simulation drivers.

This package provides the env layer of CAP's 4-layer architecture::

    cap/agent/        AGENT   LLM orchestrator, code generation
    cap/agent/tools/  TOOLS   Planning + perception (freespace_move, detect, vlm)
    cap/server/       SERVER  Router, safety, state cache, Portal RPC
    cap/env/          ENV     Robot/sim drivers — motion control, physics, rendering

Each env implements one or more protocols from ``cap.env.base``:

- ``EnvProtocol``    — required: step, observe, command, render
- ``EefControlProtocol`` — optional: per-tick EE control (compute_eef_action)
- ``SceneProtocol``      — optional: scene management (MuJoCo)
- ``TaskProtocol``       — optional: episodes, rewards, success (RoboCasa)

Adding a new robot or sim = one new file in this package.
"""

import os

from enpire.env.forge.cap.env.base import (
    EefControlProtocol,
    EnvProtocol,
    SceneProtocol,
    TaskProtocol,
)

__all__ = [
    "EefControlProtocol",
    "EnvProtocol",
    "SceneProtocol",
    "TaskProtocol",
    "create_env",
]


def create_env(env_name: str, viewer: bool = False, **kwargs):
    """Create an env by name. This is the single entry point for all envs.

    Env names:
        "yam"                          — YAM MuJoCo sim
        "yam-warp"                     — YAM GPU sim
        "robocasa"                     — RoboCasa365 (default task)
        "robocasa:TaskName"            — RoboCasa365 with specific task
        "robocasa:TaskName:RobotName"  — RoboCasa365 with specific task and robot

    Returns an env implementing EnvProtocol (and optionally others).
    """
    parts = env_name.split(":")
    env_type = parts[0]

    if env_type == "yam":
        from enpire.env.forge.cap.env.yam_mujoco import YamMuJoCoEnv

        return YamMuJoCoEnv(viewer=viewer)

    elif env_type == "yam-warp":
        from enpire.env.forge.cap.env.yam_warp import YamWarpEnv

        return YamWarpEnv()

    elif env_type == "robocasa":
        from enpire.env.forge.cap.env.profile import (
            robocasa_gr1_arms_profile,
            robocasa_panda_omron_profile,
        )
        from enpire.env.forge.cap.env.robocasa import RoboCasaEnv

        task = parts[1] if len(parts) > 1 else "PickPlaceCounterToCabinet"
        robot = parts[2] if len(parts) > 2 else "PandaOmron"

        if robot == "GR1ArmsOnly":
            profile = robocasa_gr1_arms_profile()
        else:
            profile = robocasa_panda_omron_profile()

        # Explicit kwargs take precedence over env vars.
        layout_ids = int(
            kwargs.pop("layout_ids", None) or os.environ.get("ROBOCASA_LAYOUT_ID", -3)
        )
        style_ids = int(
            kwargs.pop("style_ids", None) or os.environ.get("ROBOCASA_STYLE_ID", -3)
        )
        # Drop any passed-through controller_type silently — RoboCasaEnv is
        # pinned to OSC_POSE and no longer accepts this kwarg.
        kwargs.pop("controller_type", None)
        seed_val = kwargs.pop("seed", None)
        if seed_val is None:
            seed_str = os.environ.get("ROBOCASA_SEED", "")
            if seed_str:
                seed_val = int(seed_str)
        if seed_val is not None:
            kwargs["seed"] = seed_val

        return RoboCasaEnv(
            env_name=task,
            robot=robot,
            profile=profile,
            has_renderer=viewer,
            layout_ids=layout_ids,
            style_ids=style_ids,
            **kwargs,
        )

    elif env_type == "yam-real":
        from enpire.env.forge.cap.env.real_bimanual_yam.env import RealYamEnv

        enable_cameras = kwargs.pop("enable_cameras", True)
        return RealYamEnv(enable_cameras=enable_cameras)

    else:
        raise ValueError(
            f"Unknown env: {env_name!r}. "
            f"Available: yam, yam-warp, robocasa, robocasa:TaskName, yam-real"
        )
