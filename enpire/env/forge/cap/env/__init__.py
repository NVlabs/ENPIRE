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
- ``TaskProtocol``       — optional: episodes, rewards, success

Adding a new robot or sim = one new file in this package.
"""

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
        "yam-real"                     — real YAM bimanual hardware

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

    elif env_type == "yam-real":
        from enpire.env.forge.cap.env.real_bimanual_yam.env import RealYamEnv

        enable_cameras = kwargs.pop("enable_cameras", True)
        return RealYamEnv(enable_cameras=enable_cameras)

    else:
        raise ValueError(
            f"Unknown env: {env_name!r}. "
            f"Available: yam, yam-warp, yam-real"
        )
