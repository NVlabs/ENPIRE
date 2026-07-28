# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra-instantiated robot adapters."""

from __future__ import annotations

from typing import Any

from enpire.env.forge.cap.agent.robot_adapters.base import RobotAdapter, cfg_select
from enpire.env.forge.cap.agent.robot_adapters.study import StudyAdapter


def adapter_for_env(env_name: str | None) -> RobotAdapter:
    """Backward-compatible adapter selection by env name.

    Real-YAM hardware deploy has been removed in this sim-only build.
    """
    name = str(env_name or "")
    if name == "yam-real" or name.startswith("yam-real:"):
        raise NotImplementedError(
            "Real-YAM hardware deploy is not available in this sim-only (ENPIRE) build."
        )
    return StudyAdapter()


def get_robot_adapter(cfg: Any | None) -> RobotAdapter:
    """Instantiate ``cfg.robot.adapter`` or fall back to env-name selection."""
    adapter_cfg = cfg_select(cfg, "robot.adapter", None)
    target = cfg_select(adapter_cfg, "_target_", None)
    if target:
        from hydra.utils import instantiate

        return instantiate(adapter_cfg, _recursive_=False)
    env_name = cfg_select(cfg, "env.name", None)
    return adapter_for_env(env_name)


__all__ = [
    "RobotAdapter",
    "StudyAdapter",
    "adapter_for_env",
    "get_robot_adapter",
]
