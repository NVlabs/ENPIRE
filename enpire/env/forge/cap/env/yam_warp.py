# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAM Warp GPU simulation environment (--env yam-warp).

GPU-accelerated physics (MuJoCo Warp) with ray-traced rendering.
Import is lazy so machines without mujoco_warp installed can still import this module.
"""

from __future__ import annotations


def YamWarpEnv(**kwargs):  # type: ignore[no-untyped-def]
    """Return a WarpSimBackend instance (lazy import)."""
    from enpire.env.forge.cap.server.warp_sim_backend import WarpSimBackend
    return WarpSimBackend(**kwargs)


__all__ = ["YamWarpEnv"]
