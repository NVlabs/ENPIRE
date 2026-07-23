# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# hover_above.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill


@skill
def hover_above_v1(side, xyz, clearance=0.12):
    """Move EE above xyz by clearance metres. No orientation change."""
    import numpy as np
    target = np.array(xyz, dtype=float).copy()
    target[2] += clearance
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    success = r.status == "Success"
    print(f"hover_above_v1: target={target.tolist()} status={r.status}")
    return success, {
        "success": success,
        "status": r.status,
        "target": target.tolist(),
    }
