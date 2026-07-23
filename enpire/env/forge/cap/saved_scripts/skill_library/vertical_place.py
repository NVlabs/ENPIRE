# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vertical_place.py — skill library, append-only
from enpire.env.forge.cap.agent.skill_registry import skill
from skill_library.namespace import *  # noqa: F401, F403


@skill
def vertical_place_v1(side, target_pos, z_offset=0.03):
    """Lower EE to target_pos + [0,0,z_offset] and open gripper to release.

    No orientation change — EE keeps whatever it held after the grasp.
    """
    import numpy as np
    target = np.array(target_pos, dtype=float).copy()
    target[2] += z_offset
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    lowered = r.status == "Success"
    print(f"vertical_place_v1: target={target.tolist()} status={r.status}")
    open_gripper(side)
    return lowered, {
        "success": lowered,
        "status": r.status,
        "target": target.tolist(),
        "z_offset": z_offset,
    }
