# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# nudge_down_and_regrasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill


@skill
def nudge_down_and_regrasp_v1(side, delta_z=-0.02, hold_strength=0.3):
    """Re-open gripper, nudge down by delta_z, compliant-close, verify grasp."""
    open_gripper(side)
    r = nudge(side=side, delta_pos=[0.0, 0.0, delta_z])
    print(f"nudge_down_and_regrasp_v1: nudge success={r.success} final_pos={r.final_pos}")
    close_gripper(side, compliant=True, hold_strength=hold_strength)
    info = get_gripper_info(side)
    grasped = bool(
        info.get("has_object")
        and not info.get("is_fully_closed", False)
        and (info.get("actuator_force_N") or 0.0) > 1.0
    )
    print(f"nudge_down_and_regrasp_v1: grasped={grasped} gripper_info={info}")
    return grasped, {
        "success": grasped,
        "delta_z": delta_z,
        "gripper_info": info,
    }
