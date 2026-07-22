# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helper + single source of truth for the ziptie-handover ("hover") pose.

The 4 macros below (LEFT/RIGHT x POS/RPY) are the ONE place the handover pose lives.
They are reused by:
  - get_rew_rgb_debug.py / get_rew_rgb_debug_trt.py  (via load_module(), call move_to_rew_pose())
  - rl/ziptie_runner.py                              (imports the macros to build _HOVER_TARGET)

Keep this module dependency-free at import time: only pure-data macros + `time` at the
top level, and the skill_library import is lazy (inside move_to_rew_pose). That lets
rl/ziptie_runner.py load the macros by path without dragging in skill_library, and means
loading the module never moves the robot — the caller invokes move_to_rew_pose() explicitly.

Flat mode only: the ziptie faces upward on the table at the start (not lying on a side).
"""

import time

# ── Handover pose: single source of truth (display RPY in degrees) ──────────
# LEFT_TARGET_POS = [0.4813, 0.0502, 1.1772]; LEFT_TARGET_RPY = [10.72, 92.94, 59.80]
# RIGHT_TARGET_POS = [0.5013, -0.0751, 1.1772]; RIGHT_TARGET_RPY = [-81.25, 48.77, 5.24]
LEFT_TARGET_POS, LEFT_TARGET_RPY = [0.48+0.02, 0.105, 1.213], [0, 75.34, 61]
RIGHT_TARGET_POS, RIGHT_TARGET_RPY = [0.532, -0.036, 1.22], [-83.1, 66.07, -0.14]


def move_to_rew_pose(settle_s: float = 5.0, planning_speed: float | None = None, torque_limit_left=0.2, torque_limit_right=0.2):
    """Transport both arms to the handover pose, close grippers, then settle.

    Blocks for ``settle_s`` seconds after the grippers close so the scene is static
    before detection starts. ``planning_speed`` (rad/s) overrides the freespace_move
    speed for this move; ``None`` keeps freespace_move's default.
    """
    from skill_library.namespace import close_gripper, freespace_move

    print(f"[reward] freespace_move to ziptie-handover pose (planning_speed={planning_speed})...")
    _speed_kw = {} if planning_speed is None else {"planning_speed": float(planning_speed)}
    freespace_move(
        left_target_pos=LEFT_TARGET_POS,
        left_target_rpy=LEFT_TARGET_RPY,
        right_target_pos=RIGHT_TARGET_POS,
        right_target_rpy=RIGHT_TARGET_RPY,
        **_speed_kw,
    )
    close_gripper(side="left", torque_limit=torque_limit_left)
    close_gripper(side="right", torque_limit=torque_limit_right)
    print(f"[reward] arms in pose; settling {settle_s:.0f}s before detection starts...")
    time.sleep(settle_s)

def move_to_rew_pose_right(settle_s: float = 5.0, planning_speed: float | None = None, torque_limit=0.2):
    """Transport both arms to the handover pose, close grippers, then settle.

    Blocks for ``settle_s`` seconds after the grippers close so the scene is static
    before detection starts. ``planning_speed`` (rad/s) overrides the freespace_move
    speed for this move; ``None`` keeps freespace_move's default.
    """
    from skill_library.namespace import close_gripper, freespace_move

    print(f"[reward] freespace_move to ziptie-handover pose (planning_speed={planning_speed})...")
    _speed_kw = {} if planning_speed is None else {"planning_speed": float(planning_speed)}
    freespace_move(
        right_target_pos=RIGHT_TARGET_POS,
        right_target_rpy=RIGHT_TARGET_RPY,
        **_speed_kw,
    )
    close_gripper(side="right", torque_limit=torque_limit)
    print(f"[reward] arms in pose; settling {settle_s:.0f}s before detection starts...")
    time.sleep(settle_s)

def move_to_rew_pose_left(settle_s: float = 5.0, planning_speed: float | None = None, torque_limit=0.2):
    """Transport both arms to the handover pose, close grippers, then settle.

    Blocks for ``settle_s`` seconds after the grippers close so the scene is static
    before detection starts. ``planning_speed`` (rad/s) overrides the freespace_move
    speed for this move; ``None`` keeps freespace_move's default.
    """
    from skill_library.namespace import close_gripper, freespace_move

    print(f"[reward] freespace_move to ziptie-handover pose (planning_speed={planning_speed})...")
    _speed_kw = {} if planning_speed is None else {"planning_speed": float(planning_speed)}
    freespace_move(
        left_target_pos=LEFT_TARGET_POS,
        left_target_rpy=LEFT_TARGET_RPY,
        **_speed_kw,
    )
    close_gripper(side="left", torque_limit=torque_limit)
    print(f"[reward] arms in pose; settling {settle_s:.0f}s before detection starts...")
    time.sleep(settle_s)

move_to_rew_pose(settle_s=5.0)
