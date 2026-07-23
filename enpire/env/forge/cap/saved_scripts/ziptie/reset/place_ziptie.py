# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place the plugged-in ziptie into the white box, then home.

Loaded by ziptie/reset/flat.py via load_module(); call place_ziptie(holding, other, PLAN).

Sequence (the `other` arm keeps holding the ziptie throughout):
  0. `holding` arm releases the strip (open gripper).
  1. detect the white-box CENTROID (SAM3 mask -> depth -> world) and move `other` OVER it (+Z hover),
     KEEPING its current orientation.
  2. open `other` gripper to drop the ziptie, then both arms go home.

Kept dependency-free at import time (only `time` + pure-data macros at top level; the
skill_library / log_mask imports are lazy inside place_ziptie). Loading never moves the robot.
"""

from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG

_RC = dict(TABLE_SORT_RUN_CONFIG)
_PLAN = {k: _RC[k] for k in ("planning_speed", "ik_error_threshold", "ik_xyz_weight", "ik_rpy_weight", "planner_backend") if k in _RC}

from skill_library.namespace import freespace_move, go_home, open_gripper

freespace_move(preview_only=False, **_PLAN, **{"right_target_pos": [0.683, -0.115, 0.918], "right_target_rpy": [2.1, 137.3, -25.3]})
# 2. drop the ziptie, then both arms home.
open_gripper("right")
go_home()

