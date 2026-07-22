# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from skill_library.constants.robot import LEFT_HOME_XYZ
from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG
from skill_library.namespace import close_gripper, freespace_move, go_home, open_gripper, rotate_joint
from skill_library.pick import go_birdeye, pick_object

OBJECT_NAME = "white strip"
STRIP_LEN = 0.15

RUN_CONFIG = dict(TABLE_SORT_RUN_CONFIG)
PLAN = {k: RUN_CONFIG[k] for k in ("planning_speed", "ik_error_threshold",
        "ik_xyz_weight", "ik_rpy_weight", "planner_backend") if k in RUN_CONFIG}

go_birdeye(side="both", **RUN_CONFIG)
holding = pick_object(OBJECT_NAME, grasp_mode="2d",
                      min_grasp_site=0.10, max_grasp_site=0.20, **RUN_CONFIG)
if holding is None:
    print("[transport] initial grasp failed"); go_home(); raise SystemExit(1)
other = "right" if holding == "left" else "left"

# Mirror across world Y when right arm is holding. Convention is written for
# holding=left (sign=+1). For holding=right (sign=-1):
#   - Y offsets from centerline flip sign
#   - RPY roll & yaw flip sign (pitch unchanged)
#   - joint-6 deltas flip sign (joint axes mirror across the bimanual centerline)
sign = +1 if holding == "left" else -1
def mrpy(r, p, y): return [sign * r, p, sign * y]

MID_XYZ  = [LEFT_HOME_XYZ[0] + 0.15, 0.0, 1.10]
MID_RPY  = mrpy(-90.0, 90.0,  90.0)   # holding arm
TAIL_RPY = mrpy(-90.0, 90.0, -90.0)   # other arm
TAIL_XYZ = [MID_XYZ[0] - STRIP_LEN, MID_XYZ[1] - sign * 0.05, MID_XYZ[2] - 0.05]

print(f"[transport] holding={holding} sign={sign} MID_RPY={MID_RPY} TAIL_RPY={TAIL_RPY}")
print(f"[transport] {holding} -> middle xyz={MID_XYZ}")
freespace_move(preview_only=False, **PLAN,
               **{f"{holding}_target_pos": MID_XYZ, f"{holding}_target_rpy": MID_RPY})

print("[transport] pausing 1s so strip settles")
time.sleep(1.0)

print(f"[transport] {other} open + coarse approach tail xyz={[round(v,4) for v in TAIL_XYZ]} rpy={TAIL_RPY}")
open_gripper(other)
freespace_move(preview_only=False, **PLAN,
               **{f"{other}_target_pos": TAIL_XYZ, f"{other}_target_rpy": TAIL_RPY,
                  f"{other}_gripper_target_width": 1.0})

# Refined approach: dig ~4 cm further toward the centerline (mirror-aware).
TAIL_XYZ = [MID_XYZ[0] - STRIP_LEN, MID_XYZ[1] - sign * 0.09, MID_XYZ[2] - 0.05]
freespace_move(preview_only=False, **PLAN,
               **{f"{other}_target_pos": TAIL_XYZ, f"{other}_target_rpy": TAIL_RPY})
close_gripper(other, torque_limit=1.5)

# Synchronized inward roll of both wrists. Direction flips with holding side.
rotate_joint(rotations=[{"side": "left",  "joint": 6, "delta_deg": sign * 150.0},
                        {"side": "right", "joint": 6, "delta_deg": sign * 150.0}])
close_gripper(other, torque_limit=0.75)

print("[transport] done")

