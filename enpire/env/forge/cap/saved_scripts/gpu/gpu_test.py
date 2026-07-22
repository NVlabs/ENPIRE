# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from math import dist
from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG
from skill_library.namespace import detect_objects_oneshot, get_robot_state, go_home
from skill_library.pick_place import pick_and_place

OBJECT_NAME = "small nvidia gpu"
TARGET_NAME = "large mother board"
HOVER_CLEARANCE_M = 0.18
RUN_CONFIG = dict(TABLE_SORT_RUN_CONFIG)
gpu_det = detect_objects_oneshot(OBJECT_NAME, camera="top")[OBJECT_NAME][0]
gpu_pos = gpu_det.position_3d
state = get_robot_state()
side = "left" if dist(gpu_pos, state.left_ee_pos) <= dist(gpu_pos, state.right_ee_pos) else "right"
RUN_CONFIG[f"local_{side}_target_pos"] = [gpu_pos[0], gpu_pos[1], gpu_pos[2] + HOVER_CLEARANCE_M]
print(f"[gpu_test] hover {side} wrist above {OBJECT_NAME}, then 3D-BB grasp -> {TARGET_NAME}")
try:
    moved = pick_and_place(OBJECT_NAME, TARGET_NAME, grasp_mode="3d_bb", camera=side, **RUN_CONFIG)
    print(f"[gpu_test] moved={bool(moved)}")
finally:
    go_home()

