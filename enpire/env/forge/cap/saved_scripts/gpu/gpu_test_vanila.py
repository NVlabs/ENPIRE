# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG
from skill_library.namespace import go_home
from skill_library.pick_place import pick_and_place

OBJECT_NAME = "small nvidia gpu"
TARGET_NAME = "large mother board"
RUN_CONFIG = dict(TABLE_SORT_RUN_CONFIG)

print(f"[gpu_test] 3D-BB pick/place: {OBJECT_NAME} -> {TARGET_NAME}")
try:
    moved = pick_and_place(OBJECT_NAME, TARGET_NAME, grasp_mode="3d_bb", **RUN_CONFIG)
    print(f"[gpu_test] moved={bool(moved)}")
finally:
    go_home()

