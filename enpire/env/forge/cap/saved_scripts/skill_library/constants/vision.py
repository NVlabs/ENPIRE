# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision and perception defaults inherited from saved scripts."""

VLM_BACKEND = "gemini"
VLM_MODEL = "gemini-2.5-flash"
DEFAULT_VLM_CAMERAS = ("top", "left", "right")
NUM_VOTES = 1
MAJORITY = 1
BUNDLESDF_CAMERA = "top"
ANYGRASP_TCP_OFFSET_Z_M = 0.0
ANYGRASP_DISABLE_PLANNER_Z_CLIPPING = True
TOP_GRASP_MAX = 16
TOP_GRASP_TRY = 16
DEFAULT_TARGET_DROP_Z_OFFSET = 0.10
TABLE_TARGET_DROP_Z_OFFSETS = {
    "blue plate": 0.10,
    "yellow plate": 0.10,
    "brown cardboard box": 0.10,
}
SMALL_OBJECT_TARGET_DROP_Z_OFFSETS = {
    "blue plate": 0.10,
    "yellow plate": 0.10,
}
