# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ziptie SAM3 prompt strings — the single source shared by _compute_rew_rgb.py and
_visualize_rew.py. Pure literals with NO load_module / imports of the reward modules, so neither has to
load the other just for the prompts (that formed a circular load: _compute_rew_rgb -> _visualize_rew ->
_compute_rew_rgb -> ... -> RecursionError)."""

ZIPTIE_COLOR = "red"
# ZIPTIE_COLOR = "pink white"
ZIPTIE_STRAP_NAME_IN_TOP_CAM = f"long thin {ZIPTIE_COLOR} strip"
ZIPTIE_HEAD_NAME_IN_TOP_CAM = [f"small {ZIPTIE_COLOR} knob", f"small {ZIPTIE_COLOR} block"]
ZIPTIE_HEAD_NAME_IN_RIGHT_CAM = f"{ZIPTIE_COLOR} block" #f"small {ZIPTIE_COLOR} block"
ZIPTIE_STRAP_NAME_IN_RIGHT_CAM = f"long {ZIPTIE_COLOR} ziptie"

