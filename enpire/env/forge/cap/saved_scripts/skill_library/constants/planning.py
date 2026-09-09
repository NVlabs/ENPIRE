# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-planning defaults inherited from saved-script baselines."""

import os

# Baseline stays 1.5; ENPIRE_PLANNING_SPEED lets an operator slow a first
# run on new hardware without editing a saved-script constant.
PLANNING_SPEED = float(os.environ.get("ENPIRE_PLANNING_SPEED", "1.5"))
IK_ERROR_THRESHOLD_M = 0.01
IK_XYZ_WEIGHT = 1.0
IK_RPY_WEIGHT = 0.3
BATCH_TOP_K = 16
BATCH_SOLVER_SPEED = "fast"
BATCH_VALIDATE_TRAJECTORY = False
MOTION_PLANNER_BACKEND = "curobo"

DEFAULT_XYZ_RELAXATIONS = (
    (0.0, 0.0, 0.0),
    (0.003, 0.0, 0.0),
    (-0.003, 0.0, 0.0),
    (0.0, 0.003, 0.0),
    (0.0, -0.003, 0.0),
    (0.003, 0.003, 0.0),
    (-0.003, -0.003, 0.0),
    (-0.003, 0.003, 0.0),
)
DEFAULT_RPY_RELAXATIONS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 8.0),
    (0.0, 0.0, -8.0),
)
