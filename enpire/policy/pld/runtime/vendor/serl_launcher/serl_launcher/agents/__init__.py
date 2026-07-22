# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .continuous.sac_mini import SACMiniAgent

agents = {
    "sac_mini": SACMiniAgent,
}

