# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-surface regression for the removed internal GPU success bridge."""

from __future__ import annotations

import importlib.util


def test_internal_gpu_success_bridge_is_not_distributed() -> None:
    assert importlib.util.find_spec("enpire.policy.rl.gpu_success_full_cycle") is None
