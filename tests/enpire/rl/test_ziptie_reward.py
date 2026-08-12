# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-surface regression for internal zip-tie CaP reward scripts."""

from __future__ import annotations

from pathlib import Path

from enpire.policy.cap.launcher import list_tasks


def test_internal_ziptie_cap_reward_is_not_distributed_or_registered() -> None:
    root = Path(__file__).resolve().parents[3]
    reward = root / "enpire/env/forge/cap/saved_scripts/ziptie/reward/_compute_rew_rgb.py"
    assert not reward.exists()
    assert all(not task.name.startswith("ziptie-") for task in list_tasks())
