# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enpire.policy.rl.events import TERMINAL_REWARD_BY_EVENT


def terminal_label_options(event: str | None) -> dict:
    if event not in TERMINAL_REWARD_BY_EVENT:
        return {}
    return {
        "episode_terminal_event": event,
        "episode_terminal_reward": TERMINAL_REWARD_BY_EVENT[event],
        "episode_terminal_done": True,
    }

