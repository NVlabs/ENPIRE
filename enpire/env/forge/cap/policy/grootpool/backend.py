# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless PolicyBackend adapter for grootpool.

Each predict() call is an independent request — no session lifecycle,
no stickiness. N1.5 panda_omron uses observation_indices=[0] (current
frame only), so episodes do not share state between calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from enpire.env.forge.cap.policy.backend import PolicyBackend
from enpire.env.forge.cap.policy.grootpool.client import GrootPoolClient


class GrootpoolBackend(PolicyBackend):
    """Stateless grootpool backend. One predict() = one worker round-trip."""

    def __init__(
        self,
        task_description: str = "",
        *,
        model: str = "n15",
        endpoint: str | None = None,
    ) -> None:
        # task_description is carried in obs["annotation.human.task_description"]
        # for N1.5 — kept as param for API compatibility but not used for routing.
        self._client = GrootPoolClient(endpoint=endpoint)

    def predict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._client.predict(obs)

    def predict_batch(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._client.predict(obs)

    def reset(self) -> None:
        pass  # stateless — nothing to reset

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GrootpoolBackend":
        return self

    def __exit__(self, *args) -> None:
        self.close()
