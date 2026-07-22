# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy backend protocol — abstract interface to any VLA model server.

A backend wraps one model endpoint. It receives a raw env observation,
runs inference (handling all model-specific preprocessing internally),
and returns an un-batched action chunk.

Subclass this and implement ``predict()`` for each model family.
See ``backends/`` for concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PolicyBackend(ABC):
    """Abstract VLA model backend that returns action chunks."""

    @abstractmethod
    def predict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Single-env observation → un-batched action chunk.

        Args:
            obs: Single env observation (no batch dim).
                 e.g. ``video.res256_image_side_0`` shape ``(H, W, 3)``.

        Returns:
            Action chunk: ``{key: (action_horizon, D)}`` — no batch dim.
        """
        ...

    def predict_batch(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Batched observation → batched action chunk.

        Args:
            obs: Vectorized env observation (B already present from
                 AsyncVectorEnv). e.g. video shape ``(B, H, W, C)``.

        Returns:
            Action chunk: ``{key: (B, action_horizon, D)}``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support batched prediction. "
            "Use n_envs=1 or implement predict_batch()."
        )

    def reset(self) -> None:
        """Reset any internal state (e.g. history buffers)."""

    def close(self) -> None:
        """Release resources (sockets, GPU memory, …)."""
