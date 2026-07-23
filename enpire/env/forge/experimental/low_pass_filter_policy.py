# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-pass filter policy wrapper for smoothing policy actions."""

from typing import Any

import numpy as np

from enpire.env.forge.experimental.filter_utils import FIRFilter


class LowPassFilterPolicyWrapper:
    """Policy wrapper that applies a low-pass FIR filter to policy actions.

    This wrapper smooths the actions produced by the underlying policy using
    exponentially-weighted averaging over a rolling window of past actions.
    """

    def __init__(self, policy: Any, *, k: int, alpha: float):
        """Initialize the low-pass filter policy wrapper.

        Args:
            policy: The underlying policy to wrap
            k: Buffer size (number of samples to keep in the rolling buffer)
            alpha: Smoothing factor for exponential weights (0 < alpha <= 1).
                   Values closer to 1 give more weight to recent samples.
        """
        self.policy = policy
        self.filter = FIRFilter(k=k, alpha=alpha)

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Get action from the underlying policy and apply low-pass filtering.

        Args:
            observation: Current observation

        Returns:
            Tuple of (filtered_action, info) where filtered_action is the
            smoothed version of the policy's action
        """
        action, info = self.policy.get_action(observation)
        filtered_action = self.filter.step(action)
        return filtered_action, info

    def reset(self) -> dict[str, Any] | None:
        """Reset the policy and the low-pass filter."""
        # Reset the filter by creating a new instance with same parameters
        k = self.filter.k
        alpha = self.filter.alpha
        self.filter = FIRFilter(k=k, alpha=alpha)
        return self.policy.reset()
