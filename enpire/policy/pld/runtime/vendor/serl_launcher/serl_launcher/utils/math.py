# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import jax.numpy as jnp


def expectile_regression_loss(predicted_qs, target_qs, expectile):
    """
    Expectile regression loss
    """
    return jnp.mean(
        jnp.where(
            predicted_qs < target_qs,
            expectile * (predicted_qs - target_qs) ** 2,
            (1 - expectile) * (predicted_qs - target_qs) ** 2,
        )
    )


def gumbel_regression_loss(predicted_qs, target_qs, temperature):
    """
    Gumbel regression loss
    """
    return jnp.mean(
        jnp.where(
            predicted_qs < target_qs,
            temperature * (predicted_qs - target_qs) ** 2,
            (1 - temperature) * (predicted_qs - target_qs) ** 2,
        )
    )


def return_to_go(trajectory: List[Dict[str, Any]], discount: float):
    assert "critic_targets" in trajectory[0], "critic_targets not in trajectory"
    horizon = len(trajectory)
    trajectory[-1]["critic_targets"] = trajectory[-1]["rewards"]
    for i in range(horizon - 2, -1, -1):
        trajectory[i]["critic_targets"] = (
            trajectory[i]["rewards"] + discount * trajectory[i + 1]["critic_targets"]
        )
    return trajectory

