# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT

from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


class CovarianceNetwork(nn.Module):
    """
    Network that outputs Cholesky decomposition of covariance matrix.
    Implementation is based on https://github.com/zhouzypaul/wsrl/blob/dgn/wsrl/agents/dgn.py
    """

    encoder: nn.Module
    hidden_dims: Tuple[int, ...] = (256, 256)
    output_dim: int = None
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)
        x = obs_enc

        # MLP layers with optional dropout
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)  # TODO: changed to mish

            # Only apply dropout during training
            if train and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        # Output layer - outputs flat vector of params
        # For d-dimensional action space, we need d*(d+1)/2 elements
        num_params = self.output_dim * (self.output_dim + 1) // 2
        cholesky_elements = nn.Dense(num_params)(x)

        return cholesky_elements

