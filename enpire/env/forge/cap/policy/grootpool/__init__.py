# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T Pool — stateless multi-worker inference middleware.

N1.5 panda_omron is stateless (observation_indices=[0]).
Protocol reduced to: step(obs) → action.

Usage:
    from enpire.env.forge.cap.policy.grootpool import GrootPoolClient, GrootPoolError

    with GrootPoolClient() as client:
        action = client.predict(obs)
"""

from enpire.env.forge.cap.policy.grootpool.client import GrootPoolClient, GrootPoolError

__all__ = ["GrootPoolClient", "GrootPoolError"]
