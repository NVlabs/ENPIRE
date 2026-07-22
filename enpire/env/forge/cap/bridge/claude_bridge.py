# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible entrypoint for the generic CAP agent bridge."""

from enpire.env.forge.cap.bridge.agent_bridge import main
from enpire.env.forge.cap.bridge.agent_bridge import create_app

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
