# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward compat — re-exports from cap.env.robocasa.env."""

from enpire.env.forge.cap.env.robocasa.env import RoboCasaEnv  # noqa: F401

# Alias for create_env() compatibility
RoboCasaEnv = RoboCasaEnv
