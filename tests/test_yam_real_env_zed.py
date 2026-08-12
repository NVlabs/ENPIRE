# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-surface regression for the removed internal ZED wrapper."""

from __future__ import annotations

from enpire.env.forge.robot.yam import yam_real_env


def test_internal_nonblocking_zed_wrapper_is_not_exported() -> None:
    assert not hasattr(yam_real_env, "NonBlockingZed")
