# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-surface regression for the internal ``raiden`` calibration UI."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_internal_raiden_package_is_not_a_runtime_dependency() -> None:
    assert importlib.util.find_spec("raiden") is None
    project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '"raiden' not in project.lower()
