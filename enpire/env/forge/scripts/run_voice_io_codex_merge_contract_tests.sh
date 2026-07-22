#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest -q \
  tests/test_merge_bridge_contract.py \
  tests/test_merge_ui_contract.py \
  tests/test_merge_dependency_contract.py
