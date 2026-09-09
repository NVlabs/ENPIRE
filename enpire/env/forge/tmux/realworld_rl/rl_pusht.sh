#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Optionally set RL_KEYBOARD_DEVICE to a station-local logical input alias.

HERE="$(cd "$(dirname "$0")" && pwd)"

source "$HERE/assert_env_var.sh" || exit 1

# The supervisor's default script paths (cap/saved_scripts/..., tasks_config/...)
# are relative to the Forge root, so run from there rather than from wherever
# the operator invoked this script.
cd "$HERE/../.."

ROBOT_INTERFACE_PROFILE=1 uv run python tmux/realworld_rl/pusht_supervisor.py \
  --config-file "$HERE/tasks_config/pusht/pusht.yaml" \
  --data-saving-path "$RL_DATA_PATH" \
  "$@"
