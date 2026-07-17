#!/usr/bin/env bash
set -euo pipefail

# Optionally set RL_KEYBOARD_DEVICE to a station-local logical input alias.

source "$(dirname "$0")/assert_env_var.sh" || exit 1

ROBOT_INTERFACE_PROFILE=1 uv run python tmux/realworld_rl/pusht_supervisor.py \
  --config-file "$(dirname "$0")/tasks_config/pusht/pusht.yaml" \
  --data-saving-path "$RL_DATA_PATH" \
  "$@"
