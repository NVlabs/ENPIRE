#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest -q \
  tests/test_merge_bridge_contract.py \
  tests/test_merge_ui_contract.py \
  tests/test_merge_dependency_contract.py
