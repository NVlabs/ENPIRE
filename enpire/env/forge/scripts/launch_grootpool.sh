#!/usr/bin/env bash
# Launch the grootpool middleware on OSMO.
#
# Prereqs (one-time):
#   1. Isaac-GR00T-benchmark cloned and model_server_venv built
#      (see docs/ROBOCASA_INTEGRATION_POLICY.md §3.3).
#   2. N1.5 checkpoint staged on Lustre.
#   3. forge repo rsynced to OSMO (see .claude/skills/sync-to-osmo/).
#
# Usage:
#   bash scripts/launch_grootpool.sh [--mock] [--n-workers N] [--gpu-ids 0,1]
#
# Env vars (override shell defaults):
#   GROOT_BENCHMARK_ROOT   — path to Isaac-GR00T-benchmark clone
#   GROOTPOOL_MODEL_PATH   — path to N1.5 checkpoint
#   GROOTPOOL_N_WORKERS    — number of worker subprocesses (default 2)
#   GROOTPOOL_GPU_IDS      — comma-sep GPU ids (default 0,1)
#   GROOTPOOL_LISTEN_PORT  — ZMQ ROUTER port (default 7070)
#   GROOTPOOL_ADMIN_PORT   — FastAPI admin port (default 7071)
#   HF_HOME                — set to a Lustre path on OSMO (root disk is small)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

GROOT_BENCHMARK_ROOT="${GROOT_BENCHMARK_ROOT:-/mnt/shared/<user>/Isaac-GR00T-benchmark}"
GROOTPOOL_MODEL_PATH="${GROOTPOOL_MODEL_PATH:-/mnt/shared/<user>/PretrainedModels/gr00t_n1-5/multitask_learning/checkpoint-120000}"
GROOTPOOL_N_WORKERS="${GROOTPOOL_N_WORKERS:-2}"
GROOTPOOL_GPU_IDS="${GROOTPOOL_GPU_IDS:-0,1}"
GROOTPOOL_LISTEN_PORT="${GROOTPOOL_LISTEN_PORT:-7070}"
GROOTPOOL_ADMIN_PORT="${GROOTPOOL_ADMIN_PORT:-7071}"
GROOTPOOL_BASE_PORT="${GROOTPOOL_BASE_PORT:-5555}"
HF_HOME="${HF_HOME:-/mnt/shared/<user>/.cache/huggingface}"

# Parse --mock pass-through (everything else is forwarded via env).
EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

SERVER_PYTHON="$GROOT_BENCHMARK_ROOT/model_server_venv/bin/python"

# Detect mode from args.
MODE="real"
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        --mock) MODE="mock" ;;
        --stub) MODE="stub" ;;
    esac
done

if [[ "$MODE" == "real" ]]; then
    if [[ ! -x "$SERVER_PYTHON" ]]; then
        echo "error: $SERVER_PYTHON not found or not executable" >&2
        echo "  (build Isaac-GR00T-benchmark/model_server_venv per docs/ROBOCASA_INTEGRATION_POLICY.md §3.3)" >&2
        exit 1
    fi
    if [[ ! -d "$GROOTPOOL_MODEL_PATH" ]]; then
        echo "error: GROOTPOOL_MODEL_PATH ($GROOTPOOL_MODEL_PATH) does not exist" >&2
        exit 1
    fi
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME
export GROOT_BENCHMARK_ROOT
export GROOTPOOL_MODEL_PATH
export GROOTPOOL_N_WORKERS
export GROOTPOOL_GPU_IDS
export GROOTPOOL_LISTEN_PORT
export GROOTPOOL_ADMIN_PORT
export GROOTPOOL_BASE_PORT

echo "grootpool: mode=$MODE (defaults: n_workers=$GROOTPOOL_N_WORKERS gpus=$GROOTPOOL_GPU_IDS listen=$GROOTPOOL_LISTEN_PORT admin=$GROOTPOOL_ADMIN_PORT; CLI flags override)"

# mock and stub modes run in the forge uv venv (torch-only is enough); real
# mode needs the Isaac-GR00T-benchmark model_server_venv for gr00t + flash-attn.
if [[ "$MODE" == "real" ]]; then
    exec "$SERVER_PYTHON" -m cap.policy.grootpool.server "${EXTRA_ARGS[@]}"
else
    VENV_DIR="${VENV_DIR:-/mnt/shared/<user>/python_envs/forge}"
    FORGE_PY="${FORGE_PYTHON:-$VENV_DIR/bin/python}"
    if [[ ! -x "$FORGE_PY" ]]; then FORGE_PY="$PROJECT_DIR/.venv/bin/python"; fi
    if [[ ! -x "$FORGE_PY" ]]; then FORGE_PY=python3; fi
    exec "$FORGE_PY" -m cap.policy.grootpool.server "${EXTRA_ARGS[@]}"
fi
