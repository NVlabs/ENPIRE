#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One-command setup for RoboCasa365 benchmark eval on OSMO.
#
# Idempotent - safe to re-run. This script:
#   1. runs the GR00T pool/server setup (clone benchmark, model_server_venv, N1.5 checkpoint),
#   2. clones robocasa365,
#   3. builds an isolated benchmark client env, and
#   4. writes the right env vars into .env.grootpool so run_eval_365.py works
#      without extra hotfix exports or --client-python flags.
#
# Does NOT need setup_env_osmo.sh — run_eval_365.py uses model_server_venv and
# client_venv only; the forge CAP agent venv is irrelevant here.
#
# After this completes:
#   source .env.grootpool
#   "$GROOTPOOL_SERVER_PYTHON" cap/saved_scripts/robocasa/policy_eval/gr00t/run_eval_365.py \
#     --model-version n15 --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1 --gpus 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OSMO_ROOT="${OSMO_ROOT:-/mnt/shared/<user>}"
GROOT_BENCHMARK_ROOT="${GROOT_BENCHMARK_ROOT:-$OSMO_ROOT/Isaac-GR00T-benchmark}"
MODEL_DIR="${MODEL_DIR:-$OSMO_ROOT/PretrainedModels/gr00t_n1-5}"
MODEL_PATH="$MODEL_DIR/multitask_learning/checkpoint-120000"
HF_HOME="${HF_HOME:-$OSMO_ROOT/huggingface}"
ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$OSMO_ROOT/robocasa365}"
CLIENT_VENV_DIR="${CLIENT_VENV_DIR:-$GROOT_BENCHMARK_ROOT/client_venv}"
CLIENT_PY="$CLIENT_VENV_DIR/bin/python"
GROOTPOOL_SERVER_PYTHON="${GROOTPOOL_SERVER_PYTHON:-$GROOT_BENCHMARK_ROOT/model_server_venv/bin/python}"
ENVFILE="$PROJECT_DIR/.env.grootpool"
FORGE_ASSETS_ROOT="$PROJECT_DIR/third_party/robocasa/robocasa/models/assets"
TARGET_ASSETS_ROOT="$ROBOCASA365_ROOT/robocasa/models/assets"

# OSMO often installs from a cache on a different filesystem than Lustre.
# Copy mode avoids repeated hardlink warnings from uv.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

N=5
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }

ASSET_DIRS=(
    textures
    generative_textures
    fixtures
    objects/objaverse
    objects/aigen_objs
    objects/lightwheel
)

assets_ok() {
    local base="$1"
    local sub dir
    for sub in "${ASSET_DIRS[@]}"; do
        dir="$base/$sub"
        if [[ ! -d "$dir" ]] || [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
            return 1
        fi
    done
    return 0
}

client_ok() {
    [[ -x "$CLIENT_PY" ]] || return 1
    PYTHONPATH="$PROJECT_DIR:$GROOT_BENCHMARK_ROOT" "$CLIENT_PY" - <<'PY' >/dev/null 2>&1
import av
import cv2
import gymnasium
import msgpack
import mujoco
import numpy
import robocasa
import scipy
import zmq
from cap.policy import inference_policy
from gr00t.eval.robot import RobotInferenceClient
print("ok")
PY
}

link_assets_from_forge() {
    local src_root="$1"
    local dst_root="$2"
    local sub src dst current

    for sub in "${ASSET_DIRS[@]}"; do
        src="$src_root/$sub"
        dst="$dst_root/$sub"

        if [[ ! -d "$src" ]] || [[ -z "$(ls -A "$src" 2>/dev/null)" ]]; then
            return 1
        fi

        mkdir -p "$(dirname "$dst")"

        if [[ -L "$dst" ]]; then
            current="$(readlink "$dst")"
            if [[ "$current" == "$src" ]]; then
                continue
            fi
            rm "$dst"
        elif [[ -d "$dst" ]] && [[ -z "$(ls -A "$dst" 2>/dev/null)" ]]; then
            rmdir "$dst"
        elif [[ -e "$dst" ]]; then
            continue
        fi

        ln -s "$src" "$dst"
    done
}

if ! command -v uv >/dev/null 2>&1; then
    red "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

yellow "[1/$N] Running GR00T server/checkpoint setup..."
bash "$PROJECT_DIR/scripts/setup_grootpool.sh"
green "[1/$N] GR00T server/checkpoint ready"

if [[ -d "$ROBOCASA365_ROOT/.git" ]]; then
    green "[2/$N] robocasa365: already cloned at $ROBOCASA365_ROOT"
else
    yellow "[2/$N] Cloning robocasa365_release to $ROBOCASA365_ROOT..."
    git clone --branch robocasa365_release \
        https://github.com/robocasa/robocasa.git \
        "$ROBOCASA365_ROOT"
    green "[2/$N] robocasa365 clone done"
fi

if client_ok; then
    green "[3/$N] client_venv: already built and complete"
else
    if [[ -x "$CLIENT_PY" ]]; then
        yellow "[3/$N] Repairing client_venv packages..."
    else
        yellow "[3/$N] Building client_venv with uv..."
        uv venv --python 3.11 --seed "$CLIENT_VENV_DIR"
    fi

    uv pip install --python "$CLIENT_PY" -e "$GROOT_BENCHMARK_ROOT"
    uv pip install --python "$CLIENT_PY" -e "$ROBOCASA365_ROOT"
    uv pip install --python "$CLIENT_PY" \
        "gymnasium==1.0.0" \
        "mujoco==3.3.1" \
        pyzmq \
        msgpack \
        scipy \
        opencv-python-headless \
        tqdm \
        av

    if ! client_ok; then
        red "client_venv still missing required imports after install"
        exit 1
    fi
    green "[3/$N] client_venv ready"
fi

if assets_ok "$TARGET_ASSETS_ROOT"; then
    green "[4/$N] robocasa365 assets: already present in clone"
elif assets_ok "$FORGE_ASSETS_ROOT"; then
    yellow "[4/$N] Linking robocasa365 assets from forge/third_party/robocasa..."
    link_assets_from_forge "$FORGE_ASSETS_ROOT" "$TARGET_ASSETS_ROOT"
    if ! assets_ok "$TARGET_ASSETS_ROOT"; then
        red "asset linking completed, but robocasa365 assets are still incomplete"
        exit 1
    fi
    green "[4/$N] robocasa365 assets linked"
else
    yellow "[4/$N] Downloading robocasa365 assets into clone (~10 GB)..."
    yes | "$CLIENT_PY" -m robocasa.scripts.download_kitchen_assets --type all
    if ! assets_ok "$TARGET_ASSETS_ROOT"; then
        red "robocasa365 assets are still incomplete after download"
        exit 1
    fi
    green "[4/$N] robocasa365 assets downloaded"
fi

cat > "$ENVFILE" <<EOF
# Auto-generated by scripts/setup_robocasa365_eval.sh
# Source this before tmux/launch_grootpool.sh or run_eval_365.py.
export GROOT_BENCHMARK_ROOT="$GROOT_BENCHMARK_ROOT"
export GROOTPOOL_MODEL_PATH="$MODEL_PATH"
export GROOT_N15_MODEL_PATH="$MODEL_PATH"
export HF_HOME="$HF_HOME"
export GROOTPOOL_SERVER_PYTHON="$GROOTPOOL_SERVER_PYTHON"
export ROBOCASA365_ROOT="$ROBOCASA365_ROOT"
export ROBOCASA365_CLIENT_PYTHON="$CLIENT_PY"
EOF
green "[5/$N] wrote $ENVFILE"

echo ""
green "=== Verification ==="
echo "server python:  $GROOTPOOL_SERVER_PYTHON"
echo "client python:  $CLIENT_PY"
echo "model path:     $MODEL_PATH"
echo "robocasa365:    $ROBOCASA365_ROOT"
PYTHONPATH="$PROJECT_DIR:$GROOT_BENCHMARK_ROOT" "$CLIENT_PY" - <<'PY'
import av
import gymnasium
import mujoco
import robocasa
print("PyAV:", av.__version__)
print("gymnasium:", gymnasium.__version__)
print("MuJoCo:", mujoco.__version__)
print("robocasa:", robocasa.__path__[0])
PY

echo ""
green "=== RoboCasa365 eval setup complete ==="
echo "Run:"
echo "  cd $PROJECT_DIR"
echo "  source .env.grootpool"
echo "  \$GROOTPOOL_SERVER_PYTHON cap/saved_scripts/robocasa/policy_eval/gr00t/run_eval_365.py \\"
echo "    --model-version n15 --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1 --gpus 0"
