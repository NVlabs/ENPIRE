#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One-time OSMO environment setup for RoboCasa evaluation.
# Run from the forge repo root on OSMO:
#   bash scripts/setup_env_osmo.sh
#
# Installs: EGL libs, uv, robosuite, Python deps, cuRobo, openblas,
#           libssl1.1, graspnetAPI, RoboCasa assets.
# Skips steps that are already done (safe to re-run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Venv lives on shared Lustre so it survives workflow teardown and can be
# reused across OSMO jobs. Override with VENV_DIR=... if needed.
VENV_DIR="${VENV_DIR:-/mnt/shared/<user>/python_envs/forge}"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"

# Symlink repo-local .venv -> shared venv so tools that hardcode `.venv/`
# (IDEs, helper scripts) keep working.
if [[ ! -e .venv ]] || [[ -L .venv && "$(readlink .venv)" != "$VENV_DIR" ]]; then
    rm -f .venv
    ln -s "$VENV_DIR" .venv
fi

N=9  # total steps
print_green() { printf '\033[32m%s\033[0m\n' "$*"; }
print_yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

# ── 1. System deps (EGL for headless MuJoCo rendering) ──
if dpkg -l libegl1-mesa-dev >/dev/null 2>&1; then
    print_green "[1/$N] EGL libs: already installed"
else
    print_yellow "[1/$N] Installing EGL libs..."
    apt-get update -qq && apt-get install -y -qq \
        libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
        > /dev/null 2>&1
    print_green "[1/$N] EGL libs: done"
fi

# ── 2. uv ──
if command -v uv >/dev/null 2>&1; then
    print_green "[2/$N] uv: already installed ($(uv --version))"
else
    print_yellow "[2/$N] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    print_green "[2/$N] uv: done"
fi
export PATH="$HOME/.local/bin:$PATH"

# ── 3. robosuite (synced via rsync from local) ──
if [[ -f third_party/robosuite/setup.py ]]; then
    print_green "[3/$N] robosuite: present"
else
    print_yellow "[3/$N] ERROR: third_party/robosuite/ missing. Run sync_to_remote.sh locally first."
    exit 1
fi

# ── 4. Python deps ──
if [[ -f "$VENV_DIR/bin/python" ]]; then
    print_green "[4/$N] Python venv: already exists at $VENV_DIR"
else
    print_yellow "[4/$N] Creating venv at $VENV_DIR + installing deps..."
fi
uv sync --extra robocasa
print_green "[4/$N] Python deps: synced"

# ── 5. cuRobo ──
if "$VENV_DIR/bin/python" -c "import curobo" 2>/dev/null; then
    print_green "[5/$N] cuRobo: already installed"
else
    print_yellow "[5/$N] Building cuRobo (takes ~5 min)..."
    uv pip install --python "$VENV_DIR/bin/python" ninja 2>/dev/null || true
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
    uv pip install --python "$VENV_DIR/bin/python" -e third_party/curobo --no-build-isolation
    print_green "[5/$N] cuRobo: done"
fi

# ── 6. OpenBLAS (needed by AnyGrasp) ──
if ldconfig -p 2>/dev/null | grep -q libopenblas; then
    print_green "[6/$N] OpenBLAS: already available"
else
    print_yellow "[6/$N] Installing OpenBLAS..."
    apt-get install -y -qq libopenblas0 > /dev/null 2>&1 || \
        print_yellow "[6/$N] OpenBLAS: apt install failed — AnyGrasp may not work"
    print_green "[6/$N] OpenBLAS: done"
fi

# ── 7. libssl1.1 (needed by AnyGrasp — Ubuntu 24.04 only has OpenSSL 3) ──
LIBSSL_DIR="/root/runtime-deps/libssl11/usr/lib/x86_64-linux-gnu"
if [[ -f "$LIBSSL_DIR/libssl.so.1.1" ]]; then
    print_green "[7/$N] libssl1.1: already installed"
else
    print_yellow "[7/$N] Installing libssl1.1 from Debian Bookworm..."
    mkdir -p "$LIBSSL_DIR"
    wget -q http://deb.debian.org/debian/pool/main/o/openssl/libssl1.1_1.1.1w-0+deb11u1_amd64.deb \
        -O /tmp/libssl1.1.deb
    dpkg-deb -x /tmp/libssl1.1.deb /root/runtime-deps/libssl11/
    rm -f /tmp/libssl1.1.deb
    print_green "[7/$N] libssl1.1: done"
fi
export RUNTIME_DEPS_ROOT=/root/runtime-deps

# ── 8. graspnetAPI (needed by AnyGrasp) ──
if "$VENV_DIR/bin/python" -c "import graspnetAPI" 2>/dev/null; then
    print_green "[8/$N] graspnetAPI: already installed"
else
    print_yellow "[8/$N] Installing graspnetAPI..."
    uv pip install --python "$VENV_DIR/bin/python" graspnetAPI
    print_green "[8/$N] graspnetAPI: done"
fi

# ── 9. RoboCasa kitchen assets (~10 GB) ──
# The upstream download_kitchen_assets.py has no non-interactive skip; it always
# re-downloads. We gate here by checking that all 6 asset folders exist and are
# non-empty. If any is missing, re-run the download (it re-fetches all buckets,
# which is fine — restarts from scratch so a partial failure self-heals).
ROBOCASA_ROOT="$("$VENV_DIR/bin/python" -c 'import robocasa; print(robocasa.__path__[0])' 2>/dev/null)"
ASSETS_OK=1
for sub in textures generative_textures fixtures objects/objaverse objects/aigen_objs objects/lightwheel; do
    dir="$ROBOCASA_ROOT/models/assets/$sub"
    if [[ ! -d "$dir" ]] || [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
        print_yellow "[9/$N] Missing or empty: $dir"
        ASSETS_OK=0
    fi
done
if [[ "$ASSETS_OK" -eq 1 ]]; then
    print_green "[9/$N] RoboCasa assets: already downloaded (all 6 folders present)"
else
    print_yellow "[9/$N] Downloading RoboCasa assets (~10 GB)..."
    # Pipe `y` twice: once for the top-level confirm, then auto-accept per-bucket.
    yes | uv run --no-sync python -m robocasa.scripts.download_kitchen_assets --type all
    print_green "[9/$N] RoboCasa assets: done"
fi

# ── Verify ──
echo ""
print_green "=== Verification ==="
export MUJOCO_GL=egl
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
uv run --no-sync python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
uv run --no-sync python -c "import robosuite; print('robosuite:', robosuite.__version__)"
uv run --no-sync python -c "import robocasa; print('robocasa: OK')"
"$VENV_DIR/bin/python" -c "import curobo; print('cuRobo: OK')"
[[ -f "$LIBSSL_DIR/libssl.so.1.1" ]] && echo "libssl1.1: OK" || echo "libssl1.1: MISSING"

echo ""
print_green "=== Setup complete ==="
echo ""
echo "To run (minimal — cuRobo auto-start):"
echo "  export MUJOCO_GL=egl ANTHROPIC_API_KEY=... GEMINI_API_KEY=..."
echo "  uv run --no-sync python run_agent.py experiment=pick_place_v1 runtime.agent_name=eval_osmo runtime.curobo_port=0 llm.backend=cloud"
echo ""
echo "To launch vision servers (needs LFS .so files synced from local + SAM3 HF token):"
echo "  export RUNTIME_DEPS_ROOT=/root/runtime-deps"
echo "  See docs/plan/OSMO_ROBOCASA_EVAL.md for full server launch commands"
