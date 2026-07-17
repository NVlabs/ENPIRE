#!/usr/bin/env bash
# One-command setup for GR00T N1.5 policy pool on OSMO.
#
# Idempotent — safe to re-run. Skips steps already done.
#
# After this completes you can run:
#   bash tmux/launch_grootpool.sh 3    # start pool with 3 GR00T workers
#
# Prereq: run `scripts/setup_env_osmo.sh` first for the base forge env.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────
OSMO_ROOT="${OSMO_ROOT:-/mnt/shared/<user>}"
GROOT_BENCHMARK_ROOT="${GROOT_BENCHMARK_ROOT:-$OSMO_ROOT/Isaac-GR00T-benchmark}"
MODEL_DIR="${MODEL_DIR:-$OSMO_ROOT/PretrainedModels/gr00t_n1-5}"
MODEL_PATH="$MODEL_DIR/multitask_learning/checkpoint-120000"
HF_HOME="${HF_HOME:-$OSMO_ROOT/huggingface}"

N=4
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }

# ── 1. clone Isaac-GR00T-benchmark ────────────────────────────────────────
if [[ -d "$GROOT_BENCHMARK_ROOT/.git" ]]; then
    green "[1/$N] Isaac-GR00T-benchmark: already cloned at $GROOT_BENCHMARK_ROOT"
else
    yellow "[1/$N] Cloning Isaac-GR00T-benchmark to $GROOT_BENCHMARK_ROOT…"
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://github.com/robocasa-benchmark/Isaac-GR00T.git \
        "$GROOT_BENCHMARK_ROOT"
    (cd "$GROOT_BENCHMARK_ROOT" && git checkout HEAD -- .)
    green "[1/$N] clone done"
fi

# ── 2. build model_server_venv (via uv — 3-5× faster than pip) ────────────
VENV_DIR="$GROOT_BENCHMARK_ROOT/model_server_venv"
VENV_PY="$VENV_DIR/bin/python"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import gr00t, torch, fastapi, uvicorn" >/dev/null 2>&1; then
    green "[2/$N] model_server_venv: already built and complete"
else
    yellow "[2/$N] Building model_server_venv with uv (~5-10 min)…"
    if ! command -v uv >/dev/null 2>&1; then
        red "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # Python 3.11 (matches the forge venv on OSMO). Using 3.12 fails because
    # onnx==1.15.0 — a transitive gr00t dep — has no cp312 prebuilt wheel and
    # building from source hits a broken libprotobuf-dev on Ubuntu 24.04.
    uv venv --python 3.11 "$VENV_DIR"

    # Pin torch to 2.9 to match the flash-attn wheel below (ABI lock).
    # cu128's latest (2.11) has a different libtorch_cuda ABI → flash-attn fails
    # to load with "undefined symbol: c10_cuda_check_implementation".
    uv pip install --python "$VENV_PY" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        "torch==2.9.*" "torchvision==0.24.*"

    FLASH_WHL="flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"
    if [[ ! -f "$FLASH_WHL" ]]; then
        wget -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/$FLASH_WHL" \
            || { red "failed to fetch $FLASH_WHL — check release page at github.com/mjun0812/flash-attention-prebuild-wheels"; exit 1; }
    fi
    uv pip install --python "$VENV_PY" "$FLASH_WHL"

    # Force binary-only for onnx so uv won't try a broken source build if the
    # constraint resolves to a version without cp311 wheels.
    # gr00t's pyproject pins transformers==4.51.3, pydantic, peft, timm, etc.
    # — the -e install below pulls those in at the right versions.
    uv pip install --python "$VENV_PY" --only-binary=onnx -e "$GROOT_BENCHMARK_ROOT"

    # Only install packages NOT already pinned by gr00t's pyproject.
    # diffusers: only in gr00t's [dev] / [base] extras, not main deps.
    # fastapi / uvicorn: grootpool middleware deps, not in gr00t at all.
    # Do NOT add transformers/accelerate/pyzmq/msgpack/einops/timm/albumentations
    # here — they are gr00t main deps with pinned versions, and an unpinned
    # install would upgrade them and break the @dataclass in GR00T_N1_5_Config.
    uv pip install --python "$VENV_PY" \
        diffusers \
        fastapi 'uvicorn[standard]'

    green "[2/$N] venv built"
fi

# ── 2b. sanity: ensure transformers matches gr00t's pin ──────────────────
# (fix-up for a prior setup that accidentally upgraded it)
if [[ -x "$VENV_PY" ]]; then
    CURR_TF=$("$VENV_PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
    if [[ "$CURR_TF" != "4.51.3" ]]; then
        yellow "[2b/$N] transformers is $CURR_TF; reinstalling pinned 4.51.3…"
        uv pip install --python "$VENV_PY" "transformers==4.51.3"
        green "[2b/$N] transformers pinned to 4.51.3"
    else
        green "[2b/$N] transformers already at 4.51.3"
    fi
fi

# ── 3. N1.5 checkpoint (12 GB, ~10-30 min) ────────────────────────────────
# snapshot_download preserves repo-relative paths under local_dir, so passing
# local_dir=<parent> gets files at   <parent>/gr00t_n1-5/multitask_learning/checkpoint-120000/*
MODEL_PARENT="$(dirname "$MODEL_DIR")"
# Recover from the earlier bug: files nested one level deep at MODEL_DIR/gr00t_n1-5/…
STALE_NEST="$MODEL_DIR/gr00t_n1-5/multitask_learning/checkpoint-120000"

if [[ -d "$MODEL_PATH" ]] && [[ -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    green "[3/$N] N1.5 checkpoint: already at $MODEL_PATH"
elif [[ -d "$STALE_NEST" ]] && [[ -n "$(ls -A "$STALE_NEST" 2>/dev/null)" ]]; then
    yellow "[3/$N] Found checkpoint at wrong-nested path; relocating (no re-download)…"
    rm -rf "$MODEL_DIR/multitask_learning" 2>/dev/null || true
    mv "$MODEL_DIR/gr00t_n1-5/multitask_learning" "$MODEL_DIR/multitask_learning"
    rmdir "$MODEL_DIR/gr00t_n1-5" 2>/dev/null || true
    green "[3/$N] checkpoint at $MODEL_PATH"
else
    yellow "[3/$N] Downloading N1.5 multitask checkpoint (~12 GB)…"
    mkdir -p "$MODEL_PARENT"
    HF_HOME="$HF_HOME" python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    'robocasa/robocasa365_checkpoints',
    allow_patterns='gr00t_n1-5/multitask_learning/checkpoint-120000/*',
    local_dir='$MODEL_PARENT',
    max_workers=4,
    resume_download=True,
)
PY
    green "[3/$N] checkpoint downloaded"
fi

# ── 4. write env file for tmux launcher ───────────────────────────────────
ENVFILE="$OSMO_ROOT/forge/.env.grootpool"
cat > "$ENVFILE" <<EOF
# Auto-generated by scripts/setup_grootpool.sh — sourced by tmux/launch_grootpool.sh
export GROOT_BENCHMARK_ROOT="$GROOT_BENCHMARK_ROOT"
export GROOTPOOL_MODEL_PATH="$MODEL_PATH"
export HF_HOME="$HF_HOME"
export GROOTPOOL_SERVER_PYTHON="$VENV_PY"
EOF
green "[4/$N] wrote $ENVFILE"

green "──────────────────────────────────────────────"
green "grootpool env ready."
green "Next:  bash tmux/launch_grootpool.sh 3    # 3 workers on GPUs 0,1,2"
green "──────────────────────────────────────────────"
