#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pnp_counter_to_stove — keyframe reflection (skill_reflection enabled)
SESSION="${TMUX_SESSION:-osmo-eval}"
WINDOW="stove_kf"
PROVIDER_WINDOW="${NVIDIA_PROVIDER_WINDOW:-nv-provider}"
PROVIDER_PORT="${CAP_NVIDIA_PROVIDER_PORT:-8765}"
PROVIDER_URL="http://127.0.0.1:${PROVIDER_PORT}"
PROVIDER_PROXY_URL="${NVIDIA_PROVIDER_PROXY_URL-http://127.0.0.1:3128}"
PROJECT_DIR="/mnt/shared/<user>/forge"

tmux new-session -d -s "$SESSION" 2>/dev/null || true
if ! tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx "$PROVIDER_WINDOW"; then
    tmux new-window -t "$SESSION" -n "$PROVIDER_WINDOW" -c "$PROJECT_DIR"
    tmux send-keys -t "$SESSION:$PROVIDER_WINDOW" "UV_PROJECT_ENVIRONMENT=/mnt/shared/<user>/python_envs/forge PYTHONHASHSEED=42 \
        HTTPS_PROXY=${HTTPS_PROXY:-$PROVIDER_PROXY_URL} HTTP_PROXY=${HTTP_PROXY:-$PROVIDER_PROXY_URL} https_proxy=${https_proxy:-$PROVIDER_PROXY_URL} http_proxy=${http_proxy:-$PROVIDER_PROXY_URL} \
        NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,::1} no_proxy=${no_proxy:-localhost,127.0.0.1,::1} \
        uv run --no-sync python -m cap.agent.providers.nvidia_server serve \
        --host 127.0.0.1 --port $PROVIDER_PORT \
        --global-request-delay-s ${CAP_NVIDIA_GLOBAL_REQUEST_DELAY_S:-1.0} \
        --max-concurrent-per-key ${CAP_NVIDIA_MAX_CONCURRENT_PER_KEY:-1} \
        --dashboard" Enter
fi
tmux new-window -t "$SESSION" -n "$WINDOW" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:$WINDOW" "while true; do
    MUJOCO_GL=egl UV_PROJECT_ENVIRONMENT=/mnt/shared/<user>/python_envs/forge PYTHONHASHSEED=42 \
        CAP_NVIDIA_MAX_CONCURRENT_PER_KEY=1 \
        CAP_NVIDIA_REQUEST_DELAY_S=1.0 \
        CAP_NVIDIA_ACQUIRE_TIMEOUT_S=300 \
        CAP_NVIDIA_429_COOLDOWN_S=30 \
        CAP_NVIDIA_PROVIDER_URL=$PROVIDER_URL \
        uv run --no-sync run_agent.py \
        experiment=pnp_counter_to_stove \
        execution.n_seeds=40 execution.mode=parallel \
        wandb.enabled=true max_iterations=100 execution.record=true \
        skill_library.enabled=true agent=skill_library llm=nvidia reflection=nvidia_gemini \
        execution.seeds_per_gpu=2 oracle_api=true skill_reflection.enabled=true \
        llm.model=azure/anthropic/claude-opus-4-7
    echo '[restart] exited \$?, restarting in 5s...'
    sleep 5
done" Enter
echo "Launched window '$WINDOW' in session '$SESSION'"
echo "Attach: tmux attach -t $SESSION"
