#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT_DIR/tmp/voice_output_ui"
BRIDGE_LOG="$STATE_DIR/claude_bridge.log"
UI_LOG="$STATE_DIR/cap_ui.log"
BRIDGE_PID_FILE="$STATE_DIR/claude_bridge.pid"
UI_PID_FILE="$STATE_DIR/cap_ui.pid"
BRIDGE_URL="http://127.0.0.1:8201"
UI_URL="http://127.0.0.1:5173"

mkdir -p "$STATE_DIR"

port_pids() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

stop_port_listener() {
  local port="$1"
  local name="$2"
  local pids
  pids="$(port_pids "$port")"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "Stopping existing $name listener on port $port: $pids"
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(port_pids "$port")"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
  fi
}

is_listening() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket() as s:
    s.settimeout(0.5)
    try:
        s.connect((host, port))
    except OSError:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

wait_for_http() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ $name is ready at $url"
      return 0
    fi
    sleep 1
  done
  echo "✗ Timed out waiting for $name at $url" >&2
  return 1
}

start_bridge() {
  stop_port_listener 8201 "Claude bridge"
  echo "Starting Claude bridge..."
  (
    cd "$ROOT_DIR"
    uv sync --extra agent_voice >/dev/null
    exec .venv/bin/python -m cap.bridge.claude_bridge
  ) >"$BRIDGE_LOG" 2>&1 &
  echo $! >"$BRIDGE_PID_FILE"
  wait_for_http "$BRIDGE_URL/api/voice/status" "Claude bridge"
}

start_ui() {
  stop_port_listener 5173 "CAP UI"
  echo "Starting CAP UI dev server..."
  (
    cd "$ROOT_DIR/cap/ui"
    if [[ ! -d node_modules ]]; then
      npm ci >/dev/null
    fi
    exec npm run dev
  ) >"$UI_LOG" 2>&1 &
  echo $! >"$UI_PID_FILE"
  wait_for_http "$UI_URL" "CAP UI"
}

start_bridge
start_ui

cat <<EOF

Voice-output UI test is ready.

Open:
  $UI_URL

Then in the Chat panel click:
  Voice Test

Useful files:
  Bridge log: $BRIDGE_LOG
  UI log:     $UI_LOG

Direct curl test:
  curl -X POST $BRIDGE_URL/api/voice/test \\
    -H 'Content-Type: application/json' \\
    -d '{"text":"Hello World"}'

EOF
