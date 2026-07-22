#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Block until all grootpool workers are ready (or timeout).
#
# Usage:
#   bash scripts/wait_for_grootpool.sh [ADMIN_URL] [TIMEOUT_SEC]
#
# Defaults:
#   ADMIN_URL     http://localhost:7071/status
#   TIMEOUT_SEC   600   (workers take ~2-3 min each to load N1.5 checkpoint)
#
# Exits 0 when every worker reports alive=true and idle_workers == n_workers.
# Exits 1 on timeout; exits 2 if the middleware admin endpoint never responds.

set -u

URL="${1:-http://localhost:7071/status}"
TIMEOUT="${2:-600}"

start=$(date +%s)
last=""
while :; do
    now=$(date +%s)
    if (( now - start > TIMEOUT )); then
        echo "timeout after ${TIMEOUT}s" >&2
        exit 1
    fi

    resp=$(curl -s --max-time 2 "$URL" 2>/dev/null || true)
    if [[ -z "$resp" ]]; then
        msg="middleware not responding…"
    else
        msg=$(printf '%s' "$resp" | python3 -c '
import json, sys
s = json.load(sys.stdin)
workers = s.get("workers", [])
up = sum(1 for w in workers if w.get("alive"))
total = len(workers)
idle = s.get("idle_workers", 0)
print(f"{up}/{total} workers alive, idle_workers={idle}")
if up == total and total > 0 and idle == total:
    sys.exit(0)
sys.exit(3)
' 2>/dev/null)
        rc=$?
        if [[ "$msg" != "$last" ]]; then
            echo "[$((now-start))s] $msg"
            last="$msg"
        fi
        if [[ $rc -eq 0 ]]; then
            exit 0
        fi
    fi
    sleep 3
done
