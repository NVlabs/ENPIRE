#!/usr/bin/env bash
# Waits for run_script.py subprocesses to appear, sleeps past the env.reset
# window, then dumps every subprocess's Python stack with py-spy.
#
# Run this in a second terminal BEFORE launching run_agent.py.
#
# Usage:
#   bash scripts/dump_hung_seeds.sh [WAIT_SECONDS] [OUTPUT_FILE]
#
#   WAIT_SECONDS  seconds to wait after processes appear before dumping
#                 (default 45 — past the ~15s env.reset window)
#   OUTPUT_FILE   where to write dumps (default /tmp/pyspy_dump.txt)

WAIT="${1:-45}"
OUT="${2:-/tmp/pyspy_dump.txt}"

echo "[dump_hung_seeds] waiting for run_script.py processes..."
rm -f "$OUT"

while true; do
    pids=$(pgrep -f "run_script.py" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "[dump_hung_seeds] found PIDs: $pids — sleeping ${WAIT}s before dump"
        sleep "$WAIT"
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "=== PID $pid @ $(date '+%H:%M:%S') ===" | tee -a "$OUT"
                py-spy dump --pid "$pid" 2>&1 | tee -a "$OUT"
                echo "" | tee -a "$OUT"
            else
                echo "=== PID $pid already exited ===" | tee -a "$OUT"
            fi
        done
        echo "[dump_hung_seeds] done — output in $OUT"
        break
    fi
    sleep 2
done
