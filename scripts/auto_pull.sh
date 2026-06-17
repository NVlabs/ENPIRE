#!/usr/bin/env bash
# Run `git pull` from the repo root every 5 seconds until interrupted.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"

cd "$PROJECT_DIR"

trap 'echo; echo "Stopped auto-pull loop."; exit 0' INT TERM

echo "Auto-pulling in $PROJECT_DIR every ${INTERVAL_SECONDS}s. Press Ctrl-C to stop."

while true; do
    printf '\n[%s] Running git pull...\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    if ! git pull; then
        echo "git pull failed; retrying in ${INTERVAL_SECONDS}s."
    fi
    sleep "$INTERVAL_SECONDS"
done
