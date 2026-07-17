#!/usr/bin/env bash
# Sync HIL gripper debug logs from <robot-host> to local Mac Mini.
# Usage: ./scripts/sync_grip_logs.sh
# Ctrl-C to stop.

REMOTE="${GRIP_LOG_REMOTE:?Set GRIP_LOG_REMOTE to the robot SSH host}"
REMOTE_DIR="${GRIP_LOG_REMOTE_DIR:-logs/}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs/remote/"

echo "Syncing $REMOTE:$REMOTE_DIR → $LOCAL_DIR  (every 2s, Ctrl-C to stop)"
mkdir -p "$LOCAL_DIR"

while true; do
    rsync -az --partial "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" 2>/dev/null
    sleep 2
done
