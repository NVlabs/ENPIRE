#!/usr/bin/env bash
# Sync HIL gripper debug logs from lecar-yam-deploy to local Mac Mini.
# Usage: ./scripts/sync_grip_logs.sh
# Ctrl-C to stop.

REMOTE="lecar-yam-deploy"
REMOTE_DIR="/home/lecar/logs/"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs/remote/"

echo "Syncing $REMOTE:$REMOTE_DIR → $LOCAL_DIR  (every 2s, Ctrl-C to stop)"
mkdir -p "$LOCAL_DIR"

while true; do
    rsync -az --partial "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" 2>/dev/null
    sleep 2
done
