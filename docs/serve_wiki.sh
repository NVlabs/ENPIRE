#!/usr/bin/env bash
# Launch the YAM docs wiki server
# Usage: ./docs/serve_wiki.sh [port]
set -euo pipefail

PORT="${1:-7777}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "YAM Docs Wiki: http://0.0.0.0:${PORT}/"
exec python3 -m http.server "$PORT" --bind 0.0.0.0 -d "$DIR"
