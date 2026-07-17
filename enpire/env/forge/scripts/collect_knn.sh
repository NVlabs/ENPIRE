#!/usr/bin/env bash
set -euo pipefail

: "${KNN_DATABASE_PATH:?Set KNN_DATABASE_PATH to an external dataset directory}"

uv run run_data_collection_knn.py \
  --station="${KNN_STATION:-1}" \
  --display-image \
  --database-path="$KNN_DATABASE_PATH" \
  "$@"
