#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${KNN_DATABASE_PATH:?Set KNN_DATABASE_PATH to an external dataset directory}"

uv run run_data_collection_knn.py \
  --station="${KNN_STATION:-1}" \
  --display-image \
  --database-path="$KNN_DATABASE_PATH" \
  "$@"
