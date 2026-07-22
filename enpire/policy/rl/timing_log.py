# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from pathlib import Path


class TimingLogger:
    """Append-only JSONL event log for robot utilization analysis.

    Each log() call opens, writes, and closes the file atomically — no
    persistent file handle, no buffer accumulation, safe for multi-hour runs.
    """

    def __init__(self, output_dir: Path) -> None:
        self._path = Path(output_dir) / "timing_log.jsonl"

    def update_dir(self, output_dir: Path) -> None:
        self._path = Path(output_dir) / "timing_log.jsonl"

    def log(self, event: str, **meta) -> None:
        record = {"event": event, "t": time.time()}
        record.update(meta)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

