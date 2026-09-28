# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO-101 model path resolution."""

from __future__ import annotations

import os
from pathlib import Path

_SO101_DIR = Path(
    os.environ.get("ENPIRE_SO101_MODEL_ROOT", Path(__file__).resolve().parent)
).expanduser()


def get_so101_xml() -> Path:
    return _SO101_DIR / "so101.xml"
