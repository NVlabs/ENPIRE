#!/usr/bin/env python3
"""Standalone wrapper for the AnyGrasp debug UI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vision.serve_anygrasp_debug import main


if __name__ == "__main__":
    main()
