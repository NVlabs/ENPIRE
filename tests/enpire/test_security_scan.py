# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
    ".claude",
    ".codex",
    "cap",
    "docs",
    "enpire",
    "experimental",
    "hardware",
    "rl",
    "robot",
    "scripts",
    "tmux",
    "tools",
)
ROOT_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "SECURITY.md", "pyproject.toml")
SECRET_PATTERNS = {
    "W&B API key": re.compile(r"wandb_v1_[A-Za-z0-9_-]+"),
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
}
PRIVATE_WORKSTATION_MARKERS = (
    "/home/gear/",
    "/home/lecar/",
    "/home/yiyang/",
)


def _text_files() -> list[Path]:
    paths = [ROOT / name for name in ROOT_FILES]
    for name in SCAN_ROOTS:
        root = ROOT / name
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part in {"__pycache__", ".venv", ".ruff_cache", "vendor", "logs"}
                for part in path.parts
            )
        )
    return paths


def test_release_sources_contain_no_secret_shapes_or_private_workstation_paths() -> None:
    findings: list[str] = []
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(ROOT)
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}: {label}")
        for marker in PRIVATE_WORKSTATION_MARKERS:
            if marker in text:
                findings.append(f"{relative}: private path {marker}")

    assert findings == []

