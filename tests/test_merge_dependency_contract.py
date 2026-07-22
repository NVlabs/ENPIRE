# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_voice_modules_exist_after_merge() -> None:
    repo_root = _repo_root()
    expected = [
        "cap/voice/__init__.py",
        "cap/voice/service.py",
        "cap/voice/voice_server.py",
        "cap/voice/chat_output.py",
        "cap/voice/output.py",
        "cap/voice/speakable_text.py",
    ]
    missing = [path for path in expected if not (repo_root / path).exists()]
    assert not missing, (
        "Merge incomplete: missing voice modules from voice-input/voice-output branches. "
        f"Missing files: {missing}"
    )


def test_pyproject_registers_realtimestt_source_after_merge() -> None:
    data = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    assert "RealtimeSTT" in uv_sources, (
        "Merge incomplete: pyproject.toml should include the vendored RealtimeSTT source "
        "from the voice-input branch."
    )
