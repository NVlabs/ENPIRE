# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical filesystem locations for the self-contained Forge runtime."""

from __future__ import annotations

from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent
ENV_ROOT = FORGE_ROOT.parent
PACKAGE_ROOT = ENV_ROOT.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DOCS_ROOT = ENV_ROOT / "docs"
EXAMPLES_ROOT = ENV_ROOT / "examples"
THIRD_PARTY_ROOT = REPOSITORY_ROOT / "third_party"


def forge_path(*parts: str) -> Path:
    """Return an absolute path inside the canonical Forge implementation."""

    return FORGE_ROOT.joinpath(*parts)


def third_party_path(*parts: str) -> Path:
    """Return an absolute path inside the repository's third-party sources."""

    return THIRD_PARTY_ROOT.joinpath(*parts)
