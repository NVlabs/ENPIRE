# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")


def extract_speakable_text(text: str) -> str:
    """Convert markdown-ish assistant output into something reasonable to speak."""
    if not text:
        return ""

    cleaned = _FENCED_CODE_RE.sub(" ", text)
    cleaned = _LINK_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _HEADER_RE.sub("", cleaned)
    cleaned = _BULLET_RE.sub("", cleaned)
    cleaned = _NUMBERED_RE.sub("", cleaned)
    cleaned = cleaned.replace("|", " ")
    cleaned = cleaned.replace("—", " ")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("*", "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned
