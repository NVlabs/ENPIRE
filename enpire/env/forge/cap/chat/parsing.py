# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_BLOCK_RE = re.compile(
    r"```(?P<language>[a-zA-Z0-9_+-]*)\s*\n(?P<code>.*?)```",
    re.DOTALL,
)


@dataclass(frozen=True)
class CodeBlock:
    language: str
    code: str


def extract_code_blocks(text: str) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    for match in _CODE_BLOCK_RE.finditer(text):
        code = match.group("code").strip()
        if not code:
            continue
        language = (match.group("language") or "python").strip() or "python"
        blocks.append(CodeBlock(language=language, code=code))
    return blocks
