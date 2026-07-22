# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load reward prompt templates from JSON files under cap/prompt/.

JSON schema:
    {
        "description": "Human-readable description",
        "backend":     "gemini" | "smolvlm" | ...,
        "template":    "Prompt text with {task} placeholder"
    }

Usage:
    from enpire.env.forge.cap.reward.prompt_loader import load_reward_prompt
    template = load_reward_prompt("reward_gemini")
    prompt = template.format(task="insert the peg into the hole")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent.parent / "prompt"


def load_reward_prompt(name: str) -> str:
    """Load and return the ``template`` field from ``cap/prompt/<name>.json``.

    Args:
        name: JSON file stem, e.g. ``"reward_gemini"`` for ``reward_gemini.json``.

    Returns:
        The raw template string (contains ``{task}`` placeholder).

    Raises:
        FileNotFoundError: if the JSON file does not exist.
        KeyError: if the JSON file has no ``"template"`` field.
    """
    path = _PROMPT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Reward prompt file not found: {path}\n"
            f"Available prompts: {[p.stem for p in _PROMPT_DIR.glob('*.json')]}"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "template" not in data:
        raise KeyError(f"Prompt file {path} has no 'template' field")
    return data["template"]
