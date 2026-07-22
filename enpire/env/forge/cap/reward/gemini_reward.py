# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward function: uses Gemini VLM to judge whether a task has been completed.

Sends all available camera images (top, left, right) plus the task description
to the Gemini API and asks for a binary YES/NO success judgment.

Returns 1.0 if the model says the task is complete, 0.0 otherwise.

Because the RL loop calls get_reward at POLICY_FREQ_HZ but Gemini has strict rate limits,
this module caches the last reward and only queries the API every
GEMINI_REWARD_INTERVAL_S seconds (default 2.0, configurable via env var).

Requires:
    - GEMINI_API_KEY environment variable set
    - google-genai package installed (pip install google-genai)

Usage:
    export GEMINI_API_KEY="..."
    python -m cap.reward.reward_server --mode gemini
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from enpire.env.forge.cap.utils.image import encode_image_jpeg
from enpire.env.forge.cap.utils.prompt_loader import load_reward_prompt

logger = logging.getLogger(__name__)

_client: Any = None
_CAMERA_KEYS = ("top_camera_image", "left_camera_image", "right_camera_image")

# gemini-2.5-flash
GEMINI_MODEL = os.environ.get("GEMINI_REWARD_MODEL", "gemini-2.5-flash")
GEMINI_REWARD_INTERVAL_S = float(os.environ.get("GEMINI_REWARD_INTERVAL_S", "5.0"))

_cached_reward: float = 0.0
_last_call_time: float = 0.0
_reward_prompt: str | None = None


def _get_reward_prompt() -> str:
    global _reward_prompt
    if _reward_prompt is None:
        _reward_prompt = load_reward_prompt("reward_gemini")
    return _reward_prompt


def _get_client():
    global _client
    if _client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before starting the reward server."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def vlm_reward(obs: dict) -> float:
    """Query Gemini with all camera images and return 1.0 (success) or 0.0 (failure).

    Rate-limited: only calls the API every GEMINI_REWARD_INTERVAL_S seconds.
    Returns the cached reward for intermediate steps.
    """
    global _cached_reward, _last_call_time
    from google.genai import types

    # Collect images — bail silently if none present (e.g. print_loop with empty dict)
    contents: list = []
    image_count = 0
    for cam_key in _CAMERA_KEYS:
        img = obs.get(cam_key)
        if img is None:
            continue
        img = np.asarray(img, dtype=np.uint8)
        if img.size < 100:
            continue
        jpeg_bytes = encode_image_jpeg(img)
        cam_label = cam_key.replace("_camera_image", "")
        contents.append(f"[{cam_label} camera]")
        contents.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
        image_count += 1

    if image_count == 0:
        return _cached_reward

    # Rate limit: return cached reward if called too soon
    now = time.monotonic()
    if now - _last_call_time < GEMINI_REWARD_INTERVAL_S:
        return _cached_reward

    task = obs.get("annotation.task", "complete the manipulation task")
    prompt = _get_reward_prompt().format(task=task)
    contents.append(prompt)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8,
            ),
        )
        _last_call_time = time.monotonic()
        answer = response.text.strip().upper()
        _cached_reward = 1.0 if answer.startswith("YES") else 0.0
        print(
            f"[gemini_reward] task={task[:60]!r} images={image_count} "
            f"response={answer!r} → reward={_cached_reward:.1f}",
            flush=True,
        )
        logger.info(
            "Gemini reward response: %r reward=%.1f (task: %s)",
            answer,
            _cached_reward,
            task[:80],
        )
        return _cached_reward

    except Exception as e:
        _last_call_time = time.monotonic()
        print(f"[gemini_reward] FAILED: {e}", flush=True)
        logger.error("Gemini reward query failed: %s", e)
        return _cached_reward
