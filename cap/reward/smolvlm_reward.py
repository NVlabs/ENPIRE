"""Reward function: uses locally served SmolVLM to judge task completion.

Sends camera images plus the task description to a vLLM-hosted SmolVLM model
(OpenAI-compatible API) and asks for a binary YES/NO success judgment.

Returns 1.0 if the model says the task is complete, 0.0 otherwise.

Unlike the Gemini reward, SmolVLM runs locally on the lab GPU node with no
API quotas, so SMOLVLM_REWARD_INTERVAL_S can be set much lower (default 0.5s).

Requires:
    - vLLM server running SmolVLM (see SMOL_VLM_URL in cap/config.py)
    - No API key needed

Usage:
    python -m cap.reward.reward_server --mode smolvlm

Environment variables:
    SMOL_VLM_URL              vLLM base URL  (default: http://172.26.34.251:8401/v1)
    SMOLVLM_REWARD_INTERVAL_S seconds between API calls (default: 0.5)
    SMOLVLM_REWARD_CAMERAS    comma-separated camera list (default: top,left,right)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any

import numpy as np

from cap.config import SMOL_VLM_MODEL, SMOL_VLM_URL
from cap.utils.image import encode_image_b64
from cap.utils.prompt_loader import load_reward_prompt

logger = logging.getLogger(__name__)

_CAMERA_KEYS = ("top_camera_image", "left_camera_image", "right_camera_image")
SMOLVLM_REWARD_INTERVAL_S = float(os.environ.get("SMOLVLM_REWARD_INTERVAL_S", "0.5"))

_cached_reward: float = 0.0
_last_call_time: float = 0.0
_reward_prompt: str | None = None


def _get_reward_prompt() -> str:
    """Lazy-load prompt template on first use to avoid import-time file I/O."""
    global _reward_prompt
    if _reward_prompt is None:
        _reward_prompt = load_reward_prompt("reward_smolvlm")
    return _reward_prompt


# Positive/negative keyword sets used as a fallback when guided_choice
# is unavailable or the model still produces a longer response.
_POSITIVE_KEYWORDS = {
    "yes",
    "complete",
    "success",
    "done",
    "accomplished",
    "finished",
    "correct",
}
_NEGATIVE_KEYWORDS = {"no", "not", "fail", "incomplete", "hasn't", "incorrect", "wrong"}


def _parse_answer(answer: str) -> float:
    """Parse the model response to 1.0 (success) or 0.0 (failure).

    Checks for exact YES/NO first (expected when guided_choice is used),
    then falls back to keyword matching for models that ignore the constraint.
    """
    clean = answer.strip().upper()
    if clean == "YES":
        return 1.0
    if clean == "NO":
        return 0.0
    # Fallback: keyword scan over the full response
    words = set(clean.lower().split())
    if words & _POSITIVE_KEYWORDS and not words & _NEGATIVE_KEYWORDS:
        return 1.0
    return 0.0


def smolvlm_reward(obs: dict) -> float:
    """Query SmolVLM with camera images and return 1.0 (success) or 0.0 (failure).

    Rate-limited to SMOLVLM_REWARD_INTERVAL_S between calls.
    Returns the cached reward for intermediate steps.
    """
    global _cached_reward, _last_call_time

    # Collect images — bail silently if none present (e.g. print_loop with empty dict)
    images: list[np.ndarray] = []
    for cam_key in _CAMERA_KEYS:
        img = obs.get(cam_key)
        if img is None:
            continue
        img = np.asarray(img, dtype=np.uint8)
        if img.size < 100:
            continue
        images.append(img)

    if not images:
        return _cached_reward

    # Rate limit: return cached reward if called too soon
    now = time.monotonic()
    if now - _last_call_time < SMOLVLM_REWARD_INTERVAL_S:
        return _cached_reward
    task = obs.get("annotation.task", "complete the manipulation task")
    prompt = _get_reward_prompt().format(task=task)

    # Build OpenAI-compatible message: images first, then text instruction
    content: list[Any] = []
    for img in images:
        b64 = encode_image_b64(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": SMOL_VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
        "temperature": 0.0,
        "guided_choice": [
            "YES",
            "NO",
        ],  # Force the model to output exactly "YES" or "NO" via vLLM guided decoding.
    }

    try:
        req = urllib.request.Request(
            f"{SMOL_VLM_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())

        _last_call_time = time.monotonic()
        answer = result["choices"][0]["message"]["content"].strip().upper()
        _cached_reward = _parse_answer(answer)
        print(
            f"[smolvlm_reward] task={task[:60]!r} images={len(images)} "
            f"response={answer!r} → reward={_cached_reward:.1f}",
            flush=True,
        )
        logger.info(
            "SmolVLM reward response: %r reward=%.1f (task: %s)",
            answer,
            _cached_reward,
            task[:80],
        )
        return _cached_reward

    except Exception as e:
        _last_call_time = time.monotonic()
        print(f"[smolvlm_reward] FAILED: {e}", flush=True)
        logger.error("SmolVLM reward query failed: %s", e)
        return _cached_reward
