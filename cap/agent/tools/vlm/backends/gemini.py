"""Google Gemini (direct) and Gemini Pro backends.

Both hit ``generativelanguage.googleapis.com`` via the ``google-genai`` SDK.
Auth: ``GEMINI_API_KEY`` env var.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from cap.agent.tools.vlm.transport import register
from cap.config import GEMINI_PRO_VL_MODEL, GEMINI_VL_MODEL
from cap.utils.image import encode_image_jpeg


def _query_gemini(
    text: str,
    images: list[np.ndarray],
    model: str,
    api_key: str,
    temperature: float = 0.2,
) -> str:
    """Low-level function used by legacy callers (scripts/, benchmarks). Kept stable."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents: list = []
    for img in images:
        jpeg_bytes = encode_image_jpeg(img)
        contents.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
    contents.append(text)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=2048,
        ),
    )
    return response.text


def _query_gemini_pro(
    text: str,
    images: list[np.ndarray],
    model: str,
    api_key: str,
    thinking_budget: int = 8192,
) -> str:
    """Gemini Pro with thinking budget."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents: list = []
    for img in images:
        jpeg_bytes = encode_image_jpeg(img)
        contents.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
    contents.append(text)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=16384,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return response.text


class GeminiBackend:
    """google-genai → Gemini (non-thinking). Default model: ``gemini-2.5-flash``."""

    name = "gemini"

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
        **_kw: Any,
    ) -> str:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        return _query_gemini(
            text, images, model or GEMINI_VL_MODEL, key, temperature
        )


class GeminiProBackend:
    """Gemini Pro with thinking. Maps ``reasoning_effort`` → ``thinking_budget``."""

    name = "gemini_pro"

    _EFFORT_TO_BUDGET = {"none": 0, "low": 2048, "medium": 8192, "high": 16384}

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,  # unused; Pro uses thinking budget
        api_key: str | None = None,
        reasoning_effort: str = "high",
        thinking_budget: int | None = None,
        **_kw: Any,
    ) -> str:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        budget = (
            thinking_budget
            if thinking_budget is not None
            else self._EFFORT_TO_BUDGET.get(reasoning_effort, 8192)
        )
        return _query_gemini_pro(
            text, images, model or GEMINI_PRO_VL_MODEL, key, budget
        )


register("gemini", GeminiBackend)
register("gemini_pro", GeminiProBackend)
