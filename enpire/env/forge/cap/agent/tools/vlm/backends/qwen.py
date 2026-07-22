# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL (locally-served via vLLM, OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.vlm.transport import register
from enpire.env.forge.cap.config import QWEN_VL_MODEL, QWEN_VL_URL
from enpire.env.forge.cap.utils.image import encode_image_b64


def _query_qwen(
    text: str,
    images: list[np.ndarray],
    api_url: str,
    model: str,
    temperature: float = 0.2,
) -> str:
    """Low-level function used by legacy callers. Kept stable."""
    from openai import OpenAI

    client = OpenAI(base_url=api_url, api_key="EMPTY", timeout=30.0)
    content: list[dict] = []
    for img in images:
        b64 = encode_image_b64(img)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    content.append({"type": "text", "text": text})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content


class QwenBackend:
    """Qwen3-VL on vLLM (no API key — uses ``EMPTY``)."""

    name = "qwen"

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,
        api_url: str | None = None,
        **_kw: Any,
    ) -> str:
        return _query_qwen(
            text, images, api_url or QWEN_VL_URL, model or QWEN_VL_MODEL, temperature
        )


register("qwen", QwenBackend)
