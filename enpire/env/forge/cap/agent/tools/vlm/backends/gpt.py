# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI GPT-5 vision backend (Responses API)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.vlm.transport import register
from enpire.env.forge.cap.config import GPT_VL_MODEL
from enpire.env.forge.cap.utils.image import encode_image_b64


def _query_gpt(
    text: str,
    images: list[np.ndarray],
    model: str,
    api_key: str,
    temperature: float = 0.2,
    reasoning_effort: str = "high",
) -> str:
    """Low-level function used by legacy callers. Kept stable."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=60.0)
    content: list[dict] = []
    for img in images:
        b64 = encode_image_b64(img)
        content.append(
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
        )
    content.append({"type": "input_text", "text": text})

    kwargs: dict = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 16384,
    }
    if reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        kwargs["temperature"] = temperature

    resp = client.responses.create(**kwargs)
    return resp.output_text


class GptBackend:
    """OpenAI GPT-5 via the Responses API. Requires ``OPENAI_API_KEY``."""

    name = "gpt"

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
        reasoning_effort: str = "high",
        **_kw: Any,
    ) -> str:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        return _query_gpt(
            text, images, model or GPT_VL_MODEL, key, temperature, reasoning_effort
        )


register("gpt", GptBackend)
