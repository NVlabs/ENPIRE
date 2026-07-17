"""SmolVLM (locally-served via vLLM, OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.vlm.transport import register
from enpire.env.forge.cap.config import SMOL_VLM_MODEL, SMOL_VLM_URL
from enpire.env.forge.cap.utils.image import encode_image_b64


def _query_smolvlm(
    text: str,
    images: list[np.ndarray],
    vlm_url: str,
    model: str,
    temperature: float = 0.2,
) -> str:
    """Low-level function used by legacy callers. Kept stable."""
    import json
    import urllib.request

    content: list[dict] = []
    for img in images:
        b64 = encode_image_b64(img)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    content.append({"type": "text", "text": text})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{vlm_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


class SmolVlmBackend:
    """SmolVLM on vLLM (no API key)."""

    name = "smol_vlm"

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,
        vlm_url: str | None = None,
        api_url: str | None = None,  # alias
        **_kw: Any,
    ) -> str:
        url = vlm_url or api_url or SMOL_VLM_URL
        return _query_smolvlm(text, images, url, model or SMOL_VLM_MODEL, temperature)


register("smol_vlm", SmolVlmBackend)
