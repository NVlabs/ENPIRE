"""SmolVLM tool — vision-language model queries via vLLM."""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.request
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT, SMOL_VLM_MODEL, SMOL_VLM_URL

logger = logging.getLogger(__name__)


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode a numpy RGB image to base64 JPEG."""
    from PIL import Image

    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SmolVlmTool(Tool):
    """Query a vision-language model with an image and text prompt."""

    name = "smol_vlm"
    description = (
        "Query a vision-language model (SmolVLM) with a text prompt and "
        "optional camera image. Returns the model's text response. "
        "Use this to understand scene contents, identify objects, read text, "
        "or reason about what the robot sees."
    )
    parameters = [
        ToolParameter("text", "str", "Text prompt for the VLM"),
        ToolParameter(
            "camera",
            "str",
            'Camera name ("top", "left", "right"). Image is captured automatically.',
            required=False,
            default="top",
        ),
        ToolParameter(
            "image",
            "Any",
            "Optional numpy RGB array to use instead of capturing from camera.",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        vlm_url: str = SMOL_VLM_URL,
        vlm_model: str = SMOL_VLM_MODEL,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
    ):
        self._vlm_url = vlm_url.rstrip("/")
        self._vlm_model = vlm_model
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._portal_client = None

    def _get_cap_client(self):
        if self._portal_client is None:
            import portal
            self._portal_client = portal.Client(
                f"{self._cap_host}:{self._cap_port}"
            )
        return self._portal_client

    def _capture_image(self, camera: str) -> np.ndarray | None:
        """Capture an image from cap_server."""
        try:
            client = self._get_cap_client()
            img = client.get_camera_image(camera).result()
            img = np.asarray(img)
            if img.size < 100:  # Skip dummy 1x1 images
                return None
            return img
        except Exception as e:
            logger.warning("Failed to capture camera %s: %s", camera, e)
            return None

    def execute(self, **kwargs: Any) -> ToolResult:
        text: str = kwargs.get("text", "")
        camera: str = kwargs.get("camera", "top")
        image = kwargs.get("image", None)

        if not text:
            return ToolResult(success=False, error="text prompt is required")

        # Get image: use provided or capture from camera
        if image is None:
            image = self._capture_image(camera)

        # Build message content — image MUST come before text for SmolVLM
        # to attend to the instruction rather than free-describe the image.
        content: list[dict] = []
        if image is not None:
            b64 = _encode_image_b64(np.asarray(image))
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": text})

        # Call vLLM OpenAI-compatible API
        payload = {
            "model": self._vlm_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.2,
        }

        try:
            req = urllib.request.Request(
                f"{self._vlm_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())

            response_text = result["choices"][0]["message"]["content"]
            logger.info("SmolVLM response: %s", response_text[:200])
            return ToolResult(success=True, data=response_text)

        except Exception as e:
            logger.error("SmolVLM query failed: %s", e)
            return ToolResult(success=False, error=f"VLM query failed: {e}")
