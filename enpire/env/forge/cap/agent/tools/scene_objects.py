# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene object listing tool — uses Qwen3-VL to enumerate objects visible in the scene.

Captures the top camera image, sends it to the remote Qwen3-VL model, and
returns a structured list of object names suitable for feeding into
detect_object (SAM3 + BundleSDF) for 6-DOF pose estimation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT, QWEN_VL_MODEL, QWEN_VL_URL
from enpire.env.forge.cap.utils.image import encode_image_b64

logger = logging.getLogger(__name__)

# 默认提示词：要求返回 JSON 列表
_DEFAULT_PROMPT = (
    "List every distinct object on the table (exclude robot arms). "
    "Use short, specific names an object detector would understand "
    '(e.g. "red mug", "wooden block"). '
    "Return ONLY a JSON array of strings, nothing else."
)


class ListSceneObjectsTool(Tool):
    """Use Qwen3-VL to list all objects visible in the camera view.

    Captures an image from the specified camera (default: top), sends it to
    a remotely-served Qwen3-VL model, and parses the response into a list of
    object name strings.  These names can then be passed to ``detect_object``
    or ``track_object`` for 6-DOF pose estimation via SAM3 + BundleSDF.

    Example agent code::

        objects = list_scene_objects()           # ["red mug", "blue plate", ...]
        for obj in objects:
            detect_object(obj, backend="bundlesdf")
    """

    name = "list_scene_objects"
    description = (
        "Capture a camera image and use Qwen3-VL to list all visible objects. "
        "Returns a list of object name strings (e.g. ['red mug', 'blue plate']). "
        "Use these names with detect_object or track_object for pose estimation."
    )
    parameters = [
        ToolParameter(
            "camera",
            "str",
            'Camera to capture from: "top", "left", or "right".',
            required=False,
            default="top",
        ),
        ToolParameter(
            "prompt",
            "str",
            "Custom prompt to override the default object listing prompt.",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        qwen_url: str = QWEN_VL_URL,
        qwen_model: str = QWEN_VL_MODEL,
    ):
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._qwen_url = qwen_url
        self._qwen_model = qwen_model
        self._portal_client = None

    def _get_cap_client(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _capture_image(self, camera: str) -> np.ndarray | None:
        try:
            client = self._get_cap_client()
            img = client.get_camera_image(camera).result()
            img = np.asarray(img)
            if img.size < 100:
                return None
            return img
        except Exception as e:
            logger.warning("list_scene_objects: failed to capture camera %s: %s", camera, e)
            return None

    def _query_qwen(self, text: str, image: np.ndarray) -> str:
        """Send image + text to Qwen3-VL via vLLM OpenAI-compatible API."""
        from openai import OpenAI

        client = OpenAI(base_url=self._qwen_url, api_key="EMPTY", timeout=30.0)
        b64 = encode_image_b64(image)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": text},
        ]
        resp = client.chat.completions.create(
            model=self._qwen_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.1,  # 低温度，更确定性的输出
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content

    @staticmethod
    def _parse_object_list(raw: str) -> list[str]:
        """Parse the VLM response into a list of object name strings.

        Handles common response formats:
          - JSON array: ["red mug", "blue plate"]
          - Numbered list: 1. red mug  2. blue plate
          - Bullet list: - red mug  - blue plate
          - Comma-separated: red mug, blue plate
        """
        text = raw.strip()

        # 尝试直接解析 JSON
        # 先找到第一个 [ 和最后一个 ]，提取 JSON 子串
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                items = json.loads(text[start : end + 1])
                if isinstance(items, list) and all(isinstance(x, str) for x in items):
                    return [x.strip() for x in items if x.strip()]
            except json.JSONDecodeError:
                pass

        # 回退：按行解析
        lines = text.split("\n")
        objects = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去掉编号前缀 "1. ", "- ", "* "
            for prefix in ("- ", "* ", "• "):
                if line.startswith(prefix):
                    line = line[len(prefix) :]
                    break
            # 去掉 "1. ", "2. " 等编号
            if len(line) > 2 and line[0].isdigit() and line[1] in (".", ")"):
                line = line[2:].strip()
            elif len(line) > 3 and line[:2].isdigit() and line[2] in (".", ")"):
                line = line[3:].strip()
            if line:
                objects.append(line)

        # 如果只有一行，可能是逗号分隔的
        if len(objects) == 1 and "," in objects[0]:
            objects = [x.strip() for x in objects[0].split(",") if x.strip()]

        return objects

    def execute(self, **kwargs: Any) -> ToolResult:
        camera: str = kwargs.get("camera", "top")
        prompt: str | None = kwargs.get("prompt", None)

        if camera not in ("top", "left", "right"):
            return ToolResult(
                success=False,
                error=f"Unknown camera {camera!r}. Choose: top, left, right",
            )

        # 采集图像
        image = self._capture_image(camera)
        if image is None:
            return ToolResult(
                success=False,
                error=f"Failed to capture image from camera:{camera}",
            )

        # 查询 Qwen3-VL
        query_text = prompt or _DEFAULT_PROMPT
        try:
            raw_response = self._query_qwen(query_text, image)
        except Exception as e:
            logger.error("list_scene_objects: Qwen query failed: %s", e)
            return ToolResult(
                success=False,
                error=f"Qwen3-VL query failed: {e}",
            )

        # 解析返回结果
        objects = self._parse_object_list(raw_response)
        if not objects:
            logger.warning(
                "list_scene_objects: no objects parsed from response: %s", raw_response
            )
            return ToolResult(
                success=True,
                data={"objects": [], "raw_response": raw_response},
            )

        logger.info(
            "list_scene_objects [camera=%s]: found %d objects: %s",
            camera,
            len(objects),
            objects,
        )
        return ToolResult(
            success=True,
            data={"objects": objects, "raw_response": raw_response},
        )
