# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 text-prompted segmentation tool.

Fetches camera image from cap_server via Portal RPC or loads from a local file,
sends it to the remote SAM3 server (tools/vision/serve_sam3.py), and returns the binary mask.

Media source prefixes (same convention as vlm_query):
  camera:top / camera:left / camera:right  — live camera capture
  local:path/to/image.png                  — load from local filesystem (~ expanded)
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import requests

from enpire.env.forge.cap.config import CAP_SERVER_PORT, CAMERA_NAMES as _CAMERA_NAMES, SAM3_SERVER_HOST, SAM3_SERVER_PORT
from enpire.env.forge.cap.agent.tools.base import SegmentationResult, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.agent.tools._artifact_log import log_mask
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SegmentObjectTool(Tool):
    """Segment an object from a camera image using SAM3 text-prompted segmentation."""

    name = "segment_object"
    description = (
        "Segment an object described in natural language from a camera image. "
        "Returns a binary mask, bounding box, confidence score, and mask area. "
        'Use media="camera:top" for live cameras or media="local:path/to/img.png" '
        "to segment from a local file."
    )
    parameters = [
        ToolParameter("query", "str", "Natural-language description of the object to segment."),
        ToolParameter(
            "media", "str",
            'Image source: "camera:<name>" (top/left/right) or '
            '"local:<path>" (local file, ~ expanded, relative to project root). '
            'Defaults to "camera:top" if omitted.',
            required=False, default=None,
        ),
        ToolParameter(
            "camera", "str",
            "[DEPRECATED — use media instead] Camera: 'top', 'left', or 'right'.",
            required=False, default=None,
        ),
        ToolParameter(
            "score_thresh", "float",
            "Minimum confidence score to accept the segmentation.",
            required=False, default=0.1,
        ),
    ]

    def __init__(
        self,
        sam3_host: str = SAM3_SERVER_HOST,
        sam3_port: int = SAM3_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        timeout: float = 30.0,
        env=None,
    ):
        self._env = env
        self._sam3_url = f"http://{sam3_host}:{sam3_port}"
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._timeout = timeout
        self._portal_client = None

    def _get_portal(self):
        if self._portal_client is None:
            import portal
            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _fetch_camera_image(self, camera: str) -> np.ndarray:
        if self._env is not None:
            rgb = self._env.render_rgb(camera)
        else:
            rgb = self._get_portal().get_camera_image(camera).result()
        arr = np.asarray(rgb)
        if arr.size < 100:
            raise ValueError(f"No image returned for camera {camera!r}")
        return arr

    def _load_image(self, media: str) -> np.ndarray:
        """Resolve a media source string to a numpy RGB array."""
        if media.startswith("camera:"):
            cam_name = media[len("camera:"):]
            if cam_name not in _CAMERA_NAMES:
                raise ValueError(f"Unknown camera: {cam_name!r}")
            return self._fetch_camera_image(cam_name)
        elif media.startswith("local:"):
            import cv2
            path_str = media[len("local:"):]
            p = Path(path_str)
            if path_str.startswith("~"):
                path = p.expanduser().resolve()
            elif p.is_absolute():
                path = p.resolve()
            else:
                path = (_PROJECT_ROOT / p).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to decode image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown media prefix in {media!r}. Use camera: or local:")

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        media: str | None = kwargs.get("media", None)
        camera: str | None = kwargs.get("camera", None)
        score_thresh: float = kwargs.get("score_thresh", 0.1)

        # Resolve media source
        if media is not None:
            source = media
        elif camera is not None:
            source = f"camera:{camera}"
        else:
            source = "camera:top"

        try:
            rgb = self._load_image(source)
        except Exception as e:
            return ToolResult(success=False, error=f"Image load error ({source}): {e}")
        camera_name = None
        if source.startswith("camera:"):
            camera_name = source[len("camera:"):]

        # Encode image for transport
        buf = io.BytesIO()
        np.save(buf, rgb)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        # Send to remote SAM3 server
        try:
            resp = requests.post(
                f"{self._sam3_url}/segment",
                json={"text": query, "image_b64": image_b64},
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach SAM3 server at {self._sam3_url}",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))

        data = resp.json()
        score = data["score"]
        if score < score_thresh:
            return ToolResult(
                success=False,
                error=f"Segmentation score {score:.3f} below threshold {score_thresh}",
            )

        mask_bytes = base64.b64decode(data["mask_b64"])
        mask = np.load(io.BytesIO(mask_bytes))

        result = SegmentationResult(
            mask=mask,
            bbox_xywh=data["bbox_xywh"],
            score=score,
            mask_area=data["mask_area"],
        )

        # Save mask overlay to log
        log_mask(rgb, mask, query=query, camera=camera_name)

        return ToolResult(success=True, data=result)


class SegmentAllObjectsTool(Tool):
    """Detect all instances of an object using SAM3 multi-detection."""

    name = "segment_all_objects"
    description = (
        "Detect all instances of an object described in natural language using "
        "SAM3 multi-detection. Returns a list of SegmentationResult, one per "
        "detection, sorted by confidence."
    )
    parameters = [
        ToolParameter("query", "str", "Natural-language description of the objects to segment."),
        ToolParameter(
            "camera",
            "str",
            "Camera to use: 'top', 'left', or 'right'.",
            required=False,
            default="top",
        ),
        ToolParameter(
            "score_thresh",
            "float",
            "Minimum confidence score for each detection.",
            required=False,
            default=0.1,
        ),
    ]

    def __init__(
        self,
        sam3_host: str = SAM3_SERVER_HOST,
        sam3_port: int = SAM3_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        timeout: float = 60.0,
        env=None,
    ):
        self._env = env
        self._sam3_url = f"http://{sam3_host}:{sam3_port}"
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._timeout = timeout
        self._portal_client = None

    def _get_portal(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _fetch_camera_image(self, camera: str) -> np.ndarray:
        if self._env is not None:
            rgb = self._env.render_rgb(camera)
        else:
            rgb = self._get_portal().get_camera_image(camera).result()
        arr = np.asarray(rgb)
        if arr.size < 100:
            raise ValueError(f"No image returned for camera {camera!r}")
        return arr

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        camera: str = kwargs.get("camera", "top")
        score_thresh: float = float(kwargs.get("score_thresh", 0.1))

        try:
            rgb = self._fetch_camera_image(camera)
        except Exception as e:
            return ToolResult(success=False, error=f"Camera error ({camera}): {e}")

        buf = io.BytesIO()
        np.save(buf, rgb)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            resp = requests.post(
                f"{self._sam3_url}/segment_all",
                json={
                    "text": query,
                    "image_b64": image_b64,
                    "score_threshold": score_thresh,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach SAM3 server at {self._sam3_url}",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))

        data = resp.json()
        results = []
        for det in data.get("detections", []):
            mask_bytes = base64.b64decode(det["mask_b64"])
            mask = np.load(io.BytesIO(mask_bytes))
            results.append(
                SegmentationResult(
                    mask=mask,
                    bbox_xywh=det["bbox_xywh"],
                    score=det["score"],
                    mask_area=det["mask_area"],
                )
            )

        if results:
            combined = np.zeros_like(np.asarray(results[0].mask), dtype=np.uint8)
            for seg in results:
                combined[np.asarray(seg.mask) > 0] = 1
            log_mask(rgb, combined, query=f"{query} ×{len(results)}", tag="segment_all")

        return ToolResult(success=True, data=results)


def match_mask_shape(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compare two binary masks via Hu moments. Returns distance (lower = more similar)."""
    import cv2
    return float(cv2.matchShapes(
        mask_a.astype(np.uint8), mask_b.astype(np.uint8),
        cv2.CONTOURS_MATCH_I1, 0,
    ))
