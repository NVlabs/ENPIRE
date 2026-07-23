# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BundleSDF multi-object tracking tools wrapping tools/vision/serve_bundlesdf.py via HTTP.

Four tools map directly to the new multi-session API:
  add_detection     → POST /add_detection
  get_detection     → GET  /get_detection/{name}
  end_detection     → POST /end_detection/{name}
  list_detections   → GET  /list_detections
"""

from __future__ import annotations

import urllib.parse
from typing import Any

import requests

from enpire.env.forge.cap.agent.tools.base import Detection3D, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import BUNDLESDF_SERVER_PORT, make_bundlesdf_name


class _BundleSdfBase(Tool):
    """Shared constructor for all BundleSDF tools."""

    parameters: list[ToolParameter] = []

    def __init__(
        self,
        bundlesdf_host: str = "localhost",
        bundlesdf_port: int = BUNDLESDF_SERVER_PORT,
        timeout: float = 30.0,
    ):
        self._base_url = f"http://{bundlesdf_host}:{bundlesdf_port}"
        self._timeout = timeout

    def _conn_err(self) -> ToolResult:
        return ToolResult(
            success=False,
            error=f"Cannot reach BundleSDF server at {self._base_url}",
        )


class AddDetectionTool(_BundleSdfBase):
    """Start 6-DOF pose tracking for a named object on a camera.

    Returns immediately — use get_detection() to poll for the pose.
    """

    name = "add_detection"
    description = (
        "Start 6-DOF pose tracking for an object described in natural language. "
        "Returns immediately (async). Use get_detection() to poll the pose once ready."
    )
    parameters = [
        ToolParameter("object", "str", "Natural-language description of the object to track."),
        ToolParameter(
            "camera", "str", "Camera: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
        ToolParameter(
            "name", "str",
            "Short key for this session (optional; defaults to URL-safe object text).",
            required=False, default=None,
        ),
    ]

    def __init__(
        self,
        bundlesdf_host: str = "localhost",
        bundlesdf_port: int = BUNDLESDF_SERVER_PORT,
        timeout: float = 120.0,  # SAM3 + model load can take ~60 s
    ):
        super().__init__(bundlesdf_host, bundlesdf_port, timeout)

    def execute(self, **kwargs: Any) -> ToolResult:
        obj: str = kwargs["object"]
        camera: str = kwargs.get("camera", "top")
        name: str | None = kwargs.get("name")
        try:
            resp = requests.post(
                f"{self._base_url}/add_detection",
                json={"text": obj, "camera": camera, "name": name},
                timeout=self._timeout,
            )
            if resp.status_code == 409:
                # Already active — not an error, just return the existing name
                resolved = make_bundlesdf_name(obj, name)
                return ToolResult(success=True, data=resolved)
            resp.raise_for_status()
            data = resp.json()
            return ToolResult(success=True, data=data["name"])
        except requests.ConnectionError:
            return self._conn_err()
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetDetectionTool(_BundleSdfBase):
    """Get the latest 6-DOF pose for a tracked object."""

    name = "get_detection"
    description = (
        "Get the latest 6-DOF pose for an object currently tracked by BundleSDF. "
        "Returns a Detection3D with position_3d and quaternion_xyzw."
    )
    parameters = [
        ToolParameter(
            "object", "str",
            "Object description or session name used in add_detection.",
        ),
        ToolParameter(
            "name", "str",
            "Explicit session key if you used a custom name in add_detection.",
            required=False, default=None,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        obj: str = kwargs["object"]
        explicit_name: str | None = kwargs.get("name")
        session_name = make_bundlesdf_name(obj, explicit_name)
        try:
            resp = requests.get(
                f"{self._base_url}/get_detection/{urllib.parse.quote(session_name, safe='')}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("tracking") or data.get("position_3d") is None:
                return ToolResult(success=False, error="Pose not available yet — call again shortly")
            det = Detection3D(
                label=obj,
                score=data.get("score", 0.0),
                box_2d=[],
                position_3d=data["position_3d"],
                quaternion_xyzw=data.get("quaternion_xyzw", []),
                half_extents=data.get("half_extents") or [],
            )
            return ToolResult(success=True, data=[det])
        except requests.ConnectionError:
            return self._conn_err()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return ToolResult(
                    success=False,
                    error=f"No active detection '{session_name}' — call add_detection first",
                )
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class EndDetectionTool(_BundleSdfBase):
    """Stop tracking a specific object and free its GPU resources."""

    name = "end_detection"
    description = (
        "Stop BundleSDF tracking for a specific object and release its GPU memory. "
        "Other active detections on the same camera continue unaffected."
    )
    parameters = [
        ToolParameter(
            "object", "str",
            "Object description or session name used in add_detection.",
        ),
        ToolParameter(
            "name", "str",
            "Explicit session key if you used a custom name in add_detection.",
            required=False, default=None,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        obj: str = kwargs["object"]
        explicit_name: str | None = kwargs.get("name")
        session_name = make_bundlesdf_name(obj, explicit_name)
        try:
            resp = requests.post(
                f"{self._base_url}/end_detection/{urllib.parse.quote(session_name, safe='')}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return ToolResult(success=True, data=True)
        except requests.ConnectionError:
            return self._conn_err()
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListDetectionsTool(_BundleSdfBase):
    """List all active BundleSDF tracking sessions with current poses."""

    name = "list_detections"
    description = (
        "List all active BundleSDF object tracking sessions with their current "
        "6-DOF poses, cameras, scores, and frame counts."
    )
    parameters = []

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            resp = requests.get(
                f"{self._base_url}/list_detections",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return ToolResult(success=True, data=data.get("detections", {}))
        except requests.ConnectionError:
            return self._conn_err()
        except Exception as e:
            return ToolResult(success=False, error=str(e))
