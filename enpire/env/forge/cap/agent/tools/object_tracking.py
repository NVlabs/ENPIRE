# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-time 6-DOF object tracking via BundleSDF.

Wraps the ``tools/vision/serve_bundlesdf.py`` HTTP API so that the CAP agent (or
LLM-generated code) can start/stop tracking and read live object poses.

Coordinate frame contract
-------------------------
All poses returned by these tools are in the **robot world frame**:

    +X  forward  (toward the work table)
    +Y  left     (toward the left arm)
    +Z  up       (sky)
    Origin: URDF base_link (floor level, centred between the two arm bases)

The camera→world transform is performed **inside this tool**, using
cap_server's camera extrinsics (Pinocchio FK), so the result is correct
regardless of whether tools/vision/serve_bundlesdf.py's own ``ob_in_world`` is valid.

Usage from generated code::

    track_object("red cup", camera="top")
    pose = get_object_pose()  # Detection3D, world frame
    stop_tracking()
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Any

import numpy as np
import portal
import requests
from scipy.spatial.transform import Rotation

from enpire.env.forge.cap.config import (
    BUNDLESDF_SERVER_HOST,
    BUNDLESDF_SERVER_PORT,
    CAP_SERVER_PORT,
    make_bundlesdf_name,
)
from enpire.env.forge.cap.agent.tools.base import Detection3D, Tool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared tracking context — set by TrackObjectTool, read by GetObjectPoseTool
# ---------------------------------------------------------------------------


class _TrackingContext:
    """Mutable state shared across the three tracking tools."""

    def __init__(self) -> None:
        self.camera: str = "top"
        self.query: str = ""
        self.name: str = ""  # session key for multi-object API


# ---------------------------------------------------------------------------
# Camera-frame → world-frame conversion
# ---------------------------------------------------------------------------


def _build_cam_to_world(extrinsics: dict, camera: str = "top") -> np.ndarray:
    """Build 4×4 SE3 that maps **OpenCV camera frame → robot world frame**.

    Extrinsics from cap_server are in Pinocchio convention (+X left, +Y up,
    +Z forward).  When ``needs_optical_flip`` is set, a diag(-1,-1,1) flip
    converts to OpenCV (+X right, +Y down, +Z forward).
    """
    R = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsics["position"], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    if extrinsics.get("needs_optical_flip", True):
        F = np.diag([-1.0, -1.0, 1.0])
        T[:3, :3] = R @ F
    else:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def _se3_to_pos_quat(T: np.ndarray) -> tuple[list[float], list[float]]:
    """Extract position [x,y,z] and quaternion [x,y,z,w] from 4×4 SE3."""
    pos = T[:3, 3].tolist()
    quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()  # xyzw
    return (
        [round(x, 5) for x in pos],
        [round(x, 6) for x in quat],
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TrackObjectTool(Tool):
    """Start real-time 6-DOF object tracking via BundleSDF.

    Sends a text query to the BundleSDF server which detects the object
    with SAM3, initialises a SAM2 tracking session, and begins continuous
    6-DOF pose estimation.  Stops any previously active session first.
    """

    name = "track_object"
    description = (
        "Start real-time 6-DOF object tracking via BundleSDF. "
        "Provide a text description of the object; the server detects it "
        "in the camera view and begins continuous pose tracking. "
        "Use get_object_pose() afterwards to read the latest pose (world frame). "
        "Stops any previously running tracking session."
    )
    parameters = [
        ToolParameter(
            "query", "str", "Text description of the object to track (e.g. 'red cup')."
        ),
        ToolParameter(
            "camera",
            "str",
            "Camera to use: 'top', 'left', or 'right'.",
            required=False,
            default="top",
        ),
    ]

    def __init__(
        self,
        bundlesdf_host: str = BUNDLESDF_SERVER_HOST,
        bundlesdf_port: int = BUNDLESDF_SERVER_PORT,
        context: _TrackingContext | None = None,
        timeout: float = 120.0,
    ):
        self._base_url = f"http://{bundlesdf_host}:{bundlesdf_port}"
        self._timeout = timeout
        self._ctx = context or _TrackingContext()

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        camera: str = kwargs.get("camera", "top")
        name = make_bundlesdf_name(query)
        encoded = urllib.parse.quote(name, safe="")
        try:
            # Stop any existing session first
            if self._ctx.name:
                old_encoded = urllib.parse.quote(self._ctx.name, safe="")
                requests.post(
                    f"{self._base_url}/end_detection/{old_encoded}",
                    timeout=30,
                )

            resp = requests.post(
                f"{self._base_url}/add_detection",
                json={"text": query, "camera": camera},
                timeout=self._timeout,
            )
            if resp.status_code not in (200, 204, 409):
                resp.raise_for_status()
            data = resp.json()

            # Record which camera/name is active so get_object_pose can fetch
            # the correct extrinsics for the cam→world transform.
            self._ctx.camera = camera
            self._ctx.query = query
            self._ctx.name = name

            # Wait briefly for the tracker to converge on the first few frames
            time.sleep(1.0)

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "camera": camera,
                    "name": name,
                    "bbox": data.get("bbox", []),
                    "first_score": data.get("first_score", 0.0),
                },
            )

        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._base_url}. "
                "Is tools/vision/serve_bundlesdf.py running?",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetObjectPoseTool(Tool):
    """Read the latest 6-DOF pose from the active BundleSDF tracking session.

    The camera→world transform is performed **here**, using cap_server's
    camera extrinsics (Pinocchio FK) as the single source of truth.
    This guarantees the returned pose is in the robot world frame
    (+x forward, +y left, +z up) regardless of any server-side issues.
    """

    name = "get_object_pose"
    description = (
        "Get the latest 6-DOF pose of the currently tracked object. "
        "Returns position_3d [x,y,z] and quaternion_xyzw [x,y,z,w] in the "
        "robot world frame (+x forward, +y left, +z up). "
        "Requires track_object() to have been called first."
    )
    parameters: list[ToolParameter] = []

    def __init__(
        self,
        bundlesdf_host: str = BUNDLESDF_SERVER_HOST,
        bundlesdf_port: int = BUNDLESDF_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        context: _TrackingContext | None = None,
        timeout: float = 10.0,
    ):
        self._base_url = f"http://{bundlesdf_host}:{bundlesdf_port}"
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._timeout = timeout
        self._ctx = context or _TrackingContext()
        self._portal_client: portal.Client | None = None

    def _get_portal(self) -> portal.Client:
        if self._portal_client is None:
            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if not self._ctx.name:
                return ToolResult(
                    success=False,
                    error="No active tracking session. Call track_object() first.",
                )

            # 1. Read raw pose from BundleSDF server
            encoded = urllib.parse.quote(self._ctx.name, safe="")
            resp = requests.get(
                f"{self._base_url}/get_detection/{encoded}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("tracking"):
                return ToolResult(
                    success=False,
                    error="No active tracking session. Call track_object() first.",
                )

            ob_in_cam_raw = data.get("ob_in_cam")
            if ob_in_cam_raw is None:
                return ToolResult(
                    success=False,
                    error="Pose not yet available (tracker may still be converging).",
                )

            # 2. Get camera extrinsics from cap_server (source of truth)
            camera = self._ctx.camera
            client = self._get_portal()
            extrinsics = client.get_camera_extrinsics(camera).result()

            # 3. Transform ob_in_cam → ob_in_world (tool-side, authoritative)
            ob_in_cam = np.asarray(ob_in_cam_raw, dtype=np.float64)
            T_cam_to_world = _build_cam_to_world(extrinsics, camera)
            ob_in_world = T_cam_to_world @ ob_in_cam

            pos_world, quat_world = _se3_to_pos_quat(ob_in_world)

            # 4. Sanity check: objects on a table should have z > 0
            if pos_world[2] < 0:
                logger.warning(
                    "Object z=%.3f is below floor — coordinate frame may be wrong",
                    pos_world[2],
                )

            det = Detection3D(
                label=self._ctx.query or "tracked_object",
                score=data.get("score", 0.0),
                box_2d=[],
                position_3d=pos_world,
                quaternion_xyzw=quat_world,
                half_extents=data.get("half_extents") or [],
            )
            return ToolResult(success=True, data=det)

        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._base_url}",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class StopTrackingTool(Tool):
    """Stop the active BundleSDF tracking session and free GPU memory."""

    name = "stop_tracking"
    description = (
        "Stop the active BundleSDF tracking session and release GPU resources."
    )
    parameters: list[ToolParameter] = []

    def __init__(
        self,
        bundlesdf_host: str = BUNDLESDF_SERVER_HOST,
        bundlesdf_port: int = BUNDLESDF_SERVER_PORT,
        context: _TrackingContext | None = None,
        timeout: float = 30.0,
    ):
        self._base_url = f"http://{bundlesdf_host}:{bundlesdf_port}"
        self._timeout = timeout
        self._ctx = context or _TrackingContext()

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if not self._ctx.name:
                return ToolResult(success=True, data=True)

            encoded = urllib.parse.quote(self._ctx.name, safe="")
            resp = requests.post(
                f"{self._base_url}/end_detection/{encoded}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            self._ctx.query = ""
            self._ctx.name = ""
            return ToolResult(success=True, data=True)
        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._base_url}",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
