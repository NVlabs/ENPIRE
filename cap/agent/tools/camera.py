"""Camera tools with Portal RPC and direct-env transports."""

from __future__ import annotations

from typing import Any

import numpy as np

from cap.agent.tools.base import Tool, ToolParameter, ToolResult
from cap.config import CAP_SERVER_PORT


def _intrinsics_payload(intrinsics: Any) -> dict[str, Any]:
    fx, fy, cx, cy = [
        float(x) for x in np.asarray(intrinsics, dtype=np.float64).reshape(-1)[:4]
    ]
    K = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "intrinsics": [fx, fy, cx, cy],
        "K": K.tolist(),
    }


def _extrinsics_payload(extrinsics: dict[str, Any]) -> dict[str, Any]:
    R = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsics["position"], dtype=np.float64).reshape(3)
    needs_optical_flip = bool(extrinsics.get("needs_optical_flip", True))

    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, :3] = R @ np.diag([-1.0, -1.0, 1.0]) if needs_optical_flip else R
    T_cam_world[:3, 3] = t

    payload = dict(extrinsics)
    payload["position"] = [float(x) for x in t.tolist()]
    payload["rotation"] = [[float(x) for x in row] for row in R.tolist()]
    payload["needs_optical_flip"] = needs_optical_flip
    payload["T_cam_world"] = T_cam_world.tolist()
    return payload


class CameraTransport:
    """Shared direct/Portal camera access helper."""

    def __init__(
        self,
        *,
        env: Any | None = None,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
    ) -> None:
        self._env = env
        self._host = host
        self._port = port
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import portal

            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    def render_rgb(self, camera: str) -> Any:
        if self._env is not None:
            return self._env.render_rgb(camera)
        return self._get_client().get_camera_image(camera).result()

    def render_depth(self, camera: str) -> Any:
        if self._env is not None:
            return self._env.render_depth(camera)
        return self._get_client().get_camera_depth(camera).result()

    def get_intrinsics(self, camera: str) -> dict[str, Any]:
        if self._env is not None:
            intrinsics = self._env.get_camera_intrinsics(camera)
        else:
            intrinsics = self._get_client().get_camera_intrinsics(camera).result()
        return _intrinsics_payload(intrinsics)

    def get_extrinsics(self, camera: str) -> dict[str, Any]:
        if self._env is not None:
            extrinsics = self._env.get_camera_extrinsics(camera)
        else:
            extrinsics = self._get_client().get_camera_extrinsics(camera).result()
        return _extrinsics_payload(extrinsics)


class _CameraTool(Tool):
    parameters = [
        ToolParameter(
            "camera",
            "str",
            "Camera name: 'top', 'left', or 'right'.",
            required=False,
            default="top",
        )
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ) -> None:
        self._camera = CameraTransport(env=env, host=host, port=port)


class GetCameraIntrinsicsTool(_CameraTool):
    name = "get_camera_intrinsics"
    description = "Get camera intrinsics as fx/fy/cx/cy plus a 3x3 K matrix."

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                data=self._camera.get_intrinsics(kwargs.get("camera", "top")),
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GetCameraExtrinsicsTool(_CameraTool):
    name = "get_camera_extrinsics"
    description = "Get camera extrinsics plus OpenCV camera-to-world transform."

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                data=self._camera.get_extrinsics(kwargs.get("camera", "top")),
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class RenderRgbTool(_CameraTool):
    name = "render_rgb"
    description = "Render or fetch the latest RGB image for a camera."

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                data=self._camera.render_rgb(kwargs.get("camera", "top")),
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class RenderDepthTool(_CameraTool):
    name = "render_depth"
    description = "Render or fetch the latest depth image for a camera."

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                data=self._camera.render_depth(kwargs.get("camera", "top")),
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
