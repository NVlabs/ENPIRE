"""Atomic tool to save an image from any media source to a local file.

Media sources use the same prefix convention as vlm_query:
  camera:top / camera:left / camera:right  — live camera capture
  local:~/path/to/image.png                — copy from local filesystem
  web:https://example.com/image.png        — download from URL

Path resolution:
  ~/...  or /...  — absolute / home-relative
  other           — relative to project root
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path_str: str) -> Path:
    """Resolve a path string: ~/... expands home, /... is absolute, else project-relative."""
    p = Path(path_str)
    if path_str.startswith("~"):
        return p.expanduser().resolve()
    if p.is_absolute():
        return p.resolve()
    return (_PROJECT_ROOT / p).resolve()


class SaveImageTool(Tool):
    """Save an image from a camera, local file, or URL to a local file.

    Examples::

        save_image(media="camera:left", path="cap/tasks/peg_insertion/")
        save_image(media="camera:top", path="cap/tasks/demo/snapshot.png")
        save_image(media="web:https://example.com/ref.png", path="~/downloads/")
    """

    name = "save_image"
    description = (
        "Save an image from a media source to a local file. "
        'Media prefixes: "camera:<name>" (top/left/right), '
        '"local:<path>" (copy local file), "web:<url>" (download). '
        "If path is a directory, auto-generates a timestamped filename. "
        "Relative paths resolve against project root."
    )
    parameters = [
        ToolParameter(
            "media",
            "str",
            'Image source: "camera:top", "camera:left", "camera:right", '
            '"local:<path>", or "web:<url>".',
        ),
        ToolParameter(
            "path",
            "str",
            "Destination file path or directory. If a directory, filename is "
            "auto-generated as {source}_{timestamp}.png. Relative paths resolve "
            "against the project root.",
        ),
        ToolParameter(
            "filename",
            "str",
            "Optional filename override (used only when path is a directory).",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
    ):
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._portal_client = None

    def _get_cap_client(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _capture_camera(self, camera: str) -> np.ndarray:
        client = self._get_cap_client()
        img = client.get_camera_image(camera).result()
        img = np.asarray(img)
        if img.size < 100:
            raise RuntimeError(f"Empty image from camera:{camera}")
        return img

    def _load_media(self, media: str) -> np.ndarray:
        """Load a single media source into a numpy RGB array."""
        import cv2

        if media.startswith("camera:"):
            cam = media[len("camera:"):]
            if cam not in ("top", "left", "right"):
                raise ValueError(f"Unknown camera: {cam!r}")
            return self._capture_camera(cam)

        elif media.startswith("local:"):
            file_path = _resolve_path(media[len("local:"):])
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Failed to decode image: {file_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif media.startswith("web:"):
            import urllib.request

            url = media[len("web:"):]
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = np.frombuffer(resp.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from URL: {url}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            raise ValueError(
                f"Unknown media prefix in {media!r}. Use camera:, local:, or web:"
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        import cv2

        media: str = kwargs.get("media", "")
        path_str: str = kwargs.get("path", "")
        filename: str | None = kwargs.get("filename", None)

        if not media:
            return ToolResult(success=False, error="media is required")
        if not path_str:
            return ToolResult(success=False, error="path is required")

        try:
            img = self._load_media(media)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to load {media!r}: {e}")

        # Resolve destination path
        dest = _resolve_path(path_str)

        # If path looks like a directory (no image extension), treat as directory
        if dest.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            dest.mkdir(parents=True, exist_ok=True)
            if filename:
                dest = dest / filename
            else:
                # Auto-generate: {source_label}_{timestamp}.png
                source_label = media.replace(":", "_").replace("/", "_")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = dest / f"{source_label}_{ts}.png"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

        # Save as BGR for cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest), bgr)

        logger.info("save_image: %s -> %s", media, dest)
        return ToolResult(success=True, data=str(dest))
