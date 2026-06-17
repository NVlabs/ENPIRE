"""Offline study-mode adapter.

This adapter intentionally creates no simulator and no hardware connection.  It
is for agent loops that iterate over saved data through a task harness while
still needing a small execution namespace, especially ``vlm_query``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cap.agent.robot_adapters.base import cfg_runtime_kwargs, cfg_select


class StudyEnv:
    """Minimal no-op env compatible with run_script backend hooks."""

    def render_rgb(self, camera: str) -> np.ndarray | None:
        _ = camera
        return None

    def render_depth(self, camera: str) -> np.ndarray | None:
        _ = camera
        return None

    def close(self) -> None:
        return None


class StudyAdapter:
    """Create an offline namespace for study/harness tasks."""

    offset_service_ports = False

    def __init__(self, config_group: str = "study") -> None:
        self.config_group = config_group

    def create_runtime(
        self,
        *,
        cfg: Any | None = None,
        runtime_role: str = "script",
        env_name: str | None = None,
        viewer: bool = False,
        vlm_backend: str = "gemini",
        seed: int | None = None,
        layout_id: int | None = None,
        style_id: int | None = None,
        camera_height: int | None = None,
        camera_width: int | None = None,
        curobo_host: str = "127.0.0.1",
        curobo_port: int = 0,
        mppi_host: str = "127.0.0.1",
        mppi_port: int = 0,
    ) -> tuple[Any, dict[str, Any]]:
        kwargs = cfg_runtime_kwargs(
            cfg,
            runtime_role=runtime_role,
            env_name=env_name,
            viewer=viewer,
            vlm_backend=vlm_backend,
            seed=seed,
            layout_id=layout_id,
            style_id=style_id,
            camera_height=camera_height,
            camera_width=camera_width,
            curobo_host=curobo_host,
            curobo_port=curobo_port,
            mppi_host=mppi_host,
            mppi_port=mppi_port,
        )

        from cap.agent.tool_handle import make_tool_runner_namespace
        from cap.agent.tools.base import SegmentationResult
        from cap.agent.tools.vlm import query as _vlm_query
        from cap.env.base.skill_library import (
            segment_all_objects_sam3,
            segment_object_sam3,
        )

        env = StudyEnv()

        def _service_url(name: str, default_port: int) -> str:
            host = str(cfg_select(cfg, f"runtime.{name}_host", "127.0.0.1"))
            port = int(cfg_select(cfg, f"runtime.{name}_port", default_port) or default_port)
            return f"http://{host}:{port}"

        def _sam3_url() -> str:
            return _service_url("sam3", 6767)

        def _load_local_rgb(path_like: str) -> np.ndarray:
            import cv2

            raw = path_like
            if raw.startswith("local:"):
                raw = raw[len("local:") :]
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def _resolve_study_image(
            *,
            image: Any = None,
            media: str | None = None,
            camera: str | None = None,
        ) -> np.ndarray:
            if image is not None:
                return np.asarray(image)
            if media:
                if media.startswith("local:"):
                    return _load_local_rgb(media)
                raise RuntimeError(
                    "study-mode segmentation only supports image=... arrays or "
                    "media='local:/path/to/image.png'; live camera media is disabled."
                )
            if camera:
                raise RuntimeError(
                    "study-mode segmentation has no live cameras; pass image=rgb or "
                    "media='local:/path/to/image.png'."
                )
            raise RuntimeError("study-mode segmentation requires image=... or media='local:...'")

        def segment_object(
            query: str,
            camera: str | None = None,
            media: str | None = None,
            image: Any = None,
            score_thresh: float = 0.1,
        ) -> SegmentationResult:
            """Segment the best matching object in a saved/study image using SAM3."""
            rgb = _resolve_study_image(image=image, media=media, camera=camera)
            mask, score = segment_object_sam3(
                rgb,
                query,
                _sam3_url(),
                score_threshold=float(score_thresh),
            )
            mask = np.asarray(mask, dtype=np.int32)
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                raise RuntimeError(f"SAM3 returned empty mask for {query!r}")
            try:
                from cap.agent.tools._artifact_log import log_mask

                log_mask(rgb, mask, query=f"study:{query}", tag="segment")
            except Exception:
                pass
            return SegmentationResult(
                mask=mask,
                bbox_xywh=[
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min()),
                    int(ys.max() - ys.min()),
                ],
                score=float(score),
                mask_area=int((mask > 0).sum()),
            )

        def segment_all_objects(
            query: str,
            camera: str | None = None,
            media: str | None = None,
            image: Any = None,
            score_thresh: float = 0.1,
        ) -> list[SegmentationResult]:
            """Segment all matching objects in a saved/study image using SAM3."""
            rgb = _resolve_study_image(image=image, media=media, camera=camera)
            return segment_all_objects_sam3(
                rgb,
                query,
                _sam3_url(),
                score_threshold=float(score_thresh),
            )

        def vlm_query(
            text: str,
            camera: str = "top",
            backend: str | None = None,
            image: Any = None,
            media: list[str] | None = None,
            model: str | None = None,
            temperature: float = 0.2,
            reasoning_effort: str = "high",
            **kwargs_extra: Any,
        ) -> str:
            _ = camera
            if media:
                raise RuntimeError(
                    "study-mode vlm_query does not support live media/camera sources; "
                    "pass image=... arrays instead."
                )
            if image is None:
                raise RuntimeError("study-mode vlm_query requires image=... input")
            images = [np.asarray(im) for im in image] if isinstance(image, list) else [np.asarray(image)]
            return _vlm_query(
                backend=backend or kwargs["vlm_backend"],
                text=text,
                images=images,
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                **kwargs_extra,
            )

        namespace: dict[str, Any] = {
            "vlm_query": vlm_query,
            "segment_object": segment_object,
            "segment_all_objects": segment_all_objects,
        }
        namespace.update(make_tool_runner_namespace())
        return env, namespace

    def run_script_overrides(self, cfg: Any | None = None) -> list[str]:
        _ = cfg
        return [f"robot={self.config_group}"]

    def child_env(
        self,
        cfg: Any | None = None,
        *,
        seed: int | None = None,
        slot: int = 0,
        n_seeds: int = 1,
    ) -> dict[str, str]:
        _ = seed

        def _endpoint(name: str, default_port: int) -> tuple[str, int]:
            host = str(cfg_select(cfg, f"runtime.{name}_host", "127.0.0.1"))
            base = int(cfg_select(cfg, f"runtime.{name}_port", default_port) or default_port)
            port = base + slot if (base > 0 and n_seeds > 1) else base
            return host, port

        sam3_host, sam3_port = _endpoint("sam3", 6767)
        anygrasp_host, anygrasp_port = _endpoint("anygrasp", 8122)
        bundlesdf_host, bundlesdf_port = _endpoint("bundlesdf", 8119)
        curobo_host, curobo_port = _endpoint("curobo", 8611)
        return {
            "SAM3_SERVER_HOST": sam3_host,
            "SAM3_SERVER_PORT": str(sam3_port),
            "ANYGRASP_SERVER_HOST": anygrasp_host,
            "ANYGRASP_SERVER_PORT": str(anygrasp_port),
            "ANYGRASP_SERVICE_URL": f"http://{anygrasp_host}:{anygrasp_port}",
            "BUNDLESDF_SERVER_HOST": bundlesdf_host,
            "BUNDLESDF_SERVER_PORT": str(bundlesdf_port),
            "CAP_CUROBO_HOST": curobo_host,
            "CAP_CUROBO_PORT": str(curobo_port),
            "CAP_CUROBO_START_SERVER": "0" if curobo_port > 0 else "1",
        }
