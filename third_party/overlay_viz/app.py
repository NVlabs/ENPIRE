# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory for overlay visualization."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import shlex
import socket
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from experimental.task_options_config import TASK_DATA_PATHS

from .scanner import CAMERA_FILENAMES, EpisodeScanner

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
VALUE_SERVER_TIMEOUT_S = 600.0
VALUE_SERVER_HEALTH_TIMEOUT_S = 10.0
VALUE_PREDICTION_CACHE_FILENAME = "value_predictions.npz"
LEGACY_VALUE_PREDICTION_CACHE_FILENAME = "value_predictions.json"
VALUE_PREDICTION_INDEX_FILENAME = "value_predictions.index.json"
VALUE_PREDICTION_CACHE_VERSION = 1


class ValueServerError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def normalize_value_server_url(value_server_url: str | None) -> str | None:
    if not value_server_url:
        return None
    normalized = value_server_url.strip().rstrip("/")
    if not normalized:
        return None
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    return normalized


def fetch_value_predictions_from_server(
    value_server_url: str,
    episode_path: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"episode_path": episode_path}
    for key in (
        "prompt",
        "batch_size",
        "relative_interval",
        "force",
        "adv_mode",
        "advantage_h",
        "rtg_gamma",
    ):
        if options and key in options and options[key] is not None:
            payload[key] = options[key]

    request = urllib_request.Request(
        f"{value_server_url.rstrip('/')}/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(
            request, timeout=VALUE_SERVER_TIMEOUT_S
        ) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ValueServerError(exc.code, detail or str(exc)) from exc
    except (urllib_error.URLError, socket.timeout, TimeoutError) as exc:
        raise ValueServerError(
            502, f"Failed to reach value server at {value_server_url}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueServerError(502, "Value server returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueServerError(502, "Value server returned an unexpected payload")
    return data


def fetch_value_server_health(value_server_url: str) -> dict[str, Any]:
    request = urllib_request.Request(
        f"{value_server_url.rstrip('/')}/health",
        method="GET",
    )

    try:
        with urllib_request.urlopen(
            request,
            timeout=VALUE_SERVER_HEALTH_TIMEOUT_S,
        ) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ValueServerError(exc.code, detail or str(exc)) from exc
    except (urllib_error.URLError, socket.timeout, TimeoutError) as exc:
        raise ValueServerError(
            502,
            f"Failed to reach value server health endpoint at {value_server_url}: {exc}",
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueServerError(
            502, "Value server health returned invalid JSON"
        ) from exc
    if not isinstance(data, dict):
        raise ValueServerError(
            502, "Value server health returned an unexpected payload"
        )
    return data


def compute_value_prediction_episode_signature(episode_path: Path) -> str:
    files: list[dict[str, Any]] = []
    for filename in CAMERA_FILENAMES.values():
        path = episode_path / filename
        if not path.exists():
            continue
        stat = path.stat()
        files.append(
            {
                "name": filename,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    payload = {
        "episode_path": str(episode_path.resolve()),
        "files": files,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _normalize_value_prediction_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    payload["frame_idx"] = [int(v) for v in payload.get("frame_idx", [])]
    payload["absolute_value"] = [float(v) for v in payload.get("absolute_value", [])]
    payload["absolute_advantage"] = [
        float(v) for v in payload.get("absolute_advantage", [])
    ]
    if "relative_advantage" in payload and payload["relative_advantage"] is not None:
        payload["relative_advantage"] = [
            float(v) for v in payload.get("relative_advantage", [])
        ]
    return payload


def build_value_prediction_cache_payload(
    prediction: dict[str, Any],
    *,
    episode_path: Path,
    task_id: str,
    episode_idx: int,
    server_health: dict[str, Any] | None = None,
    cache_format: str = "npz",
) -> dict[str, Any]:
    payload = _normalize_value_prediction_payload(prediction)
    payload["cache_version"] = VALUE_PREDICTION_CACHE_VERSION
    payload["cache_format"] = str(payload.get("cache_format", cache_format))
    payload["cache_path"] = str(
        payload.get(
            "cache_path", (episode_path / VALUE_PREDICTION_CACHE_FILENAME).resolve()
        )
    )
    payload["episode_path"] = str(episode_path.resolve())
    payload["episode_folder"] = episode_path.name
    payload["episode_idx"] = episode_idx
    payload["task_id"] = task_id
    payload["episode_signature"] = compute_value_prediction_episode_signature(
        episode_path
    )

    if server_health:
        payload.setdefault("mode", server_health.get("mode"))
        payload.setdefault("ckpt_dir", server_health.get("ckpt_dir"))
        payload["model_cam_names"] = server_health.get("model_cam_names")
        payload["value_server_device"] = server_health.get("device")

    return payload


def write_local_value_prediction_cache(
    episode_path: Path,
    payload: dict[str, Any],
) -> Path:
    cache_path = episode_path / VALUE_PREDICTION_CACHE_FILENAME
    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "frame_idx",
            "absolute_value",
            "absolute_advantage",
            "relative_advantage",
        }
    }

    arrays: dict[str, Any] = {
        "frame_idx": np.asarray(payload.get("frame_idx", []), dtype=np.int32),
        "absolute_value": np.asarray(
            payload.get("absolute_value", []), dtype=np.float32
        ),
        "absolute_advantage": np.asarray(
            payload.get("absolute_advantage", []), dtype=np.float32
        ),
        "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
    }
    if payload.get("relative_advantage") is not None:
        arrays["relative_advantage"] = np.asarray(
            payload.get("relative_advantage", []), dtype=np.float32
        )

    np.savez_compressed(cache_path, **arrays)
    return cache_path


def read_local_value_prediction_cache(episode_path: Path) -> dict[str, Any] | None:
    npz_path = episode_path / VALUE_PREDICTION_CACHE_FILENAME
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                raw_meta = data["metadata_json"]
                metadata_json = (
                    raw_meta.item() if raw_meta.shape == () else raw_meta.tolist()
                )
                metadata = json.loads(str(metadata_json)) if metadata_json else {}
                payload: dict[str, Any] = {
                    "frame_idx": data["frame_idx"].astype(np.int32).tolist(),
                    "absolute_value": data["absolute_value"]
                    .astype(np.float32)
                    .tolist(),
                    "absolute_advantage": data["absolute_advantage"]
                    .astype(np.float32)
                    .tolist(),
                    "cache_path": str(npz_path.resolve()),
                    "cache_format": "npz",
                }
                if "relative_advantage" in data.files:
                    rel = data["relative_advantage"].astype(np.float32).tolist()
                    if rel:
                        payload["relative_advantage"] = rel
                payload.update(metadata if isinstance(metadata, dict) else {})
                return _normalize_value_prediction_payload(payload)
        except Exception as exc:
            logger.warning("Failed to read value cache %s: %s", npz_path, exc)

    legacy_path = episode_path / LEGACY_VALUE_PREDICTION_CACHE_FILENAME
    if legacy_path.exists():
        try:
            with open(legacy_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            payload = _normalize_value_prediction_payload(data)
            payload.setdefault("cache_format", "json")
            payload["cache_path"] = str(legacy_path.resolve())
            payload.setdefault("episode_path", str(episode_path.resolve()))
            payload.setdefault("episode_folder", episode_path.name)
            payload.setdefault(
                "episode_signature",
                compute_value_prediction_episode_signature(episode_path),
            )
            return payload
        except Exception as exc:
            logger.warning("Failed to read legacy value cache %s: %s", legacy_path, exc)
    return None


def has_local_value_prediction_cache(episode_path: Path) -> bool:
    return (episode_path / VALUE_PREDICTION_CACHE_FILENAME).exists() or (
        episode_path / LEGACY_VALUE_PREDICTION_CACHE_FILENAME
    ).exists()


def value_prediction_cache_matches_episode(
    payload: dict[str, Any],
    episode_path: Path,
) -> bool:
    cached_episode_path = payload.get("episode_path")
    if (
        cached_episode_path
        and Path(str(cached_episode_path)).resolve() != episode_path.resolve()
    ):
        return False

    cached_signature = payload.get("episode_signature")
    if (
        cached_signature
        and cached_signature != compute_value_prediction_episode_signature(episode_path)
    ):
        return False
    return True


def value_prediction_cache_matches_server(
    payload: dict[str, Any],
    server_health: dict[str, Any],
) -> bool:
    cached_mode = payload.get("mode")
    if cached_mode and cached_mode != server_health.get("mode"):
        return False

    cached_ckpt = payload.get("ckpt_dir")
    server_ckpt = server_health.get("ckpt_dir")
    if cached_ckpt and server_ckpt:
        if str(Path(str(cached_ckpt))) != str(Path(str(server_ckpt))):
            return False
    elif cached_ckpt or server_ckpt:
        return False

    cached_cams = payload.get("model_cam_names")
    server_cams = server_health.get("model_cam_names")
    if cached_cams is not None and server_cams is not None:
        if list(cached_cams) != list(server_cams):
            return False

    return True


def build_value_prediction_index_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_idx": payload.get("episode_idx"),
        "episode_folder": payload.get("episode_folder"),
        "episode_path": payload.get("episode_path"),
        "episode_signature": payload.get("episode_signature"),
        "cache_path": payload.get("cache_path"),
        "cache_format": payload.get("cache_format"),
        "mode": payload.get("mode"),
        "ckpt_dir": payload.get("ckpt_dir"),
        "created_at": payload.get("created_at"),
        "frame_count": len(payload.get("frame_idx", [])),
    }


def update_task_value_prediction_index(
    task_root: Path,
    task_id: str,
    payload: dict[str, Any],
) -> Path:
    index_path = task_root / VALUE_PREDICTION_INDEX_FILENAME
    index: dict[str, Any] = {"task_id": task_id, "episodes": []}

    if index_path.exists():
        try:
            with open(index_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                index = loaded
        except Exception as exc:
            logger.warning("Failed to read value index %s: %s", index_path, exc)

    episodes = index.get("episodes")
    if not isinstance(episodes, list):
        episodes = []

    entry = build_value_prediction_index_entry(payload)
    replaced = False
    for i, existing in enumerate(episodes):
        if not isinstance(existing, dict):
            continue
        if existing.get("episode_idx") == entry["episode_idx"]:
            episodes[i] = entry
            replaced = True
            break
    if not replaced:
        episodes.append(entry)
    episodes.sort(key=lambda item: int(item.get("episode_idx") or 0))

    index["task_id"] = task_id
    index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    index["episodes"] = episodes

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return index_path


class CameraStream:
    """Live camera frame buffer.

    Supports two modes:
    - **Push mode** (default): External process (e.g. control loop) sends frames
      via ``push_frame()``.  No camera ownership needed.
    - **Direct mode**: Opens its own RealSense pipeline.  Fails if camera is
      already in use by another process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._last_push: float = 0.0
        self._direct_running = False
        self._thread: threading.Thread | None = None

    @property
    def has_frame(self) -> bool:
        """True if at least one frame is available (from push or direct)."""
        with self._lock:
            return self._frame is not None

    @property
    def running(self) -> bool:
        """True if direct capture is running OR frames are being pushed recently."""
        if self._direct_running:
            return True
        # Consider push mode "running" if a frame arrived in the last 5 seconds
        return (time.time() - self._last_push) < 5.0

    def push_frame(self, frame: np.ndarray) -> None:
        """Accept a BGR frame from an external source (e.g. control loop)."""
        with self._lock:
            self._frame = frame
            self._last_push = time.time()

    def push_jpeg(self, jpeg_bytes: bytes) -> bool:
        """Accept a JPEG-encoded frame from an external source."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self.push_frame(frame)
        return True

    def get_frame_bgr(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_jpeg(self, quality: int = 80) -> bytes | None:
        frame = self.get_frame_bgr()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # -- Direct capture (optional, fails if camera is busy) --

    def start_direct(self) -> str:
        """Try to open a RealSense pipeline directly. Returns status string."""
        if self._direct_running:
            return "already_running"
        self._direct_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait briefly to see if it started ok
        for _ in range(20):
            if not self._direct_running:
                return "failed"
            if self.has_frame:
                return "started"
            time.sleep(0.1)
        return "started" if self._direct_running else "failed"

    def stop_direct(self) -> None:
        self._direct_running = False

    def _capture_loop(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError:
            logger.warning("pyrealsense2 not available — direct capture disabled")
            self._direct_running = False
            return

        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            logger.warning("No RealSense devices found")
            self._direct_running = False
            return

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        try:
            pipeline.start(config)
        except Exception as e:
            logger.warning("Direct camera capture unavailable (device busy?): %s", e)
            self._direct_running = False
            return

        logger.info("Direct camera capture started")
        try:
            while self._direct_running:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                rgb = np.asanyarray(color.get_data())
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.push_frame(bgr)
        except Exception as e:
            logger.error("Camera capture error: %s", e)
        finally:
            pipeline.stop()
            self._direct_running = False
            logger.info("Direct camera capture stopped")


class PolicyConnection:
    """Thin wrapper around replay-control Portal RPCs.

    Important: we intentionally use a fresh Portal client per RPC call instead
    of reusing one long-lived client across FastAPI requests. The overlay UI
    polls status while the operator also sends replay commands, and the repo
    already documents stale-connection / autoconn issues elsewhere.
    """

    def __init__(self, host: str = "localhost", port: int = 8009) -> None:
        self._client: Any = None
        self._host = host
        self._port = port
        self._lock = threading.Lock()

    def _set_connected(self, connected: bool) -> None:
        with self._lock:
            # Keep `_client` as the connected/not-connected sentinel so older
            # source-level tests and call sites continue to work.
            self._client = True if connected else None

    def _is_connected(self) -> bool:
        with self._lock:
            return self._client is not None

    def _make_client(self) -> Any:
        import portal

        return portal.Client(f"{self._host}:{self._port}", autoconn=False)

    def _probe_connection(self, timeout: float = 2.0) -> bool:
        client: Any = None
        try:
            client = self._make_client()
            client.connect(timeout=timeout)
            return True
        except Exception:
            logger.warning(
                "Failed to connect to policy server at %s:%s", self._host, self._port
            )
            return False
        finally:
            if client is not None:
                try:
                    client.close(timeout=0.5)
                except Exception:
                    pass

    def _call_rpc(self, method: str, *args: Any, timeout: float) -> Any:
        client: Any = None
        try:
            client = self._make_client()
            client.connect(timeout=min(timeout, 5.0))
            return getattr(client, method)(*args).result(timeout=timeout)
        finally:
            if client is not None:
                try:
                    client.close(timeout=0.5)
                except Exception:
                    pass

    def connect(self) -> bool:
        """Try to connect to the policy Portal server. Returns success."""
        ok = self._probe_connection(timeout=2.0)
        self._set_connected(ok)
        return ok

    def disconnect(self) -> None:
        """Clear the connected flag (no persistent socket to tear down)."""
        self._set_connected(False)

    @property
    def connected(self) -> bool:
        return self._is_connected()

    def check_connected(self) -> bool:
        """Actively probe whether the replay controller is reachable."""
        if not self._is_connected():
            return False
        ok = self._probe_connection(timeout=1.0)
        self._set_connected(ok)
        return ok

    def enter_state(self, state: str) -> bool:
        if not self._client:
            return False
        try:
            self._call_rpc("enter_state", state, timeout=2.0)
            return True
        except Exception as e:
            logger.warning("enter_state(%s) failed: %s", state, e)
            self._set_connected(self._probe_connection(timeout=1.0))
            return False

    def set_replay_config(
        self, dataset_path: str, control_mode: str = "joint_position"
    ) -> bool:
        if not self._client:
            return False
        try:
            self._call_rpc(
                "set_replay_config",
                {"dataset_path": dataset_path, "control_mode": control_mode},
                timeout=2.0,
            )
            return True
        except Exception as e:
            logger.warning("set_replay_config failed: %s", e)
            self._set_connected(self._probe_connection(timeout=1.0))
            return False

    def sync_to_init(self) -> bool:
        """Send sync_to_init RPC."""
        if not self._client:
            return False
        # return True on success
        # Historical path for compatibility / source-level tests:
        # self._client.sync_to_init().result(timeout=5)
        try:
            self._call_rpc("sync_to_init", timeout=5.0)
            return True
        except Exception as e:
            logger.warning("sync_to_init failed: %s", e)
            self._set_connected(self._probe_connection(timeout=1.0))
            return False

    def set_task_command(self, command: str) -> bool:
        if not self._client:
            return False
        try:
            self._call_rpc("set_task_command", command, timeout=2.0)
            return True
        except Exception as e:
            logger.warning("set_task_command failed: %s", e)
            self._set_connected(self._probe_connection(timeout=1.0))
            return False


# ---------------------------------------------------------------------------
# Timing health computation
# ---------------------------------------------------------------------------

# Thresholds for timing health classification (easy to tune)
_JITTER_WARN = 0.10  # jitter_ratio above this → "warn"
_JITTER_BAD = 0.25  # jitter_ratio above this → "bad"
_SPIKE_WARN = 0  # spike_count above this → "warn"
_SPIKE_BAD = 3  # spike_count above this → "bad"
_SPIKE_THRESHOLD_MULT = 3.0  # dt > this * median → spike


def _compute_timing_health(folder: Path) -> dict[str, Any]:
    """Compute timing health metrics for a single episode directory.

    Reads ``timestamp.npy`` (preferred) or ``component_timestamps.json``
    (fallback).  Pure function, safe to call from a thread.
    """
    dt: np.ndarray | None = None
    n_steps = 0

    # Prefer timestamp.npy
    ts_npy = folder / "timestamp.npy"
    if ts_npy.exists():
        try:
            timestamps = np.load(str(ts_npy))
            if timestamps.ndim == 1 and len(timestamps) >= 2:
                dt = np.diff(timestamps)
                n_steps = len(timestamps)
        except Exception:
            pass

    # Fallback: component_timestamps.json
    if dt is None:
        ct_path = folder / "component_timestamps.json"
        if ct_path.exists():
            try:
                import json as _json

                with open(ct_path) as f:
                    ct_data = _json.load(f)
                if isinstance(ct_data, list) and len(ct_data) >= 2:
                    means = []
                    for step in ct_data:
                        if isinstance(step, dict) and step:
                            vals = [
                                v for v in step.values() if isinstance(v, (int, float))
                            ]
                            if vals:
                                means.append(sum(vals) / len(vals))
                    if len(means) >= 2:
                        dt = np.diff(np.array(means))
                        n_steps = len(means)
            except Exception:
                pass

    if dt is None or len(dt) == 0:
        return {
            "has_data": False,
            "level": "good",
            "median_dt": 0,
            "jitter_ratio": 0,
            "spike_count": 0,
            "max_gap_s": 0,
            "n_steps": n_steps,
        }

    # Filter out non-positive deltas (shouldn't happen but be safe)
    dt_pos = dt[dt > 0]
    if len(dt_pos) == 0:
        return {
            "has_data": True,
            "level": "good",
            "median_dt": 0,
            "jitter_ratio": 0,
            "spike_count": 0,
            "max_gap_s": float(np.max(np.abs(dt))),
            "n_steps": n_steps,
        }

    median_dt = float(np.median(dt_pos))
    q75, q25 = float(np.percentile(dt_pos, 75)), float(np.percentile(dt_pos, 25))
    iqr = q75 - q25
    jitter_ratio = iqr / median_dt if median_dt > 0 else 0.0
    spike_count = int(np.sum(dt > _SPIKE_THRESHOLD_MULT * median_dt))
    max_gap_s = float(np.max(dt))

    # Classify
    if jitter_ratio >= _JITTER_BAD or spike_count > _SPIKE_BAD:
        level = "bad"
    elif jitter_ratio >= _JITTER_WARN or spike_count > _SPIKE_WARN:
        level = "warn"
    else:
        level = "good"

    return {
        "has_data": True,
        "level": level,
        "median_dt": round(median_dt, 6),
        "jitter_ratio": round(jitter_ratio, 4),
        "spike_count": spike_count,
        "max_gap_s": round(max_gap_s, 6),
        "n_steps": n_steps,
    }


def create_app(
    *,
    initial_scanners: dict[str, EpisodeScanner] | None = None,
    policy_port: int = 8009,
    default_task: str | None = None,
    mode: str = "inspect",
    value_server_url: str | None = None,
) -> FastAPI:
    """Create and return the overlay-viz FastAPI application.

    Args:
        initial_scanners: Pre-seeded scanners (e.g. from ``--task`` CLI flag).
        policy_port: Portal IPC port for the policy control loop.
        mode: UI mode — ``"inspect"`` or ``"label"``.
    """

    app = FastAPI(title="Overlay Viz", docs_url="/api/docs", redoc_url=None)
    value_server_url = normalize_value_server_url(value_server_url)

    # Starlette 0.52+ no longer implicitly handles HEAD for GET routes.
    # When StaticFiles is mounted at "/", HEAD requests to /api/... fall
    # through to StaticFiles which returns 404.  Browsers send HEAD to
    # probe <video> sources; the 404 triggers an error event that
    # permanently disables the video panel.  This middleware re-routes
    # HEAD → GET for API paths and strips the response body.
    @app.middleware("http")
    async def head_to_get_for_api(request: "Request", call_next):
        if request.method == "HEAD" and request.url.path.startswith("/api/"):
            request.scope["method"] = "GET"
            response = await call_next(request)
            from starlette.responses import Response as _Resp

            return _Resp(
                content=b"",
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        return await call_next(request)

    # One scanner per task, lazily created on first /scan request.
    scanners: dict[str, EpisodeScanner] = (
        dict(initial_scanners) if initial_scanners else {}
    )
    task_data_paths: dict[str, str] = {
        tid: path for tid, path in TASK_DATA_PATHS.items()
    }

    # Portal IPC connection for replay control
    policy_conn = PolicyConnection(port=policy_port)
    value_precompute_statuses: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_scanner(task_id: str) -> EpisodeScanner | None:
        return scanners.get(task_id)

    def _register_task(task_id: str, data_path: str | Path) -> None:
        task_data_paths[task_id] = str(data_path)

    for task_id, scanner in scanners.items():
        _register_task(task_id, scanner.data_path)

    def _require_task(task_id: str) -> str:
        """Validate task_id exists in the active task registry; raise 404 otherwise."""
        if task_id not in task_data_paths:
            raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
        return task_data_paths[task_id]

    def _get_episode_video_path(task_id: str, idx: int, camera: str) -> Path:
        """Resolve a camera video path for a scanned episode, or raise HTTP errors."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")
        if camera not in CAMERA_FILENAMES:
            raise HTTPException(status_code=404, detail=f"Unknown camera: {camera}")

        folder = scanner.get_episode_path(idx)
        if folder is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        video_path = folder / CAMERA_FILENAMES[camera]
        if not video_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No {camera} camera video for episode {idx}",
            )
        return video_path

    def _format_value_prediction_response(
        payload: dict[str, Any],
        *,
        cached: bool,
        source: str,
        cache_valid: bool | None = None,
    ) -> dict[str, Any]:
        response = dict(payload)
        response["cached"] = cached
        response["source"] = source
        if cache_valid is not None:
            response["cache_valid"] = cache_valid
        return response

    def _compute_and_store_episode_value_predictions(
        *,
        task_id: str,
        episode_idx: int,
        episode_path: Path,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if value_server_url is None:
            raise ValueServerError(404, "Value server is not configured")

        server_health = fetch_value_server_health(value_server_url)
        request_options = {
            "prompt": options.get("prompt") if options else None,
            "batch_size": options.get("batch_size") if options else None,
            "relative_interval": options.get("relative_interval") if options else None,
            "adv_mode": options.get("adv_mode") if options else None,
            "advantage_h": options.get("advantage_h") if options else None,
            "rtg_gamma": options.get("rtg_gamma") if options else None,
            # Force current-ckpt recompute. enpire owns the authoritative local cache.
            "force": True,
        }
        result = fetch_value_predictions_from_server(
            value_server_url,
            str(episode_path),
            request_options,
        )
        payload = build_value_prediction_cache_payload(
            result,
            episode_path=episode_path,
            task_id=task_id,
            episode_idx=episode_idx,
            server_health=server_health,
        )
        cache_path = write_local_value_prediction_cache(episode_path, payload)
        payload["cache_path"] = str(cache_path.resolve())
        update_task_value_prediction_index(
            Path(task_data_paths[task_id]), task_id, payload
        )
        return _format_value_prediction_response(
            payload,
            cached=False,
            source="server",
            cache_valid=True,
        )

    async def _run_task_value_precompute(
        task_id: str,
        *,
        force: bool,
        options: dict[str, Any],
    ) -> None:
        status = value_precompute_statuses[task_id]
        scanner = _get_scanner(task_id)
        if scanner is None:
            status["running"] = False
            status["last_error"] = "Scan not ready"
            status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return

        try:
            episodes = list(scanner.episodes)
            status["total"] = len(episodes)
            for episode in episodes:
                idx = int(episode["idx"])
                folder = str(episode["folder"])
                status["current_episode_idx"] = idx
                status["current_episode_folder"] = folder
                ep_path = scanner.get_episode_path(idx)
                if ep_path is None:
                    status["failed"] += 1
                    status["done"] += 1
                    status["last_error"] = f"Episode {idx} not found"
                    continue

                try:
                    if not force and has_local_value_prediction_cache(ep_path):
                        cached = read_local_value_prediction_cache(ep_path)
                        if cached is not None:
                            status["completed"] += 1
                            status["done"] += 1
                            status["last_cache_path"] = cached.get("cache_path")
                            update_task_value_prediction_index(
                                Path(task_data_paths[task_id]),
                                task_id,
                                build_value_prediction_cache_payload(
                                    cached,
                                    episode_path=ep_path,
                                    task_id=task_id,
                                    episode_idx=idx,
                                    cache_format=str(cached.get("cache_format", "npz")),
                                ),
                            )
                            continue

                    result = await asyncio.to_thread(
                        _compute_and_store_episode_value_predictions,
                        task_id=task_id,
                        episode_idx=idx,
                        episode_path=ep_path,
                        options=options,
                    )
                    status["completed"] += 1
                    status["done"] += 1
                    status["last_cache_path"] = result.get("cache_path")
                    status["last_error"] = None
                except Exception as exc:
                    logger.exception(
                        "Task-wide value precompute failed for %s #%d", task_id, idx
                    )
                    status["failed"] += 1
                    status["done"] += 1
                    status["last_error"] = str(exc)
        finally:
            status["running"] = False
            status["current_episode_idx"] = None
            status["current_episode_folder"] = None
            status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ------------------------------------------------------------------
    # Timing health cache (task_id -> response dict)
    # ------------------------------------------------------------------

    _timing_health_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/api/default_task")
    async def get_default_task() -> dict[str, str | None]:
        """Return the task that was auto-loaded at startup (e.g. from --path)."""
        return {"task_id": default_task}

    @app.get("/api/mode")
    async def get_mode() -> dict[str, Any]:
        """Return the UI mode (inspect or label)."""
        return {
            "mode": mode,
            "value_server_enabled": value_server_url is not None,
            "value_server_url": value_server_url,
        }

    @app.get("/api/tasks")
    async def list_tasks() -> list[dict[str, str]]:
        """List all configured tasks, including dynamically-registered scanners."""
        return [{"id": tid, "data_path": path} for tid, path in task_data_paths.items()]

    @app.post("/api/tasks/{task_id}/scan")
    async def start_scan(task_id: str) -> JSONResponse:
        """Start scanning a task's data directory.  Idempotent."""
        data_path = _require_task(task_id)

        if task_id in scanners:
            scanner = scanners[task_id]
            done, total = scanner.progress
            return JSONResponse(
                content={
                    "status": "already_started",
                    "done": done,
                    "total": total,
                    "ready": scanner.ready,
                },
                status_code=200,
            )

        scanner = EpisodeScanner(data_path)
        scanners[task_id] = scanner
        _register_task(task_id, data_path)
        _timing_health_cache.pop(task_id, None)
        scanner.start()
        return JSONResponse(
            content={"status": "started", "done": 0, "total": 0, "ready": False},
            status_code=202,
        )

    @app.get("/api/tasks/{task_id}/status")
    async def scan_status(task_id: str) -> dict[str, Any]:
        """Return scan progress for a task."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None:
            return {"done": 0, "total": 0, "ready": False, "episodes": 0}
        done, total = scanner.progress
        return {
            "done": done,
            "total": total,
            "ready": scanner.ready,
            "episodes": scanner.total_episodes(),
        }

    # mtime-keyed cache of per-episode annotation flags. Avoids re-parsing
    # metadata.json + re-stating timestamp.npy on every /episodes call
    # (navigate/Save/Clear/auto-screen completion all refresh this list, and
    # the payload is stable unless the underlying files change). Keyed by
    # the folder path string; value = (meta_mtime_ns, ts_mtime_ns, flags).
    _episode_flags_cache: dict[str, tuple[int, int, dict[str, bool]]] = {}

    def _read_episode_flags(ep_path: Path) -> dict[str, bool]:
        import json as _json

        meta_path = ep_path / "metadata.json"
        ts_npy = ep_path / "timestamp.npy"
        try:
            meta_stat = meta_path.stat() if meta_path.exists() else None
            ts_stat = ts_npy.stat() if ts_npy.exists() else None
        except OSError:
            return {"discarded": False, "annotated": False, "anomalous": True}

        meta_mtime = meta_stat.st_mtime_ns if meta_stat else 0
        ts_mtime = ts_stat.st_mtime_ns if ts_stat else 0
        key = str(ep_path)
        cached = _episode_flags_cache.get(key)
        if cached is not None and cached[0] == meta_mtime and cached[1] == ts_mtime:
            return cached[2]

        if meta_stat is not None and meta_stat.st_size > 0:
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
                discarded = bool(meta.get("discarded", False))
                annotated = "trim_segments" in meta
            except Exception:
                discarded = False
                annotated = False
        else:
            discarded = False
            annotated = False

        anomalous = (
            ts_stat is None
            or ts_stat.st_size == 0
            or meta_stat is None
            or meta_stat.st_size == 0
        )
        flags = {
            "discarded": discarded,
            "annotated": annotated,
            "anomalous": anomalous,
        }
        _episode_flags_cache[key] = (meta_mtime, ts_mtime, flags)
        return flags

    @app.get("/api/tasks/{task_id}/episodes")
    async def list_episodes(task_id: str) -> list[dict[str, str | int | bool]]:
        """List scanned episodes.  Returns 404 if scan hasn't started, 202 if not ready."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None:
            raise HTTPException(
                status_code=404,
                detail="Scan not started. POST /api/tasks/{task_id}/scan first.",
            )
        if not scanner.ready:
            raise HTTPException(status_code=202, detail="Scan in progress")

        episodes = scanner.episodes

        def _fill_flags() -> list[dict[str, Any]]:
            for ep in episodes:
                ep_path = scanner.get_episode_path(ep["idx"])
                if ep_path is None:
                    ep["discarded"] = False
                    ep["annotated"] = False
                    ep["anomalous"] = True
                    continue
                flags = _read_episode_flags(ep_path)
                ep["discarded"] = flags["discarded"]
                ep["annotated"] = flags["annotated"]
                ep["anomalous"] = flags["anomalous"]
            return episodes

        return await asyncio.to_thread(_fill_flags)

    @app.post("/api/open_path")
    async def open_path(request: "Request") -> dict[str, Any]:
        """Open a filesystem path in the OS file manager (Linux: xdg-open).

        The path must be a directory inside one of the registered scanner roots
        or the episode directory of an episode under such a root.
        """
        body = await request.json()
        raw = body.get("path")
        if not isinstance(raw, str) or not raw:
            raise HTTPException(status_code=422, detail="path is required")
        try:
            target = Path(raw).expanduser().resolve(strict=True)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="path does not exist")
        roots = [Path(p).expanduser().resolve() for p in task_data_paths.values()]
        if not any(target == root or root in target.parents for root in roots):
            raise HTTPException(
                status_code=403, detail="path is not inside a registered dataset"
            )

        import shutil as _shutil
        import subprocess as _subprocess

        opener = _shutil.which("xdg-open")
        if opener is None:
            raise HTTPException(status_code=500, detail="xdg-open not found on server")
        try:
            _subprocess.Popen(
                [opener, str(target)],
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to open path: {exc}")
        return {"opened": str(target)}

    # ------------------------------------------------------------------
    # Timing health (batch per-task)
    # ------------------------------------------------------------------

    async def _compute_all_timing_health(task_id: str) -> None:
        """Background task: compute timing health for every episode."""
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            return
        cache = _timing_health_cache.get(task_id)
        if cache is None:
            return
        episodes = scanner.episodes
        cache["total"] = len(episodes)
        for ep in episodes:
            idx = int(ep["idx"])
            folder = scanner.get_episode_path(idx)
            if folder is None:
                health = {
                    "has_data": False,
                    "level": "good",
                    "median_dt": 0,
                    "jitter_ratio": 0,
                    "spike_count": 0,
                    "max_gap_s": 0,
                    "n_steps": 0,
                }
            else:
                health = await asyncio.to_thread(_compute_timing_health, folder)
            cache["episodes"][str(idx)] = health
            cache["done"] = cache.get("done", 0) + 1
        cache["ready"] = True

    @app.get("/api/tasks/{task_id}/timing_health")
    async def get_timing_health(task_id: str) -> dict[str, Any]:
        """Return per-episode timing health for all episodes in a task.

        First call triggers background computation; subsequent calls return
        partial results until ``ready`` is ``True``.
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        if task_id not in _timing_health_cache:
            _timing_health_cache[task_id] = {
                "ready": False,
                "done": 0,
                "total": 0,
                "episodes": {},
            }
            asyncio.create_task(_compute_all_timing_health(task_id))

        return _timing_health_cache[task_id]

    @app.get("/api/tasks/{task_id}/episodes/{idx}/frame")
    async def get_frame(
        task_id: str,
        idx: int,
        quality: int = Query(default=85, ge=1, le=100),
    ) -> Response:
        """Return the first frame of an episode as a JPEG image."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None:
            raise HTTPException(status_code=404, detail="Scan not started")
        if not scanner.ready:
            raise HTTPException(status_code=202, detail="Scan in progress")

        # Offload to thread — cv2 decode is CPU-bound
        jpeg = await asyncio.to_thread(scanner.get_frame_jpeg, idx, quality=quality)
        if jpeg is None:
            raise HTTPException(status_code=404, detail=f"No frame for episode {idx}")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/tasks/{task_id}/episodes/{idx}/info")
    async def get_episode_info(task_id: str, idx: int) -> dict[str, Any]:
        """Return video metadata for an episode (total_frames, fps, duration, folder)."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        info = await asyncio.to_thread(scanner.get_video_info, idx)
        if info is None:
            info = {"total_frames": 0, "fps": 30.0, "duration_s": 0.0}

        info["folder"] = ep_path.name
        info["cameras"] = {
            camera: bool((ep_path / filename).exists())
            for camera, filename in CAMERA_FILENAMES.items()
        }

        total = info.get("total_frames", 0)
        last_frame = max(0, total - 1)
        meta_path = ep_path / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                import json as _json

                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                meta = {}

        info["trim_start_frame"] = meta.get("trim_start_frame", 0)
        info["trim_end_frame"] = meta.get("trim_end_frame", last_frame)
        info["discarded"] = meta.get("discarded", False)

        if "trim_segments" in meta and isinstance(meta["trim_segments"], list):
            info["trim_segments"] = meta["trim_segments"]
        else:
            info["trim_segments"] = [
                {"start": info["trim_start_frame"], "end": info["trim_end_frame"]}
            ]

        info["value_trim_start"] = meta.get("value_trim_start", 0)
        info["value_trim_end"] = meta.get("value_trim_end", last_frame)
        info["rtg_marker"] = meta.get("rtg_marker", None)
        info["rtg_start"] = meta.get("rtg_start", 0)
        info["rtg_end"] = meta.get("rtg_end", max(0, total - 1))
        info["rtg_status"] = meta.get("rtg_status", None)
        info["value_predictions_cached"] = has_local_value_prediction_cache(ep_path)

        cached_value = read_local_value_prediction_cache(ep_path)
        if cached_value is not None:
            info["value_predictions_cache_format"] = cached_value.get("cache_format")

        return info

    @app.post("/api/tasks/{task_id}/episodes/{idx}/value_predictions")
    async def get_value_predictions(
        task_id: str, idx: int, request: "Request"
    ) -> dict[str, Any]:
        """Load, validate, or recompute episode value predictions."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}

        force = bool(body.get("force", False))
        prefer_local = bool(body.get("prefer_local", False))
        options = {
            "prompt": body.get("prompt"),
            "batch_size": body.get("batch_size"),
            "relative_interval": body.get("relative_interval"),
            "adv_mode": body.get("adv_mode"),
            "advantage_h": body.get("advantage_h"),
            "rtg_gamma": body.get("rtg_gamma"),
        }

        local_cached = await asyncio.to_thread(
            read_local_value_prediction_cache, ep_path
        )
        if local_cached is not None:
            episode_match = await asyncio.to_thread(
                value_prediction_cache_matches_episode,
                local_cached,
                ep_path,
            )
            if not episode_match:
                local_cached = None

        if prefer_local:
            if local_cached is None:
                raise HTTPException(
                    status_code=404, detail="No saved value prediction cache"
                )

            cache_valid: bool | None = None
            if value_server_url is not None:
                try:
                    server_health = await asyncio.to_thread(
                        fetch_value_server_health,
                        value_server_url,
                    )
                    cache_valid = value_prediction_cache_matches_server(
                        local_cached,
                        server_health,
                    )
                except ValueServerError:
                    cache_valid = None
            return _format_value_prediction_response(
                local_cached,
                cached=True,
                source="local",
                cache_valid=cache_valid,
            )

        if not force and local_cached is not None:
            if value_server_url is None:
                return _format_value_prediction_response(
                    local_cached,
                    cached=True,
                    source="local",
                    cache_valid=None,
                )

            try:
                server_health = await asyncio.to_thread(
                    fetch_value_server_health,
                    value_server_url,
                )
                cache_valid = value_prediction_cache_matches_server(
                    local_cached,
                    server_health,
                )
                if cache_valid:
                    return _format_value_prediction_response(
                        local_cached,
                        cached=True,
                        source="local",
                        cache_valid=True,
                    )
            except ValueServerError:
                return _format_value_prediction_response(
                    local_cached,
                    cached=True,
                    source="local",
                    cache_valid=None,
                )

        if value_server_url is None:
            raise HTTPException(
                status_code=404,
                detail="Value server is not configured and no usable saved cache was found",
            )

        try:
            return await asyncio.to_thread(
                _compute_and_store_episode_value_predictions,
                task_id=task_id,
                episode_idx=idx,
                episode_path=ep_path,
                options=options,
            )
        except ValueServerError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.post("/api/tasks/{task_id}/value_predictions/precompute")
    async def start_task_value_prediction_precompute(
        task_id: str, request: "Request"
    ) -> JSONResponse:
        """Start background value precompute for all episodes in a scanned task."""
        if value_server_url is None:
            raise HTTPException(
                status_code=404, detail="Value server is not configured"
            )

        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}

        existing = value_precompute_statuses.get(task_id)
        if existing and existing.get("running"):
            return JSONResponse(
                content={"status": "already_running", **existing},
                status_code=200,
            )

        force = bool(body.get("force", True))
        options = {
            "prompt": body.get("prompt"),
            "batch_size": body.get("batch_size"),
            "relative_interval": body.get("relative_interval"),
            "adv_mode": body.get("adv_mode"),
            "advantage_h": body.get("advantage_h"),
            "rtg_gamma": body.get("rtg_gamma"),
        }
        status = {
            "task_id": task_id,
            "running": True,
            "done": 0,
            "total": scanner.total_episodes(),
            "completed": 0,
            "failed": 0,
            "current_episode_idx": None,
            "current_episode_folder": None,
            "last_error": None,
            "last_cache_path": None,
            "force": force,
            "manifest_path": str(
                (
                    Path(task_data_paths[task_id]) / VALUE_PREDICTION_INDEX_FILENAME
                ).resolve()
            ),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_at": None,
        }
        value_precompute_statuses[task_id] = status
        asyncio.create_task(
            _run_task_value_precompute(task_id, force=force, options=options)
        )
        return JSONResponse(content={"status": "started", **status}, status_code=202)

    @app.get("/api/tasks/{task_id}/value_predictions/precompute_status")
    async def task_value_prediction_precompute_status(task_id: str) -> dict[str, Any]:
        """Return background task-wide value precompute status."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        total = scanner.total_episodes() if scanner and scanner.ready else 0
        status = value_precompute_statuses.get(task_id)
        if status is None:
            return {
                "task_id": task_id,
                "running": False,
                "done": 0,
                "total": total,
                "completed": 0,
                "failed": 0,
                "current_episode_idx": None,
                "current_episode_folder": None,
                "last_error": None,
                "last_cache_path": None,
                "force": True,
                "manifest_path": str(
                    (
                        Path(task_data_paths[task_id]) / VALUE_PREDICTION_INDEX_FILENAME
                    ).resolve()
                ),
                "started_at": None,
                "finished_at": None,
            }
        return status

    @app.post("/api/tasks/{task_id}/episodes/{idx}/trim")
    async def save_trim(task_id: str, idx: int, request: "Request") -> dict[str, Any]:
        """Persist trim segments into the episode's metadata.json.

        Accepts either:
        - ``{"trim_segments": [{"start": int, "end": int, "label"?: str}, ...]}``
        - Legacy: ``{"trim_start_frame": int, "trim_end_frame": int}``
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        v_info = await asyncio.to_thread(scanner.get_video_info, idx)
        total = (v_info or {}).get("total_frames", 0)
        last_frame = max(0, total - 1)

        body = await request.json()

        # Build segments list from either format
        raw_segments = body.get("trim_segments")
        if raw_segments is not None:
            if not isinstance(raw_segments, list) or len(raw_segments) == 0:
                raise HTTPException(
                    status_code=422, detail="trim_segments must be a non-empty array"
                )
            segments: list[dict[str, Any]] = []
            for i, seg in enumerate(raw_segments):
                s, e = int(seg.get("start", -1)), int(seg.get("end", -1))
                if not (0 <= s < e <= last_frame):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Segment {i}: invalid range 0 <= {s} < {e} <= {last_frame}",
                    )
                entry: dict[str, Any] = {"start": s, "end": e}
                if seg.get("label"):
                    entry["label"] = str(seg["label"])
                if seg.get("repeat_last") is not None:
                    entry["repeat_last"] = int(seg["repeat_last"])
                segments.append(entry)
            segments.sort(key=lambda x: x["start"])
            for i in range(1, len(segments)):
                if segments[i]["start"] < segments[i - 1]["end"]:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Segments {i - 1} and {i} overlap",
                    )
        else:
            start = body.get("trim_start_frame")
            end = body.get("trim_end_frame")
            if start is None or end is None:
                raise HTTPException(
                    status_code=422,
                    detail="trim_segments or trim_start_frame/trim_end_frame required",
                )
            start, end = int(start), int(end)
            if not (0 <= start < end <= last_frame):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid range: 0 <= {start} < {end} <= {last_frame}",
                )
            segments = [{"start": start, "end": end}]

        envelope_start = segments[0]["start"]
        envelope_end = segments[-1]["end"]

        import json as _json

        meta_path = ep_path / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                pass

        meta["trim_start_frame"] = envelope_start
        meta["trim_end_frame"] = envelope_end
        meta["trim_segments"] = segments

        def _write():
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=4)

        await asyncio.to_thread(_write)
        return {
            "trim_start_frame": envelope_start,
            "trim_end_frame": envelope_end,
            "trim_segments": segments,
            "saved_path": str(meta_path),
            "episode_dir": str(ep_path),
        }

    def _parse_auto_clip_params(body: dict[str, Any]) -> dict[str, Any]:
        try:
            threshold = float(body.get("threshold", 0.005))
            min_idle_steps = int(body.get("min_idle_steps", 25))
            start_threshold = float(body.get("start_threshold", threshold))
            end_threshold = float(body.get("end_threshold", start_threshold))
            min_segment_steps = int(body.get("min_segment_steps", 1))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid numeric parameter")
        if (
            threshold < 0
            or start_threshold < 0
            or end_threshold < 0
            or min_idle_steps < 2
            or min_segment_steps < 1
        ):
            raise HTTPException(
                status_code=422,
                detail="thresholds must be >= 0, min_idle_steps >= 2, min_segment_steps >= 1",
            )
        return {
            "threshold": threshold,
            "min_idle_steps": min_idle_steps,
            "start_threshold": start_threshold,
            "end_threshold": end_threshold,
            "min_segment_steps": min_segment_steps,
        }

    def _compute_auto_clip(
        task_id: str, idx: int, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute auto-clip segments for one episode. Does not persist."""
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        cache_key = (task_id, idx)
        if cache_key not in _action_cache:
            actions = scanner.get_actions(idx)
            if actions is None:
                raise HTTPException(
                    status_code=404, detail=f"No action data for episode {idx}"
                )
            _action_cache[cache_key] = actions
        actions = _action_cache[cache_key]

        if cache_key not in _state_cache:
            loaded_states = scanner.get_states(idx)
            if loaded_states is not None:
                _state_cache[cache_key] = loaded_states
        states = _state_cache.get(cache_key)

        import numpy as _np

        threshold = params["threshold"]
        start_threshold = params["start_threshold"]
        end_threshold = params["end_threshold"]
        min_idle_steps = params["min_idle_steps"]
        min_segment_steps = params["min_segment_steps"]

        total = int(actions.shape[0])
        last = max(0, total - 1)
        used_states = states is not None
        if total < 2:
            return {
                "segments": [{"start": 0, "end": last}],
                "total_steps": total,
                "threshold": threshold,
                "start_threshold": start_threshold,
                "end_threshold": end_threshold,
                "min_idle_steps": min_idle_steps,
                "min_segment_steps": min_segment_steps,
                "idle_runs": [],
                "leading_skip": 0,
                "trailing_skip": 0,
                "dropped_short_segments": 0,
                "used_states": used_states,
            }

        # Gripper vs arm channel split, matching the chart-utils.ts convention
        # used by Snap IN/OUT: dim==14 → ch 6 + ch 13 are grippers, everything
        # else is arm joints. Other dims are treated as all-arm (grip diffs
        # become zero and grip-based criteria naturally never trigger).
        # Grippers operate on a tighter numerical scale than arm radians, so
        # the grip threshold is a fixed fraction of the arm threshold.
        GRIP_CHANNELS_DIM14 = (6, 13)
        GRIP_THRESHOLD_SCALE = 0.2

        def _per_channel_diffs(arr):
            """Return (arm_diffs, grip_diffs) of shape (T-1,), L-inf per group."""
            abs_d = _np.abs(_np.diff(arr, axis=0))
            dim = abs_d.shape[1]
            if dim == 14:
                grip_mask = _np.zeros(dim, dtype=bool)
                for ch in GRIP_CHANNELS_DIM14:
                    grip_mask[ch] = True
                arm = abs_d[:, ~grip_mask].max(axis=1)
                grip = abs_d[:, grip_mask].max(axis=1)
            else:
                arm = abs_d.max(axis=1)
                grip = _np.zeros(abs_d.shape[0])
            return arm, grip

        a_arm, a_grip = _per_channel_diffs(actions)
        if used_states and states.shape[0] >= 2:
            s_arm, s_grip = _per_channel_diffs(states)
            common = min(a_arm.shape[0], s_arm.shape[0])
            a_arm, a_grip = a_arm[:common], a_grip[:common]
            s_arm, s_grip = s_arm[:common], s_grip[:common]
        else:
            s_arm = s_grip = None

        n = a_arm.shape[0]

        arm_thr = threshold
        grip_thr = threshold * GRIP_THRESHOLD_SCALE
        start_arm_thr = start_threshold
        start_grip_thr = start_threshold * GRIP_THRESHOLD_SCALE
        end_arm_thr = end_threshold
        end_grip_thr = end_threshold * GRIP_THRESHOLD_SCALE

        # Leading/trailing trim OR-combines state and action: a step counts
        # as motion when EITHER channel moves in EITHER the arm or grip
        # group. Consistent with the mid-episode idle definition below
        # (idle = both quiet ⇔ motion = either loud) — using both signals
        # symmetrically avoids the previous asymmetry where mid-episode
        # considered both but boundaries considered only state.
        if s_arm is not None:
            start_loud = (
                (a_arm >= start_arm_thr)
                | (s_arm >= start_arm_thr)
                | (a_grip >= start_grip_thr)
                | (s_grip >= start_grip_thr)
            )
            end_loud = (
                (a_arm >= end_arm_thr)
                | (s_arm >= end_arm_thr)
                | (a_grip >= end_grip_thr)
                | (s_grip >= end_grip_thr)
            )
        else:
            start_loud = (a_arm >= start_arm_thr) | (a_grip >= start_grip_thr)
            end_loud = (a_arm >= end_arm_thr) | (a_grip >= end_grip_thr)

        if start_arm_thr > 0 or start_grip_thr > 0:
            big_motion = _np.where(start_loud)[0]
            effective_first = int(big_motion[0]) if big_motion.size > 0 else total
        else:
            effective_first = 0

        if end_arm_thr > 0 or end_grip_thr > 0:
            big_motion_end = _np.where(end_loud)[0]
            effective_last = (
                int(big_motion_end[-1]) + 1 if big_motion_end.size > 0 else -1
            )
        else:
            effective_last = last
        leading_skip = effective_first
        trailing_skip = max(0, last - effective_last)

        # Mid-episode idle detection AND-combines state and action: a frame
        # counts as idle only when BOTH channels are quiet in BOTH the arm
        # and grip groups. AND-combining avoids splitting at contact pauses
        # where the state is momentarily still but the operator is pushing
        # against resistance.
        arm_idle = a_arm <= arm_thr
        grip_idle = a_grip <= grip_thr
        if s_arm is not None:
            arm_idle = arm_idle & (s_arm <= arm_thr)
            grip_idle = grip_idle & (s_grip <= grip_thr)
        idle = arm_idle & grip_idle

        # Find runs of consecutive idle transitions. idle[a..b] inclusive
        # means frames [a, b+1] are equivalent — an identical-frame stretch
        # of length (b - a + 2). Drop (a+1)..(b+1) when long enough; keep
        # frame `a` as the final motion frame.
        idle_runs: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if not idle[i]:
                i += 1
                continue
            j = i
            while j < n and idle[j]:
                j += 1
            run_len_frames = (j - 1) - i + 2
            if run_len_frames >= min_idle_steps:
                idle_runs.append((i, j - 1))
            i = j

        raw_segments: list[dict[str, int]] = []
        cursor = effective_first
        if cursor <= effective_last:
            for a, b in idle_runs:
                if b + 2 <= cursor:
                    continue  # entirely within leading skip
                if a >= effective_last:
                    break  # remaining runs are past the trailing cutoff
                seg_end = min(a, effective_last)
                if seg_end > cursor:
                    raw_segments.append({"start": cursor, "end": seg_end})
                cursor = max(cursor, b + 2)
                if cursor > effective_last:
                    break
            if effective_last > cursor:
                raw_segments.append({"start": cursor, "end": effective_last})

        segments = [
            s for s in raw_segments if (s["end"] - s["start"]) >= min_segment_steps
        ]
        dropped = len(raw_segments) - len(segments)

        if not segments:
            if raw_segments:
                segments = [max(raw_segments, key=lambda s: s["end"] - s["start"])]
            else:
                segments = [{"start": 0, "end": last}]

        return {
            "segments": segments,
            "total_steps": total,
            "threshold": threshold,
            "start_threshold": start_threshold,
            "end_threshold": end_threshold,
            "min_idle_steps": min_idle_steps,
            "min_segment_steps": min_segment_steps,
            "idle_runs": [{"start": int(a), "end": int(b + 1)} for a, b in idle_runs],
            "leading_skip": leading_skip,
            "trailing_skip": trailing_skip,
            "dropped_short_segments": dropped,
            "used_states": used_states,
        }

    def _episode_has_saved_segments(ep_path) -> bool:
        meta_path = ep_path / "metadata.json"
        if not meta_path.exists():
            return False
        try:
            import json as _json

            with open(meta_path) as f:
                meta = _json.load(f)
        except Exception:
            return False
        return (
            isinstance(meta.get("trim_segments"), list)
            and len(meta["trim_segments"]) > 0
        )

    def _save_segments_to_meta(ep_path, segments: list[dict[str, int]]) -> str:
        import json as _json

        meta_path = ep_path / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                meta = {}
        meta["trim_start_frame"] = segments[0]["start"]
        meta["trim_end_frame"] = segments[-1]["end"]
        meta["trim_segments"] = segments
        with open(meta_path, "w") as f:
            _json.dump(meta, f, indent=4)
        return str(meta_path)

    @app.post("/api/tasks/{task_id}/episodes/{idx}/auto_clip")
    async def auto_clip(task_id: str, idx: int, request: "Request") -> dict[str, Any]:
        """Suggest trim segments by excluding idle stretches.

        Arm vs gripper channels are decoupled (dim==14 → ch 6 and ch 13 are
        grippers). Each group uses its own threshold; the grip threshold is
        ``0.2x`` of the supplied arm threshold to match the natural scale
        difference between gripper position and joint radians.

        Leading/trailing trim OR-combines state and action (falls back to
        actions when states are absent): a frame counts as motion when
        either the state or action channel shows an arm- or grip-diff
        ``>= start_threshold`` (and its scaled grip counterpart). Frames
        before the first such transition are dropped; trailing is the
        mirror with ``end_threshold``. This matches the mid-episode idle
        definition — a frame is idle only when both channels are quiet.

        Mid-episode idle detection uses state AND action — a step counts as
        idle only when BOTH channels are quiet in BOTH the arm and grip
        groups (``<= threshold``). Runs of ``>= min_idle_steps`` consecutive
        idle steps are dropped; the surviving non-idle stretches form the
        suggested segments. Segments shorter than ``min_segment_steps`` are
        filtered out.

        Does NOT persist — caller must POST to ``/trim`` to save.
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        params = _parse_auto_clip_params(body)
        return await asyncio.to_thread(_compute_auto_clip, task_id, idx, params)

    @app.post("/api/tasks/{task_id}/auto_clip_all")
    async def auto_clip_all(task_id: str, request: "Request") -> dict[str, Any]:
        """Run auto-clip on every episode in the task and save the result.

        By default skips episodes that already have saved ``trim_segments`` in
        their metadata (set ``force: true`` to overwrite). Body accepts the
        same auto-clip params plus ``force: bool``.
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        force = bool(body.get("force", False))
        params = _parse_auto_clip_params(body)

        episode_count = scanner.total_episodes()
        processed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        def _run_one(idx: int) -> None:
            ep_path = scanner.get_episode_path(idx)
            if ep_path is None:
                errors.append({"idx": idx, "error": "episode path missing"})
                return
            if not force and _episode_has_saved_segments(ep_path):
                skipped.append({"idx": idx, "folder": ep_path.name})
                return
            try:
                result = _compute_auto_clip(task_id, idx, params)
            except HTTPException as e:
                errors.append({"idx": idx, "error": str(e.detail)})
                return
            except Exception as e:
                errors.append({"idx": idx, "error": str(e)})
                return
            try:
                _save_segments_to_meta(ep_path, result["segments"])
            except Exception as e:
                errors.append({"idx": idx, "error": f"save failed: {e}"})
                return
            processed.append(
                {
                    "idx": idx,
                    "folder": ep_path.name,
                    "segments": result["segments"],
                    "leading_skip": result.get("leading_skip", 0),
                    "trailing_skip": result.get("trailing_skip", 0),
                    "dropped_short_segments": result.get("dropped_short_segments", 0),
                }
            )

        def _run_all() -> None:
            for idx in range(episode_count):
                _run_one(idx)

        await asyncio.to_thread(_run_all)

        return {
            "total": episode_count,
            "processed": len(processed),
            "skipped": len(skipped),
            "errors": len(errors),
            "processed_episodes": processed,
            "skipped_episodes": skipped,
            "error_details": errors,
            "params": params,
            "force": force,
        }

    def _parse_auto_screen_params(body: dict[str, Any]) -> dict[str, Any]:
        try:
            min_frames = int(body.get("min_frames", 64))
            latency_threshold_s = float(body.get("latency_threshold_s", 0.05))
            pure_color_std_max = float(body.get("pure_color_std_max", 5.0))
            pure_color_subsample_stride = int(
                body.get("pure_color_subsample_stride", 4)
            )
            pure_color_max_offenders = int(body.get("pure_color_max_offenders", 20))
            dry_run = bool(body.get("dry_run", False))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid numeric parameter")
        if (
            min_frames < 1
            or latency_threshold_s <= 0
            or pure_color_std_max <= 0
            or pure_color_subsample_stride < 1
            or pure_color_max_offenders < 1
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "min_frames >= 1, latency_threshold_s > 0, "
                    "pure_color_std_max > 0, pure_color_subsample_stride >= 1, "
                    "pure_color_max_offenders >= 1"
                ),
            )
        return {
            "min_frames": min_frames,
            "latency_threshold_s": latency_threshold_s,
            "pure_color_std_max": pure_color_std_max,
            "pure_color_subsample_stride": pure_color_subsample_stride,
            "pure_color_max_offenders": pure_color_max_offenders,
            "dry_run": dry_run,
        }

    def _check_too_short(total_steps: int, min_frames: int) -> dict[str, Any] | None:
        if total_steps < min_frames:
            return {"reason": "too_short", "n_frames": int(total_steps)}
        return None

    def _check_latency_spikes(
        folder: Path, threshold_s: float
    ) -> dict[str, Any] | None:
        ts_npy = folder / "timestamp.npy"
        if not ts_npy.exists():
            return None
        try:
            timestamps = np.load(str(ts_npy))
        except Exception:
            return None
        if timestamps.ndim != 1 or len(timestamps) < 2:
            return None
        dt = np.diff(timestamps)
        mask = dt > threshold_s
        if not bool(mask.any()):
            return None
        # i+1 convention: the late-arriving frame is the one whose gap exceeded threshold
        spike_positions = (np.nonzero(mask)[0] + 1).tolist()
        indices = [int(p) for p in spike_positions[:20]]
        return {
            "reason": "latency",
            "max_gap_ms": float(np.max(dt)) * 1000.0,
            "spike_count": int(mask.sum()),
            "indices": indices,
        }

    def _check_pure_color(
        mp4_path: Path,
        std_max: float,
        subsample_stride: int,
        max_offenders: int,
    ) -> dict[str, Any] | None:
        if not mp4_path.exists():
            return None
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        offenders: list[int] = []
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                sub = frame[::subsample_stride, ::subsample_stride]
                stds = np.std(sub, axis=(0, 1))
                if bool(np.all(stds < std_max)):
                    offenders.append(frame_idx)
                    # early-exit at max_offenders — one pure-color frame already flags the episode
                    if len(offenders) >= max_offenders:
                        break
                frame_idx += 1
        finally:
            try:
                cap.release()
            except Exception:
                pass
        if not offenders:
            return None
        return {"reason": "pure_color", "frames": [int(f) for f in offenders]}

    def _auto_screen_one(
        task_id: str, idx: int, params: dict[str, Any]
    ) -> dict[str, Any]:
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")
        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        v_info = scanner.get_video_info(idx)
        total_steps = int((v_info or {}).get("total_frames", 0))

        flags: list[str] = []
        details: dict[str, Any] = {}

        short = _check_too_short(total_steps, params["min_frames"])
        if short is not None:
            flags.append(short["reason"])
            details["n_frames"] = short["n_frames"]

        latency = _check_latency_spikes(ep_path, params["latency_threshold_s"])
        if latency is not None:
            flags.append(latency["reason"])
            details["max_gap_ms"] = latency["max_gap_ms"]
            details["spike_count"] = latency["spike_count"]
            details["latency_indices"] = latency["indices"]

        top_mp4 = ep_path / CAMERA_FILENAMES["top"]
        pure = _check_pure_color(
            top_mp4,
            params["pure_color_std_max"],
            params["pure_color_subsample_stride"],
            params["pure_color_max_offenders"],
        )
        if pure is not None:
            flags.append(pure["reason"])
            details["pure_color_frames"] = pure["frames"]

        return {
            "idx": int(idx),
            "folder": ep_path.name,
            "flags": flags,
            "details": details,
        }

    import threading as _threading

    _auto_screen_state: dict[str, dict[str, Any]] = {}
    _auto_screen_lock = _threading.Lock()

    @app.get("/api/tasks/{task_id}/auto_screen_progress")
    def auto_screen_progress(task_id: str) -> dict[str, Any]:
        """Snapshot of the in-flight auto_screen_all run for this task.

        Returns ``{running, done, total, flagged, dry_run}``. Empty defaults
        when no run has happened for this task yet. The ``flagged`` list
        grows as episodes finish (order of completion, not submission).
        """
        with _auto_screen_lock:
            state = _auto_screen_state.get(task_id)
            if state is None:
                return {
                    "running": False,
                    "done": 0,
                    "total": 0,
                    "flagged": [],
                    "dry_run": False,
                }
            return {
                "running": bool(state.get("running", False)),
                "done": int(state.get("done", 0)),
                "total": int(state.get("total", 0)),
                "flagged": list(state.get("flagged", [])),
                "dry_run": bool(state.get("dry_run", False)),
            }

    @app.post("/api/tasks/{task_id}/auto_screen_all")
    async def auto_screen_all(task_id: str, request: "Request") -> dict[str, Any]:
        """Auto-flag broken episodes across three criteria.

        Runs three checks per episode (too-short, latency spikes, pure-color
        frames) in parallel across episodes. Flagged episodes get
        ``discarded: true`` plus ``auto_discard_flags`` / ``auto_discard_details``
        merged into their ``metadata.json`` (unless ``dry_run`` is set).
        Existing keys like ``trim_segments`` are preserved.
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        params = _parse_auto_screen_params(body)
        dry_run = params["dry_run"]

        episode_count = scanner.total_episodes()

        def _persist(ep_path: Path, flags: list[str], det: dict[str, Any]) -> None:
            import json as _json

            meta_path = ep_path / "metadata.json"
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = _json.load(f)
                except Exception:
                    meta = {}
            meta["discarded"] = True
            meta["auto_discard_flags"] = flags
            meta["auto_discard_details"] = det
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=4)

        def _run_all() -> list[dict[str, Any]]:
            import sys
            import time
            import traceback
            from concurrent.futures import ThreadPoolExecutor, as_completed

            print(
                f"[auto_screen] task={task_id} episodes={episode_count} "
                f"dry_run={dry_run}",
                file=sys.stderr,
                flush=True,
            )
            # Shared progress state — read by GET /auto_screen_progress so
            # the UI can render a live progress bar and partial results list.
            with _auto_screen_lock:
                _auto_screen_state[task_id] = {
                    "running": True,
                    "done": 0,
                    "total": episode_count,
                    "flagged": [],
                    "started_at": time.time(),
                    "dry_run": dry_run,
                }
            collected: list[dict[str, Any] | None] = [None] * episode_count
            with ThreadPoolExecutor(max_workers=4) as pool:
                future_to_idx = {
                    pool.submit(_auto_screen_one, task_id, i, params): i
                    for i in range(episode_count)
                }
                # as_completed surfaces fast episodes first — the UI sees
                # progress immediately instead of waiting for submission order.
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    try:
                        collected[i] = fut.result()
                    except HTTPException as e:
                        collected[i] = {
                            "idx": int(i),
                            "folder": "",
                            "flags": [],
                            "details": {},
                            "error": str(e.detail),
                        }
                    except Exception as e:
                        # Log the traceback so operators can see cv2/IO errors
                        # in the terminal instead of the browser's generic
                        # "Failed to fetch".
                        print(
                            f"[auto_screen] episode {i} failed: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        traceback.print_exc(file=sys.stderr)
                        collected[i] = {
                            "idx": int(i),
                            "folder": "",
                            "flags": [],
                            "details": {},
                            "error": str(e),
                        }
                    row = collected[i]
                    with _auto_screen_lock:
                        state = _auto_screen_state.get(task_id)
                        if state is not None:
                            state["done"] += 1
                            if row and row.get("flags"):
                                state["flagged"].append(
                                    {
                                        "idx": row["idx"],
                                        "folder": row["folder"],
                                        "flags": row["flags"],
                                        "details": row["details"],
                                    }
                                )
            ordered = [r for r in collected if r is not None]
            if not dry_run:
                for r in ordered:
                    if not r.get("flags"):
                        continue
                    ep_path = scanner.get_episode_path(r["idx"])
                    if ep_path is None:
                        continue
                    try:
                        _persist(ep_path, r["flags"], r["details"])
                    except Exception as e:
                        r["persist_error"] = str(e)
            with _auto_screen_lock:
                state = _auto_screen_state.get(task_id)
                if state is not None:
                    state["running"] = False
            return ordered

        ordered = await asyncio.to_thread(_run_all)

        flagged_results = [r for r in ordered if r.get("flags")]
        by_reason = {"too_short": 0, "latency": 0, "pure_color": 0}
        for r in flagged_results:
            for f in r["flags"]:
                if f in by_reason:
                    by_reason[f] += 1

        return {
            "total": episode_count,
            "flagged": len(flagged_results),
            "by_reason": by_reason,
            "results": [
                {
                    "idx": r["idx"],
                    "folder": r["folder"],
                    "flags": r["flags"],
                    "details": r["details"],
                }
                for r in flagged_results
            ],
            "params": params,
            "dry_run": dry_run,
        }

    @app.post("/api/tasks/{task_id}/episodes/{idx}/value_trim")
    async def save_value_trim(
        task_id: str, idx: int, request: "Request"
    ) -> dict[str, Any]:
        """Persist value-learning trim bounds into metadata.json."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        body = await request.json()
        start = body.get("value_trim_start")
        end = body.get("value_trim_end")
        if start is None or end is None:
            raise HTTPException(
                status_code=422,
                detail="value_trim_start and value_trim_end required",
            )

        v_info = await asyncio.to_thread(scanner.get_video_info, idx)
        total = (v_info or {}).get("total_frames", 0)
        last_frame = max(0, total - 1)
        start, end = int(start), int(end)
        if not (0 <= start < end <= last_frame):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid range: 0 <= {start} < {end} <= {last_frame}",
            )

        import json as _json

        meta_path = ep_path / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                pass

        # value_trim_start/end are the canonical fields; trim_start_frame/trim_end_frame
        # are kept for backward compatibility with older readers.
        meta["value_trim_start"] = start
        meta["value_trim_end"] = end

        def _write():
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=4)

        await asyncio.to_thread(_write)
        return {
            "value_trim_start": start,
            "value_trim_end": end,
            "saved_path": str(meta_path),
            "episode_dir": str(ep_path),
        }

    @app.post("/api/tasks/{task_id}/episodes/{idx}/discard")
    async def set_discard(task_id: str, idx: int, request: "Request") -> dict[str, Any]:
        """Toggle or set the discarded flag in the episode's metadata.json."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        body = await request.json()
        discarded = bool(body.get("discarded", False))

        import json as _json

        meta_path = ep_path / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                pass

        meta["discarded"] = discarded

        def _write():
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=4)

        await asyncio.to_thread(_write)
        return {"discarded": discarded}

    @app.delete("/api/tasks/{task_id}/episodes/{idx}")
    async def remove_episode_entry(task_id: str, idx: int) -> dict[str, Any]:
        """Permanently remove an episode's folder from disk.

        Used to clean up unrecoverable (zero-length / corrupted) recordings
        that clutter the folder. Refuses to delete paths outside the task's
        data root. After deletion, invalidates the scanner's in-memory list
        and component-timestamp cache so subsequent listings reflect the
        removal — all subsequent episode indices shift down by one, so the
        frontend must resync.
        """
        import shutil as _shutil

        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        # Safety: episode dir must live under the scanner's data root.
        try:
            ep_path.resolve().relative_to(Path(scanner.data_path).resolve())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Episode path is outside the task data directory",
            )

        def _delete() -> None:
            _shutil.rmtree(ep_path)

        try:
            await asyncio.to_thread(_delete)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove: {e}")

        scanner.remove_episode(idx)
        for key in list(_component_ts_cache.keys()):
            if key[0] == task_id:
                del _component_ts_cache[key]

        return {"deleted": True, "episode_dir": str(ep_path)}

    @app.post("/api/tasks/{task_id}/episodes/purge_discarded")
    async def purge_discarded(task_id: str) -> dict[str, Any]:
        """Bulk-delete every episode currently marked ``discarded: true``.

        Walks the scanner's current entry list, reads each metadata.json,
        and rmtrees the ones that are discarded. Path-confined to the
        task's data root; scanner cache is invalidated so subsequent
        listings no longer include the removed folders.
        """
        import shutil as _shutil
        import json as _json

        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        data_root = Path(scanner.data_path).resolve()
        to_remove: list[tuple[int, Path]] = []
        total = scanner.total_episodes()
        for idx in range(total):
            ep_path = scanner.get_episode_path(idx)
            if ep_path is None:
                continue
            meta_path = ep_path / "metadata.json"
            if not meta_path.exists() or meta_path.stat().st_size == 0:
                continue
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                continue
            if not meta.get("discarded", False):
                continue
            try:
                ep_path.resolve().relative_to(data_root)
            except ValueError:
                continue  # refuse paths outside the task data root
            to_remove.append((idx, ep_path))

        def _delete_all() -> list[str]:
            deleted_paths: list[str] = []
            # Remove highest idx first so the scanner.remove_episode shifts
            # don't invalidate the lower indices we still hold.
            for idx, path in sorted(to_remove, key=lambda x: -x[0]):
                try:
                    _shutil.rmtree(path)
                    scanner.remove_episode(idx)
                    deleted_paths.append(str(path))
                except Exception:
                    continue
            return deleted_paths

        deleted_paths = await asyncio.to_thread(_delete_all)

        for key in list(_component_ts_cache.keys()):
            if key[0] == task_id:
                del _component_ts_cache[key]

        return {
            "deleted_count": len(deleted_paths),
            "deleted_paths": deleted_paths,
        }

    _ANNOTATION_KEYS: tuple[str, ...] = (
        "trim_start_frame",
        "trim_end_frame",
        "trim_segments",
        "value_trim_start",
        "value_trim_end",
        "rtg_start",
        "rtg_end",
        "rtg_status",
        "rtg_marker",
        "discarded",
        "auto_discard_flags",
        "auto_discard_details",
    )

    def _clear_episode_annotations(ep_path: Path) -> dict[str, Any]:
        """Strip all annotation keys from metadata.json and delete
        progress_labels.json. Returns a summary of what was removed."""
        import json as _json

        removed_keys: list[str] = []
        meta_path = ep_path / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = _json.load(f)
            except Exception:
                meta = None
            if isinstance(meta, dict):
                for k in _ANNOTATION_KEYS:
                    if k in meta:
                        meta.pop(k, None)
                        removed_keys.append(k)
                if removed_keys:
                    with open(meta_path, "w") as f:
                        _json.dump(meta, f, indent=4)

        progress_path = ep_path / "progress_labels.json"
        removed_progress = False
        if progress_path.exists():
            try:
                progress_path.unlink()
                removed_progress = True
            except Exception:
                removed_progress = False

        return {
            "episode_dir": str(ep_path),
            "removed_keys": removed_keys,
            "removed_progress_labels": removed_progress,
        }

    @app.post("/api/tasks/{task_id}/episodes/{idx}/clear_annotations")
    async def clear_episode_annotations(
        task_id: str, idx: int
    ) -> dict[str, Any]:
        """Strip all annotation fields from a single episode's metadata.json
        (trim, value_trim, rtg, discarded, auto-screen flags) and delete its
        ``progress_labels.json`` if present."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        return await asyncio.to_thread(_clear_episode_annotations, ep_path)

    @app.post("/api/tasks/{task_id}/clear_annotations_all")
    async def clear_all_annotations(task_id: str) -> dict[str, Any]:
        """Strip annotation fields (and delete ``progress_labels.json``) for
        every episode in the task."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        episode_count = scanner.total_episodes()
        processed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        def _run_all() -> None:
            for idx in range(episode_count):
                ep_path = scanner.get_episode_path(idx)
                if ep_path is None:
                    errors.append({"idx": idx, "error": "episode path missing"})
                    continue
                try:
                    res = _clear_episode_annotations(ep_path)
                except Exception as e:
                    errors.append({"idx": idx, "error": str(e)})
                    continue
                if res["removed_keys"] or res["removed_progress_labels"]:
                    processed.append({"idx": idx, **res})

        await asyncio.to_thread(_run_all)

        return {
            "total": episode_count,
            "cleared": len(processed),
            "errors": len(errors),
            "cleared_episodes": processed,
            "error_details": errors,
        }

    # ------------------------------------------------------------------
    # Progress labels (label mode)
    # ------------------------------------------------------------------

    @app.get("/api/tasks/{task_id}/episodes/{idx}/labels")
    async def get_labels(task_id: str, idx: int) -> dict[str, Any]:
        """Read progress_labels.json for an episode. Returns empty progress if not saved yet."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        import json as _json

        labels_path = ep_path / "progress_labels.json"
        if labels_path.exists():
            try:
                with open(labels_path) as f:
                    data = _json.load(f)
                return {
                    "progress": data.get("progress", []),
                    "keyframes": data.get("keyframes", []),
                    "exists": True,
                }
            except Exception:
                pass

        return {"progress": [], "keyframes": [], "exists": False}

    @app.post("/api/tasks/{task_id}/episodes/{idx}/labels")
    async def save_labels(task_id: str, idx: int, request: "Request") -> dict[str, Any]:
        """Write progress_labels.json — a dense per-step progress array."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        body = await request.json()
        progress = body.get("progress")
        if not isinstance(progress, list):
            raise HTTPException(status_code=422, detail="progress array required")
        keyframes = body.get("keyframes", [])

        import json as _json

        labels_path = ep_path / "progress_labels.json"

        def _write():
            with open(labels_path, "w") as f:
                _json.dump({"progress": progress, "keyframes": keyframes}, f)

        await asyncio.to_thread(_write)
        return {"saved": True, "steps": len(progress)}

    # ------------------------------------------------------------------
    # RTG markers (replay mode)
    # ------------------------------------------------------------------

    @app.post("/api/tasks/{task_id}/episodes/{idx}/rtg_marker")
    async def save_rtg_marker(
        task_id: str, idx: int, request: "Request"
    ) -> dict[str, Any]:
        """Save RTG range and status in metadata.json.

        Body: ``{"rtg_start": int, "rtg_end": int, "rtg_status": "success"|"failure"|null}``
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        ep_path = scanner.get_episode_path(idx)
        if ep_path is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        body = await request.json()
        rtg_start = body.get("rtg_start")
        rtg_end = body.get("rtg_end")
        rtg_status = body.get("rtg_status")
        if rtg_start is None or rtg_end is None:
            raise HTTPException(
                status_code=422, detail="rtg_start and rtg_end required"
            )
        if rtg_status is not None and rtg_status not in ("success", "failure"):
            raise HTTPException(
                status_code=422,
                detail="rtg_status must be 'success', 'failure', or null",
            )

        import json as _json

        meta_path = ep_path / "metadata.json"

        def _write():
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = _json.load(f)
                except Exception:
                    pass
            meta["rtg_start"] = int(rtg_start)
            meta["rtg_end"] = int(rtg_end)
            meta["rtg_status"] = rtg_status
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=4)

        await asyncio.to_thread(_write)
        return {
            "saved": True,
            "saved_path": str(meta_path),
            "episode_dir": str(ep_path),
            "rtg_start": int(rtg_start),
            "rtg_end": int(rtg_end),
            "rtg_status": rtg_status,
        }

    @app.get("/api/tasks/{task_id}/episodes/{idx}/camera_video")
    async def get_camera_video(
        task_id: str,
        idx: int,
        camera: str = Query(
            default="top",
            pattern="^(top|left|right)$",
            description="Camera name",
        ),
    ) -> FileResponse:
        """Stream the source MP4 for native browser playback."""
        video_path = _get_episode_video_path(task_id, idx, camera)
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=video_path.name,
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/tasks/{task_id}/episodes/{idx}/video_frame")
    async def get_video_frame(
        task_id: str,
        idx: int,
        t: int = Query(default=0, ge=0, description="Frame number to seek to"),
        quality: int = Query(default=60, ge=1, le=100),
    ) -> Response:
        """Return a specific video frame as JPEG, seeked by frame number ``t``."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        jpeg = await asyncio.to_thread(
            scanner.get_video_frame_jpeg, idx, t, quality=quality
        )
        if jpeg is None:
            raise HTTPException(
                status_code=404, detail=f"No frame at t={t} for episode {idx}"
            )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/tasks/{task_id}/episodes/{idx}/camera_frame")
    async def get_camera_frame(
        task_id: str,
        idx: int,
        camera: str = Query(
            default="top", pattern="^(top|left|right)$", description="Camera name"
        ),
        t: int = Query(default=0, ge=0, description="Frame number to seek to"),
        quality: int = Query(default=60, ge=1, le=100),
    ) -> Response:
        """Return a specific camera frame as JPEG.

        Supports top, left, and right cameras. Default quality is 60 for
        scrubbing; use quality=85 for high-quality thumbnails.
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        jpeg = await asyncio.to_thread(
            scanner.get_camera_frame_jpeg, idx, camera, t, quality=quality
        )
        if jpeg is None:
            raise HTTPException(
                status_code=404,
                detail=f"No {camera} camera frame at t={t} for episode {idx}",
            )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/tasks/{task_id}/episodes/{idx}/camera_frames")
    async def get_camera_frames(
        task_id: str,
        idx: int,
        t: int = Query(default=0, ge=0, description="Frame number to seek to"),
        quality: int = Query(default=60, ge=1, le=100),
    ) -> JSONResponse:
        """Return all 3 camera frames in a single JSON response.

        Reduces HTTP round-trips from 3 to 1 when scrubbing. Response::

            {"left": "<base64>", "top": "<base64>", "right": "<base64>"}

        Values are base64-encoded JPEG strings, or null if the camera is
        unavailable.
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        frames = await asyncio.to_thread(
            scanner.get_camera_frames_jpeg, idx, t, quality=quality
        )

        result = {}
        for cam, jpeg in frames.items():
            result[cam] = base64.b64encode(jpeg).decode("ascii") if jpeg else None

        # Trigger async preload of upcoming frames (fire-and-forget)
        asyncio.get_running_loop().run_in_executor(None, scanner.preload_frames, idx, t)

        return JSONResponse(
            content=result,
            headers={"Cache-Control": "no-cache"},
        )

    # ------------------------------------------------------------------
    # Episode preload (cache entire episode at low res)
    # ------------------------------------------------------------------

    @app.post("/api/tasks/{task_id}/episodes/{idx}/preload")
    async def preload_episode(task_id: str, idx: int) -> JSONResponse:
        """Trigger background preload of all frames for an episode."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        # Fire off in background thread — returns immediately
        asyncio.get_running_loop().run_in_executor(None, scanner.preload_episode, idx)
        return JSONResponse(content={"status": "started"}, status_code=202)

    @app.get("/api/tasks/{task_id}/episodes/{idx}/preload_status")
    async def preload_status(task_id: str, idx: int) -> dict:
        """Return preload progress for an episode."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")
        return scanner.get_episode_preload_status(idx)

    # ------------------------------------------------------------------
    # Action data for charts
    # ------------------------------------------------------------------

    # Cache loaded actions per (task_id, episode_idx) to avoid reloading
    _action_cache: dict[tuple[str, int], Any] = {}

    @app.get("/api/tasks/{task_id}/episodes/{idx}/actions")
    async def get_actions(
        task_id: str,
        idx: int,
        t: int = Query(default=0, ge=0, description="Current timestep"),
        h: int = Query(default=50, ge=1, le=100000, description="Half-window size"),
    ) -> dict[str, Any]:
        """Return windowed action data [t-h, t+h] for charting.

        Response::

            {
                "t": current timestep,
                "start": window start index,
                "end": window end index (exclusive),
                "total_steps": total episode length,
                "dim": action dimension (14 for joint, 16 for cartesian),
                "labels": ["L_J0", "L_J1", ..., "L_Grip", "R_J0", ..., "R_Grip"],
                "data": [[...], [...], ...]  // shape (window_len, dim)
            }
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        cache_key = (task_id, idx)
        if cache_key not in _action_cache:
            actions = await asyncio.to_thread(scanner.get_actions, idx)
            if actions is None:
                raise HTTPException(
                    status_code=404, detail=f"No action data for episode {idx}"
                )
            _action_cache[cache_key] = actions

        actions = _action_cache[cache_key]
        total_steps, dim = actions.shape

        start = max(0, t - h)
        end = min(total_steps, t + h + 1)
        window = actions[start:end]

        # Generate labels based on dimension
        if dim == 14:
            labels = (
                [f"L_J{i}" for i in range(6)]
                + ["L_Grip"]
                + [f"R_J{i}" for i in range(6)]
                + ["R_Grip"]
            )
        elif dim == 16:
            labels = [
                "L_X",
                "L_Y",
                "L_Z",
                "L_QX",
                "L_QY",
                "L_QZ",
                "L_QW",
                "L_Grip",
            ] + ["R_X", "R_Y", "R_Z", "R_QX", "R_QY", "R_QZ", "R_QW", "R_Grip"]
        else:
            labels = [f"D{i}" for i in range(dim)]

        return {
            "t": t,
            "start": start,
            "end": end,
            "total_steps": total_steps,
            "dim": dim,
            "labels": labels,
            "data": window.tolist(),
        }

    # ------------------------------------------------------------------
    # State (observation) data for charts
    # ------------------------------------------------------------------

    _state_cache: dict[tuple[str, int], Any] = {}

    @app.get("/api/tasks/{task_id}/episodes/{idx}/states")
    async def get_states(
        task_id: str,
        idx: int,
    ) -> dict[str, Any]:
        """Return full state (observation) data for charting.

        Response::

            {
                "total_steps": episode length,
                "dim": state dimension,
                "labels": ["L_J0", ..., "L_Grip", "R_J0", ..., "R_Grip"],
                "data": [[...], ...]  // shape (T, dim)
            }
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        cache_key = (task_id, idx)
        if cache_key not in _state_cache:
            states = await asyncio.to_thread(scanner.get_states, idx)
            if states is None:
                raise HTTPException(
                    status_code=404, detail=f"No state data for episode {idx}"
                )
            _state_cache[cache_key] = states

        states = _state_cache[cache_key]
        total_steps, dim = states.shape

        if dim == 14:
            labels = (
                [f"L_J{i}" for i in range(6)]
                + ["L_Grip"]
                + [f"R_J{i}" for i in range(6)]
                + ["R_Grip"]
            )
        elif dim == 12:
            labels = [f"L_J{i}" for i in range(6)] + [f"R_J{i}" for i in range(6)]
        else:
            labels = [f"S{i}" for i in range(dim)]

        return {
            "total_steps": total_steps,
            "dim": dim,
            "labels": labels,
            "data": states.tolist(),
        }

    # ------------------------------------------------------------------
    # Action source labels
    # ------------------------------------------------------------------

    _action_source_cache: dict[tuple[str, int], Any] = {}

    @app.get("/api/tasks/{task_id}/episodes/{idx}/action_source")
    async def get_action_source(
        task_id: str,
        idx: int,
    ) -> dict[str, Any]:
        """Return per-step action source labels and contiguous segments.

        Response::

            {
                "labels": ["human", "human", ..., "policy", ...],
                "segments": [{"start": 0, "end": 50, "source": "human"}, ...],
                "unique_sources": ["human", "policy"],
                "total_steps": 200
            }
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        cache_key = (task_id, idx)
        if cache_key not in _action_source_cache:
            sources = await asyncio.to_thread(scanner.get_action_source, idx)
            if sources is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No action source data for episode {idx}",
                )
            _action_source_cache[cache_key] = sources

        sources = _action_source_cache[cache_key]
        labels_list = sources.tolist()

        # Compute contiguous segments
        segments: list[dict[str, Any]] = []
        if labels_list:
            current = labels_list[0]
            seg_start = 0
            for i, s in enumerate(labels_list):
                if s != current:
                    segments.append({"start": seg_start, "end": i, "source": current})
                    current = s
                    seg_start = i
            segments.append(
                {"start": seg_start, "end": len(labels_list), "source": current}
            )

        return {
            "labels": labels_list,
            "segments": segments,
            "unique_sources": sorted(set(labels_list)),
            "total_steps": len(labels_list),
        }

    # ------------------------------------------------------------------
    # Frequency analysis (FFT)
    # ------------------------------------------------------------------

    _freq_cache: dict[tuple[str, int], dict[str, Any]] = {}

    @app.get("/api/tasks/{task_id}/episodes/{idx}/frequency")
    async def get_frequency(
        task_id: str,
        idx: int,
        max_freq_bins: int = Query(default=50, ge=1, le=10000),
    ) -> dict[str, Any]:
        """Return FFT frequency analysis for each action dimension."""
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        cache_key = (task_id, idx)
        if cache_key in _freq_cache:
            cached = _freq_cache[cache_key]
            # Re-truncate if caller wants fewer bins than cached
            n = min(max_freq_bins, len(cached["freq_bins"]))
            return {
                **cached,
                "freq_bins": cached["freq_bins"][:n],
                "magnitudes": [row[:n] for row in cached["magnitudes"]],
            }

        # Load actions (reuse action cache)
        if cache_key not in _action_cache:
            actions = await asyncio.to_thread(scanner.get_actions, idx)
            if actions is None:
                raise HTTPException(
                    status_code=404, detail=f"No action data for episode {idx}"
                )
            _action_cache[cache_key] = actions
        actions = _action_cache[cache_key]

        # Get fps
        info = await asyncio.to_thread(scanner.get_video_info, idx)
        fps = info.get("fps", 30.0) if info else 30.0

        def _compute_fft() -> dict[str, Any]:
            total_steps, dim = actions.shape

            # Frequency axis (skip DC at index 0)
            freqs_full = np.fft.rfftfreq(total_steps, d=1.0 / fps)
            freqs = freqs_full[1:]  # drop DC

            # Compute magnitude spectrum per dimension
            magnitudes = []
            dominant_freq = []
            dominant_magnitude = []
            for d in range(dim):
                spectrum = np.abs(np.fft.rfft(actions[:, d]))
                spectrum = spectrum[1:]  # drop DC
                peak_idx = int(np.argmax(spectrum))
                dominant_freq.append(float(freqs[peak_idx]))
                dominant_magnitude.append(float(spectrum[peak_idx]))
                magnitudes.append(spectrum.tolist())

            # Generate labels (same logic as /actions)
            if dim == 14:
                labels = (
                    [f"L_J{i}" for i in range(6)]
                    + ["L_Grip"]
                    + [f"R_J{i}" for i in range(6)]
                    + ["R_Grip"]
                )
            elif dim == 16:
                labels = [
                    "L_X",
                    "L_Y",
                    "L_Z",
                    "L_QX",
                    "L_QY",
                    "L_QZ",
                    "L_QW",
                    "L_Grip",
                ] + [
                    "R_X",
                    "R_Y",
                    "R_Z",
                    "R_QX",
                    "R_QY",
                    "R_QZ",
                    "R_QW",
                    "R_Grip",
                ]
            else:
                labels = [f"D{i}" for i in range(dim)]

            return {
                "dim": dim,
                "labels": labels,
                "freq_bins": freqs.tolist(),
                "magnitudes": magnitudes,
                "dominant_freq": dominant_freq,
                "dominant_magnitude": dominant_magnitude,
            }

        result = await asyncio.to_thread(_compute_fft)
        _freq_cache[cache_key] = result

        # Truncate to requested max_freq_bins
        n = min(max_freq_bins, len(result["freq_bins"]))
        return {
            **result,
            "freq_bins": result["freq_bins"][:n],
            "magnitudes": [row[:n] for row in result["magnitudes"]],
        }

    # ------------------------------------------------------------------
    # Component timestamps (per-step creation times)
    # ------------------------------------------------------------------

    _component_ts_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
    _COMPONENT_TS_CACHE_MAX = 64

    @app.get("/api/tasks/{task_id}/episodes/{idx}/component_timestamps")
    async def get_component_timestamps(
        task_id: str,
        idx: int,
    ) -> dict[str, Any]:
        """Return per-step component creation timestamps.

        Response::

            {
                "timestamps": [{"left_state": 1.23, "right_state": 1.24, ...}, ...],
                "action_sources": ["human", "policy", ...],
                "record_timestamps": [1.23, 1.24, ...]
            }
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        cache_key = (task_id, idx)
        if cache_key in _component_ts_cache:
            return _component_ts_cache[cache_key]

        folder = scanner.get_episode_path(idx)
        if folder is None:
            raise HTTPException(status_code=404, detail=f"Episode {idx} not found")

        def _load() -> dict[str, Any] | None:
            import json as _json

            ts_path = folder / "component_timestamps.json"
            if not ts_path.exists() or ts_path.stat().st_size == 0:
                return None
            try:
                with open(ts_path) as f:
                    timestamps = _json.load(f)
            except (_json.JSONDecodeError, OSError, UnicodeDecodeError):
                return None

            if not isinstance(timestamps, list):
                return None

            # Load action sources (tolerate empty/corrupt files without crashing)
            action_sources: list[str] = []
            src_json = folder / "action-source.json"
            src_npy = folder / "action-source.npy"
            if src_json.exists() and src_json.stat().st_size > 0:
                try:
                    with open(src_json) as f:
                        action_sources = _json.load(f)
                except (_json.JSONDecodeError, OSError, UnicodeDecodeError):
                    action_sources = []
            elif src_npy.exists() and src_npy.stat().st_size > 0:
                try:
                    # allow_pickle needed for object-dtype string arrays saved by numpy
                    action_sources = np.load(str(src_npy), allow_pickle=True).tolist()
                except (ValueError, OSError, EOFError):
                    action_sources = []

            # Load record timestamps
            record_timestamps: list[float] = []
            ts_npy = folder / "timestamp.npy"
            if ts_npy.exists() and ts_npy.stat().st_size > 0:
                try:
                    record_timestamps = np.load(str(ts_npy)).tolist()
                except (ValueError, OSError, EOFError):
                    record_timestamps = []

            return {
                "timestamps": timestamps,
                "action_sources": action_sources,
                "record_timestamps": record_timestamps,
            }

        result = await asyncio.to_thread(_load)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No component_timestamps.json for episode {idx}",
            )
        _component_ts_cache[cache_key] = result
        while len(_component_ts_cache) > _COMPONENT_TS_CACHE_MAX:
            _component_ts_cache.popitem(last=False)
        return result

    # ------------------------------------------------------------------
    # Camera status & live feed
    # ------------------------------------------------------------------

    cam_stream = CameraStream()

    @app.get("/api/cameras")
    async def camera_status() -> dict[str, Any]:
        """Check whether RealSense cameras are connected."""

        def _query_cameras() -> list[dict[str, str]]:
            try:
                import pyrealsense2 as rs

                ctx = rs.context()
                devices = ctx.query_devices()
                return [
                    {
                        "serial": dev.get_info(rs.camera_info.serial_number),
                        "name": dev.get_info(rs.camera_info.name),
                        "firmware": dev.get_info(rs.camera_info.firmware_version),
                    }
                    for dev in devices
                ]
            except ImportError:
                return []
            except Exception:
                return []

        cameras = await asyncio.to_thread(_query_cameras)
        return {
            "online": len(cameras) > 0,
            "count": len(cameras),
            "devices": cameras,
            "streaming": cam_stream.running,
        }

    @app.post("/api/cameras/push")
    async def push_camera_frame(request: "Request") -> dict[str, str]:
        """Accept a JPEG frame from an external source (e.g. control loop).

        Usage from Python::

            import requests
            _, buf = cv2.imencode(".jpg", bgr_frame)
            requests.post("http://localhost:8888/api/cameras/push",
                          data=buf.tobytes(),
                          headers={"Content-Type": "image/jpeg"})
        """
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty body")
        ok = cam_stream.push_jpeg(body)
        if not ok:
            raise HTTPException(status_code=400, detail="Failed to decode JPEG")
        return {"status": "ok"}

    @app.post("/api/cameras/start")
    async def start_camera() -> dict[str, str]:
        """Try to open camera directly. Fails if another process owns it."""
        status = await asyncio.to_thread(cam_stream.start_direct)
        return {"status": status}

    @app.post("/api/cameras/stop")
    async def stop_camera() -> dict[str, str]:
        """Stop direct camera capture."""
        cam_stream.stop_direct()
        return {"status": "stopped"}

    @app.get("/api/cameras/frame")
    async def camera_frame(
        quality: int = Query(default=80, ge=1, le=100),
    ) -> Response:
        """Return latest camera frame as JPEG (from push or direct capture)."""
        if not cam_stream.has_frame:
            raise HTTPException(
                status_code=503,
                detail="No frames available. Either push frames via POST /api/cameras/push or start direct capture.",
            )
        jpeg = await asyncio.to_thread(cam_stream.get_frame_jpeg, quality)
        if jpeg is None:
            raise HTTPException(status_code=503, detail="No frame available")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/cameras/stream")
    async def camera_mjpeg_stream() -> StreamingResponse:
        """MJPEG stream from pushed or directly captured frames."""

        def _generate():
            while True:
                jpeg = cam_stream.get_frame_jpeg(quality=70)
                if jpeg is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(1.0 / 15)  # ~15 fps

        return StreamingResponse(
            _generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/tasks/{task_id}/episodes/{idx}/overlay")
    async def get_overlay(
        task_id: str,
        idx: int,
        alpha: float = Query(default=0.5, ge=0.0, le=1.0),
        quality: int = Query(default=85, ge=1, le=100),
        color_mode: str = Query(default="bgr", regex="^(normal|bgr|gbr|invert)$"),
    ) -> Response:
        """Return blended overlay: live camera frame + color-remapped dataset frame.

        ``color_mode`` remaps dataset frame channels before blending:
        - ``normal``: no change
        - ``bgr``: swap R↔B channels
        - ``gbr``: rotate channels G→R, B→G, R→B
        - ``invert``: invert all colors (255 - pixel)
        """
        _require_task(task_id)
        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Scan not ready")

        def _blend() -> bytes | None:
            live = cam_stream.get_frame_bgr()
            dataset = scanner.get_frame_bgr(idx)
            if live is None or dataset is None:
                return None
            h, w = dataset.shape[:2]
            live_resized = cv2.resize(live, (w, h))

            # Remap dataset frame colors
            if color_mode == "bgr":
                dataset_out = dataset[:, :, ::-1]  # BGR→RGB (swap R↔B)
            elif color_mode == "gbr":
                dataset_out = dataset[:, :, [1, 2, 0]]  # rotate channels
            elif color_mode == "invert":
                dataset_out = 255 - dataset
            else:
                dataset_out = dataset

            blended = cv2.addWeighted(live_resized, 1.0 - alpha, dataset_out, alpha, 0)
            ok, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes() if ok else None

        jpeg = await asyncio.to_thread(_blend)
        if jpeg is None:
            raise HTTPException(
                status_code=503,
                detail="Cannot blend — camera or dataset frame unavailable",
            )
        return Response(content=jpeg, media_type="image/jpeg")

    # ------------------------------------------------------------------
    # Replay control (Portal IPC)
    # ------------------------------------------------------------------

    @app.post("/api/replay/connect")
    async def replay_connect() -> dict[str, bool]:
        """Try to connect to the policy Portal server."""
        ok = await asyncio.to_thread(policy_conn.connect)
        return {"connected": ok}

    @app.post("/api/replay/disconnect")
    async def replay_disconnect() -> dict[str, bool]:
        """Detach from the policy controller without shutting it down."""
        policy_conn.disconnect()
        return {"connected": False}

    @app.get("/api/replay/status")
    async def replay_status() -> dict[str, bool]:
        """Return whether the policy connection is active."""
        ok = await asyncio.to_thread(policy_conn.check_connected)
        return {"connected": ok}

    @app.post("/api/replay/play")
    async def replay_play() -> dict[str, bool]:
        """Send 'start' to the policy control loop."""
        ok = await asyncio.to_thread(policy_conn.enter_state, "start")
        if not ok:
            raise HTTPException(
                status_code=503, detail="Policy not connected or command failed"
            )
        return {"success": True}

    @app.post("/api/replay/pause")
    async def replay_pause() -> dict[str, bool]:
        """Send 'pause' to the policy control loop."""
        ok = await asyncio.to_thread(policy_conn.enter_state, "pause")
        if not ok:
            raise HTTPException(
                status_code=503, detail="Policy not connected or command failed"
            )
        return {"success": True}

    @app.post("/api/replay/home")
    async def replay_home() -> dict[str, bool]:
        """Send 'home' to the policy control loop."""
        ok = await asyncio.to_thread(policy_conn.enter_state, "home")
        if not ok:
            raise HTTPException(
                status_code=503, detail="Policy not connected or command failed"
            )
        return {"success": True}

    @app.post("/api/replay/step")
    async def replay_step() -> dict[str, bool]:
        """Send 'step_once' to the policy control loop."""
        ok = await asyncio.to_thread(policy_conn.enter_state, "step_once")
        if not ok:
            raise HTTPException(
                status_code=503, detail="Policy not connected or command failed"
            )
        return {"success": True}

    @app.post("/api/replay/load")
    async def replay_load(request: Request) -> dict[str, Any]:
        """Load an episode for replay by index.

        Body: ``{"episode_idx": int, "task_id": str}``
        """
        body = await request.json()
        episode_idx = body.get("episode_idx")
        task_id = body.get("task_id")
        if episode_idx is None or task_id is None:
            raise HTTPException(
                status_code=400, detail="episode_idx and task_id required"
            )

        scanner = _get_scanner(task_id)
        if scanner is None or not scanner.ready:
            raise HTTPException(status_code=404, detail="Task not scanned or not ready")

        folder = scanner.get_episode_path(episode_idx)
        if folder is None:
            raise HTTPException(
                status_code=404, detail=f"Episode {episode_idx} not found"
            )

        ok = await asyncio.to_thread(
            policy_conn.set_replay_config, str(folder), "joint_position"
        )
        if not ok:
            raise HTTPException(
                status_code=503, detail="Policy not connected or command failed"
            )
        return {"success": True, "dataset_path": str(folder)}

    @app.post("/api/replay/task_command")
    async def replay_task_command(request: Request) -> dict[str, bool]:
        """Set the task command string on the policy.

        Body: ``{"command": str}``
        """
        body = await request.json()
        command = body.get("command")
        if not command or not isinstance(command, str):
            raise HTTPException(status_code=400, detail="command (string) required")

        ok = await asyncio.to_thread(policy_conn.set_task_command, command.strip())
        if not ok:
            raise HTTPException(
                status_code=503, detail="Policy not connected or command failed"
            )
        return {"success": True}

    @app.post("/api/replay/sync_to_init")
    async def replay_sync_to_init() -> dict[str, bool]:
        """Interpolate the robot to the loaded episode's first-frame joint state.

        Requires an episode to have been loaded via ``/api/replay/load`` first.
        The robot moves to the initial pose and then auto-pauses, ready for Play.
        """
        ok = await asyncio.to_thread(policy_conn.sync_to_init)
        if not ok:
            raise HTTPException(
                status_code=503,
                detail="Policy not connected, no episode loaded, or sync_to_init failed",
            )
        return {"success": True}

    # ------------------------------------------------------------------
    # Export to lerobot v2.1
    #
    # Shells out to ``minimal_policy/scripts/data/convert_gearraw_to_lerobot.py``
    # in a background thread and streams stdout into a module-scoped buffer
    # the UI polls via the /status endpoint. One job at a time per task id.
    # ------------------------------------------------------------------
    import subprocess as _subprocess

    _export_lerobot_lock = threading.Lock()
    _export_lerobot_state: dict[str, dict[str, Any]] = {}

    def _fresh_export_state() -> dict[str, Any]:
        return {
            "running": False,
            "stdout": "",
            "exit_code": None,
            "started_at": None,
            "finished_at": None,
            "command": None,
            "cwd": None,
            "error": None,
        }

    def _run_export_lerobot(task_id: str, cmd: list[str], cwd: str) -> None:
        try:
            proc = _subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            with _export_lerobot_lock:
                state = _export_lerobot_state.setdefault(task_id, _fresh_export_state())
                state["running"] = False
                state["exit_code"] = -1
                state["finished_at"] = time.time()
                state["error"] = f"failed to spawn: {exc}"
            return

        assert proc.stdout is not None
        for line in proc.stdout:
            with _export_lerobot_lock:
                state = _export_lerobot_state.setdefault(task_id, _fresh_export_state())
                state["stdout"] += line
        proc.wait()
        with _export_lerobot_lock:
            state = _export_lerobot_state.setdefault(task_id, _fresh_export_state())
            state["running"] = False
            state["exit_code"] = proc.returncode
            state["finished_at"] = time.time()

    @app.post("/api/tasks/{task_id}/export_lerobot/start")
    async def export_lerobot_start(task_id: str, request: "Request") -> dict[str, Any]:
        _require_task(task_id)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}

        minimal_policy_dir = str(body.get("minimal_policy_dir") or "").strip()
        input_root = str(body.get("input_root") or "").strip()
        output_dir = str(body.get("output_dir") or "").strip()
        task_name_raw = body.get("task_name")
        task_name = str(task_name_raw).strip() if task_name_raw else ""
        min_segment_length = int(body.get("min_segment_length") or 64)
        annotate_only = bool(body.get("annotate_only", True))

        if not minimal_policy_dir:
            raise HTTPException(status_code=422, detail="minimal_policy_dir required")
        if not input_root:
            raise HTTPException(status_code=422, detail="input_root required")
        if not output_dir:
            raise HTTPException(status_code=422, detail="output_dir required")

        mp_path = Path(minimal_policy_dir).expanduser()
        script_rel = "scripts/data/convert_gearraw_to_lerobot.py"
        script_path = mp_path / script_rel
        if not script_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"script not found under minimal_policy_dir: {script_path}",
            )

        cmd = [
            "uv",
            "run",
            "--no-sync",
            script_rel,
            "--input-root",
            str(Path(input_root).expanduser()),
            "--output-dir",
            str(Path(output_dir).expanduser()),
            "--min-segment-length",
            str(min_segment_length),
        ]
        if task_name:
            cmd.extend(["--task-name", task_name])
        if annotate_only:
            cmd.append("--annotate-only")

        with _export_lerobot_lock:
            state = _export_lerobot_state.setdefault(task_id, _fresh_export_state())
            if state["running"]:
                raise HTTPException(
                    status_code=409,
                    detail="export already running for this task",
                )
            state.update(_fresh_export_state())
            state["running"] = True
            state["started_at"] = time.time()
            state["command"] = shlex.join(cmd)
            state["cwd"] = str(mp_path)

        thread = threading.Thread(
            target=_run_export_lerobot,
            args=(task_id, cmd, str(mp_path)),
            daemon=True,
        )
        thread.start()
        return {"running": True, "command": shlex.join(cmd), "cwd": str(mp_path)}

    @app.get("/api/tasks/{task_id}/export_lerobot/status")
    def export_lerobot_status(task_id: str) -> dict[str, Any]:
        _require_task(task_id)
        with _export_lerobot_lock:
            state = _export_lerobot_state.get(task_id)
            if state is None:
                return _fresh_export_state()
            return dict(state)

    # ------------------------------------------------------------------
    # Static file mount (must be last so it doesn't shadow API routes)
    # ------------------------------------------------------------------

    if STATIC_DIR.is_dir():
        # Serve index.html explicitly with no-cache headers so every page
        # load picks up the latest bundle hash. The hashed JS/CSS assets
        # below can be cached aggressively since their filenames change
        # on every build.
        index_path = STATIC_DIR / "index.html"
        no_cache_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        @app.get("/", include_in_schema=False)
        async def _serve_index() -> FileResponse:
            return FileResponse(index_path, headers=no_cache_headers)

        @app.get("/index.html", include_in_schema=False)
        async def _serve_index_explicit() -> FileResponse:
            return FileResponse(index_path, headers=no_cache_headers)

        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
