"""Episode scanner for overlay visualization.

Scans task data directories for episodes containing video files,
extracts first frames, and serves them as JPEG bytes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Hashable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_FILENAME = "top_camera-images-rgb.mp4"

CAMERA_FILENAMES: dict[str, str] = {
    "top": "top_camera-images-rgb.mp4",
    "left": "left_camera-images-rgb.mp4",
    "right": "right_camera-images-rgb.mp4",
}

# Max forward frames to read-and-discard instead of seeking
_SMALL_JUMP_THRESHOLD = 5


class EpisodeScanner:
    """Scans a task data directory for episodes and caches first frames.

    Thread-safe. Scanning runs in a background thread so it never blocks
    the event loop.  Frame data is served as JPEG bytes via ``get_frame_jpeg``.
    """

    def __init__(self, data_root: str | Path, *, max_cache: int = 64) -> None:
        self._data_root = Path(data_root)
        self._max_cache = max_cache

        # Populated by the background scan thread.
        self._entries: list[tuple[Path, Path]] = []  # (folder, video_path)
        self._lock = threading.Lock()
        self._scan_done = 0
        self._scan_total = 0
        self._ready = False

        # Bounded LRU cache: idx -> BGR numpy frame
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # LRU cache of open VideoCapture handles for frame seeking (max 12)
        # Values are (cap, last_read_frame_num) tuples; last_read is -1 when unknown.
        self._vcap_cache: OrderedDict[Hashable, tuple[cv2.VideoCapture, int]] = (
            OrderedDict()
        )
        self._vcap_cache_lock = threading.Lock()  # protects _vcap_cache dict
        # Per-camera locks so left/top/right can seek in parallel
        self._vcap_locks: dict[Hashable, threading.Lock] = {}
        self._vcap_locks_guard = threading.Lock()  # protects _vcap_locks dict

        # Per-frame JPEG cache: (idx, camera, frame_num, quality) -> JPEG bytes
        self._jpeg_cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._jpeg_cache_lock = threading.Lock()
        self._jpeg_cache_max = 200

        # Thread pool for parallel camera reads
        self._camera_pool = ThreadPoolExecutor(max_workers=3)

        # Episode-level pre-decoded cache (1 episode at a time)
        # {idx: {camera: [jpeg_bytes_frame_0, jpeg_bytes_frame_1, ...]}}
        self._episode_cache: dict[int, dict[str, list[bytes | None]]] = {}
        self._episode_cache_lock = threading.Lock()
        self._episode_preload_idx: int | None = None  # currently preloading
        self._episode_preload_progress: tuple[int, int] = (0, 0)  # (done, total)

        self._scan_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> str:
        """Return the scanner's dataset root as a string path."""
        return str(self._data_root)

    def start(self) -> None:
        """Begin scanning in a background thread.  No-op if already running."""
        if self._scan_thread is not None:
            return
        self._scan_thread = threading.Thread(target=self._bg_scan, daemon=True)
        self._scan_thread.start()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def progress(self) -> tuple[int, int]:
        """Return ``(done, total)`` for the current scan."""
        return self._scan_done, self._scan_total

    @property
    def episodes(self) -> list[dict[str, str | int]]:
        """Return lightweight episode metadata (index + folder name)."""
        with self._lock:
            return [
                {"idx": i, "folder": folder.name}
                for i, (folder, _) in enumerate(self._entries)
            ]

    def get_frame_jpeg(self, idx: int, *, quality: int = 85) -> bytes | None:
        """Return the first frame of episode *idx* as JPEG bytes, or ``None``."""
        frame = self._load_frame(idx)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return buf.tobytes()

    def get_frame_bgr(self, idx: int) -> np.ndarray | None:
        """Return the raw BGR numpy frame for episode *idx*."""
        return self._load_frame(idx)

    def total_episodes(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_episode_path(self, idx: int) -> Path | None:
        """Return the folder path for episode *idx*, or ``None``."""
        with self._lock:
            if idx < 0 or idx >= len(self._entries):
                return None
            return self._entries[idx][0]

    def remove_episode(self, idx: int) -> bool:
        """Drop episode *idx* from the in-memory entry list and clear caches.

        Subsequent indices shift down by one, so all idx-keyed caches are
        invalidated wholesale to keep things consistent. Returns True if
        an entry was removed.
        """
        with self._lock:
            if idx < 0 or idx >= len(self._entries):
                return False
            del self._entries[idx]
        self._frame_cache.clear()
        with self._jpeg_cache_lock:
            self._jpeg_cache.clear()
        with self._vcap_cache_lock:
            for cap, _ in self._vcap_cache.values():
                try:
                    cap.release()
                except Exception:
                    pass
            self._vcap_cache.clear()
        with self._episode_cache_lock:
            self._episode_cache.clear()
            self._episode_preload_idx = None
            self._episode_preload_progress = (0, 0)
        return True

    def get_actions(self, idx: int) -> np.ndarray | None:
        """Load action data for episode *idx*.

        Returns shape ``(T, D)`` numpy array, or ``None``.
        Auto-detects format: folder with npy files, parquet, single npy/npz.
        """
        folder = self.get_episode_path(idx)
        if folder is None:
            return None
        return self._load_actions(folder)

    @staticmethod
    def _load_actions(folder: Path) -> np.ndarray | None:
        """Load actions from an episode folder (same logic as LerobotReplayPolicy)."""
        # Raw teleop folder with separate npy files
        left_path = folder / "action-left-pos.npy"
        right_path = folder / "action-right-pos.npy"
        if left_path.exists() and right_path.exists():
            try:
                left = np.load(str(left_path), allow_pickle=True)
                right = np.load(str(right_path), allow_pickle=True)
                return np.concatenate([left, right], axis=1)
            except Exception as e:
                logger.warning("Failed to load action npy files from %s: %s", folder, e)
                return None

        # Single parquet file
        parquet_files = sorted(folder.glob("*.parquet"))
        if parquet_files:
            try:
                import pandas as pd

                df = pd.read_parquet(parquet_files[0])
                if "action" in df.columns:
                    return np.stack(df["action"].values)
            except Exception as e:
                logger.warning("Failed to load parquet from %s: %s", folder, e)
                return None

        # Single npy file named "actions.npy"
        action_npy = folder / "actions.npy"
        if action_npy.exists():
            try:
                return np.load(str(action_npy), allow_pickle=True)
            except Exception as e:
                logger.warning("Failed to load actions.npy from %s: %s", folder, e)
                return None

        return None

    def get_states(self, idx: int) -> np.ndarray | None:
        """Load state (observation) data for episode *idx*.

        Returns shape ``(T, D)`` numpy array, or ``None``.
        """
        folder = self.get_episode_path(idx)
        if folder is None:
            return None
        return self._load_states(folder)

    @staticmethod
    def _load_states(folder: Path) -> np.ndarray | None:
        """Load observation state data from an episode folder."""
        parts = []
        for name in [
            "left-joint_pos.npy",
            "left-gripper_pos.npy",
            "right-joint_pos.npy",
            "right-gripper_pos.npy",
        ]:
            path = folder / name
            if not path.exists():
                continue
            try:
                arr = np.load(str(path), allow_pickle=True)
                if arr.ndim == 1:
                    arr = arr[:, None]
                parts.append(arr)
            except Exception as e:
                logger.warning("Failed to load %s from %s: %s", name, folder, e)
        if not parts:
            return None
        try:
            return np.concatenate(parts, axis=1)
        except ValueError as e:
            logger.warning("Failed to concatenate state arrays from %s: %s", folder, e)
            return None

    def get_action_source(self, idx: int) -> np.ndarray | None:
        """Load per-step action source labels for episode *idx*.

        Returns 1D string array, or ``None``.
        """
        folder = self.get_episode_path(idx)
        if folder is None:
            return None
        return self._load_action_source(folder)

    @staticmethod
    def _load_action_source(folder: Path) -> np.ndarray | None:
        """Load action-source labels from an episode folder."""
        npy_path = folder / "action-source.npy"
        if npy_path.exists():
            try:
                return np.load(str(npy_path), allow_pickle=True)
            except Exception as e:
                logger.warning(
                    "Failed to load action-source.npy from %s: %s", folder, e
                )
        json_path = folder / "action-source.json"
        if json_path.exists():
            try:
                import json as _json

                with open(json_path) as f:
                    return np.array(_json.load(f))
            except Exception as e:
                logger.warning(
                    "Failed to load action-source.json from %s: %s", folder, e
                )
        return None

    def get_video_info(self, idx: int) -> dict | None:
        """Return video metadata for episode *idx*, or ``None``.

        Returns dict with keys: total_frames, fps, duration_s.
        """
        with self._lock:
            if idx < 0 or idx >= len(self._entries):
                return None
            video_path = self._entries[idx][1]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration_s = total_frames / fps if fps > 0 else 0.0
            return {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_s": round(duration_s, 3),
            }
        finally:
            cap.release()

    def get_video_frame_jpeg(
        self,
        idx: int,
        frame_num: int,
        *,
        quality: int = 85,
        scale: float = 1.0,
    ) -> bytes | None:
        """Return JPEG bytes of a specific video frame, or ``None``.

        Uses an LRU cache of open VideoCapture objects (max 12) to avoid
        repeatedly opening the same file when scrubbing through frames.
        Sequential reads (frame N+1 after frame N) skip the expensive
        ``cap.set()`` call entirely.

        Args:
            scale: Downscale factor (0 < scale <= 1.0). When < 1.0 the frame
                   is resized before JPEG encoding for smaller payloads.
        """
        with self._lock:
            if idx < 0 or idx >= len(self._entries):
                return None
            video_path = self._entries[idx][1]

        frame = self._read_frame_optimized(idx, video_path, frame_num)
        if frame is None:
            return None
        return self._encode_jpeg(frame, quality=quality, scale=scale)

    def get_camera_frame_jpeg(
        self,
        idx: int,
        camera: str,
        frame_num: int,
        *,
        quality: int = 85,
        scale: float = 1.0,
    ) -> bytes | None:
        """Return JPEG bytes of a specific camera frame, or ``None``.

        Uses per-frame JPEG cache and per-camera locks for parallel access.

        Args:
            idx: Episode index.
            camera: One of "top", "left", "right".
            frame_num: Frame number to seek to.
            quality: JPEG quality (1-100).
            scale: Downscale factor (0 < scale <= 1.0).
        """
        if camera not in CAMERA_FILENAMES:
            return None

        # Check per-frame JPEG cache
        jpeg_key = (idx, camera, frame_num, quality)
        cached = self._jpeg_cache_get(jpeg_key)
        if cached is not None:
            return cached

        folder = self.get_episode_path(idx)
        if folder is None:
            return None

        video_path = folder / CAMERA_FILENAMES[camera]
        if not video_path.exists():
            return None

        cache_key = (idx, camera)
        frame = self._read_frame_optimized(cache_key, video_path, frame_num)
        if frame is None:
            return None
        jpeg_bytes = self._encode_jpeg(frame, quality=quality, scale=scale)
        if jpeg_bytes is not None:
            self._jpeg_cache_put(jpeg_key, jpeg_bytes)
        return jpeg_bytes

    def get_camera_frames_jpeg(
        self, idx: int, frame_num: int, *, quality: int = 60
    ) -> dict[str, bytes | None]:
        """Return all 3 camera frames as JPEG bytes in one call.

        Uses ThreadPoolExecutor(3) to read left/top/right in parallel.

        Returns:
            Dict with keys "left", "top", "right" mapping to JPEG bytes or None.
        """
        # Fast path: check episode cache (no I/O, no threads needed)
        ep_cache = self._episode_cache.get(idx) if idx in self._episode_cache else None
        if ep_cache is not None:
            results: dict[str, bytes | None] = {}
            for cam in CAMERA_FILENAMES:
                frames = ep_cache.get(cam)
                if frames and 0 <= frame_num < len(frames):
                    results[cam] = frames[frame_num]
                else:
                    results[cam] = None
            return results

        def _read_one(camera: str) -> tuple[str, bytes | None]:
            jpeg = self.get_camera_frame_jpeg(idx, camera, frame_num, quality=quality)
            return camera, jpeg

        results: dict[str, bytes | None] = {}
        futures = [
            self._camera_pool.submit(_read_one, cam) for cam in CAMERA_FILENAMES
        ]
        for future in futures:
            camera, jpeg = future.result()
            results[camera] = jpeg

        return results

    def preload_frames(
        self, idx: int, current_frame: int, *, ahead: int = 5, quality: int = 60
    ) -> None:
        """Pre-decode a window of frames ahead into the JPEG cache.

        Populates the cache for frames [current_frame+1 .. current_frame+ahead]
        for all cameras so subsequent scrub requests hit the cache.
        Should be called asynchronously (from a background thread).
        """
        def _preload_camera(camera: str) -> None:
            for fn in range(current_frame + 1, current_frame + ahead + 1):
                jpeg_key = (idx, camera, fn, quality)
                if self._jpeg_cache_get(jpeg_key) is not None:
                    continue  # already cached
                self.get_camera_frame_jpeg(idx, camera, fn, quality=quality)

        futures = [
            self._camera_pool.submit(_preload_camera, cam)
            for cam in CAMERA_FILENAMES
        ]
        for f in futures:
            try:
                f.result()
            except Exception:
                pass  # best-effort preloading

    # ------------------------------------------------------------------
    # Episode-level preload (decode entire episode at low res)
    # ------------------------------------------------------------------

    def preload_episode(
        self, idx: int, *, scale: float = 0.5, quality: int = 40
    ) -> None:
        """Pre-decode ALL frames for every camera in episode *idx*.

        Runs synchronously (call from a background thread). Evicts any
        previous episode cache before starting. Stores low-res JPEGs keyed
        by ``{camera: [jpeg_bytes, ...]}``.
        """
        folder = self.get_episode_path(idx)
        if folder is None:
            return

        # Collect cameras that have video files
        camera_paths: list[tuple[str, Path]] = []
        total_frames = 0
        for cam, fname in CAMERA_FILENAMES.items():
            vpath = folder / fname
            if vpath.exists():
                cap = cv2.VideoCapture(str(vpath))
                if cap.isOpened():
                    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_frames += nf
                    camera_paths.append((cam, vpath))
                cap.release()

        if not camera_paths or total_frames == 0:
            return

        with self._episode_cache_lock:
            self._episode_preload_idx = idx
            self._episode_preload_progress = (0, total_frames)
            # Evict previous cache
            self._episode_cache.clear()

        result: dict[str, list[bytes | None]] = {}
        done = 0

        for cam, vpath in camera_paths:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                continue
            nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames: list[bytes | None] = []
            for _ in range(nf):
                # Bail out if a newer preload was requested
                with self._episode_cache_lock:
                    if self._episode_preload_idx != idx:
                        cap.release()
                        return
                ok, frame = cap.read()
                if ok and frame is not None:
                    jpeg = self._encode_jpeg(frame, quality=quality, scale=scale)
                    frames.append(jpeg)
                else:
                    frames.append(None)
                done += 1
                with self._episode_cache_lock:
                    self._episode_preload_progress = (done, total_frames)
            cap.release()
            result[cam] = frames

        with self._episode_cache_lock:
            # Only store if we're still the active preload
            if self._episode_preload_idx == idx:
                self._episode_cache[idx] = result
                self._episode_preload_progress = (total_frames, total_frames)
                logger.info(
                    "Episode %d preloaded: %d frames across %d cameras (~%.1f MB)",
                    idx,
                    total_frames,
                    len(camera_paths),
                    sum(
                        len(b) for fl in result.values() for b in fl if b is not None
                    )
                    / 1e6,
                )

    def get_episode_preload_status(self, idx: int) -> dict:
        """Return preload progress for episode *idx*."""
        with self._episode_cache_lock:
            if idx in self._episode_cache:
                total = sum(len(v) for v in self._episode_cache[idx].values())
                return {"ready": True, "done": total, "total": total}
            if self._episode_preload_idx == idx:
                done, total = self._episode_preload_progress
                return {"ready": False, "done": done, "total": total}
        return {"ready": False, "done": 0, "total": 0}

    def _episode_cache_get(
        self, idx: int, camera: str, frame_num: int
    ) -> bytes | None:
        """Look up a frame in the episode cache. Returns JPEG bytes or None."""
        with self._episode_cache_lock:
            ep = self._episode_cache.get(idx)
            if ep is None:
                return None
            frames = ep.get(camera)
            if frames is None or frame_num < 0 or frame_num >= len(frames):
                return None
            return frames[frame_num]

    # ------------------------------------------------------------------
    # JPEG cache helpers
    # ------------------------------------------------------------------

    def _jpeg_cache_get(self, key: tuple) -> bytes | None:
        with self._jpeg_cache_lock:
            if key in self._jpeg_cache:
                self._jpeg_cache.move_to_end(key)
                return self._jpeg_cache[key]
        return None

    def _jpeg_cache_put(self, key: tuple, data: bytes) -> None:
        with self._jpeg_cache_lock:
            if key in self._jpeg_cache:
                self._jpeg_cache.move_to_end(key)
                return
            while len(self._jpeg_cache) >= self._jpeg_cache_max:
                self._jpeg_cache.popitem(last=False)
            self._jpeg_cache[key] = data

    # ------------------------------------------------------------------
    # VideoCapture LRU cache (max 12 open handles)
    # ------------------------------------------------------------------

    def _get_vcap_lock(self, cache_key: Hashable) -> threading.Lock:
        """Return a per-VideoCapture lock, creating one if needed."""
        with self._vcap_locks_guard:
            if cache_key not in self._vcap_locks:
                self._vcap_locks[cache_key] = threading.Lock()
            return self._vcap_locks[cache_key]

    def _get_video_cap(
        self, cache_key: Hashable, video_path: Path
    ) -> tuple[cv2.VideoCapture, int] | None:
        """Return a cached ``(cap, last_read_frame)`` for *cache_key*.

        Opens a new VideoCapture if not cached, with ``last_read_frame = -1``.
        """
        with self._vcap_cache_lock:
            if cache_key in self._vcap_cache:
                self._vcap_cache.move_to_end(cache_key)
                return self._vcap_cache[cache_key]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video for seeking: %s", video_path)
            return None

        entry = (cap, -1)
        with self._vcap_cache_lock:
            # Evict oldest if at capacity
            while len(self._vcap_cache) >= 12:
                _, (old_cap, _) = self._vcap_cache.popitem(last=False)
                old_cap.release()
            self._vcap_cache[cache_key] = entry

        return entry

    def _update_vcap_last_frame(
        self, cache_key: Hashable, cap: cv2.VideoCapture, frame_num: int
    ) -> None:
        """Update the last-read frame number for a cached VideoCapture."""
        with self._vcap_cache_lock:
            if cache_key in self._vcap_cache:
                self._vcap_cache[cache_key] = (cap, frame_num)

    # ------------------------------------------------------------------
    # Optimised frame reading (sequential / small-jump / seek)
    # ------------------------------------------------------------------

    def _read_frame_optimized(
        self, cache_key: Hashable, video_path: Path, frame_num: int
    ) -> np.ndarray | None:
        """Read *frame_num* with sequential-read and small-jump optimizations.

        - If *frame_num* == last_read + 1  ->  just ``cap.read()`` (no seek).
        - If 1 < delta <= _SMALL_JUMP_THRESHOLD  ->  read-and-discard.
        - Otherwise  ->  ``cap.set(CAP_PROP_POS_FRAMES, frame_num)`` then read.

        Returns the decoded BGR frame, or ``None`` on failure.
        """
        result = self._get_video_cap(cache_key, video_path)
        if result is None:
            return None
        cap, last_frame = result

        t0 = time.monotonic()
        seek_strategy = "seq"

        per_cap_lock = self._get_vcap_lock(cache_key)
        with per_cap_lock:
            delta = frame_num - last_frame
            if delta == 1:
                # Sequential: next frame is already queued in the decoder
                seek_strategy = "seq"
            elif 1 < delta <= _SMALL_JUMP_THRESHOLD:
                # Small forward jump: faster to read-and-discard than seek
                seek_strategy = "skip"
                for _ in range(delta - 1):
                    cap.read()  # discard intermediate frames
            else:
                # Large jump or backward: must seek
                seek_strategy = "seek"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            t_seek = time.monotonic()
            ok, frame = cap.read()
            t_read = time.monotonic()

        if not ok or frame is None:
            logger.debug(
                "Frame read failed: key=%s frame=%d strategy=%s",
                cache_key,
                frame_num,
                seek_strategy,
            )
            return None

        # Update tracker
        self._update_vcap_last_frame(cache_key, cap, frame_num)

        logger.debug(
            "Frame %d [%s]: seek=%.1fms read=%.1fms total=%.1fms",
            frame_num,
            seek_strategy,
            (t_seek - t0) * 1000,
            (t_read - t_seek) * 1000,
            (t_read - t0) * 1000,
        )
        return frame

    @staticmethod
    def _encode_jpeg(
        frame: np.ndarray, *, quality: int = 85, scale: float = 1.0
    ) -> bytes | None:
        """Encode a BGR frame as JPEG bytes, optionally downscaling first."""
        t0 = time.monotonic()

        if 0 < scale < 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        t_resize = time.monotonic()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        t_encode = time.monotonic()

        if not ok:
            return None

        logger.debug(
            "JPEG encode: resize=%.1fms encode=%.1fms size=%dB",
            (t_resize - t0) * 1000,
            (t_encode - t_resize) * 1000,
            len(buf),
        )
        return buf.tobytes()

    # ------------------------------------------------------------------
    # Background scan
    # ------------------------------------------------------------------

    def _bg_scan(self) -> None:
        root = self._data_root
        if not root.exists():
            logger.warning("Data path not found: %s", root)
            self._ready = True
            return

        if (root / VIDEO_FILENAME).exists():
            folders = [root]
        else:
            folders = sorted(p for p in root.iterdir() if p.is_dir())
        self._scan_total = len(folders)
        logger.info("Scanning %d folders under %s", self._scan_total, root)

        last_log = 0.0
        for i, folder in enumerate(folders):
            matches = sorted(folder.rglob(VIDEO_FILENAME))
            if matches:
                with self._lock:
                    self._entries.append((folder, matches[0]))
            self._scan_done = i + 1

            now = time.time()
            if now - last_log > 1.0 or self._scan_done == self._scan_total:
                last_log = now
                logger.info(
                    "Scan progress: %d/%d (%d%%)",
                    self._scan_done,
                    self._scan_total,
                    int(self._scan_done / self._scan_total * 100)
                    if self._scan_total
                    else 100,
                )

        with self._lock:
            count = len(self._entries)
        if count == 0:
            logger.warning("No episodes found under %s", root)
        else:
            logger.info("Found %d episodes with video", count)

        self._ready = True

    # ------------------------------------------------------------------
    # Frame loading with bounded LRU cache
    # ------------------------------------------------------------------

    def _load_frame(self, idx: int) -> np.ndarray | None:
        with self._lock:
            if idx < 0 or idx >= len(self._entries):
                return None
            # Check cache (and move to end for LRU ordering)
            if idx in self._frame_cache:
                self._frame_cache.move_to_end(idx)
                return self._frame_cache[idx]
            video_path = self._entries[idx][1]

        frame = self._read_first_frame(video_path)
        if frame is None:
            return None

        with self._lock:
            # Evict oldest if full
            while len(self._frame_cache) >= self._max_cache:
                self._frame_cache.popitem(last=False)
            self._frame_cache[idx] = frame

        return frame

    @staticmethod
    def _read_first_frame(video_path: Path) -> np.ndarray | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video: %s", video_path)
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            logger.warning("Failed to read first frame: %s", video_path)
            return None
        return frame
