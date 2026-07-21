"""Per-camera video recording for policy evaluation.

Records one MP4 per camera per episode with text overlay.
Writes mp4v first, then re-encodes to H.264 via ffmpeg on close
(OpenCV lacks H.264 encoder on most servers).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

CAMERA_MAPS = {
    "robocasa_panda_omron": {
        "video.res256_image_side_0": "side_left",
        "video.res256_image_side_1": "side_right",
        "video.res256_image_wrist_0": "wrist",
    },
}


def _reencode_h264(src: Path, label: str | None = None, label_color: str = "white") -> None:
    """Re-encode mp4v → H.264 in-place via ffmpeg.

    If label is given, burns it as a drawtext overlay on every frame
    (top-right corner, font size scales with frame width).
    No-op if ffmpeg is missing.
    """
    if not shutil.which("ffmpeg"):
        return
    tmp = src.with_suffix(".h264.mp4")
    vf_filters = []
    if label:
        vf_filters.append(
            f"drawtext=text='{label}'"
            f":fontcolor={label_color}"
            f":fontsize=w/18"
            f":x=w-tw-4:y=h/20"
            f":box=1:boxcolor=black@0.4:boxborderw=3"
        )
    vf_filters.append("format=yuv420p")
    ret = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-profile:v", "baseline", "-level", "3.0",
         "-vf", ",".join(vf_filters), "-an", str(tmp)],
        capture_output=True,
    )
    if ret.returncode == 0 and tmp.exists():
        tmp.rename(src)
    else:
        tmp.unlink(missing_ok=True)


class EpisodeVideoRecorder:
    """Records per-camera videos for one episode.

    Output: {episode_dir}/{cam}.mp4 + episode_config.json
    """

    def __init__(self, episode_dir: str, camera_map: dict[str, str],
                 fps: int = 10, task_name: str = "", model_name: str = ""):
        import cv2
        self._cv2 = cv2
        self._dir = Path(episode_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._camera_map = camera_map
        self._fps = fps
        self._task_name = task_name
        self._model_name = model_name
        self._success: bool | None = None  # None = not yet determined
        self._writers: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}

    def set_success(self, success: bool) -> None:
        self._success = success

    def record(self, obs: dict[str, Any]) -> None:
        cv2 = self._cv2
        for cam_key, name in self._camera_map.items():
            if cam_key not in obs:
                continue
            frame = obs[cam_key]
            if frame.ndim == 4:
                frame = frame[-1]
            frame = frame.copy()
            self._overlay(frame)
            if name not in self._writers:
                h, w = frame.shape[:2]
                path = self._dir / f"{name}.mp4"
                wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     self._fps, (w, h))
                self._writers[name] = wr
                self._paths[name] = path
            self._writers[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _overlay(self, frame) -> None:
        cv2 = self._cv2
        h, w = frame.shape[:2]
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, max(0.3, w / 640), max(1, int(w / 640 * 1.5))
        y = int(16 * sc)
        # Task + model name (top-left, blue) — burned during recording
        # Success/Fail is added at re-encode time via ffmpeg drawtext (see close())
        header = self._task_name
        if self._model_name:
            header = f"{header} | {self._model_name}" if header else self._model_name
        if header:
            cv2.putText(frame, header, (4, y), font, sc * 0.7, (30, 100, 220), th)  # blue (RGB)

    def save_config(self, success: bool, extra: dict | None = None) -> None:
        meta = {"success": success, "task_name": self._task_name}
        if extra:
            meta.update(extra)
        with open(self._dir / "episode_config.json", "w") as f:
            json.dump(meta, f, indent=2)

    def close(self) -> None:
        for wr in self._writers.values():
            wr.release()
        self._writers.clear()
        label = None
        label_color = "white"
        if self._success is not None:
            label = "Success" if self._success else "Fail"
            label_color = "green" if self._success else "red"
        for path in self._paths.values():
            _reencode_h264(path, label=label, label_color=label_color)
        self._paths.clear()

    @property
    def dir(self) -> Path:
        return self._dir


class BatchedVideoTracker:
    """Manages per-env recorders for batched rollouts.

    Episodes numbered globally: ep_000/, ep_001/, ...
    """

    def __init__(self, video_dir: str, n_envs: int, camera_map: dict[str, str],
                 fps: int = 10, task_name: str = "", model_name: str = "",
                 config_extra: dict | None = None):
        self._video_dir = Path(video_dir)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._n_envs = n_envs
        self._camera_map = camera_map
        self._fps = fps
        self._task_name = task_name
        self._model_name = model_name
        self._config_extra = config_extra or {}
        self._global_ep = 0
        self._recs: list[EpisodeVideoRecorder | None] = [None] * n_envs

    def start_all(self) -> None:
        for i in range(self._n_envs):
            self._start(i)

    def record(self, obs: dict[str, Any]) -> None:
        for i in range(self._n_envs):
            rec = self._recs[i]
            if rec is None:
                continue
            env_obs = {k: obs[k][i] for k in self._camera_map if k in obs}
            rec.record(env_obs)

    def set_success(self, env_idx: int, success: bool) -> None:
        if self._recs[env_idx]:
            self._recs[env_idx].set_success(success)

    def finish_episode(self, env_idx: int, success: bool, more_episodes: bool = True,
                       extra: dict | None = None) -> None:
        rec = self._recs[env_idx]
        if rec is None:
            return
        rec.set_success(success)
        rec.close()
        merged = {**self._config_extra, **(extra or {})}
        rec.save_config(success, merged or None)
        if more_episodes:
            self._start(env_idx)
        else:
            self._recs[env_idx] = None

    def close(self) -> None:
        for i in range(self._n_envs):
            if self._recs[i]:
                # Unused recorder (no frames written) — remove ghost dir
                rec = self._recs[i]
                rec.close()
                if not any(rec.dir.glob("*.mp4")):
                    import shutil
                    shutil.rmtree(rec.dir, ignore_errors=True)
                self._recs[i] = None

    def _start(self, env_idx: int) -> None:
        ep = self._global_ep
        self._global_ep += 1
        self._recs[env_idx] = EpisodeVideoRecorder(
            str(self._video_dir / f"ep_{ep:03d}"),
            self._camera_map, self._fps, self._task_name, self._model_name,
        )
