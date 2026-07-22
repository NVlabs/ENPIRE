# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# uv run python scripts/smoke_test_camera_streams.py --source rpc 
# --scale-zed-to-640480

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import portal

from enpire.env.forge.robot.camera_factory import create_camera


CAMERAS = ("top", "left", "right")
_ZED_NATIVE_SIZES = {
    "HD2K": (2208, 1242),
    "HD1200": (1920, 1200),
    "HD1080": (1920, 1080),
    "HD720": (1280, 720),
    "SVGA": (960, 600),
    "VGA": (672, 376),
}


def _convert_to_h264_in_place(path: Path) -> None:
    tmp = path.with_name(f"{path.stem}_temp{path.suffix}")
    path.rename(tmp)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(tmp),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp.unlink()
    except Exception:
        if path.exists():
            path.unlink()
        tmp.rename(path)
        raise


def _validate_frame(camera: str, frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"{camera}: expected HxWx3 image, got shape={frame.shape}")
    if frame.shape[0] < 10 or frame.shape[1] < 10:
        raise RuntimeError(f"{camera}: invalid tiny image {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def _make_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    return writer


def _native_top_resolution() -> tuple[int, int]:
    name = os.environ.get("CAP_TOP_ZED_NATIVE_RESOLUTION", os.environ.get("CAP_ZED_NATIVE_RESOLUTION", "HD720"))
    native = _ZED_NATIVE_SIZES.get(str(name).upper())
    if native is None:
        supported = ", ".join(sorted(_ZED_NATIVE_SIZES))
        raise ValueError(f"Unsupported CAP_TOP_ZED_NATIVE_RESOLUTION={name!r}. Supported: {supported}")
    return native


class _DirectCapture:
    def __init__(self, *, scale_zed_to_640480: bool):
        top_resolution = (640, 480) if scale_zed_to_640480 else _native_top_resolution()
        self._cameras = {
            "top": create_camera("top", resolution=top_resolution, enable_depth=False),
            "left": create_camera("left", resolution=(640, 480), enable_depth=False),
            "right": create_camera("right", resolution=(640, 480), enable_depth=False),
        }

    def get_frame(self, camera: str) -> np.ndarray:
        data = self._cameras[camera].read()
        if data is None or data.images.get("rgb") is None:
            raise RuntimeError(f"{camera}: no RGB frame")
        return _validate_frame(camera, data.images["rgb"])

    def close(self) -> None:
        for camera in self._cameras.values():
            try:
                camera.stop()
            except Exception:
                pass


class _RpcCapture:
    def __init__(self, *, host: str, port: int):
        self._client = portal.Client(f"{host}:{port}")

    def get_frame(self, camera: str) -> np.ndarray:
        return _validate_frame(camera, np.asarray(self._client.get_camera_image(camera).result()))

    def close(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a short multi-camera smoke-test video.")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-h264", action="store_true", help="skip ffmpeg H.264 conversion at the end")
    parser.add_argument(
        "--scale-zed-to-640480",
        action="store_true",
        help="for the top ZED only, use the same center-crop + resize path as CAP (640x480 output)",
    )
    parser.add_argument(
        "--source",
        choices=("direct", "rpc"),
        default="direct",
        help="direct hardware capture by default; rpc uses cap_server frames instead",
    )
    parser.add_argument("--host", default="127.0.0.1", help="cap_server host when --source rpc")
    parser.add_argument("--port", type=int, default=8300, help="cap_server port when --source rpc")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path("tmp") / "camera_smoke_test" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = (
        _DirectCapture(scale_zed_to_640480=args.scale_zed_to_640480)
        if args.source == "direct"
        else _RpcCapture(host=args.host, port=args.port)
    )
    print(
        f"[smoke] source={args.source} "
        f"top_mode={'640x480 crop+resize' if args.scale_zed_to_640480 else 'native-resolution'}"
    )

    camera_writers: dict[str, cv2.VideoWriter] = {}
    combined_writer: cv2.VideoWriter | None = None
    frame_counts = {cam: 0 for cam in CAMERAS}

    period = 1.0 / max(args.fps, 0.1)
    deadline = time.time() + args.duration

    try:
        while time.time() < deadline:
            loop_start = time.time()
            frames_rgb: dict[str, np.ndarray] = {}

            for cam in CAMERAS:
                frame = capture.get_frame(cam)
                frames_rgb[cam] = frame
                frame_counts[cam] += 1

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer = camera_writers.get(cam)
                if writer is None:
                    writer = _make_writer(out_dir / f"{cam}.mp4", (frame_bgr.shape[1], frame_bgr.shape[0]), args.fps)
                    camera_writers[cam] = writer
                writer.write(frame_bgr)

            target_h = max(frame.shape[0] for frame in frames_rgb.values())
            padded = []
            for cam in CAMERAS:
                rgb = frames_rgb[cam]
                if rgb.shape[0] != target_h:
                    pad_h = target_h - rgb.shape[0]
                    rgb = np.pad(rgb, ((0, pad_h), (0, 0), (0, 0)), mode="constant")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"{cam} {frames_rgb[cam].shape[1]}x{frames_rgb[cam].shape[0]}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                padded.append(bgr)

            combined = np.hstack(padded)
            if combined_writer is None:
                combined_writer = _make_writer(
                    out_dir / "combined.mp4",
                    (combined.shape[1], combined.shape[0]),
                    args.fps,
                )
            combined_writer.write(combined)

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        for writer in camera_writers.values():
            writer.release()
        if combined_writer is not None:
            combined_writer.release()
        capture.close()

    if not args.no_h264:
        for path in [*(out_dir / f"{cam}.mp4" for cam in CAMERAS), out_dir / "combined.mp4"]:
            _convert_to_h264_in_place(path)

    print("[smoke] wrote:")
    for cam in CAMERAS:
        print(f"  {cam:>5}: {out_dir / f'{cam}.mp4'}  ({frame_counts[cam]} frames)")
    print(f"  {'all':>5}: {out_dir / 'combined.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
