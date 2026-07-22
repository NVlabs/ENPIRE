# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compress a video to H.264 in a commanded shape in a separate daemon process aysnchornously

"""

from pathlib import Path
import subprocess
import cv2
import multiprocessing as mp
import numpy as np
from enpire.env.forge.robot.constants import DEFAULT_COMPRESSED_VIDEO_SHAPE


def compress_image(image, compressed_shape=DEFAULT_COMPRESSED_VIDEO_SHAPE):
    return cv2.resize(image, compressed_shape, interpolation=cv2.INTER_LINEAR)


def blank_compressed_image(compressed_shape=DEFAULT_COMPRESSED_VIDEO_SHAPE):
    return np.zeros((compressed_shape[1], compressed_shape[0], 3), dtype=np.uint8)


# Marker filename the converter publishes inside each episode dir once every
# mp4 in that dir has been finalized (post-rename, post-ffmpeg). Consumers
# that need to operate on the FINAL mp4 contents — e.g. the auto-reward
# truncate path's _trim_mp4s_in_dir in rl/auto_reward.py — wait for this
# marker instead of heuristically polling file sizes (which raced against
# ffmpeg's mid-convert pauses and produced silently-untrimmed videos on ~40%
# of successful ziptie auto-reward episodes). Per-episode-dir marker is safe
# because _video_worker_loop runs converts on a single-threaded executor
# (max_workers=1), so no concurrent convert ever touches the same dir.
VIDEO_CONVERT_DONE_MARKER = ".videos_converted"


def _convert_videos_to_h264_worker(episode_path: str, compressed_shape: tuple = DEFAULT_COMPRESSED_VIDEO_SHAPE) -> None:
    ep = Path(episode_path)
    marker = ep / VIDEO_CONVERT_DONE_MARKER
    try:
        video_files = sorted(ep.glob("*-images-rgb.mp4"))
        if not video_files:
            return

        print(f"[Video] Converting {len(video_files)} video(s) to {compressed_shape} in H.264...")
        for video_file in video_files:
            temp_file = video_file.parent / f"{video_file.stem}_temp{video_file.suffix}"
            video_file.rename(temp_file)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(temp_file),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-vf",
                        f"scale={compressed_shape[0]}:{compressed_shape[1]}",
                        "-pix_fmt",
                        "yuv420p",
                        str(video_file),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                temp_file.unlink()
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if isinstance(e, FileNotFoundError):
                    print("[Video] Warning: ffmpeg not found, keeping original format")
                else:
                    print(f"[Video] Warning: conversion failed for {video_file.name}, keeping original")
                temp_file.rename(video_file)
    finally:
        # Touch the marker no matter what — including on the early-return path
        # (no mp4s in the dir) and the per-file ffmpeg-failure path (original
        # mp4 was renamed back). Waiters time out gracefully if we ever fail
        # to reach this finally block, but in practice we always do.
        try:
            marker.touch(exist_ok=True)
        except OSError as exc:
            print(f"[Video] Warning: could not write {marker}: {exc}")


def _video_worker_loop(
    video_queue: mp.Queue, ack_queue: mp.Queue, compressed_shape=DEFAULT_COMPRESSED_VIDEO_SHAPE
) -> None:
    # Lower this subprocess's scheduling priority so its CPU bursts (cv2
    # frame writes during episodes + ffmpeg H.264 re-encodes at episode
    # boundaries) yield to the env loop when both want the CPU. The kernel
    # still runs us when no higher-priority work is ready, and crucially we
    # remain free to use ANY core — much better than affinity-pinning to
    # one core, which would starve us during episode bursts and back up
    # the queue (causing env.reset's _flush_video_writers to block for
    # hundreds of ms while ffmpeg drains on a single core).
    # POSIX-only; ignore quietly on platforms without os.nice.
    import os

    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass

    import threading
    from concurrent.futures import ThreadPoolExecutor

    writers: dict[tuple[str, str], cv2.VideoWriter] = {}
    writers_lock = threading.Lock()

    def release_writers(tmp_dir: str | None = None) -> None:
        # tmp_dir=None -> release ALL writers (shutdown). Otherwise release only the writers
        # for that episode's tmp dir, so an async per-episode flush can finalize episode N's
        # mp4s without closing episode N+1's still-open writers.
        with writers_lock:
            if tmp_dir is None:
                for writer in writers.values():
                    writer.release()
                writers.clear()
                return
            for key in [k for k in writers if k[0] == tmp_dir]:
                writers[key].release()
                del writers[key]

    # Dedicated thread for the long-running ffmpeg H.264 re-encode step.
    # Running it inside the main loop would block "frame" and "flush"
    # commands for the entire 1-3 second encode duration — and that means
    # the env.reset that comes next has to wait for it through
    # _flush_video_writers' ack_queue.get(). max_workers=1 keeps episodes
    # encoded in submission order without piling up parallel ffmpegs.
    convert_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vid-convert",
    )

    try:
        while True:
            item = video_queue.get()
            if item is None:
                # Shut down: drain pending converts before releasing writers
                # so we don't leave half-encoded mp4s behind.
                convert_executor.shutdown(wait=True)
                release_writers()
                return

            command = item[0]
            if command == "frame":
                _, obs_key, frame, tmp_dir, fps = item
                key = (tmp_dir, obs_key)
                with writers_lock:
                    if key not in writers:
                        video_name = f"{obs_key.replace('_image', '-images-rgb')}.mp4"
                        writers[key] = cv2.VideoWriter(
                            str(Path(tmp_dir) / video_name),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            compressed_shape,
                        )
                    writers[key].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            elif command == "flush":
                # ("flush", token) -> release all; ("flush", token, tmp_dir) -> per-episode.
                token = item[1]
                tmp_dir = item[2] if len(item) > 2 else None
                release_writers(tmp_dir)
                ack_queue.put(token)
            elif command == "convert":
                # Off-thread the heavy ffmpeg re-encode so the main loop
                # can keep processing frames/flushes from the next episode
                # without waiting on ffmpeg. The env.reset path's
                # _flush_video_writers now waits ONLY for the writers from
                # the just-ended episode to release, never for an unrelated
                # convert in flight.
                _, episode_path = item
                convert_executor.submit(
                    _convert_videos_to_h264_worker,
                    episode_path,
                    compressed_shape,
                )
    finally:
        convert_executor.shutdown(wait=True)
        release_writers()

