# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-backed mirror of get_rew_rgb.py with a streaming producer-consumer
pipeline so reward latency stays flat — never grows with runtime.

WHY THE STREAMING REWRITE: the PyTorch reward loop runs synchronously
(capture → SAM3 → publish → sleep). When SAM3 takes longer than the frame
budget, the loop falls behind: latency creeps up call-over-call as backed-up
work gets serialized. The TRT backend wants a different shape:

  • capture thread     — runs as fast as cameras deliver, pushes to a
                         single-slot frame queue. If the worker is busy,
                         the old (unconsumed) frame is REPLACED silently.
  • SAM3 worker thread — pops the latest frame pair from the queue, runs
                         get_reward_from_top_right_cam(), publishes the
                         result + capture timestamp.
  • main thread        — every FPS tick, reads whatever result is most
                         recent, prints it, fires the vis save.

Net effect: SAM3 ALWAYS inferences the freshest available frame. If a new
capture lands while SAM3 is mid-forward, the previous queued frame is
dropped (counted in the per-frame trace as `dropped=N`). End-to-end reward
latency = single SAM3 forward, regardless of how long the script runs.

Quirks vs the PyTorch loop:
  • Result published lags the capture by ~SAM3 forward time (35-300 ms).
    The "stale" age of the printed reward is shown next to the timing.
  • If the worker is slower than the producer, frames are dropped. That's
    the whole point — we never queue.

Set ZIPTIE_REWARD_NO_VIS=1 to suppress vis/{top,right,merged} writes.
Pre-launch the TRT-backed SAM3 server on :6868 (see the sam3only_onnx
launch script).
"""

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from skill_library.namespace import get_camera_image

import enpire.env.forge.cap.agent.tools._artifact_log as _al
from enpire.env.forge.cap.agent.tools._artifact_log import render_pool_stats

# Pull functions AND shared constants from the TRT variant — flips SAM3_URL to :6868.
_compute_rew_func = load_module("ziptie/reward/_compute_rew_rgb_trt.py")
get_reward_from_top_right_cam                    = _compute_rew_func["get_reward_from_top_right_cam"]
save_top_right_cam_tiled_async = _compute_rew_func["save_top_right_cam_tiled_async"]
TOP_CROP                       = _compute_rew_func["TOP_CROP"]

NO_VIS = os.environ.get("ZIPTIE_REWARD_NO_VIS", "").lower() in ("1", "true", "yes")
if NO_VIS:
    _al.set_artifact_dir(None)
    print("[reward-trt] ZIPTIE_REWARD_NO_VIS=1 — vis/* writes disabled (reward decision only)")


# ── streaming primitives ──────────────────────────────────────────────────

@dataclass
class _FramePair:
    rgb_t: np.ndarray
    rgb_r: np.ndarray
    capture_ts: float
    seq: int


class _LatestFrameSlot:
    """Single-slot frame buffer with replace-on-put semantics.

    The producer always wins: putting a new frame while the slot is full
    silently drops the old one (counted). The consumer's get() blocks until
    a frame is available or the slot is stopped. This is the "queue of 2"
    pattern the user asked for — 1 frame in-flight inside the worker + 1
    waiting here. New captures replace the waiting one without queueing.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._slot: _FramePair | None = None
        self._stopped = False
        self._dropped = 0
        self._produced = 0

    def put(self, frame: _FramePair) -> None:
        with self._cond:
            if self._slot is not None:
                self._dropped += 1
            self._slot = frame
            self._produced += 1
            self._cond.notify()

    def get(self) -> _FramePair | None:
        with self._cond:
            while self._slot is None and not self._stopped:
                self._cond.wait()
            if self._slot is None:
                return None
            f, self._slot = self._slot, None
            return f

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    @property
    def stats(self) -> dict:
        with self._cond:
            return {"produced": self._produced, "dropped": self._dropped}


# ── threads ────────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_slot = _LatestFrameSlot()
_result_lock = threading.Lock()
_latest_result: dict[str, Any] | None = None
_cam_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ziptie-cap")


def _producer_loop() -> None:
    """Capture top + right in parallel as fast as the cameras + Portal RPC
    allow, push into the latest-frame slot."""
    seq = 0
    while not _stop_event.is_set():
        t0 = time.time()
        f_t = _cam_pool.submit(get_camera_image, "top")
        f_r = _cam_pool.submit(get_camera_image, "right")
        try:
            rgb_t = np.asarray(f_t.result())
            rgb_r = np.asarray(f_r.result())
        except Exception as exc:
            print(f"[reward-trt][producer] capture failed: {exc}", flush=True)
            time.sleep(0.05)
            continue
        seq += 1
        _slot.put(_FramePair(rgb_t=rgb_t, rgb_r=rgb_r, capture_ts=t0, seq=seq))
        # No sleep: we want as-fast-as-possible. If cameras saturate the loop,
        # the slot's replace-on-put handles the backpressure.


def _worker_loop() -> None:
    """Pop the latest frame, run SAM3 reward, publish the result."""
    global _latest_result
    while not _stop_event.is_set():
        frame = _slot.get()
        if frame is None:
            return
        t_inf0 = time.time()
        try:
            top_r, right_r = get_reward_from_top_right_cam(frame.rgb_t, frame.rgb_r)
        except Exception as exc:
            print(f"[reward-trt][worker] reward failed seq={frame.seq}: {exc}", flush=True)
            continue
        t_inf_done = time.time()
        with _result_lock:
            _latest_result = {
                "frame": frame,
                "top_r": top_r,
                "right_r": right_r,
                "inf_dt_ms": (t_inf_done - t_inf0) * 1000.0,
                "publish_ts": t_inf_done,
            }


# ── main printer loop ──────────────────────────────────────────────────────

FPS = 15.0
DURATION_S = 600.0
period = 1.0 / FPS
print(
    f"[reward-trt] streaming producer-consumer pipeline — capture fan-out → "
    f"single-slot frame queue → SAM3 worker. Printing at {FPS} Hz for up to "
    f"{DURATION_S:.0f}s.",
    flush=True,
)

t_start = time.time()
from enpire.env.forge.cap.agent.tools._artifact_log import (
    _artifact_dir as _ARTIFACT_DIR,  # noqa: E402
)

producer_th = threading.Thread(target=_producer_loop, name="ziptie-producer", daemon=True)
worker_th   = threading.Thread(target=_worker_loop,   name="ziptie-worker",   daemon=True)
producer_th.start()
worker_th.start()

_prev_status = "FAIL"
_last_seen_seq = -1
try:
    while time.time() - t_start < DURATION_S:
        tick_t0 = time.time()
        with _result_lock:
            res = _latest_result
        if res is None:
            # No reward published yet — wait a bit longer than usual on first tick
            # so the worker has a chance to finish its first forward.
            time.sleep(period)
            continue
        frame = res["frame"]
        top_r = res["top_r"]; right_r = res["right_r"]
        raw_t, rwd_t, fut_top, det_t   = top_r
        raw_r, rwd_r, fut_right, det_r = right_r
        age_ms = (time.time() - frame.capture_ts) * 1000.0
        is_fresh = frame.seq != _last_seen_seq
        _last_seen_seq = frame.seq

        status = "SUCCESS" if (rwd_t == 1 and rwd_r == 1) else "FAIL"
        if status == "SUCCESS" and _prev_status != "SUCCESS":
            subprocess.Popen(["spd-say", "Yeah"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _prev_status = status

        if _ARTIFACT_DIR is not None and not NO_VIS and is_fresh:
            save_top_right_cam_tiled_async(fut_top, fut_right, rwd_t, rwd_r, _ARTIFACT_DIR, TOP_CROP)

        stats = _slot.stats
        repeat_tag = "" if is_fresh else "  [repeat]"
        print(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {status}  "
            f"top raw={raw_t:5.1f} rwd={rwd_t} | right raw={raw_r:5.1f} rwd={rwd_r}  "
            f"(seq={frame.seq} age={age_ms:.0f}ms inf={res['inf_dt_ms']:.0f}ms "
            f"produced={stats['produced']} dropped={stats['dropped']}){repeat_tag}",
            flush=True,
        )

        dt = time.time() - tick_t0
        if dt < period:
            time.sleep(period - dt)
finally:
    _stop_event.set()
    _slot.stop()
    producer_th.join(timeout=2.0)
    worker_th.join(timeout=2.0)
    _stats = render_pool_stats()
    s = _slot.stats
    print(
        f"[reward-trt] loop done after {time.time() - t_start:.0f}s; "
        f"frames produced={s['produced']} dropped={s['dropped']} "
        f"(drop_rate={s['dropped']/max(s['produced'], 1)*100:.1f}%); "
        f"render pool submitted={_stats['submitted']} pending={_stats['pending']} "
        f"dropped={_stats['dropped']} — atexit will drain remaining jobs.",
        flush=True,
    )

