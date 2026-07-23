# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic async reward worker + episode truncator for RL data collection with
DELAYED, AUTONOMOUS, reward feedback.

Why this module exists — the delayed-reward problem
===================================================

When the success signal comes from a **learned reward model** (e.g. a SAM3
detector running on a separate server, taking ~100-300 ms per frame), the
moment the model says "this was a success" is not the moment the success
actually happened. By the time the worker thread finishes the per-frame
inference and the signal reaches the runner, the env has already stepped
through N more ticks at the control-loop rate. A naive recorder would:

  * stamp the terminal flag at the WRONG step (the step the runner happened
    to be on when the worker's signal arrived, ~N steps PAST the real
    success), and
  * write N trailing post-success frames into every per-step artifact —
    numpy arrays, JSON lists, and mp4 videos.

That breaks downstream consumers: the ingestor would feed a learner a
mislabeled success terminal followed by N "bonus" frames from after the
event, polluting the value-function target and the visual context. Hand-
coded sparse rewards (those that fire synchronously inside env.step from a
known condition) don't have this problem because their decision latency
is zero relative to the env tick — but the moment you swap in any
*inference-based* scorer the latency reappears.

This module is the disciplined response. It runs the slow scorer on a
daemon thread, records ``success_step_idx`` at the actual step the scorer
fired on (not the step the runner observes the signal), then performs an
atomic post-hoc truncate of every per-step artifact in the just-saved
episode dir down to ``success_step_idx + 1``. The truncate runs entirely
off the main control thread — see the threading topology below — so the
env loop never pays for the scorer's latency.

Task-agnostic: the worker doesn't know about ziptie or SAM3. Callers wire it
up with a list of camera names to snapshot per frame and a ``reward_fn``
callable that takes the per-camera snapshots and returns ``True`` for success.
The matching disk-truncation helper :func:`truncate_episode_after_success`
also lives here because it's purely numeric — it doesn't depend on which task
produced the episode.

Design summary:

* :class:`AutoRewardWorker` runs a daemon thread with a ``Queue(maxsize=2)``,
  drop-old. The runner calls :meth:`submit_frame` once per env tick from its
  learn state; each submit snapshots the env's cached camera frames (~µs of
  ``.copy()`` work, no USB activity) and pushes them to the queue.
* The worker pops frames, runs ``reward_fn``, and on success records the step
  idx + raises a single-shot signal flag. Subsequent frames in the same
  episode are scored-skipped to avoid relabeling.
* The runner polls :meth:`pop_success_signal` once per tick; on ``"success"``
  it drives the standard terminal-event sequence (terminal label, env.reset,
  state → hover) and then calls :meth:`truncate_episode_after_success` with
  the dir the env just flushed.
* :meth:`mark_episode_done` clears per-episode state at every boundary.

The truncate is exhaustive: every length-T artifact in the episode dir (npy,
reward.npz, JSON lists, mp4 videos) gets sliced to length ``success_step+1``.
``reward.npy`` is overwritten with ``[0, 0, …, 0, 1]`` (and ``reward.npz``
to match); ``dones.npy`` with ``[False, …, False, True]`` — explicit "1 at
the step the reward model scored, 0 everywhere else", as the user's spec.
Every write uses atomic rename so a concurrent learner ingest never sees a
half-written buffer.

Threading topology
==================

All the slow work (ffmpeg re-encode, ffmpeg trim, marker wait) lives on
background threads/subprocesses. The main control loop only pays a couple
of µs of bookkeeping per auto-reward success — two file renames in
``_hide_reward_files`` plus a non-blocking queue submit::

    MAIN CONTROL THREAD (rl/ziptie_runner.py → env.step / env.reset)
    ├── env.step()                            [hot path, every tick]
    ├── ctx.env.truncate_in_memory(keep)      [µs — list slicing in memory]
    ├── env.reset(...)                        [triggers _save_episode_data]
    │     └── _save_episode_data()            [unchanged, fast: tmp writes + shutil.move]
    │           └── _video_queue.put(("convert", ep))   [non-blocking queue submit]
    └── worker.request_truncate(ep)           [main thread, but only these 2 things:]
          ├── _hide_reward_files(ep)          [2 renames, µs]
          └── _truncate_queue.put((ep, k))    [non-blocking queue submit]
          ⤷ returns immediately to runner

    VIDEO-CONVERT SUBPROCESS (mp.Process spawned in record_episode_wrapper.__init__)
    └── _video_worker_loop
          └── on "convert": convert_executor.submit(_convert_videos_to_h264_worker, ep)
                └── ffmpeg re-encode per mp4 (1–2s each)
                └── finally: marker.touch()    ← VIDEO_CONVERT_DONE_MARKER

    AUTO-REWARD TRUNCATE DAEMON THREAD (rl/auto_reward.py:_truncate_thread, daemon=True)
    └── _truncate_loop                         [blocks on _truncate_queue.get(timeout=0.5)]
          └── _truncate_episode_dir(ep, keep)
                ├── (numpy/json trim — only if keep < T)
                ├── _trim_mp4s_in_dir(ep, keep)
                │     ├── _wait_for_video_convert_marker(ep, wait_s=60)   [polls, sleeps]
                │     ├── for mp4: _trim_mp4_to_tmp(mp4, keep)            [ffmpeg, seconds]
                │     └── for mp4: tmp.replace(mp4)                       [atomic rename]
                └── write dones.npy + reward.npz + reward.npy

Three threads / processes touch the episode dir for an auto-reward success:
the main control thread (does the synchronous reward-hide + queue submit
in µs), a daemon thread (does the slow numpy/mp4 trim and the canonical
reward rewrite), and a sibling subprocess (does the ffmpeg H.264 re-encode
and publishes the ``.videos_converted`` marker the daemon thread blocks on).
The daemon thread never blocks the runner — its 60 s deadline absorbs any
bursty convert queue.

Disk-state invariants (held no matter which step fails)
=======================================================

1. ``reward.npy`` is written strictly LAST inside ``_truncate_episode_dir`` —
   it's the gate file. While the truncate is in flight, the original is
   renamed to ``reward.npy.trimming`` (see ``_hide_reward_files``) so the
   ingestor's ``_episode_complete()`` returns False and the episode is
   skipped.
2. A trimmed reward.npy (``rew[-1] == 1``) NEVER coexists on disk with an
   untrimmed mp4. The mp4 trim is a two-phase commit (write all
   ``.trim.mp4`` siblings first, then atomic-replace each); if ANY mp4 fails
   to trim, the whole truncate raises and ``_resolve_hidden_reward_files``
   restores the pre-truncate reward — episode is reported as a fail to the
   ingestor instead of a corrupted success.
3. The video-convert subprocess publishes ``.videos_converted`` only after
   every mp4 in the dir has been re-encoded (or had its conversion fail
   with the original renamed back from ``_temp``) — so the trim's wait is
   a real signal, not a polling heuristic.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

# One per-camera snapshot — RGB at native resolution, depth (if the camera
# exposes it, else None), and intrinsics (dict from the camera driver).
CameraSnapshot = Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, float]]]


@dataclass
class _PendingFrame:
    """One frame in flight to the worker thread.

    ``snapshots`` maps camera_name → (rgb, depth, intrinsics). ``step_idx`` and
    ``ts_ns`` are echoed back unchanged when the worker records a success, so
    the truncator knows exactly which row in the episode arrays got scored.
    """

    snapshots: Dict[str, CameraSnapshot]
    step_idx: int
    ts_ns: int


class AutoRewardWorker:
    """Background reward thread + episode-dir truncator.

    Generic over camera set and reward computation. Caller provides:

      * ``env``: a yam_real_env (or anything with a ``get_raw_camera_data(name)``
        method returning ``(rgb, depth, intrinsics)``).
      * ``camera_names``: which cameras to snapshot per submit. Names not on
        the env get a one-time warning and the submit is dropped (so the
        runner can launch in degraded mode while a camera is brought up).
      * ``reward_fn(snapshots) -> bool``: pure function from the snapshots
        dict to a success boolean. Runs on the worker thread, so it can be
        as slow as one inference round-trip without affecting the env tick.
      * ``thread_name``: shown in ``ps``/``top`` for the daemon thread.

    The disk-truncation pieces don't depend on any of the above — they only
    care about the recorded success_step_idx and the episode dir path.

    Lifecycle inside one episode:

      1. Runner calls :meth:`submit_frame(step_idx, ts_ns)` once per env tick
         during learn. Cheap snapshot + queue push; never blocks.
      2. Worker thread pops, runs ``reward_fn``, and on success stores
         ``success_step_idx`` and sets the signal flag (one-shot).
      3. Runner polls :meth:`pop_success_signal` once per tick; on ``"success"``
         it runs the terminal sequence, calls
         :meth:`truncate_episode_after_success(env.last_episode_dir)`, and
         then :meth:`mark_episode_done` to reset for the next episode.
    """

    def __init__(
        self,
        env: Any,
        reward_fn: Callable[[Dict[str, CameraSnapshot]], bool],
        camera_names: Sequence[str],
        thread_name: str = "auto-reward",
        queue_size: int = 2,
    ):
        if not camera_names:
            raise ValueError("AutoRewardWorker needs at least one camera name")
        self._env = env
        self._reward_fn = reward_fn
        self._camera_names: Tuple[str, ...] = tuple(camera_names)
        self._queue: "queue.Queue[_PendingFrame]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._lock = threading.Lock()
        # Per-episode state (only valid while episode is in progress).
        self._success_step_idx: Optional[int] = None
        self._success_ts_ns: Optional[int] = None
        self._signal_pending = False
        self._warned_missing_cam = False  # log-once for missing-camera warning
        # Dedicated background queue for episode-truncate jobs. Each item is
        # (episode_dir, success_step_idx) captured at submission time, so the
        # truncate uses the right step idx even after _success_step_idx has
        # been reset by mark_episode_done(). Unbounded queue: truncates don't
        # pile up in practice (one per ~episode duration, each finishes in
        # well under that), but if they ever do we'd rather backlog than
        # block the runner.
        self._truncate_queue: "queue.Queue[Tuple[Path, int]]" = queue.Queue()
        self.running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=str(thread_name),
        )
        self._thread.start()
        self._truncate_thread = threading.Thread(
            target=self._truncate_loop,
            daemon=True,
            name=f"{thread_name}-trunc",
        )
        self._truncate_thread.start()

    # ---------- runner-facing API -----------------------------------------

    def submit_frame(self, step_idx: int, ts_ns: int) -> None:
        """Snapshot the env's cached camera frames and push to the worker.

        Cheap (just ``.copy()`` calls on already-cached numpy arrays in the
        env's NonBlockingCamera caches — no USB touched). Drop-old: a full
        queue evicts the stalest pending frame so the worker always scores
        the freshest input. Returns silently when any configured camera is
        missing from the env (with a one-shot warning).
        """
        snapshots: Dict[str, CameraSnapshot] = {}
        missing = []
        for name in self._camera_names:
            rgb, depth, intr = self._env.get_raw_camera_data(name)
            if rgb is None:
                missing.append(name)
            snapshots[name] = (rgb, depth, intr)
        if missing:
            if not self._warned_missing_cam:
                print(
                    f"[auto_reward] WARN cameras {missing} not exposed by env "
                    f"(needed by reward_fn). Worker is dormant until those "
                    f"cameras come up. Hint: add them via "
                    f"env.unwrapped.add_extra_camera(name).",
                    flush=True,
                )
                self._warned_missing_cam = True
            return
        frame = _PendingFrame(snapshots=snapshots, step_idx=step_idx, ts_ns=ts_ns)
        # Drop-old: if the queue is full, evict the stalest pending frame.
        # We never block the runner.
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass  # racing with worker; just drop this frame

    def pop_success_signal(self) -> Optional[str]:
        """Return ``"success"`` exactly once when the worker has flagged the
        current episode; ``None`` otherwise. Designed to be polled by the
        runner once per learn-state tick.
        """
        with self._lock:
            if self._signal_pending:
                self._signal_pending = False
                return "success"
            return None

    def clear_queue(self) -> None:
        """Drain all pending frames and the latched success signal, so the next learn episode's
        reward starts from a clean slate. Call when LEAVING learn (terminal / prepare / finish)
        and again when ENTERING learn — together with env.refresh_camera_cache() — so the worker
        only ever scores fresh, learn-phase frames (never ones cached during a reset/place handover)."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._signal_pending = False

    def request_truncate(self, episode_dir: Optional[Path]) -> None:
        """Hide reward files synchronously, then enqueue the slow trim work.

        Splits the work into two pieces deliberately:

          * **Main-thread (this call, ~µs)**: rename ``reward.npy`` and
            ``reward.npz`` to ``.trimming`` siblings. From the learner's
            ``_episode_complete()`` perspective the episode is now
            incomplete — any concurrent scan silently skips it. Doing this
            on the caller's thread closes the race between ``env.reset``
            returning (which publishes a complete episode dir with reward
            stamped at the wrong step — the SAM3-late frame) and the bg
            thread getting scheduled (the previous version had a ~ms gap
            here where a fast-scanning ingestor could read the wrong
            terminal).
          * **Background (queued)**: numpy rewrites, mp4 trim (with
            polling for the env's async video conversion to land), JSON +
            metadata updates, then the atomic publish of the new
            reward.npy / reward.npz. Episode flips back to "complete"
            only when every artifact is consistent.

        Returns in microseconds; the actual disk surgery runs on the
        ``-trunc`` daemon thread (see :meth:`_truncate_loop`). Decoupled
        from the runner's main thread so episode-length scaling never
        stalls the runner's between-episode transition or eats into the
        next learn tick's budget.

        ``success_step_idx`` is captured into the queue item NOW, before
        :meth:`mark_episode_done` clears it for the next episode — so each
        truncate job carries its own anchor and can't be confused by later
        successes.

        No-op when ``success_step_idx`` is ``None`` (e.g., episode ended by
        Fello button / timeout) or when ``episode_dir`` doesn't resolve.
        """
        if episode_dir is None or self._success_step_idx is None:
            return
        ep = Path(episode_dir)
        if not ep.is_dir():
            print(f"[auto_reward] WARN episode dir not found, skipping truncate: {ep}", flush=True)
            return
        # SYNCHRONOUS on the caller's thread: two atomic os.rename calls,
        # microseconds each. Closes the race window vs the ingestor.
        _hide_reward_files(ep)
        self._truncate_queue.put((ep, int(self._success_step_idx)))

    def mark_episode_done(self) -> None:
        """Reset per-episode state and drain the queue. Called by the runner
        on every episode boundary (whether or not auto-reward fired)."""
        with self._lock:
            self._success_step_idx = None
            self._success_ts_ns = None
            self._signal_pending = False
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def finalize_success_episode(self, ctx: Any, on_rehover: Callable[[], None]) -> None:
        """Runner-side finalize for a latency-delayed auto-reward success — the general counterpart
        to :meth:`pop_success_signal`. Call once per learn tick when the poll returns ``"success"``:
        a PAST frame was scored success on the worker thread, so stamp the success bookkeeping, trim
        the in-memory env buffers back to the scored step (so on-disk reward.npy lands at the right
        length), run ``on_rehover`` (the task's reset — it flushes+rotates the episode via env.reset
        AND returns the arms to hover), then enqueue the background dir truncate and reset per-episode
        state. Task-agnostic: ``ctx`` only needs the standard demo counters / timing_log / speech /
        env; ``on_rehover`` carries all task specifics."""
        from enpire.policy.rl.speech_announcer import TERMINAL_EVENT_PHRASES

        ctx.terminal_event = ctx.last_terminal_event = "success"
        ctx.timing_log.log("terminal", signal="auto_reward_success")
        ctx.speech_announcer.speak(TERMINAL_EVENT_PHRASES["success"])
        ctx.demo_total_count += 1
        ctx.demo_success_count += 1
        ctx.demo_rolling_window.append(True)
        if self._success_step_idx is not None:
            try:
                ctx.env.truncate_in_memory(self._success_step_idx + 1)
            except AttributeError:
                pass  # wrapper missing the method (e.g. sim env) — fall through
        on_rehover()
        self.request_truncate(getattr(ctx.env, "last_episode_dir", None))
        self.mark_episode_done()

    def close(self) -> None:
        """Stop the worker threads and drain any in-flight truncate jobs so
        we don't exit with half-trimmed episodes on disk. Daemon threads
        would die with the process anyway; this is the polite shutdown."""
        # Wait for any queued truncates to finish so the .trimming markers
        # get cleaned up and reward.npy/npz get published before we exit.
        try:
            self._truncate_queue.join()
        except Exception:
            pass
        self.running = False

    # ---------- internal worker loops -------------------------------------

    def _truncate_loop(self) -> None:
        """Background daemon: drains ``self._truncate_queue`` one job at a
        time, doing the hide → trim → publish dance for each finalized
        episode. Runs in its own thread so the runner's between-episode
        transition (do_hover + the very next env.step) never blocks on disk
        I/O — episode length can grow without ever showing up in the main
        loop's wall time.
        """
        while self.running:
            try:
                ep, keep_step = self._truncate_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if not ep.is_dir():
                    print(
                        f"[auto_reward] WARN truncate target missing: {ep}",
                        flush=True,
                    )
                    continue
                keep = keep_step + 1
                # request_truncate already hid the reward files on the
                # main thread (closes the env.reset → ingestor race);
                # here we just run the slow trim and republish.
                try:
                    _truncate_episode_dir(ep, keep)
                finally:
                    _resolve_hidden_reward_files(ep)
                print(
                    f"[auto_reward] truncated {ep.name} to {keep} steps (success at idx={keep_step})",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[auto_reward] WARN truncate raised on {ep}: {exc}",
                    flush=True,
                )
            finally:
                # task_done() must always fire so close()'s queue.join() can
                # return — even on exceptions.
                self._truncate_queue.task_done()

    def _worker_loop(self) -> None:
        while self.running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Skip if this episode already has a recorded success — every
            # subsequent frame would be a duplicate signal and risk overwriting
            # success_step_idx with a later (post-success) tick.
            with self._lock:
                if self._success_step_idx is not None:
                    continue
            try:
                is_success = bool(self._reward_fn(frame.snapshots))
            except Exception as exc:
                print(f"[auto_reward] WARN reward_fn raised: {exc}", flush=True)
                continue
            if is_success:
                with self._lock:
                    if self._success_step_idx is None:
                        self._success_step_idx = frame.step_idx
                        self._success_ts_ns = frame.ts_ns
                        self._signal_pending = True
                print(
                    f"\033[1;32m[auto_reward] SUCCESS at step={frame.step_idx} (ts_ns={frame.ts_ns})\033[0m",
                    flush=True,
                )


def build_auto_reward_worker(
    env: Any,
    reward_fn: Callable[[Dict[str, CameraSnapshot]], bool],
    reward_camera_names: Sequence[str],
    enabled_camera_names: Sequence[str],
    *,
    camera_resolutions: Optional[Dict[str, Optional[Tuple[int, int]]]] = None,
    thread_name: str = "auto-reward",
) -> AutoRewardWorker:
    """Open every reward-only camera (one in ``reward_camera_names`` but not in
    ``enabled_camera_names``) as an obs-excluded extra at ``camera_resolutions[name]`` (None → the
    camera default), then spawn an :class:`AutoRewardWorker` over ``reward_fn``. Reward-only cameras
    are added as extras so the policy obs / episode format stay untouched. Task-agnostic: the
    per-camera resolution map is caller-supplied (e.g. a reward model calibrated at a fixed frame
    size — pass ``{cam: (640, 480)}`` to force 480p)."""
    camera_resolutions = camera_resolutions or {}
    extras_added = []
    for cam in reward_camera_names:
        if cam not in enabled_camera_names:
            env.add_extra_camera(cam, resolution=camera_resolutions.get(cam))
            extras_added.append(cam)
    worker = AutoRewardWorker(env=env, reward_fn=reward_fn, camera_names=reward_camera_names, thread_name=thread_name)
    print(
        f"\033[1;36m[auto_reward] worker '{thread_name}' started "
        f"(reward_cams={list(reward_camera_names)}, obs-excluded extras={extras_added})\033[0m",
        flush=True,
    )
    return worker


# ---------- disk-truncation helpers (module-level for testability) -----------


# Names used as the "ingestor must skip me" marker. While these are renamed
# away from their canonical filenames, _episode_complete() returns False
# (see disk_buffer_ingestor._episode_complete) and the ingestor skips the
# episode without trying to read a half-trimmed state.
_HIDE_NAMES = ("reward.npy", "reward.npz")
_HIDE_SUFFIX = ".trimming"


def _hide_reward_files(ep: Path) -> None:
    """Rename ``reward.npy`` and ``reward.npz`` to ``*.trimming`` siblings so
    the learner's ``_episode_complete()`` check fails and the episode is
    skipped while we rewrite its arrays. Both renames are atomic, ~µs."""
    for name in _HIDE_NAMES:
        src = ep / name
        if src.exists():
            src.rename(ep / (name + _HIDE_SUFFIX))


def _resolve_hidden_reward_files(ep: Path) -> None:
    """Finalize the hide step. Called from a ``finally`` block.

    * Success path: the fresh ``reward.npy`` / ``reward.npz`` were written
      by :func:`_truncate_episode_dir` near the end of its work — the
      ``.trimming`` siblings are now obsolete and we delete them.
    * Failure path: :func:`_truncate_episode_dir` raised before publishing
      the new reward files — restore the ``.trimming`` originals so the
      episode reverts to its untrimmed (but still valid) state. Better to
      leave the episode untrimmed than to leave it in a permanently
      incomplete state the ingestor will never accept.
    """
    new_exists = (ep / "reward.npy").exists()
    for name in _HIDE_NAMES:
        staged = ep / (name + _HIDE_SUFFIX)
        if not staged.exists():
            continue
        if new_exists:
            try:
                staged.unlink()
            except OSError:
                pass
        else:
            staged.rename(ep / name)


def _atomic_replace(path: Path, write_tmp) -> None:
    """``write_tmp(tmp_path)`` produces the file at a sibling temp path; we
    then ``os.replace`` it onto ``path``. POSIX guarantees rename atomicity
    within the same directory, so partial writes are invisible to readers."""
    tmp = path.with_name(path.name + ".tmp")
    write_tmp(tmp)
    tmp.replace(path)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    def _w(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.save(f, arr, allow_pickle=False)

    _atomic_replace(path, _w)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    def _w(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)

    _atomic_replace(path, _w)


def _save_json_atomic(path: Path, data: Any) -> None:
    def _w(tmp: Path) -> None:
        with open(tmp, "w") as f:
            json.dump(data, f)

    _atomic_replace(path, _w)


def _wait_for_video_convert_marker(ep: Path, wait_s: float = 60.0) -> None:
    """Block until the env's video converter publishes its done marker.

    record_episode_wrapper offloads H.264 re-encode to a subprocess (see
    ``tools/data_collection/async_video_compression._video_worker_loop``)
    that writes ``VIDEO_CONVERT_DONE_MARKER`` (currently
    ``.videos_converted``) inside each finished episode dir from a finally
    block — covers the per-file ffmpeg success, the per-file failure (where
    the original was renamed back from ``_temp``), and the no-mp4 early-
    return path. Per-dir marker is race-free because the convert executor is
    single-threaded.

    Raises RuntimeError on timeout — the caller (``_truncate_episode_dir``)
    propagates it so ``_resolve_hidden_reward_files`` rolls back to the
    un-truncated reward.npy and we never end up with an untrimmed mp4 +
    rew=1 on disk together.
    """
    try:
        from enpire.env.forge.tools.data_collection.async_video_compression import (
            VIDEO_CONVERT_DONE_MARKER,
        )
    except ImportError:
        VIDEO_CONVERT_DONE_MARKER = ".videos_converted"
    marker = ep / VIDEO_CONVERT_DONE_MARKER
    deadline = time.monotonic() + wait_s
    while not marker.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"video-convert marker {marker.name} did not appear within {wait_s:.0f}s in {ep}")
        time.sleep(0.05)


def _trim_mp4_to_tmp(path: Path, keep_frames: int) -> Path:
    """ffmpeg stream-copy (no re-encode) the first ``keep_frames`` frames of
    ``path`` into a sibling ``<name>.trim.mp4``. Returns the tmp path on
    success; raises RuntimeError on any failure.

    Caller (``_trim_mp4s_in_dir``) is responsible for atomically committing
    the tmp file onto the original path AFTER every mp4 in the dir has been
    successfully written to its own tmp. Two-phase commit guarantees that
    if any single mp4 fails to trim, NO mp4 in the dir is replaced — so the
    dir never holds a half-trimmed-half-untrimmed state when paired with a
    rew=1 reward.npy.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg binary not found on PATH — cannot trim mp4s")
    if not path.exists():
        raise RuntimeError(f"mp4 missing after video-convert marker: {path}")
    tmp = path.with_name(path.name + ".trim.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-frames:v",
        str(int(keep_frames)),
        "-c:v",
        "copy",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"ffmpeg trim failed for {path.name}: {exc.stderr.decode(errors='replace')[:200]}") from exc
    return tmp


def _trim_mp4s_in_dir(ep: Path, keep_frames: int, *, wait_s: float = 60.0) -> None:
    """Atomically trim every mp4 in ``ep`` to ``keep_frames`` frames.

    Order:
      1. Wait for the env's video-convert done marker (so we're reading the
         FINAL ffmpeg-re-encoded mp4, not a mid-convert partial).
      2. Write every mp4's trimmed copy to a sibling ``.trim.mp4``. If ANY
         single mp4 fails, delete every ``.trim.mp4`` produced so far and
         raise — no original is touched.
      3. Atomically ``replace`` each original with its ``.trim.mp4``.

    The two-phase commit is what gives us the "no untrimmed mp4 in dir when
    rew=1" invariant: ``_truncate_episode_dir`` calls this BEFORE writing
    reward.npy with terminal=1, and ``_resolve_hidden_reward_files`` rolls
    back to the un-truncated reward on any exception — so either every mp4
    in the dir is trimmed AND reward.npy=1 (success), or every mp4 is
    untouched AND reward.npy is the original un-truncated array (rollback).
    Never a mix.
    """
    _wait_for_video_convert_marker(ep, wait_s=wait_s)
    mp4s = sorted(ep.glob("*.mp4"))
    if not mp4s:
        return
    # Phase 1: produce all .trim.mp4 siblings, or unwind on first failure.
    tmps: list[Path] = []
    try:
        for mp4 in mp4s:
            tmps.append(_trim_mp4_to_tmp(mp4, keep_frames))
    except Exception:
        for t in tmps:
            try:
                if t.exists():
                    t.unlink()
            except OSError:
                pass
        raise
    # Phase 2: atomic replace each. On POSIX, rename within the same dir is
    # atomic at the inode level, so each individual mp4 flips from untrimmed
    # to trimmed instantaneously. Brief intra-step window where one mp4 is
    # trimmed and the next isn't yet — but reward.npy is still hidden as
    # *.trimming, so no consumer ever sees rew=1 paired with a mixed state.
    for mp4, tmp in zip(mp4s, tmps):
        tmp.replace(mp4)


def _truncate_episode_dir(ep: Path, keep: int) -> None:
    """Trim every length-T artifact in ``ep`` to length ``keep``.

    Order is deliberate: reward.npy + reward.npz are written LAST, so that
    the moment the ingestor sees a complete episode again, every other
    artifact has already been trimmed. Combined with
    :func:`_hide_reward_files` / :func:`_resolve_hidden_reward_files` around
    this call, the learner never observes a partially-trimmed episode.

    The reward array is forced to ``[0, …, 0, 1]`` and dones to
    ``[False, …, False, True]`` — explicit per the user's spec, so even if
    the original arrays had non-zero rewards mid-episode they get wiped in
    favour of the single auto-reward terminal.
    """
    # Anchor T from timestamp.npy because reward.npy was renamed out of the
    # way by _hide_reward_files() right before this call. timestamp.npy is
    # always written by the env and has the same row count as every other
    # per-step artifact AT THE TIME _save_episode_data WROTE IT — note this
    # is NOT necessarily the original (pre-truncate) episode length, because
    # the runner calls env.truncate_in_memory(keep) BEFORE env.reset on the
    # auto-reward success path (see ziptie_runner.py:519). In that case
    # T == keep on disk for every per-step artifact, EXCEPT for the mp4s,
    # whose frames are written one-per-env.step by cv2 and stay at the
    # un-truncated frame count regardless of the in-memory slice.
    #
    # So the contract here is:
    #   * keep > T  → refuse (can't extend a shorter episode)
    #   * keep < T  → trim the length-T per-step files to length keep AND
    #                 trim every mp4 to length keep AND rewrite reward.npy.
    #   * keep == T → per-step files already at length keep (env pre-trimmed
    #                 them); STILL trim the mp4s (they're at the un-trimmed
    #                 frame count) and STILL rewrite reward.npy to the
    #                 canonical [0,...,0,1] / dones.npy [False,...,True].
    rew_path = ep / "reward.npy"
    ts_path = ep / "timestamp.npy"
    anchor = ts_path if ts_path.exists() else rew_path
    if not anchor.exists():
        print(f"[auto_reward] WARN no timestamp.npy / reward.npy in {ep}, skipping", flush=True)
        return
    T = int(np.load(anchor).shape[0])
    if keep > T:
        print(f"[auto_reward] keep={keep} > T={T}, refusing to extend episode", flush=True)
        return
    if keep < 1:
        print(f"[auto_reward] keep={keep} < 1, refusing to write empty episode", flush=True)
        return

    # 1) Every length-T numpy array EXCEPT reward.npy / dones.npy gets sliced.
    #    No-op when T == keep (the env pre-truncated everything in memory).
    if keep < T:
        for npy in ep.glob("*.npy"):
            if npy.name in ("reward.npy", "dones.npy"):
                continue
            try:
                arr = np.load(npy, allow_pickle=False)
            except ValueError:
                # string-dtype arrays (e.g. action-source.npy) need allow_pickle
                arr = np.load(npy, allow_pickle=True)
            if arr.ndim >= 1 and arr.shape[0] == T:
                _save_npy_atomic(npy, arr[:keep])

        # 2) Per-step JSON lists (component_timestamps.json, action-source.json).
        for jname in ("component_timestamps.json", "action-source.json"):
            jp = ep / jname
            if not jp.exists():
                continue
            try:
                data = json.loads(jp.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(data, list) and len(data) == T:
                _save_json_atomic(jp, data[:keep])

    # 3) metadata.json — update episode duration if the fields are present.
    md_path = ep / "metadata.json"
    if md_path.exists():
        try:
            md = json.loads(md_path.read_text())
            fps = float(md.get("env_loop_frequency", 30.0))
            if fps > 0:
                md["duration"] = float(keep) / fps
            _save_json_atomic(md_path, md)
        except (json.JSONDecodeError, ValueError):
            pass

    # 4) Videos — STRICT two-phase commit. Every mp4 in the dir is trimmed
    #    to its .trim.mp4 sibling first; only when all succeed do we atomic-
    #    replace the originals. If ANY ffmpeg call fails (or the convert
    #    marker times out), this raises and the outer _truncate_loop's
    #    finally block calls _resolve_hidden_reward_files which restores the
    #    original un-truncated reward.npy. End result: either every mp4 is
    #    trimmed AND reward.npy gets rewritten with terminal=1, or every
    #    mp4 is untouched AND reward.npy reverts to its original un-trimmed
    #    state. The ingestor (which gates on reward.npy length matching mp4
    #    frame count) NEVER sees a mismatched success episode.
    _trim_mp4s_in_dir(ep, keep)

    # 5) FINALLY: reward + dones with hard-overwritten contents, plus
    #    reward.npz. This is the publish step — the moment reward.npy lands,
    #    the ingestor sees a complete (and already-trimmed) episode. The
    #    order below matters: dones.npy and reward.npz are sibling artifacts
    #    that the ingestor may consult after reward.npy, so they must exist
    #    BEFORE reward.npy. reward.npy is the LAST write because it's the
    #    gate file (its existence + length flips the episode from "skip" to
    #    "ingest" in disk_buffer_ingestor._episode_complete).
    rew = np.zeros(keep, dtype=np.float32)
    rew[-1] = 1.0
    dones = np.zeros(keep, dtype=bool)
    dones[-1] = True
    _save_npy_atomic(ep / "dones.npy", dones)
    _save_npz_atomic(ep / "reward.npz", rewards=rew, dones=dones)
    _save_npy_atomic(rew_path, rew)  # reward.npy LAST — it's the gate file.

