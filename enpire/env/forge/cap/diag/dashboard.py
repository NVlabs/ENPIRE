#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn[standard]", "msgpack", "numpy", "portal"]
# ///
"""Standalone RL pipeline diagnostics dashboard.

Run:  uv run cap/diag/dashboard.py [--udp-port 9999] [--http-port 8888]

No dependency on CAP or bc_policy.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import socket
import statistics
import threading
import time
from typing import Any

import msgpack
import numpy as np
import portal
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

LEFT_FOLLOWER_PORT  = 11333
RIGHT_FOLLOWER_PORT = 11334
LEFT_LEADER_PORT    = 11335
RIGHT_LEADER_PORT   = 11336

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

RawEvent = dict[str, Any]

# Ordered event names for the 30 Hz cap_server loop
CAP_EVENTS = [
    "step_start", "fello_read", "obs_built", "rpc_call_start",
    "rpc_call_end", "reward_start", "reward_end", "action_applied",
    "record_done", "step_end",
]

# Segment definitions: (label, start_event, end_event)
SEGMENTS = [
    ("fello_read",    "step_start",     "fello_read"),
    ("build_obs",     "fello_read",     "obs_built"),
    ("rpc_call",      "rpc_call_start", "rpc_call_end"),
    ("reward",        "reward_start",   "reward_end"),
    ("apply_action",  "reward_end",     "action_applied"),
    ("record",        "action_applied", "record_done"),
    ("idle",          "record_done",    "step_end"),
    ("total",         "step_start",     "step_end"),
]

RPC_SEGMENTS = [
    ("base_action",    "step_recv",          "base_action_done"),
    ("sac_sample",     "base_action_done",   "sac_sample_done"),
    ("buffer_insert",  "sac_sample_done",    "buffer_insert_done"),
]


class StepData:
    __slots__ = ("ep", "step", "events", "rpc_events", "reward_events", "meta")

    def __init__(self, ep: int, step: int):
        self.ep = ep
        self.step = step
        self.events: dict[str, float] = {}        # cap_server events
        self.rpc_events: dict[str, float] = {}     # rl_policy_server events
        self.reward_events: dict[str, float] = {}  # reward_server events
        self.meta: dict[str, Any] = {}              # merged meta from all events


# ---------------------------------------------------------------------------
# Ring buffer + step index
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, raw_maxlen: int = 10_000, step_maxlen: int = 2000):
        self.lock = threading.Lock()
        self.raw: collections.deque[RawEvent] = collections.deque(maxlen=raw_maxlen)
        self.steps: dict[tuple[int, int], StepData] = {}
        self._step_order: collections.deque[tuple[int, int]] = collections.deque(maxlen=step_maxlen)
        # Aux ring buffers for sampled streams
        self.control_ticks: collections.deque[RawEvent] = collections.deque(maxlen=500)
        self.fello_ticks: collections.deque[RawEvent] = collections.deque(maxlen=500)
        self.camera_frames: collections.deque[RawEvent] = collections.deque(maxlen=500)
        self.motor_temps: dict[str, dict[str, Any]] = {}
        self.motor_temps_updated_at: float = 0.0

    def ingest(self, pkt: RawEvent) -> None:
        with self.lock:
            self.raw.append(pkt)
            src = pkt.get("src", "")
            event = pkt.get("event", "")
            ep = pkt.get("ep", 0)
            step = pkt.get("step", 0)
            t = pkt.get("t", 0.0)

            # Sampled streams
            if src == "control_loop":
                self.control_ticks.append(pkt)
                return
            if src == "fello_loop":
                self.fello_ticks.append(pkt)
                return
            if src == "camera":
                self.camera_frames.append(pkt)
                return

            key = (ep, step)
            if key not in self.steps:
                self.steps[key] = StepData(ep, step)
                self._step_order.append(key)
                # Evict old
                while len(self.steps) > self._step_order.maxlen:
                    old = self._step_order.popleft()
                    self.steps.pop(old, None)

            sd = self.steps[key]
            meta = pkt.get("meta", {})
            if src == "cap_server":
                sd.events[event] = t
                sd.meta.update(meta)
            elif src == "rl_policy_server":
                sd.rpc_events[event] = t
            elif src == "reward_server":
                sd.reward_events[event] = t

    # --- Query helpers (called with lock held externally or from snapshot) ---

    def recent_steps(self, n: int = 20) -> list[StepData]:
        with self.lock:
            keys = list(self._step_order)[-n:]
            return [self.steps[k] for k in keys if k in self.steps]

    def stats_window(self, n: int = 300) -> list[StepData]:
        with self.lock:
            keys = list(self._step_order)[-n:]
            return [self.steps[k] for k in keys if k in self.steps]

    def update_motor_temps(self, rows: dict[str, dict[str, Any]], updated_at: float) -> None:
        with self.lock:
            self.motor_temps = rows
            self.motor_temps_updated_at = updated_at


STORE = Store()

# ---------------------------------------------------------------------------
# Segment computation
# ---------------------------------------------------------------------------

def _seg_ms(events: dict[str, float], start: str, end: str) -> float | None:
    t0 = events.get(start)
    t1 = events.get(end)
    if t0 is not None and t1 is not None:
        return (t1 - t0) * 1000.0
    return None


def step_to_dict(sd: StepData) -> dict:
    segs: dict[str, float | None] = {}
    for label, s, e in SEGMENTS:
        segs[label] = _seg_ms(sd.events, s, e)

    rpc_bd: dict[str, float | None] = {}
    for label, s, e in RPC_SEGMENTS:
        rpc_bd[label] = _seg_ms(sd.rpc_events, s, e)

    # Network RTT
    cap_rpc = _seg_ms(sd.events, "rpc_call_start", "rpc_call_end")
    server_rpc = _seg_ms(sd.rpc_events, "step_recv", "step_send")
    if cap_rpc is not None and server_rpc is not None:
        rpc_bd["network_rtt"] = max(0.0, cap_rpc - server_rpc)
    else:
        rpc_bd["network_rtt"] = None

    # Extract action_source and reward from stored meta
    action_source = sd.meta.get("action_source", "unknown")
    if action_source == "unknown":
        if sd.rpc_events.get("step_send"):
            action_source = "rl"
        elif sd.events.get("action_applied"):
            action_source = "human"
    reward_val = sd.meta.get("reward")
    cumR = sd.meta.get("cumR")
    max_steps = sd.meta.get("max_steps")

    return {
        "ep": sd.ep,
        "step": sd.step,
        "max_steps": max_steps,
        "segments": {k: round(v, 2) if v is not None else None for k, v in segs.items()},
        "rpc_breakdown": {k: round(v, 2) if v is not None else None for k, v in rpc_bd.items()},
        "total_ms": round(segs["total"], 2) if segs["total"] is not None else None,
        "action_source": action_source,
        "reward": round(reward_val, 3) if reward_val is not None else None,
        "cumulative_reward": round(cumR, 2) if cumR is not None else None,
    }


def compute_stats(steps: list[StepData]) -> dict:
    all_labels = [l for l, _, _ in SEGMENTS] + [l for l, _, _ in RPC_SEGMENTS] + ["network_rtt"]
    result: dict[str, dict] = {}
    for label in all_labels:
        vals: list[float] = []
        for sd in steps:
            if label in [l for l, _, _ in SEGMENTS]:
                matching = [(s, e) for l2, s, e in SEGMENTS if l2 == label]
                if matching:
                    v = _seg_ms(sd.events, matching[0][0], matching[0][1])
                    if v is not None:
                        vals.append(v)
            elif label in [l for l, _, _ in RPC_SEGMENTS]:
                matching = [(s, e) for l2, s, e in RPC_SEGMENTS if l2 == label]
                if matching:
                    v = _seg_ms(sd.rpc_events, matching[0][0], matching[0][1])
                    if v is not None:
                        vals.append(v)
            elif label == "network_rtt":
                cap_rpc = _seg_ms(sd.events, "rpc_call_start", "rpc_call_end")
                srv_rpc = _seg_ms(sd.rpc_events, "step_recv", "step_send")
                if cap_rpc is not None and srv_rpc is not None:
                    vals.append(max(0.0, cap_rpc - srv_rpc))

        if vals:
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            result[label] = {
                "mean": round(statistics.mean(vals), 2),
                "p50": round(vals_sorted[n // 2], 2),
                "p95": round(vals_sorted[min(int(n * 0.95), n - 1)], 2),
                "max": round(vals_sorted[-1], 2),
            }
        else:
            result[label] = {"mean": None, "p50": None, "p95": None, "max": None}
    return result


def compute_health(store: Store) -> dict:
    with store.lock:
        # Latest step
        if store._step_order:
            last_key = store._step_order[-1]
            ep, step = last_key
        else:
            ep, step = 0, 0

        # Loop Hz from last 30 steps
        keys = list(store._step_order)[-30:]
        step_times = []
        for k in keys:
            sd = store.steps.get(k)
            if sd and "step_start" in sd.events:
                step_times.append(sd.events["step_start"])
        if len(step_times) >= 2:
            dts = [step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)]
            dts = [d for d in dts if d > 0]
            loop_hz = round(1.0 / statistics.mean(dts), 1) if dts else 0.0
        else:
            loop_hz = 0.0

        # Control loop Hz + jitter
        # Emitted every 20th tick — use dt_ms from meta (actual tick period)
        ctrl_dt_vals = [p.get("meta", {}).get("dt_ms") for p in store.control_ticks]
        ctrl_dt_vals = [v for v in ctrl_dt_vals if v is not None and 0 < v < 50]
        if ctrl_dt_vals:
            mean_dt_s = statistics.mean(ctrl_dt_vals) / 1000.0
            ctrl_hz = round(1.0 / mean_dt_s, 1) if mean_dt_s > 0 else 0.0
            ctrl_jitter = round(statistics.stdev(ctrl_dt_vals), 2) if len(ctrl_dt_vals) > 1 else 0.0
        else:
            ctrl_hz, ctrl_jitter = 0.0, 0.0

        # Fello Hz
        fello_ts = [p["t"] for p in store.fello_ticks]
        if len(fello_ts) >= 2:
            recent = fello_ts[-60:]
            f_dts = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            f_dts = [d for d in f_dts if 0 < d < 0.2]
            fello_hz = round(1.0 / statistics.mean(f_dts), 1) if f_dts else 0.0
        else:
            fello_hz = 0.0

        # Camera FPS per camera
        # Emitted every 10th frame — dt_ms in meta is time since last emit (covers 10 frames)
        cam_by_name: dict[str, list[float]] = {}
        for p in store.camera_frames:
            meta = p.get("meta", {})
            name = meta.get("camera_name", "unknown")
            dt = meta.get("dt_ms")
            if dt is not None and dt > 0:
                cam_by_name.setdefault(name, []).append(dt)
        camera_fps: dict[str, float] = {}
        for name, dt_vals in cam_by_name.items():
            recent = dt_vals[-50:]
            if recent:
                # dt_ms covers 10 frames, so FPS = 10 / (mean_dt / 1000)
                mean_dt_s = statistics.mean(recent) / 1000.0
                camera_fps[name] = round(10.0 / mean_dt_s, 1) if mean_dt_s > 0 else 0.0
            else:
                camera_fps[name] = 0.0

        # Latest step meta: action_source, reward, max_steps, cumR
        action_source = "unknown"
        max_steps = None
        last_reward = None
        cumulative_reward = None
        if store._step_order:
            last_sd = store.steps.get(store._step_order[-1])
            if last_sd:
                max_steps = last_sd.meta.get("max_steps")
                last_reward = last_sd.meta.get("reward")
                cumulative_reward = last_sd.meta.get("cumR")
                action_source = last_sd.meta.get("action_source", "unknown")
                if action_source == "unknown":
                    if last_sd.rpc_events.get("step_send"):
                        action_source = "rl"
                    elif last_sd.events.get("action_applied"):
                        action_source = "human"

        motor_temp_arms: dict[str, dict[str, Any]] = {}
        for arm_name, row in store.motor_temps.items():
            temps = row.get("temps") or []
            safe_temps = [float(v) if v is not None else None for v in temps]
            row_max = row.get("max")
            if row_max is not None:
                row_max = float(row_max)
            motor_temp_arms[arm_name] = {
                "status": str(row.get("status", "-")),
                "temps": safe_temps,
                "max": row_max,
            }
        arm_max_vals = [
            r["max"]
            for r in motor_temp_arms.values()
            if r.get("max") is not None and math.isfinite(float(r["max"]))
        ]
        motor_temp_max = max(arm_max_vals) if arm_max_vals else None
        motor_temp_age_s = (
            max(0.0, time.time() - store.motor_temps_updated_at)
            if store.motor_temps_updated_at > 0
            else None
        )

    return {
        "episode": ep,
        "step": step,
        "max_steps": max_steps,
        "loop_hz": loop_hz,
        "control_loop_hz": ctrl_hz,
        "control_jitter_ms": ctrl_jitter,
        "fello_hz": fello_hz,
        "camera_fps": camera_fps,
        "action_source": action_source,
        "last_reward": round(last_reward, 3) if last_reward is not None else None,
        "cumulative_reward": round(cumulative_reward, 2) if cumulative_reward is not None else None,
        "motor_temp_max": round(motor_temp_max, 1) if motor_temp_max is not None else None,
        "motor_temp_arms": motor_temp_arms,
        "motor_temp_age_s": round(motor_temp_age_s, 1) if motor_temp_age_s is not None else None,
    }


# ---------------------------------------------------------------------------
# UDP collector thread
# ---------------------------------------------------------------------------

def udp_collector(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            pkt = msgpack.unpackb(data, raw=False)
            STORE.ingest(pkt)
        except socket.timeout:
            continue
        except Exception:
            continue


MOTOR_TEMP_ENDPOINTS = [
    ("follower_left", LEFT_FOLLOWER_PORT),
    ("follower_right", RIGHT_FOLLOWER_PORT),
    ("leader_left", LEFT_LEADER_PORT),
    ("leader_right", RIGHT_LEADER_PORT),
]


def _sanitize_temp_list(values: Any) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out: list[float | None] = []
    for v in arr:
        fv = float(v)
        out.append(round(fv, 1) if math.isfinite(fv) else None)
    return out


def motor_temp_collector(host: str, refresh_hz: float, rpc_timeout_s: float) -> None:
    clients: dict[str, portal.Client] = {
        name: portal.Client(f"{host}:{port}") for name, port in MOTOR_TEMP_ENDPOINTS
    }
    period = 1.0 / max(refresh_hz, 0.2)
    while True:
        now = time.time()
        rows: dict[str, dict[str, Any]] = {}
        for name, _ in MOTOR_TEMP_ENDPOINTS:
            try:
                data = clients[name].get_motor_temperatures().result(timeout=rpc_timeout_s)
                temps = _sanitize_temp_list(data)
                valid = [v for v in temps if v is not None]
                rows[name] = {
                    "status": "ok",
                    "temps": temps,
                    "max": round(max(valid), 1) if valid else None,
                }
            except Exception as e:
                rows[name] = {"status": f"down ({type(e).__name__})", "temps": [], "max": None}
        STORE.update_motor_temps(rows, now)
        time.sleep(period)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RL Pipeline Diagnostics")


def _build_snapshot() -> dict:
    recent = STORE.recent_steps(20)
    stats_steps = STORE.stats_window(300)
    return {
        "steps": [step_to_dict(s) for s in recent],
        "stats": compute_stats(stats_steps),
        "health": compute_health(STORE),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            snapshot = _build_snapshot()
            await ws.send_text(json.dumps(snapshot))
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/api/stats")
async def api_stats():
    return _build_snapshot()


# ---------------------------------------------------------------------------
# Dashboard HTML (inline, no build step)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Pipeline Diagnostics</title>
<style>
  :root {
    --bg: #1a1a2e;
    --bg2: #16213e;
    --bg3: #0f3460;
    --fg: #e0e0e0;
    --fg2: #a0a0b8;
    --accent: #e94560;
    --blue: #4aa3df;
    --orange: #e8a838;
    --green: #44c767;
    --purple: #9b59b6;
    --gray: #555;
    --red: #e94560;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px; line-height: 1.5;
    padding: 16px;
  }
  h1 { font-size: 18px; color: var(--accent); margin-bottom: 12px; }
  h2 { font-size: 14px; color: var(--fg2); margin: 14px 0 8px 0; text-transform: uppercase; letter-spacing: 1px; }

  /* Health Panel */
  .health-panel {
    display: flex; flex-wrap: wrap; gap: 12px;
    background: var(--bg2); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 16px;
    border: 1px solid #2a2a4a;
  }
  .health-item {
    display: flex; flex-direction: column; min-width: 120px;
  }
  .health-label { font-size: 10px; color: var(--fg2); text-transform: uppercase; letter-spacing: 0.5px; }
  .health-value { font-size: 20px; font-weight: bold; color: var(--fg); }
  .health-value.good { color: var(--green); }
  .health-value.warn { color: var(--orange); }
  .health-value.bad { color: var(--red); }
  .health-value.rl { color: var(--blue); }
  .health-value.human { color: var(--orange); }

  /* Timeline */
  .timeline-container {
    background: var(--bg2); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 16px;
    border: 1px solid #2a2a4a;
    overflow-x: auto;
  }
  .step-row {
    display: flex; align-items: center; height: 22px; margin-bottom: 3px;
  }
  .step-label {
    width: 80px; flex-shrink: 0; font-size: 11px; color: var(--fg2);
    text-align: right; padding-right: 8px;
  }
  .step-bar-container {
    flex: 1; display: flex; height: 16px; border-radius: 3px; overflow: hidden;
    background: #111;
  }
  .seg {
    height: 100%; position: relative; min-width: 1px;
    transition: width 0.15s ease;
  }
  .seg:hover { opacity: 0.8; }
  .seg[data-label]::after {
    content: attr(data-label);
    position: absolute; top: -18px; left: 50%; transform: translateX(-50%);
    font-size: 9px; color: var(--fg); background: #000a; padding: 1px 4px;
    border-radius: 2px; white-space: nowrap; pointer-events: none;
    opacity: 0; transition: opacity 0.1s;
  }
  .seg:hover::after { opacity: 1; }
  .seg.fello_read  { background: #3d7ea6; }
  .seg.build_obs   { background: var(--blue); }
  .seg.rpc_call    { background: var(--orange); }
  .seg.reward      { background: var(--green); }
  .seg.apply_action { background: #5dade2; }
  .seg.record      { background: var(--purple); }
  .seg.idle        { background: var(--gray); }
  .seg.over-thresh { box-shadow: inset 0 0 0 2px var(--red); }
  .step-total {
    width: 60px; flex-shrink: 0; text-align: right; font-size: 11px;
    color: var(--fg2); padding-left: 6px;
  }

  /* Legend */
  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }
  .legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--fg2); }
  .legend-swatch { width: 12px; height: 12px; border-radius: 2px; }

  /* Stats Table */
  .stats-container {
    background: var(--bg2); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 16px;
    border: 1px solid #2a2a4a;
    overflow-x: auto;
  }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; padding: 6px 10px; font-size: 11px;
    color: var(--fg2); text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid #2a2a4a;
  }
  td {
    padding: 5px 10px; font-size: 12px; border-bottom: 1px solid #1a1a3a;
  }
  tr:hover td { background: #1a1a3a; }
  td.seg-name { color: var(--fg); font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.num.hot { color: var(--red); }

  /* Episode Overview */
  .episode-container {
    background: var(--bg2); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 16px;
    border: 1px solid #2a2a4a;
  }
  .ep-chart { display: flex; align-items: flex-end; gap: 1px; height: 60px; }
  .ep-bar {
    flex: 1; min-width: 2px; max-width: 8px;
    border-radius: 1px 1px 0 0; transition: height 0.15s ease;
  }

  .status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 4px; animation: pulse 2s infinite;
  }
  .status-dot.connected { background: var(--green); }
  .status-dot.disconnected { background: var(--red); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  .cameras-row { display: flex; flex-wrap: wrap; gap: 8px; }
  .cam-chip {
    background: var(--bg3); border-radius: 4px; padding: 3px 8px;
    font-size: 11px;
  }
  .cam-chip .cam-name { color: var(--fg2); }
  .cam-chip .cam-fps { color: var(--fg); font-weight: 600; margin-left: 4px; }

  /* Episode Progress Bar */
  .progress-bar-container {
    width: 100%; height: 6px; background: #111; border-radius: 3px;
    margin-top: 4px; overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%; border-radius: 3px; transition: width 0.2s ease;
    background: linear-gradient(90deg, var(--blue), var(--green));
  }

  .motor-details {
    margin-top: 6px;
    font-size: 11px;
    width: 100%;
  }
  .motor-details summary {
    color: var(--fg2);
    cursor: pointer;
    user-select: none;
  }
  .motor-details summary:hover {
    color: var(--fg);
  }
  .motor-grid {
    margin-top: 8px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 8px;
  }
  .motor-card {
    background: var(--bg3);
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 6px 8px;
  }
  .motor-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 4px;
  }
  .motor-name {
    color: var(--fg);
    font-weight: 600;
  }
  .motor-status {
    color: var(--fg2);
    font-size: 10px;
  }
  .motor-temps {
    color: var(--fg2);
    font-size: 11px;
    line-height: 1.4;
    word-break: break-word;
  }
</style>
</head>
<body>
<h1><span class="status-dot disconnected" id="ws-dot"></span> RL Pipeline Diagnostics</h1>

<!-- Health Panel -->
<div class="health-panel" id="health-panel">
  <div class="health-item">
    <span class="health-label">Episode</span>
    <span class="health-value" id="h-ep">--</span>
  </div>
  <div class="health-item" style="min-width:180px;">
    <span class="health-label">Step Progress</span>
    <span class="health-value" id="h-step">--</span>
    <div class="progress-bar-container">
      <div class="progress-bar-fill" id="h-progress-bar" style="width:0%"></div>
    </div>
  </div>
  <div class="health-item">
    <span class="health-label">Reward</span>
    <span class="health-value" id="h-reward">--</span>
  </div>
  <div class="health-item">
    <span class="health-label">Cum. Reward</span>
    <span class="health-value" id="h-cum-reward">--</span>
  </div>
  <div class="health-item">
    <span class="health-label">Loop Hz</span>
    <span class="health-value" id="h-loop-hz">--</span>
  </div>
  <div class="health-item">
    <span class="health-label">Action</span>
    <span class="health-value" id="h-action">--</span>
  </div>
  <div class="health-item">
    <span class="health-label">Ctrl Loop Hz</span>
    <span class="health-value" id="h-ctrl-hz">--</span>
  </div>
  <div class="health-item">
    <span class="health-label">Ctrl Jitter</span>
    <span class="health-value" id="h-ctrl-jitter">--</span>
  </div>
  <div class="health-item">
    <span class="health-label">Fello Hz</span>
    <span class="health-value" id="h-fello-hz">--</span>
  </div>
  <div class="health-item" style="flex-grow:1;">
    <span class="health-label">Camera FPS</span>
    <div class="cameras-row" id="h-cameras">--</div>
  </div>
  <div class="health-item" style="min-width:210px; flex-grow:1;">
    <span class="health-label">Motor Temp Max</span>
    <span class="health-value" id="h-motor-max">--</span>
    <details class="motor-details">
      <summary id="h-motor-summary">Show all motor temperatures</summary>
      <div class="motor-grid" id="h-motor-details">--</div>
    </details>
  </div>
</div>

<!-- Timeline -->
<h2>Live Step Timeline</h2>
<div class="timeline-container">
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#3d7ea6"></div>Fello Read</div>
    <div class="legend-item"><div class="legend-swatch" style="background:var(--blue)"></div>Build Obs</div>
    <div class="legend-item"><div class="legend-swatch" style="background:var(--orange)"></div>RPC Call</div>
    <div class="legend-item"><div class="legend-swatch" style="background:var(--green)"></div>Reward</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#5dade2"></div>Apply Action</div>
    <div class="legend-item"><div class="legend-swatch" style="background:var(--purple)"></div>Record</div>
    <div class="legend-item"><div class="legend-swatch" style="background:var(--gray)"></div>Idle</div>
  </div>
  <div id="timeline"></div>
</div>

<!-- Latency Stats -->
<h2>Latency Stats (last 300 steps, ms)</h2>
<div class="stats-container">
  <table>
    <thead>
      <tr><th>Segment</th><th style="text-align:right">Mean</th><th style="text-align:right">P50</th><th style="text-align:right">P95</th><th style="text-align:right">Max</th></tr>
    </thead>
    <tbody id="stats-body"></tbody>
  </table>
</div>

<!-- Episode Overview -->
<h2>Episode Overview (step loop time)</h2>
<div class="episode-container">
  <div class="ep-chart" id="ep-chart"></div>
</div>

<script>
const SEG_ORDER = ["fello_read","build_obs","rpc_call","reward","apply_action","record","idle"];
const RPC_ORDER = ["base_action","sac_sample","buffer_insert","network_rtt"];
const ALL_STATS = ["fello_read","build_obs","rpc_call","reward","apply_action","record","idle","total",
                   "base_action","sac_sample","buffer_insert","network_rtt"];
const THRESH_MS = {"fello_read":2,"build_obs":3,"rpc_call":20,"reward":5,"apply_action":2,"record":3,"idle":15,"total":40};
const TARGET_TOTAL = 33.3; // 30Hz

let ws;
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws");
  ws.onopen = () => { document.getElementById("ws-dot").className = "status-dot connected"; };
  ws.onclose = () => {
    document.getElementById("ws-dot").className = "status-dot disconnected";
    setTimeout(connect, 2000);
  };
  ws.onmessage = (ev) => {
    try { render(JSON.parse(ev.data)); } catch(e) { console.error(e); }
  };
}
connect();

function fmt(v) { return v != null ? v.toFixed(1) : "--"; }
function fmtHz(v) { return v != null && v > 0 ? v.toFixed(1) : "--"; }
function fmtTemp(v) { return v != null ? v.toFixed(1) : "--"; }

function hzClass(actual, target, tolerance) {
  if (actual == null || actual <= 0) return "";
  const ratio = actual / target;
  if (ratio >= (1 - tolerance)) return "good";
  if (ratio >= (1 - tolerance * 2.5)) return "warn";
  return "bad";
}

function render(data) {
  // Health
  const h = data.health || {};
  document.getElementById("h-ep").textContent = h.episode ?? "--";

  // Step progress with countdown
  const stepEl = document.getElementById("h-step");
  const progBar = document.getElementById("h-progress-bar");
  if (h.max_steps != null && h.step != null) {
    const remaining = h.max_steps - h.step;
    stepEl.textContent = h.step + " / " + h.max_steps + "  (" + remaining + " left)";
    progBar.style.width = Math.min(100, (h.step / h.max_steps) * 100) + "%";
  } else {
    stepEl.textContent = h.step ?? "--";
    progBar.style.width = "0%";
  }

  // Reward
  const rewEl = document.getElementById("h-reward");
  rewEl.textContent = h.last_reward != null ? h.last_reward.toFixed(2) : "--";
  rewEl.className = "health-value" + (h.last_reward != null && h.last_reward > 0 ? " good" : "");

  const cumREl = document.getElementById("h-cum-reward");
  cumREl.textContent = h.cumulative_reward != null ? h.cumulative_reward.toFixed(1) : "--";
  cumREl.className = "health-value" + (h.cumulative_reward != null && h.cumulative_reward > 0 ? " good" : "");

  const loopEl = document.getElementById("h-loop-hz");
  loopEl.textContent = fmtHz(h.loop_hz) + " Hz";
  loopEl.className = "health-value " + hzClass(h.loop_hz, 30, 0.05);

  const actEl = document.getElementById("h-action");
  actEl.textContent = (h.action_source || "--").toUpperCase();
  actEl.className = "health-value " + (h.action_source === "rl" ? "rl" : h.action_source === "human" ? "human" : "");

  const ctrlEl = document.getElementById("h-ctrl-hz");
  ctrlEl.textContent = fmtHz(h.control_loop_hz) + " Hz";
  ctrlEl.className = "health-value " + hzClass(h.control_loop_hz, 200, 0.02);

  const jitEl = document.getElementById("h-ctrl-jitter");
  jitEl.textContent = fmt(h.control_jitter_ms) + " ms";
  jitEl.className = "health-value " + (h.control_jitter_ms != null && h.control_jitter_ms < 1 ? "good" : h.control_jitter_ms < 3 ? "warn" : "bad");

  const felloEl = document.getElementById("h-fello-hz");
  felloEl.textContent = fmtHz(h.fello_hz) + " Hz";
  felloEl.className = "health-value " + hzClass(h.fello_hz, 30, 0.1);

  const motorMaxEl = document.getElementById("h-motor-max");
  motorMaxEl.textContent = fmtTemp(h.motor_temp_max) + " C";
  motorMaxEl.className = "health-value " + (
    h.motor_temp_max == null ? "" :
    h.motor_temp_max < 60 ? "good" :
    h.motor_temp_max < 70 ? "warn" : "bad"
  );

  // Cameras
  const camEl = document.getElementById("h-cameras");
  const cams = h.camera_fps || {};
  const camNames = Object.keys(cams).sort();
  if (camNames.length) {
    camEl.innerHTML = camNames.map(n =>
      `<div class="cam-chip"><span class="cam-name">${n}</span><span class="cam-fps">${fmtHz(cams[n])}</span></div>`
    ).join("");
  } else {
    camEl.textContent = "--";
  }

  // Motor temperatures
  const motorSummaryEl = document.getElementById("h-motor-summary");
  const motorDetailsEl = document.getElementById("h-motor-details");
  const armTemps = h.motor_temp_arms || {};
  const armNames = Object.keys(armTemps).sort();
  const age = h.motor_temp_age_s != null ? ` (updated ${fmt(h.motor_temp_age_s)}s ago)` : "";
  motorSummaryEl.textContent = "Show all motor temperatures" + age;
  if (armNames.length) {
    motorDetailsEl.innerHTML = armNames.map((name) => {
      const row = armTemps[name] || {};
      const status = row.status || "-";
      const rowMax = row.max != null ? `${fmtTemp(row.max)} C` : "--";
      const temps = Array.isArray(row.temps) ? row.temps : [];
      const tempText = temps.length
        ? temps.map((v, i) => `M${i + 1}:${fmtTemp(v)}`).join("  ")
        : "--";
      return `<div class="motor-card">
        <div class="motor-head">
          <span class="motor-name">${name}</span>
          <span class="motor-status">${status} | max ${rowMax}</span>
        </div>
        <div class="motor-temps">${tempText}</div>
      </div>`;
    }).join("");
  } else {
    motorDetailsEl.textContent = "--";
  }

  // Timeline
  renderTimeline(data.steps || []);

  // Stats
  renderStats(data.stats || {});

  // Episode chart
  renderEpisode(data.steps || []);
}

function renderTimeline(steps) {
  const el = document.getElementById("timeline");
  let html = "";
  for (const s of steps) {
    const segs = s.segments || {};
    const total = segs.total || TARGET_TOTAL;
    html += `<div class="step-row">`;
    html += `<div class="step-label">${s.ep}:${s.step}</div>`;
    html += `<div class="step-bar-container">`;
    for (const seg of SEG_ORDER) {
      const v = segs[seg];
      if (v == null || v <= 0) continue;
      const pct = Math.max(0.3, (v / total) * 100);
      const over = THRESH_MS[seg] && v > THRESH_MS[seg] ? " over-thresh" : "";
      html += `<div class="seg ${seg}${over}" style="width:${pct}%" data-label="${seg} ${v.toFixed(1)}ms"></div>`;
    }
    html += `</div>`;
    html += `<div class="step-total">${fmt(segs.total)} ms</div>`;
    html += `</div>`;
  }
  el.innerHTML = html;
}

function renderStats(stats) {
  const tbody = document.getElementById("stats-body");
  let html = "";
  for (const seg of ALL_STATS) {
    const s = stats[seg];
    if (!s) continue;
    const isRpc = RPC_ORDER.includes(seg);
    const prefix = isRpc ? "&nbsp;&nbsp;&rarr; " : "";
    const hot95 = s.p95 != null && THRESH_MS[seg] && s.p95 > THRESH_MS[seg] ? " hot" : "";
    const hotMax = s.max != null && THRESH_MS[seg] && s.max > THRESH_MS[seg] * 1.5 ? " hot" : "";
    html += `<tr>`;
    html += `<td class="seg-name">${prefix}${seg}</td>`;
    html += `<td class="num">${fmt(s.mean)}</td>`;
    html += `<td class="num">${fmt(s.p50)}</td>`;
    html += `<td class="num${hot95}">${fmt(s.p95)}</td>`;
    html += `<td class="num${hotMax}">${fmt(s.max)}</td>`;
    html += `</tr>`;
  }
  tbody.innerHTML = html;
}

function renderEpisode(steps) {
  const el = document.getElementById("ep-chart");
  if (!steps.length) { el.innerHTML = ""; return; }
  const maxT = Math.max(...steps.map(s => s.total_ms || 0), TARGET_TOTAL);
  let html = "";
  for (const s of steps) {
    const t = s.total_ms || 0;
    const h = Math.max(2, (t / maxT) * 56);
    const color = s.action_source === "rl" ? "var(--blue)" :
                  s.action_source === "human" ? "var(--orange)" : "var(--gray)";
    const border = t > TARGET_TOTAL ? "2px solid var(--red)" : "none";
    html += `<div class="ep-bar" style="height:${h}px;background:${color};border-top:${border}" title="step ${s.step}: ${fmt(t)}ms"></div>`;
  }
  el.innerHTML = html;
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Pipeline Diagnostics Dashboard")
    parser.add_argument("--udp-port", type=int, default=9999, help="UDP listen port (default: 9999)")
    parser.add_argument("--http-port", type=int, default=8888, help="HTTP/WS serve port (default: 8888)")
    parser.add_argument("--motor-host", type=str, default="localhost", help="Host for arm RPC servers")
    parser.add_argument("--motor-refresh-hz", type=float, default=2.0, help="Motor temp polling rate")
    parser.add_argument("--motor-rpc-timeout-s", type=float, default=0.5, help="Motor temp RPC timeout")
    args = parser.parse_args()

    # Start UDP collector
    collector = threading.Thread(target=udp_collector, args=(args.udp_port,), daemon=True)
    collector.start()
    motor_collector = threading.Thread(
        target=motor_temp_collector,
        args=(args.motor_host, args.motor_refresh_hz, args.motor_rpc_timeout_s),
        daemon=True,
    )
    motor_collector.start()
    print(f"[diag] UDP collector listening on :{args.udp_port}")
    print(
        f"[diag] Motor temp polling from {args.motor_host} at {args.motor_refresh_hz:.2f} Hz "
        f"(timeout={args.motor_rpc_timeout_s:.2f}s)"
    )
    print(f"[diag] Dashboard at http://localhost:{args.http_port}")

    uvicorn.run(app, host="0.0.0.0", port=args.http_port, log_level="warning")


if __name__ == "__main__":
    main()
