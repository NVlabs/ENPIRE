# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import requests

from enpire.policy.rl.context import RLContext


def dispatch_keyboard_events(ctx: RLContext, keyboard_info: dict) -> None:
    just = keyboard_info.get("_just_pressed_keys", set())
    shift = keyboard_info.get("shift_held", False)
    home_key = str(getattr(ctx.cfg, "keyboard_home_key", "KEY_H") or "").strip()
    start_key = str(getattr(ctx.cfg, "keyboard_start_key", "KEY_S") or "").strip()
    parking_key = str(getattr(ctx.cfg, "keyboard_parking_key", "KEY_P") or "").strip()
    if home_key and home_key in just:
        print("[keyboard] home -> enqueued 'home'", flush=True)
        ctx.external_event_queue.put(("home", {"source": "keyboard"}))
    if start_key and start_key in just:
        print("[keyboard] start -> enqueued 'start'", flush=True)
        ctx.external_event_queue.put(("start", {"source": "keyboard"}))
    if parking_key and parking_key in just:
        print("[keyboard] parking -> enqueued 'parking'", flush=True)
        ctx.external_event_queue.put(("parking", {"source": "keyboard"}))
    if "KEY_A" in just and ctx.cfg.enable_author_mode:
        ctx.external_event_queue.put(("author", {}))
    if "KEY_ESC" in just:
        ctx.external_event_queue.put(("discard_author", {}))
    if "KEY_R" in just:
        ctx.external_event_queue.put(("init_boundary", {}))
    if "KEY_O" in just:
        ctx.external_event_queue.put(("oor_boundary", {}))
    if "KEY_PAGEUP" in just:
        ctx.external_event_queue.put(("z_high", {}))
    if "KEY_PAGEDOWN" in just:
        ctx.external_event_queue.put(("z_low", {}))
    terminal_labels_enabled = ctx.state_machine.state == "learn"
    if terminal_labels_enabled and ctx.cfg.keyboard_success_key in just:
        print("[keyboard] success -> enqueued 'success'", flush=True)
        ctx.external_event_queue.put(("success", {"source": "keyboard"}))
    if terminal_labels_enabled and ctx.cfg.keyboard_fail_key in just:
        print("[keyboard] fail -> enqueued 'fail'", flush=True)
        ctx.external_event_queue.put(("fail", {"source": "keyboard"}))
    if ctx.cfg.restart_key in just:
        _fire_async_http(f"{ctx.fastapi_base_url}/restart")
    if ctx.cfg.auto_eval_key in just and ctx.cfg.enable_eval_mode:
        if ctx.state_machine.state in ("idle", "home"):
            ctx.external_event_queue.put(("auto_eval_start", {}))
        else:
            print(
                f"[auto_eval] ignored {ctx.cfg.auto_eval_key}: state={ctx.state_machine.state} (only allowed at idle/home)",
                flush=True,
            )
    if shift and "KEY_COMMA" in just:
        ctx.external_event_queue.put(("prev_pose", {}))
    if shift and "KEY_DOT" in just:
        ctx.external_event_queue.put(("next_pose", {}))


def _fire_async_http(url: str) -> None:
    def _post() -> None:
        try:
            requests.post(url, timeout=1.0)
        except requests.RequestException as exc:
            print(f"[Warn] keyboard->HTTP failed for {url}: {exc}", flush=True)

    threading.Thread(target=_post, daemon=True).start()

