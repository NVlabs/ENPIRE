# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from enpire.policy.rl.context import RLContext
from enpire.policy.rl.reset_options import terminal_label_options


def maybe_request_gpu_success_full_cycle(
    ctx: RLContext,
    event: str | None,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Finalize a successful GPU episode and request the external reset cycle.

    This is intentionally opt-in.  The normal GPU insertion path keeps using
    hover reset unless ``gpu_success_full_cycle_enabled`` is set by the full-cycle
    launcher.
    """

    cfg = ctx.cfg
    if not bool(getattr(cfg, "gpu_success_full_cycle_enabled", False)):
        return False
    if str(getattr(cfg, "task_name", "")) != "gpu_insertion":
        return False
    if event != "success":
        return False
    if ctx.state_machine.state != "learn":
        return False

    finalize_episode = getattr(ctx.env, "finalize_episode", None)
    if finalize_episode is None:
        raise RuntimeError("GPU full-cycle requested, but env cannot finalize episodes")

    terminal_options = terminal_label_options("success")
    finalize_episode(discard_episode=False, **terminal_options)

    request_path = _resolve_request_path(ctx)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    last_episode_dir = getattr(ctx.env, "last_episode_dir", None)
    output_dir = getattr(ctx.env, "output_dir", None)
    request = {
        "event": "success",
        "task_name": str(getattr(cfg, "task_name", "")),
        "target_socket": int(os.environ.get("GPU_TARGET_SOCKET_NUMBER", "1")),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": dict(payload or {}),
        "episode_dir": str(last_episode_dir) if last_episode_dir is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    _atomic_write_json(request_path, request)
    ctx.timing_log.log(
        "gpu_success_full_cycle_requested",
        request_path=str(request_path),
        episode_dir=request["episode_dir"],
    )
    ctx.terminal_event = None
    print(
        "[gpu_full_cycle] success episode finalized; "
        f"request written to {request_path}. Exiting RL runner for press/unplug.",
        flush=True,
    )
    return True


def _resolve_request_path(ctx: RLContext) -> Path:
    configured = getattr(ctx.cfg, "gpu_success_full_cycle_request_path", None)
    if configured:
        return Path(configured).expanduser()
    output_dir = getattr(ctx.env, "output_dir", None)
    if output_dir is None:
        raise RuntimeError(
            "gpu_success_full_cycle_request_path is required when env has no output_dir"
        )
    return Path(output_dir).expanduser() / "gpu_success_full_cycle_request.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)

