# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe client for the real-world RL reset/rollout control surface."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

_ACTIONS = {
    "health": ("GET", "healthz"),
    "help": ("GET", "help"),
    "home": ("POST", "home"),
    "pause": ("POST", "pause"),
    "restart": ("POST", "restart"),
    "resume": ("POST", "resume"),
}


@dataclass(frozen=True)
class ControlRequest:
    action: str
    method: str
    url: str


def build_control_request(action: str, base_url: str) -> ControlRequest:
    try:
        method, endpoint = _ACTIONS[action]
    except KeyError as exc:
        raise ValueError(f"Unknown control action {action!r}") from exc
    return ControlRequest(action, method, f"{base_url.rstrip('/')}/{endpoint}")


def send_control_request(request: ControlRequest, *, timeout_s: float = 3.0) -> dict:
    payload = b"" if request.method == "POST" else None
    wire_request = urllib.request.Request(request.url, data=payload, method=request.method)
    try:
        with urllib.request.urlopen(wire_request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"RL control request failed: {request.method} {request.url}: {exc}") from exc
    decoded = json.loads(body) if body else {}
    if not isinstance(decoded, dict):
        raise RuntimeError(f"RL control returned a non-object response: {decoded!r}")
    return decoded

