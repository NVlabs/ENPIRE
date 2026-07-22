# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup health checks for runtime services used by agent runs."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        cur = cfg
        for part in path.split("."):
            cur = getattr(cur, part, default)
            if cur is default:
                break
        return cur


def runtime_service_probes(cfg: Any) -> list[dict[str, Any]]:
    """Return configured runtime service probes.

    HTTP services expose `/health`. The cuRobo planner is a Portal service, so
    the startup check intentionally uses a fast TCP connect instead of a Portal
    RPC to avoid blocking the dashboard if the planner process is wedged.
    """

    def _host(name: str, default: str = "127.0.0.1") -> str:
        return str(_cfg_select(cfg, f"runtime.{name}_host", default))

    def _port(name: str, default: int = 0) -> int:
        try:
            return int(_cfg_select(cfg, f"runtime.{name}_port", default) or 0)
        except Exception:
            return 0

    probes: list[dict[str, Any]] = []
    for name in ("sam3", "anygrasp", "bundlesdf"):
        port = _port(name)
        if port > 0:
            host = _host(name)
            probes.append(
                {
                    "name": name,
                    "kind": "http",
                    "host": host,
                    "port": port,
                    "endpoint": f"http://{host}:{port}/health",
                }
            )

    curobo_port = _port("curobo")
    if curobo_port > 0:
        curobo_host = _host("curobo")
        probes.append(
            {
                "name": "curobo",
                "kind": "tcp",
                "host": curobo_host,
                "port": curobo_port,
                "endpoint": f"portal://{curobo_host}:{curobo_port}",
            }
        )
    return probes


def _probe_http(probe: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    endpoint = str(probe["endpoint"])
    request = urllib.request.Request(endpoint, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310
        status_code = int(getattr(resp, "status", 200))
        body = resp.read(256).decode("utf-8", errors="replace")
    if 200 <= status_code < 300:
        return {"status": "healthy", "http_status": status_code, "detail": body[:120]}
    return {
        "status": "unhealthy",
        "http_status": status_code,
        "detail": body[:120],
    }


def _probe_tcp(probe: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    with socket.create_connection(
        (str(probe["host"]), int(probe["port"])),
        timeout=timeout_s,
    ):
        pass
    return {"status": "healthy", "detail": "tcp connect ok"}


def _probe_one(probe: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    started = time.time()
    row = dict(probe)
    try:
        if probe.get("kind") == "http":
            result = _probe_http(probe, timeout_s)
        else:
            result = _probe_tcp(probe, timeout_s)
        row.update(result)
    except urllib.error.HTTPError as exc:
        row.update(
            {
                "status": "unhealthy",
                "http_status": exc.code,
                "error": f"HTTP {exc.code}: {exc.reason}",
            }
        )
    except Exception as exc:
        row.update({"status": "unreachable", "error": str(exc)})
    row["latency_ms"] = (time.time() - started) * 1000.0
    return row


def check_runtime_services(
    cfg: Any,
    *,
    timeout_s: float = 1.0,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Probe all configured runtime services and return dashboard rows."""

    probes = runtime_service_probes(cfg)
    if not probes:
        return []

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(probes))) as pool:
        future_to_probe = {
            pool.submit(_probe_one, probe, timeout_s): probe for probe in probes
        }
        for fut in as_completed(future_to_probe, timeout=timeout_s * len(probes) + 1.0):
            try:
                results.append(fut.result())
            except Exception as exc:
                probe = future_to_probe[fut]
                row = dict(probe)
                row.update({"status": "unreachable", "error": str(exc)})
                results.append(row)

    order = {name: i for i, name in enumerate(["sam3", "anygrasp", "bundlesdf", "curobo"])}
    results.sort(key=lambda r: order.get(str(r.get("name")), 99))
    return results


def save_service_health(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def format_service_health_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        status = str(row.get("status", "unknown"))
        endpoint = str(row.get("endpoint", ""))
        latency = row.get("latency_ms")
        latency_s = f" {float(latency):.0f}ms" if isinstance(latency, (int, float)) else ""
        err = row.get("error") or row.get("detail") or ""
        suffix = f" — {err}" if err and status != "healthy" else ""
        lines.append(f"{row.get('name')}: {status} {endpoint}{latency_s}{suffix}")
    return lines
