# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin HTTP client for the pyroki IK / trajopt service.

Mirrors cap-x's ``capx/integrations/motion/pyroki.py`` (MIT). The server is
``scripts/launch_pyroki_server.py`` in forge; it defaults to
``tcp://127.0.0.1:9600`` — one-per-node, same model as cuRobo.

Usage::

    from enpire.env.forge.cap.integrations.motion.pyroki import init_pyroki, init_pyroki_trajopt

    ik_solve_fn   = init_pyroki(server_url="http://127.0.0.1:9600")
    joints        = ik_solve_fn(target_pose_wxyz_xyz, prev_cfg=cur_joints)

    trajopt_fn    = init_pyroki_trajopt(server_url="http://127.0.0.1:9600")
    waypoints     = trajopt_fn(start_pose_wxyz_xyz, end_pose_wxyz_xyz)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:9600"


# ---------------------------------------------------------------------------
# HTTP helper (retries with exponential back-off)
# ---------------------------------------------------------------------------


def _post_with_retries(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
    retry_interval: float = 0.5,
    max_retries: int = 5,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    interval = retry_interval
    last_err: Exception | None = None
    for attempt in range(max_retries):
        if time.time() >= deadline:
            break
        try:
            resp = requests.post(url, json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(interval, max(0.0, deadline - time.time())))
            interval = min(interval * 2, 8.0)
    raise RuntimeError(
        f"pyroki POST {url} failed after {max_retries} retries "
        f"({timeout_seconds:.1f}s). Last error: {last_err}"
    )


def ping(server_url: str = DEFAULT_URL, *, timeout_seconds: float = 3.0) -> bool:
    """Cheap health check — True iff the server is up and the URDF is loaded."""
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/health", timeout=timeout_seconds)
        resp.raise_for_status()
        return bool(resp.json().get("ready", False))
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Factory functions (drop-in compatible with cap-x signatures)
# ---------------------------------------------------------------------------


def init_pyroki(
    server_url: str = DEFAULT_URL,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Return an IK solver callable.

    Signature of the returned callable:

        ``ik_solve_fn(target_pose_wxyz_xyz: (7,) float,
                      prev_cfg: (N,) float | None) -> (N,) float``

    ``N`` is the number of URDF-actuated joints — for the Panda URDF this is
    **8** (7 arm + 1 finger), not 7. Strip the last element to get the arm
    joints for a composite controller: ``arm_joints = cfg[:-1]``. Mirrors
    cap-x's convention in ``capx/integrations/franka/control.py``.

    ``prev_cfg`` (optional, recommended): full ``(N,)`` vector from the
    previous IK solve. Pins the redundant DOF of a 7-DOF arm near the
    previous configuration — prevents elbow drift across a sequence of small
    EE moves. Matches what cuRobo does internally.
    """
    server_url = server_url.rstrip("/")

    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        payload = {
            "target_pose_wxyz_xyz": np.asarray(
                target_pose_wxyz_xyz, dtype=np.float64
            ).tolist(),
            "prev_cfg": np.asarray(prev_cfg, dtype=np.float64).tolist()
            if prev_cfg is not None
            else None,
        }
        data = _post_with_retries(f"{server_url}/ik", payload, timeout_seconds=15.0)
        return np.asarray(data["joint_positions"], dtype=np.float32)

    return ik_solve_fn


def init_pyroki_trajopt(
    server_url: str = DEFAULT_URL,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a trajectory planner callable.

    Signature:

        ``trajopt_plan_fn(start_pose_wxyz_xyz, end_pose_wxyz_xyz) -> (N, 7) float``

    Uses straight-line SE(3) interpolation + per-waypoint IK on the server
    side; fine for free-space motions without clutter. For collision-aware
    planning keep using cuRobo via ``experimental.portal_motion_planner``.
    """
    server_url = server_url.rstrip("/")

    def trajopt_plan_fn(
        start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        payload = {
            "start_pose_wxyz_xyz": np.asarray(
                start_pose_wxyz_xyz, dtype=np.float64
            ).tolist(),
            "end_pose_wxyz_xyz": np.asarray(
                end_pose_wxyz_xyz, dtype=np.float64
            ).tolist(),
        }
        data = _post_with_retries(f"{server_url}/plan", payload, timeout_seconds=60.0)
        return np.asarray(data["waypoints"], dtype=np.float32)

    return trajopt_plan_fn


__all__ = ["init_pyroki", "init_pyroki_trajopt", "ping", "DEFAULT_URL"]
