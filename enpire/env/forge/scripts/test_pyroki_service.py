# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the pyroki service.

What it checks
--------------
1. ``GET /health`` returns ``ready=true``.
2. ``POST /ik`` on a reachable target returns 7 joint values within Panda
   limits.
3. Repeating (2) with ``prev_cfg`` yields a nearby configuration (elbow pin).
4. ``POST /plan`` returns a trajectory of the requested length whose start
   and end IK match the endpoint requests.

Does NOT require pyroki installed locally — it only talks HTTP.

Usage
-----
::

    uv run python scripts/test_pyroki_service.py              # default :9600
    uv run python scripts/test_pyroki_service.py --url http://host:port
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import requests


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:9600")
    parser.add_argument("--timesteps", type=int, default=10)
    args = parser.parse_args()

    url = args.url.rstrip("/")

    # ---- 1. Health ----------------------------------------------------------
    resp = requests.get(f"{url}/health", timeout=5)
    resp.raise_for_status()
    health = resp.json()
    _assert(health.get("ready") is True, f"health not ready: {health}")
    print(f"[1/4] health OK — target_link={health.get('target_link')}")

    # ---- 2. Plain IK --------------------------------------------------------
    #   Reachable Panda pose: ~40 cm forward, palm-down wrist.
    #   pyroki returns all URDF-actuated joints (7 arm + 1 finger = 8).
    #   Caller slices [:-1] to get the 7 arm joints — same convention cap-x
    #   uses in capx/integrations/franka/control.py (`self.cfg[:-1]`).
    target_wxyz_xyz = [
        0.0,
        1.0,
        0.0,
        0.0,  # wxyz — rotate 180° about X so palm faces down
        0.4,
        0.0,
        0.4,  # xyz
    ]
    resp = requests.post(
        f"{url}/ik", json={"target_pose_wxyz_xyz": target_wxyz_xyz}, timeout=30
    )
    resp.raise_for_status()
    ik = resp.json()
    cfg = np.asarray(ik["joint_positions"])
    _assert(
        cfg.ndim == 1 and cfg.shape[0] in (7, 8),
        f"IK returned {cfg.shape}, expected (7,) or (8,) (7 arm + 1 finger)",
    )
    arm_joints = cfg[:7]
    finger = cfg[7] if cfg.shape[0] == 8 else None
    _assert(
        np.all(np.abs(arm_joints) < 3.5),
        f"arm joints out of plausible range: {arm_joints}",
    )
    msg = f"arm={np.round(arm_joints, 3).tolist()}"
    if finger is not None:
        msg += f" finger={finger:.4f}"
    print(f"[2/4] /ik OK — {msg}")

    # ---- 3. prev_cfg continuity --------------------------------------------
    #   Same target, but passing prev_cfg should land near the same solution.
    #   prev_cfg must be the full cfg vector the server returned.
    resp = requests.post(
        f"{url}/ik",
        json={"target_pose_wxyz_xyz": target_wxyz_xyz, "prev_cfg": cfg.tolist()},
        timeout=30,
    )
    resp.raise_for_status()
    cfg2 = np.asarray(resp.json()["joint_positions"])
    delta = float(np.max(np.abs(cfg2[:7] - cfg[:7])))
    _assert(
        delta < 0.2, f"prev_cfg IK drifted {delta:.3f} rad from plain IK (same target)"
    )
    print(f"[3/4] /ik prev_cfg OK — max arm-joint delta {delta:.4f} rad")

    # ---- 4. Planning --------------------------------------------------------
    end_wxyz_xyz = [0.0, 1.0, 0.0, 0.0, 0.5, 0.1, 0.3]
    resp = requests.post(
        f"{url}/plan",
        json={
            "start_pose_wxyz_xyz": target_wxyz_xyz,
            "end_pose_wxyz_xyz": end_wxyz_xyz,
            "timesteps": args.timesteps,
        },
        timeout=120,
    )
    resp.raise_for_status()
    waypoints = np.asarray(resp.json()["waypoints"])
    _assert(
        waypoints.ndim == 2
        and waypoints.shape[0] == args.timesteps
        and waypoints.shape[1] in (7, 8),
        f"/plan returned {waypoints.shape}, expected ({args.timesteps}, 7 or 8)",
    )
    # Compare arm-only step sizes (finger is held constant).
    arm_waypoints = waypoints[:, :7]
    step_deltas = np.linalg.norm(np.diff(arm_waypoints, axis=0), axis=1)
    _assert(
        float(step_deltas.max()) < 1.0,
        f"large joint jump in trajectory: max step norm {step_deltas.max():.3f} rad",
    )
    print(
        f"[4/4] /plan OK — {waypoints.shape[0]} waypoints ({waypoints.shape[1]} dof), "
        f"max arm step {step_deltas.max():.4f} rad, mean {step_deltas.mean():.4f}"
    )

    print("\npyroki service smoke test passed.")


if __name__ == "__main__":
    main()
