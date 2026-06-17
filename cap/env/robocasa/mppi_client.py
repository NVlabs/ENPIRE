"""
Client for the remote MPPI navigation planner.

Mirrors the PortalMotionPlanner pattern from experimental/portal_motion_planner.py.
Connects to a remote server (anycar venv with JAX) that runs the MPPI algorithm.

Usage:
    from cap.env.robocasa.mppi_client import MPPINavigationClient

    client = MPPINavigationClient(host="127.0.0.1", port=18700)
    client.update_world(collision_data, floor_bounds)
    result = client.plan_trajectory(current_pos, current_yaw, target_pos, target_yaw)
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from typing import Any

import numpy as np
import portal

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MPPINavigationClient:
    """Client for the remote JAX MPPI navigation planner."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        start_server: bool = False,
        startup_timeout: float = 120.0,
        anycar_root: str | None = None,
    ) -> None:
        self._host = host
        self._port = port or _find_free_port()
        self._client: portal.Client | None = None
        self._process: subprocess.Popen | None = None

        if start_server:
            self._start_subprocess(anycar_root or "/home/lecar-lab/anycar")
        self._wait_for_ready(startup_timeout)

    def _start_subprocess(self, anycar_root: str) -> None:
        cmd = [
            f"{anycar_root}/.venv/bin/python",
            f"{anycar_root}/serve_mppi_planner.py",
            "--port",
            str(self._port),
        ]
        logger.info("Starting MPPI server: %s", " ".join(cmd))
        self._process = subprocess.Popen(cmd)
        logger.info(
            "MPPI server subprocess started (port=%d, pid=%d)",
            self._port,
            self._process.pid,
        )

    def _wait_for_ready(self, timeout: float) -> None:
        start = time.time()
        while time.time() - start < timeout:
            try:
                self._client = portal.Client(f"{self._host}:{self._port}", logging=False)
                if self._client.health_check().result(timeout=5):
                    logger.info(
                        "MPPI server ready at %s:%d", self._host, self._port
                    )
                    return
            except Exception:
                time.sleep(0.5)
        raise TimeoutError(
            f"MPPI server not ready at {self._host}:{self._port} "
            f"after {timeout:.0f}s"
        )

    def update_world(
        self,
        collision_data: dict[str, Any],
        floor_bounds: tuple[np.ndarray, np.ndarray],
        *,
        resolution: float = 0.08,
        inflate_radius: float = 0.15,
        height_range: tuple[float, float] = (0.05, 0.8),
    ) -> dict:
        """Convert collision_data from get_collision_geoms() to 2D boxes and
        send to the MPPI server.

        Args:
            collision_data: dict from RoboCasaEnv.get_collision_geoms()
            floor_bounds: (min_xy, max_xy) as (2,) arrays
            height_range: only include geoms within this Z range
        """
        positions = np.asarray(collision_data["positions"])
        dims = np.asarray(collision_data["dims_array"])
        n = int(collision_data["n_geoms"])

        boxes = []
        for i in range(n):
            z = float(positions[i, 2])
            h = float(dims[i, 2])
            if z - h / 2 > height_range[1] or z + h / 2 < height_range[0]:
                continue
            cx = float(positions[i, 0])
            cy = float(positions[i, 1])
            hw = float(dims[i, 0]) / 2.0
            hh = float(dims[i, 1]) / 2.0
            boxes.append([cx, cy, hw, hh])

        payload = {
            "boxes": boxes,
            "bounds_min": np.asarray(floor_bounds[0], dtype=np.float64)[:2].tolist(),
            "bounds_max": np.asarray(floor_bounds[1], dtype=np.float64)[:2].tolist(),
            "resolution": resolution,
            "inflate_radius": inflate_radius,
        }
        assert self._client is not None
        return self._client.update_world(payload).result(timeout=120)

    def plan_trajectory(
        self,
        current_pos: np.ndarray,
        current_yaw: float,
        target_pos: np.ndarray,
        target_yaw: float,
        *,
        max_steps: int = 400,
        goal_tolerance: float = 0.05,
        yaw_tolerance: float = 0.1,
    ) -> dict:
        """Plan a collision-free trajectory from current to target.

        Returns dict with: status, trajectory (T+1,3), actions (T,3), n_steps,
        final_pos_error, final_yaw_error.
        """
        payload = {
            "current_pos": np.asarray(current_pos, dtype=np.float64)[:2].tolist(),
            "current_yaw": float(current_yaw),
            "target_pos": np.asarray(target_pos, dtype=np.float64)[:2].tolist(),
            "target_yaw": float(target_yaw),
            "max_steps": max_steps,
            "goal_tolerance": goal_tolerance,
            "yaw_tolerance": yaw_tolerance,
        }
        assert self._client is not None
        return self._client.plan_trajectory(payload).result(timeout=60)

    def close(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
