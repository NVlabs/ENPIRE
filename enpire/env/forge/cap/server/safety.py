# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety module for the CAP server — e-stop and task-aware EE safety zones.

The safety zone is defined per-arm as a set of keyposes (7D: pos + quat_xyzw).
The safe exploration region is the convex hull of keypose positions expanded by
a configurable margin.  An elastic boundary attenuates actions smoothly before
a hard cutoff.  Orientation is constrained relative to the nearest keypose.

Typical usage from LLM-generated code::

    # Before RL training — LLM inspects task and sets zones
    set_safety_zone("left", [
        [0.35, 0.10, 0.25, 0.513, -0.503, 0.490, -0.493],  # pre-insert hover
        [0.30, 0.05, 0.10, 0.513, -0.503, 0.490, -0.493],  # insertion target
    ], pos_margin=0.08, ori_margin=0.3)

    learn_skill_rl("usb_insertion", {"max_steps": 500})
    clear_safety_zone()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point *p* to the line segment [a, b]."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _dist_point_to_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> float:
    """Euclidean distance from point *p* to the triangle (a, b, c)."""
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    n_len = np.linalg.norm(n)
    if n_len < 1e-10:
        # Degenerate (collinear) — fall back to edges
        return min(
            _dist_point_to_segment(p, a, b),
            _dist_point_to_segment(p, b, c),
            _dist_point_to_segment(p, a, c),
        )
    n = n / n_len
    dist_to_plane = np.dot(p - a, n)
    proj = p - dist_to_plane * n

    # Barycentric coordinates of the projection onto the plane
    v0, v1, v2 = ac, ab, proj - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    inv = 1.0 / (d00 * d11 - d01 * d01 + 1e-12)
    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv

    if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
        return abs(dist_to_plane)

    # Outside triangle — nearest edge
    return min(
        _dist_point_to_segment(p, a, b),
        _dist_point_to_segment(p, b, c),
        _dist_point_to_segment(p, a, c),
    )


def _quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance (radians) between two unit quaternions (xyzw)."""
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return float(2.0 * np.arccos(dot))


# ---------------------------------------------------------------------------
# Per-arm safety zone
# ---------------------------------------------------------------------------

@dataclass
class ArmSafetyZone:
    """Safety zone for one arm's end-effector.

    Parameters
    ----------
    keyposes : list[np.ndarray]
        Each entry is a 7-D array ``[x, y, z, qx, qy, qz, qw]``.
    pos_margin : float
        Radius (metres) around the convex hull of keypose positions that is
        considered safe.
    ori_margin : float
        Maximum angular deviation (radians) from the nearest keypose
        orientation before enforcement kicks in.
    elastic_band : float
        Width (metres) of the elastic attenuation zone beyond *pos_margin*.
        Inside this band, actions are progressively attenuated.  Beyond it
        the hard clamp fires.
    elastic_band_ori : float
        Same concept for orientation (radians).
    """

    keyposes: list[np.ndarray]
    pos_margin: float = 0.08
    ori_margin: float = 0.3
    elastic_band: float = 0.04
    elastic_band_ori: float = 0.15

    def __post_init__(self) -> None:
        self._positions = np.array([np.asarray(kp[:3], dtype=np.float64) for kp in self.keyposes])
        self._orientations = np.array([np.asarray(kp[3:7], dtype=np.float64) for kp in self.keyposes])
        # Normalise quaternions
        for i in range(len(self._orientations)):
            n = np.linalg.norm(self._orientations[i])
            if n > 1e-8:
                self._orientations[i] /= n

        # Pre-compute convex hull / Delaunay for 4+ non-degenerate points
        self._delaunay = None
        self._hull = None
        if len(self._positions) >= 4:
            try:
                from scipy.spatial import ConvexHull, Delaunay

                self._delaunay = Delaunay(self._positions)
                self._hull = ConvexHull(self._positions)
            except Exception:
                pass  # degenerate — handled by fallback paths

    # ----- distance to convex hull of keypose positions --------------------

    def distance_to_hull(self, pos: np.ndarray) -> float:
        """Distance from *pos* to the convex hull of keypose positions.

        Returns 0.0 when *pos* is inside (or on) the hull.
        """
        pos = np.asarray(pos, dtype=np.float64).ravel()[:3]
        n = len(self._positions)

        if n == 1:
            return float(np.linalg.norm(pos - self._positions[0]))

        if n == 2:
            return _dist_point_to_segment(pos, self._positions[0], self._positions[1])

        if n == 3:
            return _dist_point_to_triangle(pos, self._positions[0], self._positions[1], self._positions[2])

        # n >= 4: check if inside via Delaunay
        if self._delaunay is not None and self._delaunay.find_simplex(pos) >= 0:
            return 0.0

        # Outside hull (or degenerate): distance to nearest hull face
        if self._hull is not None:
            min_d = float("inf")
            for simplex in self._hull.simplices:
                face = self._positions[simplex]
                min_d = min(min_d, _dist_point_to_triangle(pos, face[0], face[1], face[2]))
            return min_d

        # Ultimate fallback: min distance to all pairwise edges
        min_d = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                min_d = min(min_d, _dist_point_to_segment(pos, self._positions[i], self._positions[j]))
        return min_d

    def signed_distance(self, pos: np.ndarray) -> float:
        """Signed distance to safe-zone boundary.  Negative ⇒ inside safe zone."""
        return self.distance_to_hull(pos) - self.pos_margin

    # ----- orientation constraint ------------------------------------------

    def nearest_keypose_orientation(self, pos: np.ndarray) -> np.ndarray:
        """Orientation (xyzw) of the keypose closest to *pos*."""
        pos = np.asarray(pos, dtype=np.float64).ravel()[:3]
        dists = np.linalg.norm(self._positions - pos, axis=1)
        return self._orientations[np.argmin(dists)].copy()

    def orientation_violation(self, pos: np.ndarray, quat: np.ndarray) -> float:
        """Angular distance beyond *ori_margin*.  Negative ⇒ within tolerance."""
        ref = self.nearest_keypose_orientation(pos)
        quat = np.asarray(quat, dtype=np.float64).ravel()[:4]
        return _quat_angular_distance(quat, ref) - self.ori_margin

    # ----- enforcement -----------------------------------------------------

    def compute_scale_factor(
        self, pos: np.ndarray, quat: np.ndarray,
    ) -> tuple[float, dict]:
        """Compute the enforcement scale factor for a proposed EE pose.

        Returns
        -------
        factor : float
            1.0 = fully allowed, 0.0 = hard clamp (hold position),
            intermediate = elastic attenuation.
        info : dict
            Diagnostic data (signed distances, which zone triggered, etc.).
        """
        pos_sd = self.signed_distance(pos)
        ori_viol = self.orientation_violation(pos, quat)

        info: dict = {
            "pos_signed_dist": float(pos_sd),
            "ori_violation": float(ori_viol),
            "pos_in_elastic": False,
            "ori_in_elastic": False,
            "pos_hard_clamp": False,
            "ori_hard_clamp": False,
        }

        # Both inside safe zone — no enforcement needed
        if pos_sd <= 0.0 and ori_viol <= 0.0:
            return 1.0, info

        # Position factor
        pos_factor = 1.0
        if pos_sd > 0.0:
            if self.elastic_band > 0.0 and pos_sd < self.elastic_band:
                pos_factor = 1.0 - (pos_sd / self.elastic_band)
                info["pos_in_elastic"] = True
            else:
                pos_factor = 0.0
                info["pos_hard_clamp"] = True

        # Orientation factor
        ori_factor = 1.0
        if ori_viol > 0.0:
            if self.elastic_band_ori > 0.0 and ori_viol < self.elastic_band_ori:
                ori_factor = 1.0 - (ori_viol / self.elastic_band_ori)
                info["ori_in_elastic"] = True
            else:
                ori_factor = 0.0
                info["ori_hard_clamp"] = True

        return min(pos_factor, ori_factor), info


# ---------------------------------------------------------------------------
# Task-level container
# ---------------------------------------------------------------------------

@dataclass
class TaskSafetyZone:
    """Per-arm safety zones for the current task."""

    left: ArmSafetyZone | None = None
    right: ArmSafetyZone | None = None


# ---------------------------------------------------------------------------
# Thread-safe checker (singleton owned by CapServer)
# ---------------------------------------------------------------------------

class SafetyChecker:
    """Thread-safe e-stop + task-aware EE safety zone enforcement."""

    def __init__(self) -> None:
        self._estop = False
        self._lock = threading.Lock()
        self._task_zone: TaskSafetyZone | None = None

    # ---- E-stop -----------------------------------------------------------

    def trigger_estop(self) -> None:
        with self._lock:
            self._estop = True

    def release_estop(self) -> None:
        with self._lock:
            self._estop = False

    def is_estopped(self) -> bool:
        with self._lock:
            return self._estop

    # ---- Task safety zone -------------------------------------------------

    def set_arm_zone(self, side: str, arm_zone: ArmSafetyZone) -> None:
        """Set (or update) the safety zone for one arm.  The other arm is unchanged."""
        with self._lock:
            if self._task_zone is None:
                self._task_zone = TaskSafetyZone()
            if side == "left":
                self._task_zone.left = arm_zone
            elif side == "right":
                self._task_zone.right = arm_zone
            else:
                raise ValueError(f"Unknown side: {side!r}")
            logger.info(
                "[Safety] %s zone set: %d keyposes, pos_margin=%.3fm, ori_margin=%.2frad",
                side, len(arm_zone.keyposes), arm_zone.pos_margin, arm_zone.ori_margin,
            )

    def clear_task_zone(self, side: str | None = None) -> None:
        """Clear safety zone(s).  *side*=None clears both."""
        with self._lock:
            if side is None:
                self._task_zone = None
                logger.info("[Safety] All task zones cleared")
            elif self._task_zone is not None:
                if side == "left":
                    self._task_zone.left = None
                elif side == "right":
                    self._task_zone.right = None
                if self._task_zone.left is None and self._task_zone.right is None:
                    self._task_zone = None
                logger.info("[Safety] %s zone cleared", side)

    def has_task_zone(self) -> bool:
        with self._lock:
            return self._task_zone is not None

    def get_task_zone(self) -> TaskSafetyZone | None:
        with self._lock:
            return self._task_zone

    def get_zone_config(self) -> dict:
        """Return a serialisable snapshot of the current zone config."""
        with self._lock:
            if self._task_zone is None:
                return {"active": False}
            cfg: dict = {"active": True}
            for side in ("left", "right"):
                az: ArmSafetyZone | None = getattr(self._task_zone, side)
                if az is not None:
                    cfg[side] = {
                        "keyposes": [kp.tolist() for kp in az.keyposes],
                        "pos_margin": az.pos_margin,
                        "ori_margin": az.ori_margin,
                        "elastic_band": az.elastic_band,
                        "elastic_band_ori": az.elastic_band_ori,
                    }
            return cfg

    def enforce_ee(
        self,
        side: str,
        proposed_ee_pos: np.ndarray,
        proposed_ee_quat: np.ndarray,
    ) -> tuple[float, dict]:
        """Check a proposed EE pose against the task zone for *side*.

        Returns
        -------
        scale_factor : float
            How much of the proposed action to keep (1.0 = all, 0.0 = none).
        info : dict
            Diagnostic data for the dashboard / logging.
        """
        with self._lock:
            zone = self._task_zone

        if zone is None:
            return 1.0, {}

        arm_zone = zone.left if side == "left" else zone.right
        if arm_zone is None:
            return 1.0, {}

        return arm_zone.compute_scale_factor(proposed_ee_pos, proposed_ee_quat)
