# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-joint rotation tool with clipping and smooth keypoint trajectory. 
Unlike IK related motion generatin, this is directly using joint control and thus will not fail. 
Mathematical trajectory construction
------------------------------------
The tool rotates exactly one joint while holding the other five joints fixed.
Let ``q0`` be the current joint angle and let the requested signed delta be
``dq_req`` in radians.  The unclipped target is

    q_target = q0 + dq_req.

If clipping is enabled, the commanded target is projected into the configured
joint interval ``[q_min, q_max]``:

    q1 = clip(q_target, q_min, q_max).

The actually executable rotation is therefore

    dq = q1 - q0.

To avoid an abrupt velocity discontinuity at the start and end of the rotation,
the tool samples a cubic smoothstep interpolation.  For normalized time
``u = t / T`` in ``[0, 1]``, the blend is

    s(u) = 3u^2 - 2u^3.

This polynomial satisfies ``s(0)=0``, ``s(1)=1``, and has zero endpoint
velocity because ``s'(u)=6u(1-u)``, so ``s'(0)=s'(1)=0``.  Each keypoint is
then

    q_i = q0 + s(u_i) * dq.

All non-selected joints are copied from the current joint vector for every
keypoint.  The duration is chosen from the clipped rotation magnitude and the
requested speed,

    T = max(|dq_deg| / speed_deg_s, min_duration_s),

and the number of keypoints is approximately ``|dq_deg| / keypoint_spacing_deg``.
The resulting 6-DOF joint keypoints are sent to ``move_joint_keypoints``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import portal

from enpire.env.forge.cap.config import CAP_SERVER_PORT, JOINT_LIMITS_HIGH, JOINT_LIMITS_LOW
from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult

# 1-based per-joint (lo, hi) limits for a single arm. Left/right arms share
# limits in cap.config, so we take the left arm slice.
JOINT_LIMITS: dict[int, tuple[float, float]] = {
    i + 1: (float(JOINT_LIMITS_LOW[i]), float(JOINT_LIMITS_HIGH[i])) for i in range(6)
}


class RotateJointTool(Tool):
    name = "rotate_joint"
    description = (
        "Rotate one arm joint by a signed delta in degrees. Reads current joints, "
        "clips to limits, generates the same smooth S-curve keypoint trajectory "
        "used by scripts, executes move_joint_keypoints, and returns detailed "
        "current/target/clipped/achieved metadata. Joint is 1-based (1..6)."
    )
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter("joint", "int", "1-based arm joint index, 1..6."),
        ToolParameter("delta_deg", "float", "Signed rotation delta in degrees."),
        ToolParameter("speed_deg_s", "float", "Rotation speed in deg/s.", required=False, default=45.0),
        ToolParameter("limit_rad", "float", "Optional symmetric +/- joint limit override in radians.", required=False, default=None),
        ToolParameter("lower_limit_rad", "float", "Optional lower joint limit override in radians.", required=False, default=None),
        ToolParameter("upper_limit_rad", "float", "Optional upper joint limit override in radians.", required=False, default=None),
        ToolParameter("keypoint_spacing_deg", "float", "Approx spacing between keypoints in degrees.", required=False, default=1.0),
        ToolParameter("settle_s", "float", "Sleep after execution, seconds.", required=False, default=0.25),
        ToolParameter("min_duration_s", "float", "Minimum trajectory duration, seconds.", required=False, default=0.25),
        ToolParameter("min_delta_deg", "float", "Skip motion if clipped delta magnitude is below this.", required=False, default=1.0),
        ToolParameter("clip", "bool", "Clip target to joint limits.", required=False, default=True),
        ToolParameter("clip_tolerance_rad", "float", "Tolerance for reporting hit_limit after clipping.", required=False, default=1e-4),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port
        self._client: portal.Client | None = None

    def _get_client(self) -> portal.Client:
        if self._client is None:
            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    @staticmethod
    def _limits(joint: int, kwargs: dict[str, Any]) -> tuple[float, float]:
        if kwargs.get("limit_rad") is not None:
            lim = abs(float(kwargs["limit_rad"]))
            return -lim, lim
        lo = kwargs.get("lower_limit_rad")
        hi = kwargs.get("upper_limit_rad")
        if lo is not None or hi is not None:
            default_lo, default_hi = JOINT_LIMITS[joint]
            return float(default_lo if lo is None else lo), float(default_hi if hi is None else hi)
        return JOINT_LIMITS[joint]

    def execute(self, **kwargs: Any) -> ToolResult:
        side = str(kwargs["side"])
        if side not in ("left", "right"):
            return ToolResult(success=False, error=f"invalid side: {side}")
        joint = int(kwargs["joint"])
        if joint not in JOINT_LIMITS:
            return ToolResult(success=False, error=f"joint must be 1..6, got {joint}")

        idx = joint - 1
        delta_deg = float(kwargs["delta_deg"])
        speed = max(1e-6, abs(float(kwargs.get("speed_deg_s", 45.0))))
        spacing = max(1e-6, abs(float(kwargs.get("keypoint_spacing_deg", 1.0))))
        settle_s = max(0.0, float(kwargs.get("settle_s", 0.25)))
        min_duration_s = max(0.0, float(kwargs.get("min_duration_s", 0.25)))
        min_delta_deg = max(0.0, float(kwargs.get("min_delta_deg", 1.0)))
        do_clip = bool(kwargs.get("clip", True))
        clip_tol = max(0.0, float(kwargs.get("clip_tolerance_rad", 1e-4)))
        lo, hi = self._limits(joint, kwargs)

        try:
            client = self._get_client()
            state = client.get_state().result()
            key = f"{side}_joint_pos"
            cur_joints = list(np.asarray(state[key], dtype=np.float64).ravel()[:6])
            cur = float(cur_joints[idx])
            target = cur + np.radians(delta_deg)
            clipped = float(np.clip(target, lo, hi)) if do_clip else float(target)
            actual_delta_deg = float(np.degrees(clipped - cur))
            hit_limit = bool(abs(target - clipped) > clip_tol) if do_clip else False

            data = {
                "side": side,
                "joint": joint,
                "joint_index": idx,
                "requested_delta_deg": delta_deg,
                "current_deg": float(np.degrees(cur)),
                "target_deg": float(np.degrees(target)),
                "clipped_target_deg": float(np.degrees(clipped)),
                "actual_delta_deg": actual_delta_deg,
                "hit_limit": hit_limit,
                "lower_limit_deg": float(np.degrees(lo)),
                "upper_limit_deg": float(np.degrees(hi)),
                "skipped": False,
                "achieved_delta_deg": 0.0,
            }

            if abs(actual_delta_deg) < min_delta_deg:
                data["skipped"] = True
                data["achieved_delta_deg"] = 0.0
                return ToolResult(success=True, data=data)

            n_steps = max(int(abs(actual_delta_deg) / spacing) + 1, 3)
            duration = max(abs(actual_delta_deg) / speed, min_duration_s)
            t_norm = np.linspace(0.0, 1.0, n_steps)
            s_curve = 3.0 * t_norm**2 - 2.0 * t_norm**3
            timestamps = t_norm * duration
            positions = np.zeros((n_steps, 6), dtype=np.float64)
            for i, a in enumerate(s_curve):
                positions[i] = cur_joints.copy()
                positions[i, idx] = cur + float(a) * (clipped - cur)

            result = client.move_joint_keypoints(side, timestamps, positions).result()
            if not bool(result.get("success", False)):
                return ToolResult(success=False, data=data, error=result.get("reason", "move_joint_keypoints failed"))
            if settle_s > 0:
                time.sleep(settle_s)

            state2 = client.get_state().result()
            after = float(np.asarray(state2[key], dtype=np.float64).ravel()[idx])
            data["achieved_delta_deg"] = float(np.degrees(after - cur))
            data["after_deg"] = float(np.degrees(after))
            data["duration_s"] = float(duration)
            data["n_steps"] = int(n_steps)
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
