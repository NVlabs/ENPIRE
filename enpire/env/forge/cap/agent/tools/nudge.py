# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nudge tool — apply a small delta EE pose to one arm.

Composes a position offset and/or rotation offset with the current
end-effector pose, then moves to the new target via cap_server's
``_ik_servo`` RPC or direct-env freespace planning.  Ideal for fine
adjustments after coarse positioning.

Coordinate frame contract
-------------------------
All deltas are in the **robot world frame**:

    +X  forward  (toward the work table)
    +Y  left     (toward the left arm)
    +Z  up       (sky)

``delta_pos`` is added to the current world-frame EE position.
``delta_rpy`` (extrinsic XYZ) is composed as a world-frame rotation:
``new_rot = delta_rot @ current_rot``.

The current EE pose is read from cap_server (Pinocchio FK, world frame),
and ``_ik_servo`` expects world-frame targets — so the chain is consistent.

Usage from generated code::

    nudge("left", delta_pos=[0, 0, 0.02])           # move 2 cm up (+z)
    nudge("right", delta_rpy=[0, 0, 0.1])            # yaw +0.1 rad
    nudge("left", delta_pos=[0.01, 0, 0], delta_rpy=[0, 0.05, 0])  # both
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from enpire.env.forge.cap.agent.tools.base import NudgeResult, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT, MOVE_EEF_MAX_DURATION_S, MOVE_EEF_MAX_VEL


class NudgeTool(Tool):
    """Apply a small delta movement to an arm's end-effector.

    The delta is applied in the **world frame**: delta_pos is added to the
    current position, and delta_rpy is composed as a world-frame rotation
    (new_rot = delta_rot @ current_rot).

    This is robust for small corrections — IK is reliable for nearby targets.
    """

    name = "nudge"
    description = (
        "Apply a small delta movement to one arm's end-effector. "
        "delta_pos [dx, dy, dz] in metres offsets the position. "
        "delta_rpy [droll, dpitch, dyaw] in degrees offsets the orientation. "
        "Both are in world frame. At least one of delta_pos or delta_rpy must be provided. "
        "Ideal for fine adjustments after freespace_move gets close to a target."
    )
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "delta_pos", "list[float]",
            "Position offset [dx, dy, dz] in metres (world frame). "
            "Defaults to [0, 0, 0] if omitted.",
            required=False,
        ),
        ToolParameter(
            "delta_rpy", "list[float]",
            "Orientation offset [droll, dpitch, dyaw] in degrees (world frame, extrinsic XYZ). "
            "Defaults to [0, 0, 0] if omitted.",
            required=False,
        ),
        ToolParameter(
            "max_duration_sec", "float",
            f"Timeout for the move in seconds (default {MOVE_EEF_MAX_DURATION_S}).",
            required=False, default=MOVE_EEF_MAX_DURATION_S,
        ),
        ToolParameter(
            "max_vel", "float",
            f"Max EE translation speed in m/s (default {MOVE_EEF_MAX_VEL}).",
            required=False, default=MOVE_EEF_MAX_VEL,
        ),
        ToolParameter(
            "preview_only",
            "bool",
            "Plan the nudge without executing it in direct-env mode.",
            required=False,
            default=False,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import portal

            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    def _get_env_obs(self, side: str) -> dict[str, Any]:
        if hasattr(self._env, "get_observations"):
            return self._env.get_observations(side)
        if hasattr(self._env, "get_arm_observation"):
            return self._env.get_arm_observation(side)
        raise AttributeError(
            "env must expose get_observations(side) or get_arm_observation(side)"
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        side: str = kwargs["side"]
        if side not in ("left", "right"):
            return ToolResult(
                success=False,
                error=f"Invalid side '{side}', must be 'left' or 'right'.",
            )

        delta_pos = kwargs.get("delta_pos")
        delta_rpy = kwargs.get("delta_rpy")

        if delta_pos is None and delta_rpy is None:
            return ToolResult(
                success=False,
                error="At least one of delta_pos or delta_rpy must be provided.",
            )

        d_pos = np.asarray(
            delta_pos if delta_pos is not None else [0, 0, 0], dtype=np.float64
        )
        d_rpy = np.asarray(
            delta_rpy if delta_rpy is not None else [0, 0, 0], dtype=np.float64
        )

        try:
            if self._env is not None:
                obs = self._get_env_obs(side)
                cur_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
                cur_quat_xyzw = np.asarray(obs["ee_quat"], dtype=np.float64)
                new_pos = cur_pos + d_pos
                new_quat_xyzw = (
                    Rotation.from_euler("xyz", d_rpy, degrees=True)
                    * Rotation.from_quat(cur_quat_xyzw)
                ).as_quat()

                from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool

                move = FreespaceMoveTool(
                    host=self._host,
                    port=self._port,
                    env=self._env,
                )
                move_kwargs: dict[str, Any] = {
                    f"{side}_target_pos": new_pos.tolist(),
                    f"{side}_target_quat": new_quat_xyzw.tolist(),
                    "planning_speed": float(kwargs.get("max_vel", MOVE_EEF_MAX_VEL)),
                    "preview_only": bool(kwargs.get("preview_only", False)),
                }
                result = move.execute(**move_kwargs)
                success = bool(result.success)
                final_obs = self._get_env_obs(side)
                final_pos = (
                    new_pos
                    if bool(kwargs.get("preview_only", False))
                    else np.asarray(final_obs.get("ee_pos", new_pos), dtype=np.float64)
                )
                final_quat = (
                    new_quat_xyzw
                    if bool(kwargs.get("preview_only", False))
                    else np.asarray(
                        final_obs.get("ee_quat", new_quat_xyzw), dtype=np.float64
                    )
                )
                nudge_result = NudgeResult(
                    success=success,
                    final_pos=[round(float(x), 4) for x in final_pos],
                    final_quat=[round(float(x), 4) for x in final_quat],
                )
                if not success:
                    return ToolResult(
                        success=False,
                        data=nudge_result,
                        error=result.error
                        or getattr(result.data, "reason", "nudge planning failed"),
                    )
                return ToolResult(success=True, data=nudge_result)

            client = self._get_client()
            state = client.get_state().result()

            # Current EE pose
            cur_pos = np.asarray(state[f"{side}_ee_pos"], dtype=np.float64)
            cur_quat_xyzw = np.asarray(state[f"{side}_ee_quat_xyzw"], dtype=np.float64)

            # New position: current + delta (world frame)
            new_pos = cur_pos + d_pos

            # New orientation: delta_rot @ current_rot (world frame rotation)
            cur_rot = Rotation.from_quat(cur_quat_xyzw)  # scipy uses xyzw
            delta_rot = Rotation.from_euler("xyz", d_rpy, degrees=True)
            new_rot = delta_rot * cur_rot
            new_quat_xyzw = new_rot.as_quat()  # scipy returns xyzw

            # Execute move
            max_duration = float(kwargs.get("max_duration_sec", MOVE_EEF_MAX_DURATION_S))
            max_vel = float(kwargs.get("max_vel", MOVE_EEF_MAX_VEL))

            result = client._ik_servo(
                side, new_pos, new_quat_xyzw, None, max_duration, max_vel,
            ).result()

            success = bool(result.get("success", False))
            reason = result.get("reason", "")

            nudge_result = NudgeResult(
                success=success,
                final_pos=[round(float(x), 4) for x in new_pos],
                final_quat=[round(float(x), 4) for x in new_quat_xyzw],
            )

            if not success:
                return ToolResult(success=False, data=nudge_result, error=reason)
            return ToolResult(success=True, data=nudge_result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))
