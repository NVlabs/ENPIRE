# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Follow a 3D Cartesian waypoint trajectory via mink IK + cubic spline.

Thin CAP tool wrapper around experimental.curobo_waypoint_planner.
Plans offline: seeded IK for waypoints, cubic spline for C2-smooth interpolation.
Executes via move_joint_keypoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import portal

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT

_planner = None


def _get_planner():
    global _planner
    if _planner is None:
        from enpire.env.forge.experimental.curobo_waypoint_planner import CuroboWaypointPlanner
        _planner = CuroboWaypointPlanner(device="cuda:0", solver_speed="fast")
    return _planner


class FollowTrajTool(Tool):
    name = "follow_traj"
    description = (
        "Plan and execute a Cartesian waypoint trajectory. "
        "Provide xyz waypoints and start/end display RPY orientations. "
        "Plans a C2-smooth, collision-free joint trajectory offline, then executes it. "
        "Use max_joint_vel to control speed (0.05-3.0 rad/s)."
    )
    parameters = [
        ToolParameter("waypoints_xyz", "list", "List of [x,y,z] waypoints in metres."),
        ToolParameter("start_rpy", "list[float]", "Start display RPY [roll, pitch, yaw] in degrees."),
        ToolParameter("end_rpy", "list[float]", "End display RPY [roll, pitch, yaw] in degrees."),
        ToolParameter("side", "str", "Arm: 'left' or 'right'.", default="right"),
        ToolParameter("arc_spacing", "float", "Arc-length spacing in metres.", required=False, default=0.005),
        ToolParameter("max_joint_vel", "float", "Max joint velocity in rad/s (0.05-3.0).", required=False, default=1.5),
        ToolParameter("subsample", "int", "Plan every Nth waypoint as IK target.", required=False, default=10),
        ToolParameter("preview_only", "str", "If 'true', plan but don't execute.", required=False, default="false"),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port
        self._client: portal.Client | None = None

    def _get_client(self) -> portal.Client:
        if self._client is None:
            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    def execute(self, **kwargs: Any) -> ToolResult:
        waypoints = np.asarray(kwargs["waypoints_xyz"], dtype=np.float64)
        start_rpy = list(kwargs["start_rpy"])
        end_rpy = list(kwargs["end_rpy"])
        side = str(kwargs.get("side", "right")).lower()
        arc_spacing = float(kwargs.get("arc_spacing", 0.005))
        max_joint_vel = float(kwargs.get("max_joint_vel", 1.5))
        subsample = int(kwargs.get("subsample", 10))
        preview_only = str(kwargs.get("preview_only", "false")).lower() in ("true", "1", "yes")

        try:
            client = self._get_client()
            state = client.get_state().result()
            cur_left = np.asarray(state["left_joint_pos"], dtype=np.float64)
            cur_right = np.asarray(state["right_joint_pos"], dtype=np.float64)

            planner = _get_planner()
            result = planner.plan_waypoint_trajectory(
                waypoints_xyz=waypoints,
                start_rpy_deg=start_rpy,
                end_rpy_deg=end_rpy,
                side=side,
                current_left_jp=cur_left,
                current_right_jp=cur_right,
                arc_spacing=arc_spacing,
                max_joint_vel=max_joint_vel,
                subsample=subsample,
            )

            if not result["success"]:
                return ToolResult(
                    success=False,
                    data=result,
                    error=f"Planning failed: {result['n_failed_ik']} IK failures, jumps at {result['jump_indices']}",
                )

            if preview_only:
                return ToolResult(success=True, data=result)

            exec_result = client.move_joint_keypoints(
                side,
                result["timestamps"].astype(np.float64),
                result["joints"].astype(np.float64),
            ).result()

            if not exec_result.get("success", False):
                return ToolResult(
                    success=False,
                    data=result,
                    error=exec_result.get("reason", "Execution failed"),
                )

            return ToolResult(success=True, data=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))
