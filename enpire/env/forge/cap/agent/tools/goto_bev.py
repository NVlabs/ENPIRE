# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convenience tool for sending one or both arms to the shared BEV waypoint."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT
from enpire.env.forge.cap.constants.planning import get_bev_pose


class GotoBevTool(Tool):
    name = "goto_bev"
    description = (
        "Move one or both arms to the shared bird's-eye-view (BEV) waypoint defined "
        "in cap/constants/planning.py. This is a thin convenience wrapper around "
        "freespace_move for concise BEV resets."
    )
    parameters = [
        ToolParameter(
            "side",
            "str",
            "Which arm(s) to move: 'left', 'right', or 'both'.",
        ),
        ToolParameter(
            "gripper_width",
            "float",
            "Optional single-arm gripper target width (0=closed, 1=open). "
            "For side='left' or side='right'.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "planning_speed",
            "float",
            "Motion planning speed forwarded to freespace_move.",
            required=False,
            default=1.0,
        ),
        ToolParameter(
            "left_gripper_width",
            "float",
            "Optional left gripper target width (0=closed, 1=open).",
            required=False,
            default=None,
        ),
        ToolParameter(
            "right_gripper_width",
            "float",
            "Optional right gripper target width (0=closed, 1=open).",
            required=False,
            default=None,
        ),
        ToolParameter(
            "planner_backend",
            "str",
            "Motion planner backend forwarded to freespace_move. Defaults to 'curobo'.",
            required=False,
            default="curobo",
        ),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port
        self._freespace_tool = None

    def _get_freespace_tool(self):
        if self._freespace_tool is None:
            from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool

            self._freespace_tool = FreespaceMoveTool(host=self._host, port=self._port)
        return self._freespace_tool

    def execute(self, **kwargs: Any) -> ToolResult:
        side = str(kwargs.get("side", "")).strip().lower()
        if side not in {"left", "right", "both"}:
            return ToolResult(
                success=False,
                error=f"Invalid side {side!r}. Use 'left', 'right', or 'both'.",
            )

        gripper_width = kwargs.get("gripper_width")
        left_gripper_width = kwargs.get("left_gripper_width")
        right_gripper_width = kwargs.get("right_gripper_width")
        if side == "left" and left_gripper_width is None and gripper_width is not None:
            left_gripper_width = gripper_width
        if side == "right" and right_gripper_width is None and gripper_width is not None:
            right_gripper_width = gripper_width

        move_kwargs: dict[str, Any] = {
            "planning_speed": float(kwargs.get("planning_speed", 1.0)),
            "planner_backend": str(kwargs.get("planner_backend", "curobo")),
        }
        if side in {"left", "both"}:
            left_pos, left_rpy = get_bev_pose("left")
            move_kwargs["left_target_pos"] = left_pos
            move_kwargs["left_target_rpy"] = left_rpy
        if side in {"right", "both"}:
            right_pos, right_rpy = get_bev_pose("right")
            move_kwargs["right_target_pos"] = right_pos
            move_kwargs["right_target_rpy"] = right_rpy
        if side == "both" and gripper_width is not None:
            if left_gripper_width is None:
                left_gripper_width = gripper_width
            if right_gripper_width is None:
                right_gripper_width = gripper_width
        if left_gripper_width is not None:
            move_kwargs["left_gripper_target_width"] = float(left_gripper_width)
        if right_gripper_width is not None:
            move_kwargs["right_gripper_target_width"] = float(right_gripper_width)

        result = self._get_freespace_tool().execute(**move_kwargs)
        if not result.success:
            return result

        data = result.data
        if is_dataclass(data):
            payload: dict[str, Any] = asdict(data)
        elif isinstance(data, dict):
            payload = dict(data)
        else:
            payload = {"result": data}
        payload["side"] = side
        payload["bev_targets"] = {}
        if side in {"left", "both"}:
            pos, rpy = get_bev_pose("left")
            payload["bev_targets"]["left"] = {"position": pos, "rpy": rpy}
        if side in {"right", "both"}:
            pos, rpy = get_bev_pose("right")
            payload["bev_targets"]["right"] = {"position": pos, "rpy": rpy}
        if left_gripper_width is not None:
            payload["left_gripper_width"] = float(left_gripper_width)
        if right_gripper_width is not None:
            payload["right_gripper_width"] = float(right_gripper_width)
        return ToolResult(success=True, data=payload)
