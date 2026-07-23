# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grasp tool — approach, grasp, and lift an object in one call.

Encapsulates the proven grasp patterns from oracle scripts:
  1. Open gripper
  2. Approach from above (freespace_move to hover position)
  3. Descend to grasp height (freespace_move or nudge)
  4. Close gripper
  5. Lift to hover height

Uses RPY orientation throughout. Supports horizontal approach (default)
and top-down approach for objects near the arm base.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT

logger = logging.getLogger(__name__)

# Table surface height — freespace_move targets must stay above this
_TABLE_Z = 0.75


class GraspTool(Tool):
    """Approach an object, close gripper, and lift — all in one call."""

    name = "grasp"
    description = (
        "Grasp an object at a given position. Opens gripper, approaches from above, "
        "descends to grasp height, closes gripper, and lifts. Blocking. "
        "Provide the object position from detect_object(). RPY is optional — "
        "if omitted, uses the current arm orientation. "
        "pre_height is how far above the object to hover (default 0.10 m)."
    )
    parameters = [
        ToolParameter("side", "str", '"left" or "right" — which arm to use.'),
        ToolParameter("position", "list[float]", "Object position [x, y, z] in metres (world frame)."),
        ToolParameter(
            "rpy", "list[float]",
            "Grasp orientation [roll, pitch, yaw] in degrees. "
            "If omitted, uses current arm RPY.",
            required=False, default=None,
        ),
        ToolParameter(
            "pre_height", "float",
            "Hover height above object before descending (metres, default 0.10).",
            required=False, default="0.10",
        ),
        ToolParameter(
            "z_offset", "float",
            "Extra Z offset added to grasp position for safety (metres, default 0.05). "
            "Increase for tall objects, decrease for flat objects.",
            required=False, default="0.05",
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
    ):
        self._host = host
        self._port = port
        self._portal_client = None
        self._freespace_tool = None
        self._nudge_tool = None

    def _get_portal(self):
        if self._portal_client is None:
            import portal
            self._portal_client = portal.Client(f"{self._host}:{self._port}")
        return self._portal_client

    def _get_freespace_tool(self):
        if self._freespace_tool is None:
            from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool
            self._freespace_tool = FreespaceMoveTool(host=self._host, port=self._port)
        return self._freespace_tool

    def _get_nudge_tool(self):
        if self._nudge_tool is None:
            from enpire.env.forge.cap.agent.tools.nudge import NudgeTool
            self._nudge_tool = NudgeTool(host=self._host, port=self._port)
        return self._nudge_tool

    def execute(self, **kwargs: Any) -> ToolResult:
        side: str = kwargs["side"]
        position: list[float] = list(kwargs["position"])
        rpy = kwargs.get("rpy")
        pre_height: float = float(kwargs.get("pre_height", 0.10))
        z_offset: float = float(kwargs.get("z_offset", 0.05))

        if side not in ("left", "right"):
            return ToolResult(success=False, error=f"Invalid side: {side}. Use 'left' or 'right'.")

        if len(position) != 3:
            return ToolResult(success=False, error="position must be [x, y, z].")

        # Get current state for fallback RPY
        if rpy is None:
            try:
                client = self._get_portal()
                state = client.get_robot_state().result()
                rpy = list(state[f"{side}_ee_rpy"])
                # Convert radians to degrees if needed (state returns radians)
                if all(abs(v) < 2 * math.pi for v in rpy):
                    rpy = [math.degrees(v) for v in rpy]
            except Exception as e:
                return ToolResult(success=False, error=f"Cannot read robot state: {e}")

        freespace = self._get_freespace_tool()
        nudge = self._get_nudge_tool()

        pos_key = f"{side}_target_pos"
        rpy_key = f"{side}_target_rpy"

        grasp_z = position[2] + z_offset
        hover_z = grasp_z + pre_height

        # Clamp hover above table
        if hover_z <= _TABLE_Z + 0.02:
            hover_z = _TABLE_Z + 0.10

        try:
            # 1. Open gripper
            client = self._get_portal()
            client.open_gripper(side).result()
            logger.info("[grasp] Gripper opened (%s)", side)

            # 2. Move to hover position above object
            hover_pos = [position[0], position[1], hover_z]
            result = freespace.execute(**{pos_key: hover_pos, rpy_key: rpy})
            if not result.success:
                return ToolResult(
                    success=False,
                    error=f"Failed to reach hover position: {result.error}",
                )
            logger.info("[grasp] At hover position z=%.3f", hover_z)

            # 3. Descend to grasp height
            if grasp_z > _TABLE_Z + 0.02:
                # Safe to use freespace_move
                grasp_pos = [position[0], position[1], grasp_z]
                result = freespace.execute(**{pos_key: grasp_pos, rpy_key: rpy})
                if not result.success:
                    return ToolResult(
                        success=False,
                        error=f"Failed to descend to grasp: {result.error}",
                    )
            else:
                # Too close to table — use nudge for final descent
                dz = grasp_z - hover_z
                result = nudge.execute(side=side, delta_pos=[0, 0, dz])
                if not result.success:
                    return ToolResult(
                        success=False,
                        error=f"Nudge descent failed: {result.error}",
                    )
            logger.info("[grasp] At grasp height z=%.3f", grasp_z)

            # 4. Close gripper
            client.close_gripper(side).result()
            logger.info("[grasp] Gripper closed (%s)", side)

            # 5. Lift back to hover
            result = freespace.execute(**{pos_key: hover_pos, rpy_key: rpy})
            if not result.success:
                # Try nudge up as fallback
                nudge.execute(side=side, delta_pos=[0, 0, pre_height])
            logger.info("[grasp] Lifted to z=%.3f", hover_z)

            return ToolResult(
                success=True,
                data={
                    "side": side,
                    "grasp_position": [position[0], position[1], grasp_z],
                    "grasp_rpy": rpy,
                    "hover_z": hover_z,
                },
            )

        except Exception as e:
            logger.error("[grasp] Failed: %s", e)
            return ToolResult(success=False, error=str(e))


class PlaceTool(Tool):
    """Move to a position and release the object."""

    name = "place"
    description = (
        "Place a held object at a target position. Moves to hover position above "
        "the target, descends, opens gripper, and lifts away. Blocking."
    )
    parameters = [
        ToolParameter("side", "str", '"left" or "right" — which arm is holding the object.'),
        ToolParameter("position", "list[float]", "Target position [x, y, z] in metres (world frame)."),
        ToolParameter(
            "rpy", "list[float]",
            "Orientation [roll, pitch, yaw] in degrees. If omitted, keeps current arm RPY.",
            required=False, default=None,
        ),
        ToolParameter(
            "pre_height", "float",
            "Hover height above target (metres, default 0.15).",
            required=False, default="0.15",
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
    ):
        self._host = host
        self._port = port
        self._portal_client = None
        self._freespace_tool = None

    def _get_portal(self):
        if self._portal_client is None:
            import portal
            self._portal_client = portal.Client(f"{self._host}:{self._port}")
        return self._portal_client

    def _get_freespace_tool(self):
        if self._freespace_tool is None:
            from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool
            self._freespace_tool = FreespaceMoveTool(host=self._host, port=self._port)
        return self._freespace_tool

    def execute(self, **kwargs: Any) -> ToolResult:
        side: str = kwargs["side"]
        position: list[float] = list(kwargs["position"])
        rpy = kwargs.get("rpy")
        pre_height: float = float(kwargs.get("pre_height", 0.15))

        if side not in ("left", "right"):
            return ToolResult(success=False, error=f"Invalid side: {side}.")

        if len(position) != 3:
            return ToolResult(success=False, error="position must be [x, y, z].")

        # Get current RPY if not specified
        if rpy is None:
            try:
                client = self._get_portal()
                state = client.get_robot_state().result()
                rpy = list(state[f"{side}_ee_rpy"])
                if all(abs(v) < 2 * math.pi for v in rpy):
                    rpy = [math.degrees(v) for v in rpy]
            except Exception as e:
                return ToolResult(success=False, error=f"Cannot read robot state: {e}")

        freespace = self._get_freespace_tool()
        pos_key = f"{side}_target_pos"
        rpy_key = f"{side}_target_rpy"

        hover_z = position[2] + pre_height
        if hover_z <= _TABLE_Z + 0.02:
            hover_z = _TABLE_Z + 0.10

        try:
            # 1. Move to hover above target
            hover_pos = [position[0], position[1], hover_z]
            result = freespace.execute(**{pos_key: hover_pos, rpy_key: rpy})
            if not result.success:
                return ToolResult(success=False, error=f"Failed to reach place hover: {result.error}")

            # 2. Descend to place height
            place_pos = [position[0], position[1], position[2] + 0.05]
            result = freespace.execute(**{pos_key: place_pos, rpy_key: rpy})
            if not result.success:
                return ToolResult(success=False, error=f"Failed to descend for place: {result.error}")

            # 3. Open gripper
            client = self._get_portal()
            client.open_gripper(side).result()
            logger.info("[place] Gripper opened (%s)", side)

            # 4. Lift away
            result = freespace.execute(**{pos_key: hover_pos, rpy_key: rpy})
            if not result.success:
                from enpire.env.forge.cap.agent.tools.nudge import NudgeTool
                NudgeTool(host=self._host, port=self._port).execute(
                    side=side, delta_pos=[0, 0, pre_height],
                )

            return ToolResult(
                success=True,
                data={"side": side, "place_position": place_pos, "rpy": rpy},
            )

        except Exception as e:
            logger.error("[place] Failed: %s", e)
            return ToolResult(success=False, error=str(e))
