"""Safety zone tools — set/clear/query task-aware EE safety zones on cap_server.

The LLM agent uses these before launching RL training to restrict the
end-effector to a task-relevant region.  The safe zone is defined by a set of
keyposes (7-D: position + orientation) and configurable margins.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import portal

from cap.config import CAP_SERVER_PORT
from cap.agent.tools.base import Tool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class _PortalMixin:
    """Shared Portal RPC helpers.

    Uses a fresh ``portal.Client`` per call to avoid the autoconn race
    condition in Portal's ``Future.set_result`` (see docs/SAFETY_ZONE_DESIGN.md).
    The overhead of opening a TCP connection is negligible compared to the RPC
    round-trip itself, and this eliminates the stale-connection / resend race
    that causes ``AssertionError`` with an empty message.
    """

    _host: str
    _port: int

    def _call_rpc(self, method: str, *args: Any) -> Any:
        """Call Portal RPC with a fresh client each time (avoids autoconn race)."""
        client = portal.Client(
            f"{self._host}:{self._port}", autoconn=False,
        )
        client.connect(timeout=5.0)
        try:
            return getattr(client, method)(*args).result(timeout=5.0)
        finally:
            try:
                client.close(timeout=0.5)
            except Exception:
                pass


class SetSafetyZoneTool(_PortalMixin, Tool):
    name = "set_safety_zone"
    description = (
        "Set a task-aware safety zone for one arm's end-effector before RL training. "
        "The safe exploration region is the convex hull of the given keyposes expanded "
        "by pos_margin (metres).  Orientation is constrained to within ori_margin "
        "(radians) of the nearest keypose's orientation.  Actions that would move the "
        "EE outside the safe zone are elastically attenuated, then hard-clamped. "
        "Call once per arm.  Zones persist across learn_skill_rl episodes until cleared."
    )
    parameters = [
        ToolParameter(
            "side", "str",
            "'left' or 'right' — which arm this zone applies to.",
        ),
        ToolParameter(
            "keyposes", "list[list[float]]",
            "List of 7-D keyposes [x, y, z, qx, qy, qz, qw] defining task-relevant "
            "EE poses (e.g. grasp pose, pre-insertion hover, insertion target).  The "
            "safe region is the convex hull of these positions + pos_margin.",
        ),
        ToolParameter(
            "pos_margin", "float",
            "Position margin in metres around the convex hull of keyposes. "
            "Default 0.08 (8 cm).",
            required=False, default=0.08,
        ),
        ToolParameter(
            "ori_margin", "float",
            "Orientation margin in radians — max angular deviation from nearest "
            "keypose orientation.  Default 0.3 (~17 degrees).",
            required=False, default=0.3,
        ),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        side: str = kwargs["side"]
        keyposes: list = kwargs["keyposes"]
        pos_margin: float = kwargs.get("pos_margin", 0.08)
        ori_margin: float = kwargs.get("ori_margin", 0.3)

        if side not in ("left", "right"):
            return ToolResult(success=False, error=f"side must be 'left' or 'right', got {side!r}")
        if not keyposes or not all(len(kp) == 7 for kp in keyposes):
            return ToolResult(success=False, error="keyposes must be a non-empty list of 7-D arrays [x,y,z,qx,qy,qz,qw]")

        try:
            result = self._call_rpc("set_safety_zone", side, keyposes, pos_margin, ori_margin)
            return ToolResult(success=bool(result.get("success", False)), data=result)
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}\n{tb}")


class ClearSafetyZoneTool(_PortalMixin, Tool):
    name = "clear_safety_zone"
    description = (
        "Clear the task safety zone.  Call with no arguments to clear both arms, "
        "or pass a side ('left'/'right') to clear only that arm."
    )
    parameters = [
        ToolParameter(
            "side", "str",
            "'left', 'right', or omit to clear both arms.",
            required=False, default=None,
        ),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        side = kwargs.get("side")
        try:
            # Pass side only if specified — Portal cannot serialize empty strings
            # (zero-length buffer fails SendBuffer assertion).
            if side:
                result = self._call_rpc("clear_safety_zone", side)
            else:
                result = self._call_rpc("clear_safety_zone")
            return ToolResult(success=bool(result.get("success", False)), data=result)
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}\n{tb}")


class GetSafetyZoneTool(_PortalMixin, Tool):
    name = "get_safety_zone"
    description = (
        "Query the current safety zone configuration.  Returns the active zones "
        "with their keyposes and margins, or {active: false} if no zone is set."
    )
    parameters = []

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = self._call_rpc("get_safety_zone")
            return ToolResult(success=True, data=result)
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}\n{tb}")
