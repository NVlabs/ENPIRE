"""Grasp slip detection tool based on gripper opening after close."""

from __future__ import annotations

import time
from typing import Any

import portal

from enpire.env.forge.cap.config import CAP_SERVER_PORT
from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult


class DetectGraspSlipTool(Tool):
    """Poll gripper width and report whether the grasp likely slipped.

    For these grippers, a fully closed / empty grasp reports a very small width.
    If the minimum sampled width is <= ``min_width_m``, treat it as slipped.
    """

    name = "detect_grasp_slip"
    description = (
        "Detect whether a grasp slipped by polling the selected gripper width. "
        "Returns {'slipped': bool, 'tight': bool, 'samples_m': [...], ...}."
    )
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "min_width_m",
            "float",
            "Minimum gripper opening considered a held object; <= this means slipped.",
            required=False,
            default=0.002,
        ),
        ToolParameter(
            "poll_secs",
            "float",
            "Total polling duration in seconds.",
            required=False,
            default=0.35,
        ),
        ToolParameter(
            "poll_steps",
            "int",
            "Number of gripper samples to collect.",
            required=False,
            default=4,
        ),
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
        side = str(kwargs["side"])
        if side not in ("left", "right"):
            return ToolResult(success=False, error=f"invalid side: {side}")

        min_width_m = float(kwargs.get("min_width_m", 0.002))
        poll_secs = max(0.0, float(kwargs.get("poll_secs", 0.35)))
        poll_steps = max(1, int(kwargs.get("poll_steps", 4)))
        sleep_s = poll_secs / poll_steps if poll_steps > 0 else 0.0

        try:
            client = self._get_client()
            samples: list[float] = []
            key = f"{side}_gripper_pos"
            for _ in range(poll_steps):
                state = client.get_state().result()
                raw = state[key]
                try:
                    width = float(raw[0])
                except Exception:
                    width = float(raw)
                samples.append(width)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            min_width = min(samples)
            slipped = min_width <= min_width_m
            data = {
                "side": side,
                "slipped": bool(slipped),
                "tight": bool(not slipped),
                "samples_m": samples,
                "min_width_m": float(min_width),
                "threshold_m": float(min_width_m),
            }
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
