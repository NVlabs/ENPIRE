# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand policy-output tools for external model servers.

These tools do not launch or own the model server. They assume the relevant
policy endpoint is already running and warm (for example in a separate tmux
session) and only borrow robot control when the CAP code explicitly steps the
policy or uses the one-shot wrapper.
"""

from __future__ import annotations

from typing import Any

import portal

from enpire.env.forge.cap.agent.tools.base import SkillResult, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import (
    CAP_SERVER_PORT,
    POLICY_MODEL_CONFIGS,
    PORTAL_EMPTY_SENTINEL as _PORTAL_EMPTY,
)


class _PolicyOutputToolBase(Tool):
    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
    ):
        self._host = host
        self._port = port
        self._client: portal.Client | None = None

    def _get_client(self) -> portal.Client:
        if self._client is None:
            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    @staticmethod
    def _skill_result_from_rpc(
        result: dict[str, Any],
        fallback_error: str,
    ) -> ToolResult:
        skill_result = SkillResult(
            success=bool(result.get("success", False)),
            steps_executed=int(result.get("steps_executed", 0)),
            info=result.get("info", {}),
        )
        if not skill_result.success:
            return ToolResult(
                success=False,
                error=str(result.get("reason", fallback_error)),
            )
        return ToolResult(success=True, data=skill_result)


_MODEL_DESCRIPTION = "Policy model name. Supported: " + ", ".join(
    sorted(POLICY_MODEL_CONFIGS)
)


class StartPolicyOutputTool(_PolicyOutputToolBase):
    name = "start_policy_output"
    description = (
        "Prepare a warm external policy model for later stepping. "
        "This creates a CAP-side policy session and reset boundary but does not "
        "send observations or execute robot actions yet."
    )
    parameters = [
        ToolParameter("model", "str", _MODEL_DESCRIPTION),
        ToolParameter(
            "replan_horizon",
            "int",
            "How many actions CAP should execute by default each time step_policy_output() is called.",
            required=False,
            default=30,
        ),
        ToolParameter(
            "task_description",
            "str",
            "Optional language command to pair with future observations. Uses a model-specific default when omitted.",
            required=False,
            default="",
        ),
        ToolParameter(
            "policy_server",
            "str",
            "Optional host:port override for the already-running policy server.",
            required=False,
            default="",
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        model = str(kwargs["model"]).strip()
        replan_horizon = max(1, int(kwargs.get("replan_horizon", 30)))
        task_description = (
            str(kwargs.get("task_description", "")).strip() or _PORTAL_EMPTY
        )
        policy_server = str(kwargs.get("policy_server", "")).strip() or _PORTAL_EMPTY

        try:
            result = (
                self._get_client()
                .start_policy_output(
                    model,
                    replan_horizon,
                    task_description,
                    policy_server,
                )
                .result()
            )
            return self._skill_result_from_rpc(result, "policy output start failed")
        except Exception as e:
            return ToolResult(success=False, error=str(e) or repr(e))


class StepPolicyOutputTool(_PolicyOutputToolBase):
    name = "step_policy_output"
    description = (
        "Execute a bounded burst from the active policy-output session. "
        "If other robot commands ran since the last policy step, CAP clears the "
        "local queued actions and future reset hook before sampling again."
    )
    parameters = [
        ToolParameter(
            "max_steps",
            "int",
            "Optional number of policy-control steps to execute. Defaults to the session replan_horizon.",
            required=False,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        max_steps_raw = kwargs.get("max_steps")
        max_steps = 0 if max_steps_raw in (None, "") else max(1, int(max_steps_raw))

        try:
            result = self._get_client().step_policy_output(max_steps).result()
            return self._skill_result_from_rpc(result, "policy output step failed")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class StopPolicyOutputTool(_PolicyOutputToolBase):
    name = "stop_policy_output"
    description = (
        "Stop the active policy-output session and clear CAP-side queued policy state. "
        "Safe to call even when no policy-output session is active."
    )
    parameters: list[ToolParameter] = []

    def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        try:
            result = self._get_client().stop_policy_output().result()
            return self._skill_result_from_rpc(result, "policy output stop failed")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class UsePolicyOutputTool(_PolicyOutputToolBase):
    name = "use_policy_output"
    description = (
        "One-shot convenience wrapper around start_policy_output(), "
        "step_policy_output(), and stop_policy_output(). "
        "The model's action chunk size is determined by how the server was launched; "
        "replan_horizon controls how many served actions CAP executes before returning."
    )
    parameters = [
        ToolParameter("model", "str", _MODEL_DESCRIPTION),
        ToolParameter(
            "replan_horizon",
            "int",
            "How many actions from the served chunk CAP executes before returning.",
            required=False,
            default=30,
        ),
        ToolParameter(
            "max_steps",
            "int",
            "Optional number of policy-control steps to execute. Defaults to replan_horizon.",
            required=False,
        ),
        ToolParameter(
            "task_description",
            "str",
            "Optional language command to send with the observation. Uses a model-specific default when omitted.",
            required=False,
            default="",
        ),
        ToolParameter(
            "policy_server",
            "str",
            "Optional host:port override for the already-running policy server.",
            required=False,
            default="",
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        model = str(kwargs["model"]).strip()
        replan_horizon = max(1, int(kwargs.get("replan_horizon", 30)))
        max_steps_raw = kwargs.get("max_steps")
        max_steps = 0 if max_steps_raw in (None, "") else max(1, int(max_steps_raw))
        task_description = (
            str(kwargs.get("task_description", "")).strip() or _PORTAL_EMPTY
        )
        policy_server = str(kwargs.get("policy_server", "")).strip() or _PORTAL_EMPTY

        try:
            result = (
                self._get_client()
                .use_policy_output(
                    model,
                    replan_horizon,
                    max_steps,
                    task_description,
                    policy_server,
                )
                .result()
            )
            return self._skill_result_from_rpc(result, "policy output execution failed")
        except Exception as e:
            return ToolResult(success=False, error=str(e) or repr(e))
