# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill execution tools — run a learned policy via cap_server.

The actual policy inference loop runs inside cap_server (it takes over the
POLICY_FREQ_HZ control loop for N steps).  These tools just tell cap_server which skill
to run and with what parameters.

ExecuteSkillTool: pure policy execution.
LearnSkillTool:   RL training loop — single-step actions from remote rl_policy_server
                  with human Fello takeover + data recording.
"""

from __future__ import annotations

from typing import Any

import portal

from enpire.env.forge.cap.agent.tools.base import SkillResult, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT


class ExecuteSkillTool(Tool):
    name = "execute_skill"
    description = (
        "Execute a learned skill (flow matching policy). "
        "Blocking -- runs the policy at POLICY_FREQ_HZ until completion. "
        "Returns success/failure and steps executed."
    )
    parameters = [
        ToolParameter("skill_name", "str", "Name of the skill to execute."),
        ToolParameter(
            "params", "dict", "Skill-specific parameters (e.g. target object, task description).",
            required=False, default={},
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        policy_server: str = "localhost:8964",
    ):
        self._host = host
        self._port = port
        self._policy_server = policy_server
        self._client: portal.Client | None = None

    def _get_client(self) -> portal.Client:
        if self._client is None:
            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    def execute(self, **kwargs: Any) -> ToolResult:
        skill_name: str = kwargs["skill_name"]
        params: dict = kwargs.get("params", {})
        policy_server: str = kwargs.get("policy_server", self._policy_server)

        try:
            client = self._get_client()
            result = client.execute_skill(
                skill_name, policy_server, params
            ).result()
            skill_result = SkillResult(
                success=bool(result.get("success", False)),
                steps_executed=int(result.get("steps_executed", 0)),
                info=result.get("info", {}),
            )
            return ToolResult(success=skill_result.success, data=skill_result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LearnSkillTool(Tool):
    name = "learn_skill"
    description = (
        "Run an RL training episode. Single-step actions come from a remote rl_policy_server "
        "(SAC + base agent) rather than a local flow-matching policy. Supports human Fello "
        "takeover — pressing the footswitch overrides with human joint positions (tagged as "
        "'human' for the intervention buffer). All step data is recorded to disk. "
        "Blocking — runs at POLICY_FREQ_HZ until max_steps or done. "
        "Returns success/failure, steps executed, and the episode directory."
    )
    parameters = [
        ToolParameter("skill_name", "str", "Name / description of the RL skill."),
        ToolParameter(
            "params", "dict",
            "Optional overrides: rl_host (str), rl_port (int), max_steps (int), "
            "output_dir (str), task_description (str), "
            "control_mode (str, 'left'/'right'/'both', default 'both').",
            required=False, default={},
        ),
    ]

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

    def execute(self, **kwargs: Any) -> ToolResult:
        skill_name: str = kwargs["skill_name"]
        params: dict = kwargs.get("params", {})

        try:
            client = self._get_client()
            result = client.learn_skill(skill_name, params).result()
            skill_result = SkillResult(
                success=bool(result.get("success", False)),
                steps_executed=int(result.get("steps_executed", 0)),
                info=result.get("info", {}),
            )
            return ToolResult(success=skill_result.success, data=skill_result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
