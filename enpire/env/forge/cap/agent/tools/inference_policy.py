# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference policy tool — run a VLA policy in an environment.

Wraps cap.policy.inference_policy() as a CAP agent tool.
Works with any PolicyBackend (ZMQ for GR00T, HTTP for OpenPI, etc.)
and any env with reset()/step() (RoboCasa sim, YAM real, etc.).
"""

from __future__ import annotations

from typing import Any

from enpire.env.forge.cap.agent.tools.base import SkillResult, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.policy import (
    InferencePolicyConfig,
    PolicyBackend,
    ZMQPolicyBackend,
    inference_policy,
)


class InferencePolicyTool(Tool):
    name = "inference_policy"
    description = (
        "Run a VLA policy (e.g. GR00T, Pi0) in the current environment. "
        "Connects to a policy server, executes receding-horizon action "
        "chunking, and returns success/failure with steps executed."
    )
    parameters = [
        ToolParameter("task_description", "str", "Language instruction for the policy."),
        ToolParameter("max_steps", "int", "Maximum env steps.", required=False, default=720),
        ToolParameter("n_episodes", "int", "Number of episodes.", required=False, default=1),
        ToolParameter("action_horizon", "int", "Model prediction horizon.", required=False, default=16),
        ToolParameter("replan_horizon", "int", "Steps to execute before replanning.", required=False, default=8),
        ToolParameter("use_async", "bool", "Background-thread inference.", required=False, default=False),
        ToolParameter("use_chunk_smoothing", "bool", "Linear blend at chunk boundaries.", required=False, default=False),
    ]

    def __init__(self, backend: PolicyBackend | None = None, **backend_kwargs: Any):
        """Create with an existing backend, or pass kwargs for ZMQPolicyBackend."""
        self._backend = backend
        self._backend_kwargs = backend_kwargs

    def _get_backend(self) -> PolicyBackend:
        if self._backend is None:
            self._backend = ZMQPolicyBackend(**self._backend_kwargs)
        return self._backend

    def execute(self, *, env: Any, **kwargs: Any) -> ToolResult:
        """Run the policy. Caller provides the env instance.

        Args:
            env: gym.Env or any object with reset()/step()/close().
            **kwargs: Matches self.parameters.

        Returns:
            ToolResult with SkillResult data.
        """
        config = InferencePolicyConfig(
            action_horizon=kwargs.get("action_horizon", 16),
            replan_horizon=kwargs.get("replan_horizon", 8),
            max_episode_steps=kwargs.get("max_steps", 720),
            use_async=kwargs.get("use_async", False),
            use_chunk_smoothing=kwargs.get("use_chunk_smoothing", False),
        )
        n_episodes = kwargs.get("n_episodes", 1)
        backend = self._get_backend()

        results = inference_policy(env, backend, config, n_episodes=n_episodes)

        total_steps = sum(r.steps for r in results)
        successes = sum(r.success for r in results)
        return ToolResult(
            success=successes > 0,
            data=SkillResult(
                success=successes == n_episodes,
                steps_executed=total_steps,
                info={
                    "n_episodes": n_episodes,
                    "n_successes": successes,
                    "success_rate": successes / n_episodes * 100,
                    "per_episode": [r.success for r in results],
                },
            ),
        )
