"""Deterministic reset-execute-verify trial loop."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from enpire.env.forge.artifacts import ArtifactStore
from enpire.env.forge.interface import Environment, VerificationResult
from enpire.policy.interface import Policy


@dataclass(frozen=True)
class TrialResult:
    success: bool
    steps: int
    total_reward: float
    termination: str
    verification: VerificationResult
    elapsed_s: float
    info: dict[str, Any] = field(default_factory=dict)


class TrialRunner:
    def __init__(
        self,
        environment: Environment,
        policy: Policy,
        *,
        max_steps: int = 200,
        artifacts: ArtifactStore | None = None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.environment = environment
        self.policy = policy
        self.max_steps = max_steps
        self.artifacts = artifacts

    def run(self, *, seed: int | None = None) -> TrialResult:
        started = time.monotonic()
        self.policy.reset()
        observation = self.environment.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        termination = "max_steps"

        if self.artifacts:
            self.artifacts.append_event({"event": "reset", "seed": seed})

        for index in range(self.max_steps):
            action = self.policy.act(observation)
            step = self.environment.step(action)
            observation = step.observation
            total_reward += float(step.reward)
            steps = index + 1
            if self.artifacts:
                self.artifacts.append_event(
                    {
                        "event": "step",
                        "index": index,
                        "reward": step.reward,
                        "terminated": step.terminated,
                        "truncated": step.truncated,
                        "info": step.info,
                    }
                )
            if step.terminated:
                termination = "terminated"
                break
            if step.truncated:
                termination = "truncated"
                break

        verification = self.environment.verify()
        result = TrialResult(
            success=verification.success,
            steps=steps,
            total_reward=total_reward,
            termination=termination,
            verification=verification,
            elapsed_s=time.monotonic() - started,
        )
        if self.artifacts:
            self.artifacts.write_json("result.json", asdict(result))
            self.artifacts.append_event({"event": "verify", "verification": asdict(verification)})
        return result

    def close(self) -> None:
        try:
            self.policy.close()
        finally:
            self.environment.close()
