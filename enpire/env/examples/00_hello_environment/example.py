"""Smallest complete ENPIRE reset-execute-verify loop."""

from __future__ import annotations

from pathlib import Path

from enpire.env.forge.artifacts import ArtifactStore
from enpire.env.forge.interface import StepResult, VerificationResult
from enpire.env.forge.loop import TrialRunner
from enpire.policy.interface import FunctionPolicy


class CounterEnvironment:
    def __init__(self, target: int = 3):
        self.target = target
        self.value = 0

    def reset(self, *, seed: int | None = None) -> dict[str, int | None]:
        self.value = 0
        return {"value": self.value, "seed": seed}

    def observe(self) -> dict[str, int]:
        return {"value": self.value}

    def step(self, action: int) -> StepResult:
        self.value += int(action)
        return StepResult(
            observation=self.observe(),
            reward=1.0,
            terminated=self.value >= self.target,
        )

    def verify(self) -> VerificationResult:
        return VerificationResult(
            success=self.value == self.target,
            score=self.value / self.target,
            metrics={"final_value": self.value, "target": self.target},
        )

    def close(self) -> None:
        return None


def main(*, output: str | Path = "outputs/hello-environment") -> int:
    environment = CounterEnvironment()
    policy = FunctionPolicy(lambda observation: 1)
    runner = TrialRunner(
        environment,
        policy,
        artifacts=ArtifactStore(output),
        max_steps=5,
    )
    try:
        result = runner.run(seed=0)
    finally:
        runner.close()
    print(f"success={result.success} steps={result.steps} output={Path(output)}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
