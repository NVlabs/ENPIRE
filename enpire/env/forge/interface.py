# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable contracts shared by simulation and real-world environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class StepResult:
    """Result of applying one policy action to an environment."""

    observation: Any
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """Task success determined independently from the policy."""

    success: bool
    score: float | None = None
    reason: str | None = None
    metrics: dict[str, float | int | bool | str | None] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    """Minimal reset/execute/verify interface used by the ENPIRE loop."""

    def reset(self, *, seed: int | None = None) -> Any:
        """Reset the task and return the first observation."""

    def observe(self) -> Any:
        """Return the current observation without changing the task."""

    def step(self, action: Any) -> StepResult:
        """Apply one policy action."""

    def verify(self) -> VerificationResult:
        """Evaluate success using environment-owned evidence."""

    def close(self) -> None:
        """Release hardware, processes, and other resources."""
