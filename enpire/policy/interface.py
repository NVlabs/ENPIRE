# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy interface shared by code-as-policy and learned policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    def reset(self) -> None:
        """Clear episode-local policy state."""

    def act(self, observation: Any) -> Any:
        """Return one environment action."""

    def close(self) -> None:
        """Release model, socket, or accelerator resources."""


@dataclass
class FunctionPolicy:
    """Adapt a Python function into the public Policy protocol."""

    function: Callable[[Any], Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        return None

    def act(self, observation: Any) -> Any:
        return self.function(observation)

    def close(self) -> None:
        return None
