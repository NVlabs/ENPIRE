# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM transport — one registry, one dispatch, one implementation per provider.

Two call paths use this package:

1. **Agent loop** — ``cap/agent/tools/vlm_query.py``'s ``VlmQueryTool``
   resolves ``media=...`` / ``camera=...`` to raw images and calls
   :func:`query` below.
2. **Script layer** — the closure injected into generated ``code.py`` (via
   ``cap/env/<env>/skills.py``) resolves camera names to images and calls
   :func:`query` below.

Both adapters end at the same entry point, so adding a new provider is
exactly one new file under ``backends/`` + self-registration.

Backend registry is populated on import of this package (see side-effects
below).
"""

from __future__ import annotations

from enpire.env.forge.cap.agent.tools.vlm.transport import VLMBackend, query, register  # noqa: F401

# Importing each backend module registers it with the transport registry.
from enpire.env.forge.cap.agent.tools.vlm.backends import (  # noqa: F401
    gemini,
    gpt,
    nvidia,
    qwen,
    smolvlm,
)

# Convenience re-exports for the NVIDIA-specific key rotation helpers that
# the per-seed VLM reflection uses (see cap/agent/agent_step.py).
from enpire.env.forge.cap.agent.tools.vlm.backends.nvidia import (  # noqa: F401
    list_nvidia_keys,
    pick_nvidia_key,
)
