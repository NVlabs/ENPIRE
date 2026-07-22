# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend providers for the CAP agent bridge."""

from .base import AgentBackend, AgentBackendSpec, AgentBridgeConfig, ChatSession, ProviderContext, TurnResult
from .claude_code import ClaudeCodeBackend
from .openai_codex import OpenAICodexBackend

__all__ = [
    "AgentBackend",
    "AgentBackendSpec",
    "AgentBridgeConfig",
    "ChatSession",
    "ProviderContext",
    "TurnResult",
    "ClaudeCodeBackend",
    "OpenAICodexBackend",
]
