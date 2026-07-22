# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from enpire.env.forge.cap.agent.llm.base import LLMBackend  # noqa: F401
from enpire.env.forge.cap.agent.llm.cloud import CloudLLM  # noqa: F401
from enpire.env.forge.cap.agent.llm.gemini import GeminiLLM  # noqa: F401
from enpire.env.forge.cap.agent.llm.nvidia import NvidiaLLM  # noqa: F401
from enpire.env.forge.cap.agent.llm.bridge_llm import BridgeLLMBackend  # noqa: F401


def make_llm(
    cfg,
    system_prompt: str = "",
    *,
    model_override: str | None = None,
    replace_system_prompt: bool = False,
) -> LLMBackend:
    """Build an LLMBackend from Hydra config.

    Uses ``cfg.llm.backend`` and ``cfg.llm.model`` (or *model_override*).
    """
    backend = cfg.llm.backend
    model = model_override or cfg.llm.model
    project_root = Path(__file__).resolve().parents[3]

    if backend in ("claude", "cloud"):
        # system_prompt is passed straight through to the Anthropic
        # messages.create() system field, so stage-specific prompts
        # (skill_author.md, assembly_generator.md, …) are respected.
        # If the caller leaves it empty, CloudLLM falls back to
        # build_system_prompt(tool_docs, raw_code_only=True).
        return CloudLLM(
            model=model or "claude-sonnet-4-20250514",
            system_prompt=system_prompt or None,
        )

    if backend == "gemini":
        return GeminiLLM(
            model=model or "gemini-2.5-pro",
            system_prompt=system_prompt or None,
        )

    if backend == "nvidia":
        # NVIDIA inference gateway (OpenAI-compatible). Model strings follow
        # the ``<cloud>/<provider>/<model>`` convention used by the gateway,
        # e.g. ``gcp/google/gemini-3.1-pro-preview``, ``azure/openai/gpt-5.1``,
        # ``aws/anthropic/claude-opus-4-5``. Auth via ``NVIDIA_API_KEY`` env.
        return NvidiaLLM(
            model=model or "aws/anthropic/bedrock-claude-opus-4-6",
            system_prompt=system_prompt or None,
        )

    if backend.startswith("bridge:"):
        provider_name = backend.split(":", 1)[1]
        from enpire.env.forge.cap.bridge.providers import AgentBridgeConfig, ProviderContext

        if provider_name == "claude_code":
            from enpire.env.forge.cap.bridge.providers.claude_code import ClaudeCodeBackend

            provider = ClaudeCodeBackend()
        elif provider_name in ("openai", "openai_codex"):
            from enpire.env.forge.cap.bridge.providers.openai_codex import OpenAICodexBackend

            provider = OpenAICodexBackend()
        else:
            raise ValueError(f"Unknown bridge provider: {provider_name!r}")

        bridge_cfg = AgentBridgeConfig(
            backend=provider_name,
            model=model or provider.spec.default_model,
            reasoning=provider.spec.default_reasoning,
        )
        context = ProviderContext(
            project_root=project_root,
            cap_agent_url=cfg.llm.cap_agent_url,
            mcp_config_path=project_root / cfg.llm.mcp_config_path,
        )
        return BridgeLLMBackend(
            backend=provider,
            config=bridge_cfg,
            context=context,
            system_prompt=system_prompt,
            replace_system_prompt=replace_system_prompt,
        )

    raise ValueError(
        f"Unknown LLM backend: {backend!r}. "
        "Use 'claude', 'gemini', 'nvidia', or 'bridge:<provider>'."
    )
