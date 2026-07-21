"""Provider interfaces and shared state for the CAP agent bridge."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHAT_SESSION_MESSAGES = 40


@dataclass(frozen=True)
class AgentBackendSpec:
    backend: str
    label: str
    models: tuple[str, ...] = ()
    default_model: str | None = None
    reasoning_options: tuple[str, ...] = ()
    default_reasoning: str | None = None

    def normalize(
        self,
        *,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> tuple[str | None, str | None]:
        use_model = model or self.default_model
        if self.models and use_model not in self.models:
            use_model = self.default_model

        use_reasoning = reasoning or self.default_reasoning
        if self.reasoning_options and use_reasoning not in self.reasoning_options:
            use_reasoning = self.default_reasoning
        if not self.reasoning_options:
            use_reasoning = None
        return use_model, use_reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "label": self.label,
            "models": list(self.models),
            "default_model": self.default_model,
            "reasoning_options": list(self.reasoning_options),
            "default_reasoning": self.default_reasoning,
        }


@dataclass
class AgentBridgeConfig:
    backend: str
    model: str | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "reasoning": self.reasoning,
        }


@dataclass
class ChatSession:
    conversation_dir: Path
    session_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    log_path: Path | None = None

    def append_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_CHAT_SESSION_MESSAGES:
            del self.messages[:-MAX_CHAT_SESSION_MESSAGES]

    def reset(self) -> None:
        self.session_id = None
        self.messages.clear()
        self.log_path = None

    def _ensure_log(self) -> Path:
        if self.log_path is None:
            self.conversation_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.log_path = self.conversation_dir / f"{ts}.md"
            self.log_path.write_text(f"# Conversation {ts}\n\n", encoding="utf-8")
            logger.info("Conversation log: %s", self.log_path)
        return self.log_path

    def log_user(self, message: str) -> None:
        path = self._ensure_log()
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## User\n\n{message}\n\n")

    def log_assistant(self, text: str) -> None:
        path = self._ensure_log()
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## Assistant\n\n{text}\n\n---\n\n")

    def log_tool_use(self, tool: str, tool_input: dict[str, Any]) -> None:
        path = self._ensure_log()
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"> **MCP Tool:** `{tool}({json.dumps(tool_input, default=str)})`\n\n")


@dataclass(frozen=True)
class ProviderContext:
    project_root: Path
    cap_agent_url: str
    mcp_config_path: Path


@dataclass
class TurnResult:
    full_text: str
    session_id: str | None = None


class AgentBackend(ABC):
    spec: AgentBackendSpec

    @abstractmethod
    async def run_turn(
        self,
        *,
        message: str,
        session: ChatSession,
        ws_manager: Any,
        system_prompt: str,
        config: AgentBridgeConfig,
        context: ProviderContext,
    ) -> TurnResult:
        raise NotImplementedError
