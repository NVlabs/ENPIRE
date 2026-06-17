import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

import cap.bridge.agent_bridge as agent_bridge
from cap.bridge.providers.base import (
    AgentBackend,
    AgentBackendSpec,
    AgentBridgeConfig,
    ChatSession,
    ProviderContext,
    TurnResult,
)
from cap.bridge.providers.claude_code import ClaudeCodeBackend
from cap.bridge.providers.openai_codex import OpenAICodexBackend


class _RecordingWSManager:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def broadcast(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeReadStream:
    def __init__(self, text: str) -> None:
        self._lines = [line.encode("utf-8") for line in text.splitlines(keepends=True)]
        self._remaining = text.encode("utf-8")

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""

    async def read(self) -> bytes:
        return self._remaining


class _FakeProcess:
    def __init__(
        self, stdout_lines: list[str], stderr_text: str = "", returncode: int = 0
    ) -> None:
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeReadStream(stderr_text)
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode


class _StubProvider(AgentBackend):
    def __init__(
        self,
        backend: str,
        label: str,
        default_model: str | None,
        reasoning: tuple[str, ...] = (),
        default_reasoning: str | None = None,
    ) -> None:
        self.spec = AgentBackendSpec(
            backend=backend,
            label=label,
            models=(default_model,) if default_model else (),
            default_model=default_model,
            reasoning_options=reasoning,
            default_reasoning=default_reasoning if reasoning else None,
        )
        self.calls: list[tuple[str, AgentBridgeConfig]] = []

    async def run_turn(
        self,
        *,
        message: str,
        session: ChatSession,
        ws_manager,
        system_prompt: str,
        config: AgentBridgeConfig,
        context: ProviderContext,
    ) -> TurnResult:
        self.calls.append((message, config))
        session.session_id = f"{config.backend}-session"
        await ws_manager.broadcast("chat_text_delta", {"text": "hello"})
        return TurnResult(
            full_text="hello\n```python\nprint('ok')\n```",
            session_id=session.session_id,
        )


def test_claude_backend_builds_backward_compatible_command(tmp_path: Path) -> None:
    backend = ClaudeCodeBackend()
    context = ProviderContext(
        project_root=tmp_path,
        cap_agent_url="http://localhost:8200",
        mcp_config_path=tmp_path / "mcp_config.json",
    )
    context.mcp_config_path.write_text("{}", encoding="utf-8")

    cmd = backend.build_command(
        message="hello",
        session_id="claude-session",
        system_prompt="SYSTEM",
        config=AgentBridgeConfig(
            backend="claude_code",
            model="claude-opus-4-6",
            reasoning="max",
        ),
        context=context,
    )

    assert cmd[:5] == ["claude", "-p", "hello", "--output-format", "stream-json"]
    assert "--resume" in cmd
    assert "claude-session" in cmd
    assert "--append-system-prompt" in cmd
    assert "SYSTEM" in cmd
    assert "--mcp-config" in cmd
    assert "--model" in cmd
    assert "claude-opus-4-6" in cmd
    assert "--effort" in cmd
    assert "max" in cmd


def test_codex_backend_builds_resume_command_with_model_and_reasoning(
    tmp_path: Path,
) -> None:
    backend = OpenAICodexBackend()
    output_file = tmp_path / "last.txt"
    context = ProviderContext(
        project_root=tmp_path,
        cap_agent_url="http://localhost:8200",
        mcp_config_path=tmp_path / "ignored.json",
    )
    cmd = backend.build_command(
        message="hi",
        session_id="codex-session",
        config=AgentBridgeConfig(
            backend="openai_codex",
            model="gpt-5.4-mini",
            reasoning="xhigh",
        ),
        context=context,
        output_file=output_file,
    )

    assert cmd[:4] == ["codex", "exec", "resume", "--json"]
    assert cmd[-2:] == ["codex-session", "hi"]
    assert "-m" in cmd and "gpt-5.4-mini" in cmd
    assert "-o" in cmd
    assert str(output_file) in cmd
    assert 'model_reasoning_effort="xhigh"' in cmd
    assert 'mcp_servers.cap-robot.command="uv"' in cmd
    assert "--color" not in cmd
    assert "-C" not in cmd
    assert "-s" not in cmd


def test_codex_backend_reads_output_file_and_emits_tool_events(
    monkeypatch, tmp_path: Path
) -> None:
    backend = OpenAICodexBackend()
    ws_manager = _RecordingWSManager()
    session = ChatSession(conversation_dir=tmp_path / "conv")
    context = ProviderContext(
        project_root=tmp_path,
        cap_agent_url="http://localhost:8200",
        mcp_config_path=tmp_path / "ignored.json",
    )

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        output_idx = cmd.index("--output-last-message") + 1
        Path(cmd[output_idx]).write_text("Final answer from Codex", encoding="utf-8")
        return _FakeProcess(
            stdout_lines=[
                json.dumps(
                    {
                        "type": "tool_use",
                        "name": "get_robot_state",
                        "input": {"camera": "top"},
                        "session_id": "sess-42",
                    }
                )
                + "\n"
            ]
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = asyncio.run(
        backend.run_turn(
            message="Pick up cup",
            session=session,
            ws_manager=ws_manager,
            system_prompt="SYSTEM",
            config=AgentBridgeConfig(
                backend="openai_codex",
                model="gpt-5.4",
                reasoning="high",
            ),
            context=context,
        )
    )

    assert result.session_id == "sess-42"
    assert result.full_text == "Final answer from Codex"
    assert (
        "chat_tool_use",
        {"tool": "get_robot_state", "input": {"camera": "top"}},
    ) in ws_manager.events
    assert ("chat_text_delta", {"text": "Final answer from Codex"}) in ws_manager.events


def test_agent_bridge_options_and_selected_backend_chat(
    monkeypatch, tmp_path: Path
) -> None:
    claude = _StubProvider(
        "claude_code",
        "Claude Code",
        "claude-opus-4-6",
        ("low", "medium", "high", "max"),
        default_reasoning="medium",
    )
    codex = _StubProvider(
        "openai_codex", "OpenAI Codex", "gpt-5.4", ("low", "medium", "high")
    )

    monkeypatch.setattr(
        agent_bridge,
        "build_provider_registry",
        lambda: {
            "claude_code": claude,
            "openai_codex": codex,
        },
    )
    monkeypatch.setattr(agent_bridge, "_fetch_tool_docs", lambda: "TOOLS")
    monkeypatch.setattr(agent_bridge, "build_system_prompt", lambda docs: "SYSTEM")
    monkeypatch.setattr(agent_bridge, "CONVERSATION_DIR", tmp_path / "conversation")
    monkeypatch.setattr(agent_bridge, "PROMPT_DIR", tmp_path / "prompt")
    monkeypatch.setattr(
        agent_bridge, "PER_MESSAGE_DIR", tmp_path / "prompt" / "per_message"
    )
    monkeypatch.setattr(agent_bridge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(agent_bridge, "MCP_CONFIG", tmp_path / "mcp_config.json")

    app = agent_bridge.create_app()
    with TestClient(app) as client:
        options = client.get("/api/agent/options")
        assert options.status_code == 200
        payload = options.json()
        assert payload["current"]["backend"] == "claude_code"
        assert payload["current"]["model"] == "claude-opus-4-6"
        assert payload["current"]["reasoning"] == "medium"
        assert {item["backend"] for item in payload["backends"]} == {
            "claude_code",
            "openai_codex",
        }

        resp = client.post(
            "/api/chat",
            json={
                "message": "do the thing",
                "backend": "openai_codex",
                "model": "gpt-5.4",
                "reasoning": "medium",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        deadline = time.time() + 2
        while time.time() < deadline and not codex.calls:
            time.sleep(0.02)
        status = client.get("/api/chat/status").json()
        assert status["backend"] == "openai_codex"
        assert status["model"] == "gpt-5.4"
        assert status["reasoning"] == "medium"
        assert codex.calls and not claude.calls

        reset = client.post("/api/chat/reset", json={"backend": "openai_codex"})
        assert reset.status_code == 200
        status_after_reset = client.get("/api/chat/status").json()
        assert status_after_reset["session_id"] is None


def test_agent_bridge_websocket_streams_text_and_code_blocks(
    monkeypatch, tmp_path: Path
) -> None:
    codex = _StubProvider(
        "openai_codex", "OpenAI Codex", "gpt-5.4", ("low", "medium", "high")
    )

    monkeypatch.setattr(
        agent_bridge, "build_provider_registry", lambda: {"openai_codex": codex}
    )
    monkeypatch.setattr(agent_bridge, "_fetch_tool_docs", lambda: "TOOLS")
    monkeypatch.setattr(agent_bridge, "build_system_prompt", lambda docs: "SYSTEM")
    monkeypatch.setattr(agent_bridge, "CONVERSATION_DIR", tmp_path / "conversation")
    monkeypatch.setattr(agent_bridge, "PROMPT_DIR", tmp_path / "prompt")
    monkeypatch.setattr(
        agent_bridge, "PER_MESSAGE_DIR", tmp_path / "prompt" / "per_message"
    )
    monkeypatch.setattr(agent_bridge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(agent_bridge, "MCP_CONFIG", tmp_path / "mcp_config.json")

    app = agent_bridge.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            resp = client.post(
                "/api/chat",
                json={
                    "message": "stream please",
                    "backend": "openai_codex",
                    "model": "gpt-5.4",
                    "reasoning": "high",
                },
            )
            assert resp.status_code == 200
            seen_types = []
            deadline = time.time() + 2.0
            while time.time() < deadline:
                payload = ws.receive_json()
                seen_types.append(payload["type"])
                if payload["type"] == "chat_turn_complete":
                    break

    assert "chat_text_delta" in seen_types
    assert "chat_code_block" in seen_types
    assert seen_types[-1] == "chat_turn_complete"


def test_backend_sessions_are_isolated(monkeypatch, tmp_path: Path) -> None:
    claude = _StubProvider(
        "claude_code",
        "Claude Code",
        "claude-opus-4-6",
        ("low", "medium", "high", "max"),
        default_reasoning="medium",
    )
    codex = _StubProvider(
        "openai_codex", "OpenAI Codex", "gpt-5.4", ("low", "medium", "high")
    )

    monkeypatch.setattr(
        agent_bridge,
        "build_provider_registry",
        lambda: {
            "claude_code": claude,
            "openai_codex": codex,
        },
    )
    monkeypatch.setattr(agent_bridge, "_fetch_tool_docs", lambda: "TOOLS")
    monkeypatch.setattr(agent_bridge, "build_system_prompt", lambda docs: "SYSTEM")
    monkeypatch.setattr(agent_bridge, "CONVERSATION_DIR", tmp_path / "conversation")
    monkeypatch.setattr(agent_bridge, "PROMPT_DIR", tmp_path / "prompt")
    monkeypatch.setattr(
        agent_bridge, "PER_MESSAGE_DIR", tmp_path / "prompt" / "per_message"
    )
    monkeypatch.setattr(agent_bridge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(agent_bridge, "MCP_CONFIG", tmp_path / "mcp_config.json")

    app = agent_bridge.create_app()
    with TestClient(app) as client:
        client.post(
            "/api/chat", json={"message": "hello from claude", "backend": "claude_code"}
        )
        deadline = time.time() + 2
        while time.time() < deadline and not claude.calls:
            time.sleep(0.02)
        client.post(
            "/api/chat", json={"message": "hello from codex", "backend": "openai_codex"}
        )
        deadline = time.time() + 2
        while time.time() < deadline and not codex.calls:
            time.sleep(0.02)

        status = client.get("/api/chat/status").json()
        assert status["backend"] == "openai_codex"
        assert status["session_id"] == "openai_codex-session"

        client.post("/api/chat/reset", json={"backend": "openai_codex"})
        time.sleep(0.02)
        status_after_codex_reset = client.get("/api/chat/status").json()
        assert status_after_codex_reset["session_id"] is None

        client.post(
            "/api/chat", json={"message": "resume claude", "backend": "claude_code"}
        )
        deadline = time.time() + 2
        while time.time() < deadline and len(claude.calls) < 2:
            time.sleep(0.02)
        status_after_claude = client.get("/api/chat/status").json()
        assert status_after_claude["backend"] == "claude_code"
        assert status_after_claude["session_id"] == "claude_code-session"
