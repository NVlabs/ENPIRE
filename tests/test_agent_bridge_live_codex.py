# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import enpire.env.forge.cap.bridge.agent_bridge as agent_bridge

pytestmark = pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex CLI is not installed",
)


def _should_run_live() -> bool:
    import os

    return os.environ.get("RUN_LIVE_CODEX") == "1"


@pytest.mark.skipif(not _should_run_live(), reason="set RUN_LIVE_CODEX=1 to run live Codex integration")
def test_live_codex_bridge_supports_model_switching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_bridge, "CONVERSATION_DIR", tmp_path / "conversation")
    monkeypatch.setattr(agent_bridge, "PROMPT_DIR", tmp_path / "prompt")
    monkeypatch.setattr(agent_bridge, "PER_MESSAGE_DIR", (tmp_path / "prompt" / "per_message"))
    monkeypatch.setattr(agent_bridge, "_fetch_tool_docs", lambda: "")
    monkeypatch.setattr(
        agent_bridge,
        "build_system_prompt",
        lambda docs: "Reply briefly. Do not use tools unless necessary. Always include one python fenced code block.",
    )

    app = agent_bridge.create_app()
    with TestClient(app) as client:
        for model in ("gpt-5.4-mini", "gpt-5.4"):
            client.post("/api/chat/reset", json={"backend": "openai_codex"})
            resp = client.post(
                "/api/chat",
                json={
                    "message": f"Reply with exactly: hello from {model}. Then output one python code block that prints {model!r}.",
                    "backend": "openai_codex",
                    "model": model,
                    "reasoning": "low",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

            deadline = time.time() + 120
            last = None
            while time.time() < deadline:
                last = client.get("/api/chat/status").json()
                if not last["generating"]:
                    break
                time.sleep(2)

            assert last is not None
            assert last["backend"] == "openai_codex"
            assert last["model"] == model
            assert last["reasoning"] == "low"
            assert not last["generating"]

            latest = sorted((tmp_path / "conversation").glob("*.md"))[-1]
            text = latest.read_text(encoding="utf-8")
            assert f"hello from {model}" in text.lower()
            assert "```python" in text
