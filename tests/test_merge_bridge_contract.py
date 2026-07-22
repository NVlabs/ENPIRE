# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from enpire.env.forge.cap.bridge import agent_bridge, claude_bridge


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_claude_bridge_remains_backward_compatible_wrapper() -> None:
    source = (_repo_root() / "cap" / "bridge" / "claude_bridge.py").read_text(encoding="utf-8")
    assert "from cap.bridge.agent_bridge import main" in source
    assert "if __name__ == \"__main__\":" in source
    assert not source.strip().endswith("pass")


def test_agent_bridge_still_lists_claude_and_codex_backends() -> None:
    app = agent_bridge.create_app()
    with TestClient(app) as client:
        response = client.get("/api/agent/options")
    assert response.status_code == 200
    payload = response.json()
    backends = {item["backend"] for item in payload["backends"]}
    assert {"claude_code", "openai_codex"}.issubset(backends)


def test_agent_bridge_exposes_voice_api_routes_after_merge() -> None:
    app = agent_bridge.create_app()
    route_paths = {route.path for route in app.routes}
    missing = {
        "/api/voice/status",
        "/api/voice/enabled",
        "/api/voice/test",
        "/api/voice/speak",
    } - route_paths
    assert not missing, (
        "Merge incomplete: generic agent bridge still lacks voice routes. "
        f"Missing routes: {sorted(missing)}"
    )


def test_agent_bridge_owns_voice_output_integration_after_merge() -> None:
    agent_source = (_repo_root() / "cap" / "bridge" / "agent_bridge.py").read_text(encoding="utf-8")
    claude_source = (_repo_root() / "cap" / "bridge" / "claude_bridge.py").read_text(encoding="utf-8")

    assert "ChatVoiceController" in agent_source, (
        "Merge incomplete: generic agent bridge should integrate ChatVoiceController so "
        "voice output works for both Claude and Codex."
    )
    assert "ChatVoiceController" not in claude_source, (
        "Merge regression: claude_bridge.py should stay a thin wrapper, not regain Claude-only "
        "voice-output logic."
    )


def test_claude_bridge_module_still_exports_main() -> None:
    assert callable(claude_bridge.main)
