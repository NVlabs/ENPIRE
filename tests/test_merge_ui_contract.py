# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def _source(path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).read_text(encoding="utf-8")


def test_chatpanel_keeps_agent_backend_controls() -> None:
    source = _source("cap/ui/src/components/ChatPanel.tsx")
    for label in ("Agent", "Model", "Reasoning"):
        assert label in source, f"Expected ChatPanel.tsx to keep the {label} control"
    assert "onSend(trimmed, agentConfig)" in source


def test_chatpanel_adds_voice_controls_after_merge() -> None:
    source = _source("cap/ui/src/components/ChatPanel.tsx")
    expected_snippets = [
        "voice-transcript-bar",
        "Send transcript",
        "Stop mic",
        "Voice Test",
    ]
    missing = [snippet for snippet in expected_snippets if snippet not in source]
    assert not missing, (
        "Merge incomplete: ChatPanel.tsx is still missing merged voice controls. "
        f"Missing snippets: {missing}"
    )


def test_app_wires_voice_callbacks_after_merge() -> None:
    source = _source("cap/ui/src/App.tsx")
    expected = [
        "onVoiceStart=",
        "onVoiceStop=",
        "onVoiceClear=",
        "voice=",
    ]
    missing = [snippet for snippet in expected if snippet not in source]
    assert not missing, (
        "Merge incomplete: App.tsx is not yet wiring voice state/callbacks into ChatPanel. "
        f"Missing: {missing}"
    )


def test_usechatstream_keeps_agent_config_in_send_path() -> None:
    source = _source("cap/ui/src/hooks/useChatStream.ts")
    assert "AgentBridgeConfig" in source
    assert 'body: JSON.stringify({ message: text, ...config })' in source
    assert 'body: JSON.stringify(config)' in source
