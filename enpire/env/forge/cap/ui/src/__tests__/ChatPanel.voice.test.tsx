/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ChatPanel from "../components/ChatPanel";
import type { VoiceState } from "../hooks/useChatStream";
import type { AgentBackendOption, AgentBridgeConfig } from "../api/client";

const BASE_VOICE: VoiceState = {
  listening: false,
  phase: "idle",
  partialText: "",
  finalText: "",
  error: "",
};

const BASE_AGENT_CONFIG: AgentBridgeConfig = {
  backend: "openai_codex",
  model: "gpt-5.4-mini",
  reasoning: "low",
};

const BACKEND_OPTIONS: AgentBackendOption[] = [
  {
    backend: "openai_codex",
    label: "OpenAI Codex",
    models: ["gpt-5.4-mini", "gpt-5.4"],
    default_model: "gpt-5.4-mini",
    reasoning_options: ["low", "medium", "high"],
    default_reasoning: "high",
  },
];

describe("ChatPanel voice input", () => {
  it("renders realtime transcript text below the mic button", () => {
    render(
      <ChatPanel
        messages={[]}
        isStreaming={false}
        connected={true}
        voice={{ ...BASE_VOICE, listening: true, phase: "recording", partialText: "pick up the cup" }}
        onSend={vi.fn()}
        onReset={vi.fn()}
        onEvolve={vi.fn()}
        onVoiceStart={vi.fn()}
        onVoiceStop={vi.fn()}
        onVoiceDismiss={vi.fn()}
        onVoiceClear={vi.fn()}
        agentConfig={BASE_AGENT_CONFIG}
        backendOptions={BACKEND_OPTIONS}
        onAgentConfigChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("voice-transcript-bar")).toHaveTextContent("pick up the cup");
    expect(screen.getByRole("button", { name: "Stop mic" })).toBeInTheDocument();
  });

  it("sends the finalized transcript through the normal send path", () => {
    const onSend = vi.fn();
    const onVoiceDismiss = vi.fn();
    const onVoiceClear = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        isStreaming={false}
        connected={true}
        voice={{ ...BASE_VOICE, listening: true, finalText: "move the blue block to the tray" }}
        onSend={onSend}
        onReset={vi.fn()}
        onEvolve={vi.fn()}
        onVoiceStart={vi.fn()}
        onVoiceStop={vi.fn()}
        onVoiceDismiss={onVoiceDismiss}
        onVoiceClear={onVoiceClear}
        agentConfig={BASE_AGENT_CONFIG}
        backendOptions={BACKEND_OPTIONS}
        onAgentConfigChange={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Send transcript" }));
    expect(onVoiceDismiss).toHaveBeenCalledTimes(1);
    expect(onVoiceClear).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("move the blue block to the tray", BASE_AGENT_CONFIG);
  });

  it("stops the mic before sending typed input", () => {
    const onSend = vi.fn();
    const onVoiceDismiss = vi.fn();
    const onVoiceClear = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        isStreaming={false}
        connected={true}
        voice={{ ...BASE_VOICE, listening: true }}
        onSend={onSend}
        onReset={vi.fn()}
        onEvolve={vi.fn()}
        onVoiceStart={vi.fn()}
        onVoiceStop={vi.fn()}
        onVoiceDismiss={onVoiceDismiss}
        onVoiceClear={onVoiceClear}
        agentConfig={BASE_AGENT_CONFIG}
        backendOptions={BACKEND_OPTIONS}
        onAgentConfigChange={vi.fn()}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText("Describe a task or ask a question..."), {
      target: { value: "hello" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(onVoiceDismiss).toHaveBeenCalledTimes(1);
    expect(onVoiceClear).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("hello", BASE_AGENT_CONFIG);
  });
});
