import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

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
    default_reasoning: "low",
  },
];

describe("ChatPanel voice output", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("simulates clicking Voice Test offline and posts Hello World to the bridge voice API", async () => {
    const fetchMock = vi.fn(async () =>
      ({
        ok: true,
        json: async () => ({ ok: true }),
      }) as Response,
    );
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ChatPanel
        messages={[]}
        isStreaming={false}
        connected={true}
        voice={BASE_VOICE}
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

    fireEvent.click(screen.getByRole("button", { name: "Voice Test" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledWith(
      "/bridge-api/voice/test",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "Hello World" }),
      }),
    );
  });
});
