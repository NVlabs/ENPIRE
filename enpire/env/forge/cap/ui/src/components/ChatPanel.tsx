/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { voiceTest } from "../api/client";
import type { AgentBackendOption, AgentBridgeConfig } from "../api/client";
import { ChatMessage } from "../hooks/useChatStream";
import type { VoiceState } from "../hooks/useChatStream";

interface ChatPanelProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  connected: boolean;
  voice: VoiceState;
  agentName?: string;
  onSend: (message: string, config?: AgentBridgeConfig) => void;
  onReset: (config?: AgentBridgeConfig) => void;
  onEvolve: (config?: AgentBridgeConfig) => void;
  onVoiceStart: () => void | Promise<void>;
  onVoiceStop: () => void | Promise<void>;
  onVoiceDismiss: () => void | Promise<void>;
  onVoiceClear: () => void | Promise<void>;
  agentConfig: AgentBridgeConfig;
  backendOptions: AgentBackendOption[];
  onAgentConfigChange: (config: AgentBridgeConfig) => void;
}

export default function ChatPanel({
  messages,
  isStreaming,
  connected,
  voice,
  agentName = "",
  onSend,
  onReset,
  onEvolve,
  onVoiceStart,
  onVoiceStop,
  onVoiceDismiss,
  onVoiceClear,
  agentConfig,
  backendOptions,
  onAgentConfigChange,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [voiceTesting, setVoiceTesting] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const selectedBackend =
    backendOptions.find((opt) => opt.backend === agentConfig.backend) ?? backendOptions[0];

  useEffect(() => {
    if (voice.finalText) {
      setInput(voice.finalText);
    }
  }, [voice.finalText]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    if (voice.listening) {
      void onVoiceDismiss();
    }
    void onVoiceClear();
    onSend(trimmed, agentConfig);
    setInput("");
  };

  const handleEvolve = async () => {
    if (isStreaming) return;
    try {
      onEvolve(agentConfig);
    } catch {
      // ignore — errors show in chat stream
    }
  };

  const handleVoiceTest = async () => {
    if (voiceTesting) return;
    setVoiceTesting(true);
    try {
      if (voice.listening) {
        void onVoiceDismiss();
      }
      await voiceTest("Hello World");
    } finally {
      setVoiceTesting(false);
    }
  };

  const handleVoiceSend = () => {
    const text = voice.finalText.trim() || voice.partialText.trim();
    if (!text || isStreaming) return;
    if (voice.listening) {
      void onVoiceDismiss();
    }
    void onVoiceClear();
    onSend(text, agentConfig);
    setInput("");
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-base-300 px-3 py-1.5">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Chat</span>
          <div
            className={`badge badge-xs ${connected ? "badge-success" : "badge-error"}`}
          >
            {connected ? "Bridge" : "Offline"}
          </div>
          {agentName ? (
            <span className="text-xs text-base-content/70">Name: {agentName}</span>
          ) : (
            <span className="text-xs font-medium text-error">Agent name not set</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs">
            <span className="opacity-70">Model backend</span>
            <select
              className="select select-bordered select-xs"
              value={agentConfig.backend ?? ""}
              disabled={isStreaming}
              onChange={(e) => {
                const backend = backendOptions.find((opt) => opt.backend === e.target.value);
                onAgentConfigChange({
                  backend: e.target.value,
                  model: backend?.default_model ?? null,
                  reasoning: backend?.default_reasoning ?? null,
                });
              }}
            >
              {backendOptions.map((opt) => (
                <option key={opt.backend} value={opt.backend}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          {selectedBackend?.models?.length ? (
            <label className="flex items-center gap-1 text-xs">
              <span className="opacity-70">Model</span>
              <select
                className="select select-bordered select-xs max-w-44"
                value={agentConfig.model ?? selectedBackend.default_model ?? ""}
                disabled={isStreaming}
                onChange={(e) =>
                  onAgentConfigChange({
                    ...agentConfig,
                    model: e.target.value,
                  })}
              >
                {selectedBackend.models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {selectedBackend?.reasoning_options?.length ? (
            <label className="flex items-center gap-1 text-xs">
              <span className="opacity-70">Reasoning</span>
              <select
                className="select select-bordered select-xs"
                value={agentConfig.reasoning ?? selectedBackend.default_reasoning ?? ""}
                disabled={isStreaming}
                onChange={(e) =>
                  onAgentConfigChange({
                    ...agentConfig,
                    reasoning: e.target.value,
                  })}
              >
                {selectedBackend.reasoning_options.map((reasoning) => (
                  <option key={reasoning} value={reasoning}>
                    {reasoning}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <button
            className="btn btn-ghost btn-xs"
            onClick={handleEvolve}
            disabled={isStreaming}
            title="Analyze latest conversation and extract task prompts"
          >
            Evolve
          </button>
          <button
            className="btn btn-ghost btn-xs"
            onClick={handleVoiceTest}
            disabled={voiceTesting}
            title="Play a Hello World voice-output test on the machine running the bridge"
          >
            {voiceTesting ? "Testing..." : "Voice Test"}
          </button>
          <button
            className="btn btn-ghost btn-xs"
            onClick={() => onReset(agentConfig)}
            title="Reset conversation"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-3 space-y-3"
      >
        {messages.length === 0 && (
          <div className="text-center text-base-content/40 text-sm py-8">
            Start a conversation to generate robot code...
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isStreaming && (
          <div className="flex items-center gap-2 text-sm text-base-content/50">
            <span className="loading loading-dots loading-xs" />
            Thinking...
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-base-300 p-2">
        <div className="mb-2 rounded-lg border border-base-300 bg-base-200/60 p-2">
          <div className="flex items-center gap-2">
            <button
              className={`btn btn-sm ${voice.listening ? "btn-error" : "btn-accent"}`}
              onClick={voice.listening ? onVoiceStop : onVoiceStart}
              disabled={isStreaming}
            >
              {voice.listening ? "Stop mic" : "Start mic"}
            </button>
            <button
              className="btn btn-sm btn-primary"
              onClick={handleVoiceSend}
              disabled={isStreaming || !(voice.finalText.trim() || voice.partialText.trim())}
            >
              Send transcript
            </button>
            <button
              className="btn btn-sm btn-ghost"
              onClick={onVoiceClear}
              disabled={isStreaming || (!voice.partialText && !voice.finalText && !voice.error)}
            >
              Clear
            </button>
            <span className="text-xs text-base-content/60">
              {voice.listening ? `Listening (${voice.phase})` : `Voice: ${voice.phase}`}
            </span>
          </div>
          <div
            className="mt-2 min-h-12 rounded border border-base-300 bg-base-100 px-3 py-2 text-sm"
            data-testid="voice-transcript-bar"
          >
            {voice.error ? (
              <span className="text-error">{voice.error}</span>
            ) : voice.partialText || voice.finalText ? (
              <span>{voice.finalText || voice.partialText}</span>
            ) : (
              <span className="text-base-content/40">
                Realtime transcript will appear here while you speak.
              </span>
            )}
          </div>
        </div>
        <div className="join w-full">
          <input
            type="text"
            className="input input-bordered input-sm join-item flex-1"
            placeholder={
              isStreaming
                ? "Waiting for response..."
                : "Describe a task or ask a question..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) handleSubmit();
            }}
            disabled={isStreaming}
          />
          <button
            className="btn btn-primary btn-sm join-item"
            onClick={handleSubmit}
            disabled={isStreaming || !input.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message bubble sub-component
// ---------------------------------------------------------------------------

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="chat chat-end">
        <div className="chat-bubble chat-bubble-primary text-sm whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === "tool") {
    return (
      <div className="collapse collapse-arrow bg-base-200 rounded-lg">
        <input type="checkbox" className="peer" />
        <div className="collapse-title text-xs text-base-content/60 py-1 min-h-0">
          <span className="badge badge-ghost badge-xs mr-1">MCP</span>
          {message.tool}
          {message.toolInput &&
            Object.keys(message.toolInput).length > 0 && (
              <span className="ml-1 opacity-60">
                ({Object.entries(message.toolInput)
                  .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                  .join(", ")})
              </span>
            )}
        </div>
        <div className="collapse-content text-xs">
          <pre className="whitespace-pre-wrap">
            {JSON.stringify(message.toolInput, null, 2)}
          </pre>
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="alert alert-error text-sm py-2">
        <span>{message.content}</span>
      </div>
    );
  }

  // Assistant message — code blocks auto-sent to editor, show indicator
  return (
    <div className="chat chat-start">
      <div className="chat-bubble text-sm max-w-full overflow-x-auto">
        <div className="prose prose-sm prose-invert max-w-none">
          <Markdown>{message.content}</Markdown>
        </div>

        {message.codeBlocks && message.codeBlocks.length > 0 && (
          <div className="mt-1 text-xs opacity-60">
            Code sent to editor
          </div>
        )}
      </div>
    </div>
  );
}
