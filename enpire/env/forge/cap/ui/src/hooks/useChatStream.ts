import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentBridgeConfig } from "../api/client";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool" | "error";
  content: string;
  /** For tool messages: tool name */
  tool?: string;
  /** For tool messages: tool input */
  toolInput?: Record<string, unknown>;
  /** For code blocks extracted from assistant text */
  codeBlocks?: { code: string; language: string }[];
}

type ChatWSEvent =
  | "chat_text_delta"
  | "chat_tool_use"
  | "chat_code_block"
  | "chat_turn_complete"
  | "chat_error";

type VoiceWSEvent =
  | "voice_status"
  | "voice_partial"
  | "voice_final"
  | "voice_error";

interface WSPayload {
  type: ChatWSEvent;
  data: Record<string, unknown>;
  timestamp: string;
}

interface UseChatStreamOptions {
  bridgeUrl?: string;
  bridgeWsUrl?: string;
  voiceUrl?: string;
  voiceWsUrl?: string;
  reconnectInterval?: number;
  /** Called when a code block is extracted — auto-sends to editor */
  onCodeBlock?: (code: string, language: string) => void;
}

export interface VoiceState {
  listening: boolean;
  phase: string;
  partialText: string;
  finalText: string;
  error: string;
}

const DEFAULT_VOICE_STATE: VoiceState = {
  listening: false,
  phase: "idle",
  partialText: "",
  finalText: "",
  error: "",
};

export function useChatStream(options: UseChatStreamOptions = {}) {
  const {
    bridgeUrl = "/bridge-api",
    bridgeWsUrl = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/bridge-ws/chat`,
    voiceUrl = "/voice-api",
    voiceWsUrl = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/voice-ws`,
    reconnectInterval = 2000,
    onCodeBlock,
  } = options;

  const onCodeBlockRef = useRef(onCodeBlock);
  onCodeBlockRef.current = onCodeBlock;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [connected, setConnected] = useState(false);
  const [voice, setVoice] = useState<VoiceState>(DEFAULT_VOICE_STATE);
  const chatWsRef = useRef<WebSocket | null>(null);
  const voiceWsRef = useRef<WebSocket | null>(null);
  const suppressVoiceEventsRef = useRef(false);
  const chatRetriesRef = useRef(0);
  const voiceRetriesRef = useRef(0);
  const maxRetries = 10;
  const msgIdRef = useRef(0);

  // Accumulator for the current assistant turn's streaming text
  const currentTextRef = useRef("");
  const currentCodeBlocksRef = useRef<{ code: string; language: string }[]>([]);

  const nextId = () => {
    msgIdRef.current += 1;
    return `msg-${msgIdRef.current}`;
  };

  const connectChat = useCallback(() => {
    if (chatWsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(bridgeWsUrl);

    ws.onopen = () => {
      setConnected(true);
      chatRetriesRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as WSPayload;
        handleEvent(payload);
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      // Guard against stale connection (React StrictMode double-mount race)
      if (chatWsRef.current !== ws) return;
      setConnected(false);
      chatWsRef.current = null;
      if (chatRetriesRef.current < maxRetries) {
        chatRetriesRef.current += 1;
        setTimeout(connectChat, reconnectInterval);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    chatWsRef.current = ws;
  }, [bridgeWsUrl, reconnectInterval]);

  const connectVoice = useCallback(() => {
    if (voiceWsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(voiceWsUrl);

    ws.onopen = () => {
      voiceRetriesRef.current = 0;
      fetch(`${voiceUrl}/status`)
        .then((r) => r.json())
        .then((data) => {
          setVoice({
            listening: Boolean(data.listening),
            phase: String(data.phase ?? "idle"),
            partialText: String(data.partial_text ?? ""),
            finalText: String(data.final_text ?? ""),
            error: String(data.error ?? ""),
          });
        })
        .catch(() => {
          setVoice((prev) => ({ ...prev, error: "Voice server unavailable" }));
        });
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as WSPayload;
        handleVoiceEvent(payload);
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      if (voiceWsRef.current !== ws) return;
      voiceWsRef.current = null;
      setVoice((prev) => ({
        ...prev,
        listening: false,
        phase: prev.phase === "idle" ? "idle" : "disconnected",
        error: "Voice server disconnected",
      }));
      if (voiceRetriesRef.current < maxRetries) {
        voiceRetriesRef.current += 1;
        setTimeout(connectVoice, reconnectInterval);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    voiceWsRef.current = ws;
  }, [voiceUrl, voiceWsUrl, reconnectInterval]);

  const handleEvent = (payload: WSPayload) => {
    const { type, data } = payload;

    switch (type) {
      case "chat_text_delta": {
        const text = (data.text as string) || "";
        currentTextRef.current += text;
        // Update or create the streaming assistant message
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant" && !last.id.startsWith("msg-done-")) {
            return [
              ...prev.slice(0, -1),
              { ...last, content: currentTextRef.current },
            ];
          }
          return [
            ...prev,
            {
              id: nextId(),
              role: "assistant",
              content: currentTextRef.current,
              codeBlocks: [],
            },
          ];
        });
        break;
      }
      case "chat_tool_use": {
        const tool = (data.tool as string) || "unknown";
        const input = (data.input as Record<string, unknown>) || {};
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "tool",
            content: `Using tool: ${tool}`,
            tool,
            toolInput: input,
          },
        ]);
        break;
      }
      case "chat_code_block": {
        const code = (data.code as string) || "";
        const language = (data.language as string) || "python";
        currentCodeBlocksRef.current.push({ code, language });
        // Update the assistant message with code blocks
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant") {
            return [
              ...prev.slice(0, -1),
              { ...last, codeBlocks: [...currentCodeBlocksRef.current] },
            ];
          }
          return prev;
        });
        // Auto-send to editor
        onCodeBlockRef.current?.(code, language);
        break;
      }
      case "chat_turn_complete": {
        setIsStreaming(false);
        // Finalize the assistant message ID so future deltas create a new one
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant") {
            return [
              ...prev.slice(0, -1),
              { ...last, id: `msg-done-${msgIdRef.current}` },
            ];
          }
          return prev;
        });
        currentTextRef.current = "";
        currentCodeBlocksRef.current = [];
        break;
      }
      case "chat_error": {
        const error = (data.error as string) || "Unknown error";
        setIsStreaming(false);
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "error", content: error },
        ]);
        currentTextRef.current = "";
        currentCodeBlocksRef.current = [];
        break;
      }
    }
  };

  const handleVoiceEvent = (payload: WSPayload) => {
    const type = payload.type as VoiceWSEvent;
    const data = payload.data;
    switch (type) {
      case "voice_status": {
        if (suppressVoiceEventsRef.current) {
          setVoice({
            listening: false,
            phase: "idle",
            partialText: "",
            finalText: "",
            error: "",
          });
          break;
        }
        setVoice({
          listening: Boolean(data.listening),
          phase: String(data.phase ?? "idle"),
          partialText: String(data.partial_text ?? ""),
          finalText: String(data.final_text ?? ""),
          error: String(data.error ?? ""),
        });
        break;
      }
      case "voice_partial": {
        if (suppressVoiceEventsRef.current) {
          break;
        }
        const text = String(data.text ?? "");
        setVoice((prev) => ({
          ...prev,
          listening: true,
          partialText: text,
          phase: prev.phase === "idle" ? "listening" : prev.phase,
          error: "",
        }));
        break;
      }
      case "voice_final": {
        if (suppressVoiceEventsRef.current) {
          break;
        }
        const text = String(data.text ?? "");
        setVoice((prev) => ({
          ...prev,
          listening: true,
          partialText: text,
          finalText: text,
          phase: "finalized",
          error: "",
        }));
        break;
      }
      case "voice_error": {
        const error = String(data.error ?? "Unknown voice error");
        setVoice((prev) => ({
          ...prev,
          listening: false,
          phase: "error",
          error,
        }));
        break;
      }
    }
  };

  const sendMessage = useCallback(
    async (text: string, config: AgentBridgeConfig = {}) => {
      // Add user message immediately
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", content: text },
      ]);
      setIsStreaming(true);
      currentTextRef.current = "";
      currentCodeBlocksRef.current = [];

      try {
        await fetch(`${bridgeUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, ...config }),
        });
      } catch (err) {
        setIsStreaming(false);
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "error",
            content: `Failed to send: ${err}`,
          },
        ]);
      }
    },
    [bridgeUrl],
  );

  const resetSession = useCallback(async (config: AgentBridgeConfig = {}) => {
    setMessages([]);
    setIsStreaming(false);
    currentTextRef.current = "";
    currentCodeBlocksRef.current = [];
    try {
      await fetch(`${bridgeUrl}/chat/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
    } catch {
      // ignore
    }
  }, [bridgeUrl]);

  const startVoiceListening = useCallback(async () => {
    suppressVoiceEventsRef.current = true;
    setVoice(DEFAULT_VOICE_STATE);
    await fetch(`${bridgeUrl}/voice/stop`, { method: "POST" }).catch(() => undefined);
    await fetch(`${voiceUrl}/clear`, { method: "POST" });
    suppressVoiceEventsRef.current = false;
    await fetch(`${voiceUrl}/start`, { method: "POST" });
  }, [bridgeUrl, voiceUrl]);

  const stopVoiceListening = useCallback(async () => {
    await fetch(`${voiceUrl}/stop`, { method: "POST" });
  }, [voiceUrl]);

  const clearVoiceTranscript = useCallback(async () => {
    setVoice((prev) => ({
      ...prev,
      partialText: "",
      finalText: "",
      error: "",
      phase: prev.listening ? prev.phase : "idle",
    }));
    await fetch(`${voiceUrl}/clear`, { method: "POST" });
  }, [voiceUrl]);

  const dismissVoiceCapture = useCallback(async () => {
    suppressVoiceEventsRef.current = true;
    setVoice(DEFAULT_VOICE_STATE);
    await fetch(`${voiceUrl}/stop`, { method: "POST" }).catch(() => undefined);
    await fetch(`${voiceUrl}/clear`, { method: "POST" }).catch(() => undefined);
  }, [voiceUrl]);

  useEffect(() => {
    connectChat();
    connectVoice();
    return () => {
      chatWsRef.current?.close();
      voiceWsRef.current?.close();
    };
  }, [connectChat, connectVoice]);

  return {
    messages,
    isStreaming,
    connected,
    voice,
    sendMessage,
    resetSession,
    startVoiceListening,
    stopVoiceListening,
    clearVoiceTranscript,
    dismissVoiceCapture,
  };
}
