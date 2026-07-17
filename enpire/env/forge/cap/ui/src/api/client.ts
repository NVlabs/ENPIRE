const BASE = "/api";
const REQUEST_TIMEOUT_MS = 30_000;

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API error ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  } finally {
    clearTimeout(timer);
  }
}

export function submitTask(description: string) {
  return request<{ task_id: string }>("/task", {
    method: "POST",
    body: JSON.stringify({ description }),
  });
}

export function approve() {
  return request<{ ok: boolean }>("/approve", { method: "POST" });
}

export function reject(feedback?: string) {
  return request<{ ok: boolean }>("/reject", {
    method: "POST",
    body: JSON.stringify({ feedback }),
  });
}

export function execute(code: string) {
  return request<{ ok: boolean }>("/execute", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}

export function setMode(mode: "agent" | "oracle") {
  return request<{ ok: boolean }>("/mode", {
    method: "POST",
    body: JSON.stringify({ mode }),
  });
}

export function estop() {
  return request<{ ok: boolean }>("/estop", { method: "POST" });
}

export function pause() {
  return request<{ ok: boolean }>("/pause", { method: "POST" });
}

export function resume() {
  return request<{ ok: boolean }>("/resume", { method: "POST" });
}

export function stop() {
  return request<{ ok: boolean }>("/stop", { method: "POST" });
}

export function home() {
  return request<{ ok: boolean }>("/home", { method: "POST" });
}

// Script management

export interface ScriptInfo {
  name: string;
  size: number;
  modified: string;
}

export function listScripts() {
  return request<ScriptInfo[]>("/scripts");
}

export function saveScript(name: string, code: string) {
  return request<{ ok: boolean }>("/scripts", {
    method: "POST",
    body: JSON.stringify({ name, code }),
  });
}

export function loadScript(name: string) {
  return request<{ ok: boolean; code: string }>(`/scripts/${encodeURIComponent(name)}`);
}

export function deleteScript(name: string) {
  return request<{ ok: boolean }>(`/scripts/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function renameScript(name: string, newName: string) {
  return request<{ ok: boolean }>(`/scripts/${encodeURIComponent(name)}`, {
    method: "PATCH",
    body: JSON.stringify({ new_name: newName }),
  });
}

// Debug mode

export function setDebugMode(enabled: boolean) {
  return request<{ ok: boolean }>("/debug_mode", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

// Camera list + streaming

export function listCameras() {
  return request<{ ok: boolean; cameras: string[] }>("/cameras");
}

export function setCameraStreaming(camera: string, enabled: boolean) {
  return request<{ ok: boolean }>("/camera_streaming", {
    method: "POST",
    body: JSON.stringify({ camera, enabled }),
  });
}

export function saveCameras() {
  return request<{ ok: boolean }>("/save_cameras", { method: "POST" });
}

// Bridge API (generic agent bridge)

const BRIDGE_BASE = "/bridge-api";

export interface AgentBackendOption {
  backend: string;
  label: string;
  models: string[];
  default_model: string | null;
  reasoning_options: string[];
  default_reasoning: string | null;
}

export interface AgentBridgeConfig {
  backend?: string;
  model?: string | null;
  reasoning?: string | null;
}

export interface AgentOptionsResponse {
  agent_name: string;
  current: {
    backend: string;
    model?: string | null;
    reasoning?: string | null;
  };
  backends: AgentBackendOption[];
}

export interface BridgeVoiceStatus {
  enabled: boolean;
  speaking: boolean;
  last_spoken_text: string;
  last_error?: string | null;
}

export function getAgentOptions() {
  return fetch(`${BRIDGE_BASE}/agent/options`).then(
    (r) => r.json() as Promise<AgentOptionsResponse>,
  );
}

export function sendChatMessage(message: string, config: AgentBridgeConfig = {}) {
  return fetch(`${BRIDGE_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...config }),
  }).then((r) => r.json() as Promise<{ ok: boolean }>);
}

export function resetChat(config: AgentBridgeConfig = {}) {
  return fetch(`${BRIDGE_BASE}/chat/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  }).then(
    (r) => r.json() as Promise<{ ok: boolean }>,
  );
}

export function evolveChat(config: AgentBridgeConfig = {}) {
  return fetch(`${BRIDGE_BASE}/evolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  }).then((r) => r.json() as Promise<{ ok: boolean }>);
}

export function voiceStatus() {
  return fetch(`${BRIDGE_BASE}/voice/status`).then(
    (r) => r.json() as Promise<BridgeVoiceStatus>,
  );
}

export function setVoiceEnabled(enabled: boolean) {
  return fetch(`${BRIDGE_BASE}/voice/enabled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  }).then((r) => r.json() as Promise<{ ok: boolean }>);
}

export function voiceTest(text = "Hello World") {
  return fetch(`${BRIDGE_BASE}/voice/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).then((r) => r.json() as Promise<{ ok: boolean }>);
}

export function voiceSpeak(text: string) {
  return fetch(`${BRIDGE_BASE}/voice/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).then((r) => r.json() as Promise<{ ok: boolean }>);
}

export function voiceStop() {
  return fetch(`${BRIDGE_BASE}/voice/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  }).then((r) => r.json() as Promise<{ ok: boolean }>);
}
