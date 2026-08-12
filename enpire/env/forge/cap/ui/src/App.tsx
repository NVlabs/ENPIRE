/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Panel, Group, Separator } from "react-resizable-panels";
import { useRobotState } from "./hooks/useRobotState";
import { useChatStream } from "./hooks/useChatStream";
import * as api from "./api/client";
import type { AgentBackendOption, AgentBridgeConfig } from "./api/client";

import ModeSelector from "./components/ModeSelector";
import CodeEditor from "./components/CodeEditor";
import ApprovalPanel from "./components/ApprovalPanel";
import RobotDashboard from "./components/RobotDashboard";
import ErrorDashboard from "./components/ErrorDashboard";
import CameraFeed from "./components/CameraFeed";
import ActionLog from "./components/ActionLog";
import StdoutConsole from "./components/StdoutConsole";
import Controls from "./components/Controls";
import SkillVis from "./components/SkillVis";
import ScriptBrowser, { type ScriptBrowserHandle } from "./components/ScriptBrowser";
import ChatPanel from "./components/ChatPanel";
import LearnSkillPanel from "./components/LearnSkillPanel";
import RuntimeStreamsPanel from "./components/RuntimeStreamsPanel";
import ShortcutsOverlay from "./components/ShortcutsOverlay";
import TeleopOverlay from "./components/TeleopOverlay";

const FALLBACK_AGENT_OPTIONS: AgentBackendOption[] = [
  {
    backend: "claude_code",
    label: "Claude Code",
    models: ["claude-sonnet-4-20250514"],
    default_model: "claude-sonnet-4-20250514",
    reasoning_options: [],
    default_reasoning: null,
  },
  {
    backend: "openai_codex",
    label: "OpenAI Codex",
    models: [
      "gpt-5.4",
      "gpt-5.4-mini",
      "gpt-5.3-codex",
      "gpt-5.2",
      "gpt-5.1-codex-max",
      "gpt-5.1-codex-mini",
    ],
    default_model: "gpt-5.4",
    reasoning_options: ["low", "medium", "high", "xhigh"],
    default_reasoning: "low",
  },
];

function ResizeHandle({ orientation = "horizontal" }: { orientation?: "horizontal" | "vertical" }) {
  // For horizontal group orientation, the separator is a vertical bar (col-resize)
  // For vertical group orientation, the separator is a horizontal bar (row-resize)
  const isVerticalBar = orientation === "horizontal";
  return (
    <Separator
      className={`group relative flex items-center justify-center ${
        isVerticalBar
          ? "w-1.5 cursor-col-resize"
          : "h-1.5 cursor-row-resize"
      } bg-base-300 hover:bg-primary/20 active:bg-primary/30 transition-colors`}
    >
      <div
        className={`rounded-full bg-base-content/20 group-hover:bg-primary/50 transition-colors ${
          isVerticalBar ? "h-8 w-0.5" : "w-8 h-0.5"
        }`}
      />
    </Separator>
  );
}

export default function App() {
  const {
    connected,
    robotState,
    cameras,
    actionLog,
    stdoutLog,
    clearActionLog,
    resetSkillVisState,
    skillVisEpoch,
    latestError,
    proposedCode,
    setProposedCode,
    agentStatus,
    detectionDebug,
    segmentationDebug,
    vlmResults,
    graspViz,
    motionPlannerDebug,
    contactDetectDebug,
    debugMode: wsDebugMode,
    learnSkillStatus,
  } = useRobotState();

  const chat = useChatStream({
    onCodeBlock: (code) => {
      setEditorCode(code);
      setProposedCode(code);
    },
  });

  const [mode, setMode] = useState<"agent" | "oracle">("oracle");
  const [editorCode, setEditorCode] = useState("");
  const [theme, setTheme] = useState<"dark" | "light">("light");
  const [actionLogOpen, setActionLogOpen] = useState(false);
  const [scriptBrowserOpen, setScriptBrowserOpen] = useState(false);
  const scriptBrowserRef = useRef<ScriptBrowserHandle>(null);
  const [show3DView, setShow3DView] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [showErrorDashboard, setShowErrorDashboard] = useState(true);
  const [showRuntimeStreams, setShowRuntimeStreams] = useState(true);
  const [teleopOpen, setTeleopOpen] = useState(false);
  const [teleopMounted, setTeleopMounted] = useState(false);
  const [cameraStreaming, setCameraStreaming] = useState<Record<string, boolean>>({});
  const [agentOptions, setAgentOptions] = useState<AgentBackendOption[]>(FALLBACK_AGENT_OPTIONS);
  const [agentName, setAgentName] = useState("");
  const [agentConfig, setAgentConfig] = useState<AgentBridgeConfig>({
    backend: "openai_codex",
    model: "gpt-5.4",
    reasoning: "low",
  });

  useEffect(() => {
    api.getAgentOptions()
      .then((data) => {
        setAgentOptions(data.backends);
        setAgentName(data.agent_name || "");
        setAgentConfig({
          backend: data.current.backend,
          model: data.current.model ?? null,
          reasoning: data.current.reasoning ?? null,
        });
      })
      .catch(() => {
        setAgentOptions(FALLBACK_AGENT_OPTIONS);
      });
  }, []);

  const handleScriptLoad = useCallback((code: string) => {
    setEditorCode(code);
    setProposedCode(null);
  }, []);

  const handleToggleCameraStreaming = useCallback((name: string) => {
    setCameraStreaming((prev) => {
      const next = !prev[name];
      api.setCameraStreaming(name, next);
      return { ...prev, [name]: next };
    });
  }, []);

  // Auto-save editorCode to temp.py every 2s (skips save when content unchanged)
  const editorCodeRef = useRef(editorCode);
  const lastSavedRef = useRef<string | null>(null);
  useEffect(() => {
    editorCodeRef.current = editorCode;
  }, [editorCode]);
  useEffect(() => {
    const id = setInterval(() => {
      const code = editorCodeRef.current;
      if (code.trim() && code !== lastSavedRef.current) {
        lastSavedRef.current = code;
        api.saveScript("temp", code).catch(() => {});
      }
    }, 2000);
    return () => clearInterval(id);
  }, []);

  // Ctrl+B / Cmd+B to toggle script browser
  // Ctrl+L / Cmd+L to toggle agent/oracle mode
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "b") {
        e.preventDefault();
        setScriptBrowserOpen((prev) => {
          if (!prev) {
            // Opening — focus search after the drawer animates in
            setTimeout(() => scriptBrowserRef.current?.focusSearch(), 50);
          }
          return !prev;
        });
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
        e.preventDefault();
        setActionLogOpen((prev) => !prev);
      }
      if (e.key === "Escape" && teleopOpen) {
        e.preventDefault();
        setTeleopOpen(false);
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.altKey && e.key === "l") {
        e.preventDefault();
        setMode((prev) => {
          const next = prev === "agent" ? "oracle" : "agent";
          api.setMode(next);
          return next;
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [teleopOpen]);

  const handleDebugToggle = () => {
    api.setDebugMode(!wsDebugMode);
  };

  const displayCode = proposedCode ?? editorCode;
  const hasProposal = proposedCode !== null;

  const handleModeChange = (newMode: "agent" | "oracle") => {
    setMode(newMode);
    api.setMode(newMode);
  };

  const handleApprove = () => {
    api.approve();
    setProposedCode(null);
  };

  const handleReject = () => {
    api.reject();
    setProposedCode(null);
  };

  const handleEdit = () => {
    if (proposedCode) {
      setEditorCode(proposedCode);
      setProposedCode(null);
    }
  };

  const handleExecute = () => {
    resetSkillVisState();
    api.execute(editorCode);
  };

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <ShortcutsOverlay />
      {teleopMounted && <TeleopOverlay learnSkillStatus={learnSkillStatus} visible={teleopOpen} onClose={() => setTeleopOpen(false)} />}
      {/* Navbar */}
      <div className="navbar bg-base-200 px-4 shadow-lg">
        <div className="flex-1 gap-4">
          <span className="text-lg font-bold tracking-tight">CAP</span>
          <div className="badge badge-outline text-xs">
            {connected ? "Connected" : "Disconnected"}
          </div>
          <div className="badge badge-ghost text-xs">{agentStatus}</div>
        </div>
        <div className="flex items-center gap-4">
          <ModeSelector mode={mode} onChange={handleModeChange} />
          <Controls
            agentStatus={agentStatus}
            onStart={() => api.resume()}
            onPause={() => api.pause()}
            onStop={() => api.stop()}
            onEstop={() => api.estop()}
            onHome={() => api.home()}
          />
          <button
            className={`btn btn-sm ${teleopOpen ? "btn-accent" : "btn-ghost"}`}
            onClick={() => { setTeleopOpen(!teleopOpen); setTeleopMounted(true); }}
          >
            Teleop
          </button>
          <button
            className={`btn btn-sm ${wsDebugMode ? "btn-error" : "btn-ghost"}`}
            onClick={handleDebugToggle}
          >
            {wsDebugMode && <span className="badge badge-xs badge-warning mr-1">SIM</span>}
            {wsDebugMode ? "Debug ON" : "Debug"}
          </button>
          <label className="swap swap-rotate btn btn-ghost btn-sm">
            <input
              type="checkbox"
              checked={theme === "light"}
              onChange={toggleTheme}
            />
            <svg
              className="swap-on h-5 w-5 fill-current"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
            >
              <path d="M5.64,17l-.71.71a1,1,0,0,0,0,1.41,1,1,0,0,0,1.41,0l.71-.71A1,1,0,0,0,5.64,17ZM5,12a1,1,0,0,0-1-1H3a1,1,0,0,0,0,2H4A1,1,0,0,0,5,12Zm7-7a1,1,0,0,0,1-1V3a1,1,0,0,0-2,0V4A1,1,0,0,0,12,5ZM5.64,7.05a1,1,0,0,0,.7.29,1,1,0,0,0,.71-.29,1,1,0,0,0,0-1.41l-.71-.71A1,1,0,0,0,4.93,6.34Zm12,.29a1,1,0,0,0,.7-.29l.71-.71a1,1,0,1,0-1.41-1.41L17,5.64a1,1,0,0,0,0,1.41A1,1,0,0,0,17.66,7.34ZM21,11H20a1,1,0,0,0,0,2h1a1,1,0,0,0,0-2Zm-9,8a1,1,0,0,0-1,1v1a1,1,0,0,0,2,0V20A1,1,0,0,0,12,19ZM18.36,17A1,1,0,0,0,17,18.36l.71.71a1,1,0,0,0,1.41,0,1,1,0,0,0,0-1.41ZM12,6.5A5.5,5.5,0,1,0,17.5,12,5.51,5.51,0,0,0,12,6.5Zm0,9A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z" />
            </svg>
            <svg
              className="swap-off h-5 w-5 fill-current"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
            >
              <path d="M21.64,13a1,1,0,0,0-1.05-.14,8.05,8.05,0,0,1-3.37.73A8.15,8.15,0,0,1,9.08,5.49a8.59,8.59,0,0,1,.25-2A1,1,0,0,0,8,2.36,10.14,10.14,0,1,0,22,14.05,1,1,0,0,0,21.64,13Zm-9.5,6.69A8.14,8.14,0,0,1,7.08,5.22v.27A10.15,10.15,0,0,0,17.22,15.63a9.79,9.79,0,0,0,2.1-.22A8.11,8.11,0,0,1,12.14,19.73Z" />
            </svg>
          </label>
        </div>
      </div>


      {/* Main content row: [action log drawer] [toggle] [panels] */}
      <div className="flex flex-1 min-h-0 relative">
        {/* Action Log Drawer — full-height left column */}
        <div
          className={`h-full border-r border-base-300 bg-base-100 transition-all duration-300 ease-in-out overflow-hidden shrink-0 ${
            actionLogOpen ? "w-72" : "w-0 border-r-0"
          }`}
        >
          <div className="h-full w-72">
            <div className="flex h-full flex-col">
              <div className="min-h-0 flex-[3]">
                <ActionLog entries={actionLog} onClear={clearActionLog} />
              </div>
              <div className="min-h-0 flex-[2]">
                <StdoutConsole entries={stdoutLog} />
              </div>
            </div>
          </div>
        </div>

        {/* Drawer toggle button — fixed strip on the left edge */}
        <button
          className="h-full w-6 shrink-0 flex items-center justify-center bg-base-200 hover:bg-base-300 active:bg-primary/20 transition-colors cursor-pointer border-r border-base-300"
          onClick={() => setActionLogOpen(!actionLogOpen)}
          title={actionLogOpen ? "Hide Action Log" : "Show Action Log"}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className={`w-3.5 h-3.5 text-base-content/50 transition-transform duration-300 ${
              actionLogOpen ? "rotate-180" : ""
            }`}
          >
            <path
              fillRule="evenodd"
              d="M8.22 5.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 10 8.22 6.28a.75.75 0 0 1 0-1.06Z"
              clipRule="evenodd"
            />
          </svg>
        </button>

        {/* Main panels area */}
        <div className="flex-1 min-w-0 min-h-0">
          <Group orientation="horizontal">
            {/* Left Panel */}
            <Panel defaultSize={50} minSize={25}>
              <div className="flex h-full flex-col">
                {/* Agent mode: Chat + Editor split */}
                {mode === "agent" ? (
                  <Group orientation="vertical">
                    <Panel defaultSize={55} minSize={20}>
                      <ChatPanel
                        messages={chat.messages}
                        isStreaming={chat.isStreaming}
                        connected={chat.connected}
                        voice={chat.voice}
                        agentName={agentName}
                        onSend={chat.sendMessage}
                        onReset={chat.resetSession}
                        onEvolve={(config) => { void api.evolveChat(config ?? agentConfig); }}
                        agentConfig={agentConfig}
                        backendOptions={agentOptions}
                        onAgentConfigChange={setAgentConfig}
                        onVoiceStart={chat.startVoiceListening}
                        onVoiceStop={chat.stopVoiceListening}
                        onVoiceDismiss={chat.dismissVoiceCapture}
                        onVoiceClear={chat.clearVoiceTranscript}
                      />
                    </Panel>
                    <ResizeHandle orientation="vertical" />
                    <Panel defaultSize={45} minSize={15}>
                      <div className="flex h-full flex-col">
                        <ApprovalPanel
                          visible={hasProposal}
                          onApprove={handleApprove}
                          onReject={handleReject}
                          onEdit={handleEdit}
                        />
                        <div className="flex-1 min-h-0 flex flex-col">
                          <div className="flex-1 min-h-0 overflow-hidden">
                            <CodeEditor
                              code={displayCode}
                              onChange={setEditorCode}
                              readOnly={hasProposal}
                            />
                          </div>
                          {!hasProposal && editorCode.trim() && (
                            <div className="border-t border-base-300 p-2 shrink-0">
                              <button
                                className="btn btn-primary btn-sm w-full"
                                onClick={handleExecute}
                                disabled={!editorCode.trim() || agentStatus === "executing"}
                              >
                                Execute Code
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </Panel>
                  </Group>
                ) : (
                  /* Oracle mode: full editor */
                  <div className="flex h-full flex-col">
                    <ApprovalPanel
                      visible={hasProposal}
                      onApprove={handleApprove}
                      onReject={handleReject}
                      onEdit={handleEdit}
                    />
                    <div className="flex-1 min-h-0 flex flex-col">
                      <div className="flex-1 min-h-0 overflow-hidden">
                        <CodeEditor
                          code={displayCode}
                          onChange={setEditorCode}
                          readOnly={hasProposal}
                          theme={theme}
                        />
                      </div>
                      {!hasProposal && (
                        <div className="border-t border-base-300 p-2 shrink-0">
                          <button
                            className="btn btn-primary btn-sm w-full"
                            onClick={handleExecute}
                            disabled={!editorCode.trim()}
                          >
                            Execute Code
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </Panel>

            <ResizeHandle orientation="horizontal" />

            {/* Right Panel */}
            <Panel defaultSize={50} minSize={25}>
              <Group orientation="vertical">
                {/* Camera feeds */}
                <Panel defaultSize={25} minSize={10}>
                  <div className="h-full overflow-y-auto p-2">
                    <CameraFeed cameras={cameras} streaming={cameraStreaming} onToggleStreaming={handleToggleCameraStreaming} />
                  </div>
                </Panel>
                <ResizeHandle orientation="vertical" />
                {/* 3D View (toggleable) + Skill Vis */}
                <Panel defaultSize={40} minSize={15}>
                  {show3DView ? (
                    <Group orientation="horizontal">
                      <Panel defaultSize={50} minSize={20}>
                        <div className="h-full p-2">
                          <div className="card bg-base-200 shadow-sm h-full">
                            <div className="card-body p-2 h-full">
                              <div className="flex items-center justify-between shrink-0">
                                <h3 className="card-title text-sm">3D View</h3>
                                <button
                                  className="btn btn-ghost btn-xs"
                                  onClick={() => setShow3DView(false)}
                                  title="Hide 3D View"
                                >
                                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
                                    <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                                  </svg>
                                </button>
                              </div>
                              <div className="relative flex-1 min-h-0">
                                <iframe
                                  src={`http://${window.location.hostname}:8085`}
                                  className="absolute inset-0 h-full w-full rounded border-0"
                                  title="Viser 3D View"
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      </Panel>
                      <ResizeHandle orientation="horizontal" />
                      <Panel defaultSize={50} minSize={20}>
                        <div className="h-full p-2">
                          <SkillVis key={skillVisEpoch} detectionData={detectionDebug} segmentationData={segmentationDebug} vlmResults={vlmResults} graspViz={graspViz} motionPlannerDebug={motionPlannerDebug} contactDetectDebug={contactDetectDebug} robotState={robotState} />
                        </div>
                      </Panel>
                    </Group>
                  ) : (
                    <div className="h-full p-2">
                      <SkillVis
                        key={skillVisEpoch}
                        detectionData={detectionDebug}
                        segmentationData={segmentationDebug}
                        vlmResults={vlmResults}
                        graspViz={graspViz}
                        motionPlannerDebug={motionPlannerDebug}
                        contactDetectDebug={contactDetectDebug}
                        robotState={robotState}
                        extraHeaderButton={
                          <button
                            className="btn btn-outline btn-xs"
                            onClick={() => setShow3DView(true)}
                            title="Show 3D View"
                          >
                            Show 3D
                          </button>
                        }
                      />
                    </div>
                  )}
                </Panel>
                <ResizeHandle orientation="vertical" />
                {/* Learn Skill Panel + Robot Dashboard */}
                <Panel defaultSize={35} minSize={10}>
                  <div className="flex h-full flex-col">
                    <div className="p-2 shrink-0">
                      <LearnSkillPanel status={learnSkillStatus} />
                    </div>
                    <div className="border-t border-base-300 flex-1 min-h-0 overflow-y-auto p-2 space-y-2">
                      <ErrorDashboard error={latestError} open={showErrorDashboard} onToggle={() => setShowErrorDashboard(!showErrorDashboard)} />
                      <RobotDashboard state={robotState} showDashboard={showDashboard} onToggle={() => setShowDashboard(!showDashboard)} />
                      <RuntimeStreamsPanel
                        entries={actionLog}
                        open={showRuntimeStreams}
                        onToggle={() => setShowRuntimeStreams(!showRuntimeStreams)}
                      />
                    </div>
                  </div>
                </Panel>
              </Group>
            </Panel>
          </Group>
        </div>

        {/* Script Browser toggle strip — right edge */}
        <button
          className="h-full w-6 shrink-0 flex items-center justify-center bg-base-200 hover:bg-base-300 active:bg-primary/20 transition-colors cursor-pointer border-l border-base-300"
          onClick={() => setScriptBrowserOpen(!scriptBrowserOpen)}
          title={scriptBrowserOpen ? "Hide Scripts" : "Show Scripts"}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className={`w-3.5 h-3.5 text-base-content/50 transition-transform duration-300 ${
              scriptBrowserOpen ? "" : "rotate-180"
            }`}
          >
            <path
              fillRule="evenodd"
              d="M8.22 5.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 10 8.22 6.28a.75.75 0 0 1 0-1.06Z"
              clipRule="evenodd"
            />
          </svg>
        </button>

        {/* Script Browser Drawer — full-height right column */}
        <div
          className={`h-full border-l border-base-300 bg-base-100 transition-all duration-300 ease-in-out overflow-hidden shrink-0 ${
            scriptBrowserOpen ? "w-72" : "w-0 border-l-0"
          }`}
        >
          <div className="h-full w-72">
            <ScriptBrowser
              ref={scriptBrowserRef}
              currentCode={editorCode}
              onLoad={handleScriptLoad}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
