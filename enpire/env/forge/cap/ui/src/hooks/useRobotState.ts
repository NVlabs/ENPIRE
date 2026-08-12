/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useEffect, useState } from "react";
import { useWebSocket } from "./useWebSocket";

export interface EEPose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

export interface ArmState {
  joint_positions: number[];
  gripper: number;
  ee_pose: EEPose;
}

export interface RobotState {
  arms: Record<string, ArmState>;
  timestamp: number;
}

/** Helper to get an arm by name, returning default zeros if missing. */
export function getArm(state: RobotState, name: string): ArmState {
  return state.arms[name] ?? defaultArm;
}

export interface CameraFrame {
  camera: string;
  image: string; // base64
  detections?: Detection[];
}

export interface Detection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
  position_3d?: [number, number, number];
  quaternion_xyzw?: [number, number, number, number];
  bbox_3d_projected?: [number, number][]; // 8 projected corners of oriented 3D bbox
  pose_axes?: {
    origin: [number, number];
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  is_grasp?: boolean;
  grasp_axes?: {
    origin: [number, number];
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
}

export interface ActionLogEntry {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  result?: unknown;
  stdout?: string;
  stderr?: string;
  status: "running" | "success" | "error";
  timestamp: string;
  parent_id?: string;
  node_type?: string;
}

export interface StdoutEntry {
  id: string;
  text: string;
  timestamp: string;
  tool?: string;
}

const defaultArm: ArmState = {
  joint_positions: [0, 0, 0, 0, 0, 0],
  gripper: 0,
  ee_pose: { position: [0, 0, 0], quaternion: [0, 0, 0, 1] },
};

const defaultState: RobotState = {
  arms: {},
  timestamp: 0,
};

export interface DetectionDebugData {
  image: string;
  camera: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
}

export interface VlmResult {
  prompt: string;
  backend: string;
  camera: string;
  model?: string;
  response: string;
  timestamp: number; // added client-side
}

export interface SegmentationDebugData {
  image: string; // base64 JPEG
  camera: string;
  imageWidth: number;
  imageHeight: number;
  mask_overlay: string; // base64 PNG (RGBA)
  query: string;
  score: number;
  mask_area: number;
  bbox_xywh: [number, number, number, number];
  hu_dist?: number | null;
  passed?: boolean | null;
  is_reference?: boolean;
  media?: string;
}

export interface GraspVizPayload {
  overlay_b64?: string;
  native_viz_b64?: string;
  camera?: string;
  object_name?: string;
  n_grasps?: number;
  status?: string;
  backend?: string;
  best_score?: number;
  planner_z_floor_m?: number;
  n_planner_z_clipped?: number;
  selected_side?: string;
  selected_rank?: number | null;
  selection_status?: string;
  selection_threshold_m?: number;
  planner_thresholds_m?: number[];
  attempt?: number;
  grasp_selection_state?: string;
  status_reason?: string;
  grasps?: GraspDebugRow[];
}

export interface GraspDebugRow {
  rank: number;
  score?: number;
  width?: number;
  thumbnail_b64?: string | null;
  raw_xyz?: number[];
  raw_rpy?: number[];
  planner_xyz?: number[];
  planner_rpy?: number[];
  status?: string;
  status_reason?: string | null;
  selected?: boolean;
  side?: string;
  motionplanner_side?: string;
  motionplanner_status?: string;
  motionplanner_preview_only?: boolean;
  motionplanner_reason?: string | null;
  final_pos_error_m?: number;
  final_rot_error_deg?: number;
  final_pose_error?: number;
  trajectory_steps?: number;
}

export interface MotionPlannerDebug {
  tool?: string;
  planner_backend?: string;
  side?: string;
  status?: string;
  reason?: string;
  error?: string | null;
  preview_only?: boolean;
  planning_speed?: number;
  ik_error_m?: number;
  final_pos_error_m?: number;
  final_rot_error_deg?: number;
  final_pose_error?: number;
  trajectory_steps?: number;
  executed?: boolean;
  // Dynamic arm targets (keyed by arm name)
  targets?: Record<string, { pos?: number[]; rpy?: number[] }>;
  // Legacy left/right targets (backward compat)
  left_target_pos?: number[];
  left_target_rpy?: number[];
  right_target_pos?: number[];
  right_target_rpy?: number[];
}

export interface ContactDetectDebug {
  side?: string;
  object_name?: string;
  status?: string;
  success?: boolean;
  gripper_width?: number;
  threshold_low?: number | null;
  threshold_high?: number | null;
  message?: string;
  timestamp: number;
}

export interface RuntimeErrorState {
  message: string;
  source?: string;
  detail?: string;
  timestamp: number;
}

export interface LearnSkillStatus {
  active: boolean;
  episode?: number;
  step?: number;
  max_steps?: number;
  reward?: number;
  cumulative_reward?: number;
  action_source?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function payloadDetail(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isRobotStatePayload(value: unknown): value is RobotState {
  return isRecord(value) && isRecord((value as Record<string, unknown>).arms);
}

function isCameraFramePayload(value: unknown): value is CameraFrame {
  return isRecord(value) && typeof value.camera === "string" && typeof value.image === "string";
}

function isActionLogEntryPayload(value: unknown): value is ActionLogEntry {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.tool === "string" &&
    typeof value.timestamp === "string" &&
    typeof value.status === "string"
  );
}

function isStdoutStreamPayload(value: unknown): value is { text?: string; timestamp?: string } {
  return isRecord(value);
}

function isCodeProposalPayload(value: unknown): value is { code: string } {
  return isRecord(value) && typeof value.code === "string";
}

function isStatusChangePayload(value: unknown): value is { status: string; debug_mode?: boolean } {
  return isRecord(value) && typeof value.status === "string";
}

function isErrorPayload(value: unknown): value is { message?: string; source?: string; detail?: string } {
  return isRecord(value);
}

function isDetectionDebugPayload(value: unknown): value is DetectionDebugData {
  return isRecord(value) && typeof value.camera === "string" && typeof value.image === "string";
}

function isSegmentationDebugPayload(value: unknown): value is SegmentationDebugData {
  return (
    isRecord(value) &&
    typeof value.camera === "string" &&
    typeof value.image === "string" &&
    typeof value.mask_overlay === "string"
  );
}

function isDebugModeChangePayload(value: unknown): value is { enabled: boolean } {
  return isRecord(value) && typeof value.enabled === "boolean";
}

function isVlmResultPayload(
  value: unknown
): value is { prompt: string; backend: string; camera: string; model?: string; response: string } {
  return (
    isRecord(value) &&
    typeof value.prompt === "string" &&
    typeof value.backend === "string" &&
    typeof value.camera === "string" &&
    typeof value.response === "string"
  );
}

function isGraspVizPayload(value: unknown): value is GraspVizPayload {
  return isRecord(value);
}

function isMotionPlannerDebugPayload(value: unknown): value is MotionPlannerDebug {
  return isRecord(value);
}

function isContactDetectPayload(value: unknown): value is Omit<ContactDetectDebug, "timestamp"> {
  return isRecord(value);
}

export function useRobotState() {
  const ws = useWebSocket();
  const [robotState, setRobotState] = useState<RobotState>(defaultState);
  const [cameras, setCameras] = useState<Record<string, CameraFrame>>({});
  const [actionLog, setActionLog] = useState<ActionLogEntry[]>([]);
  const [stdoutLog, setStdoutLog] = useState<StdoutEntry[]>([]);
  const [proposedCode, setProposedCode] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<string>("idle");
  const [detectionDebug, setDetectionDebug] = useState<DetectionDebugData | null>(null);
  const [segmentationDebug, setSegmentationDebug] = useState<SegmentationDebugData | null>(null);
  const [debugMode, setDebugMode] = useState(false);
  const [vlmResults, setVlmResults] = useState<VlmResult[]>([]);
  const [learnSkillStatus, setLearnSkillStatus] = useState<LearnSkillStatus>({ active: false });
  const [graspViz, setGraspViz] = useState<GraspVizPayload | null>(null);
  const [motionPlannerDebug, setMotionPlannerDebug] = useState<MotionPlannerDebug | null>(null);
  const [contactDetectDebug, setContactDetectDebug] = useState<ContactDetectDebug | null>(null);
  const [skillVisEpoch, setSkillVisEpoch] = useState(0);
  const [latestError, setLatestError] = useState<RuntimeErrorState | null>(null);

  const summarizeForActionLog = (payload: unknown): unknown => {
    if (Array.isArray(payload)) {
      return payload.slice(0, 20).map((v) => summarizeForActionLog(v));
    }
    if (payload && typeof payload === "object") {
      const obj = payload as Record<string, unknown>;
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(obj)) {
        if (k === "image" || k === "mask_overlay" || k === "overlay_b64" || k === "native_viz_b64") {
          out[k] = "[shown in Skill Vis]";
        } else {
          out[k] = summarizeForActionLog(v);
        }
      }
      return out;
    }
    return payload;
  };

  const pushSkillLog = (tool: string, result: unknown, status: "success" | "error" = "success") => {
    setActionLog((prev) => [
      ...prev,
      {
        id: `${tool}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        tool,
        args: {},
        result: summarizeForActionLog(result),
        status,
        timestamp: new Date().toISOString(),
      },
    ]);
  };

  const mergeGraspRows = (
    prevRows: GraspDebugRow[] | undefined,
    nextRows: GraspDebugRow[] | undefined,
  ): GraspDebugRow[] | undefined => {
    if (!nextRows) return prevRows;
    const merged = new Map<number, GraspDebugRow>();
    for (const row of prevRows ?? []) {
      if (typeof row.rank === "number") merged.set(row.rank, row);
    }
    for (const row of nextRows) {
      if (typeof row.rank !== "number") continue;
      merged.set(row.rank, { ...(merged.get(row.rank) ?? {}), ...row });
    }
    return Array.from(merged.values()).sort((a, b) => a.rank - b.rank);
  };

  const mergeGraspViz = (
    prev: GraspVizPayload | null,
    next: GraspVizPayload,
  ): GraspVizPayload => {
    const sameGraspRun =
      prev &&
      prev.backend === next.backend &&
      prev.object_name === next.object_name &&
      prev.camera === next.camera;
    return {
      ...(sameGraspRun ? prev : {}),
      ...next,
      grasps: sameGraspRun ? mergeGraspRows(prev?.grasps, next.grasps) : next.grasps,
    };
  };

  const applyExecutionLogToAnygraspViz = (
    prev: GraspVizPayload | null,
    entry: ActionLogEntry,
  ): GraspVizPayload | null => {
    if (!prev || prev.backend !== "anygrasp" || !prev.grasps?.length) return prev;
    const text = typeof entry.result === "string" ? entry.result : "";
    if (!text) return prev;

    const rows = prev.grasps.map((row) => ({ ...row }));
    const byRank = new Map<number, GraspDebugRow>();
    for (const row of rows) byRank.set(row.rank, row);
    let changed = false;
    let selectedRank = typeof prev.selected_rank === "number" ? prev.selected_rank : null;
    let selectedSide = prev.selected_side;
    let usedThreshold =
      typeof prev.selection_threshold_m === "number" ? prev.selection_threshold_m : undefined;
    let selectionStatus = prev.selection_status;

    const previewRegex =
      /Planner preview grasp (\d+)\/\d+ \[(left|right)\]: score=([0-9.]+), .*?pos_err=([0-9.]+)m, rot_err=([0-9.]+)deg, pose_err=([0-9.]+), traj=(\d+)/g;
    for (const match of text.matchAll(previewRegex)) {
      const rank = Number(match[1]);
      const row = byRank.get(rank);
      if (!row) continue;
      row.motionplanner_side = match[2];
      row.motionplanner_preview_only = true;
      row.score = Number.isFinite(Number(match[3])) ? Number(match[3]) : row.score;
      row.final_pos_error_m = Number(match[4]);
      row.final_rot_error_deg = Number(match[5]);
      row.final_pose_error = Number(match[6]);
      row.trajectory_steps = Number(match[7]);
      row.status = "preview_ok";
      row.status_reason = "Planner preview succeeded.";
      changed = true;
    }

    const previewFailRegex = /Planner preview grasp (\d+)\/\d+ \[(left|right)\] failed: (.+)/g;
    for (const match of text.matchAll(previewFailRegex)) {
      const rank = Number(match[1]);
      const row = byRank.get(rank);
      if (!row) continue;
      row.motionplanner_side = match[2];
      row.motionplanner_preview_only = true;
      row.status = "planner_preview_failed";
      row.status_reason = match[3];
      changed = true;
    }

    const thresholdMatch = text.match(/Using planner candidates with pos_err <= ([0-9.]+)m/);
    if (thresholdMatch) {
      usedThreshold = Number(thresholdMatch[1]);
      selectionStatus = "thresholded";
      for (const row of rows) {
        if (typeof row.final_pos_error_m !== "number") continue;
        if (row.final_pos_error_m > usedThreshold) {
          row.status = "discarded_pos_err_threshold";
          row.status_reason =
            `Discarded because planner position error ${row.final_pos_error_m.toFixed(4)}m ` +
            `exceeded threshold ${usedThreshold.toFixed(3)}m.`;
          changed = true;
        } else if (row.status === "preview_ok") {
          row.status = "preview_ok_threshold_pass";
          row.status_reason = `Passed planner threshold ${usedThreshold.toFixed(3)}m.`;
          changed = true;
        }
      }
    } else if (text.includes("no AnyGrasp candidates met the preferred position-error thresholds")) {
      selectionStatus = "no_threshold_match";
      for (const row of rows) {
        if (row.status === "preview_ok") {
          row.status = "preview_ok_no_threshold_match";
          row.status_reason =
            "Planner preview succeeded, but no candidate met the preferred " +
            "position-error thresholds.";
          changed = true;
        }
      }
    }

    const selectedMatch = text.match(
      /Selected grasp via AnyGrasp XYZ\+RPY \+ freespace preview: rank=(\d+), side=(left|right),/
    );
    if (selectedMatch) {
      selectedRank = Number(selectedMatch[1]);
      selectedSide = selectedMatch[2];
      selectionStatus = "selected";
      for (const row of rows) {
        if (row.rank === selectedRank) {
          row.selected = true;
          row.side = selectedSide;
          row.status = "selected";
          row.status_reason = "Selected as the best planner-feasible candidate.";
        } else if (
          row.status === "preview_ok" ||
          row.status === "preview_ok_threshold_pass" ||
          row.status === "preview_ok_no_threshold_match"
        ) {
          row.selected = false;
          row.status = "discarded_after_ranking";
          row.status_reason = "Passed planner thresholding but lost final ranking to a better candidate.";
        }
        changed = true;
      }
      for (const row of rows) {
        if (typeof row.final_pos_error_m !== "number" && row.rank !== selectedRank && !row.status) {
          row.status = "not_previewed_top_k";
          row.status_reason = "Not evaluated by planner preview.";
          changed = true;
        }
      }
    }

    if (text.includes("No planner-feasible top-camera grasp found in top candidates")) {
      selectionStatus = "no_planner_feasible";
      changed = true;
    }

    if (text.includes("Closed gripper (grasp-detection check disabled)") && typeof selectedRank === "number") {
      const row = byRank.get(selectedRank);
      if (row) {
        row.status = "executed";
        row.status_reason = "Robot reached the selected grasp pose and closed the gripper.";
        changed = true;
      }
    }

    if (!changed) return prev;
    return {
      ...prev,
      grasps: rows,
      selected_rank: selectedRank,
      selected_side: selectedSide,
      selection_threshold_m: usedThreshold,
      selection_status: selectionStatus,
    };
  };

  useEffect(() => {
    const unsubs = [
      ws.subscribe("state_update", (data) => {
        if (!isRobotStatePayload(data)) {
          setLatestError({
            message: "Malformed state_update payload",
            source: "state_update",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        setRobotState(data);
      }),
      ws.subscribe("camera_frame", (data) => {
        if (!isCameraFramePayload(data)) {
          setLatestError({
            message: "Malformed camera_frame payload",
            source: "camera_frame",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const frame = data;
        setCameras((prev) => ({ ...prev, [frame.camera]: frame }));
      }),
      ws.subscribe("execution_log", (data) => {
        if (!isActionLogEntryPayload(data)) {
          setLatestError({
            message: "Malformed execution_log payload",
            source: "execution_log",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const entry = data;
        setActionLog((prev) => {
          const idx = prev.findIndex((e) => e.id === entry.id);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = entry;
            return updated;
          }
          return [...prev, entry];
        });
        setGraspViz((prev) => applyExecutionLogToAnygraspViz(prev, entry));
        const resultText = typeof entry.result === "string" ? entry.result.trim() : "";
        if (
          resultText &&
          entry.args &&
          typeof entry.args === "object" &&
          "code" in entry.args &&
          !/^\d+(\.\d+)?ms$/.test(resultText)
        ) {
          setStdoutLog((prev) => [
            ...prev,
            {
              id: entry.id,
              text: resultText,
              timestamp: entry.timestamp,
              tool: entry.tool,
            },
          ]);
        }
        setGraspViz((prev) => applyExecutionLogToAnygraspViz(prev, entry));
      }),
      ws.subscribe("stdout_stream", (data) => {
        if (!isStdoutStreamPayload(data)) {
          setLatestError({
            message: "Malformed stdout_stream payload",
            source: "stdout_stream",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = data;
        const text = String(payload.text ?? "");
        if (!text) return;
        setStdoutLog((prev) => [
          ...prev,
          {
            id: `stdout-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            text,
            timestamp: payload.timestamp ?? new Date().toISOString(),
          },
        ]);
      }),
      ws.subscribe("code_proposal", (data) => {
        if (!isCodeProposalPayload(data)) {
          setLatestError({
            message: "Malformed code_proposal payload",
            source: "code_proposal",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        setProposedCode(data.code);
      }),
      ws.subscribe("status_change", (data) => {
        if (!isStatusChangePayload(data)) {
          setLatestError({
            message: "Malformed status_change payload",
            source: "status_change",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        setAgentStatus(data.status);
        if (data.debug_mode !== undefined) {
          setDebugMode(Boolean(data.debug_mode));
        }
      }),
      ws.subscribe("error", (data) => {
        if (!isErrorPayload(data)) {
          setLatestError({
            message: "Malformed error payload",
            source: "error",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = data;
        setLatestError({
          message: String(payload.message ?? payload.detail ?? "Unknown error"),
          source: typeof payload.source === "string" ? payload.source : undefined,
          detail: typeof payload.detail === "string" ? payload.detail : undefined,
          timestamp: Date.now(),
        });
      }),
      ws.subscribe("detection_debug", (data) => {
        if (!isDetectionDebugPayload(data)) {
          setLatestError({
            message: "Malformed detection_debug payload",
            source: "detection_debug",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = data;
        setDetectionDebug(payload);
        pushSkillLog("detection", payload);
      }),
      ws.subscribe("segmentation_debug", (data) => {
        if (!isSegmentationDebugPayload(data)) {
          setLatestError({
            message: "Malformed segmentation_debug payload",
            source: "segmentation_debug",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = data;
        setSegmentationDebug(payload);
        pushSkillLog("segmentation", payload);
      }),
      ws.subscribe("debug_mode_change", (data) => {
        if (!isDebugModeChangePayload(data)) {
          setLatestError({
            message: "Malformed debug_mode_change payload",
            source: "debug_mode_change",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        setDebugMode(data.enabled);
      }),
      ws.subscribe("vlm_result", (data) => {
        if (!isVlmResultPayload(data)) {
          setLatestError({
            message: "Malformed vlm_result payload",
            source: "vlm_result",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const raw = data;
        setVlmResults((prev) => {
          // Deduplicate: skip if last entry has same prompt+response+backend within 1s
          const now = Date.now();
          const last = prev[prev.length - 1];
          if (
            last &&
            now - last.timestamp < 1000 &&
            last.prompt === raw.prompt &&
            last.response === raw.response &&
            last.backend === raw.backend
          ) {
            return prev;
          }
          return [...prev, { ...raw, timestamp: now }];
        });
        pushSkillLog("vlm", raw);
      }),
      ws.subscribe("grasp_viz", (data) => {
        if (!isGraspVizPayload(data)) {
          setLatestError({
            message: "Malformed grasp_viz payload",
            source: "grasp_viz",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = data;
        setGraspViz((prev) => mergeGraspViz(prev, payload));
        pushSkillLog(
          payload.backend ?? "grasp_viz",
          payload,
          payload.status === "ok" ? "success" : "error",
        );
      }),
      ws.subscribe("motion_planner_debug", (data) => {
        if (!isMotionPlannerDebugPayload(data)) {
          setLatestError({
            message: "Malformed motion_planner_debug payload",
            source: "motion_planner_debug",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = data;
        setMotionPlannerDebug(payload);
        pushSkillLog("motion_planner", payload, payload.error ? "error" : "success");
      }),
      ws.subscribe("contact_detect", (data) => {
        if (!isContactDetectPayload(data)) {
          setLatestError({
            message: "Malformed contact_detect payload",
            source: "contact_detect",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const payload = {
          ...data,
          timestamp: Date.now(),
        };
        setContactDetectDebug(payload);
        pushSkillLog("contact_detect", payload, payload.success === false ? "error" : "success");
      }),
      ws.subscribe("learn_skill_update", (data) => {
        if (!isRecord(data)) {
          setLatestError({
            message: "Malformed learn_skill_update payload",
            source: "learn_skill_update",
            detail: payloadDetail(data),
            timestamp: Date.now(),
          });
          return;
        }
        const raw = data;
        setLearnSkillStatus({
          active: Boolean(raw.active),
          episode: raw.episode != null ? Number(raw.episode) : undefined,
          step: raw.step != null ? Number(raw.step) : undefined,
          max_steps: raw.max_steps != null ? Number(raw.max_steps) : undefined,
          reward: raw.reward != null ? Number(raw.reward) : undefined,
          cumulative_reward: raw.cumulative_reward != null ? Number(raw.cumulative_reward) : undefined,
          action_source: raw.action_source != null ? String(raw.action_source) : undefined,
        });
      }),
    ];

    return () => unsubs.forEach((unsub) => unsub());
  }, [ws]);

  const clearActionLog = () => {
    setActionLog([]);
    setStdoutLog([]);
  };

  const resetSkillVisState = () => {
    setDetectionDebug(null);
    setSegmentationDebug(null);
    setVlmResults([]);
    setGraspViz(null);
    setMotionPlannerDebug(null);
    setContactDetectDebug(null);
    setActionLog([]);
    setStdoutLog([]);
    setLatestError(null);
    setSkillVisEpoch((prev) => prev + 1);
  };

  return {
    connected: ws.connected,
    robotState,
    cameras,
    actionLog,
    stdoutLog,
    clearActionLog,
    resetSkillVisState,
    skillVisEpoch,
    proposedCode,
    setProposedCode,
    agentStatus,
    detectionDebug,
    segmentationDebug,
    vlmResults,
    debugMode,
    learnSkillStatus,
    graspViz,
    motionPlannerDebug,
    contactDetectDebug,
    latestError,
  };
}
