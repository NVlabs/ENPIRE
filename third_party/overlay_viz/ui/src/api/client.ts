// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Task,
  ScanStatus,
  EpisodeEntry,
  EpisodeInfo,
  TrimSegment,
  ChartData,
  ActionSourceData,
  FrequencyData,
  ComponentTimestampData,
  ProgressLabelData,
  CameraStatus,
  ReplayStatus,
  ValuePredictionData,
  ValuePredictionPrecomputeStatus,
  TimingHealthResponse,
} from "./types"

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`GET ${url}: ${res.status}${detail ? ` ${detail}` : ""}`)
  }
  return res.json()
}

async function post<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`POST ${url}: ${res.status}${detail ? ` ${detail}` : ""}`)
  }
  return res.json()
}

async function del<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: "DELETE" })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`DELETE ${url}: ${res.status}${detail ? ` ${detail}` : ""}`)
  }
  return res.json()
}

// Tasks
export const fetchTasks = () => get<Task[]>("/api/tasks")
export const fetchDefaultTask = () => get<{ task_id: string | null }>("/api/default_task")
export const triggerScan = (taskId: string) => post<unknown>(`/api/tasks/${taskId}/scan`)
export const fetchScanStatus = (taskId: string) => get<ScanStatus>(`/api/tasks/${taskId}/status`)
export const fetchEpisodes = (taskId: string) => get<EpisodeEntry[]>(`/api/tasks/${taskId}/episodes`)
export const fetchTimingHealth = (taskId: string) => get<TimingHealthResponse>(`/api/tasks/${taskId}/timing_health`)

// Episode data
export const fetchEpisodeInfo = (taskId: string, idx: number) =>
  get<EpisodeInfo>(`/api/tasks/${taskId}/episodes/${idx}/info`)
export const fetchActions = (taskId: string, idx: number) =>
  get<ChartData>(`/api/tasks/${taskId}/episodes/${idx}/actions?t=0&h=100000`)
export const fetchStates = (taskId: string, idx: number) =>
  get<ChartData>(`/api/tasks/${taskId}/episodes/${idx}/states`)
export const fetchActionSource = (taskId: string, idx: number) =>
  get<ActionSourceData>(`/api/tasks/${taskId}/episodes/${idx}/action_source`)
export const fetchFrequency = (taskId: string, idx: number) =>
  get<FrequencyData>(`/api/tasks/${taskId}/episodes/${idx}/frequency`)
export const fetchComponentTimestamps = (taskId: string, idx: number) =>
  get<ComponentTimestampData>(`/api/tasks/${taskId}/episodes/${idx}/component_timestamps`)
export const saveTrim = (taskId: string, idx: number, segments: TrimSegment[]) =>
  post<{ trim_segments: TrimSegment[]; saved_path: string; episode_dir: string }>(
    `/api/tasks/${taskId}/episodes/${idx}/trim`,
    { trim_segments: segments },
  )
export interface AutoClipParams {
  threshold: number
  minIdleSteps: number
  startThreshold: number
  endThreshold: number
  minSegmentSteps: number
}

export interface AutoClipResult {
  segments: TrimSegment[]
  total_steps: number
  threshold: number
  start_threshold: number
  end_threshold: number
  min_idle_steps: number
  min_segment_steps: number
  idle_runs: { start: number; end: number }[]
  leading_skip: number
  trailing_skip: number
  dropped_short_segments: number
  used_states: boolean
}

const autoClipBody = (p: AutoClipParams) => ({
  threshold: p.threshold,
  min_idle_steps: p.minIdleSteps,
  start_threshold: p.startThreshold,
  end_threshold: p.endThreshold,
  min_segment_steps: p.minSegmentSteps,
})

export const autoClipEpisode = (taskId: string, idx: number, params: AutoClipParams) =>
  post<AutoClipResult>(`/api/tasks/${taskId}/episodes/${idx}/auto_clip`, autoClipBody(params))

export const autoClipAll = (taskId: string, params: AutoClipParams, force = false) =>
  post<{
    total: number
    processed: number
    skipped: number
    errors: number
    processed_episodes: { idx: number; folder: string; segments: TrimSegment[] }[]
    skipped_episodes: { idx: number; folder: string }[]
    error_details: { idx: number; error: string }[]
  }>(
    `/api/tasks/${taskId}/auto_clip_all`,
    { ...autoClipBody(params), force },
  )
export const saveValueTrim = (taskId: string, idx: number, start: number, end: number) =>
  post<{ value_trim_start: number; value_trim_end: number; saved_path: string; episode_dir: string }>(
    `/api/tasks/${taskId}/episodes/${idx}/value_trim`,
    { value_trim_start: start, value_trim_end: end },
  )
export const setDiscard = (taskId: string, idx: number, discarded: boolean) =>
  post<{ discarded: boolean }>(
    `/api/tasks/${taskId}/episodes/${idx}/discard`,
    { discarded },
  )
export const deleteEpisode = (taskId: string, idx: number) =>
  del<{ deleted: boolean; episode_dir: string }>(
    `/api/tasks/${taskId}/episodes/${idx}`,
  )
export const purgeDiscarded = (taskId: string) =>
  post<{ deleted_count: number; deleted_paths: string[] }>(
    `/api/tasks/${taskId}/episodes/purge_discarded`,
  )

export interface ClearAnnotationsResult {
  episode_dir: string
  removed_keys: string[]
  removed_progress_labels: boolean
}
export const clearAnnotations = (taskId: string, idx: number) =>
  post<ClearAnnotationsResult>(
    `/api/tasks/${taskId}/episodes/${idx}/clear_annotations`,
  )
export const clearAllAnnotations = (taskId: string) =>
  post<{
    total: number
    cleared: number
    errors: number
    cleared_episodes: (ClearAnnotationsResult & { idx: number })[]
    error_details: { idx: number; error: string }[]
  }>(`/api/tasks/${taskId}/clear_annotations_all`)

export interface AutoScreenParams {
  min_frames?: number
  latency_threshold_s?: number
  pure_color_std_max?: number
  pure_color_subsample_stride?: number
  pure_color_max_offenders?: number
  dry_run?: boolean
}

export type AutoScreenFlag = "too_short" | "latency" | "pure_color"

export interface AutoScreenResultRow {
  idx: number
  folder: string
  flags: AutoScreenFlag[]
  details: {
    n_frames?: number
    max_gap_ms?: number
    spike_count?: number
    latency_indices?: number[]
    pure_color_frames?: number[]
  }
}

export interface AutoScreenResult {
  total: number
  flagged: number
  by_reason: { too_short: number; latency: number; pure_color: number }
  results: AutoScreenResultRow[]
  params: Record<string, unknown>
  dry_run: boolean
}

export const autoScreenAll = (taskId: string, params: AutoScreenParams) =>
  post<AutoScreenResult>(`/api/tasks/${taskId}/auto_screen_all`, params)

export interface AutoScreenProgress {
  running: boolean
  done: number
  total: number
  flagged: AutoScreenResultRow[]
  dry_run: boolean
}

export const fetchAutoScreenProgress = (taskId: string) =>
  get<AutoScreenProgress>(`/api/tasks/${taskId}/auto_screen_progress`)

// Progress labels
export const fetchLabels = (taskId: string, idx: number) =>
  get<ProgressLabelData>(`/api/tasks/${taskId}/episodes/${idx}/labels`)
export const saveLabels = (taskId: string, idx: number, progress: number[], keyframes: { step: number; progress: number }[]) =>
  post<{ saved: boolean; steps: number }>(
    `/api/tasks/${taskId}/episodes/${idx}/labels`,
    { progress, keyframes },
  )

// RTG markers
export const saveRtgMarker = (taskId: string, idx: number, rtgStart: number, rtgEnd: number, rtgStatus: string | null) =>
  post<{ saved: boolean; saved_path: string; episode_dir: string; rtg_start: number; rtg_end: number; rtg_status: string | null }>(
    `/api/tasks/${taskId}/episodes/${idx}/rtg_marker`,
    { rtg_start: rtgStart, rtg_end: rtgEnd, rtg_status: rtgStatus },
  )
export const fetchValuePredictions = (
  taskId: string,
  idx: number,
  options?: {
    force?: boolean
    preferLocal?: boolean
    batchSize?: number
    relativeInterval?: number
    prompt?: string
    advMode?: string
    advantageH?: number
    rtgGamma?: number
  },
) =>
  post<ValuePredictionData>(
    `/api/tasks/${taskId}/episodes/${idx}/value_predictions`,
    {
      force: options?.force,
      prefer_local: options?.preferLocal,
      batch_size: options?.batchSize,
      relative_interval: options?.relativeInterval,
      prompt: options?.prompt,
      adv_mode: options?.advMode,
      advantage_h: options?.advantageH,
      rtg_gamma: options?.rtgGamma,
    },
  )
export const startTaskValuePredictionPrecompute = (
  taskId: string,
  options?: {
    force?: boolean
    batchSize?: number
    relativeInterval?: number
    prompt?: string
  },
) =>
  post<ValuePredictionPrecomputeStatus & { status?: string }>(
    `/api/tasks/${taskId}/value_predictions/precompute`,
    {
      force: options?.force,
      batch_size: options?.batchSize,
      relative_interval: options?.relativeInterval,
      prompt: options?.prompt,
    },
  )
export const fetchTaskValuePredictionPrecomputeStatus = (taskId: string) =>
  get<ValuePredictionPrecomputeStatus>(
    `/api/tasks/${taskId}/value_predictions/precompute_status`,
  )

// Open a filesystem path in the OS file manager (server-side xdg-open)
export const openPath = (path: string) => post<{ opened: string }>("/api/open_path", { path })

// Export to lerobot v2.1
export interface ExportLerobotParams {
  minimal_policy_dir: string
  input_root: string
  output_dir: string
  task_name: string | null
  min_segment_length: number
  annotate_only: boolean
}
export interface ExportLerobotStatus {
  running: boolean
  stdout: string
  exit_code: number | null
  started_at: number | null
  finished_at: number | null
  command: string | null
  cwd: string | null
  error: string | null
}
export const startExportLerobot = (taskId: string, params: ExportLerobotParams) =>
  post<{ running: boolean; command: string; cwd: string }>(
    `/api/tasks/${taskId}/export_lerobot/start`,
    params,
  )
export const fetchExportLerobotStatus = (taskId: string) =>
  get<ExportLerobotStatus>(`/api/tasks/${taskId}/export_lerobot/status`)

// Mode
export const fetchMode = () =>
  get<{ mode: string; value_server_enabled?: boolean; value_server_url?: string | null }>("/api/mode")

// Episode media URLs (used as src attributes, not fetched)
export const cameraVideoUrl = (taskId: string, idx: number, camera: string) =>
  `/api/tasks/${taskId}/episodes/${idx}/camera_video?camera=${camera}`
export const cameraPosterUrl = (taskId: string, idx: number, camera: string) =>
  `/api/tasks/${taskId}/episodes/${idx}/camera_frame?camera=${camera}&t=0&quality=70`
export const overlayUrl = (taskId: string, idx: number, alpha: number, colorMode: string, bust: number) =>
  `/api/tasks/${taskId}/episodes/${idx}/overlay?alpha=${alpha}&color_mode=${colorMode}&_t=${bust}`
export const liveStreamUrl = (bust: number) => `/api/cameras/stream?${bust}`

// Cameras
export const fetchCameraStatus = () => get<CameraStatus>("/api/cameras")

// Replay control
export const replayConnect = () => post<ReplayStatus>("/api/replay/connect")
export const replayDisconnect = () => post<ReplayStatus>("/api/replay/disconnect")
export const replayStatus = () => get<ReplayStatus>("/api/replay/status")
export const replayPlay = () => post<{ success: boolean }>("/api/replay/play")
export const replayPause = () => post<{ success: boolean }>("/api/replay/pause")
export const replayStep = () => post<{ success: boolean }>("/api/replay/step")
export const replayHome = () => post<{ success: boolean }>("/api/replay/home")
export const replaySyncToInit = () => post<{ success: boolean }>("/api/replay/sync_to_init")
export const replayLoad = (taskId: string, episodeIdx: number) =>
  post<unknown>("/api/replay/load", { episode_idx: episodeIdx, task_id: taskId })
