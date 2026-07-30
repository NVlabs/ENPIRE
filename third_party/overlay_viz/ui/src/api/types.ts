// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface Task {
  id: string
  data_path: string
}

export interface ScanStatus {
  done: number
  total: number
  ready: boolean
  episodes: number
}

export interface EpisodeEntry {
  idx: number
  folder: string
  discarded?: boolean
  annotated?: boolean
  anomalous?: boolean
}

export interface TrimSegment {
  start: number
  end: number
  label?: string
  repeat_last?: number
}

export interface EpisodeInfo {
  total_frames: number
  fps: number
  duration_s: number
  folder: string
  cameras: { top: boolean; left: boolean; right: boolean }
  trim_start_frame: number
  trim_end_frame: number
  trim_segments: TrimSegment[]
  value_trim_start: number
  value_trim_end: number
  discarded: boolean
  rtg_marker?: { success_end?: number; failure_end?: number } | null
  rtg_start?: number
  rtg_end?: number
  rtg_status?: "success" | "failure" | null
  value_predictions_cached?: boolean
  value_predictions_cache_format?: string | null
}

export interface ChartData {
  labels: string[]
  data: number[][]
  dim: number
  total_steps: number
}

export interface ActionSourceData {
  segments: { start: number; end: number; source: string }[]
  unique_sources: string[]
  total_steps: number
}

export interface FrequencyData {
  dim: number
  labels: string[]
  freq_bins: number[]
  magnitudes: number[][]
}

export interface ProgressLabelData {
  progress: number[]
  keyframes: { step: number; progress: number }[]
  exists: boolean
}

export interface ComponentTimestampData {
  timestamps: Record<string, number>[]
  action_sources: string[]
  record_timestamps: number[]
}

export interface CameraStatus {
  online: boolean
  count: number
  devices: { name: string; serial: string }[]
  streaming: boolean
}

export interface ReplayStatus {
  connected: boolean
}

export interface ValuePredictionData {
  frame_idx: number[]
  absolute_value: number[]
  absolute_advantage: number[]
  relative_advantage?: number[]
  mode: "1step" | "2step" | string
  adv_mode?: string
  cached?: boolean
  prompt?: string
  ckpt_dir?: string
  created_at?: string
  source?: string
  cache_valid?: boolean
  cache_format?: string
  cache_path?: string
  task_id?: string
  episode_idx?: number
  episode_folder?: string
  episode_signature?: string
  model_cam_names?: string[]
  value_server_device?: string
}

export interface TimingHealthEntry {
  has_data: boolean
  level: "good" | "warn" | "bad"
  median_dt: number
  jitter_ratio: number
  spike_count: number
  max_gap_s: number
  n_steps: number
}

export interface TimingHealthResponse {
  ready: boolean
  done: number
  total: number
  episodes: Record<string, TimingHealthEntry>
}

export interface ValuePredictionPrecomputeStatus {
  task_id: string
  running: boolean
  done: number
  total: number
  completed: number
  failed: number
  current_episode_idx: number | null
  current_episode_folder: string | null
  last_error: string | null
  last_cache_path: string | null
  force: boolean
  manifest_path: string
  started_at: string | null
  finished_at: string | null
}
