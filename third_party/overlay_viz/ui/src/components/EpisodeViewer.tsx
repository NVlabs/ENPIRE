// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useCallback, useRef, useEffect, useMemo, type ReactNode } from "react"
import type { ChartData, EpisodeEntry } from "@/api/types"
import { ChevronLeft, ChevronRight, X, Loader2, Trash2, RotateCcw, RotateCw, Eye, EyeOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { CameraPanels } from "./CameraPanels"
import { TimelineControls, loadAutoClipParams, AUTO_CLIP_ON_LOAD_STORAGE_KEY } from "./TimelineControls"
import { TimelineChart } from "./TimelineChart"
import { FrequencyChart } from "./FrequencyChart"
import { TimestampChart } from "./TimestampChart"
import { ProgressLabelPanel } from "./ProgressLabelPanel"
import { ValueChart } from "./ValueChart"
import { DraggablePanel } from "./DraggablePanel"
import { useProgressLabels } from "@/hooks/useProgressLabels"
import { usePanelOrder, type PanelId } from "@/hooks/usePanelOrder"
import { findFirstMovementStep, findLastMovementStep } from "@/lib/chart-utils"
import * as api from "@/api/client"
import type { useEpisode } from "@/hooks/useEpisode"

const DEFAULT_JOINT_THRESHOLD = 5e-3
const DEFAULT_GRIP_THRESHOLD = 1e-3

// Fixed discrete speed set for the playback slider. Indices 0..6 map
// into this array. Default 1.0× lives at index 2.
const PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0] as const
const DEFAULT_PLAYBACK_SPEED_IDX = 2

function deriveSpeedData(src: ChartData | null | undefined): ChartData | null {
  if (!src) return null
  const T = src.data.length
  if (T === 0) return null
  const out: number[][] = new Array(T)
  for (let i = 0; i < T - 1; i++) {
    const row = new Array(src.dim) as number[]
    for (let c = 0; c < src.dim; c++) row[c] = src.data[i + 1][c] - src.data[i][c]
    out[i] = row
  }
  out[T - 1] = new Array(src.dim).fill(0) as number[]
  return { ...src, data: out }
}

function HideToggle({ hidden, onToggle }: { hidden: boolean; onToggle: () => void }) {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onToggle}
      title={hidden ? "Show panel" : "Hide panel"}
      className="ml-auto h-6 text-xs gap-1"
    >
      {hidden ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
      {hidden ? "Show" : "Hide"}
    </Button>
  )
}

interface Props {
  taskId: string
  viewerIdx: number
  episode: ReturnType<typeof useEpisode>
  episodeCount: number
  episodeEntry?: EpisodeEntry
  annotatedCount?: number
  discardedCount?: number
  unannotatedCount?: number
  replayConnected: boolean
  cameraStreaming: boolean
  mode?: string
  valueServerEnabled?: boolean
  autoPlayPending?: boolean
  onAutoPlayConsumed?: () => void
  continuousPlay?: boolean
  onContinuousPlayChange?: (v: boolean) => void
  onClose: () => void
  onNavigate: (delta: number) => void
  onCompletelyRemove?: () => void | Promise<void>
  onSaveNotice?: (title: string, savedPath: string, episodeDir: string) => void
}

export function EpisodeViewer({ taskId, viewerIdx, episode, episodeCount, episodeEntry, annotatedCount, discardedCount, unannotatedCount, replayConnected, cameraStreaming, mode, valueServerEnabled, autoPlayPending, onAutoPlayConsumed, continuousPlay, onContinuousPlayChange, onClose, onNavigate, onCompletelyRemove, onSaveNotice }: Props) {
  const ep = episode
  const isLabelMode = mode === "label"
  const isRtgMode = mode === "rtg"
  const progressLabels = useProgressLabels(taskId, viewerIdx, ep.totalSteps)
  const [actionShowLeft, setActionShowLeft] = useState(true)
  const [actionShowRight, setActionShowRight] = useState(true)
  const [actionShowGrip, setActionShowGrip] = useState(true)
  const [stateShowLeft, setStateShowLeft] = useState(true)
  const [stateShowRight, setStateShowRight] = useState(true)
  const [stateShowGrip, setStateShowGrip] = useState(true)
  const [freqShowLeft, setFreqShowLeft] = useState(true)
  const [freqShowRight, setFreqShowRight] = useState(true)
  const [freqShowGrip, setFreqShowGrip] = useState(true)
  const [actionsHidden, setActionsHidden] = useState(false)
  const [actionSpeedHidden, setActionSpeedHidden] = useState(false)
  const [statesHidden, setStatesHidden] = useState(false)
  const [stateSpeedHidden, setStateSpeedHidden] = useState(false)
  const [frequencyHidden, setFrequencyHidden] = useState(false)
  const [timestampHidden, setTimestampHidden] = useState(false)
  const [datasetPlaying, setDatasetPlaying] = useState(false)
  const [chainPlaying, setChainPlaying] = useState(false)
  const chainPlayingRef = useRef(false)
  const setChain = useCallback((v: boolean) => { chainPlayingRef.current = v; setChainPlaying(v) }, [])
  const continuousPlayRef = useRef(Boolean(continuousPlay))
  continuousPlayRef.current = Boolean(continuousPlay)
  const [overlayAlpha] = useState(0.5)
  const [colorMode] = useState("bgr")
  const [overlayBust, setOverlayBust] = useState(Date.now())
  const [rtgStart, setRtgStart] = useState(0)
  const [rtgEnd, setRtgEnd] = useState(0)
  const [rtgStatus, setRtgStatus] = useState<"success" | "failure" | null>(null)
  const [rtgDirty, setRtgDirty] = useState(false)
  const [rtgSaving, setRtgSaving] = useState(false)
  const playTimer = useRef<ReturnType<typeof setInterval> | null>(null)
  const [playbackSpeedIdx, setPlaybackSpeedIdx] = useState<number>(DEFAULT_PLAYBACK_SPEED_IDX)
  const playbackSpeed = PLAYBACK_SPEEDS[playbackSpeedIdx] ?? 1.0
  const playbackSpeedRef = useRef(playbackSpeed)
  playbackSpeedRef.current = playbackSpeed
  const [jointThreshold, setJointThresholdRaw] = useState(DEFAULT_JOINT_THRESHOLD)
  const [gripThreshold, setGripThresholdRaw] = useState(DEFAULT_GRIP_THRESHOLD)
  const [autoClipOnLoad, setAutoClipOnLoadRaw] = useState<boolean>(() => {
    try {
      return localStorage.getItem(AUTO_CLIP_ON_LOAD_STORAGE_KEY) === "1"
    } catch {
      return false
    }
  })
  const setAutoClipOnLoad = useCallback((v: boolean) => {
    setAutoClipOnLoadRaw(v)
    try {
      localStorage.setItem(AUTO_CLIP_ON_LOAD_STORAGE_KEY, v ? "1" : "0")
    } catch { /* ignore quota errors */ }
  }, [])
  // Key by folder (which only updates after the new episode's info is
  // fetched) rather than viewerIdx (which flips instantly and causes us
  // to mark "applied" while ep.* still holds the previous episode's data).
  const autoAppliedRef = useRef<string | null>(null)

  // Load per-task thresholds from localStorage whenever the task changes.
  useEffect(() => {
    if (!taskId) return
    const j = localStorage.getItem(`tbd-trim-joint-threshold:${taskId}`)
    setJointThresholdRaw(j ? parseFloat(j) || DEFAULT_JOINT_THRESHOLD : DEFAULT_JOINT_THRESHOLD)
    const g = localStorage.getItem(`tbd-trim-grip-threshold:${taskId}`)
    setGripThresholdRaw(g ? parseFloat(g) || DEFAULT_GRIP_THRESHOLD : DEFAULT_GRIP_THRESHOLD)
  }, [taskId])

  const setJointThreshold = useCallback((v: number) => {
    setJointThresholdRaw(v)
    if (taskId) localStorage.setItem(`tbd-trim-joint-threshold:${taskId}`, String(v))
  }, [taskId])
  const setGripThreshold = useCallback((v: number) => {
    setGripThresholdRaw(v)
    if (taskId) localStorage.setItem(`tbd-trim-grip-threshold:${taskId}`, String(v))
  }, [taskId])

  const actionSpeedData = useMemo(() => deriveSpeedData(ep.chartAllData), [ep.chartAllData])
  const stateSpeedData = useMemo(() => deriveSpeedData(ep.stateAllData), [ep.stateAllData])

  // Load RTG state from episode info
  useEffect(() => {
    if (!ep.info) return
    const lastFrame = Math.max(0, (ep.info.total_frames ?? 1) - 1)
    setRtgStart(ep.info.rtg_start ?? 0)
    setRtgEnd(ep.info.rtg_end ?? lastFrame)
    setRtgStatus((ep.info.rtg_status as "success" | "failure") ?? null)
    setRtgDirty(false)
    setRtgSaving(false)
  }, [ep.info])

  // Re-open every panel when the operator navigates to a new episode.
  // Hidden state shouldn't bleed across episodes.
  useEffect(() => {
    setActionsHidden(false)
    setActionSpeedHidden(false)
    setStatesHidden(false)
    setStateSpeedHidden(false)
    setFrequencyHidden(false)
    setTimestampHidden(false)
  }, [viewerIdx])

  // Apply playback rate changes live while the video is playing so a
  // slider drag takes effect immediately, not just on the next Play.
  useEffect(() => {
    if (!datasetPlaying) return
    epRef.current.setPlaybackRate(playbackSpeed)
  }, [playbackSpeed, datasetPlaying])

  useEffect(() => {
    if (!replayConnected || !cameraStreaming) return
    const id = setInterval(() => setOverlayBust(Date.now()), 500)
    return () => clearInterval(id)
  }, [replayConnected, cameraStreaming])

  useEffect(() => {
    if (!ep.info) return
    if (replayConnected || isLabelMode || isRtgMode) return
    if (!ep.chartAllData && !ep.chartsLoading && !ep.chartsError) ep.loadCharts()
    if (!ep.timestampData && !ep.timestampLoading && !ep.timestampError) ep.loadTimestamps()
    if (!ep.freqData && !ep.freqLoading && !ep.freqError) ep.loadFrequency()
  }, [ep.info, replayConnected, isLabelMode, isRtgMode, ep.chartAllData, ep.chartsLoading, ep.chartsError, ep.timestampData, ep.timestampLoading, ep.timestampError, ep.freqData, ep.freqLoading, ep.freqError, ep.loadCharts, ep.loadTimestamps, ep.loadFrequency])

  // Use refs for values accessed inside interval callbacks to avoid stale closures
  const epRef = useRef(ep)
  epRef.current = ep

  const handleSeek = useCallback((step: number) => {
    const { totalSteps, setChartStep, syncVideos } = epRef.current
    const maxValid = Math.max(0, (totalSteps || 1) - 1)
    const clamped = Math.max(0, Math.min(step, maxValid))
    setChartStep(clamped)
    syncVideos(clamped)
  }, [])

  const rtgStartRef = useRef(rtgStart)
  rtgStartRef.current = rtgStart
  const rtgEndRef = useRef(rtgEnd)
  rtgEndRef.current = rtgEnd

  const datasetPlay = useCallback(() => {
    const ep = epRef.current
    let playStart: number, playEnd: number
    if (isRtgMode) {
      playStart = rtgStartRef.current
      playEnd = rtgEndRef.current
    } else if (isLabelMode) {
      playStart = ep.valueTrimStart
      playEnd = ep.valueTrimEnd
    } else {
      playStart = ep.rangeStart
      playEnd = ep.rangeEnd
    }
    const start = (ep.chartStep < playStart || ep.chartStep >= playEnd) ? playStart : ep.chartStep
    ep.playVideos(start)
    ep.setPlaybackRate(playbackSpeedRef.current)
    ep.setChartStep(start)
    setDatasetPlaying(true)

    if (playTimer.current) clearInterval(playTimer.current)
    playTimer.current = setInterval(() => {
      const ep = epRef.current
      let end: number, clampStart: number
      if (isRtgMode) {
        end = rtgEndRef.current
        clampStart = rtgStartRef.current
      } else if (isLabelMode) {
        end = ep.valueTrimEnd
        clampStart = ep.valueTrimStart
      } else {
        end = ep.rangeEnd
        clampStart = ep.rangeStart
      }
      const t = ep.getVideoTime()
      if (t === null) return
      const step = Math.round(t * ep.fps)
      if (step >= end) {
        const segs = ep.segments
        const cur = ep.activeSegIdx
        if (chainPlayingRef.current && !isRtgMode && !isLabelMode && cur >= 0 && cur + 1 < segs.length) {
          const nextIdx = cur + 1
          const nextSeg = segs[nextIdx]
          ep.setActiveSegIdx(nextIdx)
          ep.setChartStep(nextSeg.start)
          ep.playVideos(nextSeg.start)
          return
        }
        // Ran off the end of the episode. If continuous playback is on,
        // advance to the next episode — App.tsx sets autoPlayPending, and
        // the autoPlayPending effect below restarts datasetPlay once the
        // new episode's videos are ready.
        if (continuousPlayRef.current && !isRtgMode && !isLabelMode) {
          ep.pauseVideos()
          setDatasetPlaying(false)
          setChain(false)
          if (playTimer.current) { clearInterval(playTimer.current); playTimer.current = null }
          onNavigate(1)
          return
        }
        ep.setChartStep(end)
        ep.pauseVideos()
        setDatasetPlaying(false)
        setChain(false)
        if (playTimer.current) { clearInterval(playTimer.current); playTimer.current = null }
        return
      }
      ep.setChartStep(Math.max(clampStart, step))
    }, 50)
  }, [isLabelMode, isRtgMode, onNavigate, setChain])

  const datasetStop = useCallback(() => {
    epRef.current.pauseVideos()
    setDatasetPlaying(false)
    setChain(false)
    onContinuousPlayChange?.(false)
    if (playTimer.current) { clearInterval(playTimer.current); playTimer.current = null }
  }, [setChain, onContinuousPlayChange])

  const onSwitchToNextSegment = useCallback(() => {
    if (chainPlayingRef.current) {
      datasetStop()
      return
    }
    const segs = epRef.current.segments
    if (segs.length === 0 || isRtgMode || isLabelMode) return
    const cur = epRef.current.activeSegIdx
    const nextIdx = (cur < 0 || cur >= segs.length - 1) ? 0 : cur + 1
    const nextSeg = segs[nextIdx]
    epRef.current.setActiveSegIdx(nextIdx)
    epRef.current.setChartStep(nextSeg.start)
    setChain(true)
    // Defer play until React flushes the segment change so datasetPlay reads
    // the new rangeStart/rangeEnd from epRef.
    setTimeout(() => datasetPlay(), 0)
  }, [datasetPlay, datasetStop, isRtgMode, isLabelMode, setChain])

  const handleSnapInToMovement = useCallback(() => {
    if (!ep.stateAllData || ep.activeSegIdx < 0) return
    const step = findFirstMovementStep(ep.stateAllData, jointThreshold, gripThreshold)
    ep.setRangeStart(step)
    handleSeek(step)
  }, [ep.stateAllData, ep.activeSegIdx, ep.setRangeStart, jointThreshold, gripThreshold, handleSeek])

  const handleStartFromFirstFrame = useCallback(() => {
    if (ep.activeSegIdx < 0) return
    ep.setRangeStart(0)
    handleSeek(0)
  }, [ep.activeSegIdx, ep.setRangeStart, handleSeek])

  const handleSnapOutToMovement = useCallback(() => {
    if (!ep.stateAllData || ep.activeSegIdx < 0) return
    const step = findLastMovementStep(ep.stateAllData, jointThreshold, gripThreshold)
    ep.setRangeEnd(step)
    handleSeek(step)
  }, [ep.stateAllData, ep.activeSegIdx, ep.setRangeEnd, jointThreshold, gripThreshold, handleSeek])

  const handleEndAtLastFrame = useCallback(() => {
    if (ep.activeSegIdx < 0) return
    const last = Math.max(0, ep.totalSteps - 1)
    ep.setRangeEnd(last)
    handleSeek(last)
  }, [ep.activeSegIdx, ep.setRangeEnd, ep.totalSteps, handleSeek])

  // Auto-snap both trim IN and trim OUT to the first/last movement step
  // on episode load. Gated on the same "auto-clip on load" toggle as the
  // multi-segment auto-clip below, and skipped for already-annotated
  // episodes — so when the toggle is off, nothing auto-modifies segments
  // and Save stays clean. Segments get marked dirty when we do apply.
  useEffect(() => {
    if (!autoClipOnLoad) return
    if (!ep.info || !ep.stateAllData) return
    if (autoAppliedRef.current === ep.info.folder) return
    if (ep.activeSegIdx < 0) return
    if (episodeEntry?.annotated) return
    const currentSeg = ep.segments[ep.activeSegIdx]
    if (!currentSeg) return  // segments not yet populated — retry on the next render
    // All preconditions satisfied; claim this folder so we don't re-apply.
    autoAppliedRef.current = ep.info.folder
    const startStep = findFirstMovementStep(ep.stateAllData, jointThreshold, gripThreshold)
    const endStep = findLastMovementStep(ep.stateAllData, jointThreshold, gripThreshold)
    if (endStep <= startStep) {
      handleSeek(startStep)
      return
    }
    ep.setRangeStart(startStep)
    ep.setRangeEnd(endStep)
    handleSeek(startStep)
  }, [autoClipOnLoad, ep.info, ep.stateAllData, ep.activeSegIdx, ep.segments, ep.setRangeStart, ep.setRangeEnd, jointThreshold, gripThreshold, handleSeek, episodeEntry?.annotated])

  // Auto-clip on episode load when the toggle is enabled and the episode
  // has no saved trim segments. Claims autoAppliedRef first so the Snap
  // IN/OUT effect above skips this folder — otherwise both would fight.
  // Result is left dirty: operator must press Save to persist.
  //
  // "No saved trim" is decided via episodeEntry.annotated (the scan-time
  // flag for `"trim_segments" in metadata`), not ep.info.trim_segments:
  // the server synthesizes a full-span default segment when nothing is
  // saved, so length is always ≥ 1 and cannot distinguish the two.
  useEffect(() => {
    if (!autoClipOnLoad || !ep.info) return
    if (autoAppliedRef.current === ep.info.folder) return
    if (episodeEntry?.annotated) return
    autoAppliedRef.current = ep.info.folder
    ep.autoClip(loadAutoClipParams()).catch(() => {})
  }, [autoClipOnLoad, ep.info, ep.autoClip, episodeEntry?.annotated])

  const handleRtgSave = useCallback(async () => {
    if (rtgSaving) return
    setRtgSaving(true)
    try {
      const res = await api.saveRtgMarker(taskId, viewerIdx, rtgStart, rtgEnd, rtgStatus)
      setRtgDirty(false)
      onSaveNotice?.("Saved RTG marker", res.saved_path, res.episode_dir)
    } catch (e) {
      console.error("Failed to save RTG:", e)
    } finally {
      setRtgSaving(false)
    }
  }, [taskId, viewerIdx, rtgStart, rtgEnd, rtgStatus, rtgSaving, onSaveNotice])

  useEffect(() => () => {
    if (playTimer.current) clearInterval(playTimer.current)
  }, [])

  // Auto-play when navigated via "Next" buttons. Wait for videos to have
  // metadata before starting; cap at 5s so a missing video doesn't hang.
  useEffect(() => {
    if (!autoPlayPending) return
    if (!ep.info) return
    if (replayConnected || isLabelMode) return
    let cancelled = false
    const start = Date.now()
    const tryStart = () => {
      if (cancelled) return
      const ready = epRef.current.getVideoTime() !== null
      if (ready) {
        // Under continuous play, chain through every segment of this
        // episode before the end-of-play branch advances to the next.
        if (continuousPlayRef.current && epRef.current.segments.length >= 2) {
          setChain(true)
        }
        datasetPlay()
        onAutoPlayConsumed?.()
        return
      }
      if (Date.now() - start > 5000) {
        onAutoPlayConsumed?.()
        return
      }
      setTimeout(tryStart, 100)
    }
    const id = setTimeout(tryStart, 100)
    return () => { cancelled = true; clearTimeout(id) }
  }, [autoPlayPending, ep.info, replayConnected, isLabelMode, datasetPlay, onAutoPlayConsumed])

  // Keyboard: space = play/pause, r = reset to rangeStart
  // Label mode: k = add keyframe, [ ] = prev/next keyframe, Delete = delete keyframe
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return
      if (e.key === " ") {
        e.preventDefault()
        if (datasetPlaying) datasetStop()
        else datasetPlay()
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault()
        if (isLabelMode) {
          progressLabels.save()
          epRef.current.saveValueTrim()
        } else {
          epRef.current.saveTrim()
        }
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault()
        handleSeek(isRtgMode ? rtgStartRef.current : epRef.current.rangeStart)
      } else if (isLabelMode) {
        if (e.key === "k" || e.key === "K") {
          e.preventDefault()
          progressLabels.addKeyFrame(epRef.current.chartStep)
        } else if (e.key === "[") {
          e.preventDefault()
          const prev = progressLabels.prevKeyFrame(epRef.current.chartStep)
          if (prev !== null) handleSeek(prev)
        } else if (e.key === "]") {
          e.preventDefault()
          const next = progressLabels.nextKeyFrame(epRef.current.chartStep)
          if (next !== null) handleSeek(next)
        } else if (e.key === "Delete" || e.key === "Backspace") {
          e.preventDefault()
          progressLabels.deleteKeyFrame(epRef.current.chartStep)
        }
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [datasetPlaying, datasetPlay, datasetStop, handleSeek, isLabelMode, progressLabels])

  const panelOrder = usePanelOrder()
  const [advMode, setAdvMode] = useState<string>("value_delta")
  const [advantageH, setAdvantageH] = useState<number>(50)

  if (!ep.info) return null

  const hasSavedValueCache = Boolean(ep.info.value_predictions_cached)
  const canShowValuePanel = Boolean(valueServerEnabled || hasSavedValueCache)
  const computeButtonLabel = (ep.valuePredictionData || hasSavedValueCache) ? "Recompute" : "Compute"

  const valuePanel = canShowValuePanel && (
    <div className="mt-4">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <span className="text-xs font-bold opacity-70">VALUE</span>
        {ep.valuePredictionData?.mode && <Badge variant="secondary" className="text-[10px]">{ep.valuePredictionData.mode}</Badge>}
        {ep.valuePredictionData?.adv_mode && <Badge variant="secondary" className="text-[10px]">{ep.valuePredictionData.adv_mode}</Badge>}
        {ep.valuePredictionData?.cached && <Badge variant="outline" className="text-[10px]">cached</Badge>}
        {ep.valuePredictionData?.source === "local" && <Badge variant="outline" className="text-[10px]">saved</Badge>}
        {ep.valuePredictionData?.cache_valid === false && <Badge variant="destructive" className="text-[10px]">stale ckpt</Badge>}
        <select
          value={advMode}
          onChange={e => setAdvMode(e.target.value)}
          className="h-6 rounded border bg-background px-1 text-xs"
        >
          <option value="value_delta">value_delta</option>
          <option value="recap_post_train">recap_post_train</option>
          <option value="traj_adv">traj_adv</option>
        </select>
        <label className="flex items-center gap-1 text-xs">
          <span className="opacity-70">h=</span>
          <input
            type="number"
            value={advantageH}
            onChange={e => setAdvantageH(Math.max(1, parseInt(e.target.value) || 1))}
            className="h-6 w-14 rounded border bg-background px-1 text-xs"
            min={1}
          />
        </label>
        {hasSavedValueCache && !ep.valuePredictionLoading && (
          <Button variant="outline" size="sm" onClick={() => ep.loadValuePredictions({ preferLocal: true })} className="h-6 text-xs">
            Load Saved
          </Button>
        )}
        {valueServerEnabled && !ep.valuePredictionLoading && (
          <Button variant="default" size="sm" onClick={() => ep.loadValuePredictions({ force: true, advMode })} className="h-6 text-xs">
            {computeButtonLabel}
          </Button>
        )}
      </div>

      {ep.valuePredictionData && (
        <p className="mb-2 text-xs text-muted-foreground">
          {ep.valuePredictionData.source === "local" ? "Loaded saved value prediction." : "Computed with current value server."}
          {ep.valuePredictionData.ckpt_dir ? ` ckpt=${ep.valuePredictionData.ckpt_dir}` : ""}
          {ep.valuePredictionData.created_at ? ` saved=${ep.valuePredictionData.created_at}` : ""}
        </p>
      )}

      {valueServerEnabled && (
        <div className="mb-3 rounded-lg border bg-muted/30 p-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <p className="text-sm font-medium">Task-Wide Precompute</p>
              <p className="text-xs text-muted-foreground">
                Compute and save value predictions for every episode currently loaded in this task.
              </p>
            </div>
            {!ep.valueTaskPrecomputeStatus?.running && (
              <Button
                variant="secondary"
                onClick={() => ep.startTaskValuePredictionPrecompute({ force: true })}
                className="w-full sm:w-auto"
              >
                Precompute Task
              </Button>
            )}
            {ep.valueTaskPrecomputeStatus?.running && (
              <Button variant="secondary" disabled className="w-full sm:w-auto">
                Computing Task...
              </Button>
            )}
          </div>
        </div>
      )}

      {ep.valueTaskPrecomputeStatus && (ep.valueTaskPrecomputeStatus.running || ep.valueTaskPrecomputeStatus.done > 0 || ep.valueTaskPrecomputeStatus.finished_at) && (
        <p className="mb-2 text-xs text-muted-foreground">
          Task precompute: {ep.valueTaskPrecomputeStatus.done}/{ep.valueTaskPrecomputeStatus.total} done
          {ep.valueTaskPrecomputeStatus.failed > 0 ? `, ${ep.valueTaskPrecomputeStatus.failed} failed` : ""}
          {ep.valueTaskPrecomputeStatus.current_episode_folder ? `, current=${ep.valueTaskPrecomputeStatus.current_episode_folder}` : ""}
          {ep.valueTaskPrecomputeStatus.last_error ? `, last_error=${ep.valueTaskPrecomputeStatus.last_error}` : ""}
        </p>
      )}

      {ep.valuePredictionLoading && (
        <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin" />
        </div>
      )}

      {ep.valuePredictionError && !ep.valuePredictionLoading && !ep.valuePredictionData && (
        <div className="bg-muted rounded-lg p-6 flex flex-col items-center justify-center gap-2">
          <p className="text-sm text-red-400">Failed to load value predictions</p>
          <p className="text-xs text-muted-foreground">{ep.valuePredictionError}</p>
          {hasSavedValueCache && (
            <Button variant="outline" onClick={() => ep.loadValuePredictions({ preferLocal: true })}>Load Saved</Button>
          )}
          {valueServerEnabled && (
            <Button variant="outline" onClick={() => ep.loadValuePredictions({ force: true })}>{computeButtonLabel}</Button>
          )}
        </div>
      )}

      {ep.valuePredictionData && !ep.valuePredictionLoading && (
        <div className="space-y-2">
          {ep.valuePredictionError && (
            <p className="text-xs text-red-400">Refresh failed: {ep.valuePredictionError}</p>
          )}
          <ValueChart
            data={ep.valuePredictionData}
            chartStep={ep.chartStep}
            totalSteps={ep.totalSteps}
            fps={ep.fps}
            advMode={advMode}
            advantageH={advantageH}
            rtgRange={{ start: rtgStart, end: rtgEnd, status: rtgStatus }}
            onSeek={handleSeek}
          />
        </div>
      )}
    </div>
  )

  // -- Build panel content map keyed by PanelId --
  const panelContent: Record<PanelId, ReactNode> = {
    cameras: (
      <CameraPanels
        taskId={taskId}
        viewerIdx={viewerIdx}
        info={ep.info}
        replayConnected={replayConnected}
        cameraStreaming={cameraStreaming}
        overlayAlpha={overlayAlpha}
        colorMode={colorMode}
        overlayBust={overlayBust}
        registerVideo={ep.registerVideo}
      />
    ),

    value: valuePanel || null,

    nav: (
      <div className="flex justify-center mt-3 gap-2">
        <Button variant="outline" size="sm" onClick={() => onNavigate(-1)} disabled={viewerIdx <= 0}>
          <ChevronLeft className="h-4 w-4" /> Prev
        </Button>
        <span className="text-sm self-center">{viewerIdx + 1} / {episodeCount}</span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onNavigate(1)}
          title={viewerIdx >= episodeCount - 1 ? "Finish annotating and show summary" : undefined}
        >
          Next <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    ),

    timeline: (
      <div className="mt-4">
        <TimelineControls
          chartStep={ep.chartStep}
          rangeStart={ep.rangeStart}
          rangeEnd={ep.rangeEnd}
          totalSteps={ep.totalSteps}
          fps={ep.fps}
          replayConnected={replayConnected}
          datasetPlaying={datasetPlaying}
          trimSaving={ep.trimSaving}
          trimDirty={ep.trimDirty}
          mode={mode}
          segments={ep.segments}
          activeSegIdx={ep.activeSegIdx}
          onStepChange={handleSeek}
          onRangeChange={(s, e) => { ep.setRangeStart(s); ep.setRangeEnd(e) }}
          onDatasetPlay={datasetPlay}
          onDatasetStop={datasetStop}
          onSaveTrim={ep.saveTrim}
          onResetTrim={ep.resetTrim}
          onAutoClip={ep.autoClip}
          onAutoClipAll={ep.autoClipAllAndSave}
          autoClipOnLoad={autoClipOnLoad}
          onAutoClipOnLoadChange={setAutoClipOnLoad}
          onSelectSegment={(idx) => { ep.setActiveSegIdx(idx); handleSeek(ep.segments[idx]?.start ?? 0) }}
          onAddSegment={ep.addSegment}
          onSwitchToNextSegment={onSwitchToNextSegment}
          chainPlaying={chainPlaying}
          continuousPlay={continuousPlay}
          onContinuousPlayChange={onContinuousPlayChange}
          onRemoveSegment={ep.removeSegment}
          onRenameSegment={ep.updateSegmentLabel}
          onRepeatChange={ep.updateSegmentRepeat}
          valueTrimStart={ep.valueTrimStart}
          valueTrimEnd={ep.valueTrimEnd}
          valueTrimDirty={ep.valueTrimDirty}
          valueTrimSaving={ep.valueTrimSaving}
          onValueTrimChange={(s, e) => { ep.setValueTrimStart(s); ep.setValueTrimEnd(e) }}
          onSaveValueTrim={ep.saveValueTrim}
          onResetValueTrim={ep.resetValueTrim}
          onAddKeyFrame={progressLabels.addKeyFrame}
          rtgStart={rtgStart}
          rtgEnd={rtgEnd}
          rtgStatus={rtgStatus}
          rtgDirty={rtgDirty}
          rtgSaving={rtgSaving}
          onRtgStartChange={(s) => { setRtgStart(s); setRtgDirty(true) }}
          onRtgEndChange={(s) => { setRtgEnd(s); setRtgDirty(true) }}
          onRtgStatusChange={(s) => { setRtgStatus(s); setRtgDirty(true) }}
          onRtgSave={handleRtgSave}
          jointThreshold={jointThreshold}
          gripThreshold={gripThreshold}
          onJointThresholdChange={setJointThreshold}
          onGripThresholdChange={setGripThreshold}
          onSnapInToMovement={handleSnapInToMovement}
          onStartFromFirstFrame={handleStartFromFirstFrame}
          onSnapOutToMovement={handleSnapOutToMovement}
          onEndAtLastFrame={handleEndAtLastFrame}
          playbackSpeeds={PLAYBACK_SPEEDS}
          playbackSpeedIdx={playbackSpeedIdx}
          onPlaybackSpeedIdxChange={setPlaybackSpeedIdx}
        />
      </div>
    ),

    charts: (
      <>
        {!isLabelMode && !isRtgMode && !ep.chartsError && (
          <>
            {ep.chartsLoading && (
              <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            )}

            {!ep.chartAllData && !ep.chartsLoading && (
              <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            )}

            {ep.chartAllData && (
              <div>
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">ACTIONS</span>
                  <ToggleGroup show={actionShowLeft} onToggle={() => setActionShowLeft(!actionShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={actionShowRight} onToggle={() => setActionShowRight(!actionShowRight)} label="Right" color="red" />
                  <ToggleGroup show={actionShowGrip} onToggle={() => setActionShowGrip(!actionShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={actionsHidden} onToggle={() => setActionsHidden(!actionsHidden)} />
                </div>
                {!actionsHidden && (
                  <TimelineChart
                    data={ep.chartAllData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={actionShowLeft}
                    showRight={actionShowRight}
                    showGrip={actionShowGrip}
                    segments={ep.segments}
                    activeSegIdx={ep.activeSegIdx}
                    actionSourceData={ep.actionSourceData}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}

            {actionSpeedData && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">ACTION SPEED</span>
                  <ToggleGroup show={actionShowLeft} onToggle={() => setActionShowLeft(!actionShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={actionShowRight} onToggle={() => setActionShowRight(!actionShowRight)} label="Right" color="red" />
                  <ToggleGroup show={actionShowGrip} onToggle={() => setActionShowGrip(!actionShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={actionSpeedHidden} onToggle={() => setActionSpeedHidden(!actionSpeedHidden)} />
                </div>
                {!actionSpeedHidden && (
                  <TimelineChart
                    data={actionSpeedData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={actionShowLeft}
                    showRight={actionShowRight}
                    showGrip={actionShowGrip}
                    segments={ep.segments}
                    activeSegIdx={ep.activeSegIdx}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}

            {ep.stateAllData && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">STATES</span>
                  <ToggleGroup show={stateShowLeft} onToggle={() => setStateShowLeft(!stateShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={stateShowRight} onToggle={() => setStateShowRight(!stateShowRight)} label="Right" color="red" />
                  <ToggleGroup show={stateShowGrip} onToggle={() => setStateShowGrip(!stateShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={statesHidden} onToggle={() => setStatesHidden(!statesHidden)} />
                </div>
                {!statesHidden && (
                  <TimelineChart
                    data={ep.stateAllData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={stateShowLeft}
                    showRight={stateShowRight}
                    showGrip={stateShowGrip}
                    segments={ep.segments}
                    activeSegIdx={ep.activeSegIdx}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}

            {stateSpeedData && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">STATE SPEED</span>
                  <ToggleGroup show={stateShowLeft} onToggle={() => setStateShowLeft(!stateShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={stateShowRight} onToggle={() => setStateShowRight(!stateShowRight)} label="Right" color="red" />
                  <ToggleGroup show={stateShowGrip} onToggle={() => setStateShowGrip(!stateShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={stateSpeedHidden} onToggle={() => setStateSpeedHidden(!stateSpeedHidden)} />
                </div>
                {!stateSpeedHidden && (
                  <TimelineChart
                    data={stateSpeedData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={stateShowLeft}
                    showRight={stateShowRight}
                    showGrip={stateShowGrip}
                    segments={ep.segments}
                    activeSegIdx={ep.activeSegIdx}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}
          </>
        )}

        {isRtgMode && (
          <>
            {ep.chartsLoading && (
              <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            )}

            {!ep.chartAllData && !ep.chartsLoading && !ep.chartsError && (
              <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
                <Button onClick={ep.loadCharts}>Load Charts</Button>
              </div>
            )}

            {ep.chartAllData && (
              <div>
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">ACTIONS</span>
                  <ToggleGroup show={actionShowLeft} onToggle={() => setActionShowLeft(!actionShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={actionShowRight} onToggle={() => setActionShowRight(!actionShowRight)} label="Right" color="red" />
                  <ToggleGroup show={actionShowGrip} onToggle={() => setActionShowGrip(!actionShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={actionsHidden} onToggle={() => setActionsHidden(!actionsHidden)} />
                </div>
                {!actionsHidden && (
                  <TimelineChart
                    data={ep.chartAllData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={actionShowLeft}
                    showRight={actionShowRight}
                    showGrip={actionShowGrip}
                    segments={[]}
                    activeSegIdx={-1}
                    actionSourceData={ep.actionSourceData}
                    rtgRange={{ start: rtgStart, end: rtgEnd, status: rtgStatus }}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}

            {actionSpeedData && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">ACTION SPEED</span>
                  <ToggleGroup show={actionShowLeft} onToggle={() => setActionShowLeft(!actionShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={actionShowRight} onToggle={() => setActionShowRight(!actionShowRight)} label="Right" color="red" />
                  <ToggleGroup show={actionShowGrip} onToggle={() => setActionShowGrip(!actionShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={actionSpeedHidden} onToggle={() => setActionSpeedHidden(!actionSpeedHidden)} />
                </div>
                {!actionSpeedHidden && (
                  <TimelineChart
                    data={actionSpeedData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={actionShowLeft}
                    showRight={actionShowRight}
                    showGrip={actionShowGrip}
                    segments={[]}
                    activeSegIdx={-1}
                    rtgRange={{ start: rtgStart, end: rtgEnd, status: rtgStatus }}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}

            {ep.stateAllData && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">STATES</span>
                  <ToggleGroup show={stateShowLeft} onToggle={() => setStateShowLeft(!stateShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={stateShowRight} onToggle={() => setStateShowRight(!stateShowRight)} label="Right" color="red" />
                  <ToggleGroup show={stateShowGrip} onToggle={() => setStateShowGrip(!stateShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={statesHidden} onToggle={() => setStatesHidden(!statesHidden)} />
                </div>
                {!statesHidden && (
                  <TimelineChart
                    data={ep.stateAllData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={stateShowLeft}
                    showRight={stateShowRight}
                    showGrip={stateShowGrip}
                    segments={[]}
                    activeSegIdx={-1}
                    rtgRange={{ start: rtgStart, end: rtgEnd, status: rtgStatus }}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}

            {stateSpeedData && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-xs font-semibold opacity-70">STATE SPEED</span>
                  <ToggleGroup show={stateShowLeft} onToggle={() => setStateShowLeft(!stateShowLeft)} label="Left" color="blue" />
                  <ToggleGroup show={stateShowRight} onToggle={() => setStateShowRight(!stateShowRight)} label="Right" color="red" />
                  <ToggleGroup show={stateShowGrip} onToggle={() => setStateShowGrip(!stateShowGrip)} label="Grippers" color="green" />
                  <HideToggle hidden={stateSpeedHidden} onToggle={() => setStateSpeedHidden(!stateSpeedHidden)} />
                </div>
                {!stateSpeedHidden && (
                  <TimelineChart
                    data={stateSpeedData}
                    chartStep={ep.chartStep}
                    rangeStart={ep.rangeStart}
                    rangeEnd={ep.rangeEnd}
                    fps={ep.fps}
                    showLeft={stateShowLeft}
                    showRight={stateShowRight}
                    showGrip={stateShowGrip}
                    segments={[]}
                    activeSegIdx={-1}
                    rtgRange={{ start: rtgStart, end: rtgEnd, status: rtgStatus }}
                    onSeek={handleSeek}
                  />
                )}
              </div>
            )}
          </>
        )}
      </>
    ),

    states: (!isLabelMode && ep.stateAllData) ? (
      <div className="mt-3">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <span className="text-xs font-bold opacity-70">STATES</span>
          <ToggleGroup show={stateShowLeft} onToggle={() => setStateShowLeft(!stateShowLeft)} label="Left" color="blue" />
          <ToggleGroup show={stateShowRight} onToggle={() => setStateShowRight(!stateShowRight)} label="Right" color="red" />
          <ToggleGroup show={stateShowGrip} onToggle={() => setStateShowGrip(!stateShowGrip)} label="Grippers" color="green" />
        </div>
        <TimelineChart
          data={ep.stateAllData}
          chartStep={ep.chartStep}
          rangeStart={ep.rangeStart}
          rangeEnd={ep.rangeEnd}
          fps={ep.fps}
          showLeft={stateShowLeft}
          showRight={stateShowRight}
          showGrip={stateShowGrip}
          segments={isRtgMode ? [] : ep.segments}
          activeSegIdx={isRtgMode ? -1 : ep.activeSegIdx}
          rtgRange={isRtgMode ? { start: rtgStart, end: rtgEnd, status: rtgStatus } : undefined}
          onSeek={handleSeek}
        />
      </div>
    ) : null,

    labels: isLabelMode && ep.totalSteps > 0 ? (
      <ProgressLabelPanel
        keyframes={progressLabels.keyframes}
        chartStep={ep.chartStep}
        totalSteps={ep.totalSteps}
        dirty={progressLabels.dirty}
        saving={progressLabels.saving}
        interpolate={progressLabels.interpolate}
        onAddKeyFrame={progressLabels.addKeyFrame}
        onDeleteKeyFrame={progressLabels.deleteKeyFrame}
        onUpdateKeyFrame={progressLabels.updateKeyFrame}
        onNextKeyFrame={() => {
          const next = progressLabels.nextKeyFrame(ep.chartStep)
          if (next !== null) handleSeek(next)
        }}
        onPrevKeyFrame={() => {
          const prev = progressLabels.prevKeyFrame(ep.chartStep)
          if (prev !== null) handleSeek(prev)
        }}
        onSeek={handleSeek}
        onSave={progressLabels.save}
        hasKeyFrameAt={progressLabels.hasKeyFrameAt}
        valueTrimStart={ep.valueTrimStart}
        valueTrimEnd={ep.valueTrimEnd}
      />
    ) : null,

    frequency: !replayConnected && !isLabelMode && !isRtgMode && !ep.freqError ? (
      <div className="mt-4">
        {ep.freqLoading && (
          <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}
        {!ep.freqData && !ep.freqLoading && (
          <div className="bg-muted rounded-lg h-[280px] flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}
        {ep.freqData && (
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="text-xs font-semibold opacity-70">Frequency Spectrum (FFT)</span>
              <ToggleGroup show={freqShowLeft} onToggle={() => setFreqShowLeft(!freqShowLeft)} label="Left" color="blue" />
              <ToggleGroup show={freqShowRight} onToggle={() => setFreqShowRight(!freqShowRight)} label="Right" color="red" />
              <ToggleGroup show={freqShowGrip} onToggle={() => setFreqShowGrip(!freqShowGrip)} label="Grippers" color="green" />
              <HideToggle hidden={frequencyHidden} onToggle={() => setFrequencyHidden(!frequencyHidden)} />
              <Button variant="ghost" size="sm" onClick={ep.loadFrequency} className="h-6 text-xs">Refresh</Button>
            </div>
            {!frequencyHidden && (
              <FrequencyChart data={ep.freqData} showLeft={freqShowLeft} showRight={freqShowRight} showGrip={freqShowGrip} />
            )}
          </div>
        )}
      </div>
    ) : null,

    timestamps: !replayConnected && !isLabelMode && !isRtgMode && !ep.timestampError ? (
      <div className="mt-4">
        {ep.timestampLoading && (
          <div className="bg-muted rounded-lg h-[320px] flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}
        {!ep.timestampData && !ep.timestampLoading && (
          <div className="bg-muted rounded-lg h-[320px] flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        )}
        {ep.timestampData && (
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="text-xs font-semibold opacity-70">Component Timestamps</span>
              <HideToggle hidden={timestampHidden} onToggle={() => setTimestampHidden(!timestampHidden)} />
              <Button variant="ghost" size="sm" onClick={ep.loadTimestamps} className="h-6 text-xs">Refresh</Button>
            </div>
            {!timestampHidden && (
              <TimestampChart
                data={ep.timestampData}
                chartStep={ep.chartStep}
                totalSteps={ep.totalSteps}
                fps={ep.fps}
                onSeek={handleSeek}
              />
            )}
          </div>
        )}
      </div>
    ) : null,
  }

  return (
    <div className="mb-6">
      <div className="bg-card border rounded-xl p-4">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center mb-3 gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Badge>#{viewerIdx}</Badge>
            <span className={`font-semibold text-sm truncate ${ep.discarded ? "line-through opacity-50" : ""}`}>{ep.info.folder}</span>
            {ep.discarded && <Badge variant="destructive" className="text-[10px] shrink-0">Discarded</Badge>}
          </div>
          {(annotatedCount !== undefined || discardedCount !== undefined || unannotatedCount !== undefined) && (() => {
            const done = (annotatedCount ?? 0) + (discardedCount ?? 0)
            const total = episodeCount
            const pct = total > 0 ? Math.round((done / total) * 100) : 0
            return (
              <div className="flex items-center gap-4 text-[13px] font-mono font-semibold whitespace-nowrap">
                <span className="text-sky-400">▰ progress: {pct}%</span>
                <span className="text-green-500">✓ annotated: {annotatedCount ?? 0}</span>
                <span className="text-red-500">✕ discarded: {discardedCount ?? 0}</span>
                <span className="text-amber-500">○ unannotated: {unannotatedCount ?? 0}</span>
              </div>
            )
          })()}
          <div className="flex items-center gap-1 justify-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={panelOrder.resetOrder}
              title="Reset panel order"
              className="h-7 text-xs gap-1 text-muted-foreground hover:text-foreground"
            >
              <RotateCw className="h-3 w-3" />
            </Button>
            <Button
              variant={ep.discarded ? "outline" : "ghost"}
              size="sm"
              onClick={ep.toggleDiscard}
              disabled={ep.discardSaving}
              className={`h-7 text-xs gap-1 ${ep.discarded ? "border-green-600 text-green-600 hover:bg-green-600/10" : "text-red-500 hover:bg-red-500/10"}`}
            >
              {ep.discarded ? <><RotateCcw className="h-3 w-3" /> Enable</> : <><Trash2 className="h-3 w-3" /> Discard</>}
            </Button>
            {onCompletelyRemove && (ep.discarded || episodeEntry?.anomalous || ep.chartsError || ep.timestampError || ep.freqError) && (
              <Button
                variant="outline"
                size="sm"
                onClick={onCompletelyRemove}
                title="Permanently delete this episode's folder from disk"
                className="h-7 text-xs gap-1 border-red-500/70 text-red-300 bg-red-950/50 hover:bg-red-900/70"
              >
                <Trash2 className="h-3 w-3" /> Completely Remove
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={onClose} className="h-7 w-7"><X className="h-4 w-4" /></Button>
          </div>
        </div>

        {!ep.discarded && (ep.chartsError || ep.timestampError || ep.freqError) && (
          <div className="mb-3 p-3 rounded-lg border-2 border-amber-500/70 bg-amber-950/80 text-amber-100 text-xs">
            <div className="font-semibold mb-1">⚠ This episode has broken or missing data — affected panels are hidden.</div>
            <div className="opacity-80 space-y-0.5 font-mono text-[10px]">
              {ep.chartsError && <div>• Actions/States: {ep.chartsError}</div>}
              {ep.timestampError && <div>• Component Timestamps: {ep.timestampError}</div>}
              {ep.freqError && <div>• Frequency Spectrum: {ep.freqError}</div>}
            </div>
          </div>
        )}

        {panelOrder.order.map((id) => {
          const content = panelContent[id]
          if (!content) return null
          return (
            <DraggablePanel
              key={id}
              panelId={id}
              onDragStart={(pid) => panelOrder.onDragStart(pid as PanelId)}
              onDragOver={(pid) => panelOrder.onDragOver(pid as PanelId)}
              onDragEnd={panelOrder.onDragEnd}
              isDragTarget={panelOrder.dragTarget === id}
              dragPosition={panelOrder.dragTarget === id ? panelOrder.dragPosition : null}
            >
              {content}
            </DraggablePanel>
          )
        })}
      </div>
    </div>
  )
}

function ToggleGroup({ show, onToggle, label, color }: { show: boolean; onToggle: () => void; label: string; color: string }) {
  const bg = color === "blue" ? "bg-blue-500" : color === "red" ? "bg-red-500" : "bg-green-500"
  return (
    <Button variant={show ? "default" : "ghost"} size="sm" onClick={onToggle} className={`h-6 text-xs px-2 gap-1 ${show ? "" : "opacity-50"}`}>
      <span className={`w-2 h-2 rounded-full ${bg}`} /> {label}
    </Button>
  )
}
