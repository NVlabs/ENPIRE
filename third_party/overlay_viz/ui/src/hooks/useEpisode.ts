// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useCallback, useEffect, useRef } from "react"
import * as api from "@/api/client"
import type {
  EpisodeInfo,
  TrimSegment,
  ChartData,
  ActionSourceData,
  FrequencyData,
  ComponentTimestampData,
  ValuePredictionData,
  ValuePredictionPrecomputeStatus,
} from "@/api/types"

export function useEpisode(taskId: string, viewerIdx: number | null) {
  const [info, setInfo] = useState<EpisodeInfo | null>(null)
  const [chartStep, setChartStep] = useState(0)
  const [segments, setSegments] = useState<TrimSegment[]>([])
  const [activeSegIdx, setActiveSegIdx] = useState(0)
  const [chartAllData, setChartAllData] = useState<ChartData | null>(null)
  const [stateAllData, setStateAllData] = useState<ChartData | null>(null)
  const [actionSourceData, setActionSourceData] = useState<ActionSourceData | null>(null)
  const [freqData, setFreqData] = useState<FrequencyData | null>(null)
  const [valuePredictionData, setValuePredictionData] = useState<ValuePredictionData | null>(null)
  const [chartsLoading, setChartsLoading] = useState(false)
  const [chartsError, setChartsError] = useState<string | null>(null)
  const [freqLoading, setFreqLoading] = useState(false)
  const [freqError, setFreqError] = useState<string | null>(null)
  const [timestampData, setTimestampData] = useState<ComponentTimestampData | null>(null)
  const [timestampLoading, setTimestampLoading] = useState(false)
  const [timestampError, setTimestampError] = useState<string | null>(null)
  const [valuePredictionLoading, setValuePredictionLoading] = useState(false)
  const [valuePredictionError, setValuePredictionError] = useState<string | null>(null)
  const [valueTaskPrecomputeStatus, setValueTaskPrecomputeStatus] = useState<ValuePredictionPrecomputeStatus | null>(null)
  const [trimSaving, setTrimSaving] = useState(false)
  const [trimDirty, setTrimDirty] = useState(false)
  const [valueTrimStart, setValueTrimStartRaw] = useState(0)
  const [valueTrimEnd, setValueTrimEndRaw] = useState(0)
  const [valueTrimSaving, setValueTrimSaving] = useState(false)
  const [valueTrimDirty, setValueTrimDirty] = useState(false)
  const [discarded, setDiscarded] = useState(false)
  const [discardSaving, setDiscardSaving] = useState(false)
  const [playbackRate, setPlaybackRateRaw] = useState(1.0)

  const onSaveNotice = useRef<((title: string, savedPath: string, episodeDir: string) => void) | null>(null)
  const setOnSaveNotice = useCallback((cb: (title: string, savedPath: string, episodeDir: string) => void) => {
    onSaveNotice.current = cb
  }, [])

  const totalSteps = chartAllData?.total_steps ?? info?.total_frames ?? 0
  const fps = info?.fps ?? 30

  const activeSeg = segments[activeSegIdx] ?? segments[0]
  const isRawView = activeSegIdx === -1
  const rangeStart = isRawView ? 0 : (activeSeg?.start ?? 0)
  const rangeEnd = isRawView ? Math.max(0, totalSteps - 1) : (activeSeg?.end ?? 0)

  const videoRefs = useRef<Set<HTMLVideoElement>>(new Set())
  const registerVideo = useCallback((el: HTMLVideoElement | null) => {
    if (el) {
      videoRefs.current.add(el)
    }
  }, [])

  const fpsRef = useRef(fps)
  fpsRef.current = fps

  const liveVideos = useCallback(() => {
    const live: HTMLVideoElement[] = []
    for (const v of videoRefs.current) {
      if (v.isConnected) live.push(v)
      else videoRefs.current.delete(v)
    }
    return live
  }, [])

  // Coalesce rapid seeks into one per animation frame. Scrubber input events
  // fire faster than the video decoder can seek — unthrottled, the decoder
  // falls behind and the panel displays a stale or wrong frame.
  const pendingSeekStep = useRef<number | null>(null)
  const seekRafId = useRef<number | null>(null)

  const syncVideos = useCallback((step: number) => {
    pendingSeekStep.current = step
    if (seekRafId.current !== null) return
    seekRafId.current = requestAnimationFrame(() => {
      seekRafId.current = null
      const target = pendingSeekStep.current
      pendingSeekStep.current = null
      if (target === null) return
      const t = target / fpsRef.current
      for (const v of liveVideos()) {
        if (v.readyState < 1) continue  // HAVE_METADATA — needed to set currentTime
        if (v.seeking) continue          // skip while a prior seek is still in flight
        if (Math.abs(v.currentTime - t) > 0.05) {
          try { v.currentTime = t } catch { /* ignore */ }
        }
      }
    })
  }, [liveVideos])

  const playbackRateRef = useRef(playbackRate)
  playbackRateRef.current = playbackRate

  const playVideos = useCallback((fromStep: number) => {
    const t = fromStep / fpsRef.current
    for (const v of liveVideos()) {
      if (v.readyState < 1) continue
      try {
        v.currentTime = t
        v.playbackRate = playbackRateRef.current
        v.play().catch(() => {})
      } catch { /* ignore */ }
    }
  }, [liveVideos])

  const setPlaybackRate = useCallback((rate: number) => {
    setPlaybackRateRaw(rate)
    for (const v of liveVideos()) {
      try { v.playbackRate = rate } catch { /* ignore */ }
    }
  }, [liveVideos])

  const pauseVideos = useCallback(() => {
    for (const v of liveVideos()) {
      try { v.pause() } catch { /* ignore */ }
    }
  }, [liveVideos])

  const getVideoTime = useCallback((): number | null => {
    for (const v of liveVideos()) {
      if (v.readyState >= 1) return v.currentTime
    }
    return null
  }, [liveVideos])

  const chartsWereLoaded = useRef(false)
  const precomputeWasRunning = useRef(false)

  const applyInfo = useCallback((ep: EpisodeInfo) => {
    setInfo(ep)
    const segs = ep.trim_segments?.length
      ? ep.trim_segments
      : [{ start: ep.trim_start_frame ?? 0, end: ep.trim_end_frame ?? Math.max(0, ep.total_frames - 1) }]
    setSegments(segs)
    setActiveSegIdx(0)
    setValueTrimStartRaw(ep.value_trim_start ?? 0)
    setValueTrimEndRaw(ep.value_trim_end ?? Math.max(0, ep.total_frames - 1))
    setDiscarded(ep.discarded ?? false)
  }, [])

  const refreshInfo = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    const ep = await api.fetchEpisodeInfo(taskId, viewerIdx)
    applyInfo(ep)
  }, [taskId, viewerIdx, applyInfo])

  useEffect(() => {
    // Reset ALL state immediately on any episode change to avoid stale data
    setInfo(null)
    setChartAllData(null)
    setStateAllData(null)
    setActionSourceData(null)
    setFreqData(null)
    setTimestampData(null)
    setValuePredictionData(null)
    setChartsError(null)
    setTimestampError(null)
    setFreqError(null)
    setValuePredictionError(null)
    setValuePredictionLoading(false)
    setChartStep(0)
    setSegments([])
    setActiveSegIdx(0)
    setValueTrimStartRaw(0)
    setValueTrimEndRaw(0)
    setValueTrimDirty(false)
    setTrimDirty(false)
    setDiscarded(false)
    videoRefs.current.clear()
    pendingSeekStep.current = null
    if (seekRafId.current !== null) {
      cancelAnimationFrame(seekRafId.current)
      seekRafId.current = null
    }

    if (!taskId || viewerIdx === null) return

    let cancelled = false
    api.fetchEpisodeInfo(taskId, viewerIdx).then((ep) => {
      if (cancelled) return
      applyInfo(ep)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [taskId, viewerIdx, applyInfo])

  const loadCharts = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    setChartsLoading(true)
    setChartsError(null)
    try {
      const [act, st, src] = await Promise.all([
        api.fetchActions(taskId, viewerIdx),
        api.fetchStates(taskId, viewerIdx).catch(() => null),
        api.fetchActionSource(taskId, viewerIdx).catch(() => null),
      ])
      setChartAllData(act)
      setStateAllData(st)
      setActionSourceData(src)
      chartsWereLoaded.current = true
    } catch (e) {
      console.error("loadCharts failed:", e)
      setChartsError(String(e))
    } finally {
      setChartsLoading(false)
    }
  }, [taskId, viewerIdx])

  // Auto-load charts when switching episodes if charts were previously loaded
  useEffect(() => {
    if (info && chartsWereLoaded.current && !chartAllData && !chartsLoading && !chartsError) {
      loadCharts()
    }
  }, [info, chartAllData, chartsLoading, chartsError, loadCharts])

  const loadFrequency = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    setFreqLoading(true)
    setFreqError(null)
    try {
      setFreqData(await api.fetchFrequency(taskId, viewerIdx))
    } catch (e) {
      setFreqData(null)
      setFreqError(String(e))
    } finally {
      setFreqLoading(false)
    }
  }, [taskId, viewerIdx])

  const loadTimestamps = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    setTimestampLoading(true)
    setTimestampError(null)
    try {
      setTimestampData(await api.fetchComponentTimestamps(taskId, viewerIdx))
    } catch (e) {
      console.error("loadTimestamps failed:", e)
      setTimestampData(null)
      setTimestampError(String(e))
    } finally {
      setTimestampLoading(false)
    }
  }, [taskId, viewerIdx])

  const loadValuePredictions = useCallback(async (options?: {
    force?: boolean
    preferLocal?: boolean
    batchSize?: number
    relativeInterval?: number
    prompt?: string
    advMode?: string
    advantageH?: number
    rtgGamma?: number
  }) => {
    if (!taskId || viewerIdx === null) return
    setValuePredictionLoading(true)
    setValuePredictionError(null)
    const keepExisting = Boolean(valuePredictionData)
    try {
      const next = await api.fetchValuePredictions(taskId, viewerIdx, options)
      setValuePredictionData(next)
      setInfo(prev => prev ? { ...prev, value_predictions_cached: true, value_predictions_cache_format: next.cache_format ?? prev.value_predictions_cache_format } : prev)
    } catch (e) {
      console.error("loadValuePredictions failed:", e)
      setValuePredictionError(String(e))
      if (!keepExisting) setValuePredictionData(null)
    } finally {
      setValuePredictionLoading(false)
    }
  }, [taskId, viewerIdx, valuePredictionData])

  const refreshTaskValuePredictionPrecomputeStatus = useCallback(async () => {
    if (!taskId) return
    setValueTaskPrecomputeStatus(await api.fetchTaskValuePredictionPrecomputeStatus(taskId))
  }, [taskId])

  const startTaskValuePredictionPrecompute = useCallback(async (options?: {
    force?: boolean
    batchSize?: number
    relativeInterval?: number
    prompt?: string
  }) => {
    if (!taskId) return
    const status = await api.startTaskValuePredictionPrecompute(taskId, options)
    setValueTaskPrecomputeStatus(status)
  }, [taskId])

  useEffect(() => {
    if (!taskId) return
    refreshTaskValuePredictionPrecomputeStatus().catch(() => {})
  }, [taskId, refreshTaskValuePredictionPrecomputeStatus])

  useEffect(() => {
    if (!taskId || !valueTaskPrecomputeStatus?.running) return
    const id = setInterval(() => {
      refreshTaskValuePredictionPrecomputeStatus().catch(() => {})
    }, 1000)
    return () => clearInterval(id)
  }, [taskId, valueTaskPrecomputeStatus?.running, refreshTaskValuePredictionPrecomputeStatus])

  useEffect(() => {
    if (!taskId) return
    const running = Boolean(valueTaskPrecomputeStatus?.running)
    if (running) precomputeWasRunning.current = true
    if (!running && precomputeWasRunning.current) {
      precomputeWasRunning.current = false
      refreshInfo().catch(() => {})
    }
  }, [taskId, valueTaskPrecomputeStatus?.running, refreshInfo])

  const setRangeStart = useCallback((start: number) => {
    if (activeSegIdx < 0) return
    setSegments(prev => prev.map((s, i) => i === activeSegIdx ? { ...s, start } : s))
    setTrimDirty(true)
  }, [activeSegIdx])

  const setRangeEnd = useCallback((end: number) => {
    if (activeSegIdx < 0) return
    setSegments(prev => prev.map((s, i) => i === activeSegIdx ? { ...s, end } : s))
    setTrimDirty(true)
  }, [activeSegIdx])

  const addSegment = useCallback((atStep: number) => {
    const last = Math.max(1, totalSteps - 1)
    setSegments(prev => {
      const following = prev.filter(s => s.start > atStep).sort((a, b) => a.start - b.start)
      const end = following.length > 0 ? following[0].start : last
      if (end <= atStep) return prev
      const newSeg: TrimSegment = { start: atStep, end }
      const next = [...prev, newSeg].sort((a, b) => a.start - b.start)
      setActiveSegIdx(next.findIndex(s => s.start === newSeg.start && s.end === newSeg.end))
      return next
    })
    setTrimDirty(true)
  }, [totalSteps])

  const removeSegment = useCallback((idx: number) => {
    setSegments(prev => {
      const next = prev.filter((_, i) => i !== idx)
      if (next.length === 0) {
        // Removed the last segment → drop into raw view so the operator
        // can Add a fresh one or leave the episode untrimmed.
        setActiveSegIdx(-1)
      } else {
        setActiveSegIdx(a => Math.min(Math.max(a, 0), next.length - 1))
      }
      return next
    })
    setTrimDirty(true)
  }, [])

  const updateSegmentLabel = useCallback((idx: number, label: string) => {
    setSegments(prev => prev.map((s, i) => i === idx ? { ...s, label: label || undefined } : s))
    setTrimDirty(true)
  }, [])

  const updateSegmentRepeat = useCallback((idx: number, count: number) => {
    setSegments(prev => prev.map((s, i) => i === idx ? { ...s, repeat_last: count } : s))
    setTrimDirty(true)
  }, [])

  const saveTrim = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    setTrimSaving(true)
    try {
      const res = await api.saveTrim(taskId, viewerIdx, segments)
      setTrimDirty(false)
      onSaveNotice.current?.("Saved trim", res.saved_path, res.episode_dir)
    } catch (e) {
      console.error("Failed to save trim:", e);
      setTrimSaving(false);
      return;  // don't clear dirty flag on failure
    }
    setTrimSaving(false)
  }, [taskId, viewerIdx, segments])

  const resetTrim = useCallback(() => {
    setSegments([{ start: 0, end: Math.max(1, totalSteps - 1) }])
    setActiveSegIdx(0)
    setTrimDirty(true)
  }, [totalSteps])

  const autoClip = useCallback(async (params: api.AutoClipParams) => {
    if (!taskId || viewerIdx === null) return null
    const res = await api.autoClipEpisode(taskId, viewerIdx, params)
    setSegments(res.segments)
    setActiveSegIdx(0)
    setTrimDirty(true)
    return res
  }, [taskId, viewerIdx])

  const autoClipAllAndSave = useCallback(async (params: api.AutoClipParams, force = false) => {
    if (!taskId) return null
    const res = await api.autoClipAll(taskId, params, force)
    // Refresh current episode info in case it was just modified.
    if (viewerIdx !== null) {
      try {
        const ep = await api.fetchEpisodeInfo(taskId, viewerIdx)
        applyInfo(ep)
        setTrimDirty(false)
      } catch { /* ignore */ }
    }
    return res
  }, [taskId, viewerIdx, applyInfo])

  const setValueTrimStart = useCallback((v: number) => {
    setValueTrimStartRaw(v); setValueTrimDirty(true)
  }, [])
  const setValueTrimEnd = useCallback((v: number) => {
    setValueTrimEndRaw(v); setValueTrimDirty(true)
  }, [])
  const saveValueTrim = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    setValueTrimSaving(true)
    try {
      const res = await api.saveValueTrim(taskId, viewerIdx, valueTrimStart, valueTrimEnd)
      setValueTrimDirty(false)
      onSaveNotice.current?.("Saved value trim", res.saved_path, res.episode_dir)
    } catch { /* ignore */ } finally {
      setValueTrimSaving(false)
    }
  }, [taskId, viewerIdx, valueTrimStart, valueTrimEnd])
  const resetValueTrim = useCallback(() => {
    setValueTrimStartRaw(0)
    setValueTrimEndRaw(Math.max(1, totalSteps - 1))
    setValueTrimDirty(true)
  }, [totalSteps])

  const onDiscardChanged = useRef<(() => void) | null>(null)

  const toggleDiscard = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    const next = !discarded
    setDiscardSaving(true)
    try {
      await api.setDiscard(taskId, viewerIdx, next)
      setDiscarded(next)
      onDiscardChanged.current?.()
    } catch { /* ignore */ } finally {
      setDiscardSaving(false)
    }
  }, [taskId, viewerIdx, discarded])

  const setOnDiscardChanged = useCallback((cb: () => void) => {
    onDiscardChanged.current = cb
  }, [])

  return {
    info, chartStep, setChartStep,
    segments, activeSegIdx, setActiveSegIdx,
    rangeStart, setRangeStart, rangeEnd, setRangeEnd,
    addSegment, removeSegment, updateSegmentLabel, updateSegmentRepeat,
    chartAllData, stateAllData, actionSourceData,
    freqData, freqLoading, freqError, timestampData, timestampLoading, timestampError,
    valuePredictionData, valuePredictionLoading, valuePredictionError,
    valueTaskPrecomputeStatus,
    chartsLoading, chartsError, trimSaving, trimDirty,
    valueTrimStart, setValueTrimStart, valueTrimEnd, setValueTrimEnd,
    valueTrimSaving, valueTrimDirty, saveValueTrim, resetValueTrim,
    discarded, discardSaving, toggleDiscard, setOnDiscardChanged,
    setOnSaveNotice,
    totalSteps, fps,
    loadCharts, loadFrequency, loadTimestamps, loadValuePredictions, saveTrim, resetTrim, autoClip, autoClipAllAndSave, refreshInfo,
    startTaskValuePredictionPrecompute, refreshTaskValuePredictionPrecomputeStatus,
    registerVideo, syncVideos, playVideos, pauseVideos, getVideoTime,
    playbackRate, setPlaybackRate,
  }
}
