// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useCallback, useMemo, useEffect, useRef } from "react"
import * as api from "@/api/client"

export interface KeyFrame {
  step: number
  progress: number
}

export function useProgressLabels(taskId: string, viewerIdx: number | null, totalSteps: number) {
  const [keyframes, setKeyframes] = useState<KeyFrame[]>([])
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const initialized = useRef(false)

  // Reset and load when episode changes
  useEffect(() => {
    initialized.current = false
    setDirty(false)
    setSaving(false)

    if (!taskId || viewerIdx === null || totalSteps <= 0) {
      setKeyframes([])
      return
    }

    const lastStep = Math.max(0, totalSteps - 1)

    const defaultKf: KeyFrame[] = [
      { step: 0, progress: 0 },
      { step: lastStep, progress: 1 },
    ]

    let cancelled = false
    api.fetchLabels(taskId, viewerIdx).then((data) => {
      if (cancelled) return
      if (data.exists && data.keyframes?.length >= 2) {
        setKeyframes(data.keyframes)
      } else {
        setKeyframes(defaultKf)
      }
      initialized.current = true
    }).catch(() => {
      if (cancelled) return
      setKeyframes(defaultKf)
      initialized.current = true
    })

    return () => { cancelled = true }
  }, [taskId, viewerIdx, totalSteps])

  const sorted = useMemo(
    () => [...keyframes].sort((a, b) => a.step - b.step),
    [keyframes],
  )

  const interpolate = useCallback(
    (step: number): number => {
      if (sorted.length === 0) return 0
      if (step <= sorted[0].step) return sorted[0].progress
      if (step >= sorted[sorted.length - 1].step) return sorted[sorted.length - 1].progress
      for (let i = 0; i < sorted.length - 1; i++) {
        const a = sorted[i]
        const b = sorted[i + 1]
        if (step >= a.step && step <= b.step) {
          if (a.step === b.step) return a.progress
          const t = (step - a.step) / (b.step - a.step)
          return a.progress + t * (b.progress - a.progress)
        }
      }
      return 0
    },
    [sorted],
  )

  const addKeyFrame = useCallback(
    (step: number) => {
      setKeyframes((prev) => {
        if (prev.some((kf) => kf.step === step)) return prev
        const progress = interpolateFrom(prev, step)
        return [...prev, { step, progress }]
      })
      setDirty(true)
    },
    [],
  )

  const deleteKeyFrame = useCallback(
    (step: number) => {
      setKeyframes((prev) => {
        const filtered = prev.filter((kf) => kf.step !== step)
        if (filtered.length === prev.length) return prev
        return filtered
      })
      setDirty(true)
    },
    [],
  )

  const updateKeyFrame = useCallback(
    (step: number, progress: number) => {
      const clamped = Math.max(0, Math.min(1, progress))
      setKeyframes((prev) =>
        prev.map((kf) => (kf.step === step ? { ...kf, progress: clamped } : kf)),
      )
      setDirty(true)
    },
    [],
  )

  const nextKeyFrame = useCallback(
    (currentStep: number): number | null => {
      for (const kf of sorted) {
        if (kf.step > currentStep) return kf.step
      }
      return null
    },
    [sorted],
  )

  const prevKeyFrame = useCallback(
    (currentStep: number): number | null => {
      for (let i = sorted.length - 1; i >= 0; i--) {
        if (sorted[i].step < currentStep) return sorted[i].step
      }
      return null
    },
    [sorted],
  )

  const hasKeyFrameAt = useCallback(
    (step: number): boolean => sorted.some((kf) => kf.step === step),
    [sorted],
  )

  const computeFullProgress = useCallback((): number[] => {
    const result: number[] = []
    for (let i = 0; i < totalSteps; i++) {
      result.push(interpolate(i))
    }
    return result
  }, [totalSteps, interpolate])

  const save = useCallback(async () => {
    if (!taskId || viewerIdx === null) return
    setSaving(true)
    try {
      const progress = computeFullProgress()
      await api.saveLabels(taskId, viewerIdx, progress, sorted)
      setDirty(false)
    } catch (e) {
      console.error("Failed to save labels:", e)
    } finally {
      setSaving(false)
    }
  }, [taskId, viewerIdx, computeFullProgress, sorted])

  return {
    keyframes: sorted,
    dirty,
    saving,
    interpolate,
    addKeyFrame,
    deleteKeyFrame,
    updateKeyFrame,
    nextKeyFrame,
    prevKeyFrame,
    hasKeyFrameAt,
    save,
  }
}

function interpolateFrom(keyframes: KeyFrame[], step: number): number {
  const sorted = [...keyframes].sort((a, b) => a.step - b.step)
  if (sorted.length === 0) return 0
  if (step <= sorted[0].step) return sorted[0].progress
  if (step >= sorted[sorted.length - 1].step) return sorted[sorted.length - 1].progress
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i]
    const b = sorted[i + 1]
    if (step >= a.step && step <= b.step) {
      if (a.step === b.step) return a.progress
      const t = (step - a.step) / (b.step - a.step)
      return a.progress + t * (b.progress - a.progress)
    }
  }
  return 0
}
