// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect, useRef } from "react"
import { ScanSearch, Loader2, AlertTriangle, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  autoScreenAll,
  fetchAutoScreenProgress,
  type AutoScreenProgress,
  type AutoScreenResult,
  type AutoScreenResultRow,
} from "@/api/client"

interface Props {
  taskId: string | null
  open: boolean
  onClose: () => void
  onOpenEpisode: (epIdx: number) => void
  onComplete?: (result: AutoScreenResult) => void
}

interface AutoScreenUiParams {
  minFrames: number
  latencyMs: number
  pureColorStdMax: number
  pureColorSubsampleStride: number
  pureColorMaxOffenders: number
  dryRun: boolean
}

const AUTO_SCREEN_STORAGE_KEY = "overlay-viz-auto-screen-params-v1"
const DEFAULT_AUTO_SCREEN: AutoScreenUiParams = {
  minFrames: 64,
  latencyMs: 50,
  pureColorStdMax: 5.0,
  pureColorSubsampleStride: 4,
  pureColorMaxOffenders: 20,
  dryRun: false,
}

function loadAutoScreenParams(): AutoScreenUiParams {
  try {
    const raw = localStorage.getItem(AUTO_SCREEN_STORAGE_KEY)
    if (!raw) return DEFAULT_AUTO_SCREEN
    const parsed = JSON.parse(raw) as Partial<AutoScreenUiParams>
    return {
      minFrames: typeof parsed.minFrames === "number" ? parsed.minFrames : DEFAULT_AUTO_SCREEN.minFrames,
      latencyMs: typeof parsed.latencyMs === "number" ? parsed.latencyMs : DEFAULT_AUTO_SCREEN.latencyMs,
      pureColorStdMax: typeof parsed.pureColorStdMax === "number" ? parsed.pureColorStdMax : DEFAULT_AUTO_SCREEN.pureColorStdMax,
      pureColorSubsampleStride: typeof parsed.pureColorSubsampleStride === "number" ? parsed.pureColorSubsampleStride : DEFAULT_AUTO_SCREEN.pureColorSubsampleStride,
      pureColorMaxOffenders: typeof parsed.pureColorMaxOffenders === "number" ? parsed.pureColorMaxOffenders : DEFAULT_AUTO_SCREEN.pureColorMaxOffenders,
      dryRun: typeof parsed.dryRun === "boolean" ? parsed.dryRun : DEFAULT_AUTO_SCREEN.dryRun,
    }
  } catch {
    return DEFAULT_AUTO_SCREEN
  }
}

function flagLabel(flag: string, row: AutoScreenResultRow): string {
  if (flag === "too_short") {
    const n = row.details.n_frames
    return n != null ? `too_short (${n} frames)` : "too_short"
  }
  if (flag === "latency") {
    const idxs = row.details.latency_indices ?? []
    const preview = idxs.slice(0, 3).join(", ")
    const more = idxs.length > 3 ? `, +${idxs.length - 3}` : ""
    const gap = row.details.max_gap_ms
    const gapStr = gap != null ? ` max ${gap.toFixed(1)}ms` : ""
    return idxs.length > 0 ? `latency @ [${preview}${more}]${gapStr}` : `latency${gapStr}`
  }
  if (flag === "pure_color") {
    const frs = row.details.pure_color_frames ?? []
    const preview = frs.slice(0, 3).join(", ")
    const more = frs.length > 3 ? `, +${frs.length - 3}` : ""
    return frs.length > 0 ? `pure_color @ [${preview}${more}]` : "pure_color"
  }
  return flag
}

export function AutoScreenDialog({ taskId, open, onClose, onOpenEpisode, onComplete }: Props) {
  const initial = useRef(loadAutoScreenParams()).current
  const [minFrames, setMinFrames] = useState(initial.minFrames)
  const [latencyMs, setLatencyMs] = useState(initial.latencyMs)
  const [pureColorStdMax, setPureColorStdMax] = useState(initial.pureColorStdMax)
  const [pureColorSubsampleStride, setPureColorSubsampleStride] = useState(initial.pureColorSubsampleStride)
  const [pureColorMaxOffenders, setPureColorMaxOffenders] = useState(initial.pureColorMaxOffenders)
  const [dryRun, setDryRun] = useState(initial.dryRun)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AutoScreenResult | null>(null)
  const [progress, setProgress] = useState<AutoScreenProgress | null>(null)

  // While a run is in flight, poll the server every 400ms for progress.
  // The POST holds the connection until everything is done, so without
  // polling the dialog would sit on "Running..." with no feedback.
  useEffect(() => {
    if (!busy || !taskId) return
    let cancelled = false
    const tick = async () => {
      try {
        const p = await fetchAutoScreenProgress(taskId)
        if (!cancelled) setProgress(p)
      } catch {
        /* swallow — the POST itself will surface real errors */
      }
    }
    tick()
    const id = setInterval(tick, 400)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [busy, taskId])

  useEffect(() => {
    try {
      localStorage.setItem(
        AUTO_SCREEN_STORAGE_KEY,
        JSON.stringify({
          minFrames, latencyMs, pureColorStdMax, pureColorSubsampleStride, pureColorMaxOffenders, dryRun,
        }),
      )
    } catch { /* ignore quota errors */ }
  }, [minFrames, latencyMs, pureColorStdMax, pureColorSubsampleStride, pureColorMaxOffenders, dryRun])

  if (!open) return null

  const handleRun = async () => {
    if (!taskId || busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    setProgress(null)
    try {
      const res = await autoScreenAll(taskId, {
        min_frames: minFrames,
        latency_threshold_s: latencyMs / 1000,
        pure_color_std_max: pureColorStdMax,
        pure_color_subsample_stride: pureColorSubsampleStride,
        pure_color_max_offenders: pureColorMaxOffenders,
        dry_run: dryRun,
      })
      setResult(res)
      onComplete?.(res)
    } catch (e) {
      // TypeError: Failed to fetch usually means the backend crashed mid-request
      // (e.g. cv2 segfault on a broken MP4). Check the terminal for a traceback.
      console.error("[AutoScreen] request failed:", e)
      const msg = e instanceof Error ? (e.message || e.name) : String(e)
      setError(msg.includes("fetch") ? `${msg} — check the terminal running "tbd data replay" for a Python traceback.` : msg)
    } finally {
      setBusy(false)
    }
  }

  const handleRowClick = (epIdx: number) => {
    onOpenEpisode(epIdx)
    onClose()
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-start justify-center pt-16"
      onClick={onClose}
    >
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-[640px] max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-2 border-b">
          <div className="flex items-center gap-2 text-sm font-bold tracking-wide">
            <ScanSearch className="h-4 w-4" /> AUTO-SCREEN
          </div>
          <button
            onClick={onClose}
            className="h-6 w-6 inline-flex items-center justify-center rounded hover:bg-muted text-muted-foreground"
            aria-label="Close"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="px-4 py-3 border-b grid grid-cols-2 gap-x-3 gap-y-2 text-[11px]">
          <label className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground">min frames</span>
            <input
              type="number"
              step={1}
              min={1}
              value={minFrames}
              onChange={(e) => setMinFrames(Math.max(1, parseInt(e.target.value, 10) || 1))}
              className="h-6 w-20 rounded border bg-muted px-1 font-mono outline-none"
            />
          </label>
          <label className="flex items-center justify-between gap-2" title="Flag any inter-frame gap exceeding this threshold (ms)">
            <span className="text-muted-foreground">latency (ms)</span>
            <input
              type="number"
              step={1}
              min={0}
              value={latencyMs}
              onChange={(e) => setLatencyMs(Math.max(0, parseInt(e.target.value, 10) || 0))}
              className="h-6 w-20 rounded border bg-muted px-1 font-mono outline-none"
            />
          </label>
          <label className="flex items-center justify-between gap-2" title="Per-channel std threshold below which a frame counts as pure-color">
            <span className="text-muted-foreground">std max</span>
            <input
              type="number"
              step={0.1}
              min={0}
              value={pureColorStdMax}
              onChange={(e) => setPureColorStdMax(Math.max(0, parseFloat(e.target.value) || 0))}
              className="h-6 w-20 rounded border bg-muted px-1 font-mono outline-none"
            />
          </label>
          <label className="flex items-center justify-between gap-2" title="Sample every Nth frame when scanning for pure-color">
            <span className="text-muted-foreground">subsample stride</span>
            <input
              type="number"
              step={1}
              min={1}
              value={pureColorSubsampleStride}
              onChange={(e) => setPureColorSubsampleStride(Math.max(1, parseInt(e.target.value, 10) || 1))}
              className="h-6 w-20 rounded border bg-muted px-1 font-mono outline-none"
            />
          </label>
          <label className="flex items-center justify-between gap-2" title="Stop after this many pure-color frames per episode">
            <span className="text-muted-foreground">max offenders</span>
            <input
              type="number"
              step={1}
              min={1}
              value={pureColorMaxOffenders}
              onChange={(e) => setPureColorMaxOffenders(Math.max(1, parseInt(e.target.value, 10) || 1))}
              className="h-6 w-20 rounded border bg-muted px-1 font-mono outline-none"
            />
          </label>
          <label
            className="flex items-center gap-2 cursor-pointer select-none text-muted-foreground"
            title="Preview which episodes would be flagged without writing discarded=true to any metadata.json. Use this to tune thresholds before committing."
          >
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            Dry run — preview only, no writes
          </label>
        </div>

        <div className="px-4 py-2 border-b flex items-center gap-2">
          <Button
            size="sm"
            variant="default"
            onClick={handleRun}
            disabled={busy || !taskId}
            className="h-7 text-[11px] px-2 gap-1"
          >
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <ScanSearch className="h-3 w-3" />}
            {busy ? "Running..." : "Run Auto-Screen"}
          </Button>
          {error && (
            <span className="text-[11px] text-red-400 flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" /> {error}
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {!result && !busy && (
            <div className="px-4 py-6 text-[11px] text-muted-foreground text-center">
              Run Auto-Screen to flag episodes that are too short, have latency spikes, or contain pure-color frames.
            </div>
          )}
          {busy && progress && (
            <>
              <div className="px-4 py-2 border-b bg-muted/30">
                <div className="flex items-center gap-2 text-[11px] mb-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span className="font-mono">{progress.done}/{progress.total}</span>
                  <span className="text-muted-foreground">episodes scanned</span>
                  <span className="ml-auto text-amber-400 font-mono">
                    {progress.flagged.length} flagged
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-primary h-1.5 transition-all"
                    style={{
                      width: progress.total > 0
                        ? `${(progress.done / progress.total) * 100}%`
                        : "0%",
                    }}
                  />
                </div>
              </div>
              {progress.flagged.length > 0 && (
                <ul>
                  {progress.flagged.map((row) => (
                    <li key={row.idx}>
                      <button
                        onClick={() => handleRowClick(row.idx)}
                        className="w-full text-left px-4 py-1.5 text-[11px] flex items-start gap-2 border-b border-border/50 hover:bg-accent transition-colors"
                      >
                        <span className="font-mono font-semibold shrink-0 w-12">Ep {row.idx}</span>
                        <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5 text-amber-400" />
                        <span className="flex-1 min-w-0">
                          <span className="opacity-70 mr-1">{row.folder}</span>
                          <span className="text-foreground">
                            {row.flags.map((f) => flagLabel(f, row)).join(", ")}
                          </span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
          {result && (
            <>
              <div className="px-4 py-2 border-b text-[11px] flex items-center gap-3 flex-wrap bg-muted/30">
                <span className="font-semibold">
                  {result.flagged}/{result.total} flagged
                </span>
                <span className="opacity-30">|</span>
                <span className="text-muted-foreground">too_short: <span className="font-mono text-foreground">{result.by_reason.too_short}</span></span>
                <span className="text-muted-foreground">latency: <span className="font-mono text-foreground">{result.by_reason.latency}</span></span>
                <span className="text-muted-foreground">pure_color: <span className="font-mono text-foreground">{result.by_reason.pure_color}</span></span>
                {result.dry_run && (
                  <span className="ml-auto text-amber-400 font-semibold">DRY RUN</span>
                )}
              </div>
              {result.results.length === 0 ? (
                <div className="px-4 py-6 text-[11px] text-muted-foreground text-center">
                  No episodes flagged.
                </div>
              ) : (
                <ul>
                  {result.results.map((row) => (
                    <li key={row.idx}>
                      <button
                        onClick={() => handleRowClick(row.idx)}
                        className="w-full text-left px-4 py-1.5 text-[11px] flex items-start gap-2 border-b border-border/50 hover:bg-accent transition-colors"
                      >
                        <span className="font-mono font-semibold shrink-0 w-12">Ep {row.idx}</span>
                        <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5 text-amber-400" />
                        <span className="flex-1 min-w-0">
                          <span className="opacity-70 mr-1">{row.folder}</span>
                          <span className="text-foreground">
                            {row.flags.map((f) => flagLabel(f, row)).join(", ")}
                          </span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
