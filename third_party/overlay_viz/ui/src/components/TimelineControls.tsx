import { useState, useEffect, useRef } from "react"
import { Play, Pause, RotateCcw, SkipForward, Save, Maximize, Plus, X, CheckCircle2, XCircle, Circle, Scissors, Loader2, Repeat } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { TrimSegment } from "@/api/types"

const SEG_COLORS = [
  { bg: "bg-sky-500/15", border: "border-sky-500/40", text: "text-sky-400", activeBg: "bg-sky-500/30" },
  { bg: "bg-amber-500/15", border: "border-amber-500/40", text: "text-amber-400", activeBg: "bg-amber-500/30" },
  { bg: "bg-violet-500/15", border: "border-violet-500/40", text: "text-violet-400", activeBg: "bg-violet-500/30" },
  { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", activeBg: "bg-emerald-500/30" },
  { bg: "bg-rose-500/15", border: "border-rose-500/40", text: "text-rose-400", activeBg: "bg-rose-500/30" },
  { bg: "bg-cyan-500/15", border: "border-cyan-500/40", text: "text-cyan-400", activeBg: "bg-cyan-500/30" },
]

export function segColor(idx: number) {
  return SEG_COLORS[idx % SEG_COLORS.length]
}

interface Props {
  chartStep: number
  rangeStart: number
  rangeEnd: number
  totalSteps: number
  fps: number
  replayConnected: boolean
  datasetPlaying: boolean
  trimSaving: boolean
  trimDirty: boolean
  mode?: string
  segments: TrimSegment[]
  activeSegIdx: number
  onStepChange: (step: number) => void
  onRangeChange: (start: number, end: number) => void
  onDatasetPlay: () => void
  onDatasetStop: () => void
  onSaveTrim: () => void
  onResetTrim: () => void
  onAutoClip?: (params: AutoClipUiParams) => Promise<{
    segments: { start: number; end: number }[]
    leading_skip: number
    trailing_skip: number
    dropped_short_segments: number
  } | null>
  onAutoClipAll?: (params: AutoClipUiParams, force?: boolean) => Promise<{
    total: number
    processed: number
    skipped: number
    errors: number
  } | null>
  autoClipOnLoad?: boolean
  onAutoClipOnLoadChange?: (value: boolean) => void

  onSelectSegment: (idx: number) => void
  onAddSegment: (atStep: number) => void
  onRemoveSegment: (idx: number) => void
  onRenameSegment: (idx: number, label: string) => void
  onRepeatChange: (idx: number, count: number) => void
  valueTrimStart?: number
  valueTrimEnd?: number
  valueTrimDirty?: boolean
  valueTrimSaving?: boolean
  onValueTrimChange?: (start: number, end: number) => void
  onSaveValueTrim?: () => void
  onResetValueTrim?: () => void
  onAddKeyFrame?: (step: number) => void
  rtgStart?: number
  rtgEnd?: number
  rtgStatus?: string | null
  rtgDirty?: boolean
  rtgSaving?: boolean
  onRtgStartChange?: (step: number) => void
  onRtgEndChange?: (step: number) => void
  onRtgStatusChange?: (status: "success" | "failure") => void
  onRtgSave?: () => void
  jointThreshold: number
  gripThreshold: number
  onJointThresholdChange: (v: number) => void
  onGripThresholdChange: (v: number) => void
  onSnapInToMovement: () => void
  onStartFromFirstFrame: () => void
  onSnapOutToMovement: () => void
  onEndAtLastFrame: () => void
  playbackSpeeds: readonly number[]
  playbackSpeedIdx: number
  onPlaybackSpeedIdxChange: (idx: number) => void
  chainPlaying?: boolean
  onSwitchToNextSegment?: () => void
  continuousPlay?: boolean
  onContinuousPlayChange?: (v: boolean) => void
}

interface AutoClipUiParams {
  threshold: number
  minIdleSteps: number
  startThreshold: number
  endThreshold: number
  minSegmentSteps: number
}

const AUTO_CLIP_STORAGE_KEY = "overlay-viz-auto-clip-params-v1"
export const AUTO_CLIP_ON_LOAD_STORAGE_KEY = "overlay-viz-auto-clip-on-load-v1"
const DEFAULT_AUTO_CLIP: AutoClipUiParams = {
  threshold: 0.005,
  minIdleSteps: 25,
  startThreshold: 0.02,
  endThreshold: 0.02,
  minSegmentSteps: 30,
}

export function loadAutoClipParams(): AutoClipUiParams {
  try {
    const raw = localStorage.getItem(AUTO_CLIP_STORAGE_KEY)
    if (!raw) return DEFAULT_AUTO_CLIP
    const parsed = JSON.parse(raw) as Partial<AutoClipUiParams>
    return {
      threshold: typeof parsed.threshold === "number" ? parsed.threshold : DEFAULT_AUTO_CLIP.threshold,
      minIdleSteps: typeof parsed.minIdleSteps === "number" ? parsed.minIdleSteps : DEFAULT_AUTO_CLIP.minIdleSteps,
      startThreshold: typeof parsed.startThreshold === "number" ? parsed.startThreshold : DEFAULT_AUTO_CLIP.startThreshold,
      endThreshold: typeof parsed.endThreshold === "number" ? parsed.endThreshold : DEFAULT_AUTO_CLIP.endThreshold,
      minSegmentSteps: typeof parsed.minSegmentSteps === "number" ? parsed.minSegmentSteps : DEFAULT_AUTO_CLIP.minSegmentSteps,
    }
  } catch {
    return DEFAULT_AUTO_CLIP
  }
}

export function TimelineControls(p: Props) {
  const maxStep = Math.max(1, p.totalSteps - 1)
  const stepSec = (p.chartStep / p.fps).toFixed(2)
  const totalSec = (p.totalSteps / p.fps).toFixed(2)
  const [editingLabel, setEditingLabel] = useState<number | null>(null)
  const [editValue, setEditValue] = useState("")
  const initialAutoClip = useRef(loadAutoClipParams()).current
  const [autoClipThreshold, setAutoClipThreshold] = useState(initialAutoClip.threshold)
  const [autoClipMinIdle, setAutoClipMinIdle] = useState(initialAutoClip.minIdleSteps)
  const [autoClipStartThreshold, setAutoClipStartThreshold] = useState(initialAutoClip.startThreshold)
  const [autoClipEndThreshold, setAutoClipEndThreshold] = useState(initialAutoClip.endThreshold)
  const [autoClipMinSegment, setAutoClipMinSegment] = useState(initialAutoClip.minSegmentSteps)
  const [autoClipBusy, setAutoClipBusy] = useState(false)
  const [autoClipNotice, setAutoClipNotice] = useState<string | null>(null)

  useEffect(() => {
    try {
      localStorage.setItem(
        AUTO_CLIP_STORAGE_KEY,
        JSON.stringify({
          threshold: autoClipThreshold,
          minIdleSteps: autoClipMinIdle,
          startThreshold: autoClipStartThreshold,
          endThreshold: autoClipEndThreshold,
          minSegmentSteps: autoClipMinSegment,
        }),
      )
    } catch { /* ignore quota errors */ }
  }, [autoClipThreshold, autoClipMinIdle, autoClipStartThreshold, autoClipEndThreshold, autoClipMinSegment])

  const currentAutoClipParams = (): AutoClipUiParams => ({
    threshold: autoClipThreshold,
    minIdleSteps: autoClipMinIdle,
    startThreshold: autoClipStartThreshold,
    endThreshold: autoClipEndThreshold,
    minSegmentSteps: autoClipMinSegment,
  })

  const handleAutoClip = async () => {
    if (!p.onAutoClip || autoClipBusy) return
    setAutoClipBusy(true)
    setAutoClipNotice(null)
    try {
      const res = await p.onAutoClip(currentAutoClipParams())
      if (res) {
        const parts = [`${res.segments.length} seg${res.segments.length === 1 ? "" : "s"}`]
        if (res.leading_skip > 0) parts.push(`skipped ${res.leading_skip} lead`)
        if (res.trailing_skip > 0) parts.push(`skipped ${res.trailing_skip} tail`)
        if (res.dropped_short_segments > 0) parts.push(`dropped ${res.dropped_short_segments} short`)
        setAutoClipNotice(parts.join(", "))
      }
    } catch (e) {
      setAutoClipNotice(`Failed: ${e}`)
    } finally {
      setAutoClipBusy(false)
    }
  }

  const handleAutoClipAll = async () => {
    if (!p.onAutoClipAll || autoClipBusy) return
    if (!window.confirm(
      "Auto-clip and SAVE every episode in this task that doesn't already have saved trim segments?\n\nThis modifies metadata.json files in place.",
    )) return
    setAutoClipBusy(true)
    setAutoClipNotice(null)
    try {
      const res = await p.onAutoClipAll(currentAutoClipParams(), false)
      if (res) {
        const parts = [`${res.processed}/${res.total} processed`]
        if (res.skipped > 0) parts.push(`${res.skipped} already-trimmed skipped`)
        if (res.errors > 0) parts.push(`${res.errors} errors`)
        setAutoClipNotice(parts.join(", "))
      }
    } catch (e) {
      setAutoClipNotice(`All failed: ${e}`)
    } finally {
      setAutoClipBusy(false)
    }
  }

  return (
    <div className="mb-3 space-y-2">
      {/* Row 1: Playback controls + timestamp */}
      <div className="flex items-center gap-3 p-2 bg-muted rounded-lg text-xs">
        {!p.replayConnected && (
          <div className="flex gap-1">
            {!p.datasetPlaying ? (
              <Button size="sm" onClick={p.onDatasetPlay} className="h-6 text-xs px-2">
                <Play className="h-3 w-3 mr-1" /> Play
              </Button>
            ) : (
              <Button size="sm" variant="secondary" onClick={p.onDatasetStop} className="h-6 text-xs px-2">
                <Pause className="h-3 w-3 mr-1" /> Stop
              </Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => p.onStepChange(p.rangeStart)} className="h-6 text-xs px-2">
              <RotateCcw className="h-3 w-3 mr-1" /> Reset
            </Button>
            <Button size="sm" variant="ghost" onClick={() => p.onStepChange(p.rangeEnd)} className="h-6 text-xs px-2">
              <SkipForward className="h-3 w-3 mr-1" /> End
            </Button>
          </div>
        )}
        <div className="flex items-center gap-1.5 font-mono">
          <Badge variant="outline" className="text-[10px] font-mono">t={p.chartStep}</Badge>
          <span className="opacity-60">{stepSec}s</span>
          <span className="opacity-30">/ {totalSec}s</span>
        </div>
        {!p.replayConnected && p.playbackSpeeds.length > 0 && (() => {
          const currentRate = p.playbackSpeeds[p.playbackSpeedIdx] ?? 1.0
          return (
            <div className="flex items-center gap-2 ml-auto" title="Video playback speed — does not affect saved step indices, trim, or labeling">
              <span className="text-[10px] font-semibold text-muted-foreground">Speed</span>
              <input
                type="range"
                min={0}
                max={p.playbackSpeeds.length - 1}
                step={1}
                value={p.playbackSpeedIdx}
                onChange={(e) => p.onPlaybackSpeedIdxChange(parseInt(e.target.value, 10))}
                className="h-2 w-28 accent-primary cursor-pointer"
              />
              <span className="text-[10px] font-mono w-10 text-right">{currentRate.toFixed(2)}×</span>
            </div>
          )
        })()}
      </div>

      {/* Row 2: Step scrubber */}
      <div className="flex items-center gap-2 px-1">
        <span className="text-[10px] font-semibold text-muted-foreground w-8">Step</span>
        <input
          type="range"
          min={0}
          max={maxStep}
          step={1}
          value={p.chartStep}
          onChange={(e) => p.onStepChange(parseInt(e.target.value, 10))}
          className="flex-1 h-2 accent-primary cursor-pointer"
        />
        <span className="text-[10px] font-mono text-muted-foreground w-10 text-right">{p.chartStep}</span>
      </div>

      {/* Row 3: Segment chips — hidden in label mode */}
      {p.mode !== "label" && (
        <>
          <div className="flex items-center gap-1.5 px-1 flex-wrap">
            <span className="text-[10px] font-semibold text-muted-foreground w-8">Segs</span>
            <button
              onClick={() => p.onSelectSegment(-1)}
              className={`h-6 text-[10px] px-2 rounded border font-medium transition-colors border-muted-foreground/40 ${p.activeSegIdx === -1 ? "bg-muted-foreground/20 text-foreground font-bold" : "text-muted-foreground bg-muted/40 opacity-70 hover:opacity-100"}`}
              title="Play the full untrimmed episode"
            >
              Raw episode
              <span className="ml-1 opacity-60 font-mono">0–{Math.max(0, p.totalSteps - 1)}</span>
            </button>
            {p.segments.map((seg, idx) => {
              const c = segColor(idx)
              const isActive = idx === p.activeSegIdx
              return (
                <div key={idx} className="flex items-center gap-0">
                  {editingLabel === idx ? (
                    <input
                      autoFocus
                      className="h-6 text-[10px] px-1.5 bg-muted border rounded w-24 outline-none"
                      value={editValue}
                      onChange={e => setEditValue(e.target.value)}
                      onBlur={() => { p.onRenameSegment(idx, editValue); setEditingLabel(null) }}
                      onKeyDown={e => {
                        if (e.key === "Enter") { p.onRenameSegment(idx, editValue); setEditingLabel(null) }
                        if (e.key === "Escape") setEditingLabel(null)
                      }}
                    />
                  ) : (
                    <button
                      onClick={() => p.onSelectSegment(idx)}
                      onDoubleClick={() => { setEditingLabel(idx); setEditValue(seg.label ?? `Seg ${idx + 1}`) }}
                      className={`h-6 text-[10px] px-2 rounded-l border font-medium transition-colors ${c.border} ${c.text} ${isActive ? c.activeBg + " font-bold" : c.bg + " opacity-70 hover:opacity-100"}`}
                    >
                      {seg.label || `Seg ${idx + 1}`}
                      <span className="ml-1 opacity-60 font-mono">{seg.start}–{seg.end}</span>
                      {(seg.repeat_last ?? 1) > 1 && (
                        <span className="ml-1 font-mono opacity-80">x{seg.repeat_last}</span>
                      )}
                    </button>
                  )}
                  {editingLabel !== idx && (
                    <button
                      onClick={() => p.onRemoveSegment(idx)}
                      title={p.segments.length > 1 ? "Remove this segment" : "Remove segment — switches to Raw episode view"}
                      className={`h-6 px-1 rounded-r border border-l-0 ${c.border} ${c.text} opacity-50 hover:opacity-100 hover:bg-red-500/20 transition-colors`}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )
            })}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => p.onAddSegment(p.chartStep)}
              className="h-6 text-xs px-2 gap-1"
            >
              <Plus className="h-3 w-3" /> Add
            </Button>
            {p.onSwitchToNextSegment && p.segments.length >= 2 && (
              <Button
                size="sm"
                variant={p.chainPlaying ? "default" : "ghost"}
                onClick={p.onSwitchToNextSegment}
                title={p.chainPlaying ? "Stop chained segment playback" : "Play the next segment and auto-advance through remaining segments"}
                className={`h-6 text-xs px-2 gap-1 ${p.chainPlaying ? "bg-green-500/30 text-green-300 hover:bg-green-500/40" : ""}`}
              >
                <SkipForward className="h-3 w-3" />
                {p.chainPlaying ? "Chaining…" : "Next seg"}
              </Button>
            )}
            {p.onContinuousPlayChange && (
              <Button
                size="sm"
                variant={p.continuousPlay ? "default" : "ghost"}
                onClick={() => p.onContinuousPlayChange?.(!p.continuousPlay)}
                title={p.continuousPlay ? "Stop at end of this episode instead of advancing" : "Play every segment back-to-back and auto-advance to the next episode until Stop"}
                className={`h-6 text-xs px-2 gap-1 ${p.continuousPlay ? "bg-green-500/30 text-green-300 hover:bg-green-500/40" : ""}`}
              >
                <Repeat className="h-3 w-3" />
                {p.continuousPlay ? "Continuous…" : "Continuous"}
              </Button>
            )}
          </div>

          {/* Row 4: IN/OUT controls for active segment (hidden in raw view) */}
          {p.activeSegIdx >= 0 && (
          <div className="flex items-center gap-2 px-1 flex-wrap">
            <span className="text-[10px] font-semibold text-muted-foreground w-8">Trim</span>
            <Button size="sm" variant="outline" onClick={() => p.onRangeChange(p.chartStep, Math.max(p.chartStep + 1, p.rangeEnd))} className="h-6 text-xs px-2 border-green-500/50 text-green-400 hover:bg-green-500/10">
              Set IN
            </Button>
            <Badge variant="outline" className="text-[10px] font-mono bg-green-500/10 text-green-400 border-green-500/30">
              IN {p.rangeStart}
            </Badge>
            <span className="opacity-30">—</span>
            <Badge variant="outline" className="text-[10px] font-mono bg-red-500/10 text-red-400 border-red-500/30">
              OUT {p.rangeEnd}
            </Badge>
            <Button size="sm" variant="outline" onClick={() => p.onRangeChange(Math.min(p.chartStep - 1, p.rangeStart), p.chartStep)} className="h-6 text-xs px-2 border-red-500/50 text-red-400 hover:bg-red-500/10">
              Set OUT
            </Button>
            <span className="opacity-20">|</span>
            <span className="text-[10px] text-muted-foreground">Repeat last</span>
            <select
              value={p.segments[p.activeSegIdx]?.repeat_last ?? 1}
              onChange={e => p.onRepeatChange(p.activeSegIdx, parseInt(e.target.value, 10))}
              className="h-6 text-[10px] px-1 bg-muted border rounded cursor-pointer outline-none"
            >
              {[1, 10, 20, 50].map(n => (
                <option key={n} value={n}>{n === 1 ? "1 (none)" : `${n}x`}</option>
              ))}
            </select>
            <span className="opacity-20">|</span>
            <Button
              size="sm"
              variant={p.trimDirty ? "default" : "ghost"}
              onClick={p.onSaveTrim}
              disabled={p.trimSaving}
              className={`h-6 text-xs px-2 ${p.trimDirty ? "" : "opacity-50"}`}
            >
              <Save className="h-3 w-3 mr-1" /> {p.trimSaving ? "Saving..." : "Save"}
            </Button>
            <Button size="sm" variant="ghost" onClick={p.onResetTrim} className="h-6 text-xs px-2">
              <Maximize className="h-3 w-3 mr-1" /> Full
            </Button>
            {p.onAutoClip && (
              <>
                <span className="opacity-20">|</span>
                <span className="text-[10px] text-muted-foreground">thresh</span>
                <input
                  type="number"
                  step={0.001}
                  min={0}
                  value={autoClipThreshold}
                  onChange={(e) => setAutoClipThreshold(Math.max(0, parseFloat(e.target.value) || 0))}
                  title="Idle threshold (max abs joint diff per step) for splitting"
                  className="h-6 w-16 rounded border bg-muted px-1 text-[10px] font-mono outline-none"
                />
                <span className="text-[10px] text-muted-foreground">min idle</span>
                <input
                  type="number"
                  step={1}
                  min={2}
                  value={autoClipMinIdle}
                  onChange={(e) => setAutoClipMinIdle(Math.max(2, parseInt(e.target.value, 10) || 2))}
                  title="Minimum consecutive idle steps to drop"
                  className="h-6 w-14 rounded border bg-muted px-1 text-[10px] font-mono outline-none"
                />
                <span className="text-[10px] text-muted-foreground">start thresh</span>
                <input
                  type="number"
                  step={0.001}
                  min={0}
                  value={autoClipStartThreshold}
                  onChange={(e) => setAutoClipStartThreshold(Math.max(0, parseFloat(e.target.value) || 0))}
                  title="Skip leading frames until any per-step diff reaches this value"
                  className="h-6 w-16 rounded border bg-muted px-1 text-[10px] font-mono outline-none"
                />
                <span className="text-[10px] text-muted-foreground">end thresh</span>
                <input
                  type="number"
                  step={0.001}
                  min={0}
                  value={autoClipEndThreshold}
                  onChange={(e) => setAutoClipEndThreshold(Math.max(0, parseFloat(e.target.value) || 0))}
                  title="Drop trailing frames after the last per-step diff reaching this value"
                  className="h-6 w-16 rounded border bg-muted px-1 text-[10px] font-mono outline-none"
                />
                <span className="text-[10px] text-muted-foreground">min seg</span>
                <input
                  type="number"
                  step={1}
                  min={1}
                  value={autoClipMinSegment}
                  onChange={(e) => setAutoClipMinSegment(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  title="Drop produced segments shorter than this many steps"
                  className="h-6 w-14 rounded border bg-muted px-1 text-[10px] font-mono outline-none"
                />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleAutoClip}
                  disabled={autoClipBusy}
                  className="h-6 text-xs px-2"
                >
                  {autoClipBusy ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Scissors className="h-3 w-3 mr-1" />}
                  Auto Clip
                </Button>
                {p.onAutoClipAll && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleAutoClipAll}
                    disabled={autoClipBusy}
                    title="Run auto-clip on every episode without saved trim segments and persist the result"
                    className="h-6 text-xs px-2"
                  >
                    {autoClipBusy ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Scissors className="h-3 w-3 mr-1" />}
                    Auto Clip All & Save
                  </Button>
                )}
                {p.onAutoClipOnLoadChange && (
                  <label
                    className="flex items-center gap-1 text-[10px] text-muted-foreground cursor-pointer select-none"
                    title="When enabled, auto-clip runs automatically on each episode that has no saved trim segments. Result is left dirty — Save to persist."
                  >
                    <input
                      type="checkbox"
                      checked={!!p.autoClipOnLoad}
                      onChange={(e) => p.onAutoClipOnLoadChange?.(e.target.checked)}
                      className="h-3 w-3"
                    />
                    on load
                  </label>
                )}
                {autoClipNotice && (
                  <span className="text-[10px] text-muted-foreground">{autoClipNotice}</span>
                )}
              </>
            )}
          </div>
          )}
          {/* Row 5: Auto-IN based on first-movement heuristic (hidden in raw view) */}
          {p.activeSegIdx >= 0 && (
          <div className="flex items-center gap-2 px-1 flex-wrap">
            <span className="text-[10px] font-semibold text-muted-foreground w-8">Auto</span>
            <Button size="sm" variant="outline" onClick={p.onSnapInToMovement} className="h-6 text-xs px-2" title="Set trim IN to the first step where joint L∞ or gripper speed exceeds threshold">
              Snap Trimming IN to first movement
            </Button>
            <Button size="sm" variant="ghost" onClick={p.onStartFromFirstFrame} className="h-6 text-xs px-2" title="Reset trim IN to step 0">
              Start from first frame
            </Button>
            <span className="opacity-20">|</span>
            <Button size="sm" variant="outline" onClick={p.onSnapOutToMovement} className="h-6 text-xs px-2" title="Set trim OUT to the last step where joint L∞ or gripper speed exceeds threshold">
              Snap Trimming OUT to last movement
            </Button>
            <Button size="sm" variant="ghost" onClick={p.onEndAtLastFrame} className="h-6 text-xs px-2" title="Reset trim OUT to the final frame">
              End at last frame
            </Button>
            <span className="opacity-20">|</span>
            <span className="text-[10px] text-muted-foreground">Idle threshold:</span>
            <label className="flex items-center gap-1 text-[10px] text-muted-foreground">
              joint speed &gt;
              <input
                type="number"
                step="1e-4"
                min={0}
                value={p.jointThreshold}
                onChange={e => p.onJointThresholdChange(Math.max(0, parseFloat(e.target.value) || 0))}
                className="h-6 w-20 rounded border bg-background px-1 text-[10px] font-mono outline-none"
                title="L∞ threshold on non-gripper joint state deltas"
              />
            </label>
            <label className="flex items-center gap-1 text-[10px] text-muted-foreground">
              gripper speed &gt;
              <input
                type="number"
                step="1e-4"
                min={0}
                value={p.gripThreshold}
                onChange={e => p.onGripThresholdChange(Math.max(0, parseFloat(e.target.value) || 0))}
                className="h-6 w-20 rounded border bg-background px-1 text-[10px] font-mono outline-none"
                title="Absolute threshold on gripper state delta"
              />
            </label>
          </div>
          )}
        </>
      )}

      {/* Label mode: value trim (single segment for value learning) */}
      {p.mode === "label" && p.onValueTrimChange && (
        <div className="flex items-center gap-2 px-1 flex-wrap">
          <span className="text-[10px] font-semibold text-muted-foreground w-8">VTrim</span>
          <Button size="sm" variant="outline" onClick={() => { p.onValueTrimChange!(p.chartStep, Math.max(p.chartStep + 1, p.valueTrimEnd ?? maxStep)); p.onAddKeyFrame?.(p.chartStep) }} disabled={p.chartStep >= (p.valueTrimEnd ?? maxStep) - 1} className="h-6 text-xs px-2 border-green-500/50 text-green-400 hover:bg-green-500/10">
            Set IN
          </Button>
          <Badge variant="outline" className="text-[10px] font-mono bg-green-500/10 text-green-400 border-green-500/30">
            IN {p.valueTrimStart ?? 0}
          </Badge>
          <span className="opacity-30">—</span>
          <Badge variant="outline" className="text-[10px] font-mono bg-red-500/10 text-red-400 border-red-500/30">
            OUT {p.valueTrimEnd ?? maxStep}
          </Badge>
          <Button size="sm" variant="outline" onClick={() => { p.onValueTrimChange!(Math.min(p.chartStep - 1, p.valueTrimStart ?? 0), p.chartStep); p.onAddKeyFrame?.(p.chartStep) }} disabled={p.chartStep <= (p.valueTrimStart ?? 0) + 1} className="h-6 text-xs px-2 border-red-500/50 text-red-400 hover:bg-red-500/10">
            Set OUT
          </Button>
          <span className="opacity-20">|</span>
          <Button
            size="sm"
            variant={p.valueTrimDirty ? "default" : "ghost"}
            onClick={p.onSaveValueTrim}
            disabled={p.valueTrimSaving}
            className={`h-6 text-xs px-2 ${p.valueTrimDirty ? "" : "opacity-50"}`}
          >
            <Save className="h-3 w-3 mr-1" /> {p.valueTrimSaving ? "Saving..." : "Save"}
          </Button>
          <Button size="sm" variant="ghost" onClick={p.onResetValueTrim} className="h-6 text-xs px-2">
            <Maximize className="h-3 w-3 mr-1" /> Full
          </Button>
        </div>
      )}

      {/* RTG mode: range + status */}
      {p.mode === "rtg" && p.onRtgStartChange && (
        <div className="flex items-center gap-2 px-1 flex-wrap">
          <span className="text-[10px] font-semibold text-muted-foreground w-8">RTG</span>
          <Button size="sm" variant="outline" onClick={() => p.onRtgStartChange!(p.chartStep)} disabled={p.chartStep >= (p.rtgEnd ?? maxStep)} className="h-6 text-xs px-2 border-green-500/50 text-green-400 hover:bg-green-500/10">
            Set IN
          </Button>
          <Badge variant="outline" className="text-[10px] font-mono bg-green-500/10 text-green-400 border-green-500/30">
            IN {p.rtgStart ?? 0}
          </Badge>
          <span className="opacity-30">—</span>
          <Badge variant="outline" className="text-[10px] font-mono bg-red-500/10 text-red-400 border-red-500/30">
            OUT {p.rtgEnd ?? maxStep}
          </Badge>
          <Button size="sm" variant="outline" onClick={() => p.onRtgEndChange!(p.chartStep)} disabled={p.chartStep <= (p.rtgStart ?? 0)} className="h-6 text-xs px-2 border-red-500/50 text-red-400 hover:bg-red-500/10">
            Set OUT
          </Button>
          <span className="opacity-20">|</span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => p.onRtgStatusChange!("success")}
            className={`h-6 text-xs px-2 gap-1 border-green-500/50 ${p.rtgStatus === "success" ? "bg-green-500/20 text-green-400" : "text-green-500/60 hover:bg-green-500/10"}`}
          >
            <CheckCircle2 className="h-3 w-3" /> Success
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => p.onRtgStatusChange!("failure")}
            className={`h-6 text-xs px-2 gap-1 border-red-500/50 ${p.rtgStatus === "failure" ? "bg-red-500/20 text-red-400" : "text-red-500/60 hover:bg-red-500/10"}`}
          >
            <XCircle className="h-3 w-3" /> Failure
          </Button>
          <span className="opacity-20">|</span>
          <Button
            size="sm"
            variant={p.rtgDirty ? "default" : "ghost"}
            onClick={p.onRtgSave}
            disabled={p.rtgSaving}
            className={`h-6 text-xs px-2 ${p.rtgDirty ? "" : "opacity-50"}`}
          >
            <Save className="h-3 w-3 mr-1" /> {p.rtgSaving ? "Saving..." : "Save"}
          </Button>
          {p.rtgDirty && !p.rtgSaving && (
            <span className="flex items-center gap-1 text-[10px] text-amber-400">
              <Circle className="h-2 w-2 fill-amber-400" /> unsaved
            </span>
          )}
        </div>
      )}
    </div>
  )
}
