// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRef, useEffect, useMemo, useState, useCallback } from "react"
import type { ComponentTimestampData } from "@/api/types"

interface Props {
  data: ComponentTimestampData
  chartStep: number
  totalSteps: number
  fps: number
  onSeek: (step: number) => void
}

const COMPONENT_COLORS: Record<string, string> = {
  left_state: "#ff6b6b",
  right_state: "#6b6bff",
  top_camera: "#ffa94d",
  left_camera: "#ff8787",
  right_camera: "#748ffc",
  action: "#51cf66",
  action_source: "#9775fa",
}

const COMPONENT_ORDER = [
  "left_state", "right_state",
  "top_camera", "left_camera", "right_camera",
  "action", "action_source",
]

const MARGIN = { top: 30, right: 20, bottom: 40, left: 120 }
const ROW_HEIGHT = 36
const DOT_RADIUS = 2.5
const DEFAULT_VIEW_SECONDS = 1.0
const SPREAD_THRESHOLD_SECONDS = 0.050

const ACTION_SRC_COLORS: Record<string, string> = { human: "#ff4444", policy: "#44cc44" }

export function TimestampChart({ data, chartStep, totalSteps, fps, onSeek }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  const [viewXMin, setViewXMin] = useState(0)
  const [viewXMax, setViewXMax] = useState(1)
  const [showRecordLines, setShowRecordLines] = useState(false)

  const dragState = useRef({ isDragging: false, startX: 0, startViewXMin: 0, moved: false })

  // Precompute episode data
  const epData = useMemo(() => {
    const steps = data.timestamps
    if (!steps.length) return { components: {} as Record<string, number[]>, tMin: 0, tMax: 0, nSteps: 0, duration: 0, actionSources: [] as string[], recordTimestamps: [] as number[], spreads: [] as number[], minSpread: 0, maxSpread: 0, maxSpreadStep: -1, hasSpread: false, warningSteps: [] as { step: number; spreadMs: number }[] }

    let tMin = Infinity, tMax = -Infinity
    for (const step of steps) {
      for (const v of Object.values(step)) {
        if (typeof v === "number") {
          if (v < tMin) tMin = v
          if (v > tMax) tMax = v
        }
      }
    }

    const components: Record<string, number[]> = {}
    for (const step of steps) {
      for (const [k, v] of Object.entries(step)) {
        if (typeof v !== "number") continue
        if (!components[k]) components[k] = []
        components[k].push(v - tMin)
      }
    }

    const recRel = (data.record_timestamps || []).map(t => t - tMin)

    // Compute per-step spread (max - min across components with numeric values)
    const spreads: number[] = []
    const warningSteps: { step: number; spreadMs: number }[] = []
    let minSpread = Infinity
    let maxSpread = -Infinity
    let maxSpreadStep = -1
    let hasSpread = false
    for (let s = 0; s < steps.length; s++) {
      const step = steps[s]
      let sMin = Infinity, sMax = -Infinity, count = 0
      for (const v of Object.values(step)) {
        if (typeof v !== "number") continue
        if (v < sMin) sMin = v
        if (v > sMax) sMax = v
        count++
      }
      if (count < 2) continue
      const spread = sMax - sMin
      spreads.push(spread)
      hasSpread = true
      if (spread < minSpread) minSpread = spread
      if (spread > maxSpread) { maxSpread = spread; maxSpreadStep = s }
      if (spread > SPREAD_THRESHOLD_SECONDS) {
        warningSteps.push({ step: s, spreadMs: spread * 1000 })
      }
    }
    warningSteps.sort((a, b) => a.step - b.step)
    if (!hasSpread) {
      minSpread = 0
      maxSpread = 0
    }

    return {
      components,
      tMin,
      tMax,
      nSteps: steps.length,
      duration: tMax - tMin,
      actionSources: data.action_sources || [],
      recordTimestamps: recRel,
      spreads,
      minSpread,
      maxSpread,
      maxSpreadStep,
      hasSpread,
      warningSteps,
    }
  }, [data])

  // Reset view when data changes — default to the first DEFAULT_VIEW_SECONDS
  // starting at t=0 so long episodes open zoomed into something useful.
  useEffect(() => {
    const dur = epData.duration || 1
    setViewXMin(0)
    setViewXMax(Math.min(dur, DEFAULT_VIEW_SECONDS))
  }, [epData])

  // Refs for current view (used in event handlers without re-renders)
  const viewRef = useRef({ xMin: viewXMin, xMax: viewXMax })
  viewRef.current = { xMin: viewXMin, xMax: viewXMax }

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const parent = canvas.parentElement
    if (!parent) return
    const rect = parent.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.width = rect.width + "px"
    canvas.style.height = rect.height + "px"
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const W = rect.width
    const H = rect.height
    ctx.clearRect(0, 0, W, H)

    const d = epData
    const { xMin, xMax } = viewRef.current

    if (!d.nSteps) {
      ctx.fillStyle = "#556"
      ctx.font = "14px sans-serif"
      ctx.fillText("No timestamp data", W / 2 - 60, H / 2)
      return
    }

    const plotW = W - MARGIN.left - MARGIN.right
    const plotH = H - MARGIN.top - MARGIN.bottom
    const visibleComponents = COMPONENT_ORDER.filter(c => d.components[c])
    const rowH = Math.min(ROW_HEIGHT, plotH / Math.max(visibleComponents.length, 1))

    const xScale = (t: number) => MARGIN.left + (t - xMin) / (xMax - xMin) * plotW
    const yScale = (i: number) => MARGIN.top + i * rowH + rowH / 2

    // Grid lines
    ctx.strokeStyle = "#1e2030"
    ctx.lineWidth = 1
    for (let i = 0; i < visibleComponents.length; i++) {
      const y = MARGIN.top + i * rowH + rowH
      ctx.beginPath(); ctx.moveTo(MARGIN.left, y); ctx.lineTo(W - MARGIN.right, y); ctx.stroke()
    }

    // Time axis ticks
    const range = xMax - xMin
    const rawStep = range / (plotW / 80)
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
    const nice = [1, 2, 5, 10].find(m => m * mag >= rawStep)! * mag
    const tickStart = Math.ceil(xMin / nice) * nice

    ctx.fillStyle = "#556"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "center"
    ctx.strokeStyle = "#1a1d2a"
    for (let t = tickStart; t <= xMax; t += nice) {
      const x = xScale(t)
      if (x < MARGIN.left || x > W - MARGIN.right) continue
      ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + plotH); ctx.stroke()
      let label: string
      if (nice >= 1) label = t.toFixed(0) + "s"
      else if (nice >= 0.001) label = (t * 1000).toFixed(nice >= 0.01 ? 0 : 1) + "ms"
      else label = (t * 1e6).toFixed(0) + "us"
      ctx.fillText(label, x, MARGIN.top + plotH + 16)
    }

    // X axis label
    ctx.fillStyle = "#667"
    ctx.font = "11px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Relative Time", MARGIN.left + plotW / 2, H - 4)

    // Component labels
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.font = "11px sans-serif"
    visibleComponents.forEach((comp, i) => {
      ctx.fillStyle = COMPONENT_COLORS[comp] || "#888"
      ctx.fillText(comp.replace(/_/g, " "), MARGIN.left - 10, yScale(i))
    })

    // Action source legend
    if (d.actionSources.length > 0) {
      const lx = W - MARGIN.right - 130, ly = MARGIN.top + 4
      ctx.font = "10px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ;([["#44cc44", "policy"], ["#ff4444", "human"]] as const).forEach(([col, lbl], j) => {
        const cy = ly + j * 16
        ctx.fillStyle = col
        ctx.beginPath(); ctx.arc(lx, cy, 4, 0, Math.PI * 2); ctx.fill()
        ctx.fillStyle = "#8899aa"
        ctx.fillText(lbl, lx + 10, cy)
      })
    }

    // Per-step pixel density (for adaptive opacity)
    const pxPerStep = plotW / Math.max(d.nSteps, 1) * (d.duration / Math.max(xMax - xMin, 1e-9))

    // Record timestamp vertical lines
    if (showRecordLines && d.recordTimestamps.length > 0) {
      const recAlpha = Math.min(0.5, Math.max(0.08, pxPerStep / 15))
      ctx.strokeStyle = `rgba(255,215,0,${recAlpha})`
      ctx.lineWidth = 0.8
      for (const t of d.recordTimestamps) {
        if (t < xMin || t > xMax) continue
        const x = xScale(t)
        ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + plotH); ctx.stroke()
      }
    }

    // Dots
    const colorBySource = new Set(["action", "action_source"])
    visibleComponents.forEach((comp, i) => {
      const times = d.components[comp]
      if (!times) return
      const defaultColor = COMPONENT_COLORS[comp] || "#888"
      const useSource = colorBySource.has(comp) && d.actionSources.length > 0
      const y = yScale(i)

      for (let s = 0; s < times.length; s++) {
        const t = times[s]
        if (t < xMin || t > xMax) continue
        const x = xScale(t)
        ctx.fillStyle = useSource
          ? (ACTION_SRC_COLORS[d.actionSources[s]] || defaultColor)
          : defaultColor
        ctx.beginPath(); ctx.arc(x, y, DOT_RADIUS, 0, Math.PI * 2); ctx.fill()
      }
    })

    // Per-step connecting lines
    const lineAlpha = Math.min(0.5, Math.max(0.06, pxPerStep / 20))
    ctx.strokeStyle = `rgba(160,170,190,${lineAlpha})`
    ctx.lineWidth = 0.8
    for (let s = 0; s < d.nSteps; s++) {
      const pts: { x: number; y: number }[] = []
      visibleComponents.forEach((comp, i) => {
        const times = d.components[comp]
        if (!times || s >= times.length) return
        const t = times[s]
        if (t < xMin || t > xMax) return
        pts.push({ x: xScale(t), y: yScale(i) })
      })
      if (pts.length < 2) continue
      ctx.beginPath()
      ctx.moveTo(pts[0].x, pts[0].y)
      for (let p = 1; p < pts.length; p++) ctx.lineTo(pts[p].x, pts[p].y)
      ctx.stroke()
    }

    // Step cursor — vertical line at current chartStep's timestamp
    if (chartStep >= 0 && chartStep < d.nSteps) {
      // Find a representative time for this step (mean of all component times)
      let sum = 0, count = 0
      for (const comp of visibleComponents) {
        const times = d.components[comp]
        if (times && chartStep < times.length) {
          sum += times[chartStep]
          count++
        }
      }
      if (count > 0) {
        const stepTime = sum / count
        if (stepTime >= xMin && stepTime <= xMax) {
          const x = xScale(stepTime)
          ctx.strokeStyle = "rgba(255,180,50,0.8)"
          ctx.lineWidth = 1.5
          ctx.setLineDash([4, 3])
          ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + plotH); ctx.stroke()
          ctx.setLineDash([])
          // Step label
          ctx.fillStyle = "#ffb432"
          ctx.font = "10px sans-serif"
          ctx.textAlign = "center"
          ctx.fillText(`t=${chartStep}`, x, MARGIN.top - 6)
        }
      }
    }
  }, [epData, chartStep, showRecordLines])

  // Draw on state changes
  useEffect(() => {
    draw()
  }, [draw, viewXMin, viewXMax])

  // ResizeObserver
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => draw())
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])

  // --- Interaction helpers ---
  const rafPending = useRef(false)

  const getMousePos = (e: React.MouseEvent | MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const findNearestDot = (pos: { x: number; y: number }) => {
    const d = epData
    if (!d.nSteps) return null
    const canvas = canvasRef.current
    if (!canvas) return null
    const W = canvas.width / (window.devicePixelRatio || 1)
    const plotW = W - MARGIN.left - MARGIN.right
    const plotH = (canvas.height / (window.devicePixelRatio || 1)) - MARGIN.top - MARGIN.bottom
    const visibleComponents = COMPONENT_ORDER.filter(c => d.components[c])
    const rowH = Math.min(ROW_HEIGHT, plotH / Math.max(visibleComponents.length, 1))
    const { xMin, xMax } = viewRef.current

    let bestDist = 12
    let bestComp: string | null = null, bestStep: number | null = null, bestTime: number | null = null

    visibleComponents.forEach((comp, i) => {
      const times = d.components[comp]
      if (!times) return
      const y = MARGIN.top + i * rowH + rowH / 2
      const dy = pos.y - y
      if (Math.abs(dy) > rowH / 2) return

      for (let s = 0; s < times.length; s++) {
        const t = times[s]
        if (t < xMin || t > xMax) continue
        const x = MARGIN.left + (t - xMin) / (xMax - xMin) * plotW
        const dist = Math.sqrt((pos.x - x) ** 2 + dy ** 2)
        if (dist < bestDist) {
          bestDist = dist
          bestComp = comp
          bestStep = s
          bestTime = t
        }
      }
    })

    if (bestComp === null) return null
    return { comp: bestComp, step: bestStep!, time: bestTime! }
  }

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const W = canvas.width / (window.devicePixelRatio || 1)
    const plotW = W - MARGIN.left - MARGIN.right
    const posX = e.clientX - rect.left
    const frac = (posX - MARGIN.left) / plotW
    const { xMin, xMax } = viewRef.current
    const pivot = xMin + frac * (xMax - xMin)

    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15
    const newMin = pivot - (pivot - xMin) * factor
    const newMax = pivot + (xMax - pivot) * factor
    if (newMax - newMin < 1e-6) return
    setViewXMin(newMin)
    setViewXMax(newMax)
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragState.current = { isDragging: true, startX: e.clientX, startViewXMin: viewRef.current.xMin, moved: false }
    if (canvasRef.current) canvasRef.current.style.cursor = "grabbing"
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragState.current.isDragging) {
      const dx = e.clientX - dragState.current.startX
      if (Math.abs(dx) > 3) dragState.current.moved = true
      const canvas = canvasRef.current
      if (!canvas) return
      const W = canvas.width / (window.devicePixelRatio || 1)
      const plotW = W - MARGIN.left - MARGIN.right
      const { xMin, xMax } = viewRef.current
      const dt = -dx / plotW * (xMax - xMin)
      const range = xMax - xMin
      const newMin = dragState.current.startViewXMin + dt
      setViewXMin(newMin)
      setViewXMax(newMin + range)
      return
    }

    // Tooltip — throttled via rAF to avoid O(n) hit-testing at 60Hz
    if (rafPending.current) return
    rafPending.current = true
    const savedEvent = { clientX: e.clientX, clientY: e.clientY }
    requestAnimationFrame(() => {
      rafPending.current = false
      const tooltip = tooltipRef.current
      if (!tooltip) return
      const pos = getMousePos(savedEvent as React.MouseEvent)
      const hit = findNearestDot(pos)

    if (hit) {
      const d = epData
      tooltip.style.display = "block"
      // Build tooltip content safely (no innerHTML) to avoid XSS from file data
      tooltip.textContent = ""
      const title = document.createElement("div")
      title.style.cssText = "font-weight:600;color:#a0b4ff;margin-bottom:4px"
      title.textContent = hit.comp.replace(/_/g, " ")
      tooltip.appendChild(title)
      const lines = [
        `Step: ${hit.step}`,
        `Relative: ${(hit.time * 1000).toFixed(3)} ms`,
        `Absolute: ${(hit.time + d.tMin).toFixed(6)} s`,
      ]
      if ((hit.comp === "action" || hit.comp === "action_source") && d.actionSources[hit.step]) {
        lines.push(`Source: ${d.actionSources[hit.step]}`)
      }
      lines.forEach((line, i) => {
        if (i > 0) tooltip.appendChild(document.createElement("br"))
        tooltip.appendChild(document.createTextNode(line))
      })

      let tx = pos.x + 14, ty = pos.y - 10
      const container = containerRef.current?.getBoundingClientRect()
      if (!container) return
      if (tx + 200 > container.width) tx = pos.x - 210
      if (ty < 0) ty = 10
      tooltip.style.left = tx + "px"
      tooltip.style.top = ty + "px"
    } else {
      tooltip.style.display = "none"
    }
    }) // end rAF
  }, [epData])

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    const wasDragging = dragState.current.isDragging
    const moved = dragState.current.moved
    dragState.current.isDragging = false
    dragState.current.moved = false
    if (canvasRef.current) canvasRef.current.style.cursor = "default"

    // Click to seek (only if not a drag)
    if (wasDragging && !moved) {
      const pos = getMousePos(e)
      const hit = findNearestDot(pos)
      if (hit) onSeek(hit.step)
    }
  }, [onSeek, epData])

  const handleMouseLeave = useCallback(() => {
    dragState.current.isDragging = false
    dragState.current.moved = false
    if (canvasRef.current) canvasRef.current.style.cursor = "default"
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
  }, [])

  const handleDblClick = useCallback(() => {
    const dur = epData.duration || 1
    setViewXMin(0)
    setViewXMax(Math.min(dur, DEFAULT_VIEW_SECONDS))
  }, [epData])

  // Center the view on the step with the largest component-timing spread
  // (and move the step cursor there). Keeps the current zoom width.
  const handleJumpToMaxDelay = useCallback(() => {
    const d = epData
    if (d.maxSpreadStep < 0) return
    const visibleComponents = COMPONENT_ORDER.filter(c => d.components[c])
    let sum = 0, count = 0
    for (const comp of visibleComponents) {
      const times = d.components[comp]
      if (times && d.maxSpreadStep < times.length) {
        sum += times[d.maxSpreadStep]
        count++
      }
    }
    if (count === 0) return
    const tCenter = sum / count
    const { xMin, xMax } = viewRef.current
    const w = Math.max(xMax - xMin, 1e-6)
    let newMin = tCenter - w / 2
    let newMax = tCenter + w / 2
    if (newMin < 0) { newMax -= newMin; newMin = 0 }
    if (newMax > d.duration) {
      const shift = newMax - d.duration
      newMin = Math.max(0, newMin - shift)
      newMax = d.duration
    }
    setViewXMin(newMin)
    setViewXMax(newMax)
    onSeek(d.maxSpreadStep)
  }, [epData, onSeek])

  // Attach wheel listener with passive: false
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.addEventListener("wheel", handleWheel, { passive: false })
    return () => canvas.removeEventListener("wheel", handleWheel)
  }, [handleWheel])

  const viewRange = viewXMax - viewXMin

  return (
    <>
      {epData.hasSpread && (() => {
        const exceeds = epData.maxSpread > SPREAD_THRESHOLD_SECONDS
        return (
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground mb-2 px-1">
            <span>Component timing spread:</span>
            <span className="font-mono">min {(epData.minSpread * 1000).toFixed(1)}ms</span>
            <span className={`font-mono ${exceeds ? "text-red-300 font-bold" : ""}`}>
              max {(epData.maxSpread * 1000).toFixed(1)}ms
            </span>
            <span className="font-mono">Threshold: {(SPREAD_THRESHOLD_SECONDS * 1000).toFixed(1)}ms</span>
            {exceeds && (
              <span className="px-2 py-0.5 rounded border-2 border-red-500/80 bg-red-600/40 text-red-100 text-[10px] font-bold tracking-wide uppercase animate-pulse">
                ⚠ Timing exceeded
              </span>
            )}
            {epData.maxSpreadStep >= 0 && (
              <button
                onClick={handleJumpToMaxDelay}
                className={`ml-auto px-2 py-0.5 rounded border text-[11px] font-medium transition-colors ${
                  exceeds
                    ? "border-red-500/70 bg-red-500/20 text-red-200 hover:bg-red-500/30"
                    : "border-border bg-muted/40 text-foreground hover:bg-muted"
                }`}
                title="Center the view on the step with the largest timing spread"
              >
                Jump to max delay (step {epData.maxSpreadStep})
              </button>
            )}
          </div>
        )
      })()}
      {epData.warningSteps.length > 0 && (
        <div className="mb-2 p-2 rounded border border-red-500/40 bg-red-500/10 text-xs text-red-300">
          <div className="mb-1 font-semibold">Timing violations &gt;50ms ({epData.warningSteps.length}):</div>
          <div className="flex flex-wrap gap-1">
            {epData.warningSteps.slice(0, 40).map(w => (
              <button
                key={w.step}
                onClick={() => onSeek(w.step)}
                className="px-1.5 py-0.5 rounded bg-red-500/20 hover:bg-red-500/30 font-mono"
              >
                step {w.step}: {w.spreadMs.toFixed(1)}ms
              </button>
            ))}
            {epData.warningSteps.length > 40 && (
              <span className="px-1.5 py-0.5 opacity-70">… +{epData.warningSteps.length - 40} more</span>
            )}
          </div>
        </div>
      )}
      <div ref={containerRef} className="relative rounded-lg" style={{ height: 320, background: "#0f1117" }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDblClick}
      />
      <div
        ref={tooltipRef}
        style={{
          position: "absolute", display: "none", pointerEvents: "none",
          background: "rgba(20,24,40,0.95)", border: "1px solid #3a4060", borderRadius: 6,
          padding: "8px 12px", fontSize: 11, color: "#d0d8e8", whiteSpace: "nowrap",
          boxShadow: "0 4px 12px rgba(0,0,0,0.4)", zIndex: 10,
        }}
      />
      <div style={{
        position: "absolute", bottom: 4, left: 20, right: 20,
        display: "flex", alignItems: "center", gap: 16, fontSize: 11, color: "#667",
      }}>
        <span style={{ color: "#8899aa" }}>Scroll to zoom | Drag to pan | Click dot to seek | Dbl-click to reset</span>
        <button
          onClick={() => setShowRecordLines(!showRecordLines)}
          style={{
            padding: "2px 8px", fontSize: 10, borderRadius: 3, cursor: "pointer",
            border: `1px solid ${showRecordLines ? "#4a70c0" : "#4a4a60"}`,
            background: showRecordLines ? "#2a4080" : "#1c1f2e",
            color: showRecordLines ? "#fff" : "#8899aa",
          }}
        >
          Record Lines
        </button>
        <span style={{ marginLeft: "auto", color: "#6a7a8a" }}>
          {epData.nSteps} steps | {epData.duration.toFixed(3)}s total | viewing {viewRange.toFixed(4)}s
        </span>
      </div>
      </div>
      {epData.duration > 0 && (() => {
        const range = viewXMax - viewXMin
        const maxOffset = Math.max(0, epData.duration - range)
        const fullyZoomedOut = maxOffset < 1e-6
        const sliderMax = 1000
        const sliderVal = fullyZoomedOut ? 0 : Math.round((viewXMin / maxOffset) * sliderMax)
        return (
          <div className="mt-2 flex items-center gap-2 px-1">
            <span className="text-[10px] font-mono text-muted-foreground w-14 text-right">{viewXMin.toFixed(3)}s</span>
            <input
              type="range"
              min={0}
              max={sliderMax}
              step={1}
              value={sliderVal}
              disabled={fullyZoomedOut}
              onChange={(e) => {
                const frac = parseInt(e.target.value, 10) / sliderMax
                const newMin = frac * maxOffset
                setViewXMin(newMin)
                setViewXMax(newMin + range)
              }}
              title={fullyZoomedOut ? "Zoom in (scroll on chart) to enable panning" : "Drag to pan the visible window"}
              className="flex-1 h-2 accent-primary cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            />
            <span className="text-[10px] font-mono text-muted-foreground w-14">{viewXMax.toFixed(3)}s</span>
          </div>
        )
      })()}
    </>
  )
}
