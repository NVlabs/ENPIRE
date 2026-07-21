import { useCallback, useRef, useState, useEffect, useMemo } from "react"
import { Plus, ChevronLeft, ChevronRight, Trash2, Save, Circle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { KeyFrame } from "@/hooks/useProgressLabels"

interface Props {
  keyframes: KeyFrame[]
  chartStep: number
  totalSteps: number
  dirty: boolean
  saving: boolean
  interpolate: (step: number) => number
  onAddKeyFrame: (step: number) => void
  onDeleteKeyFrame: (step: number) => void
  onUpdateKeyFrame: (step: number, progress: number) => void
  onNextKeyFrame: () => void
  onPrevKeyFrame: () => void
  onSeek: (step: number) => void
  onSave: () => void
  hasKeyFrameAt: (step: number) => boolean
  valueTrimStart?: number
  valueTrimEnd?: number
}

const MARGIN = { top: 24, right: 40, bottom: 32, left: 48 }
const HANDLE_R = 6
const HANDLE_R_HOVER = 8

export function ProgressLabelPanel({
  keyframes,
  chartStep,
  totalSteps,
  dirty,
  saving,
  interpolate,
  onAddKeyFrame,
  onDeleteKeyFrame,
  onUpdateKeyFrame,
  onNextKeyFrame,
  onPrevKeyFrame,
  onSeek,
  onSave,
  hasKeyFrameAt,
  valueTrimStart,
  valueTrimEnd,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [svgWidth, setSvgWidth] = useState(800)
  const svgHeight = 280
  const [dragKf, setDragKf] = useState<number | null>(null)
  const [hoverKf, setHoverKf] = useState<number | null>(null)

  // Responsive width
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const obs = new ResizeObserver((entries) => {
      for (const e of entries) setSvgWidth(e.contentRect.width)
    })
    obs.observe(svg.parentElement!)
    return () => obs.disconnect()
  }, [])

  const plotW = svgWidth - MARGIN.left - MARGIN.right
  const plotH = svgHeight - MARGIN.top - MARGIN.bottom
  const maxStep = Math.max(1, totalSteps - 1)

  const xScale = useCallback((step: number) => MARGIN.left + (step / maxStep) * plotW, [maxStep, plotW])
  const yScale = useCallback((progress: number) => MARGIN.top + (1 - progress) * plotH, [plotH])
  const yInverse = useCallback((py: number) => 1 - (py - MARGIN.top) / plotH, [plotH])
  const xInverse = useCallback((px: number) => Math.round(((px - MARGIN.left) / plotW) * maxStep), [maxStep, plotW])

  // Line path through keyframes
  const linePath = useMemo(() => {
    if (keyframes.length === 0) return ""
    return keyframes
      .map((kf, i) => `${i === 0 ? "M" : "L"}${xScale(kf.step)},${yScale(kf.progress)}`)
      .join(" ")
  }, [keyframes, xScale, yScale])

  // Grid ticks
  const xTicks = useMemo(() => {
    const count = Math.min(10, maxStep)
    const step = maxStep / count
    const ticks: number[] = []
    for (let i = 0; i <= count; i++) ticks.push(Math.round(i * step))
    return ticks
  }, [maxStep])

  const yTicks = [0, 0.25, 0.5, 0.75, 1.0]

  // Drag handling
  const handlePointerDown = useCallback(
    (step: number) => (e: React.PointerEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setDragKf(step)
      ;(e.target as Element).setPointerCapture(e.pointerId)
    },
    [],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (dragKf === null) return
      const svg = svgRef.current
      if (!svg) return
      const rect = svg.getBoundingClientRect()
      const py = e.clientY - rect.top
      const progress = Math.max(0, Math.min(1, yInverse(py)))
      onUpdateKeyFrame(dragKf, progress)
    },
    [dragKf, yInverse, onUpdateKeyFrame],
  )

  const handlePointerUp = useCallback(() => {
    setDragKf(null)
  }, [])

  // Click on background = seek
  const handleBgClick = useCallback(
    (e: React.MouseEvent) => {
      if (dragKf !== null) return
      const svg = svgRef.current
      if (!svg) return
      const rect = svg.getBoundingClientRect()
      const px = e.clientX - rect.left
      const step = Math.max(0, Math.min(maxStep, xInverse(px)))
      onSeek(step)
    },
    [dragKf, maxStep, xInverse, onSeek],
  )

  const currentProgress = interpolate(chartStep)

  return (
    <div className="mt-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 mb-2 px-1">
        <span className="text-xs font-bold opacity-70">PROGRESS</span>

        <Button
          size="sm"
          variant="outline"
          onClick={() => onAddKeyFrame(chartStep)}
          disabled={hasKeyFrameAt(chartStep)}
          className="h-6 text-xs px-2 gap-1"
        >
          <Plus className="h-3 w-3" /> Add KF
          <kbd className="ml-1 px-1 py-0.5 text-[9px] bg-muted rounded border border-border font-mono">K</kbd>
        </Button>

        <Button size="sm" variant="ghost" onClick={onPrevKeyFrame} className="h-6 text-xs px-2 gap-1">
          <ChevronLeft className="h-3 w-3" /> Prev
          <kbd className="ml-0.5 px-1 py-0.5 text-[9px] bg-muted rounded border border-border font-mono">[</kbd>
        </Button>
        <Button size="sm" variant="ghost" onClick={onNextKeyFrame} className="h-6 text-xs px-2 gap-1">
          Next <ChevronRight className="h-3 w-3" />
          <kbd className="ml-0.5 px-1 py-0.5 text-[9px] bg-muted rounded border border-border font-mono">]</kbd>
        </Button>

        <Button
          size="sm"
          variant="ghost"
          onClick={() => onDeleteKeyFrame(chartStep)}
          disabled={!hasKeyFrameAt(chartStep)}
          className="h-6 text-xs px-2 gap-1 text-red-400 hover:text-red-300"
        >
          <Trash2 className="h-3 w-3" /> Delete
          <kbd className="ml-0.5 px-1 py-0.5 text-[9px] bg-muted rounded border border-border font-mono">Del</kbd>
        </Button>

        <span className="opacity-20">|</span>

        {hasKeyFrameAt(chartStep) && (
          <>
            {[0, 0.25, 0.5, 0.75, 1].map(v => (
              <Button
                key={v}
                size="sm"
                variant={Math.abs(currentProgress - v) < 0.001 ? "default" : "outline"}
                onClick={() => onUpdateKeyFrame(chartStep, v)}
                className="h-6 text-[10px] px-1.5 font-mono min-w-[2rem]"
              >
                {v}
              </Button>
            ))}
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={currentProgress.toFixed(3)}
              onChange={e => {
                const v = parseFloat(e.target.value)
                if (!isNaN(v)) onUpdateKeyFrame(chartStep, Math.max(0, Math.min(1, v)))
              }}
              className="h-6 w-16 text-[10px] font-mono px-1.5 bg-muted border rounded outline-none text-center"
            />
            <span className="opacity-20">|</span>
          </>
        )}

        <Badge variant="outline" className="text-[10px] font-mono">
          {keyframes.length} keyframes
        </Badge>
        <Badge variant="outline" className="text-[10px] font-mono">
          p={currentProgress.toFixed(3)}
        </Badge>

        <div className="ml-auto flex items-center gap-1.5">
          {dirty && (
            <span className="flex items-center gap-1 text-[10px] text-amber-400">
              <Circle className="h-2 w-2 fill-amber-400" /> unsaved
            </span>
          )}
          <Button
            size="sm"
            onClick={onSave}
            disabled={saving || !dirty}
            className="h-6 text-xs px-3 gap-1"
          >
            <Save className="h-3 w-3" /> {saving ? "Saving..." : "Save"}
          </Button>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="bg-muted rounded-lg overflow-hidden" style={{ height: svgHeight }}>
        <svg
          ref={svgRef}
          width="100%"
          height={svgHeight}
          className="select-none"
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onClick={handleBgClick}
        >
          {/* Grid lines */}
          {xTicks.map((tick) => (
            <line
              key={`gx-${tick}`}
              x1={xScale(tick)}
              x2={xScale(tick)}
              y1={MARGIN.top}
              y2={MARGIN.top + plotH}
              stroke="rgba(128,128,128,0.12)"
              strokeWidth={1}
            />
          ))}
          {yTicks.map((tick) => (
            <line
              key={`gy-${tick}`}
              x1={MARGIN.left}
              x2={MARGIN.left + plotW}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="rgba(128,128,128,0.12)"
              strokeWidth={1}
            />
          ))}

          {/* Axis labels */}
          {xTicks.map((tick) => (
            <text
              key={`lx-${tick}`}
              x={xScale(tick)}
              y={MARGIN.top + plotH + 16}
              textAnchor="middle"
              fill="#888"
              fontSize={9}
            >
              {tick}
            </text>
          ))}
          {yTicks.map((tick) => (
            <text
              key={`ly-${tick}`}
              x={MARGIN.left - 8}
              y={yScale(tick) + 3}
              textAnchor="end"
              fill="#888"
              fontSize={9}
            >
              {tick.toFixed(2)}
            </text>
          ))}

          {/* Axis titles */}
          <text
            x={MARGIN.left + plotW / 2}
            y={svgHeight - 4}
            textAnchor="middle"
            fill="#888"
            fontSize={10}
          >
            Step
          </text>
          <text
            x={12}
            y={MARGIN.top + plotH / 2}
            textAnchor="middle"
            fill="#888"
            fontSize={10}
            transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
          >
            Progress
          </text>

          {/* Value trim markers */}
          {valueTrimStart != null && valueTrimStart > 0 && (
            <rect x={xScale(0)} y={MARGIN.top} width={xScale(valueTrimStart) - xScale(0)} height={plotH} fill="rgba(0,0,0,0.15)" pointerEvents="none" />
          )}
          {valueTrimEnd != null && valueTrimEnd < maxStep && (
            <rect x={xScale(valueTrimEnd)} y={MARGIN.top} width={xScale(maxStep) - xScale(valueTrimEnd)} height={plotH} fill="rgba(0,0,0,0.15)" pointerEvents="none" />
          )}
          {valueTrimStart != null && (
            <g pointerEvents="none">
              <line x1={xScale(valueTrimStart)} x2={xScale(valueTrimStart)} y1={MARGIN.top} y2={MARGIN.top + plotH} stroke="#22c55e" strokeWidth={2.5} />
              <text x={xScale(valueTrimStart) + 3} y={MARGIN.top + 10} fill="#22c55e" fontSize={9} fontWeight="bold">IN</text>
            </g>
          )}
          {valueTrimEnd != null && (
            <g pointerEvents="none">
              <line x1={xScale(valueTrimEnd)} x2={xScale(valueTrimEnd)} y1={MARGIN.top} y2={MARGIN.top + plotH} stroke="#ef4444" strokeWidth={2.5} />
              <text x={xScale(valueTrimEnd) + 3} y={MARGIN.top + 10} fill="#ef4444" fontSize={9} fontWeight="bold">OUT</text>
            </g>
          )}

          {/* Progress line */}
          {linePath && (
            <path
              d={linePath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={2}
              strokeLinejoin="round"
            />
          )}

          {/* Playhead cursor */}
          <line
            x1={xScale(chartStep)}
            x2={xScale(chartStep)}
            y1={MARGIN.top}
            y2={MARGIN.top + plotH}
            stroke="#f59e0b"
            strokeWidth={1.5}
            strokeDasharray="6,3"
            pointerEvents="none"
          />
          <text
            x={xScale(chartStep) + 4}
            y={MARGIN.top - 6}
            fill="#f59e0b"
            fontSize={9}
            pointerEvents="none"
          >
            t={chartStep}
          </text>

          {/* Current progress indicator on cursor */}
          <circle
            cx={xScale(chartStep)}
            cy={yScale(currentProgress)}
            r={3}
            fill="#f59e0b"
            pointerEvents="none"
          />

          {/* Keyframe handles */}
          {keyframes.map((kf) => {
            const isActive = dragKf === kf.step
            const isHovered = hoverKf === kf.step
            const r = isActive || isHovered ? HANDLE_R_HOVER : HANDLE_R
            return (
              <g key={kf.step}>
                <circle
                  cx={xScale(kf.step)}
                  cy={yScale(kf.progress)}
                  r={r}
                  fill="#3b82f6"
                  stroke="white"
                  strokeWidth={2}
                  style={{ cursor: "ns-resize", transition: isActive ? "none" : "r 0.1s" }}
                  onPointerDown={handlePointerDown(kf.step)}
                  onPointerEnter={() => setHoverKf(kf.step)}
                  onPointerLeave={() => setHoverKf(null)}
                />
                {/* Tooltip on hover */}
                {(isHovered || isActive) && (
                  <g pointerEvents="none">
                    <rect
                      x={xScale(kf.step) + 10}
                      y={yScale(kf.progress) - 22}
                      width={90}
                      height={20}
                      rx={4}
                      fill="rgba(0,0,0,0.85)"
                    />
                    <text
                      x={xScale(kf.step) + 14}
                      y={yScale(kf.progress) - 8}
                      fill="white"
                      fontSize={10}
                      fontFamily="monospace"
                    >
                      t={kf.step} p={kf.progress.toFixed(3)}
                    </text>
                  </g>
                )}
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
