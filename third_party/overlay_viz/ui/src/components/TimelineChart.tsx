// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useCallback } from "react"
import { ResponsiveLine } from "@nivo/line"
import { channelColors, sourceColor, isChannelVisible } from "@/lib/chart-utils"
import type { ChartData, TrimSegment, ActionSourceData } from "@/api/types"

type LineSerie = {
  id: string
  color?: string
  data: { x: number; y: number }[]
}

const SEG_STROKE_COLORS = [
  "#38bdf8", // sky
  "#fbbf24", // amber
  "#a78bfa", // violet
  "#34d399", // emerald
  "#fb7185", // rose
  "#22d3ee", // cyan
]

interface Props {
  data: ChartData | null
  chartStep: number
  rangeStart: number
  rangeEnd: number
  fps: number
  showLeft: boolean
  showRight: boolean
  showGrip: boolean
  segments: TrimSegment[]
  activeSegIdx: number
  actionSourceData?: ActionSourceData | null
  rtgMarker?: { type: string; step: number } | null
  rtgRange?: { start: number; end: number; status?: string | null } | null
  onSeek: (step: number) => void
}

export function TimelineChart({ data, chartStep, fps, showLeft, showRight, showGrip, segments, activeSegIdx, actionSourceData, rtgMarker, rtgRange, onSeek }: Props) {
  const { series, maxStep } = useMemo(() => {
    if (!data) return { series: [] as LineSerie[], maxStep: 0 }
    const colors = channelColors(data.dim)
    const ms = Math.max(0, data.total_steps - 1)

    const s: LineSerie[] = data.labels
      .map((label, ch) => ({
        id: label,
        color: colors[ch] ?? "#888",
        data: isChannelVisible(ch, data.dim, showLeft, showRight, showGrip)
          ? data.data.map((row, i) => ({ x: i, y: row[ch] }))
          : [],
      }))
      .filter((s) => s.data.length > 0)

    return { series: s, maxStep: ms }
  }, [data, showLeft, showRight, showGrip])

  const handleClick = useCallback(
    (point: any) => {
      const step = Number(point.data.x)
      if (!isNaN(step)) onSeek(step)
    },
    [onSeek],
  )

  const CursorLayer = useCallback(
    ({ xScale }: any) => {
      const x = (xScale as (v: number) => number)(chartStep)
      return (
        <g>
          <line x1={x} x2={x} y1={0} y2={1000} stroke="#f59e0b" strokeWidth={2} strokeDasharray="6,3" />
          <text x={x + 4} y={12} fill="#f59e0b" fontSize={9}>
            t={chartStep} ({(chartStep / fps).toFixed(2)}s)
          </text>
        </g>
      )
    },
    [chartStep, fps],
  )

  const TrimLayer = useCallback(
    ({ xScale, innerHeight }: any) => {
      const scale = xScale as (v: number) => number

      // Build excluded regions: gaps between (and outside) all segments
      const sorted = [...segments].sort((a, b) => a.start - b.start)
      const excluded: [number, number][] = []
      let cursor = 0
      for (const seg of sorted) {
        if (seg.start > cursor) excluded.push([cursor, seg.start])
        cursor = Math.max(cursor, seg.end)
      }
      if (cursor < maxStep) excluded.push([cursor, maxStep])

      return (
        <g>
          {/* Dim excluded regions */}
          {excluded.map(([s, e], i) => (
            <rect key={`excl-${i}`} x={scale(s)} y={0} width={scale(e) - scale(s)} height={innerHeight} fill="rgba(0,0,0,0.15)" />
          ))}

          {/* Render each segment */}
          {sorted.map((seg, origIdx) => {
            const realIdx = segments.indexOf(seg)
            const isActive = realIdx === activeSegIdx
            const color = SEG_STROKE_COLORS[realIdx % SEG_STROKE_COLORS.length]
            const xIn = scale(seg.start)
            const xOut = scale(seg.end)
            const opacity = isActive ? 1 : 0.4
            const lineWidth = isActive ? 3 : 1.5

            return (
              <g key={`seg-${origIdx}`} opacity={opacity}>
                {/* Segment fill */}
                <rect x={xIn} y={0} width={xOut - xIn} height={innerHeight} fill={color} opacity={isActive ? 0.06 : 0.03} />
                {/* IN marker */}
                <line x1={xIn} x2={xIn} y1={0} y2={innerHeight} stroke={color} strokeWidth={lineWidth} />
                {isActive && <text x={xIn + 3} y={12} fill={color} fontSize={9} fontWeight="bold">IN</text>}
                {/* OUT marker */}
                <line x1={xOut} x2={xOut} y1={0} y2={innerHeight} stroke={color} strokeWidth={lineWidth} />
                {isActive && <text x={xOut + 3} y={12} fill={color} fontSize={9} fontWeight="bold">OUT</text>}
                {/* Label */}
                {!isActive && (
                  <text x={(xIn + xOut) / 2} y={innerHeight - 4} fill={color} fontSize={8} textAnchor="middle" opacity={0.7}>
                    {seg.label || `Seg ${realIdx + 1}`}
                  </text>
                )}
              </g>
            )
          })}
        </g>
      )
    },
    [segments, activeSegIdx, maxStep],
  )

  const SourceLayer = useCallback(
    ({ xScale, innerHeight }: any) => {
      if (!actionSourceData?.segments) return null
      const scale = xScale as (v: number) => number
      return (
        <g>
          {actionSourceData.segments.map((seg, i) => (
            <rect
              key={i}
              x={scale(seg.start)}
              y={0}
              width={scale(seg.end) - scale(seg.start)}
              height={innerHeight}
              fill={sourceColor(seg.source)}
              opacity={0.08}
            />
          ))}
        </g>
      )
    },
    [actionSourceData],
  )

  const RtgMarkerLayer = useCallback(
    ({ xScale, innerHeight }: any) => {
      if (!rtgMarker) return null
      const scale = xScale as (v: number) => number
      const x = scale(rtgMarker.step)
      const color = rtgMarker.type === "success_end" ? "#22c55e" : "#ef4444"
      const label = rtgMarker.type === "success_end" ? "SUCCESS" : "FAILURE"
      return (
        <g>
          <line x1={x} x2={x} y1={0} y2={innerHeight} stroke={color} strokeWidth={2} strokeDasharray="4,4" />
          <text x={x + 4} y={innerHeight - 4} fill={color} fontSize={9} fontWeight="bold">{label}</text>
        </g>
      )
    },
    [rtgMarker],
  )

  const RtgRangeLayer = useCallback(
    ({ xScale, innerHeight }: any) => {
      if (!rtgRange) return null
      const scale = xScale as (v: number) => number
      const xIn = scale(rtgRange.start)
      const xOut = scale(rtgRange.end)
      const xZero = scale(0)
      const xMax = scale(maxStep)
      const endColor = rtgRange.status === "success" ? "#22c55e" : rtgRange.status === "failure" ? "#ef4444" : "#94a3b8"
      return (
        <g>
          {rtgRange.start > 0 && (
            <rect x={xZero} y={0} width={xIn - xZero} height={innerHeight} fill="rgba(0,0,0,0.12)" />
          )}
          {rtgRange.end < maxStep && (
            <rect x={xOut} y={0} width={xMax - xOut} height={innerHeight} fill="rgba(0,0,0,0.12)" />
          )}
          <line x1={xIn} x2={xIn} y1={0} y2={innerHeight} stroke="#22c55e" strokeWidth={2.5} strokeDasharray="5,3" />
          <text x={xIn + 3} y={12} fill="#22c55e" fontSize={9} fontWeight="bold">RTG IN</text>
          <line x1={xOut} x2={xOut} y1={0} y2={innerHeight} stroke={endColor} strokeWidth={2.5} strokeDasharray="5,3" />
          <text x={xOut + 3} y={12} fill={endColor} fontSize={9} fontWeight="bold">
            RTG OUT{rtgRange.status ? ` (${rtgRange.status})` : ""}
          </text>
        </g>
      )
    },
    [rtgRange, maxStep],
  )

  if (!data || series.length === 0) return null

  const seriesColors = series.map((s) => s.color as string)

  return (
    <div className="bg-muted rounded-lg h-[280px] w-full">
      <ResponsiveLine
        data={series}
        colors={seriesColors}
        margin={{ top: 20, right: 120, bottom: 40, left: 50 }}
        xScale={{ type: "linear", min: 0, max: maxStep }}
        yScale={{ type: "linear", min: "auto", max: "auto" }}
        curve="monotoneX"
        lineWidth={1.5}
        enablePoints={false}
        enableGridX={true}
        enableGridY={true}
        gridXValues={10}
        axisBottom={{ legend: "Step", legendOffset: 30, legendPosition: "middle", tickSize: 5 }}
        axisLeft={{ legend: "Value", legendOffset: -40, legendPosition: "middle", tickSize: 5 }}
        enableSlices="x"
        sliceTooltip={({ slice }) => (
          <div className="bg-popover border rounded-lg p-2 text-xs shadow-lg max-h-48 overflow-y-auto">
            <div className="font-semibold mb-1">Step {slice.points[0]?.data.x?.toString()}</div>
            {slice.points.map((p) => (
              <div key={p.id} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: p.seriesColor }} />
                <span>{p.seriesId}:</span>
                <span className="font-mono">{Number(p.data.y).toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
        onClick={handleClick}
        layers={["grid", SourceLayer, "markers", "axes", "areas", "lines", "crosshair", TrimLayer, RtgRangeLayer, RtgMarkerLayer, CursorLayer, "slices", "legends"]}
        legends={[
          {
            anchor: "right",
            direction: "column",
            translateX: 110,
            itemWidth: 100,
            itemHeight: 14,
            symbolSize: 8,
            symbolShape: "circle",
            itemTextColor: "#aaa",
            toggleSerie: true,
          },
        ]}
        theme={{
          background: "transparent",
          text: { fill: "#aaa", fontSize: 10 },
          axis: { ticks: { text: { fill: "#888", fontSize: 9 } }, legend: { text: { fill: "#888", fontSize: 10 } } },
          grid: { line: { stroke: "rgba(128,128,128,0.15)" } },
          crosshair: { line: { stroke: "#f59e0b", strokeOpacity: 0.5 } },
        }}
      />
    </div>
  )
}
