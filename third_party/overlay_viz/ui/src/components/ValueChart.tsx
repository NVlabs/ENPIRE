import { useCallback, useMemo } from "react"
import { ResponsiveLine } from "@nivo/line"
import type { ValuePredictionData } from "@/api/types"

type LineSerie = {
  id: string
  color?: string
  data: { x: number; y: number }[]
}

/**
 * Compute per-frame advantage client-side from absolute_value array.
 * Matches minimal_policy/value/awbc_dataset.py:compute_episode_advantage().
 * For raw eval data: assumes binary_reward=0 everywhere (TTS penalty -1).
 */
function computeAdvantage(
  values: number[],
  mode: string,
  advantageH: number = 50,
): number[] {
  const T = values.length
  if (T === 0) return []

  if (mode === "value_delta") {
    // Simple forward delta: V(t+h) - V(t)
    return values.map((v, t) => {
      const th = Math.min(T - 1, t + advantageH)
      return Math.max(-1, Math.min(1, values[th] - v))
    })
  }

  // Compute TTS-normalized RTG with synthetic all-zero binary_reward.
  // r_tts = 0 - 1 = -1 for every step. RTG[t] = sum_{k>=t} -1 = -(T-t).
  // Normalized by |RTG[0]| = T, so RTG[t] = -(T-t)/T.
  const rtg = values.map((_, t) => -(T - t) / T)

  if (mode === "traj_adv") {
    // Adv(t) = RTG(t) - V(t)
    return values.map((v, t) => Math.max(-1, Math.min(1, rtg[t] - v)))
  }

  // recap_post_train: Adv(t) = V(t+h) + RTG(t) - RTG(t+h) - V(t)
  return values.map((v, t) => {
    const th = Math.min(T - 1, t + advantageH)
    const adv = values[th] + rtg[t] - rtg[th] - v
    return Math.max(-1, Math.min(1, adv))
  })
}

interface Props {
  data: ValuePredictionData | null
  chartStep: number
  totalSteps: number
  fps: number
  advMode?: string
  advantageH?: number
  rtgRange?: { start: number; end: number; status?: string | null } | null
  onSeek: (step: number) => void
}

export function ValueChart({ data, chartStep, totalSteps, fps, advMode, advantageH = 50, rtgRange, onSeek }: Props) {
  const { series, maxStep } = useMemo(() => {
    if (!data || data.frame_idx.length === 0) {
      return { series: [] as LineSerie[], maxStep: Math.max(0, totalSteps - 1) }
    }

    // Recompute advantage client-side based on selected mode and horizon
    const advantage = advMode
      ? computeAdvantage(data.absolute_value, advMode, advantageH)
      : data.absolute_advantage

    const buildSerie = (id: string, color: string, values: number[] | undefined): LineSerie | null => {
      if (!values || values.length === 0) return null
      return {
        id,
        color,
        data: data.frame_idx.map((frame, idx) => ({ x: frame, y: values[idx] })),
      }
    }

    const nextSeries = [
      buildSerie("absolute_value", "#f59e0b", data.absolute_value),
      buildSerie("advantage", "#38bdf8", advantage),
      buildSerie("relative_advantage", "#f43f5e", data.relative_advantage),
    ].filter((serie): serie is LineSerie => Boolean(serie))

    const frameMax = data.frame_idx.reduce((acc, value) => Math.max(acc, value), 0)
    return { series: nextSeries, maxStep: Math.max(frameMax, totalSteps - 1) }
  }, [data, totalSteps, advMode, advantageH])

  const handleClick = useCallback((point: any) => {
    const step = Number(point.data.x)
    if (!Number.isNaN(step)) onSeek(step)
  }, [onSeek])

  const CursorLayer = useCallback(({ xScale }: any) => {
    const scale = xScale as (v: number) => number
    const x = scale(chartStep)
    return (
      <g>
        <line x1={x} x2={x} y1={0} y2={1000} stroke="#f59e0b" strokeWidth={2} strokeDasharray="6,3" />
        <text x={x + 4} y={12} fill="#f59e0b" fontSize={9}>
          t={chartStep} ({(chartStep / fps).toFixed(2)}s)
        </text>
      </g>
    )
  }, [chartStep, fps])

  const RtgRangeLayer = useCallback(({ xScale, innerHeight }: any) => {
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
  }, [maxStep, rtgRange])

  if (!data || series.length === 0) return null

  return (
    <div className="bg-muted rounded-lg h-[280px] w-full">
      <ResponsiveLine
        data={series}
        colors={series.map((serie) => serie.color as string)}
        margin={{ top: 20, right: 150, bottom: 40, left: 50 }}
        xScale={{ type: "linear", min: 0, max: maxStep }}
        yScale={{ type: "linear", min: "auto", max: "auto" }}
        curve="monotoneX"
        lineWidth={2}
        enablePoints={true}
        pointSize={4}
        pointBorderWidth={1}
        pointBorderColor={{ from: "serieColor" }}
        useMesh={true}
        enableGridX={true}
        enableGridY={true}
        axisBottom={{ legend: "Frame", legendOffset: 30, legendPosition: "middle", tickSize: 5 }}
        axisLeft={{ legend: "Value", legendOffset: -40, legendPosition: "middle", tickSize: 5 }}
        enableSlices="x"
        sliceTooltip={({ slice }) => (
          <div className="bg-popover border rounded-lg p-2 text-xs shadow-lg">
            <div className="font-semibold mb-1">Frame {slice.points[0]?.data.x?.toString()}</div>
            {slice.points.map((point) => (
              <div key={point.id} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: point.seriesColor }} />
                <span>{point.seriesId}:</span>
                <span className="font-mono">{Number(point.data.y).toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
        onClick={handleClick}
        layers={["grid", "markers", "axes", "areas", "lines", "points", "crosshair", RtgRangeLayer, CursorLayer, "slices", "legends"]}
        legends={[
          {
            anchor: "right",
            direction: "column",
            translateX: 130,
            itemWidth: 120,
            itemHeight: 16,
            symbolSize: 8,
            symbolShape: "circle",
            itemTextColor: "#aaa",
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
