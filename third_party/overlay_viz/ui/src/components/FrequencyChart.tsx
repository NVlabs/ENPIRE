// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react"
import { ResponsiveLine } from "@nivo/line"
import { channelColors, isChannelVisible } from "@/lib/chart-utils"
import type { FrequencyData } from "@/api/types"

type LineSerie = {
  id: string
  color?: string
  data: { x: number; y: number }[]
}

interface Props {
  data: FrequencyData | null
  showLeft: boolean
  showRight: boolean
  showGrip: boolean
}

export function FrequencyChart({ data, showLeft, showRight, showGrip }: Props) {
  const { series, colors } = useMemo(() => {
    if (!data) return { series: [] as LineSerie[], colors: [] as string[] }
    const c = channelColors(data.dim)

    const s: LineSerie[] = []
    const cols: string[] = []
    data.labels.forEach((label, d) => {
      if (!isChannelVisible(d, data.dim, showLeft, showRight, showGrip)) return
      s.push({
        id: label,
        data: data.magnitudes[d].map((v, i) => ({ x: data.freq_bins[i], y: v })),
      })
      cols.push(c[d] ?? "#888")
    })
    return { series: s, colors: cols }
  }, [data, showLeft, showRight, showGrip])

  if (!data || series.length === 0) return null

  return (
    <div className="bg-muted rounded-lg h-[280px] w-full">
      <ResponsiveLine
        data={series}
        colors={colors}
        margin={{ top: 20, right: 120, bottom: 40, left: 50 }}
        xScale={{ type: "linear", min: "auto", max: "auto" }}
        yScale={{ type: "linear", min: 0, max: "auto" }}
        curve="monotoneX"
        lineWidth={1.5}
        enablePoints={false}
        enableGridX={true}
        enableGridY={true}
        axisBottom={{ legend: "Frequency (Hz)", legendOffset: 30, legendPosition: "middle" }}
        axisLeft={{ legend: "Magnitude", legendOffset: -40, legendPosition: "middle" }}
        enableSlices="x"
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
        }}
      />
    </div>
  )
}
