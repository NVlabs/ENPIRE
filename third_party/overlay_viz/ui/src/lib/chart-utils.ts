export const CHANNEL_COLORS_14 = [
  "#3b82f6", "#60a5fa", "#93c5fd", "#2563eb", "#1d4ed8", "#1e40af",
  "#22c55e",
  "#ef4444", "#f87171", "#fca5a5", "#dc2626", "#b91c1c", "#991b1b",
  "#a855f7",
]

export function channelColors(dim: number): string[] {
  if (dim === 14) return CHANNEL_COLORS_14
  return Array.from({ length: dim }, (_, i) => `hsl(${(i * 360 / dim) % 360}, 70%, 55%)`)
}

export function sourceColor(src: string): string {
  const colors: Record<string, string> = { human: "#22c55e", policy: "#3b82f6", rl: "#8b5cf6", unknown: "#6b7280" }
  return colors[src] ?? "#6b7280"
}

export function isChannelVisible(ch: number, dim: number, showLeft: boolean, showRight: boolean, showGrip: boolean): boolean {
  if (dim !== 14) return true
  if (ch < 6) return showLeft
  if (ch === 6) return showGrip
  if (ch < 13) return showRight
  return showGrip
}

// Returns the first step index where either
//   - L-infinity of non-gripper joint deltas exceeds jointThreshold, or
//   - absolute gripper delta exceeds gripThreshold.
// Gripper channels are ch==6 and ch==13 only for dim==14; otherwise every
// channel is treated as a non-gripper joint. Returns 0 if no step meets
// the criteria (or the trajectory is too short to form a delta).
export function findFirstMovementStep(
  data: { data: number[][]; dim: number },
  jointThreshold: number,
  gripThreshold: number,
): number {
  const T = data.data.length
  if (T < 2) return 0
  const isGrip = (ch: number) => data.dim === 14 && (ch === 6 || ch === 13)
  for (let i = 0; i < T - 1; i++) {
    let jointMax = 0
    let gripMax = 0
    for (let ch = 0; ch < data.dim; ch++) {
      const d = Math.abs(data.data[i + 1][ch] - data.data[i][ch])
      if (isGrip(ch)) {
        if (d > gripMax) gripMax = d
      } else {
        if (d > jointMax) jointMax = d
      }
    }
    if (jointMax > jointThreshold || gripMax > gripThreshold) return i
  }
  return 0
}

// Mirror of findFirstMovementStep: returns the LAST step where the delta
// from the previous step exceeds thresholds — i.e., the last moment the
// arm was still moving. Falls back to totalSteps-1 when no qualifying
// step is found.
export function findLastMovementStep(
  data: { data: number[][]; dim: number },
  jointThreshold: number,
  gripThreshold: number,
): number {
  const T = data.data.length
  if (T < 2) return Math.max(0, T - 1)
  const isGrip = (ch: number) => data.dim === 14 && (ch === 6 || ch === 13)
  for (let i = T - 1; i >= 1; i--) {
    let jointMax = 0
    let gripMax = 0
    for (let ch = 0; ch < data.dim; ch++) {
      const d = Math.abs(data.data[i][ch] - data.data[i - 1][ch])
      if (isGrip(ch)) {
        if (d > gripMax) gripMax = d
      } else {
        if (d > jointMax) jointMax = d
      }
    }
    if (jointMax > jointThreshold || gripMax > gripThreshold) return i
  }
  return Math.max(0, T - 1)
}
