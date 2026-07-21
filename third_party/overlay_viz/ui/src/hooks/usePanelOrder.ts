import { useState, useCallback, useRef } from "react"

const STORAGE_KEY = "overlay-viz-panel-order-v4"

export type PanelId = "cameras" | "value" | "nav" | "timeline" | "charts" | "states" | "frequency" | "timestamps" | "labels"

export const DEFAULT_ORDER: PanelId[] = [
  "cameras",
  "value",
  "nav",
  "timeline",
  "timestamps",
  "charts",
  "states",
  "labels",
  "frequency",
]

function loadOrder(): PanelId[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_ORDER
    const parsed = JSON.parse(raw) as string[]
    // Validate: must contain exactly the known panel IDs
    const known = new Set<string>(DEFAULT_ORDER)
    const valid = parsed.filter((id) => known.has(id)) as PanelId[]
    // Add any missing panels at the end
    for (const id of DEFAULT_ORDER) {
      if (!valid.includes(id)) valid.push(id)
    }
    return valid
  } catch {
    return DEFAULT_ORDER
  }
}

function saveOrder(order: PanelId[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(order))
  } catch {
    // ignore
  }
}

export function usePanelOrder() {
  const [order, setOrder] = useState<PanelId[]>(loadOrder)
  const dragSourceRef = useRef<PanelId | null>(null)
  const dragTargetRef = useRef<PanelId | null>(null)
  const [dragTarget, setDragTarget] = useState<PanelId | null>(null)
  const [dragPosition, setDragPosition] = useState<"before" | "after" | null>(null)

  const onDragStart = useCallback((id: PanelId) => {
    dragSourceRef.current = id
  }, [])

  const onDragOver = useCallback((targetId: PanelId) => {
    const sourceId = dragSourceRef.current
    if (!sourceId || sourceId === targetId) {
      dragTargetRef.current = null
      setDragTarget(null)
      setDragPosition(null)
      return
    }
    dragTargetRef.current = targetId
    setDragTarget(targetId)
    setOrder((prev) => {
      const srcIdx = prev.indexOf(sourceId)
      const tgtIdx = prev.indexOf(targetId)
      setDragPosition(srcIdx < tgtIdx ? "after" : "before")
      return prev
    })
  }, [])

  const onDragEnd = useCallback(() => {
    const sourceId = dragSourceRef.current
    const targetId = dragTargetRef.current
    dragSourceRef.current = null
    dragTargetRef.current = null
    setDragTarget(null)
    setDragPosition(null)

    if (!sourceId || !targetId || sourceId === targetId) return

    setOrder((prev) => {
      const next = prev.filter((id) => id !== sourceId)
      const insertIdx = next.indexOf(targetId)
      const srcIdx = prev.indexOf(sourceId)
      const tgtIdx = prev.indexOf(targetId)
      // Insert before or after target depending on original positions
      if (srcIdx < tgtIdx) {
        next.splice(insertIdx + 1, 0, sourceId)
      } else {
        next.splice(insertIdx, 0, sourceId)
      }
      saveOrder(next)
      return next
    })
  }, [])

  const resetOrder = useCallback(() => {
    setOrder(DEFAULT_ORDER)
    saveOrder(DEFAULT_ORDER)
  }, [])

  return { order, dragTarget, dragPosition, onDragStart, onDragOver, onDragEnd, resetOrder }
}
