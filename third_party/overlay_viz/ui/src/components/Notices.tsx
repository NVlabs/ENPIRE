// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"

export interface Notice {
  id: number
  title: string
  detail?: string
  level?: "info" | "success" | "error"
  action?: { label: string; onClick: () => void }
}

export function useNotices() {
  const [notices, setNotices] = useState<Notice[]>([])
  // Only keep the most recent notice — repeated saves would otherwise
  // stack into a growing pile in the corner.
  const push = useCallback((n: Omit<Notice, "id">) => {
    setNotices([{ ...n, id: Date.now() + Math.random() }])
  }, [])
  const dismiss = useCallback((id: number) => {
    setNotices(prev => prev.filter(n => n.id !== id))
  }, [])
  return { notices, push, dismiss }
}

interface Props {
  notices: Notice[]
  onDismiss: (id: number) => void
}

export function Notices({ notices, onDismiss }: Props) {
  if (notices.length === 0) return null
  return (
    <div className="fixed bottom-12 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {notices.map(n => {
        const tone =
          n.level === "error" ? "bg-red-950/90 border-red-500/70 text-red-100" :
          n.level === "success" ? "bg-green-950/90 border-green-500/70 text-green-100" :
          "bg-card border-border text-foreground"
        return (
          <div key={n.id} className={`rounded-lg border-2 p-3 shadow-xl backdrop-blur-md ${tone}`}>
            <div className="flex items-start gap-2">
              <div className="flex-1">
                <div className="text-sm font-medium">{n.title}</div>
                {n.detail && <div className="text-xs opacity-90 mt-1 break-all font-mono">{n.detail}</div>}
                {n.action && (
                  <Button size="sm" variant="outline" className="mt-2 h-7 text-xs" onClick={n.action.onClick}>
                    {n.action.label}
                  </Button>
                )}
              </div>
              <button
                onClick={() => onDismiss(n.id)}
                className="opacity-60 hover:opacity-100 p-0.5 rounded hover:bg-white/10"
                aria-label="Dismiss"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
