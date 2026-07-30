// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect } from "react"
import { Camera, Sun, Moon } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface Props {
  cameraOnline: boolean
  cameraCount: number
}

export function Navbar({ cameraOnline, cameraCount }: Props) {
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return false
    return document.documentElement.classList.contains("dark") ||
      window.matchMedia("(prefers-color-scheme: dark)").matches
  })

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-muted border-b">
      <span className="text-xl font-bold tracking-tight">Data Studio</span>
      <div className="flex items-center gap-3">
        <button
          onClick={() => setDark((d) => !d)}
          className="relative flex items-center w-11 h-6 rounded-full bg-border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Toggle dark mode"
        >
          <span
            className={`absolute top-0.5 flex items-center justify-center w-5 h-5 rounded-full bg-background shadow transition-transform ${dark ? "translate-x-[22px]" : "translate-x-0.5"}`}
          >
            {dark ? <Moon className="h-3 w-3 text-foreground" /> : <Sun className="h-3 w-3 text-foreground" />}
          </span>
        </button>
        <div className="flex items-center gap-1 text-xs">
          <Camera className="h-4 w-4" />
          <Badge variant={cameraOnline ? "default" : "destructive"} className="text-[10px]">
            {cameraOnline ? `${cameraCount} cam` : "offline"}
          </Badge>
        </div>
      </div>
    </div>
  )
}
