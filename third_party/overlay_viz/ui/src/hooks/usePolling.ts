// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react"

export function usePolling(fn: () => void, intervalMs: number, enabled = true) {
  const savedFn = useRef(fn)
  savedFn.current = fn

  useEffect(() => {
    if (!enabled) return
    const id = setInterval(() => savedFn.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, enabled])
}
