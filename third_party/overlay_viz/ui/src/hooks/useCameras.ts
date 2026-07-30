// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useCallback } from "react"
import * as api from "@/api/client"
import { usePolling } from "./usePolling"

export function useCameras() {
  const [online, setOnline] = useState(false)
  const [count, setCount] = useState(0)
  const [devices, setDevices] = useState<{ name: string; serial: string }[]>([])
  const [streaming, setStreaming] = useState(false)

  const poll = useCallback(async () => {
    try {
      const s = await api.fetchCameraStatus()
      setOnline(s.online)
      setCount(s.count)
      setDevices(s.devices)
      setStreaming(s.streaming)
    } catch {
      setOnline(false)
      setCount(0)
      setDevices([])
      setStreaming(false)
    }
  }, [])

  usePolling(poll, 5000)

  return { online, count, devices, streaming }
}
