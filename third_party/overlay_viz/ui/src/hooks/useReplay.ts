import { useState, useCallback } from "react"
import * as api from "@/api/client"
import { usePolling } from "./usePolling"

export function useReplay() {
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [loadedEpisode, setLoadedEpisode] = useState<number | null>(null)
  const [notice, setNotice] = useState("")
  const [noticeLevel, setNoticeLevel] = useState<"info" | "success" | "error">("info")

  const setReplayNotice = useCallback((msg: string, level: "info" | "success" | "error" = "info") => {
    setNotice(msg)
    setNoticeLevel(level)
  }, [])

  const clearNotice = useCallback(() => setNotice(""), [])

  usePolling(
    async () => {
      try {
        const s = await api.replayStatus()
        setConnected(s.connected)
        if (!s.connected) setLoadedEpisode(null)
      } catch {
        setConnected(false)
        setLoadedEpisode(null)
      }
    },
    5000,
  )

  const connect = useCallback(async () => {
    setBusy(true)
    try {
      const r = await api.replayConnect()
      setConnected(r.connected)
      setReplayNotice(r.connected ? "Connected." : "Failed to connect.", r.connected ? "success" : "error")
    } catch (e) {
      setReplayNotice(String(e), "error")
    } finally {
      setBusy(false)
    }
  }, [setReplayNotice])

  const disconnect = useCallback(async () => {
    setBusy(true)
    try {
      await api.replayDisconnect()
    } catch { /* best effort */ }
    setConnected(false)
    setLoadedEpisode(null)
    clearNotice()
    setBusy(false)
  }, [clearNotice])

  const wrapCmd = useCallback(
    async (fn: () => Promise<unknown>, label: string) => {
      setBusy(true)
      try {
        await fn()
      } catch (e) {
        setReplayNotice(`${label} failed: ${e}`, "error")
      } finally {
        setBusy(false)
      }
    },
    [setReplayNotice],
  )

  const play = useCallback(() => wrapCmd(() => api.replayPlay(), "Play"), [wrapCmd])
  const pause = useCallback(() => wrapCmd(() => api.replayPause(), "Pause"), [wrapCmd])
  const step = useCallback(() => wrapCmd(() => api.replayStep(), "Step"), [wrapCmd])
  const home = useCallback(() => wrapCmd(() => api.replayHome(), "Home"), [wrapCmd])
  const syncToInit = useCallback(() => wrapCmd(() => api.replaySyncToInit(), "Sync to init"), [wrapCmd])

  const load = useCallback(
    async (taskId: string, idx: number) => {
      setBusy(true)
      try {
        await api.replayLoad(taskId, idx)
        setLoadedEpisode(idx)
        setReplayNotice(`Episode #${idx} loaded.`, "success")
      } catch (e) {
        setLoadedEpisode(null)
        setReplayNotice(`Load failed: ${e}`, "error")
      } finally {
        setBusy(false)
      }
    },
    [setReplayNotice],
  )

  return {
    connected, busy, loadedEpisode, notice, noticeLevel,
    setReplayNotice, clearNotice,
    connect, disconnect, play, pause, step, home, syncToInit, load,
  }
}
