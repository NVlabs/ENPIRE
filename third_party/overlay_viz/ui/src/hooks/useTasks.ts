import { useState, useCallback, useEffect, useRef } from "react"
import * as api from "@/api/client"
import type { Task, EpisodeEntry, TimingHealthEntry } from "@/api/types"

export function useTasks() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedTaskId, setSelectedTaskId] = useState("")
  const [episodes, setEpisodes] = useState<EpisodeEntry[]>([])
  const [scanning, setScanning] = useState(false)
  const [scanDone, setScanDone] = useState(0)
  const [scanTotal, setScanTotal] = useState(0)
  const [scanFound, setScanFound] = useState(0)
  const [scanReady, setScanReady] = useState(false)
  const [timingHealth, setTimingHealth] = useState<Record<string, TimingHealthEntry>>({})
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const healthPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchTasks = useCallback(async () => {
    setLoading(true)
    try {
      setTasks(await api.fetchTasks())
    } catch {
      setTasks([])
    } finally {
      setLoading(false)
    }
  }, [])

  const stopHealthPolling = useCallback(() => {
    if (healthPollRef.current) {
      clearInterval(healthPollRef.current)
      healthPollRef.current = null
    }
  }, [])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    stopHealthPolling()
  }, [stopHealthPolling])

  const startHealthPolling = useCallback((taskId: string) => {
    stopHealthPolling()
    healthPollRef.current = setInterval(async () => {
      try {
        const h = await api.fetchTimingHealth(taskId)
        setTimingHealth(h.episodes)
        if (h.ready) stopHealthPolling()
      } catch { /* ignore */ }
    }, 1000)
  }, [stopHealthPolling])

  const selectTask = useCallback(
    async (taskId: string) => {
      setSelectedTaskId(taskId)
      setEpisodes([])
      setTimingHealth({})
      setScanReady(false)
      setScanning(false)
      stopPolling()

      try {
        await api.triggerScan(taskId)
      } catch { /* ignore */ }

      setScanning(true)
      pollRef.current = setInterval(async () => {
        try {
          const st = await api.fetchScanStatus(taskId)
          setScanDone(st.done)
          setScanTotal(st.total)
          setScanFound(st.episodes)
          if (st.ready) {
            setScanning(false)
            setScanReady(true)
            if (pollRef.current) {
              clearInterval(pollRef.current)
              pollRef.current = null
            }
            setEpisodes(await api.fetchEpisodes(taskId))
            startHealthPolling(taskId)
          }
        } catch { /* ignore */ }
      }, 500)
    },
    [stopPolling, startHealthPolling],
  )

  useEffect(() => {
    fetchTasks().then(async () => {
      try {
        const d = await api.fetchDefaultTask()
        if (d.task_id) selectTask(d.task_id)
      } catch { /* optional */ }
    })
    return stopPolling
  }, [fetchTasks, selectTask, stopPolling])

  const refreshEpisodes = useCallback(async (): Promise<EpisodeEntry[]> => {
    if (!selectedTaskId) return []
    try {
      const eps = await api.fetchEpisodes(selectedTaskId)
      setEpisodes(eps)
      return eps
    } catch {
      return []
    }
  }, [selectedTaskId])

  // Re-scan the dataset directory and refresh the episode list.  Used by
  // the "Sync" button and as the first step of "Next episode".  Returns
  // the freshly fetched list so callers can act on it synchronously.
  const resync = useCallback(async (): Promise<EpisodeEntry[]> => {
    if (!selectedTaskId) return []
    try {
      await api.triggerScan(selectedTaskId)
    } catch { /* ignore */ }
    for (let i = 0; i < 60; i++) {
      try {
        const st = await api.fetchScanStatus(selectedTaskId)
        setScanDone(st.done)
        setScanTotal(st.total)
        setScanFound(st.episodes)
        if (st.ready) break
      } catch { /* ignore */ }
      await new Promise(r => setTimeout(r, 300))
    }
    try {
      const eps = await api.fetchEpisodes(selectedTaskId)
      setEpisodes(eps)
      return eps
    } catch {
      return []
    }
  }, [selectedTaskId])

  const dataPath = tasks.find(t => t.id === selectedTaskId)?.data_path ?? null

  return {
    tasks, loading, selectedTaskId, selectTask, fetchTasks,
    episodes, scanning, scanReady, scanDone, scanTotal, scanFound,
    refreshEpisodes, resync, dataPath, timingHealth,
  }
}
