import { useState, useCallback, useEffect, useMemo } from "react"
import { Navbar } from "@/components/Navbar"
import { StatusBar } from "@/components/StatusBar"
import { TaskSelector } from "@/components/TaskSelector"
import { RobotControlPanel } from "@/components/RobotControlPanel"
import { EpisodeSidebar } from "@/components/EpisodeSidebar"
import { AutoScreenDialog } from "@/components/AutoScreenDialog"
import { EpisodeViewer } from "@/components/EpisodeViewer"
import { FinishedAnnotation } from "@/components/FinishedAnnotation"
import { ExportLerobotPanel } from "@/components/ExportLerobotPanel"
import { Notices, useNotices } from "@/components/Notices"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useTasks } from "@/hooks/useTasks"
import { useCameras } from "@/hooks/useCameras"
import { useReplay } from "@/hooks/useReplay"
import { useEpisode } from "@/hooks/useEpisode"
import * as api from "@/api/client"
import type { EpisodeEntry } from "@/api/types"

// Pick the episode the operator should land on next.
// Normal case: first unannotated episode that is not discarded / anomalous.
// Exception: if the very next slot after the last annotated episode is an
// anomaly (zero-length / broken / discarded), surface it so the operator
// can press "Completely Remove" instead of silently skipping past it.
function findNextEpisodeToShow(episodes: EpisodeEntry[]): number {
  const isAnomaly = (e: EpisodeEntry) => Boolean(e.discarded || e.anomalous)
  let lastAnnotatedIdx = -1
  for (let i = 0; i < episodes.length; i++) {
    if (episodes[i].annotated) lastAnnotatedIdx = i
  }
  const nextIdx = lastAnnotatedIdx + 1
  if (nextIdx < episodes.length && isAnomaly(episodes[nextIdx])) {
    return nextIdx
  }
  const firstNormal = episodes.findIndex(e => !e.annotated && !isAnomaly(e))
  return firstNormal
}

export default function App() {
  const tasks = useTasks()
  const cameras = useCameras()
  const replay = useReplay()
  const notices = useNotices()
  const [viewerIdx, setViewerIdx] = useState<number | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [autoScreenOpen, setAutoScreenOpen] = useState(false)
  const [mode, setMode] = useState("inspect")
  const [valueServerEnabled, setValueServerEnabled] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [autoSelected, setAutoSelected] = useState(false)
  const [autoPlayPending, setAutoPlayPending] = useState(false)
  // Continuous playback: keep playing the trimmed clip end-to-end across
  // episode boundaries. Survives episode remount so autoPlayPending can
  // re-trigger play on the next episode.
  const [continuousPlay, setContinuousPlay] = useState(false)
  const [finished, setFinished] = useState(false)
  const episode = useEpisode(tasks.selectedTaskId, viewerIdx)

  useEffect(() => {
    api.fetchMode().then((res) => {
      setMode(res.mode)
      setValueServerEnabled(Boolean(res.value_server_enabled))
    }).catch(() => {})
  }, [])

  useEffect(() => {
    episode.setOnDiscardChanged(tasks.refreshEpisodes)
  }, [episode.setOnDiscardChanged, tasks.refreshEpisodes])

  useEffect(() => {
    episode.setOnSaveNotice((title, savedPath, episodeDir) => {
      notices.push({
        title,
        detail: savedPath,
        level: "success",
        action: {
          label: "Open folder",
          onClick: () => {
            api.openPath(episodeDir).catch(e => {
              notices.push({ title: "Couldn't open folder", detail: String(e), level: "error" })
            })
          },
        },
      })
    })
  }, [episode.setOnSaveNotice, notices])

  const openEpisode = useCallback((idx: number) => {
    setFinished(false)
    setViewerIdx(idx)
  }, [])

  // Clear the finished-annotation screen when switching tasks — otherwise
  // it would persist across a task change that has its own unannotated work.
  useEffect(() => {
    setFinished(false)
  }, [tasks.selectedTaskId])

  // Auto-open on first task scan: prefer the next work-item per the
  // findNextEpisodeToShow rules (skip discarded/anomalous, except when
  // the slot immediately after the last annotated episode is itself an
  // anomaly — in that case show it so the operator can act on it).
  useEffect(() => {
    if (autoSelected) return
    if (!tasks.selectedTaskId || !tasks.scanReady) return
    if (tasks.episodes.length === 0) return
    if (viewerIdx !== null) return
    const target = findNextEpisodeToShow(tasks.episodes)
    setViewerIdx(target >= 0 ? target : 0)
    setAutoSelected(true)
  }, [autoSelected, tasks.selectedTaskId, tasks.scanReady, tasks.episodes, viewerIdx])

  const closeViewer = useCallback(() => setViewerIdx(null), [])

  const navigate = useCallback(
    async (delta: number) => {
      if (viewerIdx === null) return
      // Refresh the episode list so annotation/discard counts stay fresh
      // after the operator just saved/discarded the current episode. Uses
      // refreshEpisodes (metadata-only re-read) rather than resync (full
      // directory rescan + status poll) — Next/Prev fires on every arrow
      // press, so the rescan version made every nav feel laggy.
      const eps = await tasks.refreshEpisodes().catch(() => tasks.episodes)
      // Skip past discarded episodes — auto-screen flags them and the
      // operator has already decided they're out of scope. Preserve the
      // sign of delta so reverse nav skips backward.
      const step = delta > 0 ? 1 : -1
      let next = viewerIdx + delta
      while (next >= 0 && next < eps.length && eps[next].discarded) {
        next += step
      }
      if (next >= 0 && next < eps.length) {
        if (delta > 0) setAutoPlayPending(true)
        setViewerIdx(next)
        return
      }
      // Past-the-end on forward nav: show the Finished panel. The
      // progress/counts refresh from the resync above, so what the
      // operator sees on the finish screen is current.
      if (delta > 0 && next >= eps.length) {
        setFinished(true)
      }
    },
    [viewerIdx, tasks],
  )

  const handleSync = useCallback(async () => {
    setSyncing(true)
    try {
      const eps = await tasks.resync()
      if (viewerIdx !== null && viewerIdx >= eps.length) {
        setViewerIdx(eps.length > 0 ? eps.length - 1 : null)
      }
    } finally {
      setSyncing(false)
    }
  }, [tasks, viewerIdx])

  // Mutually-exclusive tally: discarded wins over annotated, annotated over unannotated.
  const episodeCounts = useMemo(() => {
    let annotated = 0, discarded = 0, unannotated = 0
    for (const e of tasks.episodes) {
      if (e.discarded) discarded += 1
      else if (e.annotated) annotated += 1
      else unannotated += 1
    }
    return { annotated, discarded, unannotated }
  }, [tasks.episodes])

  const handleClearCurrent = useCallback(async () => {
    if (!tasks.selectedTaskId || viewerIdx === null) return
    const folder = tasks.episodes[viewerIdx]?.folder ?? `episode ${viewerIdx}`
    const ok = window.confirm(
      `Clear all annotations for ${folder}?\n\nRemoves trim, value-trim, RTG, discarded flag, and progress labels. This cannot be undone.`,
    )
    if (!ok) return
    try {
      const res = await api.clearAnnotations(tasks.selectedTaskId, viewerIdx)
      notices.push({
        title: res.removed_keys.length || res.removed_progress_labels
          ? `Cleared annotations for ${folder}`
          : `No annotations to clear for ${folder}`,
        detail: res.removed_keys.length ? `removed: ${res.removed_keys.join(", ")}` : undefined,
        level: "success",
      })
      await tasks.refreshEpisodes()
      await episode.refreshInfo?.()
    } catch (e) {
      notices.push({ title: "Clear failed", detail: String(e), level: "error" })
    }
  }, [tasks, viewerIdx, notices, episode])

  const handleClearAll = useCallback(async () => {
    if (!tasks.selectedTaskId || tasks.episodes.length === 0) return
    const ok = window.confirm(
      `Clear all annotations for every one of the ${tasks.episodes.length} episodes in this task?\n\nRemoves trim, value-trim, RTG, discarded flag, and progress labels for each. This cannot be undone.`,
    )
    if (!ok) return
    try {
      const res = await api.clearAllAnnotations(tasks.selectedTaskId)
      notices.push({
        title: `Cleared ${res.cleared} of ${res.total} episode${res.total === 1 ? "" : "s"}`,
        detail: res.errors ? `${res.errors} error${res.errors === 1 ? "" : "s"} — see server log` : undefined,
        level: res.errors ? "error" : "success",
      })
      await tasks.refreshEpisodes()
      await episode.refreshInfo?.()
    } catch (e) {
      notices.push({ title: "Clear all failed", detail: String(e), level: "error" })
    }
  }, [tasks, notices, episode])

  const handlePurgeDiscarded = useCallback(async () => {
    if (!tasks.selectedTaskId || episodeCounts.discarded === 0) return
    const ok = window.confirm(
      `Permanently delete all ${episodeCounts.discarded} episode folders marked discarded?\n\nThis will rm -rf each one. This cannot be undone.`,
    )
    if (!ok) return
    try {
      const res = await api.purgeDiscarded(tasks.selectedTaskId)
      notices.push({
        title: `Removed ${res.deleted_count} discarded episode${res.deleted_count === 1 ? "" : "s"}`,
        level: "success",
      })
      await tasks.resync()
      setViewerIdx(null)
    } catch (e) {
      notices.push({
        title: "Purge failed",
        detail: String(e),
        level: "error",
      })
    }
  }, [tasks, episodeCounts.discarded, notices])

  const handleNextEpisode = useCallback(async () => {
    setSyncing(true)
    try {
      // refreshEpisodes re-reads metadata only; use the explicit Sync
      // button if the operator needs to pick up new folders on disk.
      const eps = await tasks.refreshEpisodes()
      if (eps.length === 0) return
      const target = findNextEpisodeToShow(eps)
      if (target >= 0) {
        setAutoPlayPending(true)
        openEpisode(target)
        return
      }
      if (viewerIdx !== null && viewerIdx + 1 < eps.length) {
        setAutoPlayPending(true)
        openEpisode(viewerIdx + 1)
        return
      }
      notices.push({
        title: "Finished annotating all episodes in the folder.",
        level: "success",
      })
    } finally {
      setSyncing(false)
    }
  }, [tasks, viewerIdx, openEpisode, notices])

  const handleOpenFolder = useCallback(async () => {
    if (!tasks.dataPath) return
    try {
      await api.openPath(tasks.dataPath)
    } catch (e) {
      notices.push({
        title: "Couldn't open folder",
        detail: String(e),
        level: "error",
      })
    }
  }, [tasks.dataPath, notices])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Cmd+\ or Ctrl+\ toggles episode sidebar
      if (e.key === "\\" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setSidebarOpen((prev) => !prev)
        return
      }
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return
      if (viewerIdx === null) return
      if (e.key === "ArrowLeft") { e.preventDefault(); navigate(-1) }
      else if (e.key === "ArrowRight") { e.preventDefault(); navigate(1) }
      else if (e.key === "Escape") closeViewer()
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [viewerIdx, navigate, closeViewer])

  return (
    <TooltipProvider>
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Navbar cameraOnline={cameras.online} cameraCount={cameras.count} />

      <EpisodeSidebar
        open={sidebarOpen}
        onOpenChange={setSidebarOpen}
        episodes={tasks.episodes}
        viewerIdx={viewerIdx}
        onSelect={(idx) => { openEpisode(idx); setSidebarOpen(false) }}
        timingHealth={tasks.timingHealth}
        onSync={handleSync}
        onNextEpisode={handleNextEpisode}
        onOpenFolder={handleOpenFolder}
        onPurgeDiscarded={handlePurgeDiscarded}
        discardedCount={episodeCounts.discarded}
        syncing={syncing}
        canOpenFolder={Boolean(tasks.dataPath)}
      />

      <AutoScreenDialog
        taskId={tasks.selectedTaskId}
        open={autoScreenOpen}
        onClose={() => setAutoScreenOpen(false)}
        onOpenEpisode={(epIdx) => {
          const pos = tasks.episodes.findIndex((e) => e.idx === epIdx)
          if (pos >= 0) {
            openEpisode(pos)
            setSidebarOpen(false)
          }
        }}
        onComplete={(res) => {
          if (!res.dry_run) tasks.resync().catch(() => {})
        }}
      />

      <main className="flex-1 overflow-y-auto p-4 max-w-7xl w-full mx-auto">
        <TaskSelector
          tasks={tasks.tasks}
          loading={tasks.loading}
          selectedTaskId={tasks.selectedTaskId}
          onSelect={tasks.selectTask}
          onRefresh={tasks.fetchTasks}
          hasEpisodes={tasks.episodes.length > 0}
          onToggleSidebar={() => setSidebarOpen(true)}
          onAutoScreen={tasks.selectedTaskId ? () => setAutoScreenOpen(true) : undefined}
          autoScreenEnabled={Boolean(tasks.selectedTaskId) && tasks.scanReady && !tasks.scanning}
          onClearCurrent={tasks.selectedTaskId ? handleClearCurrent : undefined}
          clearCurrentEnabled={Boolean(tasks.selectedTaskId) && viewerIdx !== null && tasks.scanReady && !tasks.scanning}
          onClearAll={tasks.selectedTaskId ? handleClearAll : undefined}
          clearAllEnabled={Boolean(tasks.selectedTaskId) && tasks.episodes.length > 0 && tasks.scanReady && !tasks.scanning}
        />

        {tasks.scanning && (
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-1 text-sm">
              <span>Scanning: {tasks.scanDone}/{tasks.scanTotal} ({tasks.scanFound} found)</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all"
                style={{ width: tasks.scanTotal > 0 ? `${(tasks.scanDone / tasks.scanTotal) * 100}%` : "0%" }}
              />
            </div>
          </div>
        )}

        {mode !== "label" && (
          <RobotControlPanel
            connected={replay.connected}
            busy={replay.busy}
            loadedEpisode={replay.loadedEpisode}
            notice={replay.notice}
            noticeLevel={replay.noticeLevel}
            viewerIdx={viewerIdx}
            onConnect={replay.connect}
            onDisconnect={replay.disconnect}
            onPlay={replay.play}
            onPause={replay.pause}
            onStep={replay.step}
            onHome={replay.home}
            onSyncToInit={replay.syncToInit}
            onLoadEpisode={() => viewerIdx !== null && replay.load(tasks.selectedTaskId, viewerIdx)}
          />
        )}

        {tasks.selectedTaskId && (
          <ExportLerobotPanel
            taskId={tasks.selectedTaskId}
            taskDataPath={tasks.dataPath}
          />
        )}

        {finished && (
          <FinishedAnnotation
            folderPath={tasks.dataPath}
            annotatedCount={episodeCounts.annotated}
            discardedCount={episodeCounts.discarded}
            totalCount={tasks.episodes.length}
            onOpenFolder={handleOpenFolder}
            onReturn={viewerIdx !== null ? () => setFinished(false) : undefined}
          />
        )}

        {viewerIdx !== null && !finished && (
          <EpisodeViewer
            taskId={tasks.selectedTaskId}
            viewerIdx={viewerIdx}
            episode={episode}
            episodeCount={tasks.episodes.length}
            episodeEntry={tasks.episodes[viewerIdx]}
            annotatedCount={episodeCounts.annotated}
            discardedCount={episodeCounts.discarded}
            unannotatedCount={episodeCounts.unannotated}
            replayConnected={replay.connected}
            cameraStreaming={cameras.streaming}
            mode={mode}
            valueServerEnabled={valueServerEnabled}
            autoPlayPending={autoPlayPending}
            onAutoPlayConsumed={() => setAutoPlayPending(false)}
            continuousPlay={continuousPlay}
            onContinuousPlayChange={setContinuousPlay}
            onClose={closeViewer}
            onNavigate={navigate}
            onCompletelyRemove={async () => {
              const ep = tasks.episodes[viewerIdx]
              if (!ep) return
              const ok = window.confirm(
                `Permanently delete ${ep.folder}?\n\nThis will rm -rf the entire episode folder from disk. This cannot be undone.`
              )
              if (!ok) return
              try {
                await api.deleteEpisode(tasks.selectedTaskId, viewerIdx)
                notices.push({
                  title: "Episode removed",
                  detail: ep.folder,
                  level: "success",
                })
                setViewerIdx(null)
                await handleNextEpisode()
              } catch (e) {
                notices.push({
                  title: "Failed to remove episode",
                  detail: String(e),
                  level: "error",
                })
              }
            }}
            onSaveNotice={(title, savedPath, episodeDir) => {
              notices.push({
                title,
                detail: savedPath,
                level: "success",
                action: {
                  label: "Open folder",
                  onClick: () => {
                    api.openPath(episodeDir).catch(e => {
                      notices.push({ title: "Couldn't open folder", detail: String(e), level: "error" })
                    })
                  },
                },
              })
            }}
          />
        )}

        {!tasks.selectedTaskId && !tasks.loading && (
          <div className="text-center py-20 opacity-60">Select a task to view episodes</div>
        )}
        {tasks.selectedTaskId && !tasks.scanning && tasks.episodes.length === 0 && tasks.scanReady && (
          <div className="text-center py-20 opacity-60">No episodes found for this task</div>
        )}
      </main>

      <StatusBar
        selectedTaskId={tasks.selectedTaskId}
        episodeCount={tasks.episodes.length}
        viewerIdx={viewerIdx}
        replayConnected={replay.connected}
        cameraOnline={cameras.online}
        cameraCount={cameras.count}
      />

      <Notices notices={notices.notices} onDismiss={notices.dismiss} />
    </div>
    </TooltipProvider>
  )
}
