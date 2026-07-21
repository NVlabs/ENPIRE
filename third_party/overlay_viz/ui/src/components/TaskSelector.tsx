import { RefreshCw, PanelLeft, ScanSearch, Eraser } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { Task } from "@/api/types"

interface Props {
  tasks: Task[]
  loading: boolean
  selectedTaskId: string
  onSelect: (taskId: string) => void
  onRefresh: () => void
  hasEpisodes: boolean
  onToggleSidebar: () => void
  onAutoScreen?: () => void
  autoScreenEnabled?: boolean
  onClearCurrent?: () => void
  clearCurrentEnabled?: boolean
  onClearAll?: () => void
  clearAllEnabled?: boolean
}

export function TaskSelector({ tasks, loading, selectedTaskId, onSelect, onRefresh, hasEpisodes, onToggleSidebar, onAutoScreen, autoScreenEnabled, onClearCurrent, clearCurrentEnabled, onClearAll, clearAllEnabled }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3 mb-4">
      {hasEpisodes && (
        <Button variant="ghost" size="sm" onClick={onToggleSidebar}>
          <PanelLeft className="h-4 w-4 mr-1" /> Episodes
        </Button>
      )}
      <label className="font-semibold text-sm">Task:</label>
      <select
        className="border rounded px-2 py-1 text-sm bg-background"
        value={selectedTaskId}
        onChange={(e) => onSelect(e.target.value)}
        disabled={loading}
      >
        <option value="" disabled>Select a task...</option>
        {tasks.map((t) => (
          <option key={t.id} value={t.id}>{t.id}</option>
        ))}
      </select>
      <Button variant="ghost" size="icon" onClick={onRefresh} disabled={loading} className="h-8 w-8">
        <RefreshCw className="h-4 w-4" />
      </Button>
      {onAutoScreen && (
        <Button
          variant="outline"
          size="sm"
          onClick={onAutoScreen}
          disabled={!autoScreenEnabled}
          title={autoScreenEnabled ? "Scan every episode for too-short, latency spikes, and pure-color frames" : "Wait for the dataset scan to finish"}
          className="gap-1"
        >
          <ScanSearch className="h-4 w-4" /> Auto-Screen
        </Button>
      )}
      {onClearCurrent && (
        <Button
          variant="outline"
          size="sm"
          onClick={onClearCurrent}
          disabled={!clearCurrentEnabled}
          title={clearCurrentEnabled ? "Remove trim, value-trim, RTG, discarded flag, and progress labels from the open episode" : "Open an episode to clear its annotations"}
          className="gap-1"
        >
          <Eraser className="h-4 w-4" /> Clear
        </Button>
      )}
      {onClearAll && (
        <Button
          variant="outline"
          size="sm"
          onClick={onClearAll}
          disabled={!clearAllEnabled}
          title={clearAllEnabled ? "Remove all annotations from every episode in this task" : "Wait for the dataset scan to finish"}
          className="gap-1"
        >
          <Eraser className="h-4 w-4" /> Clear All
        </Button>
      )}
      {loading && <span className="text-xs text-muted-foreground">Loading...</span>}
    </div>
  )
}
