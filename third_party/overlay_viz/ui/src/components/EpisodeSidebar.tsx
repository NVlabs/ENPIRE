import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { RefreshCw, SkipForward, FolderOpen, CheckCircle2, CircleDashed, Trash2, Eye, EyeOff } from "lucide-react"
import type { EpisodeEntry, TimingHealthEntry } from "@/api/types"

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  episodes: EpisodeEntry[]
  viewerIdx: number | null
  onSelect: (idx: number) => void
  timingHealth: Record<string, TimingHealthEntry>
  onSync: () => void
  onNextEpisode: () => void
  onOpenFolder: () => void
  onPurgeDiscarded?: () => void
  discardedCount?: number
  syncing?: boolean
  canOpenFolder?: boolean
}

export function EpisodeSidebar({ open, onOpenChange, episodes, viewerIdx, onSelect, timingHealth, onSync, onNextEpisode, onOpenFolder, onPurgeDiscarded, discardedCount, syncing, canOpenFolder }: Props) {
  // Hide discarded episodes by default so auto-screen + manual discards
  // drop out of the segmentation workflow. The toggle lets the operator
  // bring them back to un-discard or inspect false positives.
  const [showDiscarded, setShowDiscarded] = useState(false)
  const visibleCount = showDiscarded
    ? episodes.length
    : episodes.filter((e) => !e.discarded).length
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-[260px] p-0">
        <SheetHeader className="px-3 py-2 border-b pr-10">
          <SheetTitle className="text-xs font-bold tracking-wide flex items-center justify-between">
            EPISODES
            <Badge variant="outline" className="text-[10px]">
              {visibleCount}{showDiscarded || (discardedCount ?? 0) === 0 ? "" : `/${episodes.length}`}
            </Badge>
          </SheetTitle>
          <div className="flex flex-wrap gap-1 mt-1">
            <Button size="sm" variant="outline" onClick={onSync} disabled={syncing} className="h-7 text-[11px] px-2 gap-1">
              <RefreshCw className={`h-3 w-3 ${syncing ? "animate-spin" : ""}`} /> Sync
            </Button>
            <Button size="sm" variant="outline" onClick={onNextEpisode} disabled={syncing} title="Jump to the first episode without a saved trim" className="h-7 text-[11px] px-2 gap-1">
              <SkipForward className="h-3 w-3" /> Next unannotated
            </Button>
            <Button size="sm" variant="outline" onClick={onOpenFolder} disabled={!canOpenFolder} className="h-7 text-[11px] px-2 gap-1">
              <FolderOpen className="h-3 w-3" /> Open folder
            </Button>
            {(discardedCount ?? 0) > 0 && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowDiscarded((v) => !v)}
                title={showDiscarded ? "Hide discarded episodes from the list" : "Show discarded episodes (greyed out)"}
                className="h-7 text-[11px] px-2 gap-1"
              >
                {showDiscarded ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                {showDiscarded ? "Hide" : "Show"} {discardedCount} discarded
              </Button>
            )}
            {onPurgeDiscarded && (discardedCount ?? 0) > 0 && (
              <Button
                size="sm"
                variant="outline"
                onClick={onPurgeDiscarded}
                disabled={syncing}
                title="Permanently delete every episode folder marked discarded"
                className="h-7 text-[11px] px-2 gap-1 border-red-500/70 text-red-300 bg-red-950/40 hover:bg-red-900/60"
              >
                <Trash2 className="h-3 w-3" /> Purge {discardedCount} discarded
              </Button>
            )}
          </div>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          {episodes.map((ep, idx) => {
            if (ep.discarded && !showDiscarded) return null
            const health = timingHealth[String(ep.idx)]
            const showDot = health && health.has_data && health.level !== "good"
            return (
              <button
                key={ep.idx}
                onClick={() => onSelect(idx)}
                className={`w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 border-b border-border/50 hover:bg-accent transition-colors ${
                  viewerIdx === idx ? "bg-primary/10 font-semibold text-primary" : ""
                } ${ep.discarded ? "opacity-40" : ""}`}
              >
                <Badge variant={viewerIdx === idx ? "default" : "outline"} className={`text-[10px] ${ep.discarded ? "border-red-500/50" : ""}`}>
                  #{ep.idx}
                </Badge>
                <Tooltip>
                  <TooltipTrigger asChild>
                    {ep.annotated ? (
                      <CheckCircle2 className="h-3 w-3 shrink-0 text-green-500" />
                    ) : (
                      <CircleDashed className="h-3 w-3 shrink-0 text-muted-foreground/50" />
                    )}
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <span className="text-xs">{ep.annotated ? "Trim saved" : "Not annotated"}</span>
                  </TooltipContent>
                </Tooltip>
                <span className={`truncate ${ep.discarded ? "line-through" : ""}`}>{ep.folder}</span>
                {showDot && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className={`ml-auto w-2 h-2 rounded-full shrink-0 ${
                        health.level === "bad" ? "bg-red-500" : "bg-amber-400"
                      }`} />
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <div className="text-xs">
                        <div className="font-semibold mb-1">Timing: {health.level}</div>
                        <div>Jitter: {(health.jitter_ratio * 100).toFixed(1)}%</div>
                        <div>Spikes: {health.spike_count}</div>
                        <div>Max gap: {(health.max_gap_s * 1000).toFixed(1)}ms</div>
                        <div>Median dt: {(health.median_dt * 1000).toFixed(1)}ms</div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                )}
              </button>
            )
          })}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}
