import { Badge } from "@/components/ui/badge"

interface Props {
  selectedTaskId: string
  episodeCount: number
  viewerIdx: number | null
  replayConnected: boolean
  cameraOnline: boolean
  cameraCount: number
}

export function StatusBar({ selectedTaskId, episodeCount, viewerIdx, replayConnected, cameraOnline, cameraCount }: Props) {
  return (
    <footer className="border-t px-4 py-1.5 text-xs text-muted-foreground flex items-center justify-between gap-2">
      <div className="flex items-center gap-3">
        {selectedTaskId && <span>Task: <b>{selectedTaskId}</b></span>}
        {episodeCount > 0 && <span>Episodes: <b>{episodeCount}</b></span>}
        {viewerIdx !== null && <span>Viewing: <b>#{viewerIdx}</b></span>}
      </div>
      <div className="flex items-center gap-2">
        <Badge variant={replayConnected ? "default" : "destructive"} className="text-[10px]">
          {replayConnected ? "robot connected" : "robot offline"}
        </Badge>
        <Badge variant={cameraOnline ? "default" : "destructive"} className="text-[10px]">
          {cameraOnline ? `${cameraCount} camera(s)` : "cameras offline"}
        </Badge>
        <span className="opacity-60">Arrow keys: navigate episodes</span>
      </div>
    </footer>
  )
}
