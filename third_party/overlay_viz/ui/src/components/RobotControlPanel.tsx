import { Play, Pause, SkipForward, Home, Zap, Unplug, Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface Props {
  connected: boolean
  busy: boolean
  loadedEpisode: number | null
  notice: string
  noticeLevel: "info" | "success" | "error"
  viewerIdx: number | null
  onConnect: () => void
  onDisconnect: () => void
  onPlay: () => void
  onPause: () => void
  onStep: () => void
  onHome: () => void
  onSyncToInit: () => void
  onLoadEpisode: () => void
}

export function RobotControlPanel(p: Props) {
  return (
    <div className={`mb-4 p-3 rounded-xl border-2 ${p.connected ? "border-green-500/40 bg-green-500/5" : "border-red-500/30 bg-muted"}`}>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="font-bold text-sm tracking-wide">ROBOT CONTROL</span>
          <Badge variant={p.connected ? "default" : "destructive"}>
            {p.connected ? "ONLINE" : "OFFLINE"}
          </Badge>
        </div>

        {!p.connected && (
          <Button size="sm" variant="outline" onClick={p.onConnect} disabled={p.busy}>
            <Zap className="h-4 w-4 mr-1" /> Connect to robot
          </Button>
        )}

        {p.connected && (
          <div className="flex gap-2">
            <Button size="sm" onClick={p.onPlay} disabled={p.busy}><Play className="h-4 w-4 mr-1" /> Play</Button>
            <Button size="sm" variant="secondary" onClick={p.onPause} disabled={p.busy}><Pause className="h-4 w-4 mr-1" /> Pause</Button>
            <Button size="sm" variant="outline" onClick={p.onStep} disabled={p.busy}><SkipForward className="h-4 w-4 mr-1" /> Step</Button>
            <Button size="sm" variant="outline" onClick={p.onHome} disabled={p.busy}><Home className="h-4 w-4 mr-1" /> Home</Button>
            <Button size="sm" variant="outline" onClick={p.onSyncToInit} disabled={p.busy || p.loadedEpisode === null}>Sync to Init</Button>
            <Button size="sm" variant="destructive" onClick={p.onDisconnect} disabled={p.busy}><Unplug className="h-4 w-4 mr-1" /> Disconnect from robot</Button>
          </div>
        )}

        {p.connected && p.viewerIdx !== null && (
          <Button size="sm" variant="secondary" onClick={p.onLoadEpisode} disabled={p.busy}>
            <Download className="h-4 w-4 mr-1" /> Load Ep #{p.viewerIdx}
          </Button>
        )}

        {p.loadedEpisode !== null && <Badge variant="outline">Episode #{p.loadedEpisode} loaded</Badge>}
        {p.busy && <Loader2 className="h-4 w-4 animate-spin" />}
      </div>

      {p.notice && (
        <div className={`mt-2 text-sm p-2 rounded ${
          p.noticeLevel === "error" ? "bg-red-500/10 text-red-400" :
          p.noticeLevel === "success" ? "bg-green-500/10 text-green-400" :
          "bg-blue-500/10 text-blue-400"
        }`}>
          {p.notice}
        </div>
      )}
    </div>
  )
}
