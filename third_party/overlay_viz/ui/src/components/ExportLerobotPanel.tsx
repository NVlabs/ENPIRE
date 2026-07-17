import { useEffect, useRef, useState } from "react"
import { ChevronDown, ChevronUp, Loader2, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import * as api from "@/api/client"

const MINIMAL_POLICY_DIR_KEY = "overlay-viz-lerobot-minimal-policy-dir-v1"

interface Props {
  taskId: string
  taskDataPath: string | null
}

export function ExportLerobotPanel({ taskId, taskDataPath }: Props) {
  const [open, setOpen] = useState(false)
  const [minimalPolicyDir, setMinimalPolicyDir] = useState<string>(() => {
    try {
      return localStorage.getItem(MINIMAL_POLICY_DIR_KEY) || ""
    } catch {
      return ""
    }
  })
  const [inputRoot, setInputRoot] = useState<string>(taskDataPath || "")
  const [outputDir, setOutputDir] = useState<string>(
    taskDataPath ? `${taskDataPath}_annotated` : "",
  )
  const [taskName, setTaskName] = useState<string>("")
  const [minSegmentLength, setMinSegmentLength] = useState<number>(64)
  const [annotateOnly, setAnnotateOnly] = useState<boolean>(true)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [status, setStatus] = useState<api.ExportLerobotStatus | null>(null)
  const stdoutRef = useRef<HTMLPreElement | null>(null)
  const userTouchedOutputRef = useRef(false)

  useEffect(() => {
    try {
      localStorage.setItem(MINIMAL_POLICY_DIR_KEY, minimalPolicyDir)
    } catch { /* ignore quota */ }
  }, [minimalPolicyDir])

  // Sync defaults from the selected task — only when the user hasn't
  // manually overridden output-dir yet; otherwise we'd clobber their edit
  // every time the task changes.
  useEffect(() => {
    if (!taskDataPath) return
    setInputRoot(taskDataPath)
    if (!userTouchedOutputRef.current) {
      setOutputDir(`${taskDataPath}_annotated`)
    }
  }, [taskDataPath])

  useEffect(() => {
    api.fetchExportLerobotStatus(taskId).then(setStatus).catch(() => {})
  }, [taskId])

  useEffect(() => {
    if (!status?.running) return
    const id = setInterval(() => {
      api.fetchExportLerobotStatus(taskId).then(setStatus).catch(() => {})
    }, 500)
    return () => clearInterval(id)
  }, [status?.running, taskId])

  // Keep the log tail in view while new lines stream in.
  useEffect(() => {
    const el = stdoutRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [status?.stdout])

  const onRun = async () => {
    setSubmitting(true)
    setSubmitError(null)
    try {
      await api.startExportLerobot(taskId, {
        minimal_policy_dir: minimalPolicyDir,
        input_root: inputRoot,
        output_dir: outputDir,
        task_name: taskName.trim() || null,
        min_segment_length: minSegmentLength,
        annotate_only: annotateOnly,
      })
      const s = await api.fetchExportLerobotStatus(taskId)
      setStatus(s)
    } catch (e) {
      setSubmitError(String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const canRun =
    !!minimalPolicyDir &&
    !!inputRoot &&
    !!outputDir &&
    !submitting &&
    !status?.running

  const statusLabel = status?.running ? (
    <span className="text-amber-500">running</span>
  ) : status?.exit_code === 0 ? (
    <span className="text-green-500">done (exit 0)</span>
  ) : status?.exit_code != null ? (
    <span className="text-red-500">exit {status.exit_code}</span>
  ) : status?.error ? (
    <span className="text-red-500">error</span>
  ) : (
    <span className="text-muted-foreground">idle</span>
  )

  return (
    <div className="mb-4 bg-card border rounded-xl overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2 text-sm font-semibold hover:bg-muted/40"
      >
        <span className="flex items-center gap-2">
          Export to lerobot v2.1
          {status?.running && <Loader2 className="h-3 w-3 animate-spin text-amber-500" />}
        </span>
        {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>

      {open && (
        <div className="px-4 py-3 border-t space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <label className="w-40 text-muted-foreground shrink-0">minimal_policy dir</label>
            <input
              type="text"
              value={minimalPolicyDir}
              onChange={(e) => setMinimalPolicyDir(e.target.value)}
              placeholder="/path/to/minimal_policy"
              className="flex-1 h-7 px-2 bg-muted rounded border font-mono text-[11px] outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="w-40 text-muted-foreground shrink-0">--input-root</label>
            <input
              type="text"
              value={inputRoot}
              onChange={(e) => setInputRoot(e.target.value)}
              className="flex-1 h-7 px-2 bg-muted rounded border font-mono text-[11px] outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="w-40 text-muted-foreground shrink-0">--output-dir</label>
            <input
              type="text"
              value={outputDir}
              onChange={(e) => {
                userTouchedOutputRef.current = true
                setOutputDir(e.target.value)
              }}
              className="flex-1 h-7 px-2 bg-muted rounded border font-mono text-[11px] outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="w-40 text-muted-foreground shrink-0">--task-name</label>
            <input
              type="text"
              value={taskName}
              onChange={(e) => setTaskName(e.target.value)}
              placeholder="(none — arg omitted)"
              className="flex-1 h-7 px-2 bg-muted rounded border font-mono text-[11px] outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="w-40 text-muted-foreground shrink-0">--min-segment-length</label>
            <input
              type="number"
              min={1}
              step={1}
              value={minSegmentLength}
              onChange={(e) =>
                setMinSegmentLength(Math.max(1, parseInt(e.target.value, 10) || 64))
              }
              className="h-7 w-24 px-2 bg-muted rounded border font-mono text-[11px] outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={annotateOnly}
                onChange={(e) => setAnnotateOnly(e.target.checked)}
              />
              <span className="font-mono">--annotate-only</span>
            </label>
          </div>
          <div className="pt-2 flex items-center gap-3">
            <Button size="sm" onClick={onRun} disabled={!canRun} className="gap-1">
              {status?.running || submitting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Play className="h-3 w-3" />
              )}
              Run export
            </Button>
            {submitError && (
              <span className="text-[10px] text-red-500 font-mono">{submitError}</span>
            )}
          </div>
        </div>
      )}

      {(status?.command || status?.stdout || status?.error) && (
        <div className="px-4 py-3 border-t text-xs space-y-2">
          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-muted-foreground">status:</span>
            {statusLabel}
            {status?.cwd && (
              <span className="text-muted-foreground font-mono truncate">
                cwd: <span className="opacity-70">{status.cwd}</span>
              </span>
            )}
          </div>
          {status?.command && (
            <pre className="font-mono text-[10px] text-muted-foreground whitespace-pre-wrap break-all">
              $ {status.command}
            </pre>
          )}
          {status?.error && (
            <pre className="font-mono text-[10px] text-red-400 whitespace-pre-wrap">
              {status.error}
            </pre>
          )}
          {status?.stdout && (
            <pre
              ref={stdoutRef}
              className="font-mono text-[10px] bg-black/50 text-green-300 rounded p-2 max-h-96 overflow-auto whitespace-pre-wrap"
            >
              {status.stdout}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
