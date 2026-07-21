import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ActionLogEntry } from "../hooks/useRobotState";

interface ActionLogProps {
  entries: ActionLogEntry[];
  onClear?: () => void;
}

type FilterTab = "skills" | "all";

function isSkillEntry(entry: ActionLogEntry): boolean {
  if (entry.node_type) return false;
  const t = entry.tool;
  return t !== "execution";
}

const STATUS_BADGE: Record<string, string> = {
  running: "badge-warning",
  success: "badge-success",
  error: "badge-error",
};

const NODE_TYPE_BADGE: Record<string, string> = {
  for: "badge-info",
  function_def: "badge-ghost",
};

const NODE_TYPE_LABEL: Record<string, string> = {
  for: "for",
  function_def: "def",
};

function EntryDetails({ entry }: { entry: ActionLogEntry }) {
  return (
    <div className="mt-2 space-y-1">
      {entry.result !== undefined && (
        <div>
          <span className="text-xs font-medium text-base-content/60">Output:</span>
          <pre className="mt-0.5 overflow-x-auto whitespace-pre-wrap break-all rounded bg-base-300 p-2 text-xs">
            {typeof entry.result === "string"
              ? entry.result
              : JSON.stringify(entry.result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function ChildEntry({ entry }: { entry: ActionLogEntry }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="flex items-start gap-1.5 py-0.5">
      <div className="mt-1 shrink-0">
        <div
          className={`h-1.5 w-1.5 rounded-full ${
            entry.status === "running"
              ? "bg-warning animate-pulse"
              : entry.status === "success"
                ? "bg-success"
                : "bg-error"
          }`}
        />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1">
          <span className={`badge badge-xs ${STATUS_BADGE[entry.status] ?? ""}`}>
            {entry.tool}
          </span>
          <button
            className="btn btn-ghost btn-xs h-4 min-h-0 px-1 text-xs"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "▲" : "▼"}
          </button>
        </div>
        {expanded && <EntryDetails entry={entry} />}
      </div>
    </div>
  );
}

function LogEntry({
  entry,
  children,
}: {
  entry: ActionLogEntry;
  children: ActionLogEntry[];
}) {
  // Loop entries default expanded so users see iterations as they come in
  const [expanded, setExpanded] = useState(entry.node_type === "for");

  return (
    <li>
      <hr />
      <div className="timeline-start text-xs text-base-content/50 font-mono">
        {new Date(entry.timestamp).toLocaleTimeString()}
      </div>
      <div className="timeline-middle">
        <div
          className={`h-2.5 w-2.5 rounded-full ${
            entry.status === "running"
              ? "bg-warning animate-pulse"
              : entry.status === "success"
                ? "bg-success"
                : "bg-error"
          }`}
        />
      </div>
      <div className="timeline-end timeline-box text-sm min-w-0 max-w-full overflow-hidden">
        <div className="flex flex-wrap items-center gap-1.5">
          {entry.node_type && NODE_TYPE_BADGE[entry.node_type] && (
            <span className={`badge badge-sm ${NODE_TYPE_BADGE[entry.node_type]}`}>
              {NODE_TYPE_LABEL[entry.node_type] ?? entry.node_type}
            </span>
          )}
          <span className={`badge badge-sm ${STATUS_BADGE[entry.status] ?? ""}`}>
            {entry.tool}
          </span>
          <button
            className="btn btn-ghost btn-xs"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Hide" : "Show"} details
          </button>
        </div>

        {/* Child entries (loop iterations) */}
        {children.length > 0 && (
          <div className="mt-1.5 border-l-2 border-base-300 pl-2 space-y-0.5">
            {children.map((child) => (
              <ChildEntry key={child.id} entry={child} />
            ))}
          </div>
        )}

        {expanded && <EntryDetails entry={entry} />}
      </div>
      <hr />
    </li>
  );
}

// Distance from bottom (px) within which auto-scroll re-engages after a manual scroll.
const SCROLL_SNAP_THRESHOLD_PX = 50;

export default function ActionLog({ entries, onClear }: ActionLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomAnchorRef = useRef<HTMLDivElement>(null);
  const [filterTab, setFilterTab] = useState<FilterTab>("skills");
  // Starts true; set to false when user scrolls up, true again when they
  // scroll back within SCROLL_SNAP_THRESHOLD_PX of the bottom.
  const shouldAutoScroll = useRef(true);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    shouldAutoScroll.current =
      scrollHeight - scrollTop - clientHeight < SCROLL_SNAP_THRESHOLD_PX;
  };

  // useLayoutEffect fires after DOM update but before paint, ensuring
  // scrollHeight is fully computed when we scroll.
  useLayoutEffect(() => {
    if (shouldAutoScroll.current) {
      bottomAnchorRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [entries, filterTab]);

  const { rootEntries, childMap } = useMemo(() => {
    if (filterTab === "skills") {
      // In skills mode, show only skill entries as a flat list —
      // promote children (e.g. _ik_servo inside a for loop) to root level.
      const root = entries.filter((e) => isSkillEntry(e));
      return { rootEntries: root, childMap: new Map<string, ActionLogEntry[]>() };
    }
    const root: ActionLogEntry[] = [];
    const children = new Map<string, ActionLogEntry[]>();
    for (const entry of entries) {
      if (entry.parent_id) {
        const arr = children.get(entry.parent_id) ?? [];
        arr.push(entry);
        children.set(entry.parent_id, arr);
      } else {
        root.push(entry);
      }
    }
    return { rootEntries: root, childMap: children };
  }, [entries, filterTab]);

  const skillCount = useMemo(
    () => entries.filter((e) => isSkillEntry(e)).length,
    [entries],
  );
  const runningEntries = useMemo(
    () => entries.filter((e) => e.status === "running"),
    [entries],
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-2 py-1 shrink-0">
        <span className="text-xs font-semibold">Action Log</span>
        {entries.length > 0 && onClear && (
          <button className="btn btn-ghost btn-xs" onClick={onClear}>
            Clear All
          </button>
        )}
      </div>
      {/* Filter tabs */}
      <div className="flex gap-1 px-2 pb-1 shrink-0">
        <button
          className={`btn btn-xs ${filterTab === "skills" ? "btn-primary" : "btn-ghost"}`}
          onClick={() => setFilterTab("skills")}
        >
          Skills{skillCount > 0 ? ` (${skillCount})` : ""}
        </button>
        <button
          className={`btn btn-xs ${filterTab === "all" ? "btn-primary" : "btn-ghost"}`}
          onClick={() => setFilterTab("all")}
        >
          All{entries.length > 0 ? ` (${entries.length})` : ""}
        </button>
      </div>
      {runningEntries.length > 0 && (
        <div className="px-2 pb-1 shrink-0">
          <div className="rounded border border-warning/30 bg-warning/10 px-2 py-1">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-warning">
              Running now
            </div>
            <div className="flex flex-col gap-1">
              {runningEntries.slice(-4).map((entry) => (
                <div key={entry.id} className="flex items-center gap-2 text-xs">
                  <span className="h-2 w-2 rounded-full bg-warning animate-pulse shrink-0" />
                  <span className="badge badge-xs badge-warning">{entry.tool}</span>
                  <span className="truncate text-base-content/70">
                    {entry.args?.code ? String(entry.args.code) : ""}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      {rootEntries.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-base-content/30 text-sm">
          {entries.length === 0 ? "No actions yet" : "No skill calls yet"}
        </div>
      ) : (
        <div
          ref={scrollRef}
          className="flex-1 min-h-0 overflow-y-auto"
          onScroll={handleScroll}
        >
          <ul className="timeline timeline-vertical timeline-compact w-full overflow-hidden">
            {rootEntries.map((entry) => (
              <LogEntry
                key={entry.id}
                entry={entry}
                children={childMap.get(entry.id) ?? []}
              />
            ))}
          </ul>
          {entries.length > 0 && !entries.some((e) => e.status === "running") && (
            <div className="text-xs text-center text-success py-2 font-semibold">
              Finished
            </div>
          )}
          <div ref={bottomAnchorRef} />
        </div>
      )}
    </div>
  );
}
