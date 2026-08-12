/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ActionLogEntry } from "../hooks/useRobotState";

interface RuntimeStreamsPanelProps {
  entries: ActionLogEntry[];
  open: boolean;
  onToggle: () => void;
}

const MAX_ROWS = 200;

export default function RuntimeStreamsPanel({
  entries,
  open,
  onToggle,
}: RuntimeStreamsPanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const streamRows = useMemo(() => {
    return entries
      .filter((entry) => Boolean(entry.stdout || entry.stderr || entry.status === "error"))
      .slice(-MAX_ROWS);
  }, [entries]);

  useLayoutEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [streamRows, open]);

  return (
    <div>
      <button
        className="btn btn-ghost btn-xs w-full justify-between"
        onClick={onToggle}
      >
        <span className="text-xs font-semibold">Runtime Streams</span>
        <span className="text-xs">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="mt-1 rounded-lg border border-base-300 bg-base-200">
          <div className="flex items-center justify-between border-b border-base-300 px-3 py-2 text-[11px] text-base-content/60">
            <span>stdout / stderr / warnings / runtime errors</span>
            <span>{streamRows.length} row{streamRows.length === 1 ? "" : "s"}</span>
          </div>
          {streamRows.length === 0 ? (
            <div className="p-3 text-xs text-base-content/40">
              No runtime stream output yet.
            </div>
          ) : (
            <div ref={scrollRef} className="max-h-64 overflow-auto">
              <table className="table table-xs">
                <thead className="sticky top-0 z-10 bg-base-200">
                  <tr>
                    <th className="w-20">Time</th>
                    <th className="w-24">Source</th>
                    <th className="w-16">Status</th>
                    <th className="w-auto">Streams</th>
                  </tr>
                </thead>
                <tbody>
                  {streamRows.map((entry) => {
                    const hasStdout = Boolean(entry.stdout);
                    const hasStderr = Boolean(entry.stderr);
                    const preview = entry.stderr || entry.stdout || String(entry.result ?? "");
                    const isExpanded = expandedId === entry.id;
                    return (
                      <tr
                        key={entry.id}
                        className="cursor-pointer hover"
                        onClick={() => setExpandedId(isExpanded ? null : entry.id)}
                      >
                        <td className="w-20 font-mono whitespace-nowrap text-[11px]">{entry.timestamp.split("T")[1] ?? entry.timestamp}</td>
                        <td className="w-24 font-mono text-[11px] truncate">{entry.tool}</td>
                        <td className="w-16">
                          <span className={`badge badge-xs ${entry.status === "error" ? "badge-error" : entry.status === "running" ? "badge-warning" : "badge-success"}`}>
                            {entry.status}
                          </span>
                        </td>
                        <td className="w-auto min-w-[22rem] max-w-0">
                          <div className="flex flex-wrap items-center gap-1">
                            {hasStdout && <span className="badge badge-ghost badge-xs">stdout</span>}
                            {hasStderr && <span className="badge badge-error badge-xs">stderr</span>}
                          </div>
                          <div className="mt-1 truncate text-xs text-base-content/70">
                            {preview || "—"}
                          </div>
                          {isExpanded && (
                            <div className="mt-2 space-y-2">
                              {entry.stdout && (
                                <div>
                                  <div className="mb-1 text-[11px] font-semibold text-base-content/60">stdout</div>
                                  <pre className="whitespace-pre-wrap break-words rounded bg-base-300 p-2 text-xs">{entry.stdout}</pre>
                                </div>
                              )}
                              {entry.stderr && (
                                <div>
                                  <div className="mb-1 text-[11px] font-semibold text-base-content/60">stderr</div>
                                  <pre className="whitespace-pre-wrap break-words rounded bg-base-300 p-2 text-xs">{entry.stderr}</pre>
                                </div>
                              )}
                              {!entry.stdout && !entry.stderr && entry.result !== undefined && (
                                <div>
                                  <div className="mb-1 text-[11px] font-semibold text-base-content/60">result</div>
                                  <pre className="whitespace-pre-wrap break-words rounded bg-base-300 p-2 text-xs">
                                    {typeof entry.result === "string" ? entry.result : JSON.stringify(entry.result, null, 2)}
                                  </pre>
                                </div>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
