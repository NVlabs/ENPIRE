/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useLayoutEffect, useRef } from "react";
import type { StdoutEntry } from "../hooks/useRobotState";

interface StdoutConsoleProps {
  entries: StdoutEntry[];
}

export default function StdoutConsole({ entries }: StdoutConsoleProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "nearest" });
  }, [entries]);

  return (
    <div className="flex h-full min-h-0 flex-col border-t border-base-300">
      <div className="px-2 py-1 text-xs font-semibold shrink-0">Stdout</div>
      {entries.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-base-content/30 text-sm">
          No stdout yet
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-y-auto bg-base-200/40 px-2 py-1 font-mono text-xs">
          {entries.map((entry) => (
            <div key={`${entry.id}-${entry.timestamp}`} className="mb-1 break-words whitespace-pre-wrap">
              <span className="text-base-content/40">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
              {entry.tool && <span className="ml-2 text-info">{entry.tool}</span>}
              <div className="text-base-content/80">{entry.text}</div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
