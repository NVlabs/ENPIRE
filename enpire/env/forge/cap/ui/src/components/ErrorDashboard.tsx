/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import type { RuntimeErrorState } from "../hooks/useRobotState";

interface Props {
  error: RuntimeErrorState | null;
  open: boolean;
  onToggle: () => void;
}

export default function ErrorDashboard({ error, open, onToggle }: Props) {
  const hasError = Boolean(error);
  return (
    <div>
      <button className="btn btn-ghost btn-xs w-full justify-between" onClick={onToggle}>
        <span className="text-xs font-semibold">Error Dashboard</span>
        <div className="flex items-center gap-2">
          <span className={`badge badge-xs ${hasError ? "badge-error" : "badge-ghost"}`}>
            {hasError ? "ERROR" : "CLEAR"}
          </span>
          <span className="text-xs">{open ? "▲" : "▼"}</span>
        </div>
      </button>
      {open && (
        <div className="mt-1 card card-compact bg-base-200 border border-base-300">
          <div className="card-body p-3 gap-2">
            {error ? (
              <>
                <div className="flex items-center justify-between gap-2">
                  <div className="text-sm font-semibold text-error break-all">{error.message}</div>
                  {error.source && <div className="badge badge-outline badge-sm">{error.source}</div>}
                </div>
                {error.detail && error.detail !== error.message && (
                  <pre className="text-xs bg-base-300 rounded p-2 whitespace-pre-wrap break-words overflow-x-auto">{error.detail}</pre>
                )}
              </>
            ) : (
              <div className="text-xs opacity-50">No current runtime errors</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
