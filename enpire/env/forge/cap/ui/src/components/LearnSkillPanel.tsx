/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useEffect, useState } from "react";
import type { LearnSkillStatus } from "../hooks/useRobotState";

interface Props {
  status: LearnSkillStatus;
}

export default function LearnSkillPanel({ status }: Props) {
  const active = status.active;
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!active) {
      setExpanded(false);
    }
  }, [active]);

  const pct =
    active && status.step != null && status.max_steps
      ? Math.min(100, (status.step / status.max_steps) * 100)
      : 0;

  const remaining =
    active && status.step != null && status.max_steps
      ? status.max_steps - status.step
      : null;

  const rewardText = status.reward != null ? status.reward.toFixed(2) : "--";
  const cumulativeRewardText =
    status.cumulative_reward != null
      ? status.cumulative_reward.toFixed(1)
      : "--";
  const sourceText = status.action_source?.toUpperCase() ?? "--";
  const sourceBadgeClass =
    status.action_source === "human"
      ? "badge-warning"
      : status.action_source
        ? "badge-info"
        : "badge-ghost";

  return (
    <div className="card bg-base-200 shadow-sm h-full">
      <div className="card-body p-0">
        <button
          type="button"
          className={`flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left ${
            active ? "cursor-pointer" : "cursor-default"
          }`}
          onClick={() => {
            if (active) {
              setExpanded((value) => !value);
            }
          }}
          aria-expanded={active ? expanded : undefined}
          title={active ? "Toggle learn skill details" : undefined}
        >
          <div className="flex min-w-0 flex-1 items-center gap-1.5 whitespace-nowrap text-[10px] sm:text-[11px]">
            <h3 className="shrink-0 text-xs font-semibold uppercase tracking-wide opacity-70">
              Learn Skill
            </h3>
            {active && (
              <div className="badge badge-primary badge-xs shrink-0">
                EP {status.episode ?? "--"}
              </div>
            )}
            <div
              className={`badge badge-xs shrink-0 ${
                active ? "badge-success" : "badge-ghost"
              }`}
            >
              {active ? "Active" : "Offline"}
            </div>
            {active ? (
              <>
                <span className="shrink-0 font-mono opacity-70">
                  {status.step ?? 0}/{status.max_steps ?? "?"}
                </span>
              </>
            ) : (
              <span className="truncate opacity-50">No active learn_skill session</span>
            )}
          </div>

          {active && (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className={`h-3.5 w-3.5 shrink-0 opacity-60 transition-transform duration-200 ${
                expanded ? "rotate-180" : ""
              }`}
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.168l3.71-3.938a.75.75 0 1 1 1.08 1.04l-4.25 4.512a.75.75 0 0 1-1.08 0L5.21 8.27a.75.75 0 0 1 .02-1.06Z"
                clipRule="evenodd"
              />
            </svg>
          )}
        </button>

        {active && expanded && (
          <div className="border-t border-base-300 px-3 pb-3 pt-2">
            <div className="space-y-3">
              <div className="w-full">
                <div className="mb-1 flex justify-between text-xs opacity-70">
                  <span>
                    Step {status.step ?? 0} / {status.max_steps ?? "?"}
                  </span>
                  {remaining != null && <span>{remaining} left</span>}
                </div>
                <progress
                  className="progress progress-primary w-full"
                  value={pct}
                  max={100}
                />
              </div>

              <div className="flex gap-4 text-sm">
                <div className="flex flex-col items-center">
                  <span className="text-xs opacity-50">Reward</span>
                  <span
                    className={`font-mono font-bold ${
                      (status.reward ?? 0) > 0
                        ? "text-success"
                        : "text-base-content"
                    }`}
                  >
                    {rewardText}
                  </span>
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-xs opacity-50">Cum. Reward</span>
                  <span
                    className={`font-mono font-bold ${
                      (status.cumulative_reward ?? 0) > 0
                        ? "text-success"
                        : "text-base-content"
                    }`}
                  >
                    {cumulativeRewardText}
                  </span>
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-xs opacity-50">Source</span>
                  <span className={`badge badge-sm ${sourceBadgeClass}`}>
                    {sourceText}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
