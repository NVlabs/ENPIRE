/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

interface ModeSelectorProps {
  mode: "agent" | "oracle";
  onChange: (mode: "agent" | "oracle") => void;
}

export default function ModeSelector({ mode, onChange }: ModeSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`text-sm font-medium ${mode === "oracle" ? "text-primary" : "text-base-content/50"}`}
      >
        Oracle
      </span>
      <input
        type="checkbox"
        className="toggle toggle-primary"
        checked={mode === "agent"}
        onChange={(e) => onChange(e.target.checked ? "agent" : "oracle")}
      />
      <span
        className={`text-sm font-medium ${mode === "agent" ? "text-primary" : "text-base-content/50"}`}
      >
        Agent
      </span>
    </div>
  );
}
