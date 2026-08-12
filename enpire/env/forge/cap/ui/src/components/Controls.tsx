/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

interface ControlsProps {
  agentStatus: string;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onEstop: () => void;
  onHome: () => void;
}

export default function Controls({
  agentStatus,
  onStart,
  onPause,
  onStop,
  onEstop,
  onHome,
}: ControlsProps) {
  const isRunning = agentStatus === "executing" || agentStatus === "generating";
  const isPaused = agentStatus === "paused";

  return (
    <div className="flex items-center gap-2">
      {isPaused ? (
        <button className="btn btn-success btn-sm" onClick={onStart}>
          Resume
        </button>
      ) : (
        <button
          className="btn btn-success btn-sm"
          onClick={onStart}
          disabled={isRunning}
        >
          Start
        </button>
      )}
      <button
        className="btn btn-warning btn-sm"
        onClick={onPause}
        disabled={!isRunning}
      >
        Pause
      </button>
      <button
        className="btn btn-neutral btn-sm"
        onClick={onStop}
        disabled={!isRunning && !isPaused}
      >
        Stop
      </button>
      <button className="btn btn-info btn-sm" onClick={onHome}>
        Home
      </button>
      <button
        className="btn btn-error btn-sm font-bold shadow-lg shadow-error/30"
        onClick={onEstop}
      >
        E-STOP
      </button>
    </div>
  );
}
