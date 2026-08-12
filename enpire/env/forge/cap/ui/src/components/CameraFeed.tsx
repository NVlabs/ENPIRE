/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useEffect, useState } from "react";
import { listCameras, saveCameras } from "../api/client";
import type { CameraFrame } from "../hooks/useRobotState";

interface CameraFeedProps {
  cameras: Record<string, CameraFrame>;
  streaming: Record<string, boolean>;
  onToggleStreaming: (name: string) => void;
}

function CameraCard({
  name,
  frame,
  streaming,
  onToggle,
}: {
  name: string;
  frame?: CameraFrame;
  streaming: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="card card-compact bg-base-200">
      <div className="card-body p-2">
        <div className="flex items-center justify-between">
          <h3 className="card-title text-xs uppercase tracking-wider">
            {name}
          </h3>
          <input
            type="checkbox"
            className="toggle toggle-xs toggle-primary"
            checked={streaming}
            onChange={onToggle}
          />
        </div>
        <div className="relative aspect-video w-full overflow-hidden rounded bg-base-300">
          {streaming && frame ? (
            <>
              <img
                src={`data:image/jpeg;base64,${frame.image}`}
                alt={`${name} camera`}
                className="h-full w-full object-contain"
              />
              {/* Detection overlay */}
              <svg
                className="absolute inset-0 h-full w-full"
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
              >
                {frame.detections?.map((det, i) => {
                  const [x1, y1, x2, y2] = det.bbox;
                  return (
                    <g key={i}>
                      <rect
                        x={x1}
                        y={y1}
                        width={x2 - x1}
                        height={y2 - y1}
                        fill="none"
                        stroke="#22d3ee"
                        strokeWidth="0.003"
                      />
                      <text
                        x={x1}
                        y={y1 - 0.005}
                        fill="#22d3ee"
                        fontSize="0.025"
                        fontFamily="monospace"
                      >
                        {det.label} ({(det.confidence * 100).toFixed(0)}%)
                      </text>
                    </g>
                  );
                })}
              </svg>
            </>
          ) : (
            <div className="flex h-full items-center justify-center text-base-content/30 text-sm">
              {streaming ? "Waiting..." : "Off"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function CameraFeed({ cameras, streaming, onToggleStreaming }: CameraFeedProps) {
  const [saving, setSaving] = useState(false);
  const [cameraNames, setCameraNames] = useState<string[]>([]);

  // Fetch available camera names from the server on mount and auto-enable streaming
  useEffect(() => {
    listCameras().then((res) => {
      if (res.ok && res.cameras.length > 0) {
        setCameraNames(res.cameras);
        // Auto-enable streaming for all cameras
        for (const name of res.cameras) {
          if (!streaming[name]) {
            onToggleStreaming(name);
          }
        }
      }
    }).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleKacha = async () => {
    setSaving(true);
    try {
      await saveCameras();
    } finally {
      setSaving(false);
    }
  };

  if (cameraNames.length === 0) {
    return <div className="text-xs text-base-content/50 p-2">Loading cameras...</div>;
  }

  return (
    <div className="space-y-1">
      <div className="flex justify-end">
        <button
          className={`btn btn-xs btn-outline ${saving ? "loading" : ""}`}
          onClick={handleKacha}
          disabled={saving}
        >
          Kacha
        </button>
      </div>
      <div className={`grid gap-2 ${cameraNames.length >= 3 ? "grid-cols-3" : cameraNames.length === 2 ? "grid-cols-2" : "grid-cols-1"}`}>
        {cameraNames.map((name) => (
          <CameraCard
            key={name}
            name={name}
            frame={cameras[name]}
            streaming={!!streaming[name]}
            onToggle={() => onToggleStreaming(name)}
          />
        ))}
      </div>
    </div>
  );
}
