/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import type { Detection } from "../hooks/useRobotState";

export interface DetectionDebugData {
  image: string; // base64 JPEG
  camera: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
}

interface DetectionDebugProps {
  data: DetectionDebugData | null;
}

// 12 edges of a box connecting 8 corners (front face, back face, connecting edges)
const BOX_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0], // bottom face
  [4, 5], [5, 6], [6, 7], [7, 4], // top face
  [0, 4], [1, 5], [2, 6], [3, 7], // vertical edges
];

export default function DetectionDebug({ data }: DetectionDebugProps) {
  if (!data) {
    return (
      <div className="card card-compact bg-base-200">
        <div className="card-body p-2">
          <h3 className="card-title text-sm">Detection Debug</h3>
          <div className="flex h-40 items-center justify-center text-base-content/30 text-sm">
            Run detect_object() to see results
          </div>
        </div>
      </div>
    );
  }

  const { image, camera, detections, imageWidth, imageHeight } = data;

  return (
    <div className="card card-compact bg-base-200">
      <div className="card-body p-2">
        <h3 className="card-title text-sm">
          Detection Debug
          <span className="badge badge-sm badge-outline">{camera}</span>
          <span className="badge badge-sm badge-info">
            {detections.length} detection{detections.length !== 1 ? "s" : ""}
          </span>
        </h3>
        <div className="relative w-full overflow-hidden rounded bg-base-300">
          <img
            src={`data:image/jpeg;base64,${image}`}
            alt="Detection debug"
            className="w-full object-contain"
          />
          {/* Bounding box overlay */}
          <svg
            className="absolute inset-0 h-full w-full"
            viewBox={`0 0 ${imageWidth} ${imageHeight}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {detections.map((det, i) => {
              const hasBbox = det.bbox && det.bbox.length === 4 && (det.bbox[2] - det.bbox[0]) > 0;
              const has3dBox = det.bbox_3d_projected && det.bbox_3d_projected.length === 8;

              if (hasBbox) {
                // 2D bounding box overlay
                const [x1, y1, x2, y2] = det.bbox;
                return (
                  <g key={i}>
                    <rect
                      x={x1}
                      y={y1}
                      width={x2 - x1}
                      height={y2 - y1}
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth="3"
                    />
                    <rect
                      x={x1}
                      y={Math.max(0, y1 - 22)}
                      width={Math.max(120, (x2 - x1))}
                      height="22"
                      fill="#ef4444"
                      opacity="0.85"
                    />
                    <text
                      x={x1 + 4}
                      y={Math.max(0, y1 - 22) + 16}
                      fill="white"
                      fontSize="14"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {det.label} {(det.confidence * 100).toFixed(0)}%
                    </text>
                    {det.position_3d && (
                      <text
                        x={x1 + 4}
                        y={y2 + 16}
                        fill="#ef4444"
                        fontSize="12"
                        fontFamily="monospace"
                      >
                        [{det.position_3d.map((v) => v.toFixed(3)).join(", ")}]
                      </text>
                    )}
                  </g>
                );
              }

              if (has3dBox) {
                // BundleSDF: oriented 3D bounding box wireframe
                const c = det.bbox_3d_projected!;
                // Centroid for label placement
                const cx = c.reduce((s, p) => s + p[0], 0) / 8;
                const minY = Math.min(...c.map((p) => p[1]));
                return (
                  <g key={i}>
                    {/* Wireframe edges */}
                    {BOX_EDGES.map(([a, b], ei) => {
                      const p1 = c[a];
                      const p2 = c[b];
                      if (!p1 || !p2) return null;
                      return (
                        <line
                          key={ei}
                          x1={p1[0]}
                          y1={p1[1]}
                          x2={p2[0]}
                          y2={p2[1]}
                          stroke="#22d3ee"
                          strokeWidth="2"
                        />
                      );
                    })}
                    {/* Corner dots */}
                    {c.map((p, ci) => (
                      <circle key={ci} cx={p[0]} cy={p[1]} r="3" fill="#22d3ee" />
                    ))}
                    {/* Label above box */}
                    <rect
                      x={cx - 80}
                      y={minY - 40}
                      width="160"
                      height="22"
                      fill="#22d3ee"
                      opacity="0.85"
                      rx="3"
                    />
                    <text
                      x={cx - 76}
                      y={minY - 24}
                      fill="black"
                      fontSize="14"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {det.label} {(det.confidence * 100).toFixed(0)}%
                    </text>
                    {/* Position + quaternion below box */}
                    {det.position_3d && (
                      <text
                        x={cx - 76}
                        y={minY - 6}
                        fill="#22d3ee"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        pos [{det.position_3d.map((v) => v.toFixed(3)).join(", ")}]
                      </text>
                    )}
                  </g>
                );
              }

              return null;
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}
