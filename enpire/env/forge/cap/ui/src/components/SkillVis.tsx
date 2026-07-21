import { useEffect, useState } from "react";
import {
  getArm,
} from "../hooks/useRobotState";
import type {
  Detection,
  VlmResult,
  SegmentationDebugData,
  GraspVizPayload,
  GraspDebugRow,
  MotionPlannerDebug,
  ContactDetectDebug,
  RobotState,
} from "../hooks/useRobotState";

export interface DetectionDebugData {
  image: string; // base64 JPEG
  camera: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
}

interface SkillVisProps {
  detectionData: DetectionDebugData | null;
  segmentationData: SegmentationDebugData | null;
  vlmResults: VlmResult[];
  graspViz?: GraspVizPayload | null;
  motionPlannerDebug?: MotionPlannerDebug | null;
  contactDetectDebug?: ContactDetectDebug | null;
  robotState: RobotState;
  extraHeaderButton?: React.ReactNode;
}

// 12 edges of a box connecting 8 corners
const BOX_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

function DetectionTab({ data }: { data: DetectionDebugData | null }) {
  if (!data) {
    return (
      <div className="flex h-40 items-center justify-center text-base-content/30 text-sm">
        Run detect_object() to see results
      </div>
    );
  }

  const { image, camera, detections, imageWidth, imageHeight } = data;
  const AXIS_COLORS = { x: "#ef4444", y: "#22c55e", z: "#3b82f6" };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header badges */}
      <div className="flex items-center gap-1 mb-1 shrink-0">
        <span className="badge badge-sm badge-outline">{camera}</span>
        <span className="badge badge-sm badge-info">
          {detections.length} detection{detections.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Image + SVG overlay */}
      <div className="relative flex-1 min-h-0 overflow-hidden rounded bg-base-300 flex items-center justify-center">
        <img
          src={`data:image/jpeg;base64,${image}`}
          alt="Detection debug"
          className="max-w-full max-h-full object-contain"
        />
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <marker id="det-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6 Z" fill="context-stroke" />
            </marker>
          </defs>
          {detections.map((det, i) => {
            const hasBbox = det.bbox && det.bbox.length === 4 && (det.bbox[2] - det.bbox[0]) > 0;
            const has3dBox = det.bbox_3d_projected && det.bbox_3d_projected.length === 8;
            const hasPoseAxes = det.pose_axes;
            const hasGraspAxes = det.is_grasp && det.grasp_axes;

            // Grasp axes (AnyGrasp) — keep existing rendering
            if (hasGraspAxes) {
              const ax = det.grasp_axes!;
              const [ox, oy] = ax.origin;
              return (
                <g key={i}>
                  {(["x", "y", "z"] as const).map((a) => (
                    <line key={a} x1={ox} y1={oy} x2={ax[a][0]} y2={ax[a][1]}
                      stroke={AXIS_COLORS[a]} strokeWidth="3" markerEnd="url(#det-arrow)" />
                  ))}
                  <circle cx={ox} cy={oy} r="5" fill="white" stroke="#000" strokeWidth="1.5" />
                  <text x={ox + 8} y={oy - 8} fill="white" fontSize="12" fontFamily="monospace"
                    stroke="black" strokeWidth="3" paintOrder="stroke">
                    #{i + 1} {(det.confidence * 100).toFixed(0)}%
                  </text>
                </g>
              );
            }

            // BundleSDF-style: pose axes + 3D bbox + label
            if (hasPoseAxes || has3dBox) {
              const ax = det.pose_axes;
              const c = det.bbox_3d_projected;

              // Determine label anchor point
              let labelX = 10;
              let labelY = 24 + i * 68;
              if (ax) {
                labelX = ax.origin[0] + 12;
                labelY = ax.origin[1] - 32;
              } else if (c) {
                labelX = c.reduce((s, p) => s + p[0], 0) / 8 - 80;
                labelY = Math.min(...c.map((p) => p[1])) - 48;
              }

              return (
                <g key={i}>
                  {/* 3D wireframe box */}
                  {c && BOX_EDGES.map(([a, b], ei) => {
                    const pa = c[a];
                    const pb = c[b];
                    if (!pa || !pb) return null;
                    return (
                      <line key={`e${ei}`} x1={pa[0]} y1={pa[1]} x2={pb[0]} y2={pb[1]}
                        stroke="#22d3ee" strokeWidth="1.5" opacity="0.6" />
                    );
                  })}

                  {/* Pose coordinate axes (BundleSDF style: X=red, Y=green, Z=blue) */}
                  {ax && (["x", "y", "z"] as const).map((a) => (
                    <line key={a}
                      x1={ax.origin[0]} y1={ax.origin[1]}
                      x2={ax[a][0]} y2={ax[a][1]}
                      stroke={AXIS_COLORS[a]} strokeWidth="3"
                      markerEnd="url(#det-arrow)" />
                  ))}
                  {ax && (
                    <circle cx={ax.origin[0]} cy={ax.origin[1]} r="4"
                      fill="white" stroke="#000" strokeWidth="1" />
                  )}

                  {/* Label panel */}
                  <rect x={labelX - 4} y={labelY - 2} width="200" height={det.position_3d ? 44 : 22}
                    fill="black" opacity="0.75" rx="3" />
                  <text x={labelX} y={labelY + 13} fill="#22d3ee" fontSize="13"
                    fontFamily="monospace" fontWeight="bold">
                    {det.label} {(det.confidence * 100).toFixed(0)}%
                  </text>
                  {det.position_3d && (
                    <text x={labelX} y={labelY + 30} fill="#a3e635" fontSize="11" fontFamily="monospace">
                      [{det.position_3d.map((v) => v.toFixed(3)).join(", ")}] m
                    </text>
                  )}
                </g>
              );
            }

            // 2D bbox only (legacy / non-BundleSDF)
            if (hasBbox) {
              const [x1, y1, x2, y2] = det.bbox;
              return (
                <g key={i}>
                  <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1}
                    fill="none" stroke="#22c55e" strokeWidth="2.5" />
                  <rect x={x1} y={Math.max(0, y1 - 22)} width={Math.max(120, (x2 - x1))} height="22"
                    fill="black" opacity="0.75" rx="2" />
                  <text x={x1 + 4} y={Math.max(0, y1 - 22) + 15} fill="#22c55e" fontSize="13"
                    fontFamily="monospace" fontWeight="bold">
                    {det.label} {(det.confidence * 100).toFixed(0)}%
                  </text>
                  {det.position_3d && (
                    <text x={x1 + 4} y={y2 + 14} fill="#a3e635" fontSize="11" fontFamily="monospace"
                      stroke="black" strokeWidth="2.5" paintOrder="stroke">
                      [{det.position_3d.map((v) => v.toFixed(3)).join(", ")}] m
                    </text>
                  )}
                </g>
              );
            }

            return null;
          })}
        </svg>
      </div>

      {/* Detection details table */}
      {detections.length > 0 && (
        <div className="mt-1 shrink-0 overflow-x-auto">
          <table className="table table-xs table-zebra w-full">
            <thead>
              <tr>
                <th>#</th>
                <th>Label</th>
                <th>Score</th>
                <th>Position (m)</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((det, i) => (
                <tr key={i}>
                  <td className="font-mono">{i + 1}</td>
                  <td>{det.label}</td>
                  <td className="font-mono">{(det.confidence * 100).toFixed(1)}%</td>
                  <td className="font-mono text-xs">
                    {det.position_3d
                      ? `[${det.position_3d.map((v) => v.toFixed(3)).join(", ")}]`
                      : "---"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function SegmentationTab({ data }: { data: SegmentationDebugData | null }) {
  if (!data) {
    return (
      <div className="flex h-40 items-center justify-center text-base-content/30 text-sm">
        Run segment_object() to see results
      </div>
    );
  }

  const { image, camera, imageWidth, imageHeight, mask_overlay, query, mask_area, bbox_xywh, hu_dist, passed, is_reference } = data;
  const [bx, by, bw, bh] = bbox_xywh;

  const statusBadge = is_reference
    ? <span className="badge badge-sm" style={{ background: "#0096ff", color: "#fff" }}>REF</span>
    : passed === true
    ? <span className="badge badge-sm badge-success">PASS</span>
    : passed === false
    ? <span className="badge badge-sm badge-error">FAIL</span>
    : null;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1 mb-1 shrink-0 flex-wrap">
        <span className="badge badge-sm badge-outline">{camera}</span>
        <span className="badge badge-sm badge-info">{query}</span>
        {statusBadge}
        {hu_dist != null && <span className="badge badge-sm badge-ghost">dist {hu_dist.toFixed(3)}</span>}
        <span className="text-xs text-base-content/50">{mask_area.toLocaleString()}px</span>
      </div>
      <div className="relative flex-1 min-h-0 overflow-hidden rounded bg-base-300 flex items-center justify-center">
        <img
          src={`data:image/jpeg;base64,${image}`}
          alt="Segmentation base"
          className="max-w-full max-h-full object-contain"
        />
        <img
          src={`data:image/png;base64,${mask_overlay}`}
          alt="Mask overlay"
          className="absolute inset-0 max-w-full max-h-full object-contain pointer-events-none"
          style={{ margin: "auto" }}
        />
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <rect x={bx} y={by} width={bw} height={bh} fill="none" stroke="#00dcdc" strokeWidth="2" strokeDasharray="6 3" />
        </svg>
      </div>
    </div>
  );
}

function SegTab({ data }: { data: SegmentationDebugData | null }) {
  if (!data?.media) {
    return (
      <div className="flex h-40 items-center justify-center text-base-content/30 text-sm">
        Run mark_segmentation(media=...) to see results
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1 mb-1 shrink-0">
        <span className="badge badge-sm badge-info">{data.query}</span>
      </div>
      <div className="relative flex-1 min-h-0 overflow-hidden rounded bg-base-300 flex items-center justify-center">
        <img
          src={data.media.startsWith("data:") ? data.media : `data:image/jpeg;base64,${data.media}`}
          alt="Segmentation media"
          className="max-w-full max-h-full object-contain"
        />
      </div>
    </div>
  );
}

function graspBackendLabel(data: GraspVizPayload | null | undefined): string {
  if (!data?.backend) return "AnyGrasp";
  if (data.backend === "3dbbgrasp") return "3D-BB";
  if (data.backend === "2dgrasp") return "2D Grasp";
  return "AnyGrasp";
}

function fmtList(values?: number[], digits = 4): string {
  if (!values || values.length === 0) return "—";
  return `[${values.map((v) => Number(v).toFixed(digits)).join(", ")}]`;
}

function fmtMetric(value: number | undefined, digits = 4, suffix = ""): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

function graspStatusClass(status?: string): string {
  if (!status) return "badge-ghost";
  if (status === "selected" || status === "executed") return "badge-success";
  if (status.includes("failed")) return "badge-error";
  if (status.includes("discarded")) return "badge-warning";
  if (status.includes("preview")) return "badge-info";
  return "badge-ghost";
}

function graspStatusLabel(status?: string): string {
  if (!status) return "unknown";
  return status.replace(/_/g, " ");
}

function getGraspAxes(
  row: GraspDebugRow,
  detectionData: DetectionDebugData | null,
): Detection["grasp_axes"] | null {
  if (!detectionData?.detections?.length) return null;
  const det = detectionData.detections[row.rank - 1];
  return det?.grasp_axes ?? null;
}

function getGraspThumbViewBox(
  row: GraspDebugRow,
  detectionData: DetectionDebugData | null,
): { x: number; y: number; w: number; h: number } | null {
  const axes = getGraspAxes(row, detectionData);
  if (!axes || !detectionData) return null;
  const pts = [axes.origin, axes.x, axes.y, axes.z];
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const margin = 28;
  const minX = Math.max(0, Math.min(...xs) - margin);
  const minY = Math.max(0, Math.min(...ys) - margin);
  const maxX = Math.min(detectionData.imageWidth, Math.max(...xs) + margin);
  const maxY = Math.min(detectionData.imageHeight, Math.max(...ys) + margin);
  return {
    x: minX,
    y: minY,
    w: Math.max(40, maxX - minX),
    h: Math.max(40, maxY - minY),
  };
}

function GraspThumbnail({
  row,
  detectionData,
}: {
  row: GraspDebugRow;
  detectionData: DetectionDebugData | null;
}) {
  if (row.thumbnail_b64) {
    return (
      <img
        src={`data:image/jpeg;base64,${row.thumbnail_b64}`}
        alt={`Grasp ${row.rank} thumbnail`}
        className="h-24 w-24 rounded bg-base-300 object-cover"
      />
    );
  }
  const viewBox = getGraspThumbViewBox(row, detectionData);
  const axes = getGraspAxes(row, detectionData);
  if (!detectionData || !viewBox || !axes) {
    return (
      <div className="flex h-24 w-24 items-center justify-center rounded bg-base-300 text-[10px] text-base-content/40">
        no thumb
      </div>
    );
  }
  return (
    <svg
      viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
      className="h-24 w-24 rounded bg-base-300"
      preserveAspectRatio="xMidYMid meet"
    >
      <image
        href={`data:image/jpeg;base64,${detectionData.image}`}
        x={0}
        y={0}
        width={detectionData.imageWidth}
        height={detectionData.imageHeight}
      />
      <line x1={axes.origin[0]} y1={axes.origin[1]} x2={axes.x[0]} y2={axes.x[1]} stroke="#ef4444" strokeWidth="3" />
      <line x1={axes.origin[0]} y1={axes.origin[1]} x2={axes.y[0]} y2={axes.y[1]} stroke="#22c55e" strokeWidth="3" />
      <line x1={axes.origin[0]} y1={axes.origin[1]} x2={axes.z[0]} y2={axes.z[1]} stroke="#3b82f6" strokeWidth="3" />
      <circle cx={axes.origin[0]} cy={axes.origin[1]} r="4" fill="white" stroke="black" strokeWidth="1.5" />
    </svg>
  );
}

function VlmTab({ results }: { results: VlmResult[] }) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  if (results.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-base-content/30 text-sm">
        Run vlm_query() to see results
      </div>
    );
  }

  const selected = selectedIdx !== null ? results[selectedIdx] : results[results.length - 1];
  const activeIdx = selectedIdx ?? results.length - 1;
  if (!selected) {
    return null;
  }

  return (
    <div className="flex flex-col gap-1 h-full min-h-0">
      {/* History selector */}
      {results.length > 1 && (
        <div className="flex gap-1 overflow-x-auto shrink-0">
          {results.map((r, i) => (
            <button
              key={i}
              className={`btn btn-xs shrink-0 ${i === activeIdx ? "btn-primary" : "btn-ghost"}`}
              onClick={() => setSelectedIdx(i)}
              title={r.prompt.slice(0, 60)}
            >
              #{i + 1} {r.backend}
            </button>
          ))}
        </div>
      )}

      {/* Selected result */}
      <div className="flex-1 min-h-0 overflow-y-auto space-y-2">
        <div className="flex items-center gap-1 flex-wrap">
          <span className="badge badge-sm badge-outline">{selected.backend}</span>
          <span className="badge badge-sm badge-outline">{selected.camera}</span>
          {selected.model && <span className="badge badge-sm badge-ghost">{selected.model}</span>}
          <span className="text-xs text-base-content/40">
            {new Date(selected.timestamp).toLocaleTimeString()}
          </span>
        </div>

        <div>
          <div className="text-xs font-semibold text-base-content/60 mb-0.5">Prompt</div>
          <pre className="text-xs bg-base-300 rounded p-2 whitespace-pre-wrap break-words">
            {selected.prompt}
          </pre>
        </div>

        <div>
          <div className="text-xs font-semibold text-base-content/60 mb-0.5">Response</div>
          <pre className="text-xs bg-base-300 rounded p-2 whitespace-pre-wrap break-words">
            {selected.response}
          </pre>
        </div>
      </div>
    </div>
  );
}

function GraspTab({
  data,
  detectionData,
}: {
  data: GraspVizPayload | null | undefined;
  detectionData: DetectionDebugData | null;
}) {
  const [mode, setMode] = useState<"overlay" | "native">("overlay");
  const backendLabel = graspBackendLabel(data);

  useEffect(() => {
    if (data?.overlay_b64) {
      setMode("overlay");
    } else if (data?.native_viz_b64) {
      setMode("native");
    }
  }, [data?.overlay_b64, data?.native_viz_b64]);

  if (!data) {
    return <div className="flex items-center justify-center h-full text-base-content/40 text-sm">No {backendLabel} viz yet</div>;
  }

  const hasOverlay = Boolean(data.overlay_b64);
  const hasNative = Boolean(data.native_viz_b64);
  const showOverlay = mode === "overlay" && hasOverlay;
  const imgBase64 = showOverlay ? data.overlay_b64 : data.native_viz_b64;
  const mimeType = showOverlay ? "image/jpeg" : "image/png";
  const matchingDetection =
    data.camera &&
    detectionData &&
    detectionData.camera === data.camera &&
    detectionData.detections.length > 0
      ? (
          detectionData.detections.find(
            (det) =>
              data.object_name &&
              det.label.trim().toLowerCase() === data.object_name.trim().toLowerCase(),
          ) ?? detectionData.detections[0]
        )
      : null;
  const correctedMarker =
    matchingDetection && detectionData
      ? matchingDetection.bbox_3d_projected && matchingDetection.bbox_3d_projected.length > 0
        ? {
            x:
              matchingDetection.bbox_3d_projected.reduce((sum, pt) => sum + pt[0], 0) /
              matchingDetection.bbox_3d_projected.length,
            y:
              matchingDetection.bbox_3d_projected.reduce((sum, pt) => sum + pt[1], 0) /
              matchingDetection.bbox_3d_projected.length,
          }
        : matchingDetection.bbox && matchingDetection.bbox.length === 4
          ? {
              x: (matchingDetection.bbox[0] + matchingDetection.bbox[2]) / 2,
              y: (matchingDetection.bbox[1] + matchingDetection.bbox[3]) / 2,
            }
          : null
      : null;

  if (!imgBase64) {
    return <div className="flex items-center justify-center h-full text-base-content/40 text-sm">No {backendLabel} viz yet</div>;
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto">
      <div className="flex items-center justify-between gap-2 mb-2 shrink-0">
        <div className="flex items-center gap-2 text-xs text-base-content/60">
          <span className="badge badge-ghost badge-xs">{backendLabel}</span>
          {data.camera && <span className="badge badge-ghost badge-xs">{data.camera}</span>}
          {data.object_name && <span className="truncate max-w-48">{data.object_name}</span>}
          {typeof data.n_grasps === "number" && (
            <span className="badge badge-ghost badge-xs">{data.n_grasps} grasps</span>
          )}
          {data.status && <span className="badge badge-ghost badge-xs">{data.status}</span>}
          {data.status_reason && <span className="text-warning truncate max-w-72">{data.status_reason}</span>}
          {data.selected_side && <span className="badge badge-ghost badge-xs">side {data.selected_side}</span>}
          {typeof data.selected_rank === "number" && (
            <span className="badge badge-success badge-xs">selected #{data.selected_rank}</span>
          )}
        </div>
        {(hasOverlay || hasNative) && (
          <div className="tabs tabs-boxed tabs-xs bg-base-300">
            {hasOverlay && (
              <button
                className={`tab tab-xs ${mode === "overlay" ? "tab-active" : ""}`}
                onClick={() => setMode("overlay")}
              >
                Overlay
              </button>
            )}
            {hasNative && (
              <button
                className={`tab tab-xs ${mode === "native" ? "tab-active" : ""}`}
                onClick={() => setMode("native")}
              >
                Native
              </button>
            )}
          </div>
        )}
      </div>
      {data.object_name && (
        <div className="mb-2 shrink-0">
          <div className="rounded bg-base-300/80 px-3 py-2 text-xs">
            <span className="font-semibold text-base-content/70 mr-2">SAM3 query</span>
            <span className="font-mono break-all">{data.object_name}</span>
          </div>
        </div>
      )}
      <div className="rounded border border-base-300 bg-base-200/40 p-2">
        <div className="mb-2 flex items-center justify-between text-xs text-base-content/60">
          <span>Main debug grasp image</span>
          <span className="badge badge-outline badge-xs">
            rows {data.grasps?.length ?? 0}
          </span>
        </div>
        <div className="flex min-h-[220px] items-center justify-center">
          <div className="relative inline-block max-w-full">
            <img
              src={`data:${mimeType};base64,${imgBase64}`}
              className="max-h-[42vh] w-auto max-w-full object-contain rounded"
              alt={showOverlay ? `${backendLabel} overlay visualization` : `${backendLabel} native visualization`}
            />
            {showOverlay && correctedMarker && detectionData && (
              <svg
                className="absolute inset-0 h-full w-full pointer-events-none"
                viewBox={`0 0 ${detectionData.imageWidth} ${detectionData.imageHeight}`}
                preserveAspectRatio="xMidYMid meet"
              >
                <circle
                  cx={correctedMarker.x}
                  cy={correctedMarker.y}
                  r="8"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="3"
                />
                <line
                  x1={correctedMarker.x - 12}
                  y1={correctedMarker.y}
                  x2={correctedMarker.x + 12}
                  y2={correctedMarker.y}
                  stroke="#ef4444"
                  strokeWidth="3"
                />
                <line
                  x1={correctedMarker.x}
                  y1={correctedMarker.y - 12}
                  x2={correctedMarker.x}
                  y2={correctedMarker.y + 12}
                  stroke="#ef4444"
                  strokeWidth="3"
                />
              </svg>
            )}
            {data.object_name && (
              <div className="absolute left-2 top-2 max-w-[calc(100%-1rem)] rounded bg-black/70 px-2 py-1 text-[11px] text-white shadow">
                <span className="font-semibold mr-1">Object:</span>
                <span className="font-mono break-all">{data.object_name}</span>
              </div>
            )}
            {showOverlay && correctedMarker && (
              <div className="absolute right-2 top-2 rounded bg-red-500/85 px-2 py-1 text-[11px] text-white shadow">
                BundleSDF XYZ
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="rounded border border-base-300 bg-base-200/40 p-2">
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-base-content/60">
          <span className="font-semibold">Returned grasps</span>
          <span className="badge badge-ghost badge-xs">{data.grasps?.length ?? 0}</span>
          {!data.grasps || data.grasps.length === 0 ? (
            <span className="text-warning">No grasp rows are being rendered.</span>
          ) : null}
          {data.planner_thresholds_m && data.planner_thresholds_m.length > 0 && (
            <span className="badge badge-ghost badge-xs">
              thresholds {data.planner_thresholds_m.map((v) => v.toFixed(3)).join(", ")} m
            </span>
          )}
          {typeof data.selection_threshold_m === "number" && (
            <span className="badge badge-ghost badge-xs">
              used {data.selection_threshold_m.toFixed(3)} m
            </span>
          )}
        </div>
        {data.grasps && data.grasps.length > 0 ? (
          <div className="space-y-2">
            {data.grasps.map((row) => (
              <div key={row.rank} className="rounded border border-base-300 bg-base-100/60 p-2">
                <div className="flex gap-3">
                  <GraspThumbnail row={row} detectionData={detectionData} />
                  <div className="min-w-0 flex-1">
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <span className="badge badge-outline badge-sm">#{row.rank}</span>
                      {typeof row.score === "number" && (
                        <span className="badge badge-ghost badge-sm">score {row.score.toFixed(4)}</span>
                      )}
                      {typeof row.width === "number" && (
                        <span className="badge badge-ghost badge-sm">width {row.width.toFixed(4)}</span>
                      )}
                      <span className={`badge badge-sm ${graspStatusClass(row.status)}`}>
                        {graspStatusLabel(row.status)}
                      </span>
                      {row.motionplanner_side && (
                        <span className="badge badge-ghost badge-sm">{row.motionplanner_side}</span>
                      )}
                    </div>

                    <div className="grid grid-cols-1 gap-2 text-[11px] md:grid-cols-2 xl:grid-cols-3">
                      <div className="rounded bg-base-200 p-2 font-mono">
                        <div className="mb-1 text-[10px] font-semibold uppercase text-base-content/50">Raw XYZ</div>
                        <div>{fmtList(row.raw_xyz, 4)}</div>
                      </div>
                      <div className="rounded bg-base-200 p-2 font-mono">
                        <div className="mb-1 text-[10px] font-semibold uppercase text-base-content/50">Raw RPY</div>
                        <div>{fmtList(row.raw_rpy, 1)}</div>
                      </div>
                      <div className="rounded bg-base-200 p-2 font-mono">
                        <div className="mb-1 text-[10px] font-semibold uppercase text-base-content/50">Planner XYZ</div>
                        <div>{fmtList(row.planner_xyz, 4)}</div>
                      </div>
                      <div className="rounded bg-base-200 p-2 font-mono">
                        <div className="mb-1 text-[10px] font-semibold uppercase text-base-content/50">Planner RPY</div>
                        <div>{fmtList(row.planner_rpy, 1)}</div>
                      </div>
                      <div className="rounded bg-base-200 p-2 font-mono">
                        <div className="mb-1 text-[10px] font-semibold uppercase text-base-content/50">Planner Error</div>
                        <div>pos {fmtMetric(row.final_pos_error_m, 4, " m")}</div>
                        <div>rot {fmtMetric(row.final_rot_error_deg, 2, " deg")}</div>
                        <div>pose {fmtMetric(row.final_pose_error, 5)}</div>
                        <div>traj {fmtMetric(row.trajectory_steps, 0)}</div>
                      </div>
                      <div className="rounded bg-base-200 p-2 text-[11px]">
                        <div className="mb-1 text-[10px] font-semibold uppercase text-base-content/50">Status</div>
                        <div className="font-semibold">{graspStatusLabel(row.status)}</div>
                        <div className="mt-1 whitespace-pre-wrap break-words text-base-content/70">
                          {row.status_reason || row.motionplanner_reason || "—"}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex h-24 flex-col items-center justify-center text-sm text-base-content/40 gap-1">
            <div>No grasp rows available to render.</div>
            {data.status_reason && <div className="text-warning text-xs">{data.status_reason}</div>}
          </div>
        )}
      </div>
    </div>
  );
}

function MotionPlannerTab({
  data,
  robotState,
}: {
  data: MotionPlannerDebug | null | undefined;
  robotState: RobotState;
}) {
  if (!data) {
    return <div className="flex items-center justify-center h-full text-base-content/40 text-sm">No motion planner result yet</div>;
  }

  const targets = [
    data.left_target_pos || data.left_target_rpy ? {
      side: "left",
      pos: data.left_target_pos,
      rpy: data.left_target_rpy,
    } : null,
    data.right_target_pos || data.right_target_rpy ? {
      side: "right",
      pos: data.right_target_pos,
      rpy: data.right_target_rpy,
    } : null,
  ].filter(Boolean) as Array<{ side: string; pos?: number[]; rpy?: number[] }>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_220px] gap-3 h-full min-h-0">
      <div className="flex flex-col gap-2 min-h-0 overflow-y-auto">
        <div className="flex flex-wrap items-center gap-1">
          {data.planner_backend && (
            <span className="badge badge-sm badge-primary">{data.planner_backend}</span>
          )}
          {data.side && <span className="badge badge-sm badge-outline">{data.side}</span>}
          {data.status && (
            <span className={`badge badge-sm ${data.status === "Success" ? "badge-success" : "badge-error"}`}>
              {data.status}
            </span>
          )}
          <span className={`badge badge-sm ${data.preview_only ? "badge-warning" : "badge-info"}`}>
            {data.preview_only ? "preview" : "execute"}
          </span>
          {typeof data.planning_speed === "number" && (
            <span className="badge badge-sm badge-ghost">speed {data.planning_speed.toFixed(2)}</span>
          )}
          {typeof data.trajectory_steps === "number" && (
            <span className="badge badge-sm badge-ghost">{data.trajectory_steps} steps</span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="bg-base-300 rounded p-2">
            <div className="text-xs font-semibold text-base-content/60 mb-1">Position Error</div>
            <div className="font-mono text-sm">{(data.final_pos_error_m ?? data.ik_error_m ?? 0).toFixed(4)} m</div>
          </div>
          <div className="bg-base-300 rounded p-2">
            <div className="text-xs font-semibold text-base-content/60 mb-1">Orientation Error</div>
            <div className="font-mono text-sm">{(data.final_rot_error_deg ?? 0).toFixed(2)} deg</div>
          </div>
          <div className="bg-base-300 rounded p-2">
            <div className="text-xs font-semibold text-base-content/60 mb-1">Combined Pose Error</div>
            <div className="font-mono text-sm">{(data.final_pose_error ?? 0).toFixed(5)}</div>
          </div>
          <div className="bg-base-300 rounded p-2">
            <div className="text-xs font-semibold text-base-content/60 mb-1">Executed</div>
            <div className="font-mono text-sm">{String(Boolean(data.executed))}</div>
          </div>
        </div>

        {targets.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-base-content/60 mb-1">Targets</div>
            <div className="space-y-1">
              {targets.map((target) => (
                <div key={target.side} className="bg-base-300 rounded p-2 text-xs font-mono">
                  <div className="mb-1 uppercase text-base-content/60">{target.side}</div>
                  {target.pos && <div>xyz [{target.pos.map((v) => v.toFixed(4)).join(", ")}]</div>}
                  {target.rpy && <div>rpy [{target.rpy.map((v) => v.toFixed(2)).join(", ")}]</div>}
                </div>
              ))}
            </div>
          </div>
        )}

        {(data.reason || data.error) && (
          <div>
            <div className="text-xs font-semibold text-base-content/60 mb-1">Failure Info</div>
            <pre className="text-xs bg-base-300 rounded p-2 whitespace-pre-wrap break-words">
              {data.reason || data.error}
              {data.reason && data.error && data.error !== data.reason ? `\n\nerror: ${data.error}` : ""}
            </pre>
          </div>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <div className="text-xs font-semibold text-base-content/60">Gripper Width</div>
        {Object.entries(robotState.arms).map(([name, arm]) => (
          <div key={name} className="rounded bg-base-300 p-3">
            <div className="text-xs font-semibold text-base-content/60 mb-1">{name}</div>
            <div className="font-mono text-sm">{arm.gripper.toFixed(4)}</div>
          </div>
        ))}
        {data.side && data.side !== "both" && (
          <div className="rounded bg-base-300 p-3">
            <div className="text-xs font-semibold text-base-content/60 mb-1">Active Side</div>
            <div className="font-mono text-sm">
              {getArm(robotState, data.side).gripper.toFixed(4)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ContactDetectTab({
  data,
  robotState,
}: {
  data: ContactDetectDebug | null | undefined;
  robotState: RobotState;
}) {
  const liveWidth = data?.side ? getArm(robotState, data.side).gripper : null;

  if (!data) {
    return (
      <div className="flex flex-col gap-3 h-full justify-center text-sm text-base-content/50">
        <div className="text-center">No contact detect result yet</div>
        <div className="grid grid-cols-2 gap-2 text-xs max-w-sm mx-auto w-full">
          {Object.entries(robotState.arms).map(([name, arm]) => (
            <div key={name} className="rounded bg-base-300 p-3">
              <div className="font-semibold text-base-content/70 mb-1">{name} gripper</div>
              <div className="font-mono">{arm.gripper.toFixed(4)}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 h-full min-h-0 overflow-y-auto">
      <div className="flex flex-wrap items-center gap-2">
        {data.side && <span className="badge badge-outline badge-sm">{data.side}</span>}
        <span className={`badge badge-sm ${data.success ? "badge-success" : "badge-error"}`}>
          {data.success ? "grasp success" : "grasp failed"}
        </span>
        {data.status && <span className="badge badge-ghost badge-sm">{data.status}</span>}
        <span className="text-xs text-base-content/50">
          {new Date(data.timestamp).toLocaleTimeString()}
        </span>
      </div>

      {data.object_name && (
        <div>
          <div className="text-xs font-semibold text-base-content/60 mb-1">Object</div>
          <div className="rounded bg-base-300 p-2 text-sm font-mono break-all">{data.object_name}</div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div className="rounded bg-base-300 p-3">
          <div className="text-xs font-semibold text-base-content/60 mb-1">Measured gripper width</div>
          <div className="font-mono text-sm">
            {typeof data.gripper_width === "number" ? data.gripper_width.toFixed(4) : "—"}
          </div>
        </div>
        <div className="rounded bg-base-300 p-3">
          <div className="text-xs font-semibold text-base-content/60 mb-1">Current gripper width</div>
          <div className="font-mono text-sm">
            {typeof liveWidth === "number" ? liveWidth.toFixed(4) : "—"}
          </div>
        </div>
      </div>

      <div className="rounded bg-base-300 p-3 text-xs">
        <div className="font-semibold text-base-content/60 mb-1">Decision rule</div>
        <div className="font-mono break-words">
          {typeof data.threshold_high === "number"
            ? `fail if width ≤ ${(data.threshold_low ?? 0.02).toFixed(4)} or width ≥ ${data.threshold_high.toFixed(4)}`
            : `fail if width ≤ ${(data.threshold_low ?? 0.02).toFixed(4)}`}
        </div>
      </div>

      {data.message && (
        <div>
          <div className="text-xs font-semibold text-base-content/60 mb-1">Message</div>
          <pre className="text-xs bg-base-300 rounded p-2 whitespace-pre-wrap break-words">{data.message}</pre>
        </div>
      )}
    </div>
  );
}

export default function SkillVis({ detectionData, segmentationData, vlmResults, graspViz, motionPlannerDebug, contactDetectDebug, robotState, extraHeaderButton }: SkillVisProps) {
  const [tab, setTab] = useState<"detection" | "segmentation" | "seg" | "vlm" | "grasp" | "motion" | "contact">("vlm");
  const graspLabel = graspBackendLabel(graspViz);
  const availableTabs: Array<{ key: "detection" | "segmentation" | "seg" | "vlm" | "grasp" | "motion" | "contact"; label: string; badge?: number }> = [
    ...(vlmResults.length > 0 ? [{ key: "vlm" as const, label: "VLM", badge: vlmResults.length }] : []),
    ...(detectionData ? [{ key: "detection" as const, label: "Detection" }] : []),
    ...(graspViz ? [{ key: "grasp" as const, label: graspLabel }] : []),
    ...(motionPlannerDebug ? [{ key: "motion" as const, label: "Motion Planner" }] : []),
    ...(contactDetectDebug ? [{ key: "contact" as const, label: "Gripper Detection" }] : []),
    ...(segmentationData?.mask_overlay ? [{ key: "segmentation" as const, label: "Segment" }] : []),
    ...(segmentationData?.media ? [{ key: "seg" as const, label: "Seg" }] : []),
  ];

  useEffect(() => {
    const firstTab = availableTabs[0];
    if (!firstTab) return;
    if (!availableTabs.some((entry) => entry.key === tab)) {
      setTab(firstTab.key);
    }
  }, [availableTabs, tab]);

  return (
    <div className="card card-compact bg-base-200 h-full flex flex-col">
      <div className="card-body p-2 flex flex-col min-h-0">
        <div className="flex items-center justify-between shrink-0 gap-2">
          <h3 className="card-title text-sm">Skill Vis</h3>
          <div className="flex items-center gap-2">
            {extraHeaderButton}
          <div className="tabs tabs-boxed tabs-xs bg-base-300">
            {availableTabs.map((entry) => (
              <button
                key={entry.key}
                className={`tab tab-xs ${tab === entry.key ? "tab-active" : ""}`}
                onClick={() => setTab(entry.key)}
              >
                {entry.label}
                {typeof entry.badge === "number" && entry.badge > 0 && (
                  <span className="badge badge-xs badge-primary ml-1">{entry.badge}</span>
                )}
              </button>
            ))}
          </div>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          {availableTabs.length === 0 ? (
            <div className="flex h-full items-center justify-center text-base-content/40 text-sm">
              No active skill visualizations yet
            </div>
          ) : tab === "detection" ? (
            <DetectionTab data={detectionData} />
          ) : tab === "segmentation" ? (
            <SegmentationTab data={segmentationData} />
          ) : tab === "seg" ? (
            <SegTab data={segmentationData} />
          ) : tab === "grasp" ? (
            <GraspTab data={graspViz} detectionData={detectionData} />
          ) : tab === "contact" ? (
            <ContactDetectTab data={contactDetectDebug} robotState={robotState} />
          ) : tab === "motion" ? (
            <MotionPlannerTab data={motionPlannerDebug} robotState={robotState} />
          ) : (
            <VlmTab results={vlmResults} />
          )}
        </div>
      </div>
    </div>
  );
}
