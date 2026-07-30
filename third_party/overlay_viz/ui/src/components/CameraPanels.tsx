// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cameraVideoUrl, cameraPosterUrl, liveStreamUrl, overlayUrl } from "@/api/client"
import type { EpisodeInfo } from "@/api/types"

interface Props {
  taskId: string
  viewerIdx: number
  info: EpisodeInfo
  replayConnected: boolean
  cameraStreaming: boolean
  overlayAlpha: number
  colorMode: string
  overlayBust: number
  registerVideo: (el: HTMLVideoElement | null) => void
}

function VideoPanel({ src, poster, label, registerVideo }: {
  src: string; poster: string; label: string; registerVideo: (el: HTMLVideoElement | null) => void
}) {
  return (
    <div className="flex flex-col">
      <div className="text-xs font-semibold text-center mb-1 opacity-70">{label}</div>
      <div className="bg-muted rounded-lg overflow-hidden aspect-[4/3] flex items-center justify-center">
        <video
          key={src}
          ref={registerVideo}
          src={src}
          poster={poster}
          className="w-full h-full object-contain"
          muted
          playsInline
          preload="metadata"
        />
      </div>
    </div>
  )
}

export function CameraPanels(p: Props) {
  if (p.replayConnected) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="flex flex-col">
          <div className="text-xs font-semibold text-center mb-1 opacity-70">Real-world (Live)</div>
          <div className="bg-muted rounded-lg overflow-hidden aspect-[4/3] flex items-center justify-center">
            {p.cameraStreaming ? (
              <img src={liveStreamUrl(p.overlayBust)} className="w-full h-full object-contain" alt="Live" />
            ) : (
              <span className="text-xs opacity-50">Waiting for live frames...</span>
            )}
          </div>
        </div>
        <VideoPanel
          src={cameraVideoUrl(p.taskId, p.viewerIdx, "top")}
          poster={cameraPosterUrl(p.taskId, p.viewerIdx, "top")}
          label="Dataset Frame"
          registerVideo={p.registerVideo}
        />
        <div className="flex flex-col">
          <div className="text-xs font-semibold text-center mb-1 opacity-70">Overlay ({Math.round(p.overlayAlpha * 100)}%)</div>
          <div className="bg-muted rounded-lg overflow-hidden aspect-[4/3] flex items-center justify-center">
            {p.cameraStreaming ? (
              <img src={overlayUrl(p.taskId, p.viewerIdx, p.overlayAlpha, p.colorMode, p.overlayBust)} className="w-full h-full object-contain" alt="Overlay" />
            ) : (
              <span className="text-xs opacity-50">Needs live camera</span>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
      {(["left", "top", "right"] as const).map((cam) => (
        <VideoPanel
          key={cam}
          src={p.info.cameras[cam] ? cameraVideoUrl(p.taskId, p.viewerIdx, cam) : ""}
          poster={p.info.cameras[cam] ? cameraPosterUrl(p.taskId, p.viewerIdx, cam) : ""}
          label={`${cam.charAt(0).toUpperCase() + cam.slice(1)} Camera`}
          registerVideo={p.registerVideo}
        />
      ))}
    </div>
  )
}
