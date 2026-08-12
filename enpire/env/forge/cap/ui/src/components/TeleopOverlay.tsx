/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import type { LearnSkillStatus } from "../hooks/useRobotState";
import LearnSkillPanel from "./LearnSkillPanel";

interface TeleopOverlayProps {
  learnSkillStatus: LearnSkillStatus;
  visible: boolean;
  onClose: () => void;
}

export default function TeleopOverlay({ learnSkillStatus, visible, onClose }: TeleopOverlayProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-base-100"
      style={{ display: visible ? undefined : "none" }}
    >
      <div className="relative w-[80vw] h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between shrink-0 pb-2">
          <span className="text-sm font-bold tracking-wider text-base-content/60">TELEOP</span>
          <button
            className="btn btn-ghost btn-sm"
            onClick={onClose}
          >
            Close (Esc)
          </button>
        </div>

        {/* 3D View */}
        <div className="flex-1 min-h-0 rounded overflow-hidden">
          <iframe
            src={`http://${window.location.hostname}:8080`}
            className="h-full w-full border-0"
            title="Viser 3D View"
          />
        </div>

        {/* Learn Skill status */}
        <div className="shrink-0 pt-2">
          <LearnSkillPanel status={learnSkillStatus} />
        </div>
      </div>
    </div>
  );
}
