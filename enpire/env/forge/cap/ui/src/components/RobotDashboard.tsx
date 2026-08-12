/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import type { ArmState, RobotState } from "../hooks/useRobotState";

interface RobotDashboardProps {
  state: RobotState;
  showDashboard: boolean;
  onToggle: () => void;
}

function fmt(n: number): string {
  return n.toFixed(3);
}

function ArmTable({
  label,
  arm,
}: {
  label: string;
  arm: ArmState;
}) {
  const { joint_positions, gripper, ee_pose } = arm;

  return (
    <div className="card card-compact bg-base-200">
      <div className="card-body p-3">
        <h3 className="card-title text-sm">{label} Arm</h3>

        <table className="table table-xs">
          <thead>
            <tr>
              <th>Joint</th>
              <th>Position (rad)</th>
            </tr>
          </thead>
          <tbody>
            {joint_positions.map((val, i) => (
              <tr key={i}>
                <td className="font-mono">J{i + 1}</td>
                <td className="font-mono">{fmt(val)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="flex items-center gap-4 text-xs">
          <span>
            Gripper:{" "}
            <span className="badge badge-sm font-mono">
              {fmt(gripper)}
            </span>
          </span>
        </div>

        <div className="text-xs">
          <span className="font-medium">EE Pos: </span>
          <span className="font-mono">
            [{fmt(ee_pose.position[0])}, {fmt(ee_pose.position[1])},{" "}
            {fmt(ee_pose.position[2])}]
          </span>
        </div>
        <div className="text-xs">
          <span className="font-medium">EE Quat: </span>
          <span className="font-mono">
            [{fmt(ee_pose.quaternion[0])}, {fmt(ee_pose.quaternion[1])},{" "}
            {fmt(ee_pose.quaternion[2])}, {fmt(ee_pose.quaternion[3])}]
          </span>
        </div>
      </div>
    </div>
  );
}

export default function RobotDashboard({ state, showDashboard, onToggle }: RobotDashboardProps) {
  return (
    <div>
      <button
        className="btn btn-ghost btn-xs w-full justify-between"
        onClick={onToggle}
      >
        <span className="text-xs font-semibold">Robot Dashboard</span>
        <span className="text-xs">{showDashboard ? "▲" : "▼"}</span>
      </button>
      {showDashboard && (
        <div className={`grid gap-2 mt-1 ${Object.keys(state.arms).length > 1 ? "grid-cols-2" : "grid-cols-1"}`}>
          {Object.keys(state.arms).length === 0 ? (
            <div className="text-xs text-base-content/50 p-3">No robot connected</div>
          ) : (
            Object.entries(state.arms).map(([name, arm]) => (
              <ArmTable key={name} label={name.charAt(0).toUpperCase() + name.slice(1)} arm={arm} />
            ))
          )}
        </div>
      )}
    </div>
  );
}
