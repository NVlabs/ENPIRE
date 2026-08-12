/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

interface ApprovalPanelProps {
  visible: boolean;
  onApprove: () => void;
  onReject: () => void;
  onEdit: () => void;
}

export default function ApprovalPanel({
  visible,
  onApprove,
  onReject,
  onEdit,
}: ApprovalPanelProps) {
  if (!visible) return null;

  return (
    <div className="alert alert-info flex items-center gap-2">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        className="h-6 w-6 shrink-0 stroke-current"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
      <span className="flex-1 text-sm">LLM generated code awaiting review</span>
      <div className="flex gap-2">
        <button className="btn btn-success btn-sm" onClick={onApprove}>
          Approve
        </button>
        <button className="btn btn-warning btn-sm" onClick={onEdit}>
          Edit
        </button>
        <button className="btn btn-error btn-sm" onClick={onReject}>
          Reject
        </button>
      </div>
    </div>
  );
}
