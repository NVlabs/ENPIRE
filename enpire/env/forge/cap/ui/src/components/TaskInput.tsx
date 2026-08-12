/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useState } from "react";

interface TaskInputProps {
  onSubmit: (description: string) => void;
  disabled?: boolean;
}

export default function TaskInput({ onSubmit, disabled }: TaskInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <div className="join w-full">
      <input
        type="text"
        className="input input-bordered join-item flex-1"
        placeholder="Describe a task... e.g. 'Pick up the red cup'"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSubmit();
        }}
        disabled={disabled}
      />
      <button
        className="btn btn-primary join-item"
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
      >
        Submit Task
      </button>
    </div>
  );
}
