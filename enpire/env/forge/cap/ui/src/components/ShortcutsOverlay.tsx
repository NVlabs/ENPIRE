/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useEffect, useRef, useState } from "react";

const HOLD_DURATION_MS = 1000;

const isMac = typeof navigator !== "undefined" && /Mac/.test(navigator.userAgent);
const modKey = isMac ? "Cmd" : "Ctrl";

const SHORTCUTS = [
  { keys: `${modKey}+B`, description: "Toggle Script Browser" },
  { keys: `${modKey}+\\`, description: "Toggle Action Log" },
  { keys: `${modKey}+Alt+L`, description: "Toggle Agent / Oracle mode" },
  { keys: `${modKey}+P`, description: "Focus script search" },
  { keys: `${modKey} (hold 1s)`, description: "Show this overlay" },
];

export default function ShortcutsOverlay() {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeRef = useRef(false);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Only trigger on bare Ctrl/Meta without other keys
      if (!(e.key === "Control" || e.key === "Meta")) return;
      if (activeRef.current) return;

      activeRef.current = true;
      timerRef.current = setTimeout(() => {
        setVisible(true);
      }, HOLD_DURATION_MS);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Control" || e.key === "Meta") {
        activeRef.current = false;
        if (timerRef.current) {
          clearTimeout(timerRef.current);
          timerRef.current = null;
        }
        setVisible(false);
      }
    };

    // Also dismiss if window loses focus
    const onBlur = () => {
      activeRef.current = false;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      setVisible(false);
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  if (!visible) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
      <div className="bg-base-300/85 backdrop-blur-md rounded-2xl shadow-2xl border border-base-content/10 px-8 py-6 max-w-sm w-full">
        <h2 className="text-base font-semibold text-center mb-4 text-base-content/80">
          Keyboard Shortcuts
        </h2>
        <div className="space-y-2">
          {SHORTCUTS.map((s) => (
            <div key={s.keys} className="flex items-center justify-between gap-4">
              <kbd className="kbd kbd-sm bg-base-100/60 text-base-content/90 font-mono whitespace-nowrap">
                {s.keys}
              </kbd>
              <span className="text-sm text-base-content/70 text-right">{s.description}</span>
            </div>
          ))}
        </div>
        <div className="text-xs text-center text-base-content/30 mt-4">
          Release {modKey} to dismiss
        </div>
      </div>
    </div>
  );
}
