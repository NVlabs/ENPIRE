# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset both inserted GPUs using the soft-bar unplug primitive.

Runs slot 3 first, then slot 1, using `gpu_reset_soft_bar.py` for each slot.
This keeps both unplug operations in one run_script process while preserving
the slot-specific soft-bar defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
import traceback


RESET_SCRIPT = Path.cwd() / "cap" / "saved_scripts" / "gpu" / "gpu_reset_soft_bar.py"
SCRIPT_COMPLETED = False
SLOT_RESULTS = []


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _slot_place_env(slot: int) -> dict[str, str]:
    if int(slot) == 3:
        return {
            "GPU_UNPLUG_LEFT_TABLE_PLACE_X": _env("GPU_DUAL_SLOT3_DROP_X", "0.528"),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_Y": _env("GPU_DUAL_SLOT3_DROP_Y", "-0.240"),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_Z": _env("GPU_DUAL_SLOT3_DROP_Z", "0.864"),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_ROLL_DEG": _env(
                "GPU_DUAL_SLOT3_DROP_ROLL_DEG",
                "-52.2",
            ),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_PITCH_DEG": _env(
                "GPU_DUAL_SLOT3_DROP_PITCH_DEG",
                "143.7",
            ),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_YAW_DEG": _env(
                "GPU_DUAL_SLOT3_DROP_YAW_DEG",
                "16.7",
            ),
        }
    if int(slot) == 1:
        return {
            "GPU_UNPLUG_LEFT_TABLE_PLACE_X": _env("GPU_DUAL_SLOT1_DROP_X", "0.528"),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_Y": _env("GPU_DUAL_SLOT1_DROP_Y", "-0.154"),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_Z": _env("GPU_DUAL_SLOT1_DROP_Z", "0.864"),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_ROLL_DEG": _env(
                "GPU_DUAL_SLOT1_DROP_ROLL_DEG",
                "-52.2",
            ),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_PITCH_DEG": _env(
                "GPU_DUAL_SLOT1_DROP_PITCH_DEG",
                "143.7",
            ),
            "GPU_UNPLUG_LEFT_TABLE_PLACE_YAW_DEG": _env(
                "GPU_DUAL_SLOT1_DROP_YAW_DEG",
                "16.7",
            ),
        }
    raise ValueError(f"unsupported reset slot: {slot}")


def _run_gpu_reset_slot(slot: int, *, go_home_on_start: bool) -> dict:
    old_env = os.environ.copy()
    updates = {
        "GPU_TARGET_SOCKET_NUMBER": str(int(slot)),
        "GPU_UNPLUG_GO_HOME_ON_START": "1" if go_home_on_start else "0",
    }
    updates.update(_slot_place_env(slot))
    os.environ.update(updates)
    print(
        "[gpu_reset_soft_bar_dual] Running gpu_reset_soft_bar.py for "
        f"slot {int(slot)} with place=("
        f"{updates['GPU_UNPLUG_LEFT_TABLE_PLACE_X']}, "
        f"{updates['GPU_UNPLUG_LEFT_TABLE_PLACE_Y']}, "
        f"{updates['GPU_UNPLUG_LEFT_TABLE_PLACE_Z']}) "
        f"go_home_on_start={go_home_on_start}"
    )
    try:
        namespace = {
            key: value
            for key, value in globals().items()
            if key not in {"__name__", "__file__", "__package__", "__spec__"}
        }
        namespace.update(
            {
                "__builtins__": __builtins__,
                "__file__": str(RESET_SCRIPT),
                "__name__": f"gpu_reset_soft_bar_slot_{int(slot)}",
            }
        )
        code = RESET_SCRIPT.read_text(encoding="utf-8")
        exec(compile(code, str(RESET_SCRIPT), "exec"), namespace)  # noqa: S102
        info_fn = namespace.get("get_task_info")
        info = dict(info_fn()) if callable(info_fn) else {}
        success = bool(info.get("success", False))
        info.update({"slot": int(slot), "success": success})
        if not success:
            raise RuntimeError(
                f"slot {int(slot)} gpu_reset_soft_bar.py did not report success: {info}"
            )
        return info
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"slot {int(slot)} reset failed: {exc}") from exc
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def main():
    global SCRIPT_COMPLETED, SLOT_RESULTS
    SLOT_RESULTS = []
    for index, slot in enumerate((3, 1)):
        go_home_default = index == 0
        go_home = _env(
            f"GPU_DUAL_SLOT{slot}_RESET_GO_HOME_ON_START",
            "1" if go_home_default else "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        SLOT_RESULTS.append(_run_gpu_reset_slot(slot, go_home_on_start=go_home))
    SCRIPT_COMPLETED = True
    print("[gpu_reset_soft_bar_dual] Complete: slot 3 and slot 1 reset succeeded.")


def get_task_info() -> dict:
    return {
        "success": bool(SCRIPT_COMPLETED),
        "reward": 1.0 if SCRIPT_COMPLETED else 0.0,
        "method": "dual_gpu_reset_soft_bar",
        "slots": SLOT_RESULTS,
    }


main()

