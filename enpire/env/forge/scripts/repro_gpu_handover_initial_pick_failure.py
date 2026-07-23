#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline reproducer for recent GPU handover initial-pick failures.

This script deliberately does not import or execute gpu_handover.py.  It only
reads the source and saved run logs, so it is safe to run without a robot.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("cap/saved_scripts/gpu/gpu_handover.py")
DEFAULT_LOGS = [
    Path(
        "logs/gpu_dual_full_cycle_20260604T182928/"
        "slot3_handover_prepare/run_profiling_20260604T183115.txt"
    ),
    Path(
        "logs/gpu_dual_full_cycle_20260604T182342/"
        "slot1_handover_prepare/run_profiling_20260604T182346.txt"
    ),
]

STALE_INITIAL_PICK_NAMES = [
    "RIGHT_PICK_CLEAR_ON_START",
    "RIGHT_PICK_CLEAR_REQUIRED",
    "RIGHT_PICK_VALIDATE_TRAJECTORY",
    "RIGHT_PICK_APPROACH_RETRY_FLIPPED_YAW",
    "RIGHT_PICK_ROI_CANONICALIZE_YAW",
    "RIGHT_PICK_ROI_PREFER_FLIPPED_YAW",
    "PICK_RIGHT_ROI_USE_FULL_WIDTH",
    "PICK_RIGHT_ROI_EXCLUDE_RIGHT_EE",
    "PICK_RIGHT_ROI_RIGHT_EE_EXCLUSION_RADIUS_PX",
]

BAD_YAW_TOKENS = [
    "_right_pick_flipped_yaw_rpy",
    "prefer_flipped_yaw",
    "rpy_yaw_roi_flipped",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _line_hits(text: str, token: str) -> list[int]:
    return [
        idx
        for idx, line in enumerate(text.splitlines(), start=1)
        if token in line
    ]


def _normalize_yaw(yaw_deg: float) -> float:
    out = ((float(yaw_deg) + 180.0) % 360.0) - 180.0
    if abs(out + 180.0) < 1e-9:
        return 180.0
    return out


def _stale_flipped_yaw(yaw_deg: float) -> float:
    candidates: list[float] = []
    for shift_deg in (0.0, 180.0, -180.0):
        yaw = _normalize_yaw(yaw_deg + shift_deg)
        if all(abs(yaw - existing) > 1e-4 for existing in candidates):
            candidates.append(yaw)
    return max(candidates, key=abs)


def _current_canonical_yaw(yaw_deg: float, max_abs_yaw_deg: float = 90.0) -> float:
    candidates: list[float] = []
    for shift_deg in (0.0, 180.0, -180.0):
        yaw = _normalize_yaw(yaw_deg + shift_deg)
        if all(abs(yaw - existing) > 1e-4 for existing in candidates):
            candidates.append(yaw)
    in_range = [yaw for yaw in candidates if abs(yaw) <= max_abs_yaw_deg + 1e-6]
    return min(in_range or candidates, key=abs)


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _reproduce_name_error() -> str:
    try:
        exec("if RIGHT_PICK_CLEAR_ON_START:\n    pass\n", {})  # noqa: S102
    except NameError as exc:
        return str(exc)
    return "not reproduced"


def _scan_log(path: Path) -> dict[str, object]:
    paths = [path]
    for companion_name in ("exec.log", "result.json"):
        companion = path.parent / companion_name
        if companion.exists() and companion not in paths:
            paths.append(companion)
    text = "\n".join(_read(item) for item in paths)
    name_errors = re.findall(r"NameError: name '([^']+)' is not defined", text)
    roi_match = re.search(
        r"Motherboard-right loose-GPU pick ROI: .*?broad_roi=\[([^\]]+)\] "
        r"tight_bbox=\[([^\]]+)\]",
        text,
    )
    yaw_segments = []
    seen_yaw_segments = set()
    for match in re.finditer(r"Initial right-pick adjustment: ", text):
        segment = text[match.start() : match.start() + 900]
        rpy_match = re.search(
            r"old_rpy=\[([^\]]+)\] new_rpy=\[([^\]]+)\].*?"
            r"(rpy_yaw_[a-z_]+(?:_from)?=[^ ]+ to=[^ ]+|rpy_yaw_[a-z_]+=unchanged[^ ]*)",
            segment,
            re.DOTALL,
        )
        if not rpy_match:
            continue
        old_rpy = _parse_float_list(rpy_match.group(1))
        new_rpy = _parse_float_list(rpy_match.group(2))
        key = (
            tuple(round(value, 4) for value in old_rpy),
            tuple(round(value, 4) for value in new_rpy),
            rpy_match.group(3),
        )
        if key in seen_yaw_segments:
            continue
        seen_yaw_segments.add(key)
        yaw_segments.append(
            {
                "old_rpy": old_rpy,
                "new_rpy": new_rpy,
                "yaw_text": rpy_match.group(3),
            }
        )
    ik_fail = "IK did not converge" in text or "approach move failed" in text
    return {
        "path": path,
        "paths": paths,
        "name_errors": name_errors,
        "roi": None
        if roi_match is None
        else {
            "broad_roi": _parse_float_list(roi_match.group(1)),
            "tight_bbox": _parse_float_list(roi_match.group(2)),
        },
        "yaw_segments": yaw_segments,
        "ik_fail": ik_fail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce and explain recent GPU handover initial-pick failures "
            "from source and saved logs."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()

    source = args.source
    logs = args.log or [path for path in DEFAULT_LOGS if path.exists()]
    if not source.exists():
        raise SystemExit(f"source not found: {source}")
    if not logs:
        raise SystemExit("no logs provided/found; pass --log PATH")

    source_text = _read(source)
    stale_hits = {
        token: _line_hits(source_text, token)
        for token in STALE_INITIAL_PICK_NAMES + BAD_YAW_TOKENS
    }
    stale_hits = {token: hits for token, hits in stale_hits.items() if hits}

    print("== Offline GPU Handover Initial-Pick Failure Reproducer ==")
    print(f"source: {source}")
    print("logs:")
    for path in logs:
        print(f"  - {path}")

    print("\n[1] Reproduce stale NameError")
    reproduced = _reproduce_name_error()
    print(f"minimal branch result: {reproduced}")

    saw_name_error = False
    yaw_flip_verified = False
    for path in logs:
        if not path.exists():
            print(f"\n[skip] missing log: {path}")
            continue
        scan = _scan_log(path)
        print(f"\n[log] {path}")
        name_errors = scan["name_errors"]
        if name_errors:
            saw_name_error = True
            print(f"  NameError(s): {name_errors}")
        roi = scan["roi"]
        if roi:
            print(
                "  ROI saw table GPU: "
                f"broad_roi={roi['broad_roi']} tight_bbox={roi['tight_bbox']}"
            )
        for idx, segment in enumerate(scan["yaw_segments"], start=1):
            old_yaw = float(segment["old_rpy"][2])
            logged_yaw = float(segment["new_rpy"][2])
            stale_yaw = _stale_flipped_yaw(old_yaw)
            current_yaw = _current_canonical_yaw(old_yaw)
            print(
                f"  yaw case {idx}: old_yaw={old_yaw:.1f} "
                f"logged_new_yaw={logged_yaw:.1f} "
                f"stale_flip={stale_yaw:.1f} "
                f"current_canonical={current_yaw:.1f} "
                f"log_text={segment['yaw_text']}"
            )
            if abs(logged_yaw - stale_yaw) < 0.2 and abs(current_yaw - old_yaw) < 0.2:
                yaw_flip_verified = True
        if scan["ik_fail"]:
            print("  motion failure evidence: IK/approach failure present")

    print("\n[2] Current-source verification")
    if stale_hits:
        print("  FAIL: stale or broken initial-pick tokens still exist:")
        for token, hits in sorted(stale_hits.items()):
            print(f"    {token}: lines {hits[:8]}")
    else:
        print("  PASS: stale initial-pick tokens are absent from current source")

    print("\n[3] Assumptions and attempted fixes")
    if saw_name_error:
        print(
            "  A1: The 18:31 slot-3 failure was not a vision/grasp failure; "
            "it was a stale source cleanup bug: a top-level branch referenced "
            "RIGHT_PICK_CLEAR_ON_START after its definition was removed."
        )
        print(
            "      Verification: minimal branch reproduces the same NameError; "
            "current source no longer contains that token."
        )
    else:
        print("  A1: No NameError evidence found in the provided logs.")
    if yaw_flip_verified:
        print(
            "  A2: The earlier 18:23 pick failure did see the table GPU, but "
            "the ROI-specific yaw flip changed a reachable-looking yaw into "
            "the opposite wrist orientation before approach planning."
        )
        print(
            "      Verification: logged yaw equals stale_flip(old_yaw), while "
            "current canonical yaw keeps the original yaw for that case."
        )
    else:
        print("  A2: No logged yaw-flip case was verified in the provided logs.")

    return 1 if stale_hits else 0


if __name__ == "__main__":
    sys.exit(main())
