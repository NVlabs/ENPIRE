# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# bowl_stacking.py — place a small bowl on top of (or inside) a larger bowl
import time

import numpy as np
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import (
    debug_marker_v1,
    detect_sam3_depth_candidates_v1,
    show_debug_markers_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)

BOWL_QUERIES_V1 = (
    "bowl",
    "small bowl",
    "ceramic bowl",
    "mixing bowl",
    "round bowl",
)
BOWL_CAMERAS_V1 = ("top", "right", "wrist")


def detect_bowls_v1(
    *,
    cameras=BOWL_CAMERAS_V1,
    score_threshold=0.20,
    max_results=12,
    min_count=2,
):
    """Detect candidate bowl positions. Returns a list sorted by inferred size
    (largest first) so callers can pick base then top.
    """
    raw = detect_sam3_depth_candidates_v1(
        list(BOWL_QUERIES_V1),
        cameras,
        score_threshold=score_threshold,
        max_results=max_results,
        min_depth_m=0.10,
        min_valid_points=20,
    )
    # Dedupe by xy position (different cameras may detect the same bowl twice).
    deduped = []
    for det in raw:
        pos = np.asarray(det["pos"], dtype=float)
        dup = False
        for prev in deduped:
            ppos = np.asarray(prev["pos"], dtype=float)
            if float(np.linalg.norm(pos[:2] - ppos[:2])) < 0.04:
                dup = True
                break
        if not dup:
            deduped.append(det)

    if len(deduped) < int(min_count):
        raise RuntimeError(
            f"only detected {len(deduped)} unique bowls (need >= {min_count})"
        )

    # bbox_xywh area is a proxy for size; bigger w*h ~= larger bowl.
    def _size(d):
        bbox = d.get("bbox_xywh") or [0, 0, 1, 1]
        return float(bbox[2]) * float(bbox[3])

    deduped.sort(key=_size, reverse=True)
    print("  detected bowls (largest first):")
    for i, det in enumerate(deduped):
        print(
            f"    [{i}] {det['query']!r} from {det['camera']} "
            f"score={float(det['score']):.3f} size_proxy={_size(det):.0f} "
            f"pos={fmt_xyz_v1(det['pos'])}"
        )
    return deduped


def plan_stack_target_v1(lower_bowl_pos, *, vertical_clearance_m=0.04):
    """Compute a target position for placing the upper bowl on top of the lower."""
    lower = np.asarray(lower_bowl_pos, dtype=float)
    target = lower.copy()
    target[2] += float(vertical_clearance_m)
    return target


def place_bowl_at_v1(
    side,
    place_pos,
    *,
    hover_clearance_m=0.16,
    place_z_offset_m=0.02,
    xy_offsets_m=(
        (0.0, 0.0),
        (0.02, 0.0),
        (-0.02, 0.0),
        (0.0, 0.02),
        (0.0, -0.02),
        (0.03, 0.03),
        (-0.03, -0.03),
    ),
    retreat_distance_m=0.20,
):
    """Place a held bowl at a target position (counter, cabinet, or on top of
    another bowl).
    """
    place_pos = np.asarray(place_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover = place_pos.copy()
    hover[2] += float(hover_clearance_m)
    hover_status = "Skipped"
    for dx, dy in xy_offsets_m:
        candidate = hover.copy()
        candidate[0] += float(dx)
        candidate[1] += float(dy)
        result = freespace_move(
            right_target_pos=candidate.tolist(),
            right_target_quat=current_quat.tolist(),
            side=side,
            gripper=0.1,
            auto_update_world=True,
        )
        hover_status = str(getattr(result, "status", result))
        print(f"  bowl-place hover xy=({dx:+.3f},{dy:+.3f}): status={hover_status}")
        if hover_status == "Success":
            break
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    place_z = float(place_pos[2]) + float(place_z_offset_m)
    lower_status = "Skipped"
    for dx, dy in xy_offsets_m:
        candidate = place_pos.copy()
        candidate[0] += float(dx)
        candidate[1] += float(dy)
        candidate[2] = place_z
        result = freespace_move(
            right_target_pos=candidate.tolist(),
            side=side,
            gripper=0.1,
            auto_update_world=True,
        )
        lower_status = str(getattr(result, "status", result))
        print(f"  bowl-place lower xy=({dx:+.3f},{dy:+.3f}): status={lower_status}")
        if lower_status == "Success":
            break

    if lower_status != "Success":
        nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -(float(hover_clearance_m) - float(place_z_offset_m))],
            n_steps=10,
        )

    print("  open gripper to release bowl")
    open_gripper(side)
    time.sleep(0.4)

    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=8,
    )
    return True, {
        "phase": "done",
        "place_z": float(place_z),
        "retreat_success": bool(getattr(retreat, "success", False)),
    }


def show_bowl_debug_markers_v1(*, lower=None, upper=None, target=None):
    markers = []
    if lower is not None:
        markers.append(
            debug_marker_v1(
                "bowl_lower",
                np.asarray(lower, dtype=float),
                (96, 192, 255),
                radius=0.025,
            )
        )
    if upper is not None:
        markers.append(
            debug_marker_v1(
                "bowl_upper",
                np.asarray(upper, dtype=float),
                (96, 255, 192),
                radius=0.022,
            )
        )
    if target is not None:
        markers.append(
            debug_marker_v1(
                "bowl_stack_target",
                np.asarray(target, dtype=float),
                (255, 80, 80),
                radius=0.020,
            )
        )
    show_debug_markers_v1(markers)
