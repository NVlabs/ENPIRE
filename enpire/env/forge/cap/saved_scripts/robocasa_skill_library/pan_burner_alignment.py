# pan_burner_alignment.py — place a grasped pan/kettle/pot onto a stove burner
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import (
    debug_marker_v1,
    detect_sam3_depth_candidates_v1,
    show_debug_markers_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)


BURNER_QUERIES_V1 = (
    "stove burner",
    "stove burner ring",
    "burner on stove",
    "stove cooktop circle",
    "stovetop burner",
    "induction cooktop element",
)
STOVE_QUERIES_V1 = (
    "stove cooktop",
    "stove",
    "stovetop",
    "cooktop",
)
BURNER_CAMERAS_V1 = ("top", "right")


def detect_stove_top_v1(cameras=BURNER_CAMERAS_V1):
    """Detect the stove cooktop bulk position. Used as a spatial anchor."""
    try:
        raw = detect_sam3_depth_candidates_v1(
            list(STOVE_QUERIES_V1),
            cameras,
            score_threshold=0.20,
            max_results=8,
            min_depth_m=0.10,
            min_valid_points=80,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"could not detect stove cooktop: {exc}")
    raw.sort(key=lambda d: float(d["score"]), reverse=True)
    chosen = raw[0]
    print(
        f"  stove top: {chosen['query']!r} from {chosen['camera']} "
        f"score={chosen['score']:.3f} pos={fmt_xyz_v1(chosen['pos'])}"
    )
    return chosen


def detect_burner_sites_v1(
    *,
    cameras=BURNER_CAMERAS_V1,
    score_threshold=0.10,
    max_results=20,
    stove_anchor_pos=None,
    max_dist_from_stove_m=0.45,
):
    """Detect burner candidate positions. Returns a list of dicts sorted by score."""
    raw = detect_sam3_depth_candidates_v1(
        list(BURNER_QUERIES_V1),
        cameras,
        score_threshold=score_threshold,
        max_results=max_results,
        min_depth_m=0.10,
        min_valid_points=15,
    )
    if stove_anchor_pos is not None:
        center = np.asarray(stove_anchor_pos, dtype=float)
        kept = []
        for det in raw:
            pos = np.asarray(det["pos"], dtype=float)
            if float(np.linalg.norm(pos[:2] - center[:2])) <= float(max_dist_from_stove_m):
                kept.append(det)
        if kept:
            raw = kept
            print(
                f"  burners filtered to {len(kept)} within "
                f"{float(max_dist_from_stove_m):.2f}m of stove anchor"
            )

    deduped = _dedupe_burner_positions_v1(raw, dist_thresh_m=0.04)
    if not deduped:
        raise RuntimeError("no burner candidates after dedupe")
    print(f"  burners after dedupe: {len(deduped)}")
    for i, det in enumerate(deduped):
        print(
            f"    burner[{i}] {det['query']!r} from {det['camera']} "
            f"score={float(det['score']):.3f} pos={fmt_xyz_v1(det['pos'])}"
        )
    return deduped


def _dedupe_burner_positions_v1(detections, dist_thresh_m=0.04):
    ordered = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    kept = []
    for det in ordered:
        pos = np.asarray(det["pos"], dtype=float)
        if any(
            float(np.linalg.norm(pos[:2] - np.asarray(prev["pos"], dtype=float)[:2]))
            < float(dist_thresh_m)
            for prev in kept
        ):
            continue
        kept.append(det)
    return kept


def choose_target_burner_v1(
    burner_dets,
    *,
    side="right",
    prefer="closest_to_arm",
):
    """Pick a single burner. ``prefer`` is one of:
    - "closest_to_arm": closest XY to current EE position (good default)
    - "highest_score": highest SAM3 score
    - "rightmost": largest X coordinate (matches "right arm reach")
    """
    if not burner_dets:
        raise RuntimeError("no burner candidates to choose from")
    if prefer == "highest_score":
        chosen = max(burner_dets, key=lambda d: float(d["score"]))
        reason = "highest_score"
    elif prefer == "rightmost":
        chosen = max(burner_dets, key=lambda d: float(d["pos"][0]))
        reason = "rightmost"
    else:
        ee_pos = np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)
        chosen = min(
            burner_dets,
            key=lambda d: float(
                np.linalg.norm(np.asarray(d["pos"], dtype=float)[:2] - ee_pos[:2])
            ),
        )
        reason = "closest_to_arm"
    print(
        f"  chosen burner ({reason}): pos={fmt_xyz_v1(chosen['pos'])} "
        f"score={float(chosen['score']):.3f}"
    )
    return chosen


def align_pan_on_burner_v1(
    side,
    burner_pos,
    *,
    hover_clearance_m=0.18,
    place_z_offset_m=0.04,
    xy_offsets_m=(
        (0.0, 0.0),
        (0.02, 0.0),
        (-0.02, 0.0),
        (0.0, 0.02),
        (0.0, -0.02),
        (0.03, 0.03),
        (-0.03, -0.03),
        (0.05, 0.0),
        (-0.05, 0.0),
    ),
    retreat_distance_m=0.25,
    release_settle_s=0.4,
):
    """Place a grasped object on the burner site.

    Hover above with extra Z, lower with XY-candidate offsets, open gripper,
    retreat upward. Default `retreat_distance_m=0.25` matches the default
    `gripper_obj_far` threshold checked by KettleBoiling/SearingMeat.
    Returns ``(success, info)``.
    """
    burner_pos = np.asarray(burner_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover_pos = burner_pos.copy()
    hover_pos[2] += float(hover_clearance_m)
    print(
        "\n--- Burner align: hover above burner ---\n"
        f"  burner={fmt_xyz_v1(burner_pos)} hover={fmt_xyz_v1(hover_pos)}"
    )
    hover_status = "Skipped"
    for dx, dy in xy_offsets_m:
        candidate = hover_pos.copy()
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
        print(
            f"  hover xy=({dx:+.3f},{dy:+.3f}): status={hover_status} "
            f"target={fmt_xyz_v1(candidate)}"
        )
        if hover_status == "Success":
            break
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    place_z = float(burner_pos[2]) + float(place_z_offset_m)
    lower_status = "Skipped"
    used_offset_xy = (0.0, 0.0)
    for dx, dy in xy_offsets_m:
        candidate = burner_pos.copy()
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
        print(
            f"  lower xy=({dx:+.3f},{dy:+.3f}): status={lower_status} "
            f"target={fmt_xyz_v1(candidate)}"
        )
        if lower_status == "Success":
            used_offset_xy = (float(dx), float(dy))
            break

    if lower_status != "Success":
        # Fallback: brutal nudge straight down to seat the object on the burner.
        nudge_result = nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -(float(hover_clearance_m) - float(place_z_offset_m))],
            n_steps=10,
        )
        nudge_ok = bool(getattr(nudge_result, "success", False))
        print(f"  nudge-down fallback: success={nudge_ok}")

    print("  open gripper to release")
    open_gripper(side)
    time.sleep(float(release_settle_s))

    print(f"  retreat upward {float(retreat_distance_m):.2f}m")
    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=10,
    )
    retreat_ok = bool(getattr(retreat, "success", False))
    print(f"  retreat success={retreat_ok}")

    return True, {
        "phase": "done",
        "used_offset_xy": list(used_offset_xy),
        "place_z": float(place_z),
        "retreat_success": retreat_ok,
    }


def show_burner_debug_markers_v1(*, burner_dets=None, chosen=None, stove_pos=None):
    markers = []
    if stove_pos is not None:
        markers.append(
            debug_marker_v1(
                "stove_top",
                np.asarray(stove_pos, dtype=float),
                (180, 96, 255),
                radius=0.04,
            )
        )
    for i, det in enumerate(burner_dets or []):
        pos = np.asarray(det["pos"], dtype=float)
        markers.append(
            debug_marker_v1(
                f"burner_{i}",
                pos,
                (96, 192, 255),
                radius=0.020,
            )
        )
    if chosen is not None:
        markers.append(
            debug_marker_v1(
                "burner_chosen",
                np.asarray(chosen["pos"], dtype=float),
                (255, 80, 80),
                radius=0.026,
            )
        )
    show_debug_markers_v1(markers)
