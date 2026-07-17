# stove_knob_detection.py - TurnOffStove vision detection and target selection
from skill_library.namespace import *  # noqa: F401, F403

import re

import cv2
import numpy as np

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.stove_knob_common import *  # noqa: F401, F403


def detect_turn_off_stove_knobs_v1(
    *,
    queries=STOVE_KNOB_QUERIES_V1,
    cameras=STOVE_KNOB_CAMERAS_V1,
    score_threshold=0.1,
    max_results=12,
    show_markers=True,
    target_knob=None,
):
    task_description = stove_task_description_v1()
    if target_knob is None:
        target_knob = parse_stove_target_knob_v1(task_description)
    else:
        target_knob = str(target_knob)
        print(f"  using explicit target_knob={target_knob!r} (skipping task parse)")
    groups = []

    for camera in cameras:
        depth = np.asarray(get_camera_depth(camera), dtype=np.float32)
        transform, fx, fy, cx, cy = camera_transform_v1(camera)
        for query in queries:
            try:
                segmentations = segment_object_all(
                    query,
                    camera=camera,
                    score_thresh=float(score_threshold),
                    max_results=int(max_results),
                )
            except Exception as exc:
                print(f"  SAM skip {camera}/{query}: {type(exc).__name__}: {exc}")
                continue

            comps = []
            for det_i, seg in enumerate(segmentations):
                mask = np.asarray(seg["mask"], dtype=np.int32)
                for comp_i, comp in enumerate(stove_connected_components_v1(mask)):
                    bp = backproject_mask_with_pixel_v1(
                        comp,
                        depth,
                        transform,
                        fx,
                        fy,
                        cx,
                        cy,
                        lower_middle_z=True,
                    )
                    if bp is None:
                        continue
                    comps.append(
                        {
                            "camera": camera,
                            "query": query,
                            "score": float(seg.get("score", 0.0)),
                            "pos": np.asarray(bp["pos"], dtype=float),
                            "pixel_uv": bp["pixel_uv"],
                            "n_valid": int(bp["n_valid"]),
                            "world_z_range_m": float(bp["world_z_range_m"]),
                            "world_z_p90_p10_m": float(bp["world_z_p90_p10_m"]),
                            "world_z_std_m": float(bp["world_z_std_m"]),
                            "pos_mode": str(bp.get("pos_mode", "mask_median")),
                            "z_band_center_m": bp.get("z_band_center_m"),
                            "z_band_n": int(bp.get("z_band_n", 0)),
                            "area": int((comp > 0).sum()),
                            "det_index": int(det_i),
                            "component_index": int(comp_i),
                            "bbox_xywh": seg.get("bbox_xywh"),
                        }
                    )

            if comps:
                filtered = filter_stove_knob_row_candidates_v1(comps)
                if filtered:
                    groups.append(
                        {
                            "camera": camera,
                            "query": query,
                            "score": max(float(c["score"]) for c in filtered),
                            "raw_count": len(comps),
                            "candidates": filtered,
                        }
                    )

    if not groups:
        raise RuntimeError("no stove knob candidates from SAM/depth")

    groups.sort(
        key=lambda g: (
            min(len(g["candidates"]), 8),
            float(g["score"]),
            -tuple(cameras).index(g["camera"]),
        ),
        reverse=True,
    )
    selected_group = groups[0]
    candidates = list(selected_group["candidates"])
    normal_seed = np.median([np.asarray(c["pos"]) for c in candidates], axis=0)
    surface_normal = stove_camera_surface_normal_v1(normal_seed)
    right_axis = stove_right_axis_v1(surface_normal)

    for i, cand in enumerate(candidates):
        cand["visible_index"] = i
        cand["predicted"] = False
    candidates, gap_info = predict_blocked_stove_knobs_v1(candidates, right_axis)
    for i, cand in enumerate(candidates):
        cand["index"] = i

    target_candidates = select_stove_knob_candidates_by_target_v1(
        candidates,
        target_knob,
        right_axis,
    )
    selected = target_candidates[0]
    state = {
        "task_description": task_description,
        "target_knob": target_knob,
        "selected_group": selected_group,
        "candidates": candidates,
        "target_candidates": target_candidates,
        "selected_candidate": selected,
        "surface_normal": surface_normal,
        "right_axis": right_axis,
        "gap_prediction": gap_info,
    }
    prepare_stove_pose_from_selected_v1(state)
    print_stove_detection_summary_v1(state)
    if show_markers:
        show_stove_debug_markers_v1(state)
    return state



def stove_task_description_v1():
    try:
        return str(get_task_description())
    except Exception:
        info = get_task_info()
        for key in ("task_description", "language", "description"):
            if key in info:
                return str(info[key])
        return str(info.get("env_name", ""))


def parse_stove_target_knob_v1(text):
    normalized = str(text).lower().replace("-", " ").replace("_", " ")
    patterns = (
        ("front_left", r"\bfront\s+left\b"),
        ("front_center", r"\bfront\s+center\b|\bfront\s+middle\b"),
        ("front_right", r"\bfront\s+right\b"),
        ("rear_left", r"\brear\s+left\b|\bback\s+left\b"),
        ("rear_center", r"\brear\s+center\b|\brear\s+middle\b|\bback\s+center\b"),
        ("rear_right", r"\brear\s+right\b|\bback\s+right\b"),
        ("center", r"\bcenter\b|\bmiddle\b"),
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
    )
    for name, pattern in patterns:
        if re.search(pattern, normalized):
            return name
    raise RuntimeError(f"could not parse target knob from task text: {text!r}")


def stove_connected_components_v1(mask):
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    image_area = int(binary.shape[0] * binary.shape[1])
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 25 or area > int(0.20 * image_area):
            continue
        components.append((labels == label_id).astype(np.int32))
    components.sort(key=lambda comp: int((comp > 0).sum()), reverse=True)
    return components[:12]


def filter_stove_knob_row_candidates_v1(candidates):
    if not candidates:
        return []
    usable = [
        c
        for c in candidates
        if (
            50 <= int(c.get("area", 0)) <= 1000
            and float(c.get("score", 0.0)) >= 0.1
        )
    ]
    if not usable:
        usable = list(candidates)

    max_v = max(float(c["pixel_uv"][1]) for c in usable)
    row = [c for c in usable if float(c["pixel_uv"][1]) >= max_v - 50.0]
    if len(row) >= 2:
        usable = row

    usable = deduplicate_stove_candidates_v1(usable)
    max_score = max(float(c.get("score", 0.0)) for c in usable)
    score_floor = max(0.18, 0.45 * max_score)
    strong = [c for c in usable if float(c.get("score", 0.0)) >= score_floor]
    if len(strong) >= 2:
        usable = strong

    usable = deduplicate_stove_candidates_v1(usable)
    if len(usable) > 10:
        usable = sorted(
            usable,
            key=lambda c: (float(c.get("score", 0.0)), int(c.get("area", 0))),
            reverse=True,
        )[:10]
    return usable


def deduplicate_stove_candidates_v1(candidates):
    ordered = sorted(
        candidates,
        key=lambda c: (float(c.get("score", 0.0)), int(c.get("area", 0))),
        reverse=True,
    )
    kept = []
    for cand in ordered:
        pos = np.asarray(cand["pos"], dtype=float)
        uv = np.asarray(cand["pixel_uv"], dtype=float)
        duplicate = False
        for prev in kept:
            prev_pos = np.asarray(prev["pos"], dtype=float)
            prev_uv = np.asarray(prev["pixel_uv"], dtype=float)
            if (
                float(np.linalg.norm(pos - prev_pos)) < 0.035
                or float(np.linalg.norm(uv - prev_uv)) < 12.0
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def predict_blocked_stove_knobs_v1(candidates, row_axis):
    if len(candidates) < 2:
        return candidates, {"gaps": [], "nominal_gap_m": None, "predicted": []}

    axis = normalize_v1(np.asarray(row_axis, dtype=float))
    ordered = sorted(
        candidates,
        key=lambda c: float(np.dot(np.asarray(c["pos"], dtype=float), axis)),
    )
    gaps = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_pos = np.asarray(left["pos"], dtype=float)
        right_pos = np.asarray(right["pos"], dtype=float)
        gaps.append(
            {
                "left_visible_index": int(left.get("visible_index", -1)),
                "right_visible_index": int(right.get("visible_index", -1)),
                "gap_m": float(np.dot(right_pos - left_pos, axis)),
            }
        )

    positive_gaps = [g["gap_m"] for g in gaps if 0.035 <= float(g["gap_m"]) <= 0.25]
    if not positive_gaps:
        return candidates, {"gaps": gaps, "nominal_gap_m": None, "predicted": []}

    gap_arr = np.asarray(positive_gaps, dtype=float)
    initial_gap = float(np.median(gap_arr))
    close_gaps = gap_arr[gap_arr <= initial_gap * 1.35]
    nominal_gap = float(np.median(close_gaps if len(close_gaps) else gap_arr))
    if not 0.045 <= nominal_gap <= 0.18:
        return candidates, {"gaps": gaps, "nominal_gap_m": nominal_gap, "predicted": []}

    expanded = list(candidates)
    predicted = []
    for pair_idx, (left, right, gap) in enumerate(zip(ordered[:-1], ordered[1:], gaps)):
        gap_m = float(gap["gap_m"])
        missing = int(round(gap_m / nominal_gap)) - 1
        if gap_m < nominal_gap * STOVE_GAP_INSERT_RATIO_V1:
            missing = 0
        missing = int(np.clip(missing, 0, 4))
        if missing <= 0:
            continue

        left_pos = np.asarray(left["pos"], dtype=float)
        right_pos = np.asarray(right["pos"], dtype=float)
        left_uv = np.asarray(left.get("pixel_uv", [0.0, 0.0]), dtype=float)
        right_uv = np.asarray(right.get("pixel_uv", left_uv), dtype=float)
        for k in range(1, missing + 1):
            t = k / float(missing + 1)
            pos = left_pos * (1.0 - t) + right_pos * t
            uv = left_uv * (1.0 - t) + right_uv * t
            pred = {
                "camera": str(left.get("camera", "")),
                "query": "predicted blocked knob",
                "score": min(float(left.get("score", 0.0)), float(right.get("score", 0.0))),
                "pos": pos,
                "pixel_uv": uv,
                "n_valid": 0,
                "area": 0,
                "det_index": -1,
                "component_index": -1,
                "bbox_xywh": None,
                "predicted": True,
                "prediction_pair": pair_idx,
                "prediction_gap_m": gap_m,
                "prediction_nominal_gap_m": nominal_gap,
                "prediction_fraction": t,
            }
            for stat_key in ("world_z_range_m", "world_z_p90_p10_m", "world_z_std_m"):
                left_stat = left.get(stat_key)
                right_stat = right.get(stat_key)
                if left_stat is not None and right_stat is not None:
                    pred[stat_key] = (
                        float(left_stat) * (1.0 - t) + float(right_stat) * t
                    )
            expanded.append(pred)
            predicted.append(
                {
                    "between": (
                        int(left.get("visible_index", -1)),
                        int(right.get("visible_index", -1)),
                    ),
                    "pos": pos.tolist(),
                    "gap_m": gap_m,
                    "fraction": t,
                }
            )

    expanded.sort(key=lambda c: float(np.dot(np.asarray(c["pos"], dtype=float), axis)))
    return expanded, {"gaps": gaps, "nominal_gap_m": nominal_gap, "predicted": predicted}


def stove_camera_surface_normal_v1(_anchor_pos=None):
    top_pos = np.asarray(get_camera_extrinsics("top")["position"], dtype=float)
    right_pos = np.asarray(get_camera_extrinsics("right")["position"], dtype=float)
    camera_vec = np.array(
        [right_pos[0] - top_pos[0], right_pos[1] - top_pos[1], 0.0],
        dtype=float,
    )
    normal = xy_cw_vertical_v1(camera_vec)
    print(f"  stove_top_camera_pos={fmt_xyz_v1(top_pos)}")
    print(f"  stove_right_camera_pos={fmt_xyz_v1(right_pos)}")
    print(f"  stove_camera_vec_xy={fmt_xyz_v1(camera_vec)}")
    print(f"  stove_surface_normal_raw={fmt_xyz_v1(normal)}")
    return normal


def stove_right_axis_v1(surface_normal):
    axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), surface_normal)
    if float(np.linalg.norm(axis)) < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize_v1(axis)


def stove_approach_direction_from_candidate_v1(cand, surface_normal):
    z_spread = cand.get("world_z_p90_p10_m")
    if z_spread is not None and float(z_spread) <= STOVE_FLAT_KNOB_Z_P90_P10_M_V1:
        return np.array([0.0, 0.0, -1.0], dtype=float), "flat_z_down"
    return -normalize_v1(np.asarray(surface_normal, dtype=float)), "surface_normal"


def select_stove_knob_candidates_by_target_v1(candidates, target_knob, right_axis):
    ordered = sorted(
        candidates,
        key=lambda c: float(np.dot(np.asarray(c["pos"], dtype=float), right_axis)),
    )
    for rank, cand in enumerate(ordered):
        cand["rank_left_to_right"] = rank

    n_candidates = len(ordered)
    if n_candidates == 0:
        raise RuntimeError("no stove knob candidates")

    matches = stove_target_order_matches_v1(n_candidates, target_knob)
    selected_by_index = {}
    selected_candidates = []
    for match in matches:
        rank = int(np.clip(match["rank"], 0, n_candidates - 1))
        cand = ordered[rank]
        index = int(cand["index"])
        if index not in selected_by_index:
            selected_by_index[index] = cand
            selected_candidates.append(cand)
            cand["possible_order_matches"] = []
        selected_by_index[index]["possible_order_matches"].append(match)

    for attempt_rank, cand in enumerate(selected_candidates):
        cand["target_attempt_rank"] = attempt_rank
        matches_for_candidate = cand.get("possible_order_matches", [])
        first_match = matches_for_candidate[0] if matches_for_candidate else {}
        cand["expected_order"] = tuple(
            first_match.get("order")
            or stove_expected_order_v1(n_candidates, target_knob)
        )
        reason_parts = []
        for match in matches_for_candidate:
            detail = match.get("reason")
            if not detail:
                detail = f"via {list(match['order'])}"
            reason_parts.append(
                f"rank={int(match['rank'])} {detail}"
            )
        cand["selection_reason"] = (
            f"candidate {attempt_rank + 1}/{len(selected_candidates)} for "
            f"{target_knob}: " + "; ".join(reason_parts)
        )
    return selected_candidates


def select_stove_knob_by_target_v1(candidates, target_knob, right_axis):
    return select_stove_knob_candidates_by_target_v1(
        candidates,
        target_knob,
        right_axis,
    )[0]


def stove_target_order_matches_v1(n_candidates, target_knob):
    indices = stove_target_index_candidates_v1(n_candidates, target_knob)
    matches = [
        {
            "rank": int(rank),
            "order": (),
            "reason": (
                "from signed rank list "
                f"indices={list(indices)}"
            ),
        }
        for rank in indices
    ]

    if matches:
        return matches
    if target_knob == "left":
        return [{"rank": 0, "order": ("left", "right")}]
    if target_knob == "right":
        return [{"rank": n_candidates - 1, "order": ("left", "right")}]
    return [{"rank": n_candidates // 2, "order": (target_knob,)}]


def stove_target_index_candidates_v1(n_candidates, target_knob):
    if n_candidates <= 0:
        return ()
    if target_knob == "left":
        return (0,)
    if target_knob == "right":
        return (n_candidates - 1,)

    signed_ranks = STOVE_TARGET_SIGNED_RANK_CANDIDATES_V1.get(target_knob, ())
    valid = tuple(
        dict.fromkeys(
            idx
            for rank in signed_ranks
            for idx in [signed_stove_rank_to_index_v1(rank, n_candidates)]
            if idx is not None
        )
    )
    if valid:
        return valid

    fallback = []
    for order in stove_possible_orders_v1(n_candidates, target_knob):
        if target_knob in order:
            fallback.append(order.index(target_knob))
    return tuple(
        dict.fromkeys(
            int(i)
            for i in fallback
            if 0 <= int(i) < int(n_candidates)
        )
    )


def signed_stove_rank_to_index_v1(signed_rank, n_candidates):
    signed_rank = int(signed_rank)
    n_candidates = int(n_candidates)
    if signed_rank > 0:
        idx = signed_rank - 1
    elif signed_rank < 0:
        idx = n_candidates + signed_rank
    else:
        return None
    if 0 <= idx < n_candidates:
        return int(idx)
    return None


def stove_possible_orders_v1(n_candidates, target_knob):
    real_orders = [
        order
        for order in STOVE_REAL_ORDERS_V1
        if len(order) == n_candidates and target_knob in order
    ]
    if real_orders:
        return tuple(real_orders)

    fallback_orders = []
    for order in STOVE_INDEX_ORDERS_V1.values():
        if len(order) == n_candidates and target_knob in order:
            fallback_orders.append(order)
    if target_knob in {"front_left", "rear_left", "rear_right", "front_right"}:
        for order in STOVE_REAL_ORDERS_V1:
            if target_knob in order:
                rank = order.index(target_knob)
                if 0 <= rank < n_candidates:
                    fallback_orders.append(order)
    if fallback_orders:
        unique = []
        seen = set()
        for order in fallback_orders:
            key = tuple(order)
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return tuple(unique)
    return (stove_expected_order_v1(n_candidates, target_knob),)


def stove_expected_order_v1(n_candidates, target_knob):
    if target_knob in {"front_center", "rear_center"}:
        if n_candidates >= 6:
            return STOVE_INDEX_ORDERS_V1["6_standard"]
        return STOVE_INDEX_ORDERS_V1["5_center"]
    if target_knob == "center":
        if n_candidates >= 7:
            return STOVE_INDEX_ORDERS_V1["7_aux_left"]
        return STOVE_INDEX_ORDERS_V1["5_standard"]
    if target_knob in {"front_left", "rear_left", "rear_right", "front_right"}:
        if n_candidates >= 7:
            return STOVE_INDEX_ORDERS_V1["7_aux_left"]
        if n_candidates >= 6:
            return STOVE_INDEX_ORDERS_V1["6_standard"]
        if n_candidates == 5:
            return STOVE_INDEX_ORDERS_V1["5_standard"]
        return STOVE_INDEX_ORDERS_V1["4_standard"]
    if n_candidates >= 6:
        return STOVE_INDEX_ORDERS_V1["6_standard"]
    if n_candidates == 5:
        return STOVE_INDEX_ORDERS_V1["5_standard"]
    if n_candidates == 4:
        return STOVE_INDEX_ORDERS_V1["4_standard"]
    if n_candidates == 3:
        return ("left", "center", "right")
    if n_candidates == 2:
        return ("left", "right")
    return (target_knob,)


def prepare_stove_pose_from_selected_v1(stove_state):
    selected = stove_state["selected_candidate"]
    knob_pos = np.asarray(selected["pos"], dtype=float)
    surface_normal = stove_camera_surface_normal_v1(knob_pos)
    right_axis = stove_right_axis_v1(surface_normal)
    approach_direction, approach_mode = stove_approach_direction_from_candidate_v1(
        selected,
        surface_normal,
    )
    quat = make_gripper_z_quat_v1(approach_direction)
    stove_state["surface_normal"] = surface_normal
    stove_state["right_axis"] = right_axis
    stove_state["approach_direction"] = approach_direction
    stove_state["approach_mode"] = approach_mode
    stove_state["approach_roll_deg"] = 0.0
    stove_state["quat"] = quat
    stove_state["hover_pos"] = knob_pos - approach_direction * 0.10
    return stove_state


def print_stove_detection_summary_v1(stove_state):
    selected = stove_state["selected_candidate"]
    group = stove_state["selected_group"]
    gap_info = stove_state.get("gap_prediction") or {}
    expected_order = selected.get("expected_order", ())
    target_candidates = list(stove_state.get("target_candidates") or [selected])
    print(
        "  Detected stove knobs: "
        f"group={group['camera']}/{group['query']} "
        f"n={len(stove_state['candidates'])}/{int(group.get('raw_count', 0))} "
        f"pred={len(gap_info.get('predicted') or [])} "
        f"selected={int(selected['index'])}"
    )
    print(f"  target_knob={stove_state['target_knob']}")
    print(f"  fallback_left_to_right_order={list(expected_order)}")
    print(
        "  possible_target_candidates="
        + ", ".join(
            (
                f"{attempt + 1}:{int(c['index'])}"
                f"(rank={c.get('rank_left_to_right')})"
            )
            for attempt, c in enumerate(target_candidates)
        )
    )
    print(f"  surface_normal={fmt_xyz_v1(stove_state['surface_normal'])}")
    print(f"  right_axis={fmt_xyz_v1(stove_state['right_axis'])}")
    print(
        "  selected: "
        f"index={int(selected['index'])} "
        f"camera={selected['camera']} score={float(selected['score']):.3f} "
        f"rank={selected.get('rank_left_to_right')} "
        f"predicted={bool(selected.get('predicted', False))} "
        f"pos={fmt_xyz_v1(selected['pos'])}"
    )
    print(
        "  selected_z_p90_p10_m="
        f"{float(selected.get('world_z_p90_p10_m', float('nan'))):.3f}"
    )
    print(
        "  selected_pos_mode="
        f"{selected.get('pos_mode', 'mask_median')} "
        f"z_band_n={int(selected.get('z_band_n', 0))}"
    )
    print(f"  selection_reason={selected.get('selection_reason', '')}")
    print(
        f"  approach_direction={fmt_xyz_v1(stove_state['approach_direction'])} "
        f"mode={stove_state['approach_mode']}"
    )
    if gap_info.get("nominal_gap_m") is not None:
        print(f"  nominal_knob_gap_m={float(gap_info['nominal_gap_m']):.3f}")
    for cand in stove_state["candidates"]:
        print(
            "  Candidate "
            f"[{int(cand['index'])}] "
            f"{cand['camera']} score={float(cand['score']):.3f} "
            f"rank={cand.get('rank_left_to_right')} "
            f"pred={bool(cand.get('predicted', False))} "
            f"pos={fmt_xyz_v1(cand['pos'])}"
        )


def show_stove_debug_markers_v1(stove_state):
    set_debug_markers_fn = globals().get("set_debug_markers")
    if not callable(set_debug_markers_fn):
        return False

    markers = []
    selected = stove_state.get("selected_candidate")
    selected_idx = None if selected is None else int(selected["index"])
    target_candidate_indices = {
        int(c["index"]) for c in stove_state.get("target_candidates", [])
    }
    for cand in stove_state.get("candidates", []):
        idx = int(cand["index"])
        color = (255, 215, 64) if cand.get("predicted") else (80, 180, 255)
        radius = 0.012 if cand.get("predicted") else 0.014
        if idx in target_candidate_indices:
            color = (255, 64, 220)
            radius = 0.018
        if selected_idx is not None and idx == selected_idx:
            color = (255, 80, 80)
            radius = 0.020
        target_text = " target" if idx in target_candidate_indices else ""
        markers.append(
            {
                "name": f"stove_knob_{idx}",
                "label": (
                    f"{idx}: {cand['camera']} "
                    f"score={float(cand['score']):.2f} "
                    f"rank={cand.get('rank_left_to_right')}"
                    f"{target_text}"
                ),
                "position": [float(x) for x in np.asarray(cand["pos"], dtype=float)],
                "color": list(color),
                "radius_m": radius,
                "alpha": 0.80,
            }
        )

    hover_pos = stove_state.get("hover_pos")
    if hover_pos is not None:
        markers.append(
            {
                "name": "stove_knob_hover",
                "label": "hover",
                "position": [float(x) for x in np.asarray(hover_pos, dtype=float)],
                "color": [255, 160, 64],
                "radius_m": 0.018,
                "alpha": 0.80,
            }
        )

    try:
        set_debug_markers_fn(markers)
        print(f"debug_markers: updated {len(markers)} marker(s)")
        return True
    except Exception as exc:
        print(f"debug_markers: failed to update ({exc})")
        return False

