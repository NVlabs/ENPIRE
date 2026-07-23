# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generic RoboCasa vision helpers.
import numpy as np
from skill_library.namespace import *  # noqa: F401, F403


def detect_object_v1(object_query, cameras="top", *, required=True):
    """Detect prompts from one camera or rank same-prompt hits across cameras."""
    prompts = _prompt_list(object_query)
    cameras = _camera_order(cameras)
    last_error = None

    if len(cameras) == 1:
        camera = cameras[0]
        print("\n--- Detect object ---")
        for prompt in prompts:
            try:
                det = _detect_one(prompt, camera)
            except Exception as exc:
                last_error = exc
                print(f"  {camera} {prompt!r}: {type(exc).__name__}: {exc}")
                continue
            if det is None:
                print(f"  {camera} {prompt!r}: no detection")
                continue
            print(
                f"  Selected: query={det['query']!r} camera={det['camera']} "
                f"score={det['score']:.3f} pos={_fmt(det['pos'])}"
            )
            return det
        if required:
            raise RuntimeError(
                f"could not detect any of {tuple(prompts)!r} from {camera!r}: "
                f"{last_error}"
            )
        return None

    print("\n--- Detect object rank ---")
    for prompt in prompts:
        print(f"  Prompt {prompt!r}")
        best = None
        for camera in cameras:
            try:
                det = _detect_one(prompt, camera)
            except Exception as exc:
                last_error = exc
                print(f"    {camera}: {type(exc).__name__}: {exc}")
                continue
            if det is None:
                print(f"    {camera}: no detection")
                continue
            print(f"    {camera}: pos={_fmt(det['pos'])} score={det['score']:.3f}")
            if best is None or float(det["score"]) > float(best["score"]):
                best = det
        if best is not None:
            print(
                "  Selected: "
                f"query={best['query']!r} camera={best['camera']} "
                f"score={best['score']:.3f} pos={_fmt(best['pos'])}"
            )
            return best

    if required:
        raise RuntimeError(
            f"could not detect any of {tuple(prompts)!r} from {tuple(cameras)!r}: "
            f"{last_error}"
        )
    return None


def detect_sam3_depth_candidates_v1(
    query_list,
    cameras,
    *,
    score_threshold=0.3,
    max_results=25,
    min_depth_m=0.1,
    min_valid_points=20,
):
    results = []
    for camera in _camera_order(cameras):
        depth = np.asarray(get_camera_depth(camera), dtype=np.float32)
        transform, fx, fy, cx, cy = _camera_transform(camera)
        for query in _prompt_list(query_list):
            try:
                segmentations = segment_object_all(
                    query,
                    camera=camera,
                    score_thresh=score_threshold,
                    max_results=max_results,
                )
            except Exception as exc:
                print(f"  SAM3 failed for {query!r} from {camera}: {exc}")
                continue
            for instance_idx, seg in enumerate(segmentations):
                pos, n_valid = _backproject_mask(
                    seg["mask"],
                    depth,
                    transform,
                    fx,
                    fy,
                    cx,
                    cy,
                    min_depth_m=min_depth_m,
                    min_valid_points=min_valid_points,
                )
                if pos is None:
                    print(
                        f"  Skip {query!r}#{instance_idx} from {camera}: "
                        f"only {n_valid} valid depth points"
                    )
                    continue
                results.append(
                    {
                        "query": query,
                        "camera": camera,
                        "instance_idx": instance_idx,
                        "pos": np.asarray(pos, dtype=float),
                        "score": float(seg["score"]),
                        "bbox_xywh": seg["bbox_xywh"],
                    }
                )

    results.sort(key=lambda det: det["score"], reverse=True)
    if not results:
        raise RuntimeError(
            f"Could not detect any of {query_list} with score >= "
            f"{score_threshold:.3f}"
        )
    print(f"  Found {len(results)} raw candidate(s); selecting one target")
    return results


def debug_marker_v1(name, pos, color, radius=0.03):
    return {
        "name": name,
        "label": name,
        "position": [float(x) for x in pos],
        "color": list(color),
        "radius_m": float(radius),
        "alpha": 0.75,
    }


def show_debug_markers_v1(markers):
    set_debug_markers_fn = globals().get("set_debug_markers")
    if not callable(set_debug_markers_fn):
        return False
    try:
        set_debug_markers_fn(markers)
        print(f"debug_markers: updated {len(markers)} marker(s)")
        return True
    except Exception as exc:
        print(f"debug_markers: failed to update ({exc})")
        return False


def filter_candidates_near_v1(detections, center_pos, max_dist_m, label="target"):
    center = np.asarray(center_pos, dtype=float)
    kept = []
    for det in detections:
        pos = np.asarray(det["pos"], dtype=float)
        dist = float(np.linalg.norm(pos - center))
        if dist <= max_dist_m:
            item = dict(det)
            item["filter_distance_m"] = dist
            kept.append(item)
    if not kept:
        raise RuntimeError(
            f"No candidates within {max_dist_m:.3f} m of {label} "
            f"at {[round(float(x), 3) for x in center]}"
        )
    print(
        f"  Kept {len(kept)}/{len(detections)} candidate(s) within "
        f"{max_dist_m:.3f} m of {label}"
    )
    return kept


def select_highest_score_near_v1(detections, center_pos, max_dist_m, label="target"):
    kept = filter_candidates_near_v1(
        detections,
        center_pos,
        max_dist_m,
        label=label,
    )
    selected_idx, selected = max(
        enumerate(kept),
        key=lambda item: float(item[1]["score"]),
    )
    print(
        f"  Selected highest-score near {label} [{selected_idx}] "
        f"{selected['query']!r} from {selected['camera']} "
        f"score={selected['score']:.3f} "
        f"dist={selected['filter_distance_m']:.3f} "
        f"pos={[round(float(x), 3) for x in selected['pos']]}"
    )
    return selected, kept


def rank_lowest_z_candidates_v1(detections):
    positions = [np.asarray(det["pos"], dtype=float) for det in detections]
    return sorted(
        range(len(detections)),
        key=lambda idx: (
            float(positions[idx][2]),
            -float(detections[idx]["score"]),
        ),
    )


def select_candidate_v1(detections, candidate_order):
    selected_idx = int(candidate_order[0])
    det = detections[selected_idx]
    print(
        f"  Selected one lowest-z target [{selected_idx}] {det['query']!r} "
        f"from {det['camera']} score={det['score']:.3f} "
        f"pos={[round(float(x), 3) for x in det['pos']]}"
    )
    return det


def _detect_one(query, camera):
    result = detect_objects_oneshot(query, camera=camera)
    detections = result.get(query, [])
    if not detections:
        return None
    det = detections[0]
    return {
        "query": query,
        "camera": camera,
        "pos": np.asarray(det.position_3d, dtype=float),
        "score": float(getattr(det, "score", 0.0)),
    }


def _prompt_list(query):
    raw_prompts = query if isinstance(query, (list, tuple)) else [query]
    prompts = []
    for q in raw_prompts:
        q = str(q or "").strip()
        if q and q not in prompts:
            prompts.append(q)
    if not prompts:
        raise RuntimeError(f"no specific object prompts from query {query!r}")
    return prompts


def _camera_order(cameras):
    raw = [cameras] if isinstance(cameras, str) else list(cameras or ())
    ordered = []
    for camera in raw:
        camera = str(camera).strip()
        if camera and camera not in ordered:
            ordered.append(camera)
    return tuple(ordered or ("top", "wrist", "right"))


def _camera_transform(camera):
    fx, fy, cx, cy = [float(x) for x in get_camera_intrinsics(camera)]
    extr = get_camera_extrinsics(camera)
    rot = np.asarray(extr["rotation"], dtype=float).reshape(3, 3)
    pos = np.asarray(extr["position"], dtype=float)
    transform = np.eye(4, dtype=float)
    if extr.get("needs_optical_flip", True):
        transform[:3, :3] = rot @ np.diag([-1.0, -1.0, 1.0])
    else:
        transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform, fx, fy, cx, cy


def _backproject_mask(
    mask,
    depth,
    transform,
    fx,
    fy,
    cx,
    cy,
    *,
    min_depth_m,
    min_valid_points,
):
    valid = (mask > 0) & np.isfinite(depth) & (depth > min_depth_m)
    n_valid = int(valid.sum())
    if n_valid < min_valid_points:
        return None, n_valid
    vs, us = np.where(valid)
    zs = depth[valid].astype(float)
    xs = (us.astype(float) - cx) * zs / fx
    ys = (vs.astype(float) - cy) * zs / fy
    pts_cam = np.stack([xs, ys, zs], axis=1)
    centroid_cam = np.median(pts_cam, axis=0)
    return transform[:3, :3] @ centroid_cam + transform[:3, 3], n_valid


def _fmt(pos):
    return [round(float(x), 3) for x in np.asarray(pos, dtype=float)]
