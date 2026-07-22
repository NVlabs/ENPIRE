# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from skill_library.namespace import *  # noqa: F401,F403
except AttributeError:
    pass
from enpire.env.forge.cap.agent.skill_registry import skill  # noqa: F401

import collections
import math

import numpy as np

try:
    from skill_library.constants.vision import (  # type: ignore
        ANYGRASP_DISABLE_PLANNER_Z_CLIPPING,
        ANYGRASP_TCP_OFFSET_Z_M,
    )
except Exception:
    ANYGRASP_TCP_OFFSET_Z_M = 0.0
    ANYGRASP_DISABLE_PLANNER_Z_CLIPPING = True

try:
    from skill_library.constants.manipulation import GRIPPER_WIDTH_M  # type: ignore
except Exception:
    GRIPPER_WIDTH_M = 0.08

TWO_D_TOP_DOWN_Z_M = 0.79
SAFE_REL_TOL = 0.12
SAFE_ABS_TOL_M = 0.006


def _tool(name):
    fn = globals().get(name)
    if fn is not None:
        return fn
    import skill_library.namespace as namespace

    fn = getattr(namespace, name, None)
    if fn is None:
        raise RuntimeError(f"Required tool is not available in skill_library.namespace: {name}")
    return fn


def _as_grasp(position, rpy, score=1.0, width=None, trajectory_cache_key=None):
    Grasp = collections.namedtuple(
        "GraspCandidate", ["position", "rpy", "score", "width", "trajectory_cache_key"]
    )
    return Grasp(
        position=[float(x) for x in position],
        rpy=[float(x) for x in rpy],
        score=float(score),
        width=float(GRIPPER_WIDTH_M if width is None else width),
        trajectory_cache_key=trajectory_cache_key,
    )


def _clip_grasp_min_z(grasp, min_z):
    clipped_z = max(float(grasp.position[2]), float(min_z))
    if clipped_z <= float(grasp.position[2]):
        return grasp
    position = [float(x) for x in grasp.position]
    original_z = position[2]
    position[2] = clipped_z
    print(f"  Clipping AnyGrasp z from {original_z:.4f}m to {clipped_z:.4f}m")
    return _as_grasp(
        position,
        getattr(grasp, "rpy", [0.0, 180.0, 0.0]),
        score=getattr(grasp, "score", 1.0),
        width=getattr(grasp, "width", GRIPPER_WIDTH_M),
        trajectory_cache_key=getattr(grasp, "trajectory_cache_key", None),
    )


def _display_rpy_to_rotation(rpy):
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    return Rotation.from_euler("xyz", [-pitch, roll, -yaw - 90.0], degrees=True)


def _rotation_to_display_rpy(rot):
    ex, ey, ez = rot.as_euler("xyz", degrees=True)
    disp = np.array([ey, -ex, -ez - 90.0], dtype=float)
    return ((disp + 180.0) % 360.0 - 180.0).tolist()


def _grasp_rpy(grasp):
    if isinstance(grasp, dict):
        return grasp.get("rpy")
    return getattr(grasp, "rpy", None)


def _copy_grasp_with_rpy(grasp, rpy):
    if isinstance(grasp, dict):
        out = dict(grasp)
        out["rpy"] = [float(x) for x in rpy]
        return out
    return _as_grasp(
        getattr(grasp, "position"),
        rpy,
        score=getattr(grasp, "score", 1.0),
        width=getattr(grasp, "width", GRIPPER_WIDTH_M),
        trajectory_cache_key=getattr(grasp, "trajectory_cache_key", None),
    )


def _wrist_camera_y_dot_from_rpy(rpy):
    rot = _display_rpy_to_rotation(rpy)
    return float(rot.as_matrix()[1, 1])


def _yaw_flip_rpy_for_wrist_camera(rpy):
    from scipy.spatial.transform import Rotation

    rot = _display_rpy_to_rotation(rpy)
    flipped = rot * Rotation.from_euler("z", 180.0, degrees=True)
    return _rotation_to_display_rpy(flipped)


def filter_anygrasp_wrist_camera_y(
    grasps,
    *,
    threshold=0.0,
    allow_yaw_flip=True,
    log=True,
):
    """Keep AnyGrasp poses whose wrist-camera/local +Y axis points toward world +Y."""
    if not grasps:
        return []

    kept = []
    n_flipped = 0
    n_filtered = 0
    n_unchecked = 0
    threshold = float(threshold)
    for grasp in grasps:
        rpy = _grasp_rpy(grasp)
        if rpy is None:
            kept.append(grasp)
            n_unchecked += 1
            continue
        y_dot = _wrist_camera_y_dot_from_rpy(rpy)
        if y_dot >= threshold:
            kept.append(grasp)
            continue
        if allow_yaw_flip:
            flipped_rpy = _yaw_flip_rpy_for_wrist_camera(rpy)
            flipped_y_dot = _wrist_camera_y_dot_from_rpy(flipped_rpy)
            if flipped_y_dot >= threshold:
                kept.append(_copy_grasp_with_rpy(grasp, flipped_rpy))
                n_flipped += 1
                continue
        n_filtered += 1

    if log and (n_flipped or n_filtered or n_unchecked):
        print(
            "  AnyGrasp wrist-camera +Y filter: "
            f"kept={len(kept)}/{len(grasps)} yaw_flipped={n_flipped} "
            f"filtered={n_filtered} unchecked={n_unchecked} threshold={threshold:.3f}"
        )
    return kept


def sample_anygrasp(
    object_name,
    camera="top",
    max_grasps=None,
    tcp_offset_z_m=None,
    disable_planner_z_clipping=None,
    clip_min_z=None,
    filter_wrist_camera_y=True,
    wrist_camera_y_threshold=0.0,
    allow_wrist_camera_yaw_flip=True,
    image_bbox=None,
):
    kwargs = {
        "object_name": object_name,
        "camera": camera,
        "object_input_mode": "segmented_object_cloud",
    }
    if max_grasps is not None:
        kwargs["max_grasps"] = int(max_grasps)
    if image_bbox is not None:
        kwargs["image_bbox"] = list(image_bbox)
    tcp_offset = ANYGRASP_TCP_OFFSET_Z_M if tcp_offset_z_m is None else tcp_offset_z_m
    disable_clip = (
        ANYGRASP_DISABLE_PLANNER_Z_CLIPPING
        if disable_planner_z_clipping is None
        else disable_planner_z_clipping
    )
    try:
        grasps = _tool("sample_grasp_pose_anygrasp")(
            **kwargs,
            tcp_offset_z_m=tcp_offset,
            disable_planner_z_clipping=disable_clip,
            filter_wrist_camera_y=bool(filter_wrist_camera_y),
            wrist_camera_y_threshold=float(wrist_camera_y_threshold),
            allow_wrist_camera_yaw_flip=bool(allow_wrist_camera_yaw_flip),
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        grasps = _tool("sample_grasp_pose_anygrasp")(**kwargs)
    except Exception as exc:
        print(f"  AnyGrasp query {object_name!r} failed: {exc}")
        return []
    if filter_wrist_camera_y:
        try:
            grasps = filter_anygrasp_wrist_camera_y(
                grasps,
                threshold=wrist_camera_y_threshold,
                allow_yaw_flip=allow_wrist_camera_yaw_flip,
            )
        except Exception as exc:
            print(f"  AnyGrasp wrist-camera +Y filter skipped: {exc}")
    if clip_min_z is None:
        return grasps
    return [_clip_grasp_min_z(grasp, clip_min_z) for grasp in grasps]


def sample_obb(
    object_name,
    camera="top",
    queries=None,
    tcp_offset_z_m=0.0,
    relax=False,
    relax_xyz_offsets_m=None,
    relax_yaw_offsets_deg=None,
    image_bbox=None,
    min_world_z=None,
    max_world_z=None,
    candidate_filter_fn=None,
):
    last_error = None
    filtered_queries = []
    for query in list(queries or [object_name]):
        try:
            grasps = _tool("sample_grasp_pose_3d_bb")(
                object_name=query,
                camera=camera,
                tcp_offset_z_m=tcp_offset_z_m,
                relax=bool(relax),
                relax_xyz_offsets_m=relax_xyz_offsets_m,
                relax_yaw_offsets_deg=relax_yaw_offsets_deg,
                image_bbox=list(image_bbox) if image_bbox is not None else None,
                min_world_z=min_world_z,
                max_world_z=max_world_z,
            )
            if grasps and callable(candidate_filter_fn):
                try:
                    original_count = len(grasps)
                    grasps = candidate_filter_fn(
                        grasps,
                        query=query,
                        object_name=object_name,
                        camera=camera,
                    )
                except TypeError:
                    original_count = len(grasps)
                    grasps = candidate_filter_fn(grasps)
                if original_count > 0 and not grasps:
                    filtered_queries.append(str(query))
            if grasps:
                return grasps
        except Exception as exc:
            last_error = exc
            print(f"  3D-BB query {query!r} failed: {exc}")
    if filtered_queries:
        filtered_text = ", ".join(repr(q) for q in filtered_queries)
        if last_error is not None:
            raise RuntimeError(
                f"all 3D-BB queries failed or were filtered out for {object_name!r}; "
                f"filtered_queries=[{filtered_text}]; last_error={last_error}"
            )
        raise RuntimeError(
            f"all 3D-BB queries were filtered out for {object_name!r}; "
            f"filtered_queries=[{filtered_text}]"
        )
    if last_error is not None:
        raise RuntimeError(
            f"all 3D-BB queries failed for {object_name!r}; last_error={last_error}"
        )
    return []


def sample_2d(
    object_name,
    camera="top",
    grasp_z_m=None,
    max_grasps=10,
    return_debug=True,
    reuse_cached_frame=False,
    image_bbox=None,
):
    return _tool("sample_grasp_pose_2d")(
        object_name=object_name,
        camera=camera,
        max_grasps=int(max_grasps),
        grasp_z_m=grasp_z_m,
        return_debug=bool(return_debug),
        reuse_cached_frame=bool(reuse_cached_frame),
        image_bbox=list(image_bbox) if image_bbox is not None else None,
    )


def _normalize_angle_deg(angle_deg):
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _project_to_segment_2d(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    t = np.clip(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    foot = a + t * ab
    return foot, float(np.linalg.norm(point - foot))


def _camera_matrices(camera):
    params_tool = globals().get("get_camera_params")
    if params_tool is not None:
        params = params_tool(camera=camera)
        return np.asarray(params["K"], dtype=float), np.asarray(params["T_cam_world"], dtype=float)
    K = np.asarray(_tool("get_camera_intrinsics")(camera=camera), dtype=float)
    T = np.asarray(_tool("get_camera_extrinsics")(camera=camera), dtype=float)
    return K, T


def _project_pixel_to_plane_world(pixel_xy, cam_K, T_cam_world, plane_z_m):
    u, v = [float(x) for x in pixel_xy]
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    origin_world = np.asarray(T_cam_world[:3, 3], dtype=float)
    ray_world = np.asarray(T_cam_world[:3, :3], dtype=float) @ ray_cam
    dz = float(ray_world[2])
    if abs(dz) < 1e-9:
        return None
    scale = (float(plane_z_m) - float(origin_world[2])) / dz
    if scale <= 0.0:
        return None
    return origin_world + scale * ray_world


def detect_part_yaw(part_query, body_query, camera, plane_z_m=None, publish_debug=True):
    part_out = sample_2d(part_query, camera=camera, max_grasps=10, return_debug=True)
    body_out = sample_2d(
        body_query,
        camera=camera,
        max_grasps=4,
        return_debug=True,
        reuse_cached_frame=True,
    )
    part_grasps = part_out.get("grasps", part_out) if isinstance(part_out, dict) else part_out
    body_grasps = body_out.get("grasps", body_out) if isinstance(body_out, dict) else body_out
    if not part_grasps or not body_grasps:
        return None

    part_debug = part_out.get("debug", {}) if isinstance(part_out, dict) else {}
    body_debug = body_out.get("debug", {}) if isinstance(body_out, dict) else {}
    corners = body_debug.get("bbox_corners_px")
    body_center = body_debug.get("bbox_center_px")
    part_center = part_debug.get("bbox_center_px")
    if corners is None or body_center is None or part_center is None:
        return None

    body_center = np.asarray(body_center, dtype=float)
    part_center = np.asarray(part_center, dtype=float)
    corners = [np.asarray(corner, dtype=float) for corner in corners]
    part_vec = part_center - body_center
    part_norm = float(np.linalg.norm(part_vec))
    edge_dirs = []
    for i, j in ((0, 1), (1, 2)):
        edge = corners[j] - corners[i]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm > 1e-6:
            edge_dirs.append(edge / edge_norm)
    if not edge_dirs:
        return None
    ref_dir = min(edge_dirs, key=lambda d: abs(float(np.dot(d, part_vec / part_norm)))) if part_norm > 1e-6 else edge_dirs[0]
    ts = [float(np.dot(corner - body_center, ref_dir)) for corner in corners]
    ref_a = body_center + min(ts) * ref_dir
    ref_b = body_center + max(ts) * ref_dir
    intersection, _ = _project_to_segment_2d(part_center, ref_a, ref_b)

    cam_K, T_cam_world = _camera_matrices(camera)
    if plane_z_m is None:
        first_body = body_grasps[0]
        body_position = getattr(first_body, "position", None)
        if body_position is None and isinstance(first_body, dict):
            body_position = first_body.get("position")
        if body_position is None:
            return None
        plane_z_m = body_position[2]
    foot_world = _project_pixel_to_plane_world(intersection, cam_K, T_cam_world, plane_z_m)
    part_world = _project_pixel_to_plane_world(part_center, cam_K, T_cam_world, plane_z_m)
    if foot_world is None or part_world is None:
        return None
    axis = np.asarray(part_world[:2], dtype=float) - np.asarray(foot_world[:2], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None
    axis = axis / norm
    yaw = float(math.degrees(math.atan2(float(axis[1]), float(axis[0]))))
    print(f"  part yaw {part_query!r} relative to {body_query!r}: {yaw:.1f} deg")
    return yaw


def detect_signed_axis(body_query, sign_query, camera="top"):
    yaw = detect_part_yaw(sign_query, body_query, camera)
    if yaw is None:
        return None
    rad = math.radians(yaw)
    return np.array([math.cos(rad), math.sin(rad)], dtype=float)


def topdown_candidates(
    center_xyz,
    yaw_deg,
    width=None,
    xy_offsets=None,
    yaw_offsets=(0.0, 180.0),
    score=1.0,
):
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    offsets = xy_offsets if xy_offsets is not None else [(0.0, 0.0)]
    candidates = []
    for xy_offset in offsets:
        offset = np.asarray(xy_offset, dtype=float).reshape(2)
        for yaw_offset in yaw_offsets:
            yaw = _normalize_angle_deg(float(yaw_deg) + float(yaw_offset))
            candidates.append(
                _as_grasp(
                    [center[0] + offset[0], center[1] + offset[1], center[2]],
                    [0.0, 180.0, yaw],
                    score=score,
                    width=width,
                )
            )
    return candidates


def tilted_axis_candidates(
    ref_xy,
    axis_xy,
    half_len,
    fractions,
    tilt_dir,
    tilt_deg,
    grasp_z,
    score_fn=None,
    width=None,
):
    ref = np.asarray(ref_xy, dtype=float).reshape(2)
    axis = np.asarray(axis_xy, dtype=float).reshape(2)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    tilt = np.asarray(tilt_dir, dtype=float).reshape(2)
    tilt = tilt / max(float(np.linalg.norm(tilt)), 1e-9)
    yaw = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    roll = float(tilt_deg) * float(np.sign(np.cross(axis, tilt)))
    candidates = []
    for index, fraction in enumerate(fractions):
        xy = ref + axis * float(half_len) * float(fraction)
        score = score_fn(fraction, index) if score_fn is not None else 1.0
        candidates.append(_as_grasp([xy[0], xy[1], grasp_z], [roll, 180.0 - abs(roll), yaw], score=score, width=width))
    return candidates


def detect_obb_safe_state(object_name, camera, queries=None, rel_tol=None, abs_tol_m=None):
    grasps = sample_obb(object_name, camera=camera, queries=queries)
    if not grasps:
        raise RuntimeError(f"{camera} 3D-BB could not detect {object_name!r}")
    grasp = grasps[0]
    bbox = getattr(grasp, "bbox_result", None)
    if bbox is None:
        raise RuntimeError("3D-BB result missing bbox_result")

    extents = np.asarray(bbox.obb_extents, dtype=float)
    normal = np.asarray(bbox.top_normal_world, dtype=float)
    standing_extent = float(bbox.top_normal_extent)
    min_extent = float(np.min(extents))
    tol = max(float(abs_tol_m if abs_tol_m is not None else SAFE_ABS_TOL_M), float(rel_tol if rel_tol is not None else SAFE_REL_TOL) * min_extent)
    is_safe = standing_extent <= min_extent + tol
    return {
        "ok": True,
        "state": "safe_short_side_up" if is_safe else "not_safe_not_short_side_up",
        "is_safe": bool(is_safe),
        "side": camera,
        "query": object_name,
        "grasp": grasp,
        "bbox": bbox,
        "metrics": {
            "obb_extents_m": [float(v) for v in extents],
            "standing_axis_extent_m": standing_extent,
            "shortest_extent_m": min_extent,
            "safe_tolerance_m": tol,
            "top_normal_world": [float(v) for v in normal],
            "z_alignment": float(abs(normal @ np.array([0.0, 0.0, 1.0])) / max(float(np.linalg.norm(normal)), 1e-9)),
        },
    }
