# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# stand_mixer_motion.py — shared OpenStandMixerHead motion helpers
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np
from scipy.spatial.transform import Rotation as R


def refresh_stand_mixer_planner_world_v1(reason):
    world_info = update_planner_world()
    print(f"[curobo] {reason}: {world_info['n_obstacles']} obstacles")
    return world_info


def compute_stand_mixer_geometry_v1(
    mixer_det,
    *,
    offset_candidates_m=(0.20, 0.25, 0.30),
):
    mixer_pos = np.asarray(mixer_det["pos"], dtype=float)
    normal_info = surface_normal_from_cameras_v1()
    surface_normal = np.asarray(normal_info["surface_normal"], dtype=float)
    tangent_direction = stand_mixer_tangent_direction_v1(surface_normal)
    offset_direction = stand_mixer_offset_direction_v1(surface_normal)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    move_direction = _normalize(tangent_direction + world_up)
    if float(np.linalg.norm(move_direction)) < 1e-6:
        raise RuntimeError("stand mixer move direction is degenerate")
    home_quat = make_stand_mixer_home_quat_v1(surface_normal)
    offset_candidates = [
        (
            float(offset_m),
            mixer_pos + offset_direction * float(offset_m),
        )
        for offset_m in offset_candidates_m
    ]
    return {
        "mixer_pos": mixer_pos,
        "surface_normal": surface_normal,
        "tangent_direction": tangent_direction,
        "offset_direction": offset_direction,
        "move_direction": move_direction,
        "home_quat": home_quat,
        "offset_candidates": offset_candidates,
        "camera_vector_xy": normal_info["camera_vector_xy"],
        "top_camera_pos": normal_info["top_pos"],
        "right_camera_pos": normal_info["right_pos"],
    }


def surface_normal_from_cameras_v1():
    top_extr = get_camera_extrinsics("top")
    right_extr = get_camera_extrinsics("right")
    top_pos = np.asarray(top_extr["position"], dtype=float)
    right_pos = np.asarray(right_extr["position"], dtype=float)
    camera_vector_xy = right_pos - top_pos
    camera_vector_xy[2] = 0.0
    surface_normal = _xy_cw_vertical(camera_vector_xy)
    if float(np.linalg.norm(surface_normal)) < 1e-6:
        raise RuntimeError("camera-derived surface normal is degenerate")
    return {
        "top_pos": top_pos,
        "right_pos": right_pos,
        "camera_vector_xy": camera_vector_xy,
        "surface_normal": surface_normal,
    }


def stand_mixer_tangent_direction_v1(surface_normal):
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    tangent = _normalize(np.cross(world_up, np.asarray(surface_normal, dtype=float)))
    if float(np.linalg.norm(tangent)) < 1e-6:
        raise RuntimeError("cross(z+, surface_normal) is degenerate")
    return tangent


def stand_mixer_offset_direction_v1(surface_normal):
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    offset_direction = _normalize(np.cross(np.asarray(surface_normal, dtype=float), world_up))
    if float(np.linalg.norm(offset_direction)) < 1e-6:
        raise RuntimeError("cross(surface_normal, z+) is degenerate")
    return offset_direction


def make_stand_mixer_home_quat_v1(surface_normal):
    z_axis = stand_mixer_tangent_direction_v1(surface_normal)
    x_axis = _normalize(-np.asarray(surface_normal, dtype=float))
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = _normalize(x_axis - z_axis * np.dot(x_axis, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return _normalize_quat(R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat())


def move_to_stand_mixer_offset_v1(
    *,
    side,
    mixer_det,
    offset_candidates_m,
):
    geometry = compute_stand_mixer_geometry_v1(
        mixer_det,
        offset_candidates_m=offset_candidates_m,
    )
    home_quat = np.asarray(geometry["home_quat"], dtype=float)
    offset_candidates = list(geometry["offset_candidates"])
    if not offset_candidates:
        raise RuntimeError("no stand mixer offset candidates")

    world_info = refresh_stand_mixer_planner_world_v1("mixer offset")
    last_status = None
    last_reason = ""
    for attempt_idx, (offset_m, target_pos) in enumerate(offset_candidates, start=1):
        result = freespace_move_world_pose_v1(
            side,
            target_pos,
            home_quat,
            auto_update_world=True,
        )
        last_status = result["status"]
        last_reason = result.get("reason", "")
        print(
            f"mixer_offset attempt={attempt_idx}/{len(offset_candidates)} "
            f"offset_m={float(offset_m):.3f} status={last_status} "
            f"target={[round(float(x), 3) for x in target_pos]} "
            f"quat={[round(float(x), 4) for x in home_quat]}"
        )
        if last_status == "Success":
            return {
                "geometry": geometry,
                "world_info": world_info,
                "offset_m": float(offset_m),
                "target_pos": np.asarray(target_pos, dtype=float),
                "target_quat": home_quat,
                "attempt_idx": attempt_idx,
                "n_candidates": len(offset_candidates),
            }

    raise RuntimeError(
        f"failed {len(offset_candidates)} mixer offset candidates "
        f"(last_status={last_status}, reason={last_reason})"
    )


def nudge_stand_mixer_tangent_v1(
    *,
    side,
    motion_state,
    nudge_m,
    nudge_n_steps,
):
    geometry = motion_state["geometry"]
    tangent_direction = np.asarray(geometry["tangent_direction"], dtype=float)
    delta = tangent_direction * float(nudge_m)
    return _nudge_stand_mixer_v1(
        side=side,
        label="cross_z_normal",
        delta=delta,
        nudge_n_steps=nudge_n_steps,
    )


def nudge_stand_mixer_bisector_v1(
    *,
    side,
    motion_state,
    nudge_m,
    nudge_n_steps,
    repeat_idx=1,
    repeat_total=1,
):
    geometry = motion_state["geometry"]
    move_direction = np.asarray(geometry["move_direction"], dtype=float)
    delta = move_direction * float(nudge_m)
    return _nudge_stand_mixer_v1(
        side=side,
        label=f"bisector[{repeat_idx}/{repeat_total}]",
        delta=delta,
        nudge_n_steps=nudge_n_steps,
    )


def stand_mixer_progress_snapshot_v1():
    task = get_task_info()
    try:
        oracle = get_oracle_targets()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "success": bool(task.get("success", False)),
            "reward": float(task.get("reward", 0.0)),
        }
    fixture_state = oracle.get("fixture_state", {})
    if not isinstance(fixture_state, dict):
        fixture_state = {}
    head = fixture_state.get("head")
    head_f = float(head) if head is not None else float("nan")
    return {
        "available": True,
        "success": bool(task.get("success", False)),
        "reward": float(task.get("reward", 0.0)),
        "head": head_f,
        "remaining_to_open": max(0.0, 0.99 - head_f),
        "success_standard": "stand_mixer head > 0.99",
        "fixture_state": fixture_state,
    }


def stand_mixer_progress_lines_v1(progress, *, prefix):
    if not progress.get("available", False):
        return [
            f"{prefix}_progress_unavailable={progress.get('error')}",
            f"{prefix}_success={bool(progress.get('success', False))}",
            f"{prefix}_reward={float(progress.get('reward', 0.0)):.3f}",
        ]
    return [
        f"{prefix}_head={float(progress['head']):.3f}",
        f"{prefix}_success={bool(progress['success'])}",
        f"{prefix}_reward={float(progress['reward']):.3f}",
        f"{prefix}_remaining_to_open={float(progress['remaining_to_open']):.3f}",
    ]


def freespace_move_world_pose_v1(
    side,
    target_pos,
    target_quat,
    *,
    auto_update_world=True,
):
    result = freespace_move(
        right_target_pos=np.asarray(target_pos, dtype=float).tolist(),
        right_target_quat=np.asarray(target_quat, dtype=float).tolist(),
        side=side,
        planning_speed=1.5,
        speed=1.0,
        max_waypoints=24,
        auto_update_world=bool(auto_update_world),
    )
    status = getattr(result, "status", "Error")
    if status != "Success":
        return {
            "status": status,
            "reason": getattr(result, "reason", status),
        }
    return {
        "status": "Success",
        "reason": "",
        "trajectory_steps": int(getattr(result, "trajectory_steps", 0)),
        "final_pos_error_m": float(getattr(result, "final_pos_error_m", 0.0)),
    }


def _nudge_stand_mixer_v1(*, side, label, delta, nudge_n_steps):
    progress_before = stand_mixer_progress_snapshot_v1()
    print(
        f"{label}: delta={[round(float(x), 3) for x in delta]} "
        f"steps={int(nudge_n_steps)}"
    )
    for line in stand_mixer_progress_lines_v1(progress_before, prefix="before"):
        print(f"  {line}")

    result = nudge_brutal(
        side=side,
        delta_pos=np.asarray(delta, dtype=float).tolist(),
        n_steps=int(nudge_n_steps),
    )
    time.sleep(0.25)
    progress_after = stand_mixer_progress_snapshot_v1()
    print(
        f"{label}: success={result.success} "
        f"final_pos="
        f"{[round(float(x), 3) for x in np.asarray(result.final_pos, dtype=float)]} "
        f"task_success={bool(progress_after.get('success', False))} "
        f"reward={float(progress_after.get('reward', 0.0)):.3f}"
    )
    for line in stand_mixer_progress_lines_v1(progress_after, prefix="after"):
        print(f"  {line}")
    return {
        "result": result,
        "delta": np.asarray(delta, dtype=float),
        "progress_before": progress_before,
        "progress_after": progress_after,
    }


def _xy_cw_vertical(vec):
    vec = np.asarray(vec, dtype=float).copy()
    vec[2] = 0.0
    return _normalize(np.array([vec[1], -vec[0], 0.0], dtype=float))


def _normalize(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > eps else np.zeros_like(vec)


def _normalize_quat(quat):
    quat = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm
