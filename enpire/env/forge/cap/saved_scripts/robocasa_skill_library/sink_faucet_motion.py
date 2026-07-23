# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# sink_faucet_motion.py — shared sink faucet motion helpers
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from skill_library.namespace import *  # noqa: F401, F403


def refresh_sink_faucet_planner_world_v1(reason):
    world_info = update_planner_world()
    print(f"[curobo] {reason}: {world_info['n_obstacles']} obstacles")
    return world_info


def compute_sink_faucet_geometry_v1(sink_faucet_det, faucet_handle_det):
    sink_faucet_pos = np.asarray(sink_faucet_det["pos"], dtype=float)
    faucet_handle_pos = np.asarray(faucet_handle_det["pos"], dtype=float)
    handle_delta = faucet_handle_pos - sink_faucet_pos
    handle_delta_xy = handle_delta.copy()
    handle_delta_xy[2] = 0.0
    if float(np.linalg.norm(handle_delta_xy)) < 1e-6:
        raise RuntimeError("faucet handle - sink faucet vector is degenerate")

    normal_info = surface_normal_from_cameras_v1()
    surface_normal = np.asarray(normal_info["surface_normal"], dtype=float)
    handle_quat = make_press_quat_v1(surface_normal)

    tangent_axis = _xy_cw_vertical(surface_normal)
    base_axis_name, base_axis = _snap_xy_axis(tangent_axis)
    projection_on_base = float(np.dot(handle_delta_xy, base_axis))
    move_direction = base_axis.copy()
    move_axis_name = base_axis_name
    if projection_on_base < 0.0:
        move_direction = -base_axis
        move_axis_name = (
            f"-{base_axis_name[1]}"
            if base_axis_name.startswith("+")
            else f"+{base_axis_name[1]}"
        )

    move_axis_projection = float(np.dot(handle_delta_xy, move_direction))
    if abs(move_axis_projection) < 1e-6:
        raise RuntimeError("handle vector has no projection on movement axes")

    return {
        "sink_faucet_pos": sink_faucet_pos,
        "faucet_handle_pos": faucet_handle_pos,
        "handle_delta": handle_delta,
        "surface_normal": surface_normal,
        "handle_quat": handle_quat,
        "move_direction": move_direction,
        "move_axis_name": move_axis_name,
        "move_axis_projection": move_axis_projection,
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


def make_press_quat_v1(surface_normal, roll_deg=0.0):
    z_axis = _normalize(-np.asarray(surface_normal, dtype=float))
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = _normalize(world_up - z_axis * np.dot(world_up, z_axis))
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = _normalize(x_axis - z_axis * np.dot(x_axis, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    rot = R.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    if abs(float(roll_deg)) > 1e-6:
        rot = R.from_rotvec(z_axis * np.radians(float(roll_deg))) * rot
    return _normalize_quat(rot.as_quat())


def make_sink_faucet_handle_grip_map_candidates_v1(
    geometry,
    faucet_handle_det,
    grip_map_offsets_m,
    handle_z_offset_m,
    handle_move_dir_offset_m,
):
    faucet_handle_pos = np.asarray(faucet_handle_det["pos"], dtype=float)
    move_direction = np.asarray(geometry["move_direction"], dtype=float)
    base_handle_pos = faucet_handle_pos.copy()
    base_handle_pos[2] += float(handle_z_offset_m)
    base_handle_pos += move_direction * float(handle_move_dir_offset_m)
    offset_pairs = [
        (0.0, 0.0),
        *[
            (float(x_offset), float(y_offset))
            for x_offset in grip_map_offsets_m
            for y_offset in grip_map_offsets_m
            if abs(float(x_offset)) > 1e-9 or abs(float(y_offset)) > 1e-9
        ],
    ]
    candidates = []
    for x_offset, y_offset in offset_pairs:
        name = f"x{x_offset:+.2f}_y{y_offset:+.2f}"
        pos = base_handle_pos + np.array([x_offset, y_offset, 0.0], dtype=float)
        candidates.append((name, pos))
    return candidates


def make_sink_faucet_pre_pre_candidates_v1(
    geometry,
    faucet_handle_det,
    grip_map_offsets_m,
    handle_z_offset_m,
    handle_move_dir_offset_m,
    pre_pre_handle_m,
):
    surface_normal = np.asarray(geometry["surface_normal"], dtype=float)
    return [
        (
            grip_map_name,
            handle_pos + surface_normal * float(pre_pre_handle_m),
            handle_pos,
        )
        for grip_map_name, handle_pos in make_sink_faucet_handle_grip_map_candidates_v1(
            geometry,
            faucet_handle_det,
            grip_map_offsets_m,
            handle_z_offset_m,
            handle_move_dir_offset_m,
        )
    ]


def move_to_sink_faucet_pre_pre_handle_v1(
    *,
    side,
    sink_faucet_det,
    faucet_handle_det,
    grip_map_offsets_m,
    handle_z_offset_m,
    handle_move_dir_offset_m,
    pre_pre_handle_m,
    pre_handle_m,
):
    geometry = compute_sink_faucet_geometry_v1(sink_faucet_det, faucet_handle_det)
    handle_quat = np.asarray(geometry["handle_quat"], dtype=float)
    surface_normal = np.asarray(geometry["surface_normal"], dtype=float)
    pre_pre_candidates = make_sink_faucet_pre_pre_candidates_v1(
        geometry,
        faucet_handle_det,
        grip_map_offsets_m,
        handle_z_offset_m,
        handle_move_dir_offset_m,
        pre_pre_handle_m,
    )
    if not pre_pre_candidates:
        raise RuntimeError("no faucet pre-pre-handle candidates")

    print("  Opening gripper for pre-pre handle move")
    open_gripper(side)
    world_info = refresh_sink_faucet_planner_world_v1("pre-pre handle")

    last_status = None
    last_reason = ""
    for attempt_idx, (grip_map_name, pre_pre_pos, handle_pos) in enumerate(
        pre_pre_candidates,
        start=1,
    ):
        result = freespace_move_world_pose_v1(
            side,
            pre_pre_pos,
            handle_quat,
            auto_update_world=True,
        )
        last_status = result["status"]
        last_reason = result.get("reason", "")
        print(
            f"pre_pre_handle attempt={attempt_idx}/{len(pre_pre_candidates)} "
            f"grip_map={grip_map_name} status={last_status} "
            f"target={[round(float(x), 3) for x in pre_pre_pos]} "
            f"quat={[round(float(x), 4) for x in handle_quat]}"
        )
        if last_status == "Success":
            pre_handle_pos = handle_pos + surface_normal * float(pre_handle_m)
            return {
                "geometry": geometry,
                "world_info": world_info,
                "grip_map_name": grip_map_name,
                "pre_pre_handle_pos": np.asarray(pre_pre_pos, dtype=float),
                "pre_handle_pos": np.asarray(pre_handle_pos, dtype=float),
                "handle_pos": np.asarray(handle_pos, dtype=float),
                "handle_quat": handle_quat,
                "attempt_idx": attempt_idx,
                "n_candidates": len(pre_pre_candidates),
            }

    raise RuntimeError(
        f"failed {len(pre_pre_candidates)} pre-pre-handle candidates "
        f"(last_status={last_status}, reason={last_reason})"
    )


def nudge_to_sink_faucet_pre_handle_v1(
    *,
    side,
    motion_state,
    base_nudge_steps,
):
    pre_pre_pos = np.asarray(motion_state["pre_pre_handle_pos"], dtype=float)
    pre_pos = np.asarray(motion_state["pre_handle_pos"], dtype=float)
    delta = pre_pos - pre_pre_pos
    nudge_steps = max(
        int(base_nudge_steps),
        int(np.ceil(float(np.linalg.norm(delta)) / 0.01)),
    )
    print("  Opening gripper for pre-handle nudge")
    open_gripper(side)
    print(
        f"pre_handle_nudge: delta={[round(float(x), 3) for x in delta]} "
        f"steps={nudge_steps}"
    )
    result = nudge_brutal(
        side=side,
        delta_pos=delta.tolist(),
        n_steps=nudge_steps,
    )
    time.sleep(0.25)
    print(
        f"pre_handle_nudge: success={result.success} "
        f"final_pos="
        f"{[round(float(x), 3) for x in np.asarray(result.final_pos, dtype=float)]}"
    )
    return {
        "result": result,
        "delta": delta,
        "nudge_steps": nudge_steps,
    }


def nudge_sink_faucet_surface_normal_v1(
    *,
    side,
    motion_state,
    normal_nudge_extra_m,
    normal_nudge_n_steps,
):
    geometry = motion_state["geometry"]
    surface_normal = np.asarray(geometry["surface_normal"], dtype=float)
    pre_pos = np.asarray(motion_state["pre_handle_pos"], dtype=float)
    handle_pos = np.asarray(motion_state["handle_pos"], dtype=float)
    normal_distance = abs(float(np.dot(pre_pos - handle_pos, surface_normal)))
    normal_nudge_m = normal_distance + float(normal_nudge_extra_m)
    delta = -surface_normal * normal_nudge_m
    print("  Opening gripper for -surface-normal nudge")
    open_gripper(side)
    print(
        f"normal_nudge: pre_to_handle_normal_dist={normal_distance:.3f} "
        f"extra={float(normal_nudge_extra_m):.3f} "
        f"normal_nudge_m={normal_nudge_m:.3f} "
        f"delta={[round(float(x), 3) for x in delta]}"
    )
    result = nudge_brutal(
        side=side,
        delta_pos=delta.tolist(),
        n_steps=int(normal_nudge_n_steps),
    )
    time.sleep(0.25)
    task = get_task_info()
    print(
        f"normal_nudge: success={result.success} "
        f"final_pos="
        f"{[round(float(x), 3) for x in np.asarray(result.final_pos, dtype=float)]} "
        f"task_success={bool(task.get('success', False))} "
        f"reward={float(task.get('reward', 0.0)):.3f}"
    )
    return {
        "result": result,
        "delta": delta,
        "normal_distance": normal_distance,
        "normal_nudge_m": normal_nudge_m,
        "task": task,
    }


def nudge_sink_faucet_move_direction_v1(
    *,
    side,
    motion_state,
    move_nudge_m,
    move_nudge_n_steps,
    repeat_idx=1,
    repeat_total=1,
):
    geometry = motion_state["geometry"]
    move_direction = np.asarray(geometry["move_direction"], dtype=float)
    move_axis_name = str(geometry["move_axis_name"])
    move_axis_projection = float(geometry["move_axis_projection"])
    delta = move_direction * float(move_nudge_m)
    print("  Keeping gripper open for move-direction nudge")
    open_gripper(side)
    time.sleep(0.2)

    progress_before = sink_faucet_progress_snapshot_v1()
    print(
        f"move_direction_nudge[{repeat_idx}/{repeat_total}]: "
        f"move_axis={move_axis_name} projection={move_axis_projection:.3f} "
        f"delta={[round(float(x), 3) for x in delta]}"
    )
    for line in sink_faucet_progress_lines_v1(progress_before, prefix="before"):
        print(f"  {line}")

    result = nudge_brutal(
        side=side,
        delta_pos=delta.tolist(),
        n_steps=int(move_nudge_n_steps),
    )
    time.sleep(0.25)
    task = get_task_info()
    progress_after = sink_faucet_progress_snapshot_v1()
    handle_delta_text = "handle_joint_delta=unknown"
    if progress_before.get("available", False) and progress_after.get(
        "available",
        False,
    ):
        handle_delta = float(progress_after["handle_joint"]) - float(
            progress_before["handle_joint"]
        )
        handle_delta_text = f"handle_joint_delta={handle_delta:.3f}"

    print(
        f"move_direction_nudge[{repeat_idx}/{repeat_total}]: "
        f"closed_gripper_before_nudge=False "
        f"gripper_open_before_nudge=True "
        f"success={result.success} "
        f"final_pos="
        f"{[round(float(x), 3) for x in np.asarray(result.final_pos, dtype=float)]} "
        f"{handle_delta_text} "
        f"task_success={bool(task.get('success', False))} "
        f"reward={float(task.get('reward', 0.0)):.3f}"
    )
    for line in sink_faucet_progress_lines_v1(progress_after, prefix="after"):
        print(f"  {line}")

    return {
        "result": result,
        "delta": delta,
        "task": task,
        "progress_before": progress_before,
        "progress_after": progress_after,
    }


def sink_faucet_progress_snapshot_v1():
    try:
        oracle = get_oracle_targets()
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    fixture_state = oracle.get("fixture_state", {})
    if not isinstance(fixture_state, dict):
        fixture_state = {}
    handle_joint = fixture_state.get("handle_joint")
    handle_joint_f = (
        float(handle_joint) if handle_joint is not None else float("nan")
    )
    return {
        "available": True,
        "success": bool(oracle.get("task_info", {}).get("success", False)),
        "reward": float(oracle.get("task_info", {}).get("reward", 0.0)),
        "handle_joint": handle_joint_f,
        "water_on": bool(fixture_state.get("water_on", False)),
        "water_pressure": fixture_state.get("water_pressure"),
        "water_temp_state": fixture_state.get("water_temp_state"),
        "remaining_to_on": max(0.0, 0.40 - handle_joint_f),
        "margin_to_off_high": float(np.pi) - handle_joint_f,
        "success_standard": "0.40 < handle_joint < pi",
    }


def sink_faucet_progress_lines_v1(progress, *, prefix):
    if not progress.get("available", False):
        return [f"{prefix}_progress_unavailable={progress.get('error')}"]
    return [
        f"{prefix}_handle_joint={float(progress['handle_joint']):.3f}",
        f"{prefix}_water_on={bool(progress['water_on'])}",
        f"{prefix}_water_pressure={progress.get('water_pressure')}",
        f"{prefix}_success={bool(progress['success'])}",
        f"{prefix}_remaining_to_on={float(progress['remaining_to_on']):.3f}",
        f"{prefix}_margin_to_off_high="
        f"{float(progress['margin_to_off_high']):.3f}",
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


def _snap_xy_axis(vec):
    vec_xy = np.asarray(vec, dtype=float).copy()
    vec_xy[2] = 0.0
    if float(np.linalg.norm(vec_xy)) < 1e-6:
        raise RuntimeError("cannot snap degenerate XY axis")
    if abs(float(vec_xy[0])) >= abs(float(vec_xy[1])):
        sign = 1.0 if float(vec_xy[0]) >= 0.0 else -1.0
        return (
            "+x" if sign > 0.0 else "-x",
            np.array([sign, 0.0, 0.0], dtype=float),
        )
    sign = 1.0 if float(vec_xy[1]) >= 0.0 else -1.0
    return (
        "+y" if sign > 0.0 else "-y",
        np.array([0.0, sign, 0.0], dtype=float),
    )


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
