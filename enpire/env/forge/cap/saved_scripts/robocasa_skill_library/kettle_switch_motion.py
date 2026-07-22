# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# kettle_switch_motion.py — shared kettle switch motion helpers
from skill_library.namespace import *  # noqa: F401, F403

import numpy as np


def refresh_kettle_planner_world_v1(reason):
    world_info = update_planner_world()
    print(f"[curobo] {reason}: {world_info['n_obstacles']} obstacles")
    return world_info


def make_quat_candidates_v1(items):
    return [(name, _normalize_quat(quat)) for name, quat in items]


def make_xy_candidates_v1(items):
    return [
        (name, np.array([dx, dy, 0.0], dtype=float))
        for name, (dx, dy) in items
    ]


def move_to_switch_hover_v1(
    *,
    side,
    detections,
    candidate_order,
    ee_offset_world,
    hover_heights,
    xy_candidates,
    orientation_candidates,
):
    print("  Opening gripper for hover")
    open_gripper(side)
    attempt_candidates = _build_hover_attempts(
        hover_heights,
        xy_candidates,
        orientation_candidates,
    )
    print(
        f"  orientation_candidates={len(orientation_candidates)} "
        f"position_candidates={len(xy_candidates)} "
        f"total_attempts={len(attempt_candidates)}"
    )

    last_status = None
    last_reason = ""
    for candidate_rank, det_idx in enumerate(candidate_order, start=1):
        switch_det = detections[int(det_idx)]
        switch_pos = np.asarray(switch_det["pos"], dtype=float)
        print(
            f"  Trying switch candidate {candidate_rank}/{len(candidate_order)} "
            f"[{det_idx}] from {switch_det['camera']} "
            f"pos={[round(float(x), 3) for x in switch_pos]} "
            f"score={switch_det['score']:.3f}"
        )
        for attempt_idx, attempt in enumerate(attempt_candidates, start=1):
            (
                position_name,
                xy_offset,
                hover_name,
                hover_z,
                orientation_name,
                move_quat,
            ) = attempt
            hover_pos = switch_pos + np.asarray(ee_offset_world, dtype=float) + xy_offset
            hover_pos[2] += hover_z
            result = freespace_move_world_pose_v1(side, hover_pos, move_quat)
            last_status = result["status"]
            last_reason = result.get("reason", "")
            print(
                f"switch_hover candidate={candidate_rank}/{len(candidate_order)} "
                f"attempt={attempt_idx}/{len(attempt_candidates)} "
                f"position={position_name} "
                f"height={hover_name} z={hover_z:.2f} "
                f"orientation={orientation_name}: status={last_status} "
                f"target={[round(float(x), 3) for x in hover_pos]} "
                f"quat={[round(float(x), 4) for x in move_quat]}"
            )
            if last_status == "Success":
                return (
                    switch_det,
                    hover_pos,
                    move_quat,
                    hover_name,
                    orientation_name,
                    position_name,
                )

    raise RuntimeError(
        f"failed to reach kettle switch hover "
        f"(last_status={last_status}, reason={last_reason})"
    )


def freespace_move_world_pose_v1(side, target_pos, target_quat):
    result = freespace_move(
        right_target_pos=np.asarray(target_pos, dtype=float).tolist(),
        right_target_quat=np.asarray(target_quat, dtype=float).tolist(),
        side=side,
        auto_update_world=False,
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


def _build_hover_attempts(hover_heights, xy_candidates, orientation_candidates):
    attempts = [
        (position_name, xy_offset, hover_name, hover_z, orientation_name, move_quat)
        for position_name, xy_offset in xy_candidates
        for hover_name, hover_z in hover_heights
        for orientation_name, move_quat in orientation_candidates[:1]
    ]
    attempts.extend(
        (position_name, xy_offset, hover_name, hover_z, orientation_name, move_quat)
        for orientation_name, move_quat in orientation_candidates[1:]
        for position_name, xy_offset in xy_candidates
        for hover_name, hover_z in hover_heights
    )
    return attempts


def _normalize_quat(quat):
    quat = np.array(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm
