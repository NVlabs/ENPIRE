# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# move_to_target_xyz.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill


MOVE_XYZ_BATCH_TOP_K = 16


@skill
def move_to_target_xyz_v1(
    side,
    target_xyz,
    nominal_rpy=(180.0, 0.0, 0.0),
    pitch_offsets=(-30.0, -15.0, 0.0, 15.0),
    yaw_offsets=(-45.0, -15.0, 0.0, 30.0),
    batch_top_k=MOVE_XYZ_BATCH_TOP_K,
    augment_yaw_flip=False,
):
    """Move the EE to ``target_xyz`` after picking the best IK-feasible orientation.

    Builds a |pitch_offsets|×|yaw_offsets| grid of display-RPY candidates around
    ``nominal_rpy``, batch-IKs them via :func:`select_best_grasp`, then executes
    the top feasible pose with :func:`freespace_move`. Useful when the caller
    knows where to put the EE but wants the planner to choose an orientation
    that is reachable from the current arm pose.

    Adapted from ``yam_autorl/pin_insert/move_to_target_xyz_v1`` for the
    RoboCasa skill library. Differences vs. the YAM version:
    - ``nominal_rpy`` defaults to ``(180, 0, 0)`` (vertical-down in the
      display-RPY convention used by ``select_best_grasp`` /
      ``display_rpy_to_quat`` — equivalent to "gripper pointing -Z"). The YAM
      version pulled nominal pose from a side-specific birdseye anchor; here
      the caller supplies it explicitly so the same skill works for any
      RoboCasa task.
    - The fingertip-offset compensation (``target_is_fingertip``,
      ``TILT_FINGERTIP_OFFSET_*``) is dropped. Pre-offset ``target_xyz``
      yourself if you want to anchor at the fingertip rather than the EE site.
    - Uses RoboCasa's :func:`freespace_move` (with quaternion via
      :func:`display_rpy_to_quat`) instead of YAM's ``safe_move_detailed``.

    Args:
        side: ``"left"`` or ``"right"``.
        target_xyz: World [x, y, z] in metres. Same point used for every
            candidate orientation — orientation is the only thing being swept.
        nominal_rpy: Display-RPY [roll, pitch, yaw] in degrees that the offsets
            are applied around. Default is vertical-down. Pass e.g.
            ``(150, 0, 45)`` for a tilted approach.
        pitch_offsets: Pitch deltas (deg) added to ``nominal_rpy[1]``.
        yaw_offsets: Yaw deltas (deg) added to ``nominal_rpy[2]``.
        batch_top_k: At most this many candidates are scored (top-score-first).
            Total candidate count is ``len(pitch_offsets) * len(yaw_offsets)``;
            with the defaults that's 16, matching the YAM constant.
        augment_yaw_flip: Forwarded to :func:`select_best_grasp`. If True,
            doubles candidates with 180°-yaw-flipped variants — useful for
            symmetric parallel-jaw grasps where +yaw and -yaw are equivalent.

    Returns:
        ``(success: bool, info: dict)``. On success, ``info`` includes the
        selected position/rpy, IK error, and the executed move's status. On
        failure (no IK-feasible candidate, or the move itself failed), ``info``
        carries ``reason``/``batch_status`` so the caller can decide whether to
        widen the offsets, change ``nominal_rpy``, or fall back.
    """
    import numpy as np
    from types import SimpleNamespace

    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3).tolist()
    nom_roll, nom_pitch, nom_yaw = (float(v) for v in nominal_rpy)

    # Build a candidate grid. Score = closeness to nominal so the planner
    # picks orientations that move the wrist least when ties exist.
    candidates = []
    for p_off in pitch_offsets:
        for y_off in yaw_offsets:
            penalty = abs(float(p_off)) + abs(float(y_off))
            candidates.append(
                SimpleNamespace(
                    position=list(target_xyz),
                    rpy=[nom_roll, nom_pitch + float(p_off), nom_yaw + float(y_off)],
                    score=1.0 - 0.001 * penalty,
                    width=0.08,
                )
            )

    batch = select_best_grasp(
        candidates,
        side=side,
        batch_top_k=int(batch_top_k),
        augment_yaw_flip=bool(augment_yaw_flip),
    )
    best = getattr(batch, "best_candidate", None)
    if best is None or not getattr(best, "is_executable", False):
        return False, {
            "success": False,
            "side": side,
            "target_xyz": list(target_xyz),
            "reason": getattr(batch, "reason", "no IK-feasible XYZ pose"),
            "batch_status": getattr(batch, "status", None),
            "candidate_count": len(candidates),
            "evaluated": int(getattr(batch, "evaluated_candidate_count", 0)),
        }

    best_rpy = [float(v) for v in best.rpy]
    best_quat = list(display_rpy_to_quat(best_rpy))
    move_result = freespace_move(
        right_target_pos=[float(v) for v in best.position],
        right_target_quat=best_quat,
        side=side,
    )
    move_status = getattr(move_result, "status", "Unknown")
    move_err = getattr(move_result, "final_pos_error_m", None)
    success = move_status == "Success"

    return bool(success), {
        "success": bool(success),
        "side": side,
        "target_xyz": list(target_xyz),
        "selected_position": [float(v) for v in best.position],
        "selected_rpy": best_rpy,
        "selected_rank": int(getattr(best, "rank", -1)),
        "selected_ik_error_m": getattr(best, "ik_error_m", None),
        "candidate_count": len(candidates),
        "batch_status": getattr(batch, "status", None),
        "evaluated": int(getattr(batch, "evaluated_candidate_count", 0)),
        "executed_status": move_status,
        "executed_pos_error_m": move_err,
    }
