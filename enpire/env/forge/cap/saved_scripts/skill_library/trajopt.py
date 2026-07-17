from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill  # noqa: F401

from skill_library.constants.planning import (
    BATCH_SOLVER_SPEED,
    BATCH_TOP_K,
    BATCH_VALIDATE_TRAJECTORY,
    DEFAULT_RPY_RELAXATIONS,
    DEFAULT_XYZ_RELAXATIONS,
    IK_ERROR_THRESHOLD_M,
    IK_RPY_WEIGHT,
    IK_XYZ_WEIGHT,
    MOTION_PLANNER_BACKEND,
    PLANNING_SPEED,
)


def _as_list(value):
    if value is None:
        return None
    return [float(x) for x in value]


def _candidate_get(candidate, name, default=None):
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _candidate_to_dict(candidate):
    position = _candidate_get(candidate, "planner_xyz", None)
    if position is None:
        position = _candidate_get(candidate, "position", None)
    rpy = _candidate_get(candidate, "planner_rpy", None)
    if rpy is None:
        rpy = _candidate_get(candidate, "rpy", None)

    out = {
        "position": _as_list(position),
        "score": float(_candidate_get(candidate, "score", 1.0)),
    }
    if rpy is not None:
        out["rpy"] = _as_list(rpy)
    quat = _candidate_get(candidate, "quat", None)
    if quat is None:
        quat = _candidate_get(candidate, "quaternion", None)
    if quat is not None:
        out["quat"] = _as_list(quat)
        if "rpy" not in out:
            out["rpy"] = _quat_to_display_rpy(out["quat"])
    width = _candidate_get(candidate, "width", None)
    if width is not None:
        out["width"] = float(width)
    label = _candidate_get(candidate, "label", None)
    if label is not None:
        out["label"] = str(label)
    return out


def _candidate_pose(candidate):
    return {
        "position": _as_list(_candidate_get(candidate, "position", [])),
        "rpy": _as_list(_candidate_get(candidate, "rpy", [])),
        "score": float(_candidate_get(candidate, "score", 0.0)),
        "width": float(_candidate_get(candidate, "width", 0.0) or 0.0),
        "trajectory_cache_key": _candidate_get(candidate, "trajectory_cache_key", None),
        "planner_status": _candidate_get(candidate, "planner_status", None),
        "ik_error_m": _candidate_get(candidate, "ik_error_m", None),
        "ik_rot_error_deg": _candidate_get(candidate, "ik_rot_error_deg", None),
    }


def _motion_summary(result):
    return {
        "status": getattr(result, "status", None),
        "reason": getattr(result, "reason", ""),
        "executed": getattr(result, "executed", None),
        "planning_mode": getattr(result, "planning_mode", None),
        "side": getattr(result, "side", None),
        "ik_error_m": getattr(result, "ik_error_m", None),
        "final_pos_error_m": getattr(result, "final_pos_error_m", None),
        "final_rot_error_deg": getattr(result, "final_rot_error_deg", None),
        "trajectory_cache_key": getattr(result, "trajectory_cache_key", None),
        "trajectory_steps": getattr(result, "trajectory_steps", None),
    }


def _quat_to_display_rpy(quat_xyzw):
    from scipy.spatial.transform import Rotation

    ex, ey, ez = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    return [float(ey), float(-ex), float(-ez - 90.0)]


def _result_ok(result):
    return getattr(result, "status", None) in (None, "Success", "success", "done")


def pose_candidates(
    target_pos,
    target_rpy=None,
    target_quat=None,
    score=1.0,
    width=None,
    label="target",
):
    """Build a single planner-facing candidate from a target pose."""
    candidate = {
        "position": _as_list(target_pos),
        "score": float(score),
        "label": str(label),
    }
    if target_rpy is not None:
        candidate["rpy"] = _as_list(target_rpy)
    if target_quat is not None:
        candidate["quat"] = _as_list(target_quat)
        if target_rpy is None:
            candidate["rpy"] = _quat_to_display_rpy(candidate["quat"])
    if width is not None:
        candidate["width"] = float(width)
    return [candidate]


def relax_candidates(candidates, pos_offsets=None, rpy_offsets=None, score_decay=0.95):
    """Expand candidates with small XYZ/RPY offsets for fallback planning."""
    base = [_candidate_to_dict(c) for c in candidates]
    xyz_offsets = pos_offsets if pos_offsets is not None else DEFAULT_XYZ_RELAXATIONS
    rpy_offsets_ = rpy_offsets if rpy_offsets is not None else DEFAULT_RPY_RELAXATIONS
    relaxed = []
    for candidate in base:
        pos = candidate.get("position")
        rpy = candidate.get("rpy")
        if pos is None:
            continue
        for pi, xyz_offset in enumerate(xyz_offsets):
            if rpy is None:
                rpy_iter = [(0.0, 0.0, 0.0)]
            else:
                rpy_iter = rpy_offsets_
            for ri, rpy_offset in enumerate(rpy_iter):
                next_candidate = dict(candidate)
                next_candidate["position"] = [
                    float(pos[i]) + float(xyz_offset[i]) for i in range(3)
                ]
                if rpy is not None:
                    next_candidate["rpy"] = [
                        float(rpy[i]) + float(rpy_offset[i]) for i in range(3)
                    ]
                penalty = int(pi != 0) + int(ri != 0)
                next_candidate["score"] = float(candidate.get("score", 1.0)) * (
                    float(score_decay) ** penalty
                )
                relaxed.append(next_candidate)
    return relaxed


def solve_candidates(candidates, side, batch_top_k=None, label="trajopt", pad=True):
    """Batch-rank candidates with freespace_move and return structured metadata."""
    top_k = int(batch_top_k or BATCH_TOP_K)
    ranked = sorted(
        [_candidate_to_dict(c) for c in candidates],
        key=lambda c: -float(c.get("score", 0.0)),
    )[:top_k]
    if not ranked:
        return {
            "success": False,
            "label": label,
            "side": side,
            "feasible_count": 0,
            "selected_pose": None,
            "trajectory_cache_key": None,
            "failure_reasons": ["no candidates"],
        }

    payload = list(ranked)
    if pad and len(payload) < top_k:
        payload.extend([payload[0]] * (top_k - len(payload)))

    batch = freespace_move(
        grasp_candidates=payload,
        batch_side=side,
        batch_top_k=len(payload),
        solver_speed=BATCH_SOLVER_SPEED,
        batch_validate_trajectory=BATCH_VALIDATE_TRAJECTORY,
        planning_speed=PLANNING_SPEED,
        ik_error_threshold=IK_ERROR_THRESHOLD_M,
        ik_xyz_weight=IK_XYZ_WEIGHT,
        ik_rpy_weight=IK_RPY_WEIGHT,
        planner_backend=MOTION_PLANNER_BACKEND,
    )
    batch_candidates = list(getattr(batch, "batch_candidates", []) or [])
    feasible = [
        c
        for c in batch_candidates
        if getattr(c, "motion_plan_error", True) is False
        and getattr(c, "trajectory_cache_key", None) is not None
    ]
    best = feasible[0] if feasible else None
    failure_reasons = [
        str(getattr(c, "motion_plan_reason", "") or getattr(c, "planner_status", ""))
        for c in batch_candidates
        if all(c is not f for f in feasible)
    ]
    summary = {
        "success": best is not None,
        "label": label,
        "side": side,
        "input_count": int(getattr(batch, "input_candidate_count", len(ranked))),
        "evaluated_count": int(getattr(batch, "evaluated_candidate_count", len(payload))),
        "payload_count": len(payload),
        "feasible_count": len(feasible),
        "selected_pose": _candidate_pose(best) if best is not None else None,
        "trajectory_cache_key": getattr(best, "trajectory_cache_key", None),
        "best_candidate": best,
        "feasible_candidates": feasible,
        "batch_result": batch,
        "failure_reasons": [r for r in failure_reasons if r],
        "curobo_solve_time_ms": getattr(batch, "curobo_solve_time_ms", None),
        "curobo_graph_time_ms": getattr(batch, "curobo_graph_time_ms", None),
        "curobo_ik_time_ms": getattr(batch, "curobo_ik_time_ms", None),
    }
    print(f"  batch [{label}]: {len(feasible)}/{len(payload)} executable")
    return summary


def solve_groups(
    groups,
    side,
    relax=False,
    pos_offsets=None,
    rpy_offsets=None,
    group_labels=None,
):
    """Try candidate groups in order and return the first feasible solution."""
    labels = list(group_labels or [])
    attempts = []
    for i, group in enumerate(groups):
        label = labels[i] if i < len(labels) else f"group-{i}"
        candidates = list(group)
        if relax:
            candidates = relax_candidates(candidates, pos_offsets, rpy_offsets)
        solution = solve_candidates(candidates, side=side, label=label)
        attempts.append(solution)
        if solution.get("success"):
            solution["attempts"] = attempts
            return solution
    return {
        "success": False,
        "side": side,
        "feasible_count": 0,
        "selected_pose": None,
        "trajectory_cache_key": None,
        "attempts": attempts,
        "failure_reasons": [
            reason
            for attempt in attempts
            for reason in attempt.get("failure_reasons", [])
        ],
    }


def execute_solution(solution, side=None):
    """Execute a cached solution or its selected pose."""
    if solution is None:
        return {"success": False, "reason": "missing solution", "result": None}
    key = solution.get("trajectory_cache_key")
    if key is None and solution.get("best_candidate") is not None:
        key = getattr(solution["best_candidate"], "trajectory_cache_key", None)
    if key is not None:
        result = freespace_move(preview_only=False, trajectory_cache_key=key)
        return {
            "success": _result_ok(result),
            "result": result,
            "summary": _motion_summary(result),
            "trajectory_cache_key": key,
        }

    pose = solution.get("selected_pose") or solution
    pos = pose.get("position")
    rpy = pose.get("rpy")
    quat = pose.get("quat")
    move_side = side or solution.get("side")
    return safe_move(move_side, target_pos=pos, target_rpy=rpy, target_quat=quat)


def safe_move(
    side,
    target_pos=None,
    target_rpy=None,
    target_quat=None,
    trajectory_cache_key=None,
):
    """Execute a single freespace move with saved-script planner defaults."""
    if trajectory_cache_key is not None:
        result = freespace_move(preview_only=False, trajectory_cache_key=trajectory_cache_key)
    else:
        move_kwargs = {
            "planning_speed": PLANNING_SPEED,
            "ik_error_threshold": IK_ERROR_THRESHOLD_M,
            "ik_xyz_weight": IK_XYZ_WEIGHT,
            "ik_rpy_weight": IK_RPY_WEIGHT,
            "planner_backend": MOTION_PLANNER_BACKEND,
            "preview_only": False,
        }
        if target_pos is not None:
            move_kwargs[f"{side}_target_pos"] = _as_list(target_pos)
        if target_quat is not None:
            move_kwargs[f"{side}_target_quat"] = _as_list(target_quat)
        elif target_rpy is not None:
            move_kwargs[f"{side}_target_rpy"] = _as_list(target_rpy)
        result = freespace_move(**move_kwargs)
    summary = _motion_summary(result)
    success = _result_ok(result)
    if not success:
        print(f"  move failed: {summary}")
    return {"success": success, "result": result, "summary": summary}
