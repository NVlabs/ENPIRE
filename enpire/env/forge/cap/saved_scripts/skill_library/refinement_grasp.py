# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from skill_library.namespace import *  # noqa: F401,F403
except AttributeError:
    pass
import collections

import numpy as np

from enpire.env.forge.cap.agent.skill_registry import skill  # noqa: F401
from skill_library import grasp_geometry
from skill_library.pick_place import (
    _birdseye_config,
    _gripper_pos_for_side,
    _place_with_recovery,
    _safe_move,
    birdseye_pose,
    go_birdeye,
)

try:
    from skill_library.constants.manipulation import MAX_GRASP_ATTEMPTS  # type: ignore
except Exception:
    MAX_GRASP_ATTEMPTS = 5

try:
    from skill_library.constants.planning import (  # type: ignore
        BATCH_SOLVER_SPEED,
        BATCH_TOP_K,
        BATCH_VALIDATE_TRAJECTORY,
        IK_ERROR_THRESHOLD_M,
        IK_RPY_WEIGHT,
        IK_XYZ_WEIGHT,
        MOTION_PLANNER_BACKEND,
        PLANNING_SPEED,
    )
except Exception:
    BATCH_SOLVER_SPEED = "fast"
    BATCH_TOP_K = 16
    BATCH_VALIDATE_TRAJECTORY = False
    IK_ERROR_THRESHOLD_M = 0.01
    IK_RPY_WEIGHT = 0.3
    IK_XYZ_WEIGHT = 1.0
    MOTION_PLANNER_BACKEND = "curobo"
    PLANNING_SPEED = 1.5

try:
    from skill_library.constants.vision import (  # type: ignore
        ANYGRASP_DISABLE_PLANNER_Z_CLIPPING,
        ANYGRASP_TCP_OFFSET_Z_M,
        BUNDLESDF_CAMERA,
        TOP_GRASP_MAX,
        TOP_GRASP_TRY,
    )
except Exception:
    ANYGRASP_DISABLE_PLANNER_Z_CLIPPING = True
    ANYGRASP_TCP_OFFSET_Z_M = 0.0
    BUNDLESDF_CAMERA = "top"
    TOP_GRASP_MAX = 16
    TOP_GRASP_TRY = 16


DEFAULT_HOVER_YAWS_DEG = (0.0, 45.0, 90.0, 135.0)
DEFAULT_RELAXED_RPY_OFFSETS_DEG = (
    (8.0, 0.0, 0.0),
    (-8.0, 0.0, 0.0),
    (0.0, 8.0, 0.0),
    (0.0, -8.0, 0.0),
    (0.0, 0.0, 12.0),
    (0.0, 0.0, -12.0),
)
DEFAULT_ALIGNMENT_SCORE_WEIGHT = 0.35
DEFAULT_ALIGNMENT_IK_SLACK_M = 0.004
DEFAULT_ALIGNMENT_IK_ROT_SLACK_DEG = 5.0
DEFAULT_REFINEMENT_PREGRASP_CLEARANCE_M = 0.07
DEFAULT_REFINEMENT_FINAL_APPROACH_SPEED_MPS = 0.15

SelectedAlignedGrasp = collections.namedtuple(
    "SelectedAlignedGrasp",
    [
        "position",
        "rpy",
        "score",
        "width",
        "trajectory_cache_key",
        "alignment_error_deg",
        "pregrasp_position",
        "approach_delta",
    ],
)


def _tool_or_none(name):
    fn = globals().get(name)
    if fn is not None:
        return fn
    try:
        import skill_library.namespace as namespace

        fn = getattr(namespace, name, None)
        if fn is not None:
            return fn
    except Exception:
        pass
    return None


def _state_attr(state, side, suffix):
    direct = f"{side}_{suffix}"
    if hasattr(state, direct):
        return getattr(state, direct)
    arms = getattr(state, "arms", None)
    if arms is not None:
        arm = arms[side] if isinstance(arms, dict) else getattr(arms, side)
        return getattr(arm, suffix)
    raise AttributeError(f"robot state missing {direct}")


def _result_ok(result):
    return getattr(result, "status", None) in (None, "Success", "success", "done")


def _motion_summary(result):
    return {
        "status": getattr(result, "status", None),
        "reason": getattr(result, "reason", ""),
        "planning_mode": getattr(result, "planning_mode", None),
        "trajectory_cache_key": getattr(result, "trajectory_cache_key", None),
        "ik_error_m": getattr(result, "ik_error_m", None),
        "final_pos_error_m": getattr(result, "final_pos_error_m", None),
        "final_rot_error_deg": getattr(result, "final_rot_error_deg", None),
    }


def _candidate_summary(candidate):
    if candidate is None:
        return None
    rank = getattr(candidate, "rank", None)
    source_index = getattr(candidate, "source_index", None)
    return {
        "position": [float(x) for x in getattr(candidate, "position", [])],
        "rpy": [float(x) for x in getattr(candidate, "rpy", [])],
        "score": float(getattr(candidate, "score", 0.0)),
        "rank": int(rank) if rank is not None else None,
        "source_index": int(source_index) if source_index is not None else None,
        "trajectory_cache_key": getattr(candidate, "trajectory_cache_key", None),
    }


def _field(candidate, name, default=None):
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _display_rpy_to_quat(rpy):
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    return Rotation.from_euler("xyz", [-pitch, roll, -yaw - 90.0], degrees=True).as_quat()


def _approach_axis_from_rpy(rpy):
    from scipy.spatial.transform import Rotation

    axis = Rotation.from_quat(_display_rpy_to_quat(rpy)).as_matrix()[:, 2]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return np.asarray(axis, dtype=float) / norm


def _pregrasp_from_grasp(grasp, clearance_m):
    grasp_pos = np.asarray(_field(grasp, "position"), dtype=float)
    rpy = [float(x) for x in _field(grasp, "rpy")]
    approach_axis = _approach_axis_from_rpy(rpy)
    delta = approach_axis * float(clearance_m)
    pregrasp_pos = grasp_pos - delta
    return (
        [float(x) for x in pregrasp_pos.tolist()],
        [float(x) for x in delta.tolist()],
    )


def _quat_alignment_error_deg(target_quat, reference_quat):
    from scipy.spatial.transform import Rotation

    target = Rotation.from_quat(np.asarray(target_quat, dtype=float))
    reference = Rotation.from_quat(np.asarray(reference_quat, dtype=float))
    delta = reference.inv() * target
    return float(np.degrees(delta.magnitude()))


def _current_ee_quat(side):
    try:
        state = get_robot_state()
    except Exception as exc:
        print(f"  Could not read current gripper orientation: {exc}")
        return None

    for suffix in ("ee_quat", "ee_quat_xyzw"):
        try:
            quat = _state_attr(state, side, suffix)
        except Exception:
            quat = None
        if quat is not None and len(quat) == 4:
            return [float(x) for x in quat]

    for attr in (f"{side}_ee_quat", f"{side}_ee_quat_xyzw"):
        if hasattr(state, attr):
            quat = getattr(state, attr)
            if quat is not None and len(quat) == 4:
                return [float(x) for x in quat]
    if isinstance(state, dict):
        for key in (f"{side}_ee_quat", f"{side}_ee_quat_xyzw"):
            quat = state.get(key)
            if quat is not None and len(quat) == 4:
                return [float(x) for x in quat]
    return None


def _copy_grasp_for_alignment(
    grasp,
    score,
    alignment_error_deg,
    pregrasp_position=None,
    approach_delta=None,
    trajectory_cache_key=None,
):
    return SelectedAlignedGrasp(
        position=[float(x) for x in _field(grasp, "position", [])],
        rpy=[float(x) for x in _field(grasp, "rpy", [])],
        score=float(score),
        width=float(_field(grasp, "width", 0.0) or 0.0),
        trajectory_cache_key=trajectory_cache_key,
        alignment_error_deg=float(alignment_error_deg),
        pregrasp_position=(
            [float(x) for x in pregrasp_position]
            if pregrasp_position is not None
            else None
        ),
        approach_delta=(
            [float(x) for x in approach_delta]
            if approach_delta is not None
            else None
        ),
    )


def _annotate_grasps_by_orientation_alignment(grasps, reference_quat, **config):
    weight = float(config.get("orientation_alignment_score_weight", DEFAULT_ALIGNMENT_SCORE_WEIGHT))
    clearance_m = float(
        config.get("refinement_pregrasp_clearance_m", DEFAULT_REFINEMENT_PREGRASP_CLEARANCE_M)
    )
    annotated = []
    for grasp in grasps:
        rpy = _field(grasp, "rpy")
        base_score = float(_field(grasp, "score", 0.0) or 0.0)
        if reference_quat is None:
            alignment_error = 0.0
        else:
            try:
                alignment_error = _quat_alignment_error_deg(_display_rpy_to_quat(rpy), reference_quat)
            except Exception as exc:
                print(f"  Could not score grasp orientation alignment: {exc}")
                alignment_error = 180.0
        pregrasp_position, approach_delta = _pregrasp_from_grasp(grasp, clearance_m)
        adjusted_score = base_score - weight * (alignment_error / 180.0)
        annotated.append(
            _copy_grasp_for_alignment(
                grasp,
                adjusted_score,
                alignment_error,
                pregrasp_position=pregrasp_position,
                approach_delta=approach_delta,
            )
        )
    annotated.sort(key=lambda grasp: (float(grasp.alignment_error_deg), -float(grasp.score)))
    return annotated


def _select_best_aligned_grasp(grasps, side, label="grasp", **config):
    if not grasps:
        return None
    prefer_alignment = bool(config.get("prefer_current_gripper_orientation", True))
    reference_quat = _current_ee_quat(side) if prefer_alignment else None
    if reference_quat is None:
        print("  Current gripper orientation unavailable; ranking reachable pregrasps")

    candidates = _annotate_grasps_by_orientation_alignment(
        grasps,
        reference_quat,
        **config,
    )
    preview = ", ".join(
        f"{float(candidate.alignment_error_deg):.1f}deg"
        for candidate in candidates[: min(5, len(candidates))]
    )
    print(f"  Wrist grasp orientation alignment to pregrasp: best=[{preview}]")

    batch_top_k = max(int(config.get("batch_top_k", BATCH_TOP_K)), len(candidates))
    pregrasp_candidates = [
        {
            "position": list(candidate.pregrasp_position),
            "rpy": list(candidate.rpy),
            "score": float(candidate.score),
            "width": float(candidate.width),
        }
        for candidate in candidates
    ]
    batch = freespace_move(
        grasp_candidates=pregrasp_candidates,
        batch_side=side,
        batch_top_k=batch_top_k,
        solver_speed=config.get("solver_speed", BATCH_SOLVER_SPEED),
        batch_validate_trajectory=config.get(
            "batch_validate_trajectory", BATCH_VALIDATE_TRAJECTORY
        ),
        planning_speed=config.get("planning_speed", PLANNING_SPEED),
        ik_error_threshold=config.get("ik_error_threshold", IK_ERROR_THRESHOLD_M),
        ik_xyz_weight=config.get("ik_xyz_weight", IK_XYZ_WEIGHT),
        ik_rpy_weight=config.get("ik_rpy_weight", IK_RPY_WEIGHT),
        planner_backend=config.get("planner_backend", MOTION_PLANNER_BACKEND),
    )
    print(
        f"  Batched cuRobo {label} pregrasp ranking [{side}]: "
        f"input={int(getattr(batch, 'input_candidate_count', len(candidates)))}, "
        f"evaluated={int(getattr(batch, 'evaluated_candidate_count', 0))}, "
        f"truncated={int(getattr(batch, 'truncated_input_count', 0))}, "
        f"solve={float(getattr(batch, 'curobo_solve_time_ms', 0.0)):.1f}ms, "
        f"graph={float(getattr(batch, 'curobo_graph_time_ms', 0.0)):.1f}ms, "
        f"ik={float(getattr(batch, 'curobo_ik_time_ms', 0.0)):.1f}ms, "
        f"mode={getattr(batch, 'planning_mode', 'unknown')}"
    )

    feasible = [
        candidate
        for candidate in list(getattr(batch, "batch_candidates", []) or [])
        if getattr(candidate, "motion_plan_error", True) is False
    ]
    if not feasible:
        failures = []
        for candidate in list(getattr(batch, "batch_candidates", []) or [])[:5]:
            if getattr(candidate, "motion_plan_error", False):
                failures.append(
                    f"rank={int(getattr(candidate, 'rank', 0))}: "
                    f"{getattr(candidate, 'motion_plan_reason', None) or 'Motion plan error'}"
                )
        raise RuntimeError(
            f"Batch cuRobo returned no feasible {label} pregrasp on {side}. "
            f"{getattr(batch, 'reason', '') or ''}"
            + (f" Top failures: {'; '.join(failures)}" if failures else "")
        )

    def metric(candidate, attr, default=float("inf")):
        value = getattr(candidate, attr, None)
        return default if value is None else float(value)

    planner_best = getattr(batch, "best_candidate", None)
    if planner_best not in feasible:
        planner_best = feasible[0]
    best_pos_err = metric(planner_best, "ik_error_m")
    best_rot_err = metric(planner_best, "ik_rot_error_deg")
    pos_slack = float(config.get("orientation_alignment_ik_slack_m", DEFAULT_ALIGNMENT_IK_SLACK_M))
    rot_slack = float(
        config.get("orientation_alignment_ik_rot_slack_deg", DEFAULT_ALIGNMENT_IK_ROT_SLACK_DEG)
    )

    def alignment_for(candidate):
        source_index = int(getattr(candidate, "source_index", 0) or 0)
        if 0 <= source_index < len(candidates):
            return float(candidates[source_index].alignment_error_deg)
        return 180.0

    eligible = [
        candidate
        for candidate in feasible
        if metric(candidate, "ik_error_m", 0.0) <= best_pos_err + pos_slack
        and metric(candidate, "ik_rot_error_deg", 0.0) <= best_rot_err + rot_slack
    ]
    pool = eligible or feasible

    def source_for(candidate):
        source_index = int(getattr(candidate, "source_index", 0) or 0)
        if 0 <= source_index < len(candidates):
            return candidates[source_index]
        return candidates[0]

    pool.sort(
        key=lambda candidate: (
            -float(source_for(candidate).score),
            metric(candidate, "ik_error_m", 0.0),
            metric(candidate, "ik_rot_error_deg", 0.0),
            alignment_for(candidate),
        )
    )
    best = pool[0]
    source = source_for(best)
    alignment_error = float(source.alignment_error_deg)
    print(
        f"  Selected {label}: rank={int(getattr(best, 'rank', 0))}, side={side}, "
        f"alignment={alignment_error:.1f}deg, "
        f"pregrasp_xyz={[round(float(x), 4) for x in getattr(best, 'position', [])]}, "
        f"grasp_xyz={[round(float(x), 4) for x in source.position]}, "
        f"rpy={[round(float(x), 1) for x in source.rpy]}, "
        f"score={float(getattr(best, 'score', 0.0)):.3f}"
    )
    return _copy_grasp_for_alignment(
        source,
        score=float(getattr(best, "score", 0.0)),
        alignment_error_deg=alignment_error,
        pregrasp_position=source.pregrasp_position,
        approach_delta=source.approach_delta,
        trajectory_cache_key=getattr(best, "trajectory_cache_key", None),
    )


def _execute_refinement_grasp_from_pregrasp(side, grasp, label="grasp", **config):
    if grasp is None:
        return False
    if grasp.pregrasp_position is None or grasp.approach_delta is None:
        print("  Refinement grasp missing pregrasp or approach delta")
        return False

    open_gripper(side)
    planning_speed = float(config.get("planning_speed", PLANNING_SPEED))
    approach_speed = float(
        config.get(
            "refinement_final_approach_speed_mps",
            DEFAULT_REFINEMENT_FINAL_APPROACH_SPEED_MPS,
        )
    )
    planned_approach_delta = np.asarray(grasp.approach_delta, dtype=float)
    print(
        f"  Moving to {label} pregrasp at normal speed: "
        f"xyz={[round(float(x), 4) for x in grasp.pregrasp_position]}, "
        f"rpy={[round(float(x), 1) for x in grasp.rpy]}, speed={planning_speed:.2f}"
    )
    if not _safe_move(
        side,
        grasp.pregrasp_position,
        grasp.rpy,
        trajectory_cache_key=getattr(grasp, "trajectory_cache_key", None),
        **config,
    ):
        return False

    nudge_fn = _tool_or_none("nudge")
    if nudge_fn is None:
        print("  nudge tool unavailable")

    backend = str(config.get("refinement_final_approach_backend", "pyroki")).lower()
    if backend == "pyroki":
        pyroki_fn = _tool_or_none("pyroki_final_approach")
        if pyroki_fn is None:
            print("  PyRoki final approach tool unavailable; falling back to nudge")
        else:
            try:
                print(
                    "  PyRoki straight final approach: "
                    f"target_xyz={[round(float(x), 4) for x in grasp.position]}, "
                    f"speed={approach_speed:.2f}m/s"
                )
                pyroki_result = pyroki_fn(
                    side=side,
                    target_pos=[float(x) for x in grasp.position],
                    target_rpy=[float(x) for x in grasp.rpy],
                    server_url=str(
                        config.get("pyroki_server_url", "http://127.0.0.1:9600")
                    ),
                    timesteps=int(config.get("pyroki_final_approach_timesteps", 16)),
                    max_cartesian_speed_mps=approach_speed,
                    max_joint_vel_rad_s=float(
                        config.get("pyroki_final_approach_max_joint_vel_rad_s", 0.8)
                    ),
                    min_duration_s=float(
                        config.get("pyroki_final_approach_min_duration_s", 0.6)
                    ),
                )
                print(f"  PyRoki final approach result: {pyroki_result}")
                close_gripper(side)
                gripper_pos = _gripper_pos_for_side(get_robot_state(), side)
                print(f"  Gripper pos after close: {gripper_pos:.4f}")
                if gripper_pos > 0.0:
                    print("  Grasp check passed")
                    return True
                print("  Grasp check failed (gripper at/near zero)")
                open_gripper(side)
                return False
            except Exception as exc:
                print(f"  PyRoki final approach failed: {exc}; falling back to nudge")

    if nudge_fn is None:
        return False

    try:
        current_pos = np.asarray(_state_attr(get_robot_state(), side, "ee_pos"), dtype=float)
        approach_delta = np.asarray(grasp.position, dtype=float) - current_pos
        lateral_correction = float(np.linalg.norm(approach_delta - planned_approach_delta))
    except Exception as exc:
        print(f"  Could not read actual pregrasp pose; using planned approach delta: {exc}")
        approach_delta = planned_approach_delta
        lateral_correction = 0.0

    approach_distance = float(np.linalg.norm(approach_delta))
    duration = max(1.0, approach_distance / max(approach_speed, 1e-3) + 0.5)
    print(
        f"  Straight final approach: "
        f"delta={[round(float(x), 4) for x in approach_delta.tolist()]}, "
        f"distance={approach_distance:.3f}m, speed={approach_speed:.2f}m/s, "
        f"pregrasp_correction={lateral_correction:.4f}m"
    )
    try:
        result = nudge_fn(
            side=side,
            delta_pos=[float(x) for x in approach_delta.tolist()],
            max_vel=approach_speed,
            max_duration_sec=duration,
        )
    except Exception as exc:
        print(f"  Straight final approach failed: {exc}")
        return False
    if hasattr(result, "success") and not bool(result.success):
        print(f"  Straight final approach failed: {getattr(result, 'error', '')}")
        return False

    close_gripper(side)
    gripper_pos = _gripper_pos_for_side(get_robot_state(), side)
    print(f"  Gripper pos after close: {gripper_pos:.4f}")
    if gripper_pos > 0.0:
        print("  Grasp check passed")
        return True
    print("  Grasp check failed (gripper at/near zero)")
    open_gripper(side)
    return False


def _detections_from_result(result, object_name):
    if isinstance(result, dict):
        if object_name in result:
            return list(result.get(object_name) or [])
        out = []
        for dets in result.values():
            out.extend(list(dets or []))
        return out
    return list(result or [])


def _detect_bundlesdf_pose(object_name, camera):
    detect_oneshot = _tool_or_none("detect_objects_oneshot")
    errors = []
    if detect_oneshot is not None:
        for args, kwargs in (
            ((object_name,), {"camera": camera}),
            ((), {"query": object_name, "camera": camera}),
        ):
            try:
                dets = _detections_from_result(detect_oneshot(*args, **kwargs), object_name)
                if dets:
                    det = dets[0]
                    if getattr(det, "position_3d", None):
                        return det
            except Exception as exc:
                errors.append(str(exc))

    detect = _tool_or_none("detect_object")
    if detect is not None:
        for args, kwargs in (
            ((object_name,), {"camera": camera, "backend": "bundlesdf"}),
            ((), {"query": object_name, "camera": camera, "backend": "bundlesdf"}),
            ((object_name,), {"camera": camera}),
            ((), {"query": object_name, "camera": camera}),
        ):
            try:
                dets = _detections_from_result(detect(*args, **kwargs), object_name)
                if dets:
                    det = dets[0]
                    if getattr(det, "position_3d", None):
                        return det
            except Exception as exc:
                errors.append(str(exc))

    suffix = f"; last errors: {' | '.join(errors[-3:])}" if errors else ""
    raise RuntimeError(f"No BundleSDF pose with position_3d for {object_name!r}{suffix}")


def _choose_closer_arm_from_pose(object_pos):
    state = get_robot_state()
    obj = np.asarray(object_pos, dtype=float)
    left_dist = float(np.linalg.norm(obj - np.asarray(_state_attr(state, "left", "ee_pos"))))
    right_dist = float(np.linalg.norm(obj - np.asarray(_state_attr(state, "right", "ee_pos"))))
    side = "left" if left_dist <= right_dist else "right"
    print(f"  BundleSDF object xyz={[round(float(x), 4) for x in obj.tolist()]}")
    print(f"  Distance to left gripper: {left_dist:.3f} m")
    print(f"  Distance to right gripper: {right_dist:.3f} m")
    print(f"  Refinement arm: {side}")
    return side, {"left": left_dist, "right": right_dist}


def _clamped_hover_height(hover_height_m, min_hover_m, max_hover_m):
    requested = float(hover_height_m)
    return min(max(requested, float(min_hover_m)), float(max_hover_m))


def _build_hover_candidates(
    hover_pos,
    hover_yaws_deg=None,
    relaxed_rpy_offsets_deg=None,
    include_relaxed=True,
):
    yaws = tuple(DEFAULT_HOVER_YAWS_DEG if hover_yaws_deg is None else hover_yaws_deg)
    relaxed_offsets = tuple(
        DEFAULT_RELAXED_RPY_OFFSETS_DEG
        if relaxed_rpy_offsets_deg is None
        else relaxed_rpy_offsets_deg
    )
    candidates = []
    for yaw_i, yaw in enumerate(yaws):
        base_rpy = [0.0, 180.0, float(yaw)]
        candidates.append(
            {
                "position": [float(x) for x in hover_pos],
                "rpy": base_rpy,
                "score": 1.0 - 0.01 * yaw_i,
                "label": f"strict_topdown_yaw_{float(yaw):.1f}",
            }
        )
        if not include_relaxed:
            continue
        for offset_i, rpy_offset in enumerate(relaxed_offsets):
            candidates.append(
                {
                    "position": [float(x) for x in hover_pos],
                    "rpy": [
                        float(base_rpy[i]) + float(rpy_offset[i]) for i in range(3)
                    ],
                    "score": 0.9 - 0.01 * yaw_i - 0.002 * offset_i,
                    "label": f"relaxed_topdown_yaw_{float(yaw):.1f}_{offset_i}",
                }
            )
    return candidates


def _hover_batch_kwargs(config, candidate_count):
    return {
        "batch_side": config.get("side"),
        "batch_top_k": max(int(config.get("batch_top_k", BATCH_TOP_K)), int(candidate_count)),
        "solver_speed": config.get("solver_speed", BATCH_SOLVER_SPEED),
        "batch_validate_trajectory": config.get(
            "batch_validate_trajectory", BATCH_VALIDATE_TRAJECTORY
        ),
        "planning_speed": config.get("planning_speed", PLANNING_SPEED),
        "ik_error_threshold": config.get("ik_error_threshold", IK_ERROR_THRESHOLD_M),
        "ik_xyz_weight": config.get("ik_xyz_weight", IK_XYZ_WEIGHT),
        "ik_rpy_weight": config.get("ik_rpy_weight", IK_RPY_WEIGHT),
        "planner_backend": config.get("planner_backend", MOTION_PLANNER_BACKEND),
    }


def _select_hover_plan(candidates, side, **config):
    if not candidates:
        return {
            "success": False,
            "side": side,
            "reason": "no hover candidates",
            "best_candidate": None,
        }
    batch_config = dict(config, side=side)
    kwargs = _hover_batch_kwargs(batch_config, len(candidates))
    try:
        batch = freespace_move(grasp_candidates=list(candidates), **kwargs)
    except Exception as exc:
        print(f"  Batched refinement hover planning failed [{side}]: {exc}")
        return {
            "success": False,
            "side": side,
            "reason": str(exc),
            "best_candidate": None,
        }
    print(
        f"  Batched cuRobo refinement hover [{side}]: "
        f"input={int(getattr(batch, 'input_candidate_count', len(candidates)))}, "
        f"evaluated={int(getattr(batch, 'evaluated_candidate_count', 0))}, "
        f"truncated={int(getattr(batch, 'truncated_input_count', 0))}, "
        f"solve={float(getattr(batch, 'curobo_solve_time_ms', 0.0)):.1f}ms, "
        f"graph={float(getattr(batch, 'curobo_graph_time_ms', 0.0)):.1f}ms, "
        f"ik={float(getattr(batch, 'curobo_ik_time_ms', 0.0)):.1f}ms, "
        f"mode={getattr(batch, 'planning_mode', 'unknown')}"
    )
    best = getattr(batch, "best_candidate", None)
    if best is not None and getattr(best, "motion_plan_error", True) is False:
        source_index = int(getattr(best, "source_index", 0) or 0)
        source = candidates[source_index] if source_index < len(candidates) else {}
        print(
            f"  Selected refinement hover: rank={int(getattr(best, 'rank', 0))}, "
            f"label={source.get('label', 'unknown')}, "
            f"xyz={[round(float(x), 4) for x in getattr(best, 'position', [])]}, "
            f"rpy={[round(float(x), 1) for x in getattr(best, 'rpy', [])]}"
        )
        return {
            "success": True,
            "side": side,
            "batch_result": batch,
            "best_candidate": best,
            "selected_source": source,
            "selected_pose": _candidate_summary(best),
            "trajectory_cache_key": getattr(best, "trajectory_cache_key", None),
        }

    failures = []
    for candidate in list(getattr(batch, "batch_candidates", []) or [])[:5]:
        if getattr(candidate, "motion_plan_error", False):
            failures.append(
                f"rank={int(getattr(candidate, 'rank', 0))}: "
                f"{getattr(candidate, 'motion_plan_reason', None) or 'Motion plan error'}"
            )
    return {
        "success": False,
        "side": side,
        "batch_result": batch,
        "best_candidate": best,
        "reason": getattr(batch, "reason", "") or "no feasible hover",
        "failure_reasons": failures,
    }


def _execute_hover_plan(solution, side, **config):
    best = solution.get("best_candidate")
    key = solution.get("trajectory_cache_key")
    try:
        if key is not None:
            result = freespace_move(preview_only=False, trajectory_cache_key=key)
        elif best is not None:
            prefix = "left" if side == "left" else "right"
            result = freespace_move(
                **{
                    f"{prefix}_target_pos": [float(x) for x in best.position],
                    f"{prefix}_target_rpy": [float(x) for x in best.rpy],
                    "planning_speed": config.get("planning_speed", PLANNING_SPEED),
                    "ik_error_threshold": config.get("ik_error_threshold", IK_ERROR_THRESHOLD_M),
                    "ik_xyz_weight": config.get("ik_xyz_weight", IK_XYZ_WEIGHT),
                    "ik_rpy_weight": config.get("ik_rpy_weight", IK_RPY_WEIGHT),
                    "planner_backend": config.get("planner_backend", MOTION_PLANNER_BACKEND),
                    "preview_only": False,
                }
            )
        else:
            return {"success": False, "reason": "missing hover solution", "summary": {}}
    except Exception as exc:
        print(f"  Refinement hover execution failed [{side}]: {exc}")
        return {"success": False, "reason": str(exc), "summary": {}}
    summary = _motion_summary(result)
    success = _result_ok(result)
    print(f"  Refinement hover execute: status={summary.get('status')} success={success}")
    return {"success": success, "summary": summary}


def _public_hover_solution(solution):
    out = {
        key: value
        for key, value in solution.items()
        if key not in ("batch_result", "best_candidate")
    }
    return out


@skill
def plan_refinement_hover(
    object_name,
    object_pos=None,
    side=None,
    bundlesdf_camera=None,
    hover_height_m=0.15,
    min_hover_m=0.10,
    max_hover_m=0.20,
    hover_yaws_deg=None,
    relaxed_rpy_offsets_deg=None,
    include_relaxed=True,
    **kwargs,
):
    """Plan and execute a batched top-down hover above a BundleSDF object pose.

    Returns (success, info). The selected arm is the closer arm to the top-camera
    object pose unless side is provided.
    """
    camera = bundlesdf_camera or kwargs.get("bundlesdf_camera", BUNDLESDF_CAMERA)
    det = None
    if object_pos is None:
        det = _detect_bundlesdf_pose(object_name, camera)
        object_pos = det.position_3d
    obj = np.asarray(object_pos, dtype=float).reshape(3)
    if side is None:
        side, distances = _choose_closer_arm_from_pose(obj)
    else:
        distances = {}

    hover_height = _clamped_hover_height(hover_height_m, min_hover_m, max_hover_m)
    hover_pos = obj.copy()
    hover_pos[2] += hover_height
    hover_planning_speed = float(kwargs.get("planning_speed", PLANNING_SPEED))
    strict_candidates = _build_hover_candidates(
        hover_pos,
        hover_yaws_deg=hover_yaws_deg,
        relaxed_rpy_offsets_deg=relaxed_rpy_offsets_deg,
        include_relaxed=False,
    )
    print(
        f"  Planning refinement hover: object={object_name!r}, side={side}, "
        f"camera={camera}, hover={hover_height:.3f}m, "
        f"speed={hover_planning_speed:.2f}, strict_candidates={len(strict_candidates)}"
    )
    solution = _select_hover_plan(strict_candidates, side, **kwargs)
    hover_phase = "strict_topdown"
    candidate_count = len(strict_candidates)
    strict_solution = _public_hover_solution(solution)
    if not solution.get("success") and include_relaxed:
        print(
            "  Strict top-down refinement hover failed; trying relaxed top-down "
            "roll/pitch/yaw offsets"
        )
        relaxed_candidates = [
            candidate
            for candidate in _build_hover_candidates(
                hover_pos,
                hover_yaws_deg=hover_yaws_deg,
                relaxed_rpy_offsets_deg=relaxed_rpy_offsets_deg,
                include_relaxed=True,
            )
            if str(candidate.get("label", "")).startswith("relaxed_")
        ]
        candidate_count += len(relaxed_candidates)
        solution = _select_hover_plan(relaxed_candidates, side, **kwargs)
        hover_phase = "relaxed_topdown"
    info = {
        "success": False,
        "object_name": object_name,
        "object_pos": [float(x) for x in obj.tolist()],
        "bundlesdf_camera": camera,
        "bundle_score": getattr(det, "score", None),
        "side": side,
        "distances": distances,
        "hover_height_m": hover_height,
        "hover_pos": [float(x) for x in hover_pos.tolist()],
        "hover_planning_speed": hover_planning_speed,
        "candidate_count": candidate_count,
        "hover_phase": hover_phase,
        "strict_hover_solution": strict_solution,
        "hover_solution": _public_hover_solution(solution),
    }
    if not solution.get("success"):
        info["reason"] = solution.get("reason", "hover plan failed")
        return False, info
    execute = _execute_hover_plan(solution, side, **kwargs)
    info["hover_execute"] = execute
    info["success"] = bool(execute.get("success"))
    if not info["success"]:
        info["reason"] = "hover execution failed"
    return info["success"], info


@skill
def refinement_grasp(
    object_name,
    bundlesdf_camera=None,
    wrist_camera=None,
    max_attempts=None,
    max_grasps=None,
    top_grasp_try=None,
    hover_height_m=0.15,
    min_hover_m=0.10,
    max_hover_m=0.20,
    **kwargs,
):
    """Hover from top-camera pose, sample AnyGrasp from the chosen wrist, then grasp.

    Returns (success, info). info["side"] is set when an arm was selected.
    """
    attempts = int(MAX_GRASP_ATTEMPTS if max_attempts is None else max_attempts)
    max_grasps = int(TOP_GRASP_MAX if max_grasps is None else max_grasps)
    top_grasp_try = int(TOP_GRASP_TRY if top_grasp_try is None else top_grasp_try)
    all_attempts = []

    for attempt in range(1, attempts + 1):
        print(f"\n  --- Refinement grasp attempt {attempt}/{attempts} for {object_name} ---")
        try:
            ok, hover_info = plan_refinement_hover(
                object_name,
                bundlesdf_camera=bundlesdf_camera,
                hover_height_m=hover_height_m,
                min_hover_m=min_hover_m,
                max_hover_m=max_hover_m,
                **kwargs,
            )
        except Exception as exc:
            hover_info = {"success": False, "reason": str(exc)}
            ok = False
        attempt_info = {"attempt": attempt, "hover": hover_info}
        all_attempts.append(attempt_info)
        if not ok:
            print(f"  Refinement hover failed: {hover_info.get('reason')}")
            continue

        side = hover_info["side"]
        camera = wrist_camera or side
        grasps = grasp_geometry.sample_anygrasp(
            object_name,
            camera=camera,
            max_grasps=max_grasps,
            tcp_offset_z_m=kwargs.get("tcp_offset_z_m", ANYGRASP_TCP_OFFSET_Z_M),
            disable_planner_z_clipping=kwargs.get(
                "disable_planner_z_clipping", ANYGRASP_DISABLE_PLANNER_Z_CLIPPING
            ),
            clip_min_z=kwargs.get("clip_min_z"),
        )
        attempt_info["wrist_camera"] = camera
        attempt_info["grasp_candidate_count"] = len(grasps or [])
        print(f"  Wrist AnyGrasp camera={camera}: candidates={len(grasps or [])}")
        if not grasps:
            attempt_info["reason"] = "no wrist AnyGrasp candidates"
            continue

        attempt_info["refinement_pregrasp_planning_speed"] = float(
            kwargs.get("planning_speed", PLANNING_SPEED)
        )
        attempt_info["refinement_final_approach_speed_mps"] = float(
            kwargs.get(
                "refinement_final_approach_speed_mps",
                DEFAULT_REFINEMENT_FINAL_APPROACH_SPEED_MPS,
            )
        )
        print(
            "  Refinement wrist pregrasp planning speed: "
            f"{float(kwargs.get('planning_speed', PLANNING_SPEED)):.2f}; "
            "final approach speed: "
            f"{float(kwargs.get('refinement_final_approach_speed_mps', DEFAULT_REFINEMENT_FINAL_APPROACH_SPEED_MPS)):.2f}m/s"
        )
        try:
            selected = _select_best_aligned_grasp(
                grasps[: min(top_grasp_try, len(grasps))],
                side,
                label=f"{camera} wrist AnyGrasp refinement",
                **kwargs,
            )
        except Exception as exc:
            print(f"  Wrist AnyGrasp ranking failed: {exc}")
            attempt_info["reason"] = str(exc)
            continue

        attempt_info["selected_grasp"] = {
            "position": [float(x) for x in selected.position],
            "rpy": [float(x) for x in selected.rpy],
            "score": float(selected.score),
            "width": float(selected.width),
            "trajectory_cache_key": getattr(selected, "trajectory_cache_key", None),
            "alignment_error_deg": getattr(selected, "alignment_error_deg", None),
            "pregrasp_position": [float(x) for x in selected.pregrasp_position],
            "approach_delta": [float(x) for x in selected.approach_delta],
        }
        if _execute_refinement_grasp_from_pregrasp(
            side,
            selected,
            label=f"{camera} wrist AnyGrasp refinement",
            **kwargs,
        ):
            return True, {
                "success": True,
                "side": side,
                "object_name": object_name,
                "wrist_camera": camera,
                "attempts": all_attempts,
            }
        attempt_info["reason"] = "grasp execution failed"

    return False, {
        "success": False,
        "side": all_attempts[-1]["hover"].get("side") if all_attempts else None,
        "object_name": object_name,
        "attempts": all_attempts,
        "reason": f"giving up after {attempts} attempts",
    }


def pick_object_with_refinement(object_name, **kwargs):
    """Compatibility wrapper returning the picked side or None."""
    success, info = refinement_grasp(object_name, **kwargs)
    return info.get("side") if success else None


def pick_and_place_with_refinement(object_name, target_name, **kwargs):
    print(f"\n{'=' * 50}")
    print(f"Picking: {object_name} -> {target_name} with refinement grasp")
    print(f"{'=' * 50}")
    success, info = refinement_grasp(object_name, **kwargs)
    side = info.get("side")
    if not success or side is None:
        try:
            go_home()
        except Exception:
            go_birdeye()
        print(f"  Failed to refinement-pick {object_name}: {info.get('reason')}")
        return False

    birdseye_pos, transport_rpy = birdseye_pose(side, **_birdseye_config(kwargs))
    print(
        f"  Moving {side} arm to BirdEyeView: "
        f"xyz={[round(float(x), 4) for x in birdseye_pos]}, "
        f"rpy={[round(float(x), 1) for x in transport_rpy]}"
    )
    _safe_move(side, birdseye_pos, transport_rpy, **kwargs)
    ok = _place_with_recovery(side, object_name, target_name, transport_rpy, **kwargs)
    if not ok:
        try:
            go_home()
        except Exception:
            go_birdeye()
        return False
    try:
        go_home()
    except Exception:
        go_birdeye()
    print(f"  {object_name} placed at {target_name}!")
    return bool(ok)
