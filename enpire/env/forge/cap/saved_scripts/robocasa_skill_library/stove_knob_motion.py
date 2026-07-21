# stove_knob_motion.py - TurnOffStove hover, contact, and rotation helpers
from skill_library.namespace import *  # noqa: F401, F403

import math
import time

import numpy as np

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.stove_knob_common import *  # noqa: F401, F403
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.stove_knob_detection import (  # noqa: E402
    show_stove_debug_markers_v1,
)


def refresh_turn_off_stove_planner_world_v1(reason):
    world = update_planner_world()
    print(f"[curobo] {reason}: {world['n_obstacles']} obstacles")
    return world


def move_to_stove_knob_hover_v1(
    *,
    side,
    stove_state,
    hover_offsets_m,
    roll_candidates_deg=STOVE_APPROACH_ROLL_CANDIDATES_DEG_V1,
    base_backward_retries=0,
    base_backward_distance_m=0.10,
    base_backward_action=-1.0,
    base_backward_chunk_steps=3,
    base_backward_max_chunks=20,
    base_backward_settle_s=0.0,
    pre_rotate_total_clockwise_deg=180.0,
    pre_rotate_angle_deg=90.0,
    hover_pos_tol_m=STOVE_HOVER_POS_TOL_M_V1,
    hover_rot_tol_deg=STOVE_HOVER_ROT_TOL_DEG_V1,
    hover_pose_retries=STOVE_HOVER_POSE_RETRIES_V1,
):
    print("  Opening gripper for hover")
    open_gripper(side)
    time.sleep(0.1)

    selected = stove_state["selected_candidate"]
    knob_pos = np.asarray(selected["pos"], dtype=float)
    approach = np.asarray(stove_state["approach_direction"], dtype=float)
    attempts = []
    max_round = int(base_backward_retries) + 1
    for retry_idx in range(max_round):
        if retry_idx > 0:
            result = move_stove_base_backward_v1(
                distance_m=base_backward_distance_m,
                action=base_backward_action,
                chunk_steps=base_backward_chunk_steps,
                max_chunks=base_backward_max_chunks,
                retry_idx=retry_idx,
                retry_total=int(base_backward_retries),
            )
            stove_state.setdefault("base_backward_results", []).append(result)
            settle_s = float(base_backward_settle_s)
            if settle_s > 0.0:
                settle_result = settle_stove_base_v1(
                    settle_s,
                    retry_idx=retry_idx,
                    retry_total=int(base_backward_retries),
                )
                stove_state.setdefault("base_settle_results", []).append(settle_result)
            refresh_turn_off_stove_planner_world_v1(
                f"stove hover base retry {retry_idx}"
            )
        for hover_offset_m in hover_offsets_m:
            target_pos = knob_pos - approach * float(hover_offset_m)
            for roll_deg in roll_candidates_deg:
                quat = make_gripper_z_quat_v1(approach, roll_deg=roll_deg)
                for pose_retry in range(int(hover_pose_retries) + 1):
                    result = freespace_move_world_pose_v1(
                        side,
                        target_pos,
                        quat,
                        auto_update_world=True,
                    )
                    status = str(getattr(result, "status", result))
                    attempts.append(
                        (
                            retry_idx,
                            float(hover_offset_m),
                            float(roll_deg),
                            pose_retry,
                            status,
                        )
                    )
                    print(
                        "stove_hover "
                        f"base_retry={retry_idx}/{int(base_backward_retries)} "
                        f"offset={float(hover_offset_m):.3f} "
                        f"roll={float(roll_deg):.0f} "
                        f"pose_retry={pose_retry}/{int(hover_pose_retries)}: "
                        f"status={status} "
                        f"target={fmt_xyz_v1(target_pos)} "
                        f"quat={fmt_xyz_v1(quat, 4)}"
                    )
                    if status != "Success":
                        break

                    check = check_stove_hover_pose_v1(
                        side,
                        target_pos,
                        quat,
                        pos_tol_m=hover_pos_tol_m,
                        rot_tol_deg=hover_rot_tol_deg,
                        label="stove_hover_pose_check",
                    )
                    stove_state.setdefault("hover_pose_checks", []).append(check)
                    if not check["ok"]:
                        if pose_retry < int(hover_pose_retries):
                            print("  Hover pose not reached; retrying same pose")
                            continue
                        break

                    stove_state["hover_offset_m"] = float(hover_offset_m)
                    stove_state["hover_pos"] = np.asarray(target_pos, dtype=float)
                    stove_state["quat"] = np.asarray(quat, dtype=float)
                    stove_state["approach_roll_deg"] = float(roll_deg)
                    stove_state["hover_base_retry"] = retry_idx
                    stove_state["hover_attempts"] = attempts
                    show_stove_debug_markers_v1(stove_state)
                    pre_result = pre_rotate_if_clockwise_limited_v1(
                        side=side,
                        total_clockwise_deg=pre_rotate_total_clockwise_deg,
                        pre_rotate_angle_deg=pre_rotate_angle_deg,
                    )
                    if pre_result is not None:
                        stove_state["pre_rotate_result"] = pre_result
                        post_check = check_stove_hover_pose_v1(
                            side,
                            target_pos,
                            quat,
                            pos_tol_m=hover_pos_tol_m,
                            rot_tol_deg=hover_rot_tol_deg,
                            label="stove_hover_pose_after_pre_rotate",
                        )
                        stove_state.setdefault("hover_pose_checks", []).append(
                            post_check
                        )
                        if not post_check["ok"]:
                            print(
                                "  Pre-rotate moved away from hover; retrying hover"
                            )
                            continue
                    return stove_state

    stove_state["hover_attempts"] = attempts
    raise RuntimeError(
        "failed to reach stove knob hover; last="
        + (str(attempts[-1]) if attempts else "none")
    )


def move_stove_base_backward_v1(
    *,
    distance_m,
    action,
    chunk_steps,
    max_chunks,
    retry_idx,
    retry_total,
):
    if "execute_base_trajectory" not in globals():
        raise RuntimeError("execute_base_trajectory is not available")
    base0 = current_base_pos_v1()
    if base0 is None:
        raise RuntimeError("base position is unavailable")

    target_m = float(distance_m)
    stop_tol_m = min(0.010, max(0.003, target_m * 0.10))
    moved_m = 0.0
    result = None
    chunks = 0
    while moved_m < target_m - stop_tol_m and chunks < int(max_chunks):
        chunks += 1
        # Fourth field asks the base executor for a short 2-sim-step pulse.
        actions = [[float(action), 0.0, 0.0, 2.0]] * int(chunk_steps)
        result = execute_base_trajectory(actions)
        base_now = current_base_pos_v1()
        if base_now is None:
            raise RuntimeError("base position became unavailable")
        moved_m = float(np.linalg.norm(base_now[:2] - base0[:2]))

    base1 = current_base_pos_v1()
    print(
        "stove_hover_base_backward "
        f"retry={int(retry_idx)}/{int(retry_total)} "
        f"target_distance_m={target_m:.3f} "
        f"moved_m={moved_m:.3f} "
        f"stop_tol_m={stop_tol_m:.3f} "
        f"action={float(action):.3f} "
        f"chunk_steps={int(chunk_steps)} sim_steps_per_action=2 "
        f"chunks={chunks}/{int(max_chunks)} "
        f"base_before={fmt_xyz_v1(base0)} "
        f"base_after={fmt_xyz_v1(base1) if base1 is not None else []}"
    )
    return result


def settle_stove_base_v1(settle_s, *, retry_idx, retry_total):
    if "execute_base_trajectory" not in globals():
        raise RuntimeError("execute_base_trajectory is not available")
    settle_s = float(settle_s)
    # RoboCasa runs at 20 Hz. Each action below requests two sim steps, so
    # 10 actions per second makes the pause visible in direct-mode videos.
    n_actions = max(1, int(math.ceil(settle_s * 10.0)))
    result = execute_base_trajectory([[0.0, 0.0, 0.0, 2.0]] * n_actions)
    print(
        "stove_hover_base_backward_settle "
        f"retry={int(retry_idx)}/{int(retry_total)} "
        f"sim_s={settle_s:.2f} "
        f"actions={n_actions} sim_steps_per_action=2"
    )
    return result


def current_base_pos_v1():
    if "get_state" in globals():
        try:
            state = get_state()
            if "base_pos" in state:
                return np.asarray(state["base_pos"], dtype=float)
        except Exception:
            pass
    if "get_base_state" in globals():
        try:
            state = get_base_state()
            if "pos" in state:
                return np.asarray(state["pos"], dtype=float)
        except Exception:
            pass
    return None


def nudge_stove_knob_approach_v1(
    *,
    side,
    stove_state,
    step_m,
    extra_after_target_m,
    backoff_after_m=0.0,
):
    approach = np.asarray(stove_state["approach_direction"], dtype=float)
    hover_offset_m = float(stove_state.get("hover_offset_m", 0.10))
    total_m = max(float(step_m), hover_offset_m + float(extra_after_target_m))
    n_steps = max(1, int(math.ceil(total_m / float(step_m))))
    delta = approach * total_m
    result = nudge_brutal(side=side, delta_pos=delta.tolist(), n_steps=n_steps)
    final_pos = np.asarray(getattr(result, "final_pos", []), dtype=float)
    print(
        "stove_approach_nudge: "
        f"hover_offset={hover_offset_m:.3f} "
        f"extra_after_target={float(extra_after_target_m):.3f} "
        f"total_m={total_m:.3f} "
        f"step_m={float(step_m):.3f} "
        f"n_steps={n_steps} "
        f"success={bool(getattr(result, 'success', False))} "
        f"delta={fmt_xyz_v1(delta)} "
        f"final_pos={fmt_xyz_v1(final_pos) if final_pos.size else []}"
    )
    stove_state["approach_nudge_result"] = result
    stove_state["approach_nudge_total_m"] = total_m
    stove_state["approach_nudge_steps"] = n_steps
    backoff_m = max(0.0, float(backoff_after_m))
    if backoff_m > 0.0:
        backoff_delta = -approach * backoff_m
        backoff_result = nudge_brutal(
            side=side,
            delta_pos=backoff_delta.tolist(),
            n_steps=1,
        )
        backoff_final_pos = np.asarray(
            getattr(backoff_result, "final_pos", []),
            dtype=float,
        )
        print(
            "stove_approach_backoff: "
            f"distance_m={backoff_m:.3f} "
            f"delta={fmt_xyz_v1(backoff_delta)} "
            f"success={bool(getattr(backoff_result, 'success', False))} "
            f"final_pos={fmt_xyz_v1(backoff_final_pos) if backoff_final_pos.size else []}"
        )
        stove_state["approach_backoff_result"] = backoff_result
        stove_state["approach_backoff_m"] = backoff_m
    return result


def close_and_rotate_stove_knob_clockwise_v1(
    *,
    side,
    angle_deg=90.0,
    runtime_direction="clockwise",
    close_first=True,
):
    if close_first:
        close_gripper(side)
        time.sleep(0.15)
    result = rotate_gripper_in_place(
        side=side,
        angle_deg=float(angle_deg),
        direction=runtime_direction,
        axis="local_z",
        n_steps=None,
        settle_steps=5,
        gripper=0.0,
    )
    print(
        "stove_close_rotate: "
        f"requested_visual=clockwise "
        f"runtime_direction={runtime_direction} "
        f"close_first={bool(close_first)} "
        f"status={result.get('status', 'Unknown')} "
        f"requested_angle_deg={float(result.get('requested_angle_deg', 0.0)):.1f} "
        f"angle_deg={float(result.get('angle_deg', 0.0)):.1f} "
        f"limited={bool(result.get('limited_by_joint_limit', False))} "
        f"pos_drift_m={float(result.get('pos_drift_m', 0.0)):.4f} "
        f"rot_error_deg={float(result.get('rot_error_deg', 0.0)):.1f}"
    )
    return result


def pre_rotate_if_clockwise_limited_v1(
    *,
    side,
    total_clockwise_deg,
    pre_rotate_angle_deg,
):
    joint7 = current_joint7_v1(side)
    if joint7 is None:
        print("  pre-rotate check skipped: no joint7 in robot state")
        return None

    lower = -2.8973
    upper = 2.8973
    margin = 0.0873
    safe_upper = upper - margin
    available_visual_clockwise_deg = math.degrees(max(0.0, safe_upper - joint7))
    if available_visual_clockwise_deg >= float(total_clockwise_deg):
        print(
            "  pre-rotate check: "
            f"joint7={joint7:.3f} "
            f"available_visual_clockwise={available_visual_clockwise_deg:.1f} "
            f"need={float(total_clockwise_deg):.1f}; no pre-rotate"
        )
        return None

    print(
        "  pre-rotate check: "
        f"joint7={joint7:.3f} "
        f"available_visual_clockwise={available_visual_clockwise_deg:.1f} "
        f"need={float(total_clockwise_deg):.1f}; "
        f"open + runtime-clockwise {float(pre_rotate_angle_deg):.1f} deg"
    )
    open_gripper(side)
    time.sleep(0.1)
    result = rotate_gripper_in_place(
        side=side,
        angle_deg=float(pre_rotate_angle_deg),
        direction="clockwise",
        axis="local_z",
        n_steps=None,
        settle_steps=5,
        gripper=1.0,
    )
    print(
        "  pre_rotate_open: "
        f"status={result.get('status', 'Unknown')} "
        f"angle_deg={float(result.get('angle_deg', 0.0)):.1f} "
        f"pos_drift_m={float(result.get('pos_drift_m', 0.0)):.4f}"
    )
    return result


def freespace_move_world_pose_v1(
    side,
    target_pos,
    target_quat,
    *,
    auto_update_world=False,
):
    return freespace_move(
        right_target_pos=np.asarray(target_pos, dtype=float).tolist(),
        right_target_quat=np.asarray(target_quat, dtype=float).tolist(),
        side=side,
        auto_update_world=bool(auto_update_world),
    )


def current_joint7_v1(side):
    try:
        joint_pos = np.asarray(get_robot_state().arms[side].joint_pos, dtype=float)
    except Exception:
        return None
    if joint_pos.size < 7:
        return None
    return float(joint_pos[-1])


def check_stove_hover_pose_v1(
    side,
    target_pos,
    target_quat,
    *,
    pos_tol_m=STOVE_HOVER_POS_TOL_M_V1,
    rot_tol_deg=STOVE_HOVER_ROT_TOL_DEG_V1,
    label="stove_hover_pose_check",
):
    arm = get_robot_state().arms[side]
    ee_pos = np.asarray(arm.ee_pos, dtype=float)
    ee_quat = np.asarray(
        getattr(arm, "ee_quat", getattr(arm, "ee_quat_xyzw", [0.0, 0.0, 0.0, 1.0])),
        dtype=float,
    )
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)
    pos_error_m = float(np.linalg.norm(ee_pos - target_pos))
    rot_error_deg = quat_angle_error_deg_v1(ee_quat, target_quat)
    ok = (
        pos_error_m <= float(pos_tol_m)
        and rot_error_deg <= float(rot_tol_deg)
    )
    print(
        f"{label}: "
        f"{'reached' if ok else 'not_reached'} "
        f"pos_error={pos_error_m:.3f}m/{float(pos_tol_m):.3f} "
        f"rot_error={rot_error_deg:.1f}deg/{float(rot_tol_deg):.1f} "
        f"target={fmt_xyz_v1(target_pos)} "
        f"ee={fmt_xyz_v1(ee_pos)}"
    )
    return {
        "ok": ok,
        "pos_error_m": pos_error_m,
        "rot_error_deg": rot_error_deg,
        "target_pos": target_pos,
        "target_quat": target_quat,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
    }


