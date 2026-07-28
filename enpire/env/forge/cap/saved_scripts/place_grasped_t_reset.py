# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset the PushT T to the recorded reset pose from the beginning.

Sequence: locate -> hover -> descend -> grasp -> lift -> move above reset ->
descend to reset release pose -> open gripper -> go home.

If the T is in the upside-down 180-degree configuration, first normalize it:
reverse-grasp -> lift/rotate 180 -> put down -> release, then run the standard
reset sequence from a fresh detection.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from skill_library.reset_t_skill import (
    RESET_OK_REFERENCE_IMAGE,
    RESET_OK_REFERENCE_META,
    RESET_T_HOLD_GRIPPER_POS,
    _as_rgb_uint8,
    _red_mask,
    _reset_ok_dominant_component,
    _reset_ok_load_reference,
    move_to_t_grasp_pose_v1,
    normalize_upside_down_t_v1,
    place_grasped_t_at_reset_v1,
    place_grasped_t_for_left_handoff_v1,
    reset_ok_v1,
)

SIDE = os.environ.get("PUSHT_RESET_SIDE", "auto").strip().lower() or "auto"
RIGHT_TO_LEFT_HANDOFF = os.environ.get(
    "PUSHT_RIGHT_TO_LEFT_HANDOFF", "1"
).strip().lower() not in {"0", "false", "no", "off"}
LOOP_COUNT = 10
RESET_OK_THRESHOLD = 0.4
DETECTION_RETRIES = 100
DETECTION_RETRY_SLEEP_S = 0.1
GRASP_HOVER_Z = 0.93
GRASP_WAIT_S = 0.2
LIFT_Z = 0.90
PRE_HOME_LIFT_M = 0.08
PRE_HOME_LIFT_MAX_Z = 0.93
DRY_RUN = os.environ.get("PUSHT_RESET_DRY_RUN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RESET_RESULT_PATH = os.environ.get("PUSHT_RESET_RESULT_PATH", "").strip()
FINAL_FRAME_DIR = os.environ.get("PUSHT_FINAL_FRAME_DIR", "").strip()
GOAL_IMAGE_PATH = os.environ.get("PUSHT_GOAL_IMAGE", "cap/tasks/pusht/goal_top.png").strip()
INITIAL_FRAME_PATH = os.environ.get("PUSHT_INITIAL_FRAME", "").strip()


def _state_ee_pos(state, side):
    direct = getattr(state, f"{side}_ee_pos", None)
    if direct is not None:
        return [float(v) for v in direct[:3]]
    if isinstance(state, dict):
        direct = state.get(f"{side}_ee_pos")
        if direct is not None:
            return [float(v) for v in direct[:3]]
    arms = state.get("arms", {}) if isinstance(state, dict) else getattr(state, "arms", {})
    arm = arms.get(side) if isinstance(arms, dict) else getattr(arms, side, None)
    if arm is None:
        return None
    ee_pos = arm.get("ee_pos") if isinstance(arm, dict) else getattr(arm, "ee_pos", None)
    if ee_pos is None:
        return None
    return [float(v) for v in ee_pos[:3]]


def _lift_before_home():
    if DRY_RUN:
        print("[place_grasped_t_reset] dry_run=True; skip pre-home lift")
        return {"success": True, "dry_run": True}
    move = globals().get("freespace_move")
    read_state = globals().get("get_robot_state")
    if not callable(move) or not callable(read_state):
        print("[place_grasped_t_reset] pre_home_lift skipped: missing freespace_move/get_robot_state")
        return {"success": False, "skipped": True, "reason": "missing_skill"}

    state = read_state()
    results = {}
    for side in ("left", "right"):
        pos = _state_ee_pos(state, side)
        if pos is None:
            results[side] = {"success": False, "skipped": True, "reason": "missing_ee_pos"}
            continue
        target = list(pos)
        target[2] = min(float(PRE_HOME_LIFT_MAX_Z), target[2] + float(PRE_HOME_LIFT_M))
        if target[2] <= pos[2] + 0.005:
            results[side] = {"success": True, "skipped": True, "reason": "already_high", "start": pos}
            continue
        print(f"[place_grasped_t_reset] pre_home_lift {side}: {pos} -> {target}")
        kwargs = {
            "side": side,
            f"{side}_target_pos": target,
            "planning_speed": 4.0,
            "backend": "rrt-connect",
            "planner_backend": "rrtconnect",
            "ik_rpy_weight": 0.0,
            "ik_error_threshold": 0.03,
        }
        try:
            result = move(**kwargs)
            ok = getattr(result, "status", None) in {None, "Success", "success", "done"}
            results[side] = {"success": bool(ok), "status": getattr(result, "status", None), "result": str(result)}
        except Exception as exc:
            results[side] = {"success": False, "error": str(exc)}
            print(f"[place_grasped_t_reset] pre_home_lift {side} failed: {exc}")
    return {"success": any(v.get("success") for v in results.values()), "results": results}


def _write_reset_result(success, **payload):
    if not RESET_RESULT_PATH:
        return
    path = Path(RESET_RESULT_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "success": bool(success),
        "time_s": time.time(),
        "final_frame_artifacts": globals().get("final_frame_artifacts", {}),
        **payload,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))


def _is_retryable_detection_error(exc):
    msg = str(exc)
    return (
        "No 8-corner red T contour found" in msg
        or "Largest red contour does not match T topology" in msg
        or "_pusht_" in msg
    )


def _go_home_at_start():
    if DRY_RUN:
        print("[place_grasped_t_reset] dry_run=True; skip initial go_home_fast")
        return {"success": True, "dry_run": True}
    lift_result = _lift_before_home()
    print(f"[place_grasped_t_reset] pre_home_lift result={lift_result}")
    fast_home = globals().get("go_home_fast")
    if callable(fast_home):
        print("[place_grasped_t_reset] initial_go_home_fast start")
        result = fast_home()
        print(f"[place_grasped_t_reset] initial_go_home_fast result={result}")
        return result
    home = globals().get("go_home")
    if callable(home):
        print("[place_grasped_t_reset] initial_go_home fallback start")
        result = home()
        print(f"[place_grasped_t_reset] initial_go_home result={result}")
        return result
    raise RuntimeError("Neither go_home_fast nor go_home is available in script namespace")


_go_home_at_start()


def _save_final_frame_and_overlay():
    if not FINAL_FRAME_DIR:
        print("[place_grasped_t_reset] final_frame_dir not set; skip final frame capture")
        return {}
    import cv2
    import numpy as np

    out_dir = Path(FINAL_FRAME_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = _as_rgb_uint8(get_camera_image("top"))
    final_path = out_dir / "final_frame.png"
    cv2.imwrite(str(final_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    goal_bgr = cv2.imread(str(Path(GOAL_IMAGE_PATH)), cv2.IMREAD_COLOR)
    if goal_bgr is None:
        raise FileNotFoundError(f"goal image not readable: {GOAL_IMAGE_PATH}")
    if goal_bgr.shape[:2] != bgr.shape[:2]:
        goal_bgr = cv2.resize(goal_bgr, (bgr.shape[1], bgr.shape[0]))
    goal_mask, _goal_info = _reset_ok_dominant_component(_red_mask(goal_bgr))
    overlay = bgr.copy()
    goal = goal_mask > 0
    overlay[goal] = (
        0.35 * overlay[goal] + 0.65 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay_path = out_dir / "t_with_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    _reference_crop, init_mask, _reference_info, crop = _reset_ok_load_reference(
        RESET_OK_REFERENCE_IMAGE,
        RESET_OK_REFERENCE_META,
        margin=90,
        target_shape=bgr.shape,
    )
    init_overlay = bgr.copy()
    x, y, w, h = crop
    init_crop = init_overlay[y : y + h, x : x + w]
    init_goal = init_mask > 0
    init_crop[init_goal] = (
        0.35 * init_crop[init_goal] + 0.65 * np.array([255, 160, 0])
    ).astype(np.uint8)
    cv2.rectangle(init_overlay, (x, y), (x + w, y + h), (255, 160, 0), 2)
    init_overlay_path = out_dir / "t_init_overlay.png"
    cv2.imwrite(str(init_overlay_path), init_overlay)

    init_final_overlay_path = None
    if INITIAL_FRAME_PATH:
        initial_bgr = cv2.imread(str(Path(INITIAL_FRAME_PATH)), cv2.IMREAD_COLOR)
        if initial_bgr is None:
            print(
                "[place_grasped_t_reset] init_final_overlay skipped: "
                f"initial frame not readable: {INITIAL_FRAME_PATH}"
            )
        else:
            if initial_bgr.shape[:2] != bgr.shape[:2]:
                initial_bgr = cv2.resize(initial_bgr, (bgr.shape[1], bgr.shape[0]))
            init_mask, _init_info = _reset_ok_dominant_component(_red_mask(initial_bgr))
            final_mask, _final_info = _reset_ok_dominant_component(_red_mask(bgr))
            init_final_overlay = bgr.copy()
            init_px = init_mask > 0
            final_px = final_mask > 0
            overlap_px = init_px & final_px
            init_only = init_px & ~final_px
            final_only = final_px & ~init_px
            init_final_overlay[init_only] = (
                0.35 * init_final_overlay[init_only] + 0.65 * np.array([255, 160, 0])
            ).astype(np.uint8)
            init_final_overlay[final_only] = (
                0.35 * init_final_overlay[final_only] + 0.65 * np.array([0, 255, 0])
            ).astype(np.uint8)
            init_final_overlay[overlap_px] = (
                0.35 * init_final_overlay[overlap_px] + 0.65 * np.array([255, 255, 255])
            ).astype(np.uint8)
            init_final_overlay_path = out_dir / "init_final_overlay.png"
            cv2.imwrite(str(init_final_overlay_path), init_final_overlay)
    print(
        "[place_grasped_t_reset] saved final frame artifacts "
        f"final_frame={final_path} overlay={overlay_path} "
        f"init_overlay={init_overlay_path} init_final_overlay={init_final_overlay_path}"
    )
    artifacts = {
        "final_frame": str(final_path),
        "t_with_overlay": str(overlay_path),
        "t_init_overlay": str(init_overlay_path),
    }
    if init_final_overlay_path is not None:
        artifacts["init_final_overlay"] = str(init_final_overlay_path)
        artifacts["initial_frame"] = str(Path(INITIAL_FRAME_PATH))
    return artifacts


final_frame_artifacts = {}
try:
    final_frame_artifacts = _save_final_frame_and_overlay()
except Exception as exc:
    print(f"[place_grasped_t_reset] final_frame_capture_failed={exc}")

reset_ok = False
last_reset_ok_log = {}
try:
    reset_ok, last_reset_ok_log = reset_ok_v1(threshold=RESET_OK_THRESHOLD)
    print(f"[place_grasped_t_reset] initial_reset_ok={reset_ok}")
    print(f"[place_grasped_t_reset] initial_reset_ok_log={last_reset_ok_log}")
except Exception as exc:
    print(f"[place_grasped_t_reset] initial_reset_ok_check_failed={exc}")

if reset_ok:
    print(
        "[place_grasped_t_reset] success=True "
        f"reason=already_reset_ok score={last_reset_ok_log.get('score')}"
    )
else:
    active_side = SIDE
    for iteration in range(1, LOOP_COUNT + 1):
        prefix = f"[place_grasped_t_reset] iter={iteration}/{LOOP_COUNT}"
        print(f"{prefix} start")

        normalize_ok = False
        normalize_log = {}
        normalize_error = None
        for attempt in range(1, DETECTION_RETRIES + 1):
            try:
                normalize_ok, normalize_log = normalize_upside_down_t_v1(
                    side=active_side,
                    hover_z=GRASP_HOVER_Z,
                    wait_s=GRASP_WAIT_S,
                    close_pos=RESET_T_HOLD_GRIPPER_POS,
                    dry_run=DRY_RUN,
                )
                normalize_error = None
                break
            except RuntimeError as exc:
                normalize_error = exc
                print(
                    f"{prefix} normalize_attempt={attempt}/{DETECTION_RETRIES} "
                    f"failed: {exc}"
                )
                if not _is_retryable_detection_error(exc):
                    raise
                if attempt < DETECTION_RETRIES and DETECTION_RETRY_SLEEP_S > 0:
                    time.sleep(DETECTION_RETRY_SLEEP_S)
        print(f"{prefix} normalize_success={normalize_ok}")
        print(f"{prefix} normalize_log={normalize_log}")

        if not normalize_ok:
            if normalize_error is not None:
                _write_reset_result(
                    False,
                    stage="normalize",
                    iteration=iteration,
                    detection_retries=DETECTION_RETRIES,
                    error=str(normalize_error),
                )
                raise RuntimeError(
                    "PushT reset failed during upside-down normalization "
                    f"after {DETECTION_RETRIES} detection attempts at iter={iteration}"
                ) from normalize_error
            _write_reset_result(
                False,
                stage="normalize",
                iteration=iteration,
                detection_retries=DETECTION_RETRIES,
                error="normalize returned False",
            )
            raise RuntimeError(
                f"PushT reset failed during upside-down normalization at iter={iteration}"
            )

        standard_pose = None
        if not normalize_log.get("normalized", False):
            standard_pose = normalize_log.get("pose")

        grasp_ok, grasp_log = move_to_t_grasp_pose_v1(
            side=active_side,
            hover_z=GRASP_HOVER_Z,
            wait_s=GRASP_WAIT_S,
            close_after_wait=True,
            close_pos=RESET_T_HOLD_GRIPPER_POS,
            dry_run=DRY_RUN,
            detected_pose=standard_pose,
        )
        print(f"{prefix} grasp_success={grasp_ok}")
        print(f"{prefix} grasp_log={grasp_log}")

        if not grasp_ok:
            _write_reset_result(
                False,
                stage="grasp",
                iteration=iteration,
                grasp_log=grasp_log,
            )
            raise RuntimeError(
                f"PushT reset failed during locate/hover/reach/grasp at iter={iteration}"
            )

        grasp_side = str(grasp_log.get("side", active_side)).strip().lower()
        if RIGHT_TO_LEFT_HANDOFF and grasp_side == "right":
            handoff_ok, handoff_log = place_grasped_t_for_left_handoff_v1(
                side="right",
                lift_z=LIFT_Z,
                dry_run=DRY_RUN,
            )
            print(f"{prefix} handoff_success={handoff_ok}")
            print(f"{prefix} handoff_log={handoff_log}")
            if not handoff_ok:
                _write_reset_result(
                    False,
                    stage="right_to_left_handoff",
                    iteration=iteration,
                    handoff_log=handoff_log,
                )
                raise RuntimeError(
                    f"PushT reset failed during right-to-left table handoff at iter={iteration}"
                )
            active_side = "left"
            print(f"{prefix} handoff_complete=True next_side=left")
            continue

        place_ok, place_log = place_grasped_t_at_reset_v1(
            side=grasp_side,
            lift_z=LIFT_Z,
            dry_run=DRY_RUN,
        )
        print(f"{prefix} place_success={place_ok}")
        print(f"{prefix} place_log={place_log}")

        if not place_ok:
            _write_reset_result(
                False,
                stage="place",
                iteration=iteration,
                place_log=place_log,
            )
            raise RuntimeError(
                f"PushT reset failed during lift/place/open/home at iter={iteration}"
            )

        reset_ok, last_reset_ok_log = reset_ok_v1(threshold=RESET_OK_THRESHOLD)
        print(f"{prefix} reset_ok={reset_ok}")
        print(f"{prefix} reset_ok_log={last_reset_ok_log}")

        if reset_ok:
            print(
                f"{prefix} success=True "
                f"score={last_reset_ok_log.get('score')} threshold={RESET_OK_THRESHOLD}"
            )
            break

        print(
            f"{prefix} success=False reason=reset_score_below_threshold "
            f"score={last_reset_ok_log.get('score')} threshold={RESET_OK_THRESHOLD}"
        )
    else:
        _write_reset_result(
            False,
            stage="reset_ok",
            loop_count=LOOP_COUNT,
            last_reset_ok_log=last_reset_ok_log,
        )
        raise RuntimeError(
            "PushT reset failed: reset_ok score stayed below threshold "
            f"after {LOOP_COUNT} iterations; last_log={last_reset_ok_log}"
        )

_write_reset_result(
    True,
    stage="done",
    score=last_reset_ok_log.get("score"),
    threshold=RESET_OK_THRESHOLD,
    reset_ok_log=last_reset_ok_log,
)
print(
    "[place_grasped_t_reset] success=True "
    f"max_loop_count={LOOP_COUNT} final_score={last_reset_ok_log.get('score')}"
)

