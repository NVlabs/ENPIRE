# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Press a successfully inserted GPU, then run the existing GPU unplug reset.

This script is intended for the opt-in RL full-cycle launcher.  It assumes the
left gripper is holding the GPU at the end of a successful insertion episode.
It then applies a small bounded downward press, releases/retracts the gripper,
goes home, and delegates unplugging to ``gpu_reset.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or raw == "" else raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None or raw == "" else raw)


PRESS_STRATEGY = (
    os.environ.get("GPU_INSERT_PRESS_STRATEGY", "edge_pair").strip().lower()
    or "edge_pair"
)
PRESS_DOWN_M = _env_float("GPU_INSERT_PRESS_DOWN_M", 0.020)
PRESS_DURATION_S = _env_float("GPU_INSERT_PRESS_DURATION_S", 1.25)
PRESS_STEPS = max(2, _env_int("GPU_INSERT_PRESS_STEPS", 14))
PRESS_HOLD_S = _env_float("GPU_INSERT_PRESS_HOLD_S", 0.75)
PRESS_VERTICAL_ONLY = _env_flag("GPU_INSERT_PRESS_VERTICAL_ONLY", True)
PRESS_MAX_XY_DRIFT_M = _env_float("GPU_INSERT_PRESS_MAX_XY_DRIFT_M", 0.003)
PRESS_STRICT_VERTICAL = _env_flag("GPU_INSERT_PRESS_STRICT_VERTICAL", False)
EXTRA_FIRST_EDGE_PRESS_COUNT = max(
    0,
    _env_int(
        "GPU_INSERT_EXTRA_FIRST_EDGE_PRESS_COUNT",
        _env_int("GPU_INSERT_EXTRA_TOP_PRESS_COUNT", 1),
    ),
)
POST_CLOSE_SETTLE_S = _env_float("GPU_INSERT_PRESS_POST_CLOSE_SETTLE_S", 0.15)
POST_OPEN_SETTLE_S = _env_float("GPU_INSERT_PRESS_POST_OPEN_SETTLE_S", 0.15)
EDGE_PRESS_X_M = _env_float("GPU_INSERT_EDGE_PRESS_X_M", 0.028)
FIRST_EDGE_PRESS_EXTRA_X_M = _env_float(
    "GPU_INSERT_FIRST_EDGE_PRESS_EXTRA_X_M",
    0.014,
)
# Dedicated -x offset for the repeat (3rd+) edge presses so they can land
# independently of the initial -x press (which uses first_x_offset).
REPEAT_EDGE_PRESS_X_M = _env_float("GPU_INSERT_REPEAT_EDGE_PRESS_X_M", 0.045)
EDGE_PRESS_CLEAR_Z_M = _env_float("GPU_INSERT_EDGE_PRESS_CLEAR_Z_M", 0.006)
EDGE_PRESS_MOVE_DURATION_S = _env_float("GPU_INSERT_EDGE_PRESS_MOVE_DURATION_S", 0.45)
EDGE_PRESS_MOVE_STEPS = max(2, _env_int("GPU_INSERT_EDGE_PRESS_MOVE_STEPS", 8))
RETRACT_M = _env_float("GPU_INSERT_PRESS_RETRACT_M", 0.100)
RETRACT_DURATION_S = _env_float("GPU_INSERT_PRESS_RETRACT_DURATION_S", 0.55)
RETRACT_STEPS = max(2, _env_int("GPU_INSERT_PRESS_RETRACT_STEPS", 8))
LEFT_CLOSE_VEL_LIMIT = _env_float("GPU_INSERT_PRESS_LEFT_CLOSE_VEL_LIMIT", 1.0)
LEFT_CLOSE_TORQUE_LIMIT = _env_float("GPU_INSERT_PRESS_LEFT_CLOSE_TORQUE_LIMIT", 1.8)
LEFT_OPEN_VEL_LIMIT = _env_float("GPU_INSERT_PRESS_LEFT_OPEN_VEL_LIMIT", 3.0)
LEFT_REPOSITION_OPEN_DELTA = _env_float("GPU_INSERT_PRESS_REPOSITION_OPEN_DELTA", 0.28)
LEFT_REPOSITION_OPEN_MAX_POS = _env_float("GPU_INSERT_PRESS_REPOSITION_OPEN_MAX_POS", 0.65)
LEFT_REPOSITION_OPEN_MIN_POS = _env_float("GPU_INSERT_PRESS_REPOSITION_OPEN_MIN_POS", 0.16)
LEFT_REPOSITION_OPEN_DURATION_S = _env_float(
    "GPU_INSERT_PRESS_REPOSITION_OPEN_DURATION_S",
    0.25,
)
LEFT_FINAL_OPEN_DELTA = _env_float("GPU_INSERT_PRESS_FINAL_OPEN_DELTA", 0.18)
LEFT_FINAL_OPEN_MAX_POS = _env_float("GPU_INSERT_PRESS_FINAL_OPEN_MAX_POS", 0.28)
LEFT_FINAL_OPEN_MIN_POS = _env_float("GPU_INSERT_PRESS_FINAL_OPEN_MIN_POS", 0.24)
LEFT_FINAL_OPEN_DURATION_S = _env_float(
    "GPU_INSERT_PRESS_FINAL_OPEN_DURATION_S",
    0.25,
)
OPEN_AFTER_PRESS = _env_flag("GPU_INSERT_PRESS_OPEN_AFTER", True)
GO_HOME_AFTER_PRESS = _env_flag("GPU_INSERT_PRESS_GO_HOME", True)
RUN_UNPLUG_AFTER_PRESS = _env_flag("GPU_INSERT_PRESS_RUN_UNPLUG", True)
DRY_RUN = _env_flag("GPU_INSERT_PRESS_DRY_RUN", False)
SCRIPT_COMPLETED = False


def _tool_env_from_callable(fn):
    seen = set()

    def _search(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        env = getattr(obj, "_env", None)
        if env is not None:
            return env
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            found = _search(wrapped)
            if found is not None:
                return found
        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    found = _search(cell.cell_contents)
                except ValueError:
                    continue
                if found is not None:
                    return found
        return None

    return _search(fn)


def _direct_env():
    env = _tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for GPU insertion press")
    if not hasattr(env, "_kin_lock") or not hasattr(env, "kin"):
        raise RuntimeError("direct YAM env is missing kinematics support for press move")
    return env


def _move_left_to_world_pos(
    target_pos,
    *,
    duration_s: float,
    steps: int,
    label: str,
    lock_xy=None,
) -> np.ndarray:
    env = _direct_env()
    obs_left = env.get_observations("left")
    obs_right = env.get_observations("right")
    left_jp = np.asarray(obs_left["joint_pos"], dtype=np.float64).reshape(6)
    right_jp = np.asarray(obs_right["joint_pos"], dtype=np.float64).reshape(6)
    left_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gp = float(np.asarray(obs_right["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    left_start_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    left_start_quat = np.asarray(obs_left["ee_quat"], dtype=np.float64).reshape(4)
    right_hold_pos = np.asarray(obs_right["ee_pos"], dtype=np.float64).reshape(3)
    right_hold_quat = np.asarray(obs_right["ee_quat"], dtype=np.float64).reshape(4)
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    lock_xy_arr = None
    if lock_xy is not None:
        lock_xy_arr = np.asarray(lock_xy, dtype=np.float64).reshape(2)
        target_pos = target_pos.copy()
        target_pos[:2] = lock_xy_arr

    print(
        f"[gpu_press_then_unplug] {label}: "
        f"start={[round(float(v), 4) for v in left_start_pos.tolist()]} "
        f"target={[round(float(v), 4) for v in target_pos.tolist()]} "
        f"duration_s={float(duration_s):.2f} steps={int(steps)}"
        + (
            " lock_xy="
            f"{[round(float(v), 4) for v in lock_xy_arr.tolist()]}"
            if lock_xy_arr is not None
            else ""
        )
    )
    if DRY_RUN:
        return left_start_pos

    left_waypoints = [left_jp.copy()]
    right_waypoints = [right_jp.copy()]
    left_grippers = [[left_gp]]
    right_grippers = [[right_gp]]
    timestamps = [0.0]
    cur_left_jp = left_jp.copy()
    cur_right_jp = right_jp.copy()
    with env._kin_lock:
        env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
        for step_index in range(1, int(steps) + 1):
            alpha = float(step_index) / float(steps)
            interp_pos = left_start_pos + alpha * (target_pos - left_start_pos)
            if lock_xy_arr is not None:
                interp_pos[:2] = lock_xy_arr
            env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
            next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                interp_pos,
                left_start_quat,
                right_hold_pos,
                right_hold_quat,
                seeded=True,
                dt=0.01,
                solver="daqp",
                damping=1e-3,
                err_threshold=1e-4,
                max_iters=40,
            )
            cur_left_jp = np.asarray(next_left_jp, dtype=np.float64).reshape(6)
            cur_right_jp = np.asarray(next_right_jp, dtype=np.float64).reshape(6)
            left_waypoints.append(cur_left_jp.copy())
            right_waypoints.append(cur_right_jp.copy())
            left_grippers.append([left_gp])
            right_grippers.append([right_gp])
            timestamps.append(alpha * float(duration_s))

    result = env.move_bimanual_joint_keypoints(
        timestamps=timestamps,
        left_joint_positions=left_waypoints,
        right_joint_positions=right_waypoints,
        left_gripper_positions=left_grippers,
        right_gripper_positions=right_grippers,
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.0,
    )
    if not bool(result.get("success", False)):
        raise RuntimeError(result.get("reason", f"{label} failed"))
    obs_after = env.get_observations("left")
    actual_pos = np.asarray(obs_after["ee_pos"], dtype=np.float64).reshape(3)
    print(
        f"[gpu_press_then_unplug] {label} actual: "
        f"pos={[round(float(v), 4) for v in actual_pos.tolist()]} "
        f"delta={[round(float(v), 4) for v in (actual_pos - left_start_pos).tolist()]} "
        f"target_error_m={float(np.linalg.norm(actual_pos - target_pos)):.4f}"
    )
    return actual_pos


def _move_left_world_delta(
    delta_xyz_m,
    *,
    duration_s: float,
    steps: int,
    label: str,
) -> np.ndarray:
    env = _direct_env()
    obs_left = env.get_observations("left")
    left_start_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    delta = np.asarray(delta_xyz_m, dtype=np.float64).reshape(3)
    return _move_left_to_world_pos(
        left_start_pos + delta,
        duration_s=duration_s,
        steps=steps,
        label=label,
    )


def _move_left_world_z(
    delta_z_m: float,
    *,
    duration_s: float,
    steps: int,
    label: str,
) -> np.ndarray:
    return _move_left_world_delta(
        [0.0, 0.0, float(delta_z_m)],
        duration_s=duration_s,
        steps=steps,
        label=label,
    )


def _move_left_vertical_press(
    delta_z_m: float,
    *,
    duration_s: float,
    steps: int,
    label: str,
) -> np.ndarray:
    env = _direct_env()
    obs_left = env.get_observations("left")
    start_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    target_pos = start_pos.copy()
    target_pos[2] += float(delta_z_m)

    actual_pos = _move_left_to_world_pos(
        target_pos,
        duration_s=duration_s,
        steps=steps,
        label=f"{label} vertical-only",
        lock_xy=start_pos[:2],
    )
    if DRY_RUN:
        return actual_pos

    xy_drift_m = float(np.linalg.norm(actual_pos[:2] - start_pos[:2]))
    actual_z_delta_m = float(actual_pos[2] - start_pos[2])
    print(
        "[gpu_press_then_unplug] vertical press check: "
        f"label={label!r} "
        f"xy_drift_m={xy_drift_m:.4f} "
        f"max_xy_drift_m={float(PRESS_MAX_XY_DRIFT_M):.4f} "
        f"commanded_z_delta_m={float(delta_z_m):.4f} "
        f"actual_z_delta_m={actual_z_delta_m:.4f}"
    )
    if xy_drift_m > float(PRESS_MAX_XY_DRIFT_M):
        msg = (
            "vertical press exceeded XY drift limit: "
            f"label={label!r} xy_drift_m={xy_drift_m:.4f} "
            f"max_xy_drift_m={float(PRESS_MAX_XY_DRIFT_M):.4f}"
        )
        if PRESS_STRICT_VERTICAL:
            raise RuntimeError(msg)
        print(f"[gpu_press_then_unplug] WARNING: {msg}")
    return actual_pos


def _open_left_fully(label: str) -> None:
    print(
        f"[gpu_press_then_unplug] {label}: full open left gripper "
        f"vel_limit={float(LEFT_OPEN_VEL_LIMIT):.3f}"
    )
    if not DRY_RUN:
        open_gripper("left", vel_limit=float(LEFT_OPEN_VEL_LIMIT))
    if float(POST_OPEN_SETTLE_S) > 0.0:
        time.sleep(float(POST_OPEN_SETTLE_S))


def _move_left_gripper_position(target_gp: float, *, duration_s: float, label: str) -> float:
    env = _direct_env()
    target_gp = min(1.0, max(0.0, float(target_gp)))
    duration_s = max(0.1, float(duration_s))
    obs_left = env.get_observations("left")
    obs_right = env.get_observations("right")
    left_jp = np.asarray(obs_left["joint_pos"], dtype=np.float64).reshape(6)
    right_jp = np.asarray(obs_right["joint_pos"], dtype=np.float64).reshape(6)
    left_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gp = float(np.asarray(obs_right["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    if abs(left_gp - target_gp) <= 0.01:
        print(
            f"[gpu_press_then_unplug] {label}: left gripper already near target "
            f"(current={left_gp:.3f}, target={target_gp:.3f})"
        )
        return left_gp

    print(
        f"[gpu_press_then_unplug] {label}: direct gripper position move "
        f"current={left_gp:.3f} target={target_gp:.3f} "
        f"duration_s={duration_s:.2f}"
    )
    result = env.move_bimanual_joint_keypoints(
        timestamps=[0.0, duration_s],
        left_joint_positions=[left_jp.tolist(), left_jp.tolist()],
        right_joint_positions=[right_jp.tolist(), right_jp.tolist()],
        left_gripper_positions=[[left_gp], [target_gp]],
        right_gripper_positions=[[right_gp], [right_gp]],
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.0,
    )
    if not bool(result.get("success", False)):
        raise RuntimeError(result.get("reason", f"{label} gripper position move failed"))
    actual_gp = float(
        np.asarray(
            env.get_observations("left")["gripper_pos"],
            dtype=np.float64,
        ).reshape(-1)[0]
    )
    print(
        f"[gpu_press_then_unplug] {label}: direct gripper actual "
        f"pos={actual_gp:.3f} target_error={abs(actual_gp - target_gp):.3f}"
    )
    return actual_gp


def _release_left_for_reposition(label: str) -> None:
    _release_left_to_target(
        label,
        open_delta=float(LEFT_REPOSITION_OPEN_DELTA),
        min_pos=float(LEFT_REPOSITION_OPEN_MIN_POS),
        max_pos=float(LEFT_REPOSITION_OPEN_MAX_POS),
        duration_s=float(LEFT_REPOSITION_OPEN_DURATION_S),
    )


def _release_left_to_target(
    label: str,
    *,
    open_delta: float,
    min_pos: float,
    max_pos: float,
    duration_s: float,
) -> None:
    env = _direct_env()
    obs_left = env.get_observations("left")
    current_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    target_gp = min(
        float(max_pos),
        max(
            float(min_pos),
            current_gp + max(0.0, float(open_delta)),
        ),
    )
    print(
        f"[gpu_press_then_unplug] {label}: partial release left gripper "
        f"current={current_gp:.3f} target={target_gp:.3f} "
        f"delta={float(open_delta):.3f} "
        f"max={float(max_pos):.3f} "
        f"vel_limit={float(LEFT_OPEN_VEL_LIMIT):.3f}"
    )
    if not DRY_RUN:
        _move_left_gripper_position(
            target_gp,
            duration_s=float(duration_s),
            label=label,
        )
    if float(POST_OPEN_SETTLE_S) > 0.0:
        time.sleep(float(POST_OPEN_SETTLE_S))


def _close_left_for_press() -> None:
    print(
        "[gpu_press_then_unplug] firm close left gripper: "
        f"vel_limit={float(LEFT_CLOSE_VEL_LIMIT):.3f} "
        f"torque_limit={float(LEFT_CLOSE_TORQUE_LIMIT):.3f}"
    )
    if not DRY_RUN:
        close_gripper(
            "left",
            vel_limit=float(LEFT_CLOSE_VEL_LIMIT),
            torque_limit=float(LEFT_CLOSE_TORQUE_LIMIT),
        )
    if float(POST_CLOSE_SETTLE_S) > 0.0:
        time.sleep(float(POST_CLOSE_SETTLE_S))


def _release_and_retract() -> None:
    if OPEN_AFTER_PRESS:
        _release_left_to_target(
            "small release after final press",
            open_delta=float(LEFT_FINAL_OPEN_DELTA),
            min_pos=float(LEFT_FINAL_OPEN_MIN_POS),
            max_pos=float(LEFT_FINAL_OPEN_MAX_POS),
            duration_s=float(LEFT_FINAL_OPEN_DURATION_S),
        )
    if float(RETRACT_M) > 0.0:
        _move_left_world_z(
            abs(float(RETRACT_M)),
            duration_s=float(RETRACT_DURATION_S),
            steps=int(RETRACT_STEPS),
            label="retract after press",
        )


def _press_down(label: str) -> None:
    if PRESS_VERTICAL_ONLY:
        _move_left_vertical_press(
            -abs(float(PRESS_DOWN_M)),
            duration_s=float(PRESS_DURATION_S),
            steps=int(PRESS_STEPS),
            label=label,
        )
    else:
        _move_left_world_z(
            -abs(float(PRESS_DOWN_M)),
            duration_s=float(PRESS_DURATION_S),
            steps=int(PRESS_STEPS),
            label=label,
        )
    if float(PRESS_HOLD_S) > 0.0:
        print(f"[gpu_press_then_unplug] hold press for {float(PRESS_HOLD_S):.2f}s")
        time.sleep(float(PRESS_HOLD_S))


def _run_center_press() -> None:
    _close_left_for_press()
    _press_down("downward insertion press")


def _run_edge_pair_press() -> None:
    env = _direct_env()
    base_obs = env.get_observations("left")
    base_pos = np.asarray(base_obs["ee_pos"], dtype=np.float64).reshape(3)
    x_offset = abs(float(EDGE_PRESS_X_M))
    first_x_offset = x_offset + max(0.0, float(FIRST_EDGE_PRESS_EXTRA_X_M))
    clear_z = max(0.0, float(EDGE_PRESS_CLEAR_Z_M))
    move_duration_s = float(EDGE_PRESS_MOVE_DURATION_S)
    move_steps = int(EDGE_PRESS_MOVE_STEPS)

    print(
        "[gpu_press_then_unplug] edge-pair press sequence: "
        f"base={[round(float(v), 4) for v in base_pos.tolist()]} "
        f"x_offset_m={x_offset:.4f} "
        f"first_x_offset_m={first_x_offset:.4f} "
        f"repeat_x_offset_m={abs(float(REPEAT_EDGE_PRESS_X_M)):.4f} "
        f"clear_z_m={clear_z:.4f}"
    )
    _release_left_for_reposition("release for -x edge reposition")

    repeat_x_offset = abs(float(REPEAT_EDGE_PRESS_X_M))
    neg_edge_clear = base_pos + np.array([-first_x_offset, 0.0, clear_z], dtype=np.float64)
    neg_edge_press = base_pos + np.array([-first_x_offset, 0.0, 0.0], dtype=np.float64)
    pos_edge_clear = base_pos + np.array([x_offset, 0.0, clear_z], dtype=np.float64)
    pos_edge_press = base_pos + np.array([x_offset, 0.0, 0.0], dtype=np.float64)
    repeat_edge_clear = base_pos + np.array([-repeat_x_offset, 0.0, clear_z], dtype=np.float64)
    repeat_edge_press = base_pos + np.array([-repeat_x_offset, 0.0, 0.0], dtype=np.float64)

    if clear_z > 0.0:
        _move_left_to_world_pos(
            neg_edge_clear,
            duration_s=move_duration_s,
            steps=move_steps,
            label="move to -x edge press clearance",
        )
    _move_left_to_world_pos(
        neg_edge_press,
        duration_s=move_duration_s,
        steps=move_steps,
        label="move to -x edge press pose",
    )
    _close_left_for_press()
    _press_down("-x edge downward insertion press")
    _release_left_for_reposition("release for +x edge reposition")

    if clear_z > 0.0:
        _move_left_to_world_pos(
            neg_edge_clear,
            duration_s=move_duration_s,
            steps=move_steps,
            label="lift from -x edge press",
        )
        _move_left_to_world_pos(
            pos_edge_clear,
            duration_s=move_duration_s,
            steps=move_steps,
            label="move to +x edge press clearance",
        )
    _move_left_to_world_pos(
        pos_edge_press,
        duration_s=move_duration_s,
        steps=move_steps,
        label="move to +x edge press pose",
    )
    _close_left_for_press()
    _press_down("+x edge downward insertion press")

    for press_index in range(int(EXTRA_FIRST_EDGE_PRESS_COUNT)):
        _release_left_for_reposition(
            f"release for repeat -x edge press {press_index + 1}"
        )
        if clear_z > 0.0:
            _move_left_to_world_pos(
                pos_edge_clear,
                duration_s=move_duration_s,
                steps=move_steps,
                label=f"lift from +x edge press {press_index + 1}",
            )
            _move_left_to_world_pos(
                repeat_edge_clear,
                duration_s=move_duration_s,
                steps=move_steps,
                label=f"move back to -x edge clearance {press_index + 1}",
            )
        _move_left_to_world_pos(
            repeat_edge_press,
            duration_s=move_duration_s,
            steps=move_steps,
            label=f"move back to -x edge press pose {press_index + 1}",
        )
        _close_left_for_press()
        _press_down(f"repeat -x edge downward insertion press {press_index + 1}")


def _run_gpu_reset() -> None:
    reset_path = Path.cwd() / "cap" / "saved_scripts" / "gpu" / "gpu_reset.py"
    if not reset_path.exists():
        raise FileNotFoundError(f"GPU reset script not found: {reset_path}")
    if DRY_RUN:
        print(
            "[gpu_press_then_unplug] dry-run: would execute unplug reset: "
            f"{reset_path}"
        )
        return
    os.environ.setdefault("GPU_UNPLUG_GO_HOME_ON_START", "0")
    print(f"[gpu_press_then_unplug] executing unplug reset: {reset_path}")
    source = reset_path.read_text(encoding="utf-8")
    old_file = globals().get("__file__")
    globals()["__file__"] = str(reset_path)
    try:
        exec(compile(source, str(reset_path), "exec"), globals())  # noqa: S102
    finally:
        if old_file is None:
            globals().pop("__file__", None)
        else:
            globals()["__file__"] = old_file


def main() -> None:
    global SCRIPT_COMPLETED
    print(
        "[gpu_press_then_unplug] Config: "
        f"strategy={PRESS_STRATEGY!r} "
        f"press_down_m={float(PRESS_DOWN_M):.4f} "
        f"press_duration_s={float(PRESS_DURATION_S):.2f} "
        f"press_steps={int(PRESS_STEPS)} "
        f"hold_s={float(PRESS_HOLD_S):.2f} "
        f"vertical_only={bool(PRESS_VERTICAL_ONLY)} "
        f"max_xy_drift_m={float(PRESS_MAX_XY_DRIFT_M):.4f} "
        f"strict_vertical={bool(PRESS_STRICT_VERTICAL)} "
        f"extra_first_edge_press_count={int(EXTRA_FIRST_EDGE_PRESS_COUNT)} "
        f"edge_x_m={float(EDGE_PRESS_X_M):.4f} "
        f"first_edge_extra_x_m={float(FIRST_EDGE_PRESS_EXTRA_X_M):.4f} "
        f"edge_clear_z_m={float(EDGE_PRESS_CLEAR_Z_M):.4f} "
        f"reposition_open_delta={float(LEFT_REPOSITION_OPEN_DELTA):.3f} "
        f"reposition_open_max={float(LEFT_REPOSITION_OPEN_MAX_POS):.3f} "
        f"reposition_open_min={float(LEFT_REPOSITION_OPEN_MIN_POS):.3f} "
        f"reposition_open_duration_s={float(LEFT_REPOSITION_OPEN_DURATION_S):.2f} "
        f"final_open_delta={float(LEFT_FINAL_OPEN_DELTA):.3f} "
        f"final_open_max={float(LEFT_FINAL_OPEN_MAX_POS):.3f} "
        f"final_open_min={float(LEFT_FINAL_OPEN_MIN_POS):.3f} "
        f"final_open_duration_s={float(LEFT_FINAL_OPEN_DURATION_S):.2f} "
        f"open_after={bool(OPEN_AFTER_PRESS)} "
        f"retract_m={float(RETRACT_M):.4f} "
        f"go_home={bool(GO_HOME_AFTER_PRESS)} "
        f"run_unplug={bool(RUN_UNPLUG_AFTER_PRESS)} "
        f"dry_run={bool(DRY_RUN)}"
    )
    if PRESS_STRATEGY in {"center", "single", "single_center"}:
        _run_center_press()
    elif PRESS_STRATEGY in {"edge_pair", "edges", "two_edge", "two_edges"}:
        _run_edge_pair_press()
    else:
        raise ValueError(
            "GPU_INSERT_PRESS_STRATEGY must be 'edge_pair' or 'center', "
            f"got {PRESS_STRATEGY!r}"
        )
    _release_and_retract()
    if GO_HOME_AFTER_PRESS:
        print("[gpu_press_then_unplug] go home before unplug reset")
        if not DRY_RUN:
            go_home()
    if RUN_UNPLUG_AFTER_PRESS:
        _run_gpu_reset()
    SCRIPT_COMPLETED = True


def get_task_info() -> dict:
    return {
        "success": bool(SCRIPT_COMPLETED),
        "reward": 1.0 if SCRIPT_COMPLETED else 0.0,
        "press_strategy": str(PRESS_STRATEGY),
        "press_vertical_only": bool(PRESS_VERTICAL_ONLY),
        "run_unplug_after_press": bool(RUN_UNPLUG_AFTER_PRESS),
    }


main()

