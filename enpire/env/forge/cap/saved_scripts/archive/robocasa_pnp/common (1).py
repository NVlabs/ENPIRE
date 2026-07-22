# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation


_TOOL_GLOBALS: dict[str, object] = {}


def _capture_tool_globals() -> None:
    if _TOOL_GLOBALS:
        return
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        while frame is not None:
            globals_dict = frame.f_globals
            if "get_oracle_targets" in globals_dict and "freespace_move" in globals_dict:
                _TOOL_GLOBALS.update(globals_dict)
                return
            frame = frame.f_back
    finally:
        del frame
    raise RuntimeError("oracle common.py could not resolve run_script tool globals")


def _tool(name: str):
    _capture_tool_globals()
    if name not in _TOOL_GLOBALS:
        raise RuntimeError(f"oracle common.py missing tool: {name}")
    return _TOOL_GLOBALS[name]


def _arm_name() -> str:
    state = _tool("get_robot_state")()
    return list(state.arms.keys())[0]


def _arm_state():
    arm = _arm_name()
    return arm, _tool("get_robot_state")().arms[arm]


def _vec(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = _vec(vec)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def current_pose():
    arm, state = _arm_state()
    return arm, _vec(state.ee_pos), _vec(state.ee_quat)


def move_with_orientation(
    arm: str,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
):
    return _tool("freespace_move")(
        right_target_pos=_vec(target_pos).tolist(),
        right_target_quat=_vec(target_quat).tolist(),
        side=arm,
    )


def move_checked(
    arm: str,
    label: str,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    max_err: float,
) -> bool:
    result = move_with_orientation(arm, target_pos, target_quat)
    err = float(getattr(result, "final_pos_error_m", 1e9) or 1e9)
    status = getattr(result, "status", "Unknown")
    print(
        f"{label}: status={status} err={err:.4f} "
        f"pos={[round(float(x), 3) for x in _vec(target_pos)]}"
    )
    return status == "Success" and err <= max_err


def nudge_checked(
    arm: str,
    label: str,
    *,
    delta_pos: np.ndarray | None = None,
    delta_rpy: np.ndarray | None = None,
) -> bool:
    kwargs = {}
    if delta_pos is not None:
        kwargs["delta_pos"] = _vec(delta_pos).tolist()
    if delta_rpy is not None:
        kwargs["delta_rpy"] = _vec(delta_rpy).tolist()
    result = _tool("nudge")(arm, **kwargs)
    success = bool(result["success"] if isinstance(result, dict) else getattr(result, "success", False))
    _, pos, quat = current_pose()
    print(
        f"{label}: success={success} "
        f"delta_pos={kwargs.get('delta_pos', [])} "
        f"delta_rpy={kwargs.get('delta_rpy', [])} "
        f"pos={[round(float(x), 3) for x in pos]}"
    )
    return success


def make_press_quat(surface_normal: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    # Panda gripper +Z is fingertip-forward.
    z_axis = normalize(-surface_normal)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = normalize(world_up - z_axis * np.dot(world_up, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = normalize(x_axis - z_axis * np.dot(x_axis, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    base_rot = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    if abs(roll_deg) > 1e-6:
        base_rot = Rotation.from_rotvec(z_axis * np.radians(roll_deg)) * base_rot
    return base_rot.as_quat()


def _lane_entry(stage_far: np.ndarray, ee_pos: np.ndarray, lift_height: float) -> np.ndarray:
    lane = _vec(stage_far).copy()
    lane[2] = lift_height
    planar_delta = np.abs((_vec(ee_pos) - lane)[:2])
    planar_axis = int(np.argmax(planar_delta))
    lane[planar_axis] = ee_pos[planar_axis]
    return lane


def _panel_axes(surface_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    panel_up = normalize(world_up - surface_normal * np.dot(world_up, surface_normal))
    if np.linalg.norm(panel_up) < 1e-6:
        panel_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    panel_side = normalize(np.cross(panel_up, surface_normal))
    return panel_up, panel_side


def run_press_ladder(
    arm: str,
    *,
    approach_pos: np.ndarray,
    approach_quat: np.ndarray,
    release_pos: np.ndarray,
    target_pos: np.ndarray,
    surface_normal: np.ndarray,
    press_distance: float,
    success_check: Callable[[], bool] | None = None,
) -> bool:
    panel_up, panel_side = _panel_axes(surface_normal)
    lateral_offsets = [
        np.zeros(3, dtype=np.float64),
        panel_side * 0.010,
        -panel_side * 0.010,
        panel_up * 0.006,
        -panel_up * 0.006,
    ]
    depths = [press_distance, press_distance + 0.012]
    check = success_check or (lambda: bool(_tool("get_task_info")().get("success", False)))

    for depth_idx, depth in enumerate(depths, start=1):
        for offset_idx, offset_vec in enumerate(lateral_offsets, start=1):
            if not move_checked(
                arm,
                f"press_reset_{depth_idx}_{offset_idx}",
                approach_pos,
                approach_quat,
                max_err=0.05,
            ):
                continue

            press_pos = _vec(target_pos) - _vec(surface_normal) * depth + offset_vec
            result = move_with_orientation(arm, press_pos, approach_quat)
            err = float(getattr(result, "final_pos_error_m", 1e9) or 1e9)
            status = getattr(result, "status", "Unknown")
            print(
                f"press_attempt_{depth_idx}_{offset_idx}: status={status} "
                f"err={err:.4f} pos={[round(float(x), 3) for x in press_pos]}"
            )
            if status != "Success":
                continue
            time.sleep(0.25)
            if check():
                return True

            if not move_checked(
                arm,
                f"press_release_{depth_idx}_{offset_idx}",
                release_pos,
                approach_quat,
                max_err=0.16,
            ):
                continue
            time.sleep(0.25)
            if check():
                return True

    return False


def run_press_task(expected_task: str) -> None:
    _capture_tool_globals()
    oracle = _tool("get_oracle_targets")()
    if oracle.get("task_name") != expected_task:
        raise RuntimeError(
            f"wrong task: expected {expected_task}, got {oracle.get('task_name')}"
        )
    if not oracle.get("supported"):
        raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

    target = oracle["target"]
    target_pos = _vec(target["pos"])
    surface_normal = normalize(_vec(target["surface_normal"]))
    standoff = float(target.get("recommended_standoff", 0.06))
    press_distance = float(target.get("recommended_press_distance", 0.025))
    retreat_distance = max(float(target.get("recommended_retreat_distance", 0.10)), 0.16)

    print(
        f"Task:{expected_task}"
        f"Fixture:{oracle.get('fixture', {}).get('type', 'Unknown')}"
        f"Target pos:{[round(float(x), 3) for x in target_pos]}"
        f"Approach dir:{[round(float(x), 3) for x in surface_normal]}"
    )

    arm, ee_pos, travel_quat = current_pose()
    _tool("close_gripper")(arm)

    press_quat = make_press_quat(surface_normal)
    lift_height = max(float(ee_pos[2]), float(target_pos[2])) + 0.10
    stage_far = target_pos + surface_normal * (standoff + 0.18)
    stage_far[2] = lift_height
    stage_hover = target_pos + surface_normal * (standoff + 0.06)
    stage_hover[2] = lift_height
    approach_pos = target_pos + surface_normal * standoff
    retreat_pos = target_pos + surface_normal * retreat_distance
    retreat_pos[2] = max(float(target_pos[2]) + 0.05, lift_height)
    lane_entry = _lane_entry(stage_far, ee_pos, lift_height)

    if not move_checked(arm, "lane_entry", lane_entry, travel_quat, max_err=0.08):
        raise RuntimeError("failed to reach lane_entry")
    if not move_checked(arm, "stage_far", stage_far, travel_quat, max_err=0.08):
        raise RuntimeError("failed to reach stage_far")
    if not move_checked(arm, "stage_hover", stage_hover, press_quat, max_err=0.10):
        raise RuntimeError("failed to reach stage_hover")
    if not move_checked(arm, "approach", approach_pos, press_quat, max_err=0.08):
        raise RuntimeError("failed to reach approach")

    pressed = run_press_ladder(
        arm,
        approach_pos=approach_pos,
        approach_quat=press_quat,
        release_pos=retreat_pos,
        target_pos=target_pos,
        surface_normal=surface_normal,
        press_distance=press_distance,
    )
    if not pressed:
        move_checked(arm, "retreat", retreat_pos, press_quat, max_err=0.16)

    final_task = _tool("get_task_info")()
    print(f"Success:{final_task.get('success', False)}")
    print(f"Reward:{final_task.get('reward', 0.0)}")


def run_turn_task(expected_task: str) -> None:
    _capture_tool_globals()
    oracle = _tool("get_oracle_targets")()
    if oracle.get("task_name") != expected_task:
        raise RuntimeError(
            f"wrong task: expected {expected_task}, got {oracle.get('task_name')}"
        )
    if not oracle.get("supported"):
        raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

    target = oracle["target"]
    target_pos = _vec(target["pos"])
    surface_normal = normalize(_vec(target["surface_normal"]))
    axis_world = normalize(_vec(target["axis_world"]))
    anchor_world = _vec(target["anchor_world"])
    turn_amount = float(target.get("recommended_turn_amount", 0.30))
    desired_fraction = float(target.get("desired_fraction", 1.0))
    current_fraction = float(target.get("normalized_qpos", 0.0))
    preferred_sign = 1.0 if desired_fraction >= current_fraction else -1.0

    fixture_state = oracle.get("fixture_state")
    print(
        f"Task:{expected_task}"
        f"Target pos:{[round(float(x), 3) for x in target_pos]}"
        f"Joint type:{target.get('joint_type')}"
        f"Axis world:{[round(float(x), 3) for x in axis_world]}"
        f"Surface normal:{[round(float(x), 3) for x in surface_normal]}"
        f"Initial state:{fixture_state}"
        f"Attempt sign:{preferred_sign}"
    )

    arm, ee_pos, travel_quat = current_pose()
    _tool("close_gripper")(arm)

    lift_height = max(float(ee_pos[2]), float(target_pos[2])) + 0.10
    stage_height = max(float(target_pos[2]) + 0.14, lift_height - 0.08)
    approach_height = max(float(target_pos[2]) + 0.07, lift_height - 0.18)

    stage_xy = target_pos[:2] + 0.38 * (ee_pos[:2] - target_pos[:2])
    approach_xy = target_pos[:2] + 0.14 * (ee_pos[:2] - target_pos[:2])
    contact_xy = target_pos[:2] + 0.03 * (ee_pos[:2] - target_pos[:2])

    lane_entry = np.array([ee_pos[0], ee_pos[1], lift_height], dtype=np.float64)
    stage_pos = np.array([stage_xy[0], stage_xy[1], stage_height], dtype=np.float64)
    approach_pos = np.array([approach_xy[0], approach_xy[1], approach_height], dtype=np.float64)
    mid_approach_pos = np.array(
        [
            0.5 * (stage_pos[0] + approach_pos[0]),
            0.5 * (stage_pos[1] + approach_pos[1]),
            0.5 * (stage_pos[2] + approach_pos[2]),
        ],
        dtype=np.float64,
    )
    contact_pos = np.array([contact_xy[0], contact_xy[1], target_pos[2] + 0.06], dtype=np.float64)
    mid_contact_pos = np.array(
        [
            0.5 * (approach_pos[0] + contact_pos[0]),
            0.5 * (approach_pos[1] + contact_pos[1]),
            max(float(target_pos[2]) + 0.12, 0.5 * (approach_pos[2] + contact_pos[2])),
        ],
        dtype=np.float64,
    )

    if not move_checked(arm, "lane_entry", lane_entry, travel_quat, max_err=0.06):
        raise RuntimeError("failed to reach lane_entry")
    _, _, stage_quat = current_pose()
    if not move_checked(arm, "stage", stage_pos, stage_quat, max_err=0.08):
        raise RuntimeError("failed to reach stage")
    _, _, approach_quat = current_pose()
    if not move_checked(arm, "mid_approach", mid_approach_pos, approach_quat, max_err=0.09):
        raise RuntimeError("failed to reach mid_approach")
    _, _, approach_quat = current_pose()
    if not move_checked(arm, "approach", approach_pos, approach_quat, max_err=0.07):
        raise RuntimeError("failed to reach approach")
    _, _, contact_quat = current_pose()
    if not move_checked(arm, "mid_contact", mid_contact_pos, contact_quat, max_err=0.10):
        raise RuntimeError("failed to reach mid_contact")
    _, _, contact_quat = current_pose()
    if not move_checked(arm, "contact", contact_pos, contact_quat, max_err=0.06):
        raise RuntimeError("failed to reach contact")

    radial = normalize(contact_pos - anchor_world)
    if np.linalg.norm(radial) < 1e-6:
        radial = normalize(surface_normal)
    tangent = normalize(np.cross(axis_world, radial))
    if np.linalg.norm(tangent) < 1e-6:
        tangent = normalize(np.cross(axis_world, surface_normal))

    for attempt_sign in [preferred_sign, -preferred_sign]:
        for step_idx, scale in enumerate([0.12, 0.24], start=1):
            step_target = contact_pos + tangent * (attempt_sign * turn_amount * scale)
            if not move_checked(
                arm,
                f"turn_step_{step_idx}",
                step_target,
                contact_quat,
                max_err=0.05,
            ):
                break
            time.sleep(0.15)
            fixture_state = _tool("get_oracle_targets")().get("fixture_state")
            print(f"fixture_state:{fixture_state}")
            task = _tool("get_task_info")()
            if task.get("success", False):
                print(f"Success:{task.get('success', False)}")
                print(f"Reward:{task.get('reward', 0.0)}")
                return

    final_task = _tool("get_task_info")()
    print(f"Final state:{_tool('get_oracle_targets')().get('fixture_state')}")
    print(f"Success:{final_task.get('success', False)}")
    print(f"Reward:{final_task.get('reward', 0.0)}")


def run_handle_task(expected_task: str) -> None:
    _capture_tool_globals()
    oracle = _tool("get_oracle_targets")()
    if oracle.get("task_name") != expected_task:
        raise RuntimeError(
            f"wrong task: expected {expected_task}, got {oracle.get('task_name')}"
        )
    if not oracle.get("supported"):
        raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

    target = oracle["target"]
    target_pos = _vec(target["pos"])
    surface_normal = normalize(_vec(target["surface_normal"]))
    axis_world = normalize(_vec(target["axis_world"]))
    anchor_world = _vec(target["anchor_world"])
    desired_fraction = float(target.get("desired_fraction", 1.0))
    current_fraction = float(target.get("normalized_qpos", 0.0))
    travel_distance = float(target.get("recommended_travel_distance", 0.18))
    contact_offset = float(target.get("recommended_contact_offset", 0.015))
    retreat_distance = float(target.get("recommended_retreat_distance", 0.10))
    kind = target.get("kind", "")

    print(
        f"Task:{expected_task}"
        f"Kind:{kind}"
        f"Target pos:{[round(float(x), 3) for x in target_pos]}"
        f"Current fraction:{current_fraction:.3f}"
        f"Desired fraction:{desired_fraction:.3f}"
    )

    arm, ee_pos, travel_quat = current_pose()
    _tool("close_gripper")(arm)

    lift_height = max(float(ee_pos[2]), float(target_pos[2])) + 0.12
    stage_far = target_pos + surface_normal * 0.18
    stage_far[2] = lift_height
    lane_entry = _lane_entry(stage_far, ee_pos, lift_height)
    stage_pos = target_pos + surface_normal * 0.12
    stage_pos[2] = max(float(target_pos[2]) + 0.14, lift_height - 0.06)
    approach_pos = target_pos + surface_normal * (contact_offset + 0.05)
    approach_pos[2] = max(float(target_pos[2]) + 0.10, lift_height - 0.14)
    contact_pos = target_pos + surface_normal * contact_offset
    contact_pos[2] = max(float(target_pos[2]) + 0.04, float(approach_pos[2]) - 0.12)
    retreat_pos = target_pos + surface_normal * retreat_distance
    retreat_pos[2] = max(float(target_pos[2]) + 0.08, lift_height)

    if lift_height - float(ee_pos[2]) > 0.16:
        lift_1 = np.array([ee_pos[0], ee_pos[1], min(lift_height, float(ee_pos[2]) + 0.12)], dtype=np.float64)
        if not move_checked(arm, "lift_1", lift_1, travel_quat, max_err=0.08):
            raise RuntimeError("failed to reach lift_1")
        if lift_height - float(lift_1[2]) > 0.08:
            lift_2 = np.array([ee_pos[0], ee_pos[1], min(lift_height, float(ee_pos[2]) + 0.24)], dtype=np.float64)
            if not move_checked(arm, "lift_2", lift_2, travel_quat, max_err=0.08):
                raise RuntimeError("failed to reach lift_2")

    if not move_checked(arm, "lane_entry", lane_entry, travel_quat, max_err=0.08):
        raise RuntimeError("failed to reach lane_entry")
    _, _, stage_quat = current_pose()
    if not move_checked(arm, "stage", stage_pos, stage_quat, max_err=0.26):
        raise RuntimeError("failed to reach stage")
    _, _, approach_quat = current_pose()
    if not move_checked(arm, "approach", approach_pos, approach_quat, max_err=0.20):
        raise RuntimeError("failed to reach approach")
    _, _, contact_quat = current_pose()
    if not move_checked(arm, "contact", contact_pos, contact_quat, max_err=0.20):
        raise RuntimeError("failed to reach contact")

    sign = 1.0 if desired_fraction >= current_fraction else -1.0
    radial = normalize(target_pos - anchor_world)
    if np.linalg.norm(radial) < 1e-6:
        radial = normalize(surface_normal)
    tangent = normalize(np.cross(axis_world, radial))
    if np.linalg.norm(tangent) < 1e-6:
        tangent = normalize(np.cross(axis_world, surface_normal))

    if kind == "slider_handle":
        motion_dirs = [normalize(axis_world) * sign, normalize(axis_world) * -sign]
        step_scales = [1.0] * 6
    else:
        motion_dirs = [tangent * sign, -tangent * sign]
        step_scales = [0.5, 1.0]

    for dir_idx, motion_dir in enumerate(motion_dirs, start=1):
        for step_idx, scale in enumerate(step_scales, start=1):
            if kind == "slider_handle":
                delta_mag = max(0.04, min(0.07, travel_distance / 3.0)) * scale
                inward_bias = -surface_normal * max(0.004, min(0.012, contact_offset * 0.8))
                if not nudge_checked(
                    arm,
                    f"handle_step_{dir_idx}_{step_idx}",
                    delta_pos=motion_dir * delta_mag + inward_bias,
                ):
                    break
            else:
                step_target = contact_pos + motion_dir * (travel_distance * scale)
                if not move_checked(
                    arm,
                    f"handle_step_{dir_idx}_{step_idx}",
                    step_target,
                    contact_quat,
                    max_err=0.18,
                ):
                    break
            time.sleep(0.15)
            if _tool("get_task_info")().get("success", False):
                print(f"Success:{_tool('get_task_info')().get('success', False)}")
                print(f"Reward:{_tool('get_task_info')().get('reward', 0.0)}")
                return

    _, _, retreat_quat = current_pose()
    move_checked(arm, "retreat", retreat_pos, retreat_quat, max_err=0.16)
    final_task = _tool("get_task_info")()
    print(f"Success:{final_task.get('success', False)}")
    print(f"Reward:{final_task.get('reward', 0.0)}")


def run_pick_place_task(expected_task: str) -> None:
    _capture_tool_globals()
    oracle = _tool("get_oracle_targets")()
    if oracle.get("task_name") != expected_task:
        raise RuntimeError(
            f"wrong task: expected {expected_task}, got {oracle.get('task_name')}"
        )
    if not oracle.get("supported"):
        raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

    task_info = oracle.get("task_info", {})
    target = oracle["target"]
    target_pos = _vec(target["pos"])

    if "lid_fixture" in oracle:
        source_pos = _vec(oracle["lid_fixture"]["pos"])
    elif "obj_pos" in task_info:
        source_pos = _vec(task_info["obj_pos"])
    else:
        raise RuntimeError("pick-place source object pose unavailable")

    arm, _, travel_quat = current_pose()
    _tool("open_gripper")(arm)

    pick_hover = source_pos + np.array([0.0, 0.0, 0.12], dtype=np.float64)
    pick_pos = source_pos + np.array([0.0, 0.0, 0.03], dtype=np.float64)
    place_hover = target_pos + np.array([0.0, 0.0, 0.14], dtype=np.float64)
    place_pos = target_pos + np.array([0.0, 0.0, 0.04], dtype=np.float64)

    if not move_checked(arm, "pick_hover", pick_hover, travel_quat, max_err=0.08):
        raise RuntimeError("failed to reach pick_hover")
    if not move_checked(arm, "pick_pos", pick_pos, travel_quat, max_err=0.06):
        raise RuntimeError("failed to reach pick_pos")
    _tool("close_gripper")(arm)
    time.sleep(0.25)
    move_checked(arm, "lift", pick_hover, travel_quat, max_err=0.10)
    if not move_checked(arm, "place_hover", place_hover, travel_quat, max_err=0.10):
        raise RuntimeError("failed to reach place_hover")
    if not move_checked(arm, "place_pos", place_pos, travel_quat, max_err=0.08):
        raise RuntimeError("failed to reach place_pos")
    _tool("open_gripper")(arm)
    time.sleep(0.25)
    move_checked(arm, "retreat", place_hover, travel_quat, max_err=0.10)

    final_task = _tool("get_task_info")()
    print(f"Success:{final_task.get('success', False)}")
    print(f"Reward:{final_task.get('reward', 0.0)}")


def run_navigate_task(expected_task: str) -> None:
    _capture_tool_globals()
    oracle = _tool("get_oracle_targets")()
    if oracle.get("task_name") != expected_task:
        raise RuntimeError(
            f"wrong task: expected {expected_task}, got {oracle.get('task_name')}"
        )
    if not oracle.get("supported"):
        raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

    target = oracle["target"]
    pos = _vec(target["pos"])
    yaw = float(target["yaw"])
    print(
        f"Task:{expected_task} Base target:{[round(float(x), 3) for x in pos]} "
        f"yaw={yaw:.3f}"
    )
    result = _tool("move_base")(float(pos[0]), float(pos[1]), yaw)
    print(f"move_base:{result}")
