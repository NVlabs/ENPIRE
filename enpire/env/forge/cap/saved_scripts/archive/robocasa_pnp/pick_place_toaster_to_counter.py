# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

SIDE = "right"
CUROBO_SPEED = "slow"
JOINT_SPEED = 1.0
HOVER_HEIGHT = 0.12
LIFT_HEIGHT = 0.20
PLACE_CLEARANCE = 0.03
TOASTER_RELEASE_HEIGHT = 0.05
TOASTER_EXTRACT_DISTANCES = [0.10, 0.14, 0.18]
TOASTER_EXTRACT_RISE = 0.02
TOASTER_TRANSIT_CLEARANCE = 0.08
TOASTER_LIFT_STEPS = [0.05, 0.05, 0.05, 0.05]
NON_SINK_GRASP_Z_OFFSET = 0.01
TOASTER_GRASP_Z_OFFSET = 0.06

HOVER_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.04],
    [0.0, 0.0, 0.08],
    [0.0, 0.0, 0.12],
    [0.03, 0.0, 0.04],
    [-0.03, 0.0, 0.04],
    [0.0, 0.03, 0.04],
    [0.0, -0.03, 0.04],
    [0.05, 0.0, 0.08],
    [-0.05, 0.0, 0.08],
    [0.0, 0.05, 0.08],
    [0.0, -0.05, 0.08],
    [0.03, 0.03, 0.08],
    [0.03, -0.03, 0.08],
    [-0.03, 0.03, 0.08],
    [-0.03, -0.03, 0.08],
]
GRASP_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.02],
    [0.02, 0.0, 0.02],
    [-0.02, 0.0, 0.02],
    [0.0, 0.02, 0.02],
    [0.0, -0.02, 0.02],
]
TOASTER_GRASP_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.02],
    [0.0, 0.0, 0.04],
    [0.01, 0.0, 0.0],
    [-0.01, 0.0, 0.0],
    [0.0, 0.01, 0.0],
    [0.0, -0.01, 0.0],
    [0.01, 0.0, 0.02],
    [-0.01, 0.0, 0.02],
    [0.0, 0.01, 0.02],
    [0.0, -0.01, 0.02],
]
LIFT_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.05],
    [0.0, 0.0, 0.10],
    [0.03, 0.0, 0.02],
    [-0.03, 0.0, 0.02],
    [0.0, 0.03, 0.02],
    [0.0, -0.03, 0.02],
]
TOASTER_LIFT_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.03],
    [0.0, 0.0, 0.06],
    [0.02, 0.0, 0.02],
    [-0.02, 0.0, 0.02],
    [0.0, 0.02, 0.02],
    [0.0, -0.02, 0.02],
    [0.02, 0.0, 0.06],
    [-0.02, 0.0, 0.06],
]
PLACE_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.05],
    [0.02, 0.0, 0.05],
    [-0.02, 0.0, 0.05],
    [0.0, 0.02, 0.05],
    [0.0, -0.02, 0.05],
]
ORIENTATION_OFFSETS_DEG = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, -25.0],
    [0.0, 0.0, 25.0],
    [0.0, 0.0, -45.0],
    [0.0, 0.0, 45.0],
    [0.0, 0.0, -70.0],
    [0.0, 0.0, 70.0],
    [-20.0, 0.0, 0.0],
    [20.0, 0.0, 0.0],
    [-35.0, 0.0, 0.0],
    [35.0, 0.0, 0.0],
    [0.0, -20.0, 0.0],
    [0.0, 20.0, 0.0],
    [0.0, -35.0, 0.0],
    [0.0, 35.0, 0.0],
    [0.0, -55.0, 0.0],
    [0.0, 55.0, 0.0],
    [-20.0, 0.0, -45.0],
    [-20.0, 0.0, 45.0],
    [20.0, 0.0, -45.0],
    [20.0, 0.0, 45.0],
    [0.0, -20.0, -45.0],
    [0.0, -20.0, 45.0],
    [0.0, 20.0, -45.0],
    [0.0, 20.0, 45.0],
    [0.0, -35.0, -70.0],
    [0.0, -35.0, 70.0],
    [0.0, 35.0, -70.0],
    [0.0, 35.0, 70.0],
]
HOVER_GRIPPER_DIRECTIONS = [
    ("+z", [0.0, 0.0, 1.0]),
    ("-z", [0.0, 0.0, -1.0]),
    ("+x", [1.0, 0.0, 0.0]),
    ("-x", [-1.0, 0.0, 0.0]),
    ("+y", [0.0, 1.0, 0.0]),
    ("-y", [0.0, -1.0, 0.0]),
]

_TOOL_GLOBALS: dict[str, object] = {}


def _capture_tool_globals() -> None:
    if _TOOL_GLOBALS:
        return
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        while frame is not None:
            globals_dict = frame.f_globals
            if "get_task_info" in globals_dict and "create_motion_planner" in globals_dict:
                _TOOL_GLOBALS.update(globals_dict)
                return
            frame = frame.f_back
    finally:
        del frame
    raise RuntimeError("pick_place_curobo_common.py could not resolve run_script tool globals")


def _tool(name: str):
    _capture_tool_globals()
    if name not in _TOOL_GLOBALS:
        raise RuntimeError(f"pick_place_curobo_common.py missing tool: {name}")
    return _TOOL_GLOBALS[name]


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def find_place_target(info: dict, obj_pos: np.ndarray) -> tuple[np.ndarray, str]:
    env_name = str(info.get("env_name", ""))
    if "CounterToCabinet" in env_name and "distr_cab_pos" in info:
        return np.asarray(info["distr_cab_pos"], dtype=float), "cabinet"

    preferred_keys = [
        ("container_pos", "container"),
        ("plate_pos", "plate"),
        ("distr_counter_pos", "counter"),
    ]
    for key, label in preferred_keys:
        if key in info:
            return np.asarray(info[key], dtype=float), label

    if "distr_pos" in info:
        return np.asarray(info["distr_pos"], dtype=float), "distr"

    for key, value in info.items():
        if not key.endswith("_pos") or key.startswith("robot") or key == "obj_pos":
            continue
        return np.asarray(value, dtype=float), key.removesuffix("_pos")

    return obj_pos + np.array([0.15, 0.0, 0.05], dtype=float), "offset fallback"


def build_timestamps(positions: np.ndarray) -> list[float]:
    timestamps = [0.0]
    for i in range(positions.shape[0] - 1):
        dt = max(
            0.01,
            float(np.max(np.abs(positions[i + 1] - positions[i]))) / JOINT_SPEED,
        )
        timestamps.append(timestamps[-1] + dt)
    return timestamps


def quat_from_gripper_direction(direction_world: np.ndarray) -> np.ndarray:
    z_axis = normalize(direction_world)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = normalize(np.cross(helper, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()


def build_hover_base_quats(current_quat_world: np.ndarray) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = [
        ("current", np.asarray(current_quat_world, dtype=float))
    ]
    for label, direction in HOVER_GRIPPER_DIRECTIONS:
        candidates.append(
            (f"dir{label}", quat_from_gripper_direction(np.asarray(direction, dtype=float)))
        )
    return candidates


def build_toaster_grasp_base_quats(
    current_quat_world: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = [
        ("current", np.asarray(current_quat_world, dtype=float))
    ]
    for label, direction in (
        ("dir-z", [0.0, 0.0, -1.0]),
        ("dir+x", [1.0, 0.0, 0.0]),
        ("dir-x", [-1.0, 0.0, 0.0]),
        ("dir+y", [0.0, 1.0, 0.0]),
        ("dir-y", [0.0, -1.0, 0.0]),
    ):
        candidates.append(
            (label, quat_from_gripper_direction(np.asarray(direction, dtype=float)))
        )
    return candidates


def execute_trajectory(
    positions: np.ndarray,
    *,
    label: str,
    gripper: float | None = None,
) -> bool:
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if positions.ndim != 2 or positions.shape[0] == 0:
        print(f"[{label}] Empty trajectory from cuRobo")
        return False

    timestamps = build_timestamps(positions)
    gripper_positions = None
    if gripper is not None:
        gripper_positions = [[float(gripper)]] * len(timestamps)
    try:
        _tool("move_joint_keypoints")(
            side=SIDE,
            timestamps=timestamps,
            joint_positions=positions.tolist(),
            gripper_positions=gripper_positions,
        )
    except RuntimeError as exc:
        print(f"[{label}] execution failed: {exc}")
        return False

    print(f"[{label}] done ({timestamps[-1]:.2f}s)")
    return True


def run_pick_place_curobo_task(expected_task: str) -> None:
    _capture_tool_globals()
    planner = _tool("create_motion_planner")(solver_speed=CUROBO_SPEED)

    base_pos = np.zeros(3, dtype=np.float64)
    base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    r_base = R.from_quat(base_quat)

    def refresh_planner_world(*, reason: str, exclude_body_prefixes: list[str] | None = None) -> None:
        nonlocal base_pos, base_quat, r_base
        print(f"[curobo] Refreshing collision world ({reason})")
        world_info = _tool("update_planner_world")(
            planner,
            exclude_body_prefixes=exclude_body_prefixes,
        )
        base_pos = np.asarray(world_info["base_pos"], dtype=np.float64)
        base_quat = np.asarray(world_info["base_quat_xyzw"], dtype=np.float64)
        r_base = R.from_quat(base_quat)
        print(f"[curobo] Loaded {world_info['n_obstacles']} obstacles")

    def world_to_base(
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_b = r_base.inv().apply(np.asarray(pos, dtype=np.float64) - base_pos)
        quat_b = (r_base.inv() * R.from_quat(np.asarray(quat_xyzw, dtype=np.float64))).as_quat()
        return pos_b, quat_b

    def plan_and_move(
        target_pos_world: np.ndarray,
        target_quat_world: np.ndarray,
        *,
        label: str,
        gripper: float | None = None,
    ) -> bool:
        cur_state = _tool("get_robot_state")()
        cur_jp = np.asarray(cur_state.arms[SIDE].joint_pos, dtype=np.float64)
        target_pos_b, target_quat_b = world_to_base(target_pos_world, target_quat_world)
        if gripper is not None and hasattr(planner, "set_gripper_qpos"):
            planner.set_gripper_qpos(right_gripper=float(gripper))

        result = planner.plan_to_pose(
            current_left_jp=np.zeros(0),
            current_right_jp=cur_jp,
            target_right_pos=target_pos_b,
            target_right_quat_xyzw=target_quat_b,
            side=SIDE,
        )
        status = result.get("status", "Error")
        if status != "Success":
            print(f"[{label}] cuRobo failed: {status} - {result.get('status_detail', '')}")
            return False
        return execute_trajectory(
            np.asarray(result["right_positions"], dtype=np.float64),
            label=label,
            gripper=gripper,
        )

    def plan_with_pose_candidates(
        target_pos_world: np.ndarray,
        target_quat_world: np.ndarray,
        *,
        label: str,
        pos_offsets: list[list[float]],
        orientation_offsets_deg: list[list[float]],
        gripper: float | None = None,
        base_quat_candidates: list[tuple[str, np.ndarray]] | None = None,
    ) -> bool:
        if base_quat_candidates is None:
            base_quat_candidates = [("current", np.asarray(target_quat_world, dtype=float))]

        seen: set[tuple[str, tuple[float, ...], tuple[float, ...]]] = set()
        for base_name, base_quat_world in base_quat_candidates:
            base_rot = R.from_quat(np.asarray(base_quat_world, dtype=float))
            for offset in pos_offsets:
                offset_arr = np.asarray(offset, dtype=float)
                for rpy_offset_deg in orientation_offsets_deg:
                    rpy_offset_arr = np.asarray(rpy_offset_deg, dtype=float)
                    key = (
                        str(base_name),
                        tuple(np.round(offset_arr, 4).tolist()),
                        tuple(np.round(rpy_offset_arr, 4).tolist()),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    candidate_pos = np.asarray(target_pos_world, dtype=float) + offset_arr
                    candidate_quat = (
                        base_rot * R.from_euler("xyz", rpy_offset_arr, degrees=True)
                    ).as_quat()
                    candidate_label = label
                    if base_name != "current":
                        candidate_label += f"+{base_name}"
                    if np.linalg.norm(offset_arr) > 1e-6:
                        candidate_label += f"+d{np.round(offset_arr, 3).tolist()}"
                    if np.linalg.norm(rpy_offset_arr) > 1e-6:
                        candidate_label += f"+rpy{np.round(rpy_offset_arr, 1).tolist()}"

                    if plan_and_move(
                        candidate_pos,
                        candidate_quat,
                        label=candidate_label,
                        gripper=gripper,
                    ):
                        return True
        return False

    refresh_planner_world(reason="run start")

    info = _tool("get_task_info")()
    env_name = str(info.get("env_name", ""))
    if env_name != expected_task:
        raise RuntimeError(f"wrong task: expected {expected_task}, got {env_name}")

    obj_pos = np.asarray(info["obj_pos"], dtype=float)
    place_pos, place_label = find_place_target(info, obj_pos)
    obj_name = info.get("obj_name", "object")
    print(f"Task: {env_name}")
    print(f"Object ({obj_name}): {[round(float(x), 3) for x in obj_pos]}")
    print(f"Place target ({place_label}): {[round(float(x), 3) for x in place_pos]}")

    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)

    print("\n--- Step 1: Open gripper ---")
    _tool("open_gripper")(SIDE)
    if hasattr(planner, "set_gripper_qpos"):
        planner.set_gripper_qpos(right_gripper=1.0)

    print("\n--- Step 2: Hover above object ---")
    hover = obj_pos.copy()
    hover[2] += HOVER_HEIGHT
    if not plan_with_pose_candidates(
        hover,
        ee_quat,
        label="hover",
        pos_offsets=HOVER_POS_OFFSETS,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=1.0,
        base_quat_candidates=build_hover_base_quats(ee_quat),
    ):
        raise RuntimeError("Failed to reach hover pose")

    print("\n--- Step 3: Descend to object ---")
    obj_now = np.asarray(_tool("get_task_info")()["obj_pos"], dtype=float)
    grasp_pos_offsets = GRASP_POS_OFFSETS
    grasp_base_quats = None
    if "ToasterToCounter" in env_name:
        obj_now[2] += TOASTER_GRASP_Z_OFFSET
        grasp_pos_offsets = TOASTER_GRASP_POS_OFFSETS
        grasp_base_quats = build_toaster_grasp_base_quats(
            np.asarray(_tool("get_robot_state")().arms[SIDE].ee_quat, dtype=float)
        )
    elif "SinkToCounter" not in env_name:
        obj_now[2] += NON_SINK_GRASP_Z_OFFSET

    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
    if not plan_with_pose_candidates(
        obj_now,
        ee_quat,
        label="grasp-approach",
        pos_offsets=grasp_pos_offsets,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=1.0,
        base_quat_candidates=grasp_base_quats,
    ):
        raise RuntimeError("Failed to reach object pose")

    print("\n--- Step 4: Close gripper ---")
    _tool("close_gripper")(SIDE)
    time.sleep(0.25)
    state = _tool("get_robot_state")()
    grip_val = float(np.asarray(state.arms[SIDE].gripper_pos, dtype=float).reshape(-1)[0])
    grasped = 0.02 < grip_val < 0.95
    print(f"Gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED'}")
    if not grasped:
        print("Grasp was not confirmed. Going home.")
        _tool("open_gripper")(SIDE)
        if hasattr(planner, "set_gripper_qpos"):
            planner.set_gripper_qpos(right_gripper=1.0)
        _tool("go_home")(SIDE)
        final = _tool("get_task_info")()
        print(f"\nSuccess: {final.get('success', False)}")
        print(f"Reward:  {final.get('reward', 0.0)}")
        return
    if hasattr(planner, "set_gripper_qpos"):
        planner.set_gripper_qpos(right_gripper=0.0)
    _tool("set_gripper")(SIDE, 0.0)
    time.sleep(0.10)

    refresh_planner_world(
        reason="post grasp",
        exclude_body_prefixes=["robot0", "gripper", "mobilebase", "obj"],
    )

    if "ToasterToCounter" in env_name:
        print("\n--- Step 5: Extract from toaster ---")
        state = _tool("get_robot_state")()
        ee_pos = np.asarray(state.arms[SIDE].ee_pos, dtype=float)
        ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
        extract_dir = place_pos[:2] - ee_pos[:2]
        if float(np.linalg.norm(extract_dir)) < 1e-6:
            extract_dir = np.array([1.0, 0.0], dtype=float)
        else:
            extract_dir = normalize(extract_dir)
        extracted = False
        for distance in TOASTER_EXTRACT_DISTANCES:
            extract_pos = ee_pos.copy()
            extract_pos[:2] += extract_dir * distance
            extract_pos[2] += TOASTER_EXTRACT_RISE
            label = f"extract@{distance:.2f}"
            if plan_with_pose_candidates(
                extract_pos,
                ee_quat,
                label=label,
                pos_offsets=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]],
                orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
                gripper=0.0,
                base_quat_candidates=build_toaster_grasp_base_quats(ee_quat),
            ):
                extracted = True
                break
        if not extracted:
            raise RuntimeError("Failed to extract object from toaster")

    print("\n--- Step 5: Lift ---")
    if "ToasterToCounter" in env_name:
        lifted = True
        for step_idx, dz in enumerate(TOASTER_LIFT_STEPS, start=1):
            delta = np.array([0.0, 0.0, dz], dtype=float)
            print(
                f"[lift_nudge_{step_idx}] delta:",
                [round(float(x), 3) for x in delta],
            )
            result = _tool("nudge")(SIDE, delta_pos=delta.tolist())
            success = bool(
                result["success"] if isinstance(result, dict) else getattr(result, "success", False)
            )
            if not success:
                lifted = False
                break
            _tool("set_gripper")(SIDE, 0.0)
            time.sleep(0.05)
        if not lifted:
            raise RuntimeError("Failed to lift object")
        refresh_planner_world(
            reason="post lift",
            exclude_body_prefixes=["robot0", "gripper", "mobilebase", "obj"],
        )
    else:
        state = _tool("get_robot_state")()
        lift_pos = np.asarray(state.arms[SIDE].ee_pos, dtype=float)
        lift_pos[2] += LIFT_HEIGHT
        ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
        if not plan_with_pose_candidates(
            lift_pos,
            ee_quat,
            label="lift",
            pos_offsets=LIFT_POS_OFFSETS,
            orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
            gripper=0.0,
        ):
            raise RuntimeError("Failed to lift object")

    if "ToasterToCounter" in env_name:
        print("\n--- Step 6: Transit clear of toaster ---")
        state = _tool("get_robot_state")()
        ee_pos = np.asarray(state.arms[SIDE].ee_pos, dtype=float)
        transit_dir = place_pos[:2] - ee_pos[:2]
        transit_delta = np.array([0.0, 0.0, TOASTER_TRANSIT_CLEARANCE], dtype=float)
        if float(np.linalg.norm(transit_dir)) >= 1e-6:
            transit_delta[:2] = normalize(transit_dir) * 0.08
        print(
            "[transit-clear] nudge delta:",
            [round(float(x), 3) for x in transit_delta],
        )
        transit_result = _tool("nudge")(SIDE, delta_pos=transit_delta.tolist())
        transit_success = bool(
            transit_result["success"]
            if isinstance(transit_result, dict)
            else getattr(transit_result, "success", False)
        )
        if not transit_success:
            raise RuntimeError("Failed to clear toaster after lift")
        _tool("set_gripper")(SIDE, 0.0)
        time.sleep(0.10)
        refresh_planner_world(
            reason="post toaster transit clear",
            exclude_body_prefixes=["robot0", "gripper", "mobilebase", "obj"],
        )

    print(f"\n--- Step 7: Move above {place_label} ---")
    place_hover = place_pos.copy()
    place_hover[2] += HOVER_HEIGHT
    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
    if not plan_with_pose_candidates(
        place_hover,
        ee_quat,
        label="place-hover",
        pos_offsets=HOVER_POS_OFFSETS,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=0.0,
    ):
        raise RuntimeError("Failed to reach place hover pose")

    print(f"\n--- Step 8: Lower to {place_label} ---")
    place_target = place_pos.copy()
    place_target[2] += PLACE_CLEARANCE
    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
    place_succeeded = False
    if plan_with_pose_candidates(
        place_target,
        ee_quat,
        label="place-lower",
        pos_offsets=PLACE_POS_OFFSETS,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=0.0,
    ):
        place_succeeded = True
    else:
        shallow_target = place_pos.copy()
        shallow_target[2] += TOASTER_RELEASE_HEIGHT
        print(
            "[place-lower] falling back to shallow release height:",
            [round(float(x), 3) for x in shallow_target],
        )
        if plan_with_pose_candidates(
            shallow_target,
            ee_quat,
            label="place-shallow",
            pos_offsets=PLACE_POS_OFFSETS,
            orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
            gripper=0.0,
        ):
            place_succeeded = True
    if not place_succeeded:
        raise RuntimeError("Failed to reach place pose")

    print("\n--- Step 9: Release ---")
    _tool("open_gripper")(SIDE)
    time.sleep(0.25)
    if hasattr(planner, "set_gripper_qpos"):
        planner.set_gripper_qpos(right_gripper=1.0)

    print("\n--- Step 10: Retract ---")
    _tool("go_home")(SIDE)

    final = _tool("get_task_info")()
    print(f"\nSuccess: {final.get('success', False)}")
    print(f"Reward:  {final.get('reward', 0.0)}")


run_pick_place_curobo_task("PickPlaceToasterToCounter")
