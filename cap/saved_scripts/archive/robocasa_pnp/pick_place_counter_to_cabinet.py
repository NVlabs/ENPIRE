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
NON_SINK_GRASP_Z_OFFSET = 0.01
GRASP_ALIGN_Z_THRESH = 0.04
GRASP_ALIGN_XY_THRESH = 0.03
GRASP_ALIGN_MAX_ITERS = 3
GRASP_ALIGN_MAX_NUDGE = 0.03
GRASP_PROBE_LIFT = 0.04
GRASP_CONFIRM_OBJ_RISE = 0.02
PRE_CLOSE_INSERT_DISTANCE = 0.015
TOASTER_GRASP_Z_OFFSET = 0.06
CABINET_APPROACH_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.03],
    [0.0, 0.0, 0.06],
    [0.02, 0.0, 0.0],
    [-0.02, 0.0, 0.0],
    [0.0, 0.02, 0.0],
    [0.0, -0.02, 0.0],
    [0.04, 0.0, 0.02],
    [-0.04, 0.0, 0.02],
    [0.0, 0.04, 0.02],
    [0.0, -0.04, 0.02],
]
CABINET_INSERT_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.02],
    [0.0, 0.0, 0.04],
    [0.01, 0.0, 0.0],
    [-0.01, 0.0, 0.0],
    [0.0, 0.01, 0.0],
    [0.0, -0.01, 0.0],
    [0.02, 0.0, 0.02],
    [-0.02, 0.0, 0.02],
    [0.0, 0.02, 0.02],
    [0.0, -0.02, 0.02],
]
CABINET_FALLBACK_APPROACH_DISTANCE = 0.18
CABINET_FALLBACK_INSERT_DISTANCE = 0.08
CABINET_FALLBACK_SETTLE_RISE = 0.0
CABINET_SETTLE_POS_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, -0.01],
    [0.0, 0.0, 0.01],
    [0.01, 0.0, 0.0],
    [-0.01, 0.0, 0.0],
    [0.0, 0.01, 0.0],
    [0.0, -0.01, 0.0],
]

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
    if "CounterToCabinet" in env_name:
        for key in ("cab_settle_pos", "cab_insert_pos", "cab_front_center_pos"):
            if key in info:
                return np.asarray(info[key], dtype=float), "cabinet"
        if "distr_cab_pos" in info:
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


def build_cabinet_base_quats(
    current_quat_world: np.ndarray,
    front_normal_world: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    front_normal = normalize(front_normal_world)
    candidates: list[tuple[str, np.ndarray]] = [
        ("current", np.asarray(current_quat_world, dtype=float))
    ]
    for label, direction in (
        ("cab-in", -front_normal),
        ("cab-out", front_normal),
        ("down", np.array([0.0, 0.0, -1.0], dtype=float)),
    ):
        candidates.append(
            (label, quat_from_gripper_direction(np.asarray(direction, dtype=float)))
        )
    return candidates


def estimate_cabinet_front_normal(
    info: dict,
    *,
    place_pos: np.ndarray,
    current_ee_pos: np.ndarray,
) -> np.ndarray:
    if "cab_front_normal" in info:
        return normalize(np.asarray(info["cab_front_normal"], dtype=float))

    if "cab_front_center_pos" in info and "cab_back_center_pos" in info:
        front_center = np.asarray(info["cab_front_center_pos"], dtype=float)
        back_center = np.asarray(info["cab_back_center_pos"], dtype=float)
        return normalize(front_center - back_center)

    front = np.asarray(current_ee_pos, dtype=float) - np.asarray(place_pos, dtype=float)
    front[2] = 0.0
    if np.linalg.norm(front) < 1e-6:
        front = np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize(front)


def get_cabinet_waypoints(
    info: dict,
    *,
    place_pos: np.ndarray,
    current_ee_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    front_normal = estimate_cabinet_front_normal(
        info,
        place_pos=place_pos,
        current_ee_pos=current_ee_pos,
    )

    settle_pos = np.asarray(
        info.get("cab_settle_pos", place_pos),
        dtype=float,
    ).copy()
    if "cab_settle_pos" not in info:
        settle_pos[2] += CABINET_FALLBACK_SETTLE_RISE

    insert_pos = np.asarray(
        info.get("cab_insert_pos", settle_pos + front_normal * CABINET_FALLBACK_INSERT_DISTANCE),
        dtype=float,
    )
    approach_pos = np.asarray(
        info.get(
            "cab_approach_pos",
            settle_pos + front_normal * CABINET_FALLBACK_APPROACH_DISTANCE,
        ),
        dtype=float,
    )

    if "cab_front_center_pos" in info:
        front_center = np.asarray(info["cab_front_center_pos"], dtype=float)
        approach_pos[2] = front_center[2]
        insert_pos[2] = front_center[2]

    return {
        "approach_pos": approach_pos,
        "insert_pos": insert_pos,
        "settle_pos": settle_pos,
        "front_normal": front_normal,
    }


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


def get_obj_gap(info: dict) -> np.ndarray:
    return np.asarray(info["obj_to_robot0_eef_pos"], dtype=float)


def maybe_align_grasp_with_nudges() -> None:
    for idx in range(GRASP_ALIGN_MAX_ITERS):
        info = _tool("get_task_info")()
        gap_local = get_obj_gap(info)
        ee_quat = np.asarray(_tool("get_robot_state")().arms[SIDE].ee_quat, dtype=float)
        gap_world = R.from_quat(ee_quat).apply(gap_local)
        xy_norm = float(np.linalg.norm(gap_world[:2]))
        z_gap = float(gap_world[2])
        print(
            f"[grasp-align-{idx+1}] gap_local="
            f"{[round(float(x), 4) for x in gap_local.tolist()]} "
            f"gap_world={ [round(float(x), 4) for x in gap_world.tolist()] }"
        )
        if abs(z_gap) <= GRASP_ALIGN_Z_THRESH and xy_norm <= GRASP_ALIGN_XY_THRESH:
            return

        delta = np.zeros(3, dtype=float)
        delta[0] = float(
            np.clip(gap_world[0], -GRASP_ALIGN_MAX_NUDGE, GRASP_ALIGN_MAX_NUDGE)
        )
        delta[1] = float(
            np.clip(gap_world[1], -GRASP_ALIGN_MAX_NUDGE, GRASP_ALIGN_MAX_NUDGE)
        )
        if abs(z_gap) > 0.015:
            target_z_gap = 0.015 * np.sign(z_gap)
            delta[2] = float(
                np.clip(z_gap - target_z_gap, -GRASP_ALIGN_MAX_NUDGE, GRASP_ALIGN_MAX_NUDGE)
            )
        print(f"[grasp-align-{idx+1}] nudge={np.round(delta, 4).tolist()}")
        _tool("nudge")(SIDE, delta_pos=delta.tolist())


def preclose_insert_along_gripper() -> None:
    ee_quat = np.asarray(_tool("get_robot_state")().arms[SIDE].ee_quat, dtype=float)
    insert_world = R.from_quat(ee_quat).apply(
        np.array([0.0, 0.0, -PRE_CLOSE_INSERT_DISTANCE], dtype=float)
    )
    print(f"[grasp-preclose] insert_world={np.round(insert_world, 4).tolist()}")
    _tool("nudge")(SIDE, delta_pos=insert_world.tolist())


def counter_grasp_from_hover() -> bool:
    for idx in range(6):
        info = _tool("get_task_info")()
        gap_local = get_obj_gap(info)
        ee_quat = np.asarray(_tool("get_robot_state")().arms[SIDE].ee_quat, dtype=float)
        gap_world = R.from_quat(ee_quat).apply(gap_local)
        xy_norm = float(np.linalg.norm(gap_world[:2]))
        z_gap = float(gap_world[2])
        print(
            f"[counter-grasp-{idx+1}] gap_local="
            f"{[round(float(x), 4) for x in gap_local.tolist()]} "
            f"gap_world={ [round(float(x), 4) for x in gap_world.tolist()] }"
        )
        if abs(z_gap) <= 0.02 and xy_norm <= 0.02:
            return True

        delta = np.zeros(3, dtype=float)
        delta[0] = float(np.clip(gap_world[0], -0.03, 0.03))
        delta[1] = float(np.clip(gap_world[1], -0.03, 0.03))
        target_z_gap = 0.01 * np.sign(z_gap) if abs(z_gap) > 0.01 else z_gap
        delta[2] = float(np.clip(z_gap - target_z_gap, -0.04, 0.01))
        print(f"[counter-grasp-{idx+1}] nudge={np.round(delta, 4).tolist()}")
        result = _tool("nudge")(SIDE, delta_pos=delta.tolist())
        success = bool(getattr(result, "success", False))
        if not success:
            return False
    final_gap = R.from_quat(
        np.asarray(_tool("get_robot_state")().arms[SIDE].ee_quat, dtype=float)
    ).apply(get_obj_gap(_tool("get_task_info")()))
    return abs(float(final_gap[2])) <= 0.03 and float(np.linalg.norm(final_gap[:2])) <= 0.025


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
    cabinet_waypoints = None
    obj_name = info.get("obj_name", "object")
    print(f"Task: {env_name}")
    print(f"Object ({obj_name}): {[round(float(x), 3) for x in obj_pos]}")
    print(f"Place target ({place_label}): {[round(float(x), 3) for x in place_pos]}")

    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
    if "CounterToCabinet" in env_name:
        cabinet_waypoints = get_cabinet_waypoints(
            info,
            place_pos=place_pos,
            current_ee_pos=np.asarray(state.arms[SIDE].ee_pos, dtype=float),
        )
        print(
            "Cabinet waypoints: "
            f"approach={[round(float(x), 3) for x in cabinet_waypoints['approach_pos']]} "
            f"insert={[round(float(x), 3) for x in cabinet_waypoints['insert_pos']]} "
            f"settle={[round(float(x), 3) for x in cabinet_waypoints['settle_pos']]} "
            f"front_normal={[round(float(x), 3) for x in cabinet_waypoints['front_normal']]}"
        )

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
    grasp_success = plan_with_pose_candidates(
        obj_now,
        ee_quat,
        label="grasp-approach",
        pos_offsets=grasp_pos_offsets,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=1.0,
        base_quat_candidates=grasp_base_quats,
    )
    if not grasp_success and grasp_base_quats is None:
        grasp_success = plan_with_pose_candidates(
            obj_now,
            ee_quat,
            label="grasp-approach-fallback",
            pos_offsets=grasp_pos_offsets,
            orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
            gripper=1.0,
            base_quat_candidates=build_hover_base_quats(ee_quat),
        )
    if not grasp_success:
        raise RuntimeError("Failed to reach object pose")

    print("\n--- Step 4: Close gripper ---")
    _tool("close_gripper")(SIDE)
    time.sleep(0.25)
    _tool("set_gripper")(SIDE, 0.0)
    time.sleep(0.10)
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

    print("\n--- Step 5: Lift ---")
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

    if cabinet_waypoints is not None:
        print("\n--- Step 6: Move to cabinet opening ---")
        place_hover = cabinet_waypoints["approach_pos"].copy()
        place_hover_offsets = CABINET_APPROACH_POS_OFFSETS
        place_hover_label = "cabinet-approach"
    else:
        print(f"\n--- Step 6: Move above {place_label} ---")
        place_hover = place_pos.copy()
        place_hover[2] += HOVER_HEIGHT
        place_hover_offsets = HOVER_POS_OFFSETS
        place_hover_label = "place-hover"
    place_hover_quats = None
    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
    if cabinet_waypoints is not None:
        place_hover_quats = build_cabinet_base_quats(
            ee_quat, cabinet_waypoints["front_normal"]
        )
    if not plan_with_pose_candidates(
        place_hover,
        ee_quat,
        label=place_hover_label,
        pos_offsets=place_hover_offsets,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=0.0,
        base_quat_candidates=place_hover_quats,
    ):
        raise RuntimeError("Failed to reach place hover pose")

    if cabinet_waypoints is not None:
        print("\n--- Step 7: Insert into cabinet ---")
        place_target = cabinet_waypoints["insert_pos"].copy()
        place_target_offsets = CABINET_INSERT_POS_OFFSETS
        place_target_label = "cabinet-insert"
    else:
        print(f"\n--- Step 7: Lower to {place_label} ---")
        place_target = place_pos.copy()
        place_target[2] += PLACE_CLEARANCE
        place_target_offsets = PLACE_POS_OFFSETS
        place_target_label = "place-lower"
    place_target_quats = None
    state = _tool("get_robot_state")()
    ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
    if cabinet_waypoints is not None:
        place_target_quats = build_cabinet_base_quats(
            ee_quat, cabinet_waypoints["front_normal"]
        )
    if not plan_with_pose_candidates(
        place_target,
        ee_quat,
        label=place_target_label,
        pos_offsets=place_target_offsets,
        orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
        gripper=0.0,
        base_quat_candidates=place_target_quats,
    ):
        raise RuntimeError("Failed to reach place pose")

    if cabinet_waypoints is not None:
        print("\n--- Step 7b: Settle inside cabinet ---")
        state = _tool("get_robot_state")()
        ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
        settle_quats = build_cabinet_base_quats(
            ee_quat, cabinet_waypoints["front_normal"]
        )
        settled = plan_with_pose_candidates(
            cabinet_waypoints["settle_pos"],
            ee_quat,
            label="cabinet-settle",
            pos_offsets=CABINET_SETTLE_POS_OFFSETS,
            orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
            gripper=0.0,
            base_quat_candidates=settle_quats,
        )
        if not settled:
            print("[cabinet-settle] settle refinement failed; releasing from insert pose")

    print("\n--- Step 8: Release ---")
    _tool("open_gripper")(SIDE)
    time.sleep(0.25)
    if hasattr(planner, "set_gripper_qpos"):
        planner.set_gripper_qpos(right_gripper=1.0)

    print("\n--- Step 9: Retract ---")
    if cabinet_waypoints is not None:
        state = _tool("get_robot_state")()
        ee_quat = np.asarray(state.arms[SIDE].ee_quat, dtype=float)
        retract_quats = build_cabinet_base_quats(
            ee_quat, cabinet_waypoints["front_normal"]
        )
        retracted = plan_with_pose_candidates(
            cabinet_waypoints["approach_pos"],
            ee_quat,
            label="cabinet-retract",
            pos_offsets=CABINET_APPROACH_POS_OFFSETS,
            orientation_offsets_deg=ORIENTATION_OFFSETS_DEG,
            gripper=1.0,
            base_quat_candidates=retract_quats,
        )
        if not retracted:
            print("[cabinet-retract] failed; falling back to go_home")
    _tool("go_home")(SIDE)

    final = _tool("get_task_info")()
    print(f"\nSuccess: {final.get('success', False)}")
    print(f"Reward:  {final.get('reward', 0.0)}")


run_pick_place_curobo_task("PickPlaceCounterToCabinet")
