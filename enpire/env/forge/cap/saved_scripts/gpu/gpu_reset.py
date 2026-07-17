"""Minimal single-slot GPU unplug script for real YAM.

Behavior:
  1. Detect the motherboard and derive the target-slot hover using the same
     motherboard-relative localization path as `gpu_debug_left_slot_hover.py`.
  2. Move the left arm to that hover with the gripper open.
  3. Use the top camera to find a bottom-end / metal-bar target near the
     inserted GPU, pull there first to loosen the card, then optionally
     re-hover to the slot-centered grasp and pull the GPU fully free.

This assumes a GPU is already plugged into the selected slot and uses motherboard-relative
slot geometry plus a lightweight top-view VLM query for the initial loosening
pull. After unplugging, the left arm carries the GPU to a fixed right-side
table parking pose, sets it down, retracts, and goes home.

Run with:
  source .forge_env && uv run python run_script.py robot=real_yam \
      script_file=cap/saved_scripts/gpu/gpu_reset.py \
      env.name=yam-real skill_library_path=cap/saved_scripts/skill_library
"""

from __future__ import annotations

import ast
import copy
import numpy as np
import os
from pathlib import Path
import time
import types


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or raw == "" else raw)


def _load_left_slot_hover_helpers():
    script_path = Path.cwd() / "cap" / "saved_scripts" / "gpu" / "gpu_debug_left_slot_hover.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    keep = []
    for idx, node in enumerate(tree.body):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            keep.append(node)
        elif (
            idx == 0
            and isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    helpers = types.ModuleType("gpu_left_slot_hover_helpers")
    helpers.__file__ = str(script_path)
    helpers.__dict__.update({"__builtins__": __builtins__})
    for name in [
        "close_gripper",
        "freespace_move",
        "get_camera_extrinsics",
        "get_camera_image",
        "get_camera_intrinsics",
        "get_robot_state",
        "go_home",
        "open_gripper",
        "render_depth",
        "sample_grasp_pose_3d_bb",
        "segment_object",
    ]:
        if name in globals():
            helpers.__dict__[name] = globals()[name]
    exec(compile(module, str(script_path), "exec"), helpers.__dict__)  # noqa: S102
    return helpers


def _configure_hover_env():
    primary_camera = os.environ.get("GPU_UNPLUG_CAMERA", "").strip() or "top"
    aux_camera = os.environ.get("GPU_UNPLUG_AUX_CAMERA", "").strip() or "left_third"
    aux_prefer_world_pose = os.environ.get("GPU_UNPLUG_AUX_PREFER_WORLD_POSE", "").strip() or "1"
    aux_required = os.environ.get("GPU_UNPLUG_REQUIRE_AUX_CAMERA", "").strip() or "0"
    save_hover_artifacts = os.environ.get("GPU_UNPLUG_SAVE_HOVER_ARTIFACTS", "").strip() or "1"
    os.environ["GPU_SLOT_DEBUG_CAMERA"] = primary_camera
    os.environ["GPU_SLOT_DEBUG_AUX_CAMERA"] = aux_camera
    os.environ["GPU_SLOT_DEBUG_AUX_CAMERA_PREFER_WORLD_POSE"] = aux_prefer_world_pose
    os.environ["GPU_SLOT_DEBUG_REQUIRE_AUX_CAMERA"] = aux_required
    os.environ["GPU_SLOT_DEBUG_SAVE_ARTIFACTS"] = save_hover_artifacts
    os.environ["GPU_TARGET_SOCKET_NUMBER"] = os.environ.get("GPU_TARGET_SOCKET_NUMBER", "1")


_configure_hover_env()
hover = _load_left_slot_hover_helpers()
gh = hover.gh
HOVER_ARTIFACT_DIR = gh._artifact_output_dir("gpu_handover")
TARGET_SOCKET_NUMBER = int(os.environ.get("GPU_TARGET_SOCKET_NUMBER", "1"))
SLOT3_HOVER_Z_EXTRA_M = _env_float("GPU_UNPLUG_SLOT3_HOVER_Z_EXTRA_M", 0.003)
TRANSIT_HOVER_Z_EXTRA_M = _env_float("GPU_UNPLUG_TRANSIT_HOVER_Z_EXTRA_M", 0.025)

GO_HOME_ON_START = _env_flag("GPU_UNPLUG_GO_HOME_ON_START", True)
REFRESH_BEFORE_DESCENT = _env_flag(
    "GPU_UNPLUG_REFRESH_BEFORE_DESCENT",
    int(TARGET_SOCKET_NUMBER) != 3,
)
PRESERVE_REFRESH_Y = _env_flag("GPU_UNPLUG_PRESERVE_REFRESH_Y", True)
SLOT3_PRESERVE_REFRESH_Y = _env_flag("GPU_UNPLUG_SLOT3_PRESERVE_REFRESH_Y", True)
OPEN_RIGHT_ON_START = _env_flag("GPU_UNPLUG_OPEN_RIGHT_ON_START", False)
SCRIPT_COMPLETED = False
PRE_DESCEND_SETTLE_S = _env_float("GPU_UNPLUG_PRE_DESCEND_SETTLE_S", 0.15)
POST_CLOSE_SETTLE_S = _env_float("GPU_UNPLUG_POST_CLOSE_SETTLE_S", 0.25)
POST_LIFT_SETTLE_S = _env_float("GPU_UNPLUG_POST_LIFT_SETTLE_S", 0.15)
POST_RECLAMP_SETTLE_S = _env_float("GPU_UNPLUG_POST_RECLAMP_SETTLE_S", 0.18)
POST_LOOSEN_RELEASE_SETTLE_S = _env_float("GPU_UNPLUG_POST_LOOSEN_RELEASE_SETTLE_S", 0.15)
DESCEND_FROM_HOVER_M = _env_float("GPU_UNPLUG_DESCEND_FROM_HOVER_M", 0.060)
MIN_GRASP_Z_ABOVE_BOARD_M = _env_float("GPU_UNPLUG_MIN_GRASP_Z_ABOVE_BOARD_M", 0.035)
MAX_GRASP_Z_BELOW_HOVER_M = _env_float("GPU_UNPLUG_MAX_GRASP_Z_BELOW_HOVER_M", 0.010)
LIFT_FROM_GRASP_M = _env_float("GPU_UNPLUG_LIFT_FROM_GRASP_M", 0.085)
LIFT_ABOVE_HOVER_M = _env_float("GPU_UNPLUG_LIFT_ABOVE_HOVER_M", 0.030)
FINAL_PULL_X_OFFSET_M = _env_float("GPU_UNPLUG_FINAL_PULL_X_OFFSET_M", -0.015)
BAR_TARGET_X_OFFSET_M = _env_float("GPU_UNPLUG_BAR_TARGET_X_OFFSET_M", -0.108)
SOCKET3_BAR_TARGET_X_OFFSET_M = _env_float(
    "GPU_UNPLUG_SOCKET3_BAR_TARGET_X_OFFSET_M",
    -0.093,
)
BAR_TARGET_Y_OFFSET_M = _env_float("GPU_UNPLUG_BAR_TARGET_Y_OFFSET_M", 0.0)
BAR_DESCEND_FROM_HOVER_M = _env_float("GPU_UNPLUG_BAR_DESCEND_FROM_HOVER_M", 0.010)
BAR_MIN_GRASP_Z_ABOVE_BOARD_M = _env_float(
    "GPU_UNPLUG_BAR_MIN_GRASP_Z_ABOVE_BOARD_M",
    0.050,
)
BAR_MAX_GRASP_Z_BELOW_HOVER_M = _env_float(
    "GPU_UNPLUG_BAR_MAX_GRASP_Z_BELOW_HOVER_M",
    float(MAX_GRASP_Z_BELOW_HOVER_M),
)
BAR_LOOSEN_LIFT_M = _env_float("GPU_UNPLUG_BAR_LOOSEN_LIFT_M", 0.028)
BAR_PULL_FORWARD_X_M = _env_float("GPU_UNPLUG_BAR_PULL_FORWARD_X_M", 0.018)
BAR_PULL_FORWARD_Y_M = _env_float("GPU_UNPLUG_BAR_PULL_FORWARD_Y_M", 0.0)
BAR_PULL_ARC_MID_FRACTION = _env_float("GPU_UNPLUG_BAR_PULL_ARC_MID_FRACTION", 0.45)
BAR_RELEASE_SUPPORT_ABOVE_GRASP_M = _env_float(
    "GPU_UNPLUG_BAR_RELEASE_SUPPORT_ABOVE_GRASP_M",
    0.006,
)
BAR_STAGE_ENABLED = _env_flag("GPU_UNPLUG_BAR_STAGE_ENABLED", True)
BAR_STAGE_REGRASP_ENABLED = _env_flag("GPU_UNPLUG_BAR_STAGE_REGRASP_ENABLED", True)
BAR_HOVER_REFINEMENT_ENABLED = _env_flag("GPU_UNPLUG_BAR_HOVER_REFINEMENT_ENABLED", False)
BAR_HOVER_REFINEMENT_MIN_SHIFT_M = _env_float(
    "GPU_UNPLUG_BAR_HOVER_REFINEMENT_MIN_SHIFT_M",
    0.0025,
)
BAR_HOVER_REFINEMENT_SETTLE_S = _env_float(
    "GPU_UNPLUG_BAR_HOVER_REFINEMENT_SETTLE_S",
    0.10,
)
BAR_HOVER_REFINEMENT_DURATION_S = _env_float(
    "GPU_UNPLUG_BAR_HOVER_REFINEMENT_DURATION_S",
    0.30,
)
BAR_HOVER_REFINEMENT_STEPS = max(
    4,
    int(_env_float("GPU_UNPLUG_BAR_HOVER_REFINEMENT_STEPS", 6)),
)
BAR_KEEP_CURRENT_Y_MAX_DELTA_M = _env_float(
    "GPU_UNPLUG_BAR_KEEP_CURRENT_Y_MAX_DELTA_M",
    0.015,
)
LEFT_X_REFINEMENT_ENABLED = _env_flag("GPU_UNPLUG_LEFT_X_REFINEMENT_ENABLED", False)
LEFT_X_REFINEMENT_CAMERA = os.environ.get("GPU_UNPLUG_LEFT_X_REFINEMENT_CAMERA", "").strip() or "left"
LEFT_X_REFINEMENT_MIN_SHIFT_M = _env_float(
    "GPU_UNPLUG_LEFT_X_REFINEMENT_MIN_SHIFT_M",
    0.0015,
)
LEFT_X_REFINEMENT_MAX_SHIFT_M = _env_float(
    "GPU_UNPLUG_LEFT_X_REFINEMENT_MAX_SHIFT_M",
    0.060,
)
LEFT_TABLE_PLACE_X = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PLACE_X",
    0.528,
)
LEFT_TABLE_PLACE_Y = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PLACE_Y",
    -0.154,
)
LEFT_TABLE_PLACE_Z = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PLACE_Z",
    0.864,
)
LEFT_TABLE_PLACE_RPY = [
    _env_float("GPU_UNPLUG_LEFT_TABLE_PLACE_ROLL_DEG", -52.2),
    _env_float("GPU_UNPLUG_LEFT_TABLE_PLACE_PITCH_DEG", 143.7),
    _env_float("GPU_UNPLUG_LEFT_TABLE_PLACE_YAW_DEG", 16.7),
]
LEFT_TABLE_RELEASE_APPROACH_OFFSET_M = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_RELEASE_APPROACH_OFFSET_M",
    0.04,
)
UNPLUG_GUIDED_DURATION_SCALE = min(
    1.5,
    max(0.5, float(_env_float("GPU_UNPLUG_GUIDED_DURATION_SCALE", 0.90))),
)
UNPLUG_MOVE_PLANNING_SPEED = _env_float(
    "GPU_UNPLUG_MOVE_PLANNING_SPEED",
    float(gh.POST_PICK_PLANNING_SPEED) * 1.10,
)
LEFT_TABLE_PRE_RELEASE_REORIENT_LIFT_M = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PRE_RELEASE_REORIENT_LIFT_M",
    0.08,
)
LEFT_TABLE_PRE_RELEASE_REORIENT_CLEARANCE_M = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PRE_RELEASE_REORIENT_CLEARANCE_M",
    0.03,
)
LEFT_TABLE_RELEASE_GLIDE_DURATION_S = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_RELEASE_GLIDE_DURATION_S",
    0.20,
)
LEFT_TABLE_RELEASE_GLIDE_STEPS = max(
    3,
    int(_env_float("GPU_UNPLUG_LEFT_TABLE_RELEASE_GLIDE_STEPS", 6)),
)
LEFT_TABLE_RELEASE_OPEN_START_ALPHA = min(
    0.95,
    max(0.0, float(_env_float("GPU_UNPLUG_LEFT_TABLE_RELEASE_OPEN_START_ALPHA", 0.65))),
)
LEFT_TABLE_PLACE_POST_RELEASE_SETTLE_S = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PLACE_POST_RELEASE_SETTLE_S",
    0.18,
)
LEFT_TABLE_PLACE_RETRACT_M = _env_float(
    "GPU_UNPLUG_LEFT_TABLE_PLACE_RETRACT_M",
    0.10,
)
DESCEND_DURATION_S = _env_float("GPU_UNPLUG_DESCEND_DURATION_S", 1.20)
DESCEND_STEPS = max(2, int(_env_float("GPU_UNPLUG_DESCEND_STEPS", 12)))
LIFT_DURATION_S = _env_float("GPU_UNPLUG_LIFT_DURATION_S", 1.40)
LIFT_STEPS = max(2, int(_env_float("GPU_UNPLUG_LIFT_STEPS", 14)))
LEFT_TRANSIT_PREOPEN_POS = _env_float("GPU_UNPLUG_LEFT_TRANSIT_PREOPEN_POS", 0.6)
LEFT_GRASP_PREOPEN_POS = _env_float("GPU_UNPLUG_LEFT_GRASP_PREOPEN_POS", 0.6)
LEFT_BAR_PREOPEN_POS = _env_float("GPU_UNPLUG_LEFT_BAR_PREOPEN_POS", 0.6)
LEFT_PREOPEN_DURATION_S = _env_float("GPU_UNPLUG_LEFT_PREOPEN_DURATION_S", 0.35)
LEFT_CLOSE_VEL_LIMIT = _env_float(
    "GPU_UNPLUG_LEFT_CLOSE_VEL_LIMIT",
    float(gh.LEFT_REGRASP_CLOSE_VEL_LIMIT),
)
LEFT_CLOSE_TORQUE_LIMIT = _env_float(
    "GPU_UNPLUG_LEFT_CLOSE_TORQUE_LIMIT",
    float(max(0.8, gh.LEFT_REGRASP_CLOSE_TORQUE_LIMIT, gh.PICK_GRIPPER_CLOSE_TORQUE_LIMIT)),
)
LEFT_RECLAMP_VEL_LIMIT = _env_float(
    "GPU_UNPLUG_LEFT_RECLAMP_VEL_LIMIT",
    float(max(0.8, min(LEFT_CLOSE_VEL_LIMIT, 1.2))),
)
LEFT_RECLAMP_TORQUE_LIMIT = _env_float(
    "GPU_UNPLUG_LEFT_RECLAMP_TORQUE_LIMIT",
    float(min(LEFT_CLOSE_TORQUE_LIMIT, 0.65)),
)
LEFT_RECLAMP_ATTEMPTS = max(
    0,
    int(_env_float("GPU_UNPLUG_LEFT_RECLAMP_ATTEMPTS", 2)),
)
BAR_STAGE_RECLAMP_ATTEMPTS = max(
    0,
    int(_env_float("GPU_UNPLUG_BAR_STAGE_RECLAMP_ATTEMPTS", 0)),
)
LEFT_MIN_HOLD_GRIPPER_POS = _env_float(
    "GPU_UNPLUG_LEFT_MIN_HOLD_GRIPPER_POS",
    0.015,
)


def _best_effort_go_home(context: str):
    try:
        gh.go_home()
        return True
    except Exception as exc:
        print(f"[gpu_unplug] go_home failed during {context}: {exc}")
        return False


def _best_effort_open_grippers():
    _set_left_gripper_position(
        LEFT_TRANSIT_PREOPEN_POS,
        duration_s=float(LEFT_PREOPEN_DURATION_S),
        label="startup safe-open",
    )
    if OPEN_RIGHT_ON_START:
        gh._open_gripper("right", vel_limit=gh.PICK_GRIPPER_OPEN_VEL_LIMIT)


def _apply_target_slot_hover_z_trim(scene, *, stage_label):
    extra_z_m = float(SLOT3_HOVER_Z_EXTRA_M) if int(TARGET_SOCKET_NUMBER) == 3 else 0.0
    if abs(extra_z_m) <= 1e-9:
        return scene
    scene["hover_z"] = float(scene["hover_z"]) + extra_z_m
    hover_world = np.asarray(scene["socket_hover_world"], dtype=np.float64).reshape(3).copy()
    hover_world[2] = float(scene["hover_z"])
    scene["socket_hover_world"] = hover_world
    print(
        "[gpu_unplug] slot hover-z trim applied: "
        f"target_socket={int(TARGET_SOCKET_NUMBER)} "
        f"stage={stage_label} "
        f"extra_z_m={extra_z_m:.4f} "
        f"hover_z={float(scene['hover_z']):.4f}"
    )
    return scene


def _with_transit_hover_clearance(scene, *, stage_label):
    extra_z_m = float(TRANSIT_HOVER_Z_EXTRA_M)
    if abs(extra_z_m) <= 1e-9:
        return scene
    clearance_scene = copy.deepcopy(scene)
    base_hover_z = float(clearance_scene["hover_z"])
    hover_world = np.asarray(
        clearance_scene["socket_hover_world"],
        dtype=np.float64,
    ).reshape(3).copy()
    clearance_scene["hover_z"] = base_hover_z + extra_z_m
    hover_world[2] = float(clearance_scene["hover_z"])
    clearance_scene["socket_hover_world"] = hover_world
    print(
        "[gpu_unplug] transit hover clearance applied: "
        f"stage={stage_label} "
        f"base_hover_z={base_hover_z:.4f} "
        f"extra_z_m={extra_z_m:.4f} "
        f"hover_z={float(clearance_scene['hover_z']):.4f}"
    )
    return clearance_scene


def _preserve_refresh_y(previous_scene, refined_scene, *, stage_label):
    preserve_y = bool(PRESERVE_REFRESH_Y)
    if int(TARGET_SOCKET_NUMBER) == 3:
        preserve_y = preserve_y or bool(SLOT3_PRESERVE_REFRESH_Y)
    if not preserve_y:
        return refined_scene
    try:
        previous_hover = np.asarray(
            previous_scene["socket_hover_world"],
            dtype=np.float64,
        ).reshape(3)
        refined_hover = np.asarray(
            refined_scene["socket_hover_world"],
            dtype=np.float64,
        ).reshape(3).copy()
    except Exception as exc:
        print(f"[gpu_unplug] refresh Y preserve skipped: {exc}")
        return refined_scene

    old_y = float(refined_hover[1])
    target_y = float(previous_hover[1])
    delta_y = target_y - old_y
    if abs(delta_y) <= 1e-9:
        return refined_scene

    refined_hover[1] = target_y
    refined_scene["socket_hover_world"] = refined_hover
    if refined_scene.get("socket_hover_center_offset_world") is not None:
        center_offset_world = np.asarray(
            refined_scene["socket_hover_center_offset_world"],
            dtype=np.float64,
        ).reshape(3).copy()
        center_offset_world[1] += delta_y
        refined_scene["socket_hover_center_offset_world"] = center_offset_world
    if refined_scene.get("socket_record") is not None:
        try:
            refined_scene["socket_record"]["world"] = refined_hover.copy()
        except Exception:
            pass

    cam_K = gh._camera_matrix(hover.CAMERA)
    T_cam_world = hover._tracking_camera_T_cam_world(hover.CAMERA)
    if cam_K is not None and T_cam_world is not None:
        hover._set_scene_hover_target(
            refined_scene,
            refined_hover,
            cam_K,
            T_cam_world,
        )
    print(
        "[gpu_unplug] refresh Y preserved: "
        f"stage={stage_label} "
        f"target_socket={int(TARGET_SOCKET_NUMBER)} "
        f"refreshed_y={old_y:.4f} "
        f"kept_y={target_y:.4f} "
        f"delta_y_m={delta_y:.4f} "
        f"hover_world={[round(float(v), 4) for v in refined_hover.tolist()]}"
    )
    return refined_scene


def _bar_target_x_offset_m():
    if int(TARGET_SOCKET_NUMBER) == 3:
        return float(SOCKET3_BAR_TARGET_X_OFFSET_M)
    return float(BAR_TARGET_X_OFFSET_M)


def _set_left_gripper_position(target_gp, *, duration_s, label):
    env = gh._tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for left gripper position move")

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
            f"[gpu_unplug] {label}: left gripper already near target "
            f"(current={left_gp:.3f}, target={target_gp:.3f})"
        )
        return left_gp

    print(
        f"[gpu_unplug] {label}: set left gripper position "
        f"current={left_gp:.3f} target={target_gp:.3f} duration_s={duration_s:.2f}"
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
        raise RuntimeError(result.get("reason", "left gripper position move failed"))
    return target_gp


def _left_release_glide(target_pos, target_rpy, *, duration_s, num_steps, open_start_alpha):
    env = gh._tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for left release glide")

    duration_s = max(0.08, float(duration_s))
    num_steps = max(3, int(num_steps))
    open_start_alpha = min(0.95, max(0.0, float(open_start_alpha)))
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = gh._display_rpy_to_rotation(target_rpy).as_quat().astype(np.float64)

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

    start_rot = gh.Rotation.from_quat(left_start_quat)
    target_rot = gh.Rotation.from_quat(target_quat)
    delta_rot = target_rot * start_rot.inv()

    left_waypoints = []
    right_waypoints = []
    left_grippers = []
    right_grippers = []
    timestamps = []

    with env._kin_lock:
        env.kin.forward_kinematics(left_jp, right_jp)
        cur_left_jp = left_jp.copy()
        cur_right_jp = right_jp.copy()
        for step_index in range(1, num_steps + 1):
            alpha = float(step_index) / float(num_steps)
            interp_pos = left_start_pos + alpha * (target_pos - left_start_pos)
            interp_rot = gh.Rotation.from_rotvec(delta_rot.as_rotvec() * alpha) * start_rot
            interp_quat = interp_rot.as_quat().astype(np.float64)
            env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
            next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                interp_pos,
                interp_quat,
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
            if alpha <= open_start_alpha:
                interp_gp = left_gp
            else:
                open_alpha = (alpha - open_start_alpha) / max(1e-6, 1.0 - open_start_alpha)
                interp_gp = left_gp + open_alpha * (1.0 - left_gp)
            left_waypoints.append(cur_left_jp.copy())
            right_waypoints.append(cur_right_jp.copy())
            left_grippers.append([float(np.clip(interp_gp, 0.0, 1.0))])
            right_grippers.append([right_gp])
            timestamps.append(alpha * duration_s)

    print(
        "[gpu_unplug] left release glide: "
        f"target_pos={[round(float(v), 4) for v in target_pos.tolist()]} "
        f"target_rpy={[round(float(v), 1) for v in target_rpy]} "
        f"duration_s={duration_s:.2f} steps={num_steps} "
        f"open_start_alpha={open_start_alpha:.2f}"
    )
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
        raise RuntimeError(result.get("reason", "left release glide failed"))


def _build_grasp_pos(scene):
    hover_world = [float(v) for v in scene["socket_hover_world"].tolist()]
    return _build_grasp_pos_for_world(
        scene,
        hover_world,
        descend_from_hover_m=float(DESCEND_FROM_HOVER_M),
    )


def _build_grasp_pos_for_world(
    scene,
    target_world,
    *,
    descend_from_hover_m,
    min_grasp_z_above_board_m=None,
    max_grasp_z_below_hover_m=None,
):
    target_world = [float(v) for v in target_world]
    hover_z = float(scene["hover_z"])
    top_z = float(scene["motherboard_top_z"])
    min_grasp_z_above_board_m = float(
        MIN_GRASP_Z_ABOVE_BOARD_M
        if min_grasp_z_above_board_m is None
        else min_grasp_z_above_board_m
    )
    max_grasp_z_below_hover_m = float(
        MAX_GRASP_Z_BELOW_HOVER_M
        if max_grasp_z_below_hover_m is None
        else max_grasp_z_below_hover_m
    )
    grasp_z = max(
        top_z + min_grasp_z_above_board_m,
        hover_z - float(descend_from_hover_m),
    )
    grasp_z = min(grasp_z, hover_z - max_grasp_z_below_hover_m)
    if grasp_z <= top_z:
        raise RuntimeError(
            f"computed grasp z is not above board plane: grasp_z={grasp_z:.4f} top_z={top_z:.4f}"
        )
    print(
        "[gpu_unplug] computed grasp target: "
        f"target_world={[round(float(v), 4) for v in target_world]} "
        f"top_z={top_z:.4f} hover_z={hover_z:.4f} grasp_z={grasp_z:.4f} "
        f"descend_from_hover_m={float(descend_from_hover_m):.4f} "
        f"min_grasp_z_above_board_m={min_grasp_z_above_board_m:.4f}"
    )
    return [float(target_world[0]), float(target_world[1]), float(grasp_z)]


def _build_lift_pos(scene, grasp_pos):
    hover_z = float(scene["hover_z"])
    lift_z = max(
        float(grasp_pos[2]) + float(LIFT_FROM_GRASP_M),
        hover_z + float(LIFT_ABOVE_HOVER_M),
    )
    lift_pos = [
        float(grasp_pos[0]) + float(FINAL_PULL_X_OFFSET_M),
        float(grasp_pos[1]),
        float(lift_z),
    ]
    print(
        "[gpu_unplug] final pull lift target: "
        f"grasp_pos={[round(float(v), 4) for v in grasp_pos]} "
        f"x_offset_m={float(FINAL_PULL_X_OFFSET_M):.4f} "
        f"lift_pos={[round(float(v), 4) for v in lift_pos]}"
    )
    return lift_pos


def _build_short_lift_pos(scene, grasp_pos):
    hover_z = float(scene["hover_z"])
    lift_z = max(
        float(grasp_pos[2]) + float(BAR_LOOSEN_LIFT_M),
        min(float(hover_z), float(grasp_pos[2]) + float(BAR_LOOSEN_LIFT_M)),
    )
    return [float(grasp_pos[0]), float(grasp_pos[1]), float(lift_z)]


def _build_bar_arc_lift_waypoints(scene, grasp_pos):
    grasp_pos = [float(v) for v in grasp_pos]
    final_pos = _build_short_lift_pos(scene, grasp_pos)
    final_pos[0] += float(BAR_PULL_FORWARD_X_M)
    final_pos[1] += float(BAR_PULL_FORWARD_Y_M)

    mid_fraction = min(0.9, max(0.1, float(BAR_PULL_ARC_MID_FRACTION)))
    mid_pos = [
        float(grasp_pos[0]) + float(BAR_PULL_FORWARD_X_M) * mid_fraction,
        float(grasp_pos[1]) + float(BAR_PULL_FORWARD_Y_M) * mid_fraction,
        float(grasp_pos[2]) + (float(final_pos[2]) - float(grasp_pos[2])) * mid_fraction,
    ]
    print(
        "[gpu_unplug] bar pull arc: "
        f"grasp_pos={[round(float(v), 4) for v in grasp_pos]} "
        f"mid_pos={[round(float(v), 4) for v in mid_pos]} "
        f"final_pos={[round(float(v), 4) for v in final_pos]} "
        f"forward_x_m={float(BAR_PULL_FORWARD_X_M):.4f} "
        f"forward_y_m={float(BAR_PULL_FORWARD_Y_M):.4f}"
    )
    return mid_pos, final_pos


def _build_bar_release_support_pos(grasp_pos):
    return [
        float(grasp_pos[0]),
        float(grasp_pos[1]),
        float(grasp_pos[2]) + float(BAR_RELEASE_SUPPORT_ABOVE_GRASP_M),
    ]


def _guided_left_move(target_pos, target_rpy, *, duration_s, num_steps, label):
    requested_duration_s = float(duration_s)
    duration_s = max(0.08, requested_duration_s * float(UNPLUG_GUIDED_DURATION_SCALE))
    print(
        f"[gpu_unplug] {label}: target_pos="
        f"{[round(float(v), 4) for v in target_pos]} "
        f"target_rpy={[round(float(v), 1) for v in target_rpy]} "
        f"duration_s={float(duration_s):.2f} "
        f"(requested={requested_duration_s:.2f}) "
        f"steps={int(num_steps)}"
    )
    gh._guided_hover_plane_move(
        "left",
        [float(v) for v in target_pos],
        [float(v) for v in target_rpy],
        duration_s=float(duration_s),
        num_steps=int(num_steps),
    )


def _save_hover_visualization(scene, artifact_prefix, *, camera=None):
    if not bool(getattr(hover, "SAVE_DEBUG_ARTIFACTS", False)):
        return
    camera = str(camera or hover.CAMERA)
    try:
        rgb = get_camera_image(camera=camera)
        seg_record = gh._segment_motherboard_mask(camera=camera)
        hover._save_hover_debug_artifacts(
            rgb,
            camera,
            scene,
            seg_record=seg_record,
            artifact_prefix=str(artifact_prefix),
        )
        print(
            "[gpu_unplug] saved hover visualization: "
            f"artifact_prefix={artifact_prefix!r} "
            f"camera={camera!r} "
            f"dir={HOVER_ARTIFACT_DIR}"
        )
    except Exception as exc:
        print(
            "[gpu_unplug] hover visualization save failed: "
            f"artifact_prefix={artifact_prefix!r} camera={camera!r} exc={exc}"
        )


def _apply_left_x_refinement(reference_scene, scene, *, stage_label, artifact_prefix):
    if not LEFT_X_REFINEMENT_ENABLED:
        return scene
    if str(hover.CAMERA).strip().lower() == "left":
        return scene
    if str(LEFT_X_REFINEMENT_CAMERA).strip().lower() != "left":
        print(
            "[gpu_unplug] left-x refinement skipped: "
            f"unsupported camera={LEFT_X_REFINEMENT_CAMERA!r}"
        )
        return scene
    try:
        aux_obs = hover._camera_segmentation_observation(
            LEFT_X_REFINEMENT_CAMERA,
            float(reference_scene["motherboard_top_z"]),
        )
    except Exception as exc:
        print(f"[gpu_unplug] left-x refinement unavailable during {stage_label}: {exc}")
        return scene

    if not bool(aux_obs.get("guard_valid")) or not bool(aux_obs.get("world_valid")):
        print(
            "[gpu_unplug] left-x refinement skipped: "
            f"stage={stage_label} "
            f"guard_valid={bool(aux_obs.get('guard_valid'))} "
            f"world_valid={bool(aux_obs.get('world_valid'))} "
            f"score={float(aux_obs.get('score', 0.0) or 0.0):.3f}"
        )
        return scene

    top_center_world = np.asarray(scene["motherboard_center_world"], dtype=np.float64).reshape(3)
    left_center_world = np.asarray(aux_obs["center_world"], dtype=np.float64).reshape(3)
    delta_x_m = float(left_center_world[0] - top_center_world[0])
    abs_delta_x_m = abs(delta_x_m)
    if abs_delta_x_m < float(LEFT_X_REFINEMENT_MIN_SHIFT_M):
        print(
            "[gpu_unplug] left-x refinement skipped: "
            f"stage={stage_label} delta_x_m={delta_x_m:.4f} below min shift"
        )
        return scene

    clipped_delta_x_m = float(
        max(-float(LEFT_X_REFINEMENT_MAX_SHIFT_M), min(float(LEFT_X_REFINEMENT_MAX_SHIFT_M), delta_x_m))
    )
    chosen_center_world = top_center_world.copy()
    chosen_center_world[0] += clipped_delta_x_m
    chosen_anchor_world = np.asarray(
        scene["motherboard_right_edge_anchor_world"],
        dtype=np.float64,
    ).reshape(3).copy()
    chosen_anchor_world[0] += clipped_delta_x_m
    center_update = dict(scene.get("motherboard_center_update") or {})
    center_update.update(
        {
            "proposed_shift_m": float(abs_delta_x_m),
            "applied_shift_m": float(abs(clipped_delta_x_m)),
            "held": bool(abs_delta_x_m > float(LEFT_X_REFINEMENT_MAX_SHIFT_M)),
            "hold_reason": (
                "left_x_clamped"
                if abs_delta_x_m > float(LEFT_X_REFINEMENT_MAX_SHIFT_M)
                else "left_x"
            ),
        }
    )
    registration = {
        "method": "left_x_only",
        "camera": str(LEFT_X_REFINEMENT_CAMERA),
        "score": float(aux_obs.get("score", 0.0) or 0.0),
        "dx_px": 0,
        "dy_px": 0,
    }
    refined_scene = hover._apply_world_estimate_to_scene(
        reference_scene,
        scene,
        center_world=chosen_center_world,
        right_edge_anchor_world=chosen_anchor_world,
        registration=registration,
        cam_K=gh._camera_matrix(hover.CAMERA),
        T_cam_world=hover._tracking_camera_T_cam_world(hover.CAMERA),
        center_update=center_update,
        pose_source_camera=f"{hover.CAMERA}+left_x",
        aux_score=float(aux_obs.get("score", 0.0) or 0.0),
        localization_mode="edge",
    )
    print(
        "[gpu_unplug] left-x refinement applied: "
        f"stage={stage_label} "
        f"top_center_x={float(top_center_world[0]):.4f} "
        f"left_center_x={float(left_center_world[0]):.4f} "
        f"delta_x_m={delta_x_m:.4f} "
        f"applied_delta_x_m={clipped_delta_x_m:.4f} "
        f"hover_world={[round(float(v), 4) for v in refined_scene['socket_hover_world'].tolist()]}"
    )
    _save_hover_visualization(refined_scene, artifact_prefix)
    return refined_scene


def _refresh_hover_scene(reference_scene, current_scene):
    if not REFRESH_BEFORE_DESCENT:
        return current_scene
    try:
        refined_scene = hover._refresh_scene_from_sam3(
            reference_scene,
            current_scene,
            camera=hover.CAMERA,
            artifact_prefix="pre_unplug",
        )
        refined_scene = _apply_left_x_refinement(
            reference_scene,
            refined_scene,
            stage_label="pre_unplug_refresh",
            artifact_prefix="gpu_unplug_pre_unplug_hover_left_x",
        )
        refined_scene = _preserve_refresh_y(
            current_scene,
            refined_scene,
            stage_label="pre_unplug_refresh",
        )
        refined_scene = _apply_target_slot_hover_z_trim(
            refined_scene,
            stage_label="pre_unplug_refresh",
        )
        hover_pos = hover._move_left_to_hover(
            refined_scene,
            hover._resolve_hover_rpy(),
            fixed_plane_z=float(refined_scene["hover_z"]),
            guided=True,
            guided_duration_s=float(hover.REACTIVE_GUIDED_DURATION_S),
            guided_steps=int(hover.REACTIVE_GUIDED_STEPS),
        )
        print(
            "[gpu_unplug] refreshed hover before descent: "
            f"pos={[round(float(v), 4) for v in hover_pos]}"
        )
        _save_hover_visualization(refined_scene, "gpu_unplug_pre_unplug_hover")
        return refined_scene
    except Exception as exc:
        print(f"[gpu_unplug] pre-descent hover refresh failed, keeping current hover: {exc}")
        return current_scene


def _build_bar_hover_scene(scene):
    bar_scene = copy.deepcopy(scene)
    bar_target_x_offset_m = _bar_target_x_offset_m()
    if bar_scene.get("socket_hover_center_offset_world") is not None:
        center_offset_world = np.asarray(
            bar_scene["socket_hover_center_offset_world"],
            dtype=np.float64,
        ).reshape(3).copy()
        center_offset_world[0] += bar_target_x_offset_m
        center_offset_world[1] += float(BAR_TARGET_Y_OFFSET_M)
        center_offset_world[2] = 0.0
        bar_scene["socket_hover_center_offset_world"] = center_offset_world
    if bar_scene.get("socket_hover_x_from_right_edge_m") is not None:
        bar_scene["socket_hover_x_from_right_edge_m"] = float(
            bar_scene["socket_hover_x_from_right_edge_m"]
        ) + bar_target_x_offset_m
    if bar_scene.get("socket_hover_edge_offset_m") is not None:
        side_sign = float(bar_scene.get("socket_hover_side_sign", 1.0) or 1.0)
        bar_scene["socket_hover_edge_offset_m"] = float(
            bar_scene["socket_hover_edge_offset_m"]
        ) - side_sign * float(BAR_TARGET_Y_OFFSET_M)

    top_z = float(bar_scene["motherboard_top_z"])
    bar_hover_world = hover._derive_biased_socket_hover_world(bar_scene, top_z)
    bar_hover_world = np.asarray(bar_hover_world, dtype=np.float64).reshape(3)
    if abs(float(BAR_TARGET_Y_OFFSET_M)) <= 1e-9:
        current_y = float(gh._robot_vec(get_robot_state(), "left", "ee_pos")[1])
        y_delta_m = float(current_y - bar_hover_world[1])
        if abs(y_delta_m) <= float(BAR_KEEP_CURRENT_Y_MAX_DELTA_M):
            bar_hover_world[1] = current_y
            if bar_scene.get("socket_hover_center_offset_world") is not None:
                center_offset_world = np.asarray(
                    bar_scene["socket_hover_center_offset_world"],
                    dtype=np.float64,
                ).reshape(3).copy()
                center_offset_world[1] += y_delta_m
                bar_scene["socket_hover_center_offset_world"] = center_offset_world
            print(
                "[gpu_unplug] bar-stage target keeps current Y before -x move: "
                f"current_y={current_y:.4f} "
                f"perception_y={float(bar_hover_world[1] - y_delta_m):.4f} "
                f"applied_delta_y_m={y_delta_m:.4f}"
            )
    bar_scene["socket_hover_world"] = bar_hover_world

    cam_K = gh._camera_matrix(hover.CAMERA)
    T_cam_world = hover._tracking_camera_T_cam_world(hover.CAMERA)
    if cam_K is not None and T_cam_world is not None:
        hover._set_scene_hover_target(
            bar_scene,
            bar_hover_world,
            cam_K,
            T_cam_world,
        )
    print(
        "[gpu_unplug] bar-stage scene from slot geometry: "
        f"base_hover_world={[round(float(v), 4) for v in np.asarray(scene['socket_hover_world'], dtype=np.float64).reshape(3).tolist()]} "
        f"bar_hover_world={[round(float(v), 4) for v in bar_hover_world.tolist()]} "
        f"x_offset_m={bar_target_x_offset_m:.4f} "
        f"y_offset_m={float(BAR_TARGET_Y_OFFSET_M):.4f} "
        f"localization_mode={bar_scene.get('socket_hover_localization_mode', 'edge')!r}"
    )
    return bar_scene


def _acquire_bar_hover_scene(reference_scene, current_scene, hover_rpy):
    bar_reference_scene = _build_bar_hover_scene(current_scene)
    bar_reference_scene = _apply_left_x_refinement(
        bar_reference_scene,
        bar_reference_scene,
        stage_label="bar_reference",
        artifact_prefix="gpu_unplug_bar_reference_hover_left_x",
    )
    print("[gpu_unplug] bar-stage hover acquisition via helper motherboard relocalization")
    bar_hover_pos, bar_current_scene = hover._acquire_initial_hover(
        reference_scene=bar_reference_scene,
        hover_rpy=hover_rpy,
        camera=hover.CAMERA,
    )
    print(
        "[gpu_unplug] bar-stage hover acquired: "
        f"pos={[round(float(v), 4) for v in bar_hover_pos]}"
    )
    _save_hover_visualization(bar_current_scene, "gpu_unplug_bar_hover")
    return bar_reference_scene, bar_current_scene


def _refine_bar_hover_scene(reference_scene, current_scene, hover_rpy):
    if not BAR_HOVER_REFINEMENT_ENABLED:
        return current_scene
    if float(BAR_HOVER_REFINEMENT_SETTLE_S) > 0.0:
        time.sleep(float(BAR_HOVER_REFINEMENT_SETTLE_S))
    try:
        refined_scene = hover._refresh_scene_from_sam3(
            reference_scene,
            current_scene,
            camera=hover.CAMERA,
            artifact_prefix="bar_hover",
        )
        prev_hover_world = np.asarray(
            current_scene["socket_hover_world"],
            dtype=np.float64,
        ).reshape(3)
        refined_hover_world = np.asarray(
            refined_scene["socket_hover_world"],
            dtype=np.float64,
        ).reshape(3)
        delta_xy_m = float(np.linalg.norm(refined_hover_world[:2] - prev_hover_world[:2]))
        print(
            "[gpu_unplug] bar hover refinement: "
            f"prev_hover_world={[round(float(v), 4) for v in prev_hover_world.tolist()]} "
            f"refined_hover_world={[round(float(v), 4) for v in refined_hover_world.tolist()]} "
            f"delta_xy_m={delta_xy_m:.4f}"
        )
        if delta_xy_m >= float(BAR_HOVER_REFINEMENT_MIN_SHIFT_M):
            hover._move_left_to_hover(
                refined_scene,
                hover_rpy,
                fixed_plane_z=float(refined_scene["hover_z"]),
                guided=True,
                guided_duration_s=float(BAR_HOVER_REFINEMENT_DURATION_S),
                guided_steps=int(BAR_HOVER_REFINEMENT_STEPS),
            )
        _save_hover_visualization(refined_scene, "gpu_unplug_bar_hover_refined")
        return refined_scene
    except Exception as exc:
        print(f"[gpu_unplug] bar hover refinement failed, keeping prior bar hover: {exc}")
        return current_scene


def _verify_left_hold_or_raise(context: str):
    left_gp = gh._gripper_pos("left")
    print(f"[gpu_unplug] {context}: left gripper pos={left_gp:.4f}")
    if left_gp <= float(LEFT_MIN_HOLD_GRIPPER_POS):
        raise RuntimeError(
            f"{context}: left gripper hold looks empty "
            f"(gripper_pos={left_gp:.4f}, required>{float(LEFT_MIN_HOLD_GRIPPER_POS):.4f})"
        )
    return left_gp


def _firm_close_left_gripper(*, reclamp_attempts=None, label_prefix="close left gripper"):
    reclamp_attempts = int(
        LEFT_RECLAMP_ATTEMPTS if reclamp_attempts is None else max(0, int(reclamp_attempts))
    )
    print(
        f"[gpu_unplug] {label_prefix}: "
        f"vel_limit={float(LEFT_CLOSE_VEL_LIMIT):.3f} "
        f"torque_limit={float(LEFT_CLOSE_TORQUE_LIMIT):.3f}"
    )
    gh._close_gripper(
        "left",
        vel_limit=LEFT_CLOSE_VEL_LIMIT,
        torque_limit=LEFT_CLOSE_TORQUE_LIMIT,
    )
    if float(POST_CLOSE_SETTLE_S) > 0.0:
        time.sleep(float(POST_CLOSE_SETTLE_S))
    left_gp = gh._gripper_pos("left")
    print(f"[gpu_unplug] post-close gripper pos={left_gp:.4f}")

    for attempt in range(1, reclamp_attempts + 1):
        print(
            "[gpu_unplug] re-clamp left gripper: "
            f"attempt={attempt}/{reclamp_attempts} "
            f"vel_limit={float(LEFT_RECLAMP_VEL_LIMIT):.3f} "
            f"torque_limit={float(LEFT_RECLAMP_TORQUE_LIMIT):.3f}"
        )
        gh._close_gripper(
            "left",
            vel_limit=LEFT_RECLAMP_VEL_LIMIT,
            torque_limit=LEFT_RECLAMP_TORQUE_LIMIT,
        )
        if float(POST_RECLAMP_SETTLE_S) > 0.0:
            time.sleep(float(POST_RECLAMP_SETTLE_S))
        left_gp = gh._gripper_pos("left")
        print(f"[gpu_unplug] post-reclamp gripper pos={left_gp:.4f}")
    return left_gp


def _place_left_held_gpu_on_table():
    place_release_pos = [
        float(LEFT_TABLE_PLACE_X),
        float(LEFT_TABLE_PLACE_Y),
        float(LEFT_TABLE_PLACE_Z),
    ]
    place_rpy = [float(v) for v in LEFT_TABLE_PLACE_RPY]
    state = get_robot_state()
    current_left_pos = np.asarray(gh._robot_vec(state, "left", "ee_pos"), dtype=np.float64).reshape(3)
    reorient_pos = current_left_pos.copy()
    reorient_pos[2] = max(
        float(current_left_pos[2]) + float(LEFT_TABLE_PRE_RELEASE_REORIENT_LIFT_M),
        float(LEFT_TABLE_PLACE_Z) + float(LEFT_TABLE_PRE_RELEASE_REORIENT_CLEARANCE_M),
    )
    reorient_pos = [float(v) for v in reorient_pos.tolist()]
    print(
        "[gpu_unplug] Step 12a: reorient left-held GPU horizontal before table transfer "
        f"target_pos={[round(float(v), 4) for v in reorient_pos]} "
        f"target_rpy={[round(float(v), 1) for v in place_rpy]}"
    )
    gh._move_with_speed(
        "left",
        reorient_pos,
        place_rpy,
        planning_speed=float(UNPLUG_MOVE_PLANNING_SPEED),
    )
    state = get_robot_state()
    current_left_pos = np.asarray(gh._robot_vec(state, "left", "ee_pos"), dtype=np.float64).reshape(3)
    release_pos_arr = np.asarray(place_release_pos, dtype=np.float64).reshape(3)
    approach_dir = current_left_pos - release_pos_arr
    approach_norm = float(np.linalg.norm(approach_dir))
    if approach_norm < 1e-6:
        approach_pos = release_pos_arr.copy()
    else:
        approach_pos = release_pos_arr + (
            approach_dir / approach_norm
        ) * float(LEFT_TABLE_RELEASE_APPROACH_OFFSET_M)
    place_approach_pos = [float(v) for v in approach_pos.tolist()]
    print(
        "[gpu_unplug] Step 12: move left-held GPU to fixed right-side release approach "
        f"approach_pos={[round(float(v), 4) for v in place_approach_pos]} "
        f"release_pos={[round(float(v), 4) for v in place_release_pos]} "
        f"rpy={[round(float(v), 1) for v in place_rpy]}"
    )
    gh._move_with_speed(
        "left",
        place_approach_pos,
        place_rpy,
        planning_speed=float(UNPLUG_MOVE_PLANNING_SPEED),
    )
    print("[gpu_unplug] Step 13: glide into fixed release pose while opening left gripper")
    _left_release_glide(
        place_release_pos,
        place_rpy,
        duration_s=float(LEFT_TABLE_RELEASE_GLIDE_DURATION_S),
        num_steps=int(LEFT_TABLE_RELEASE_GLIDE_STEPS),
        open_start_alpha=float(LEFT_TABLE_RELEASE_OPEN_START_ALPHA),
    )
    if float(LEFT_TABLE_PLACE_POST_RELEASE_SETTLE_S) > 0.0:
        time.sleep(float(LEFT_TABLE_PLACE_POST_RELEASE_SETTLE_S))
    gh._cartesian_retract_up_after_release("left", float(LEFT_TABLE_PLACE_RETRACT_M))
    gh._move_side_to_joint_home("left", close_gripper_after=False)


def main():
    global SCRIPT_COMPLETED
    print(
        "[gpu_unplug] Config: "
        f"camera={hover.CAMERA!r} "
        f"aux_camera={hover.AUX_CAMERA!r} "
        f"aux_prefer_world_pose={bool(hover.AUX_CAMERA_PREFER_WORLD_POSE)} "
        f"aux_required={bool(hover.AUX_CAMERA_REQUIRED)} "
        f"aux_has_calibrated_xml_world={bool(hover._calibrated_aux_camera_T_cam_world(hover.AUX_CAMERA) is not None)} "
        f"left_x_refinement_enabled={bool(LEFT_X_REFINEMENT_ENABLED)} "
        f"left_x_refinement_camera={LEFT_X_REFINEMENT_CAMERA!r} "
        f"target_socket={int(TARGET_SOCKET_NUMBER)} "
        f"guided_duration_scale={float(UNPLUG_GUIDED_DURATION_SCALE):.3f} "
        f"move_planning_speed={float(UNPLUG_MOVE_PLANNING_SPEED):.3f} "
        f"slot3_hover_z_extra_m={float(SLOT3_HOVER_Z_EXTRA_M):.4f} "
        f"transit_hover_z_extra_m={float(TRANSIT_HOVER_Z_EXTRA_M):.4f} "
        f"go_home_on_start={bool(GO_HOME_ON_START)} "
        f"refresh_before_descent={bool(REFRESH_BEFORE_DESCENT)} "
        f"preserve_refresh_y={bool(PRESERVE_REFRESH_Y)} "
        f"slot3_preserve_refresh_y={bool(SLOT3_PRESERVE_REFRESH_Y)} "
        f"descend_from_hover_m={float(DESCEND_FROM_HOVER_M):.4f} "
        f"min_grasp_z_above_board_m={float(MIN_GRASP_Z_ABOVE_BOARD_M):.4f} "
        f"lift_from_grasp_m={float(LIFT_FROM_GRASP_M):.4f} "
        f"lift_above_hover_m={float(LIFT_ABOVE_HOVER_M):.4f} "
        f"final_pull_x_offset_m={float(FINAL_PULL_X_OFFSET_M):.4f} "
        f"bar_stage_enabled={bool(BAR_STAGE_ENABLED)} "
        f"bar_stage_regrasp_enabled={bool(BAR_STAGE_REGRASP_ENABLED)} "
        f"bar_target_x_offset_m={float(BAR_TARGET_X_OFFSET_M):.4f} "
        f"socket3_bar_target_x_offset_m={float(SOCKET3_BAR_TARGET_X_OFFSET_M):.4f} "
        f"active_bar_target_x_offset_m={_bar_target_x_offset_m():.4f} "
        f"bar_target_y_offset_m={float(BAR_TARGET_Y_OFFSET_M):.4f} "
        f"bar_descend_from_hover_m={float(BAR_DESCEND_FROM_HOVER_M):.4f} "
        f"bar_min_grasp_z_above_board_m={float(BAR_MIN_GRASP_Z_ABOVE_BOARD_M):.4f} "
        f"bar_loosen_lift_m={float(BAR_LOOSEN_LIFT_M):.4f} "
        f"bar_pull_forward_x_m={float(BAR_PULL_FORWARD_X_M):.4f} "
        f"bar_pull_forward_y_m={float(BAR_PULL_FORWARD_Y_M):.4f} "
        f"bar_stage_reclamp_attempts={int(BAR_STAGE_RECLAMP_ATTEMPTS)} "
        f"bar_hover_refinement_enabled={bool(BAR_HOVER_REFINEMENT_ENABLED)} "
        f"bar_keep_current_y_max_delta_m={float(BAR_KEEP_CURRENT_Y_MAX_DELTA_M):.4f} "
        f"left_transit_preopen_pos={float(LEFT_TRANSIT_PREOPEN_POS):.3f} "
        f"left_bar_preopen_pos={float(LEFT_BAR_PREOPEN_POS):.3f} "
        f"left_grasp_preopen_pos={float(LEFT_GRASP_PREOPEN_POS):.3f} "
        f"left_close_torque={float(LEFT_CLOSE_TORQUE_LIMIT):.3f} "
        f"left_reclamp_torque={float(LEFT_RECLAMP_TORQUE_LIMIT):.3f} "
        f"left_reclamp_attempts={int(LEFT_RECLAMP_ATTEMPTS)} "
        f"left_table_pre_release_reorient_lift_m={float(LEFT_TABLE_PRE_RELEASE_REORIENT_LIFT_M):.4f} "
        f"left_table_pre_release_reorient_clearance_m={float(LEFT_TABLE_PRE_RELEASE_REORIENT_CLEARANCE_M):.4f} "
        f"left_table_place_xyz=({float(LEFT_TABLE_PLACE_X):.3f}, {float(LEFT_TABLE_PLACE_Y):.3f}, {float(LEFT_TABLE_PLACE_Z):.3f}) "
        f"left_table_place_rpy=({float(LEFT_TABLE_PLACE_RPY[0]):.1f}, {float(LEFT_TABLE_PLACE_RPY[1]):.1f}, {float(LEFT_TABLE_PLACE_RPY[2]):.1f})"
    )

    if GO_HOME_ON_START:
        print("[gpu_unplug] Step 0: go home")
        _best_effort_go_home("script start")

    print("[gpu_unplug] Step 1: open grippers")
    _best_effort_open_grippers()

    print(f"[gpu_unplug] Step 2: build motherboard-relative slot-{int(TARGET_SOCKET_NUMBER)} reference")
    reference_scene = hover._build_initial_reference_scene(camera=hover.CAMERA)
    reference_scene = _apply_left_x_refinement(
        reference_scene,
        reference_scene,
        stage_label="initial_reference",
        artifact_prefix="gpu_unplug_initial_reference_left_x",
    )
    reference_scene = _apply_target_slot_hover_z_trim(
        reference_scene,
        stage_label="initial_reference",
    )
    hover_rpy = hover._resolve_hover_rpy()
    print(
        f"[gpu_unplug] Slot-{int(TARGET_SOCKET_NUMBER)} hover target: "
        f"hover_world={[round(float(v), 4) for v in reference_scene['socket_hover_world'].tolist()]} "
        f"top_z={float(reference_scene['motherboard_top_z']):.4f} "
        f"hover_z={float(reference_scene['hover_z']):.4f} "
        f"rpy={[round(float(v), 1) for v in hover_rpy]}"
    )

    print("[gpu_unplug] Step 3: acquire initial hover")
    initial_clearance_scene = _with_transit_hover_clearance(
        reference_scene,
        stage_label="initial_hover_acquire",
    )
    if initial_clearance_scene is not reference_scene:
        hover._move_left_to_hover(
            initial_clearance_scene,
            hover_rpy,
            fixed_plane_z=float(initial_clearance_scene["hover_z"]),
            guided=True,
            guided_duration_s=float(hover.REACTIVE_GUIDED_DURATION_S),
            guided_steps=int(hover.REACTIVE_GUIDED_STEPS),
        )
    hover_pos, current_scene = hover._acquire_initial_hover(
        reference_scene=reference_scene,
        hover_rpy=hover_rpy,
        camera=hover.CAMERA,
    )
    current_scene = _apply_target_slot_hover_z_trim(
        current_scene,
        stage_label="initial_hover",
    )
    print(
        "[gpu_unplug] Initial hover acquired: "
        f"pos={[round(float(v), 4) for v in hover_pos]}"
    )
    _save_hover_visualization(current_scene, "gpu_unplug_initial_hover")

    print("[gpu_unplug] Step 4: optional pre-unplug hover refresh")
    current_scene = _refresh_hover_scene(reference_scene, current_scene)

    bar_reference_scene = None
    bar_current_scene = None
    if BAR_STAGE_ENABLED:
        print("[gpu_unplug] Step 5: derive bar-stage hover scene from slot geometry")
        try:
            bar_reference_scene, bar_current_scene = _acquire_bar_hover_scene(
                reference_scene,
                current_scene,
                hover_rpy,
            )
        except Exception as exc:
            print(f"[gpu_unplug] bar-stage hover acquisition failed, falling back to center pull: {exc}")

    bar_stage_completed = False
    if bar_current_scene is not None:
        print("[gpu_unplug] Step 6: optional bar hover refinement")
        bar_current_scene = _refine_bar_hover_scene(
            bar_reference_scene,
            bar_current_scene,
            hover_rpy,
        )

        print("[gpu_unplug] Step 7: descend on metal bar and loosen GPU")
        _set_left_gripper_position(
            LEFT_BAR_PREOPEN_POS,
            duration_s=float(LEFT_PREOPEN_DURATION_S),
            label="bar-stage pre-open",
        )
        if float(PRE_DESCEND_SETTLE_S) > 0.0:
            time.sleep(float(PRE_DESCEND_SETTLE_S))
        bar_grasp_pos = _build_grasp_pos_for_world(
            bar_current_scene,
            np.asarray(bar_current_scene["socket_hover_world"], dtype=np.float64).reshape(3).tolist(),
            descend_from_hover_m=float(BAR_DESCEND_FROM_HOVER_M),
            min_grasp_z_above_board_m=float(BAR_MIN_GRASP_Z_ABOVE_BOARD_M),
            max_grasp_z_below_hover_m=float(BAR_MAX_GRASP_Z_BELOW_HOVER_M),
        )
        _guided_left_move(
            bar_grasp_pos,
            hover_rpy,
            duration_s=float(DESCEND_DURATION_S),
            num_steps=int(DESCEND_STEPS),
            label="bar-stage descend",
        )
        _firm_close_left_gripper(
            reclamp_attempts=BAR_STAGE_RECLAMP_ATTEMPTS,
            label_prefix="close left gripper on metal bar",
        )
        _verify_left_hold_or_raise("bar-stage hold check")

        loosen_lift_mid_pos, loosen_lift_pos = _build_bar_arc_lift_waypoints(
            bar_current_scene,
            bar_grasp_pos,
        )
        _guided_left_move(
            loosen_lift_mid_pos,
            hover_rpy,
            duration_s=float(max(0.35, 0.30 * LIFT_DURATION_S)),
            num_steps=int(max(4, LIFT_STEPS // 3)),
            label="bar-stage arc lift 1",
        )
        _guided_left_move(
            loosen_lift_pos,
            hover_rpy,
            duration_s=float(max(0.45, 0.40 * LIFT_DURATION_S)),
            num_steps=int(max(5, LIFT_STEPS // 3)),
            label="bar-stage arc lift 2",
        )
        _verify_left_hold_or_raise("bar-stage post-loosen hold check")
        bar_stage_completed = True

        if BAR_STAGE_REGRASP_ENABLED:
            print("[gpu_unplug] Step 8: release bar grasp and re-hover to center")
            bar_release_pos = _build_bar_release_support_pos(bar_grasp_pos)
            _guided_left_move(
                bar_release_pos,
                hover_rpy,
                duration_s=float(max(0.5, 0.55 * DESCEND_DURATION_S)),
                num_steps=int(max(6, DESCEND_STEPS // 2)),
                label="bar-stage support return",
            )
            _set_left_gripper_position(
                LEFT_GRASP_PREOPEN_POS,
                duration_s=float(LEFT_PREOPEN_DURATION_S),
                label="bar-stage release pre-open",
            )
            if float(POST_LOOSEN_RELEASE_SETTLE_S) > 0.0:
                time.sleep(float(POST_LOOSEN_RELEASE_SETTLE_S))
            hover._move_left_to_hover(
                current_scene,
                hover_rpy,
                fixed_plane_z=float(current_scene["hover_z"]),
                guided=True,
                guided_duration_s=float(hover.REACTIVE_GUIDED_DURATION_S),
                guided_steps=int(hover.REACTIVE_GUIDED_STEPS),
            )

    if bar_stage_completed and not BAR_STAGE_REGRASP_ENABLED:
        print("[gpu_unplug] Success: GPU loosened and lifted using bar-stage grasp.")
        print("[gpu_unplug] Leaving the left arm holding the lifted GPU.")
        return

    print("[gpu_unplug] Step 9: descend to center grasp")
    _set_left_gripper_position(
        LEFT_GRASP_PREOPEN_POS,
        duration_s=float(LEFT_PREOPEN_DURATION_S),
        label="center-grasp pre-open",
    )
    if float(PRE_DESCEND_SETTLE_S) > 0.0:
        time.sleep(float(PRE_DESCEND_SETTLE_S))
    grasp_pos = _build_grasp_pos(current_scene)
    _guided_left_move(
        grasp_pos,
        hover_rpy,
        duration_s=float(DESCEND_DURATION_S),
        num_steps=int(DESCEND_STEPS),
        label="center descend",
    )

    print("[gpu_unplug] Step 10: close left gripper")
    _firm_close_left_gripper()
    _verify_left_hold_or_raise("center post-close hold check")

    print("[gpu_unplug] Step 11: pull GPU off the motherboard completely")
    lift_pos = _build_lift_pos(current_scene, grasp_pos)
    _guided_left_move(
        lift_pos,
        hover_rpy,
        duration_s=float(LIFT_DURATION_S),
        num_steps=int(LIFT_STEPS),
        label="center full lift",
    )
    if float(POST_LIFT_SETTLE_S) > 0.0:
        time.sleep(float(POST_LIFT_SETTLE_S))
    _verify_left_hold_or_raise("center post-lift hold check")

    print("[gpu_unplug] Step 11b: return left-held GPU to hover pose before handover")
    hover_return_pos = [
        float(current_scene["socket_hover_world"][0]),
        float(current_scene["socket_hover_world"][1]),
        float(current_scene["hover_z"]),
    ]
    _guided_left_move(
        hover_return_pos,
        hover_rpy,
        duration_s=float(max(0.6, 0.65 * LIFT_DURATION_S)),
        num_steps=int(max(6, LIFT_STEPS // 2)),
        label="post-unplug hover return",
    )
    _verify_left_hold_or_raise("post-unplug hover-return hold check")

    print(
        "[gpu_unplug] Success: GPU should now be unplugged and returned to hover. "
        f"hover_pos={[round(float(v), 4) for v in hover_return_pos]}"
    )
    _place_left_held_gpu_on_table()

    print("[gpu_unplug] Step 15: go home after left-arm setdown")
    if not _best_effort_go_home("post-setdown cleanup"):
        raise RuntimeError("go_home failed after left-arm setdown")
    SCRIPT_COMPLETED = True


def get_task_info() -> dict:
    return {
        "success": bool(SCRIPT_COMPLETED),
        "reward": 1.0 if SCRIPT_COMPLETED else 0.0,
        "target_socket": int(TARGET_SOCKET_NUMBER),
        "method": "script_completion",
        "bar_stage_enabled": bool(BAR_STAGE_ENABLED),
        "bar_stage_regrasp_enabled": bool(BAR_STAGE_REGRASP_ENABLED),
    }


main()

