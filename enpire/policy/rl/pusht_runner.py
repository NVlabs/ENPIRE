# ruff: noqa: E402  (imports must follow the sys.path / bootstrap block)
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

from enpire.env.forge.paths import FORGE_ROOT, REPOSITORY_ROOT

REPO_ROOT = FORGE_ROOT
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from enpire.env.forge.tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPOSITORY_ROOT, required_modules=["gymnasium", "tyro"])

sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np
from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.rl_interface import PolicyAdapters
import requests
import tyro
import yaml

from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.context import RLContext, build_context
from enpire.policy.rl.events import TERMINAL_EVENTS
from enpire.policy.rl.author import do_author, enter_author, exit_author
from enpire.policy.rl.handlers import (
    do_change_pose,
    do_home,
    do_parking,
    handle_restart,
)
from enpire.policy.rl.policy import PolicyRouter
from enpire.policy.rl.reset_options import terminal_label_options
from enpire.env.forge.robot.yam.kinematics import (
    YamKinematics,
    _quat_xyzw_to_rot6d,
    _rot6d_to_rot_matrix,
    _rpy_display_to_quat_xyzw,
)
from enpire.policy.rl.pusht.reward import (
    SUCCESS_RECT_KEY,
    _as_bgr,
    _crop,
    _dominant_component,
    _draw_reward_overlay,
    _load_meta,
    _red_mask,
    _rect_full_to_crop,
    _scale_rect,
    _score_success,
    gate_score_by_avoid,
)
from enpire.policy.rl.pusht import gripper_overlay as pusht_gripper_overlay
from enpire.env.forge.display_utils import put_latest_image


PUSHT_EXIT_RESET_REQUESTED = 10
DELTA_EE_CONTROL_MODES = ("delta_ee_pose", "delta_ee_pose_translation")
IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
PUSHT_SPECIAL_ACTION_IDS = {
    "avoid": 1,
    "back": 2,
}
PUSHT_SPECIAL_ACTION_BY_ID = {
    value: key for key, value in PUSHT_SPECIAL_ACTION_IDS.items()
}
PUSHT_SPECIAL_ACTION_KEY = "__pusht_special_action_id"


@dataclass
class PushTConfig(DataCollectionConfig):
    enabled_camera_names: tuple[str, ...] = ("top", "left", "right")
    pusht_start_on_launch: bool = False
    pusht_keyboard_shortcuts_enabled: bool = True
    pusht_lock_rotation: bool = True
    pusht_locked_rpy_deg: tuple[float, float, float] = (0.0, -180.0, 0.0)
    pusht_lock_gripper: bool = True
    pusht_locked_gripper_pos: float = 0.0
    pusht_lock_z: bool = True
    pusht_locked_z: float = 0.8069
    pusht_bbox_x_lim: tuple[float, float] = (0.3085, 0.78)
    pusht_bbox_y_lim: tuple[float, float] = (-0.3, 0.3)
    pusht_bbox_z_lim: tuple[float, float] = (0.8069, 0.8469)
    pusht_bbox_release_orientation: bool = True
    pusht_bbox_orientation_cost: float = 0.1
    pusht_bbox_point_down_after_position_only: bool = True
    pusht_delta_action_scale_m: float = 0.01
    pusht_teleop_position_only_planning: bool = True
    pusht_teleop_orientation_cost: float = 0.1
    pusht_teleop_point_down_after_position_only: bool = False
    pusht_teleop_point_down_orientation_cost: float = 0.02
    pusht_teleop_point_down_max_pos_err_m: float = 0.01
    pusht_teleop_replan_deadband_m: float = 0.001
    pusht_avoid_hover_z: float = 0.93
    pusht_avoid_hover_clearance_m: float = 0.08
    pusht_avoid_home_reset_max_joint_velocity: float = 4.8
    pusht_back_hover_z: float = 0.93
    pusht_back_hover_clearance_m: float = 0.08
    pusht_back_hover_reset_max_joint_velocity: float = 4.8
    pusht_bbox_yaw_candidates_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 180.0)
    pusht_reward_goal_image: str = ""
    pusht_match_reward_camera_image_shape: bool = True
    pusht_camera_image_shape: tuple[int, int] = (1280, 720)
    pusht_viewer_camera_tile_height: int = 720
    pusht_viewer_reward_tile_height: int = 180
    pusht_auto_reset_on_reward: bool = True
    pusht_auto_reset_threshold: float = 0.5
    pusht_auto_reset_exit_after: bool = True
    pusht_auto_reset_script_file: str = "cap/saved_scripts/place_grasped_t_reset.py"
    pusht_auto_reset_skill_library_path: str = "cap/saved_scripts/pusht"
    pusht_auto_reset_ok_exit_codes: tuple[int, ...] = (0, 139)
    pusht_auto_reset_timeout_s: float = 240.0
    pusht_auto_reset_cleanup_stale_processes: bool = True
    pusht_auto_reset_stale_process_age_s: float = 120.0
    pusht_reset_request_path: str | None = None
    pusht_overlay_calibration_enabled: bool = True
    pusht_overlay_calibration_host: str = "127.0.0.1"
    pusht_overlay_calibration_port: int = 8205
    pusht_overlay_table_correction_enabled: bool = True
    pusht_overlay_table_grid_size: int = 100
    pusht_overlay_table_min_samples: int = 3
    pusht_overlay_table_idw_neighbors: int = 12
    pusht_overlay_table_idw_power: float = 2.0
    pusht_overlay_sample_file_dir: str = "cap/tasks/pusht/overlay_tables"
    pusht_overlay_station_name: str | None = None
    pusht_overlay_manual_anchor_radius_m: float = 0.015
    pusht_gripper_dot_observation_key: str = "top_gripper_dot_camera_image"


class PushTPolicyRouter(PolicyRouter):
    """PushT-specific router: left-arm PushT constraints for Cartesian/delta EE."""

    def __init__(self, *args, cfg: PushTConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._enabled_side_set = set(self.enabled_sides)
        self._locked_rot6d = _quat_xyzw_to_rot6d(
            _rpy_display_to_quat_xyzw(cfg.pusht_locked_rpy_deg)
        )

    def route_action(
        self, obs, fello_action, fello_info, *, use_rl: bool, keyboard_info
    ):
        try:
            action, elapsed_ms = super().route_action(
                obs,
                fello_action,
                fello_info,
                use_rl=use_rl,
                keyboard_info=keyboard_info,
            )
        except Exception as exc:
            print(
                f"[PushT] policy action failed; holding current pose: {exc}",
                flush=True,
            )
            action = self._hold_cartesian_action(obs)
            elapsed_ms = 0
        return self._constrain_cartesian_action(obs, action), elapsed_ms

    def _hold_cartesian_action(self, obs: dict) -> dict:
        action = {"source": "hold"}
        for side in ("left", "right"):
            if self.cfg.control_mode in DELTA_EE_CONTROL_MODES:
                action[f"{side}_ee_pos"] = np.zeros(3, dtype=np.float32)
                action[f"{side}_ee_rot6d"] = IDENTITY_ROT6D.copy()
            else:
                action[f"{side}_ee_pos"] = np.asarray(
                    obs[f"{side}_ee_pos"], dtype=np.float32
                ).reshape(3)
                action[f"{side}_ee_rot6d"] = np.asarray(
                    obs[f"{side}_ee_rot6d"], dtype=np.float32
                ).reshape(6)
            action[f"{side}_gripper_pos"] = np.asarray(
                obs.get(f"{side}_gripper_pos", np.zeros(1, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(1)
        return action

    def _constrain_cartesian_action(self, obs: dict, action: dict) -> dict:
        is_delta_ee = self.cfg.control_mode in DELTA_EE_CONTROL_MODES
        if self.cfg.control_mode != "cartesian_position" and not is_delta_ee:
            return action

        constrained = dict(action)
        source = constrained.pop("source", None)
        for side in ("left", "right"):
            rot6d_key = f"{side}_ee_rot6d"
            if side not in self._enabled_side_set:
                for suffix in ("ee_pos", "ee_rot6d", "ee_quat_xyzw", "gripper_pos"):
                    constrained.pop(f"{side}_{suffix}", None)
                if is_delta_ee:
                    constrained[f"{side}_ee_pos"] = np.zeros(3, dtype=np.float32)
                    constrained[rot6d_key] = IDENTITY_ROT6D.copy()
                else:
                    constrained[f"{side}_ee_pos"] = np.asarray(
                        obs[f"{side}_ee_pos"], dtype=np.float32
                    ).reshape(3)
                    constrained[rot6d_key] = np.asarray(
                        obs[f"{side}_ee_rot6d"], dtype=np.float32
                    ).reshape(6)
                constrained[f"{side}_gripper_pos"] = np.asarray(
                    obs.get(f"{side}_gripper_pos", np.zeros(1, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(1)
                continue

            quat_key = f"{side}_ee_quat_xyzw"
            if quat_key in constrained and rot6d_key not in constrained:
                constrained[rot6d_key] = _quat_xyzw_to_rot6d(constrained[quat_key])
            constrained.pop(quat_key, None)
            if (
                is_delta_ee
                and f"{side}_ee_pos" in constrained
                and rot6d_key not in constrained
            ):
                constrained[rot6d_key] = IDENTITY_ROT6D.copy()

            lock_rotation = self.cfg.pusht_lock_rotation and not (
                side == "left" and self.cfg.pusht_teleop_position_only_planning
            )
            if lock_rotation and f"{side}_ee_pos" in constrained and not is_delta_ee:
                constrained[rot6d_key] = self._locked_rot6d.copy()
            if (
                self.cfg.pusht_lock_z
                and f"{side}_ee_pos" in constrained
                and not is_delta_ee
            ):
                pos = np.asarray(
                    constrained[f"{side}_ee_pos"], dtype=np.float32
                ).reshape(3)
                pos[2] = float(self.cfg.pusht_locked_z)
                constrained[f"{side}_ee_pos"] = pos
            if side == "left" and f"{side}_ee_pos" in constrained:
                pos = np.asarray(
                    constrained[f"{side}_ee_pos"], dtype=np.float32
                ).reshape(3)
                x_lo, x_hi = self.cfg.pusht_bbox_x_lim
                y_lo, y_hi = self.cfg.pusht_bbox_y_lim
                before = pos.copy()
                if is_delta_ee:
                    pos[2] = 0.0
                    action_units = np.clip(pos, -1.0, 1.0).astype(np.float32)
                    if not np.allclose(action_units, pos):
                        print(
                            f"[PushT] clipped left delta action units "
                            f"{np.round(pos, 4)} -> {np.round(action_units, 4)}",
                            flush=True,
                        )
                    pos = action_units * float(self.cfg.pusht_delta_action_scale_m)
                    current = np.asarray(
                        obs[f"{side}_ee_pos"], dtype=np.float32
                    ).reshape(3)
                    target = current + pos
                    clipped_target = target.copy()
                    clipped_target[0] = float(np.clip(clipped_target[0], x_lo, x_hi))
                    clipped_target[1] = float(np.clip(clipped_target[1], y_lo, y_hi))
                    if (
                        self.cfg.pusht_lock_z
                        or self.cfg.control_mode == "delta_ee_pose_translation"
                    ):
                        clipped_target[2] = current[2]
                    pos = (clipped_target - current).astype(np.float32)
                    constrained[f"{side}_ee_pos"] = pos
                    clipped_for_log = target
                    clipped_to_log = clipped_target
                else:
                    pos[0] = float(np.clip(pos[0], x_lo, x_hi))
                    pos[1] = float(np.clip(pos[1], y_lo, y_hi))
                    if self.cfg.pusht_lock_z:
                        pos[2] = float(self.cfg.pusht_locked_z)
                    else:
                        pos[2] = float(
                            np.asarray(
                                obs[f"{side}_ee_pos"], dtype=np.float32
                            ).reshape(3)[2]
                        )
                    constrained[f"{side}_ee_pos"] = pos
                    clipped_for_log = before
                    clipped_to_log = pos
                if not np.allclose(clipped_for_log, clipped_to_log):
                    print(
                        f"[PushT] clipped left_xyz {np.round(clipped_for_log, 4)} -> "
                        f"{np.round(clipped_to_log, 4)}",
                        flush=True,
                    )
            if self.cfg.pusht_lock_gripper:
                constrained[f"{side}_gripper_pos"] = np.asarray(
                    [self.cfg.pusht_locked_gripper_pos], dtype=np.float32
                )

        if source is not None:
            constrained["source"] = source
        return constrained


def pusht_map_observation(obs: dict):
    from PIL import Image

    images = {}
    for key in sorted(k for k in obs if k.endswith("_camera_image")):
        images[key] = Image.fromarray(np.asarray(obs[key], dtype=np.uint8))
    proprio = {
        "left_joint_pos": np.asarray(
            obs.get("left_joint_pos", np.zeros(6)), dtype=np.float32
        ).reshape(6),
        "left_target_eef": np.asarray(
            obs.get("left_ee_pos", np.zeros(3)), dtype=np.float32
        ).reshape(3),
        "state_eef_rot6d": np.concatenate(
            [
                np.asarray(
                    obs.get("left_ee_pos", np.zeros(3)), dtype=np.float32
                ).reshape(3),
                np.asarray(
                    obs.get("left_ee_rot6d", np.zeros(6)), dtype=np.float32
                ).reshape(6),
                np.asarray(
                    obs.get("left_gripper_pos", np.zeros(1)), dtype=np.float32
                ).reshape(1),
                np.asarray(
                    obs.get("right_ee_pos", np.zeros(3)), dtype=np.float32
                ).reshape(3),
                np.asarray(
                    obs.get("right_ee_rot6d", np.zeros(6)), dtype=np.float32
                ).reshape(6),
                np.asarray(
                    obs.get("right_gripper_pos", np.zeros(1)), dtype=np.float32
                ).reshape(1),
            ]
        ),
    }
    return images, proprio


def pusht_map_action(action: dict):
    mode = action.get("mode")
    if mode is not None:
        mode = str(mode)
        value = action.get("action")
        if mode == "control":
            pos = np.asarray(value, dtype=np.float32).reshape(-1)
            if pos.size == 2:
                pos = np.asarray([pos[0], pos[1], 0.0], dtype=np.float32)
            elif pos.size == 3:
                pos = pos.astype(np.float32)
                pos[2] = 0.0
            else:
                raise ValueError(
                    f"PushT control action must have 2 or 3 values, got {pos.size}"
                )
            return {"left_ee_pos": pos.reshape(3)}
        if mode == "special":
            special = str(value)
            if special not in PUSHT_SPECIAL_ACTION_IDS:
                raise ValueError(f"unknown PushT special action: {special!r}")
            return {
                PUSHT_SPECIAL_ACTION_KEY: np.asarray(
                    [PUSHT_SPECIAL_ACTION_IDS[special]], dtype=np.float32
                )
            }
        raise ValueError(f"unknown PushT action mode: {mode!r}")

    mapped = {}
    key_map = {
        "ee_pos_action_left": "left_ee_pos",
        "ee_quat_action_left": "left_ee_rot6d",
        "gripper_pos_action_left": "left_gripper_pos",
        "ee_pos_action_right": "right_ee_pos",
        "ee_quat_action_right": "right_ee_rot6d",
        "gripper_pos_action_right": "right_gripper_pos",
    }
    env_keys = {
        "left_ee_pos",
        "left_ee_rot6d",
        "left_ee_quat_xyzw",
        "left_gripper_pos",
        "right_ee_pos",
        "right_ee_rot6d",
        "right_ee_quat_xyzw",
        "right_gripper_pos",
    }
    for key, value in action.items():
        if key in env_keys:
            mapped[key] = value
        elif key in key_map:
            if key.startswith("ee_quat_action_"):
                mapped[key_map[key]] = _quat_xyzw_to_rot6d(value)
            else:
                mapped[key_map[key]] = value
    return mapped if mapped else action


def build_pusht_context(cfg: PushTConfig) -> RLContext:
    if cfg.pusht_match_reward_camera_image_shape:
        _patch_pusht_camera_image_shape(cfg.pusht_camera_image_shape)
    print(
        "[PushT] active bbox "
        f"x={cfg.pusht_bbox_x_lim} y={cfg.pusht_bbox_y_lim} "
        f"z_action={cfg.pusht_locked_z} z_r_vertices={cfg.pusht_bbox_z_lim}",
        flush=True,
    )
    ctx = build_context(cfg)
    command_sides = (
        {"left", "right"} if cfg.enabled_sides == "both" else {cfg.enabled_sides}
    )
    env = ctx.env.unwrapped
    if hasattr(env, "set_command_enabled_sides"):
        env.set_command_enabled_sides(command_sides)
    if hasattr(env, "set_delta_ee_translation_mask"):
        env.set_delta_ee_translation_mask(enabled_sides=command_sides)
    ctx.policy_router.rl_policy.adapters = PolicyAdapters(
        map_observation=pusht_map_observation,
        map_action=pusht_map_action,
    )
    ctx.policy_router = PushTPolicyRouter(
        fello_policy=ctx.policy_router.fello_policy,
        spacemouse_policy=ctx.policy_router.spacemouse_policy,
        rl_policy=ctx.policy_router.rl_policy,
        enabled_sides=cfg.enabled_sides,
        z_up_step_m=cfg.delta_ee_translation_xyz_max[0],
        demo_collection=cfg.demo_collection,
        cfg=cfg,
    )
    start_pusht_overlay_calibration_server(ctx)
    return ctx


def do_pusht_hover(ctx: RLContext) -> None:
    opts = {
        "target_ee_pose": ctx.initial_pose_manager.build_episode_start_pose(),
        "task_name": ctx.cfg.task_name,
        "start_new_episode": True,
    }
    opts.update(terminal_label_options(ctx.terminal_event))
    print(
        f"[INFO] Episode start from {ctx.initial_pose_manager.describe_current()} "
        f"offset={np.round(ctx.initial_pose_manager.last_offset, 4)}",
        flush=True,
    )
    ctx.obs, _ = ctx.env.reset(options=opts)
    update_pusht_gripper_dot_observation(ctx)
    reset_info = ctx.policy_router.rl_policy.reset(ctx.obs)
    print(f"[PushT] policy reset with initial observation: {reset_info}", flush=True)
    ctx.event_router.reset_timer()
    ctx.terminal_event = None
    ctx.speech_announcer.speak("learn")
    ctx.state_machine.state = "learn"


def _patch_pusht_camera_image_shape(shape_wh: tuple[int, int]) -> None:
    width, height = [int(v) for v in shape_wh]
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid pusht_camera_image_shape={shape_wh}")

    from enpire.policy.rl import record_episode_wrapper
    import enpire.env.forge.robot.constants as robot_constants
    import enpire.env.forge.robot.yam._base_yam_env as base_yam_env
    import enpire.env.forge.robot.yam.yam_real_env as yam_real_env

    for camera_name in ("top", "left", "right"):
        os.environ[f"CAP_{camera_name.upper()}_CAMERA_RESOLUTION"] = f"{width}x{height}"

    def _resize_for_pusht_reward(image):
        arr = np.asarray(image)
        if arr.shape[:2] == (height, width):
            return arr
        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)

    def _blank_for_pusht_reward():
        return np.zeros((height, width, 3), dtype=np.uint8)

    robot_constants.DEFAULT_COMPRESSED_VIDEO_SHAPE = (width, height)
    base_yam_env.DEFAULT_COMPRESSED_VIDEO_SHAPE = (width, height)
    yam_real_env.compress_image = _resize_for_pusht_reward
    yam_real_env.blank_compressed_image = _blank_for_pusht_reward
    record_episode_wrapper.DEFAULT_COMPRESSED_VIDEO_SHAPE = (width, height)
    print(
        f"[PushT] camera observations resized to {width}x{height} "
        "and camera capture resolution forced for top/left/right",
        flush=True,
    )


def dispatch_pusht_keyboard_events(ctx: RLContext, keyboard_info: dict) -> None:
    just = keyboard_info.get("_just_pressed_keys", set())
    shift = keyboard_info.get("shift_held", False)
    supervisor_url = os.environ.get("PUSHT_SUPERVISOR_URL", "http://127.0.0.1:8203")
    if "KEY_H" in just:
        _fire_async_http(f"{supervisor_url}/home")
    if "KEY_S" in just:
        _fire_async_http(f"{supervisor_url}/start")
    if "KEY_P" in just:
        ctx.external_event_queue.put(("parking", {}))
    if "KEY_A" in just:
        ctx.external_event_queue.put(("author", {}))
    if "KEY_ESC" in just:
        ctx.external_event_queue.put(("discard_author", {}))
    if "KEY_R" in just:
        reset_to_next_pusht_bbox_vertex(ctx)
    if "KEY_O" in just:
        ctx.external_event_queue.put(("oor_boundary", {}))
    if "KEY_PAGEUP" in just:
        ctx.external_event_queue.put(("z_high", {}))
    if "KEY_PAGEDOWN" in just:
        ctx.external_event_queue.put(("z_low", {}))
    if ctx.cfg.restart_key in just:
        _fire_async_http(f"{supervisor_url}/restart")
    if shift and "KEY_COMMA" in just:
        ctx.external_event_queue.put(("prev_pose", {}))
    if shift and "KEY_DOT" in just:
        ctx.external_event_queue.put(("next_pose", {}))


def reset_to_next_pusht_bbox_vertex(ctx: RLContext) -> None:
    idx = int(getattr(ctx, "pusht_bbox_vertex_idx", 0))
    z_idx = 0 if idx < 4 else 1
    corner_idx = idx % 4
    x_lo, x_hi = ctx.cfg.pusht_bbox_x_lim
    y_lo, y_hi = ctx.cfg.pusht_bbox_y_lim
    z_lo, z_hi = ctx.cfg.pusht_bbox_z_lim
    xy_corners = (
        (x_hi, y_hi),  # left upper
        (x_lo, y_hi),
        (x_lo, y_lo),  # right lower
        (x_hi, y_lo),
    )
    x, y = xy_corners[corner_idx]
    z = z_lo if z_idx == 0 else z_hi
    left_xyz = np.array([float(x), float(y), float(z)], dtype=np.float64)
    print(
        f"[PushT] OOR bbox vertex {idx + 1}/8 "
        f"corner={corner_idx + 1}/4 z={'low' if z_idx == 0 else 'high'} "
        f"left_xyz={np.round(left_xyz, 4)}",
        flush=True,
    )
    ctx.policy_router.rl_policy.reset()
    if ctx.cfg.pusht_bbox_release_orientation:
        _reset_to_pusht_bbox_position_only(ctx, left_xyz, idx)
        setattr(ctx, "pusht_bbox_vertex_idx", (idx + 1) % 8)
        return

    last_exc: Exception | None = None
    try:
        for yaw in ctx.cfg.pusht_bbox_yaw_candidates_deg:
            pose = ctx.initial_pose_manager.build_center_pose()
            pose["left"]["position"] = left_xyz.tolist()
            pose["left"]["rpy_deg"] = [0.0, -180.0, float(yaw)]
            print(
                f"[PushT]   trying left_rpy={np.round(pose['left']['rpy_deg'], 2)}",
                flush=True,
            )
            try:
                ctx.obs, _ = ctx.env.reset(
                    options={
                        "target_ee_pose": pose,
                        "discard_episode": True,
                    }
                )
                ctx.terminal_event = None
                ctx.state_machine.state = "idle"
                print(
                    f"[PushT] bbox vertex {idx + 1}/8 reached "
                    f"left_xyz={np.round(left_xyz, 4)} yaw={yaw:.1f}",
                    flush=True,
                )
                return
            except Exception as exc:
                last_exc = exc
                print(
                    f"[PushT]   yaw={yaw:.1f} failed: {exc}",
                    flush=True,
                )
        print(
            f"[PushT] bbox vertex {idx + 1}/8 reset failed for all yaw candidates; "
            f"last_error={last_exc}",
            flush=True,
        )
    finally:
        setattr(ctx, "pusht_bbox_vertex_idx", (idx + 1) % 8)


def _reset_to_pusht_bbox_position_only(
    ctx: RLContext,
    left_xyz: np.ndarray,
    idx: int,
) -> None:
    env = ctx.env.unwrapped
    if not hasattr(ctx, "pusht_bbox_kinematics"):
        setattr(
            ctx,
            "pusht_bbox_kinematics",
            YamKinematics(
                position_cost=1.0,
                orientation_cost=float(ctx.cfg.pusht_bbox_orientation_cost),
            ),
        )
    kinematics = getattr(ctx, "pusht_bbox_kinematics")
    current = env._read_current_joint_state()
    nominal_quat = _rpy_display_to_quat_xyzw(ctx.cfg.pusht_locked_rpy_deg)
    left_joint_pos, _, _, _ = kinematics.inverse_kinematics_full(
        left_xyz,
        nominal_quat,
        None,
        None,
        left_seed=np.asarray(current["left_joint_pos"], dtype=np.float32).reshape(6),
        right_seed=np.asarray(current["right_joint_pos"], dtype=np.float32).reshape(6),
        max_iters=40,
    )
    if ctx.cfg.pusht_bbox_point_down_after_position_only:
        left_joint_pos, _, orient_err, _ = (
            kinematics.inverse_kinematics_orientation_only(
                np.asarray(left_joint_pos, dtype=np.float32).reshape(6),
                nominal_quat,
                np.asarray(current["right_joint_pos"], dtype=np.float32).reshape(6),
                None,
                max_iters=40,
            )
        )
    else:
        orient_err = float("nan")
    target_state = {
        "left_joint_pos": np.asarray(left_joint_pos, dtype=np.float32).reshape(6),
        "left_gripper_pos": np.asarray(
            current.get("left_gripper_pos", np.zeros(1, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1),
        "right_joint_pos": np.asarray(
            current["right_joint_pos"], dtype=np.float32
        ).reshape(6),
        "right_gripper_pos": np.asarray(
            current.get("right_gripper_pos", np.zeros(1, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1),
    }
    fk_left, _, _, _ = kinematics.forward_kinematics(
        target_state["left_joint_pos"], target_state["right_joint_pos"]
    )
    pos_err = float(np.linalg.norm(np.asarray(fk_left, dtype=np.float64) - left_xyz))
    print(
        f"[PushT] bbox vertex {idx + 1}/8 position-only IK "
        f"target={np.round(left_xyz, 4)} fk={np.round(fk_left, 4)} "
        f"pos_err={pos_err:.4f}m point_down_err={orient_err:.4f}",
        flush=True,
    )
    try:
        ctx.obs, _ = ctx.env.reset(
            options={
                "target_joint_position": target_state,
                "discard_episode": True,
            }
        )
        ctx.terminal_event = None
        ctx.state_machine.state = "idle"
    except Exception as exc:
        print(
            f"[PushT] bbox vertex {idx + 1}/8 position-only reset failed: {exc}",
            flush=True,
        )


def _copy_joint_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in state.items()
        if key.endswith("_joint_pos") or key.endswith("_gripper_pos")
    }


def _reset_robot_without_episode_finalize(
    ctx: RLContext,
    options: dict,
    *,
    reset_max_joint_velocity: float | None = None,
) -> None:
    env = ctx.env.unwrapped
    previous_velocity = None
    if reset_max_joint_velocity is not None and hasattr(
        env, "reset_max_joint_velocity"
    ):
        previous_velocity = float(env.reset_max_joint_velocity)
        env.reset_max_joint_velocity = float(reset_max_joint_velocity)
    try:
        obs, info = env.reset(options=options)
    finally:
        if previous_velocity is not None:
            env.reset_max_joint_velocity = previous_velocity
    ctx.obs = obs
    update_pusht_gripper_dot_observation(ctx)
    if hasattr(ctx.env, "prev_obs"):
        ctx.env.prev_obs = obs
    if hasattr(ctx.env, "_prev_obs_timestamps"):
        ctx.env._prev_obs_timestamps = info.get("__timestamps")


def handle_pusht_avoid(ctx: RLContext, *, keep_learning: bool = True) -> None:
    if getattr(ctx, "pusht_pre_avoid_joint_state", None) is not None:
        print(
            "[PushT] special avoid ignored: already avoided; "
            "policy must send special action='back' before another avoid",
            flush=True,
        )
        return
    pre_avoid_sample = _save_pusht_overlay_sample(ctx, label="pre_avoid")
    if pre_avoid_sample.get("success"):
        _freeze_pusht_pre_avoid_overlay(ctx, pre_avoid_sample["sample"])
        print(
            f"[PushT] special avoid: saved pre-avoid green-dot sample "
            f"{pre_avoid_sample['sample']['id']}",
            flush=True,
        )
    else:
        print(
            f"[PushT] special avoid: pre-avoid green-dot save skipped: "
            f"{pre_avoid_sample.get('error', 'unknown error')}",
            flush=True,
        )
    env = ctx.env.unwrapped
    if not hasattr(env, "_read_current_joint_state"):
        print(
            "[PushT] special avoid ignored: env cannot read current joint state",
            flush=True,
        )
        return
    state = _copy_joint_state(env._read_current_joint_state())
    setattr(ctx, "pusht_pre_avoid_joint_state", state)
    hover_state = _build_pusht_hover_state_from_joint_state(
        ctx,
        state,
        hover_z=float(ctx.cfg.pusht_avoid_hover_z),
        clearance_m=float(ctx.cfg.pusht_avoid_hover_clearance_m),
        label="avoid",
    )
    print(
        "[PushT] special avoid: saved current joint state, lifting, then going home",
        flush=True,
    )
    _reset_robot_without_episode_finalize(
        ctx,
        {"target_joint_position": hover_state},
    )
    _reset_robot_without_episode_finalize(
        ctx,
        {"alias": "home"},
        reset_max_joint_velocity=float(
            ctx.cfg.pusht_avoid_home_reset_max_joint_velocity
        ),
    )
    ctx.terminal_event = None
    if not keep_learning:
        ctx.state_machine.state = "idle"


def handle_pusht_back(ctx: RLContext, *, keep_learning: bool = True) -> None:
    state = getattr(ctx, "pusht_pre_avoid_joint_state", None)
    if not state:
        _clear_pusht_pre_avoid_overlay(ctx)
        print("[PushT] special back ignored: no saved pre-avoid pose", flush=True)
        return
    setattr(ctx, "pusht_pre_avoid_joint_state", None)
    hover_state = _build_pusht_hover_state_from_joint_state(
        ctx,
        state,
        hover_z=float(ctx.cfg.pusht_back_hover_z),
        clearance_m=float(ctx.cfg.pusht_back_hover_clearance_m),
        label="back",
    )
    print(
        "[PushT] special back: moving to hover, then returning to saved pre-avoid joint state",
        flush=True,
    )
    _reset_robot_without_episode_finalize(
        ctx,
        {"target_joint_position": hover_state},
        reset_max_joint_velocity=float(
            ctx.cfg.pusht_back_hover_reset_max_joint_velocity
        ),
    )
    _reset_robot_without_episode_finalize(
        ctx,
        {"target_joint_position": _copy_joint_state(state)},
    )
    _clear_pusht_pre_avoid_overlay(ctx)
    ctx.terminal_event = None
    if not keep_learning:
        ctx.state_machine.state = "idle"


def _freeze_pusht_pre_avoid_overlay(ctx: RLContext, sample: dict[str, Any]) -> None:
    frozen = {
        key: value
        for key, value in sample.items()
        if key not in {"id", "label", "snapshot"}
    }
    frozen["frozen_pre_avoid"] = True
    frozen["frozen_pre_avoid_sample_id"] = sample.get("id")
    frozen["frozen_pre_avoid_snapshot"] = sample.get("snapshot")
    setattr(ctx, "pusht_pre_avoid_overlay_projection", frozen)


def _clear_pusht_pre_avoid_overlay(ctx: RLContext) -> None:
    if hasattr(ctx, "pusht_pre_avoid_overlay_projection"):
        delattr(ctx, "pusht_pre_avoid_overlay_projection")
    if hasattr(ctx, "pusht_reject_control_until_back_warned"):
        delattr(ctx, "pusht_reject_control_until_back_warned")


def _build_pusht_hover_state_from_joint_state(
    ctx: RLContext,
    saved_state: dict[str, np.ndarray],
    *,
    hover_z: float,
    clearance_m: float,
    label: str,
) -> dict[str, np.ndarray]:
    if not hasattr(ctx, "pusht_special_hover_kinematics"):
        setattr(
            ctx,
            "pusht_special_hover_kinematics",
            YamKinematics(position_cost=1.0, orientation_cost=1.0),
        )
    kinematics = getattr(ctx, "pusht_special_hover_kinematics")
    saved = _copy_joint_state(saved_state)
    left_joint = np.asarray(saved["left_joint_pos"], dtype=np.float32).reshape(6)
    right_joint = np.asarray(saved["right_joint_pos"], dtype=np.float32).reshape(6)
    left_pos, left_quat, _right_pos, _right_quat = kinematics.forward_kinematics(
        left_joint,
        right_joint,
    )
    hover_pos = np.asarray(left_pos, dtype=np.float32).reshape(3).copy()
    hover_pos[2] = max(
        float(hover_z),
        float(hover_pos[2]) + float(clearance_m),
    )
    hover_left_joint, _, _, _ = kinematics.inverse_kinematics_full(
        hover_pos,
        np.asarray(left_quat, dtype=np.float32).reshape(4),
        None,
        None,
        left_seed=left_joint,
        right_seed=right_joint,
        max_iters=40,
    )
    hover_state = {
        "left_joint_pos": np.asarray(hover_left_joint, dtype=np.float32).reshape(6),
        "left_gripper_pos": saved["left_gripper_pos"].copy(),
        "right_joint_pos": right_joint.copy(),
        "right_gripper_pos": saved["right_gripper_pos"].copy(),
    }
    fk_hover, _, _, _ = kinematics.forward_kinematics(
        hover_state["left_joint_pos"],
        hover_state["right_joint_pos"],
    )
    print(
        f"[PushT] special {label} hover target={np.round(hover_pos, 4)} "
        f"fk={np.round(fk_hover, 4)}",
        flush=True,
    )
    return hover_state


def _pusht_special_action(action: dict) -> str | None:
    raw = action.get(PUSHT_SPECIAL_ACTION_KEY)
    if raw is None:
        return None
    arr = np.asarray(raw).reshape(-1)
    if arr.size == 0:
        return None
    try:
        action_id = int(arr[0])
    except (TypeError, ValueError):
        return None
    return PUSHT_SPECIAL_ACTION_BY_ID.get(action_id)


def handle_pusht_special_action(ctx: RLContext, action: dict) -> bool:
    special = _pusht_special_action(action)
    if special is None:
        return False
    print(f"[PushT] policy special action={special}", flush=True)
    if special == "avoid":
        handle_pusht_avoid(ctx, keep_learning=True)
    elif special == "back":
        handle_pusht_back(ctx, keep_learning=True)
    return True


def _queue_pusht_restart(ctx: RLContext) -> None:
    output_dir = getattr(ctx.env, "output_dir", None)
    if output_dir is None:
        print("[PushT] restart ignored: env has no output_dir", flush=True)
        return
    root = Path(output_dir).parent
    new_dir = root / time.strftime("%Y%m%d-%H%M%S")
    suffix = 1
    while new_dir.exists():
        new_dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
        suffix += 1
    new_dir.mkdir(parents=True, exist_ok=True)
    ctx.external_event_queue.put(("restart", {"path": str(new_dir)}))
    print(f"[PushT] keyboard restart -> {new_dir}", flush=True)


def _fire_async_http(url: str) -> None:
    def _post() -> None:
        try:
            requests.post(url, timeout=1.0)
        except requests.RequestException as exc:
            print(f"[Warn] keyboard->HTTP failed for {url}: {exc}", flush=True)

    threading.Thread(target=_post, daemon=True).start()


def render_pusht(ctx: RLContext) -> None:
    if not ctx.cfg.display_image or ctx.img_queue is None:
        return
    frames = []
    for camera_name in ctx.cfg.enabled_camera_names:
        frame = ctx.obs.get(f"{camera_name}_camera_image")
        if frame is not None:
            frames.append((camera_name, frame))
    gripper_overlay = _build_pusht_gripper_top_overlay(ctx)
    if gripper_overlay is not None:
        frames.append(gripper_overlay)
    reward = score_pusht_reward(ctx)
    reward_panel = build_pusht_reward_panel(ctx, reward)
    if reward_panel is not None:
        frames.append(("reward", reward_panel))
    if not frames:
        return

    grid_rows, grid_cols = _pusht_camera_grid_shape(len(frames))
    grid_h = max(1, int(ctx.cfg.pusht_viewer_camera_tile_height))
    tile_h = max(1, grid_h // max(1, grid_rows))
    aspects = [
        frame.shape[1] / frame.shape[0]
        for _, frame in frames
        if np.asarray(frame).ndim >= 2 and frame.shape[0] > 0
    ]
    aspect = float(np.median(aspects)) if aspects else 16.0 / 9.0
    tile_w = max(1, int(round(tile_h * aspect)))
    tiles: list[np.ndarray] = []
    for name, frame in frames:
        frame = _resize_pusht_viewer_tile(frame, tile_w, tile_h)
        font_scale = max(0.45, min(0.75, tile_h / 720.0 * 0.7))
        thickness = max(1, int(round(tile_h / 360.0)))
        cv2.putText(
            frame,
            name,
            (8, max(22, int(0.07 * tile_h))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        tiles.append(frame)

    empty_tile = np.zeros((tile_h, tile_w, 3), dtype=tiles[0].dtype)
    rows = []
    for row_idx in range(grid_rows):
        row_tiles = []
        for col_idx in range(grid_cols):
            tile_idx = row_idx * grid_cols + col_idx
            row_tiles.append(
                tiles[tile_idx] if tile_idx < len(tiles) else empty_tile.copy()
            )
        rows.append(np.concatenate(row_tiles, axis=1))
    img = np.concatenate(rows, axis=0)
    padded = np.zeros((img.shape[0] + 8, img.shape[1] + 8, 3), dtype=img.dtype)
    padded[4:-4, 4:-4] = img
    left_pos = ctx.obs.get("left_ee_pos")
    if left_pos is not None:
        p = np.asarray(left_pos, dtype=float).reshape(3)
        cv2.putText(
            padded,
            f"L xyz {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}",
            (12, padded.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        padded,
        ctx.state_machine.state,
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    put_latest_image(ctx.img_queue, padded)


def _pusht_camera_grid_shape(count: int) -> tuple[int, int]:
    if count <= 6:
        return 2, 3
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    return rows, cols


def _build_pusht_gripper_top_overlay(ctx: RLContext) -> tuple[str, np.ndarray] | None:
    top = ctx.obs.get("top_camera_image")
    if top is None:
        return None

    frame = pusht_gripper_overlay.normalize_rgb_frame(top)
    if frame is None:
        return None

    projection = _pusht_overlay_projection_payload(ctx)
    if projection is None:
        cv2.putText(
            frame,
            "left tip projection unavailable",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return "top + left tip", frame

    overlay = pusht_gripper_overlay.draw_green_dot(frame, projection["corrected_uv"])
    return "top + gripper dot", frame if overlay is None else overlay


def _pusht_left_gripper_tip_world(ctx: RLContext) -> np.ndarray | None:
    pos = ctx.obs.get("left_ee_pos")
    if pos is None:
        return None
    tip = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    rot6d = ctx.obs.get("left_ee_rot6d")
    if rot6d is None:
        return tip
    try:
        rot = _rot6d_to_rot_matrix(rot6d)
    except Exception:
        return tip
    offset_m = float(os.environ.get("PUSHT_VIEWER_GRIPPER_TIP_OFFSET_M", "0.1347"))
    return tip + np.asarray(rot, dtype=np.float64).reshape(3, 3)[:, 2] * offset_m


def _pusht_overlay_adjust_px(ctx: RLContext, tip_world: Any | None = None) -> np.ndarray:
    value = getattr(ctx, "pusht_overlay_adjust_px", None)
    if value is None:
        value = np.zeros(2, dtype=np.float64)
        setattr(ctx, "pusht_overlay_adjust_px", value)
    arr = np.asarray(value, dtype=np.float64).reshape(2)
    anchor = getattr(ctx, "pusht_overlay_adjust_anchor_tip_world", None)
    if tip_world is not None and anchor is not None:
        tip = np.asarray(tip_world, dtype=np.float64).reshape(3)
        anchor_arr = np.asarray(anchor, dtype=np.float64).reshape(3)
        radius = float(getattr(ctx.cfg, "pusht_overlay_manual_anchor_radius_m", 0.015))
        if float(np.linalg.norm(tip[:2] - anchor_arr[:2])) > radius:
            arr = np.zeros(2, dtype=np.float64)
            setattr(ctx, "pusht_overlay_adjust_px", arr)
            delattr(ctx, "pusht_overlay_adjust_anchor_tip_world")
            return arr
    return arr


def _set_pusht_overlay_adjust_px(
    ctx: RLContext,
    value: Any,
    tip_world: Any | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(2)
    setattr(ctx, "pusht_overlay_adjust_px", arr)
    if tip_world is None or float(np.linalg.norm(arr)) < 1e-9:
        if hasattr(ctx, "pusht_overlay_adjust_anchor_tip_world"):
            delattr(ctx, "pusht_overlay_adjust_anchor_tip_world")
    else:
        setattr(
            ctx,
            "pusht_overlay_adjust_anchor_tip_world",
            np.asarray(tip_world, dtype=np.float64).reshape(3),
        )
    return arr


def _pusht_overlay_table_prediction(ctx: RLContext, tip_world: Any) -> dict[str, Any]:
    empty = {
        "enabled": False,
        "offset_px": [0.0, 0.0],
        "sample_count": 0,
        "occupied_cells": 0,
        "cell": None,
        "in_bounds": False,
    }
    if not bool(getattr(ctx.cfg, "pusht_overlay_table_correction_enabled", True)):
        return empty
    cache = _pusht_overlay_table_cache(ctx)
    return pusht_gripper_overlay.table_prediction(
        cache,
        tip_world,
        min_samples=int(getattr(ctx.cfg, "pusht_overlay_table_min_samples", 3)),
        neighbors=int(getattr(ctx.cfg, "pusht_overlay_table_idw_neighbors", 12)),
        power=float(getattr(ctx.cfg, "pusht_overlay_table_idw_power", 2.0)),
    )


def _pusht_overlay_table_cache(ctx: RLContext) -> dict[str, Any]:
    sample_file = _pusht_overlay_sample_file(ctx)
    stat_key = pusht_gripper_overlay.sample_file_key(sample_file)
    grid_size = max(2, int(getattr(ctx.cfg, "pusht_overlay_table_grid_size", 100)))
    xlim = tuple(float(v) for v in getattr(ctx.cfg, "pusht_bbox_x_lim", (0.3085, 0.78)))
    ylim = tuple(float(v) for v in getattr(ctx.cfg, "pusht_bbox_y_lim", (-0.3, 0.3)))
    cache_key = (stat_key, grid_size, xlim, ylim)
    cached = getattr(ctx, "pusht_overlay_table_cache", None)
    if cached is not None and cached.get("cache_key") == cache_key:
        return cached

    cache = pusht_gripper_overlay.build_table_cache(
        sample_file,
        grid_size=grid_size,
        xlim=xlim,
        ylim=ylim,
    )
    cache["cache_key"] = cache_key
    setattr(ctx, "pusht_overlay_table_cache", cache)
    return cache


def _pusht_overlay_table_cell(
    x: float,
    y: float,
    xlim: np.ndarray,
    ylim: np.ndarray,
    grid_size: int,
) -> tuple[int, int]:
    x_span = max(float(xlim[1] - xlim[0]), 1e-9)
    y_span = max(float(ylim[1] - ylim[0]), 1e-9)
    ix = int(np.floor((float(x) - float(xlim[0])) / x_span * grid_size))
    iy = int(np.floor((float(y) - float(ylim[0])) / y_span * grid_size))
    return int(np.clip(ix, 0, grid_size - 1)), int(np.clip(iy, 0, grid_size - 1))


def _pusht_overlay_projection_payload(ctx: RLContext) -> dict[str, Any] | None:
    frozen = getattr(ctx, "pusht_pre_avoid_overlay_projection", None)
    if frozen is not None:
        payload = dict(frozen)
        payload["time_s"] = time.time()
        return payload

    top = ctx.obs.get("top_camera_image")
    tip = _pusht_left_gripper_tip_world(ctx)
    if top is None or tip is None:
        return None
    uv_depth = _project_world_point_to_top_camera(ctx, tip, np.asarray(top).shape)
    if uv_depth is None:
        return None
    u, v, depth = uv_depth
    table = _pusht_overlay_table_prediction(ctx, tip)
    manual = _pusht_overlay_adjust_px(ctx, tip)
    learned = np.asarray(table["offset_px"], dtype=np.float64)
    adjust = learned + manual
    raw_uv = np.asarray([float(u), float(v)], dtype=np.float64)
    table_uv = raw_uv + learned
    corrected = raw_uv + adjust
    return {
        "time_s": time.time(),
        "tip_world": [float(x) for x in np.asarray(tip).reshape(3)],
        "left_ee_pos": [
            float(x) for x in np.asarray(ctx.obs.get("left_ee_pos", np.zeros(3))).reshape(3)
        ],
        "left_ee_rot6d": [
            float(x) for x in np.asarray(ctx.obs.get("left_ee_rot6d", np.zeros(6))).reshape(6)
        ],
        "predicted_uv": [float(u), float(v)],
        "raw_predicted_uv": [float(u), float(v)],
        "table_predicted_uv": [float(table_uv[0]), float(table_uv[1])],
        "corrected_uv": [float(corrected[0]), float(corrected[1])],
        "offset_px": [float(adjust[0]), float(adjust[1])],
        "table_offset_px": [float(learned[0]), float(learned[1])],
        "manual_offset_px": [float(manual[0]), float(manual[1])],
        "table": table,
        "depth_m": float(depth),
        "tip_offset_m": float(os.environ.get("PUSHT_VIEWER_GRIPPER_TIP_OFFSET_M", "0.1347")),
    }


def update_pusht_gripper_dot_observation(ctx: RLContext) -> None:
    top = ctx.obs.get("top_camera_image")
    key = str(
        getattr(ctx.cfg, "pusht_gripper_dot_observation_key", "top_gripper_dot_camera_image")
    )
    if top is None:
        ctx.obs.pop(key, None)
        return
    projection = _pusht_overlay_projection_payload(ctx)
    if projection is None:
        ctx.obs.pop(key, None)
        return
    overlay = pusht_gripper_overlay.draw_green_dot(top, projection["corrected_uv"])
    if overlay is None:
        ctx.obs.pop(key, None)
        return
    ctx.obs[key] = overlay


def _project_world_point_to_top_camera(
    ctx: RLContext,
    point_world: Any,
    image_shape: tuple[int, ...],
) -> tuple[int, int, float] | None:
    calibration = _pusht_top_camera_calibration(ctx, image_shape)
    T_world_cam = _pusht_top_camera_to_world(ctx)
    if calibration is None or T_world_cam is None:
        return None

    K, dist = calibration
    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    T_cam_world = np.linalg.inv(T_world_cam)
    point_cam = T_cam_world[:3, :3] @ point + T_cam_world[:3, 3]
    z = float(point_cam[2])
    if abs(z) < 1e-9:
        return None
    projected, _ = cv2.projectPoints(
        point_cam.reshape(1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        dist,
    )
    u, v = [int(round(float(x))) for x in projected.reshape(2)]
    return u, v, z


def _pusht_top_camera_calibration(
    ctx: RLContext,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    env = getattr(ctx.env, "unwrapped", ctx.env)
    cameras = getattr(env, "cameras", {})
    cam = cameras.get("top") if isinstance(cameras, dict) else None
    intr = None
    if cam is not None:
        source = getattr(cam, "camera", cam)
        get_intrinsics = getattr(source, "get_intrinsics", None)
        if callable(get_intrinsics):
            try:
                intr = get_intrinsics()
            except Exception:
                intr = None
    if not intr:
        return None

    if isinstance(intr, dict):
        fx, fy, cx, cy = [float(intr[k]) for k in ("fx", "fy", "cx", "cy")]
        src_w = float(intr.get("width", image_shape[1]))
        src_h = float(intr.get("height", image_shape[0]))
        dist = np.asarray(intr.get("distortion_coeffs", []), dtype=np.float64).reshape(-1)
    else:
        fx, fy, cx, cy = [float(v) for v in intr[:4]]
        src_w = float(image_shape[1])
        src_h = float(image_shape[0])
        dist = np.zeros(5, dtype=np.float64)
    dst_h, dst_w = image_shape[:2]
    if src_w > 0 and src_h > 0:
        sx = float(dst_w) / src_w
        sy = float(dst_h) / src_h
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy
    if min(fx, fy) <= 0:
        return None
    if dist.size == 0:
        dist = np.zeros(5, dtype=np.float64)
    return (
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        dist,
    )


def _pusht_top_camera_to_world(ctx: RLContext) -> np.ndarray | None:
    cached = getattr(ctx, "pusht_top_camera_to_world", None)
    if cached is not None:
        return np.asarray(cached, dtype=np.float64).reshape(4, 4)
    calibrated_xml = os.environ.get("YAM_STATION_CALIBRATED_XML_PATH", "").strip()
    if calibrated_xml:
        try:
            import mujoco
            from enpire.env.forge.robot.models.station.paths import get_top_camera_frame

            model = mujoco.MjModel.from_xml_path(calibrated_xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            camera_name = os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame())
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                raise RuntimeError(f"camera {camera_name!r} not found")
            T_world_cam = np.eye(4, dtype=np.float64)
            T_world_cam[:3, :3] = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
            T_world_cam[:3, 3] = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).reshape(3)
            setattr(ctx, "pusht_top_camera_to_world", T_world_cam)
            return T_world_cam
        except Exception as exc:
            if not getattr(ctx, "pusht_top_xml_projection_warned", False):
                print(
                    f"[PushT] calibrated XML top projection unavailable: {exc}",
                    flush=True,
                )
                setattr(ctx, "pusht_top_xml_projection_warned", True)
    try:
        import yourdfpy
        from enpire.env.forge.robot.models.station.paths import (
            get_station_urdf,
            get_top_camera_frame,
            needs_optical_flip,
        )

        urdf = yourdfpy.URDF.load(str(get_station_urdf()))
        cam_frame = get_top_camera_frame()
        T_world_cam = np.asarray(
            urdf.get_transform(cam_frame, "base_link"), dtype=np.float64
        )
        if needs_optical_flip("top"):
            T_world_cam = T_world_cam @ np.diag([-1.0, -1.0, 1.0, 1.0])
        setattr(ctx, "pusht_top_camera_to_world", T_world_cam)
        return T_world_cam
    except Exception as exc:
        if not getattr(ctx, "pusht_top_projection_warned", False):
            print(f"[PushT] top projection unavailable: {exc}", flush=True)
            setattr(ctx, "pusht_top_projection_warned", True)
        return None


def start_pusht_overlay_calibration_server(ctx: RLContext) -> None:
    if not bool(getattr(ctx.cfg, "pusht_overlay_calibration_enabled", True)):
        return
    if getattr(ctx, "pusht_overlay_calibration_server_started", False):
        return
    try:
        import uvicorn
        from fastapi import Body, FastAPI, Response
    except Exception as exc:
        print(f"[PushT] overlay calibration UI disabled: {exc}", flush=True)
        return

    app = FastAPI(title="PushT overlay calibration")

    @app.get("/")
    def index() -> Response:
        return Response(_PUSHT_OVERLAY_CALIBRATION_HTML, media_type="text/html")

    @app.get("/state")
    def state() -> dict:
        payload = _pusht_overlay_projection_payload(ctx) or {}
        payload["sample_count"] = _pusht_overlay_sample_count(ctx)
        payload["station_name"] = _pusht_overlay_station_name(ctx)
        payload["sample_file"] = str(_pusht_overlay_sample_file(ctx))
        payload["sample_file_name"] = _pusht_overlay_sample_file_name(ctx)
        payload["sample_files"] = _pusht_overlay_sample_file_options(ctx)
        return payload

    @app.get("/frame.jpg")
    def frame() -> Response:
        image = _pusht_overlay_calibration_frame(ctx)
        if image is None:
            return Response("no frame", status_code=503, media_type="text/plain")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            return Response("encode failed", status_code=500, media_type="text/plain")
        return Response(bytes(encoded), media_type="image/jpeg")

    @app.get("/coverage.jpg")
    def coverage() -> Response:
        image = _pusht_overlay_coverage_frame(ctx)
        if image is None:
            return Response("no frame", status_code=503, media_type="text/plain")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            return Response("encode failed", status_code=500, media_type="text/plain")
        return Response(bytes(encoded), media_type="image/jpeg")

    @app.post("/nudge")
    def nudge(payload: dict = Body(default={})) -> dict:
        if getattr(ctx, "pusht_pre_avoid_overlay_projection", None) is not None:
            return state()
        dx = float(payload.get("dx", 0.0))
        dy = float(payload.get("dy", 0.0))
        tip = _pusht_left_gripper_tip_world(ctx)
        adjust = _pusht_overlay_adjust_px(ctx, tip) + np.asarray([dx, dy], dtype=np.float64)
        _set_pusht_overlay_adjust_px(ctx, adjust, tip)
        return state()

    @app.post("/set_observed_uv")
    def set_observed_uv(payload: dict = Body(default={})) -> dict:
        if getattr(ctx, "pusht_pre_avoid_overlay_projection", None) is not None:
            return state()
        projection = _pusht_overlay_projection_payload(ctx)
        if projection is None:
            return {"success": False, "error": "projection unavailable"}
        u = float(payload["u"])
        v = float(payload["v"])
        pred = np.asarray(projection["predicted_uv"], dtype=np.float64)
        learned = np.asarray(projection["table_offset_px"], dtype=np.float64)
        observed_offset = np.asarray([u, v], dtype=np.float64) - pred
        tip = projection.get("tip_world")
        _set_pusht_overlay_adjust_px(ctx, observed_offset - learned, tip)
        return state()

    @app.post("/save")
    def save(payload: dict = Body(default={})) -> dict:
        label = str(payload.get("label", "")).strip()
        return _save_pusht_overlay_sample(ctx, label=label)

    @app.post("/select_sample_file")
    def select_sample_file(payload: dict = Body(default={})) -> dict:
        file_name = str(payload.get("file", "")).strip()
        try:
            selected = _validate_pusht_overlay_sample_file_name(file_name)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        setattr(ctx, "pusht_overlay_sample_file_name", selected)
        _set_pusht_overlay_adjust_px(ctx, np.zeros(2, dtype=np.float64))
        if hasattr(ctx, "pusht_overlay_table_cache"):
            delattr(ctx, "pusht_overlay_table_cache")
        return {"success": True, **state()}

    host = str(getattr(ctx.cfg, "pusht_overlay_calibration_host", "127.0.0.1"))
    port = int(getattr(ctx.cfg, "pusht_overlay_calibration_port", 8205))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(
        target=server.run,
        name="pusht-overlay-calibration-ui",
        daemon=True,
    )
    thread.start()
    setattr(ctx, "pusht_overlay_calibration_server_started", True)
    print(
        f"[PushT] overlay calibration UI listening on http://{host}:{port}",
        flush=True,
    )


def _pusht_overlay_calibration_frame(ctx: RLContext) -> np.ndarray | None:
    top = ctx.obs.get("top_camera_image")
    if top is None:
        return None
    frame = np.asarray(top).copy()
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    frame = frame[:, :, :3].copy()
    projection = _pusht_overlay_projection_payload(ctx)
    if projection is None:
        cv2.putText(
            frame,
            "projection unavailable",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame
    pred = np.asarray(projection["predicted_uv"], dtype=np.float64)
    table_uv = np.asarray(projection["table_predicted_uv"], dtype=np.float64)
    corr = np.asarray(projection["corrected_uv"], dtype=np.float64)
    h, w = frame.shape[:2]
    pred_pt = (int(np.clip(round(pred[0]), 0, w - 1)), int(np.clip(round(pred[1]), 0, h - 1)))
    table_pt = (
        int(np.clip(round(table_uv[0]), 0, w - 1)),
        int(np.clip(round(table_uv[1]), 0, h - 1)),
    )
    corr_pt = (int(np.clip(round(corr[0]), 0, w - 1)), int(np.clip(round(corr[1]), 0, h - 1)))
    cv2.circle(frame, pred_pt, 8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, table_pt, 10, (0, 128, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, corr_pt, 12, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, corr_pt, 15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, pred_pt, corr_pt, (0, 255, 255), 2, cv2.LINE_AA)
    offset = projection["offset_px"]
    table = projection["table"]
    cv2.putText(
        frame,
        f"cyan=raw blue=table green=final offset=({offset[0]:+.1f},{offset[1]:+.1f}) cells={table['occupied_cells']}",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if projection.get("frozen_pre_avoid"):
        cv2.putText(
            frame,
            "pre-avoid dot frozen until back finishes",
            (12, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def _pusht_overlay_coverage_frame(ctx: RLContext) -> np.ndarray | None:
    top = ctx.obs.get("top_camera_image")
    if top is None:
        return None
    return pusht_gripper_overlay.coverage_frame(top, _pusht_overlay_sample_file(ctx))


def _pusht_overlay_samples(ctx: RLContext) -> list[dict[str, Any]]:
    return pusht_gripper_overlay.load_samples(_pusht_overlay_sample_file(ctx))


def _pusht_overlay_station_name(ctx: RLContext) -> str:
    raw = (
        getattr(ctx.cfg, "pusht_overlay_station_name", None)
        or os.environ.get("PUSHT_OVERLAY_STATION_NAME")
        or socket.gethostname()
    )
    name = "".join(
        ch.lower() if ch.isalnum() or ch in "._-" else "-"
        for ch in str(raw).strip()
    ).strip("._-")
    return name or "unknown-station"


def _pusht_overlay_sample_dir(ctx: RLContext) -> Path:
    root = (
        Path(ctx.cfg.data_saving_path).expanduser()
        if ctx.cfg.data_saving_path is not None
        else REPO_ROOT / "tmp"
    )
    path = root / "pusht_overlay_calibration"
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshots").mkdir(parents=True, exist_ok=True)
    return path


def _pusht_overlay_sample_file_dir(ctx: RLContext) -> Path:
    root = Path(
        str(
            getattr(
                ctx.cfg,
                "pusht_overlay_sample_file_dir",
                "cap/tasks/pusht/overlay_tables",
            )
        )
    ).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_pusht_overlay_sample_file_name(file_name: str) -> str:
    name = str(file_name).strip()
    if not name:
        raise ValueError("missing sample file name")
    if Path(name).name != name:
        raise ValueError("sample file name must not include a directory")
    if not name.endswith(".jsonl"):
        raise ValueError("sample file name must end with .jsonl")
    if any(ch not in "._-" and not ch.isalnum() for ch in name):
        raise ValueError("sample file name may only contain letters, numbers, ., _, -")
    return name


def _pusht_overlay_default_sample_file_name(ctx: RLContext) -> str:
    return f"{_pusht_overlay_station_name(ctx)}_samples.jsonl"


def _pusht_overlay_sample_file_name(ctx: RLContext) -> str:
    current = getattr(ctx, "pusht_overlay_sample_file_name", None)
    if current:
        try:
            return _validate_pusht_overlay_sample_file_name(str(current))
        except ValueError:
            pass
    return _pusht_overlay_default_sample_file_name(ctx)


def _pusht_overlay_sample_file(ctx: RLContext) -> Path:
    return _pusht_overlay_sample_file_dir(ctx) / _pusht_overlay_sample_file_name(ctx)


def _pusht_overlay_sample_file_options(ctx: RLContext) -> list[dict[str, Any]]:
    root = _pusht_overlay_sample_file_dir(ctx)
    names = {p.name for p in root.glob("*.jsonl") if p.is_file()}
    names.add(_pusht_overlay_default_sample_file_name(ctx))
    selected = _pusht_overlay_sample_file_name(ctx)
    if selected not in names:
        names.add(selected)
    options: list[dict[str, Any]] = []
    for name in sorted(names):
        path = root / name
        options.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "selected": name == selected,
                "sample_count": _pusht_overlay_sample_count_for_file(path),
            }
        )
    return options


def _pusht_overlay_sample_count(ctx: RLContext) -> int:
    return _pusht_overlay_sample_count_for_file(_pusht_overlay_sample_file(ctx))


def _pusht_overlay_sample_count_for_file(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except OSError:
        return 0


def _save_pusht_overlay_sample(ctx: RLContext, *, label: str = "") -> dict:
    projection = _pusht_overlay_projection_payload(ctx)
    if projection is None:
        return {"success": False, "error": "projection unavailable"}
    out_dir = _pusht_overlay_sample_dir(ctx)
    sample_id = time.strftime("%Y%m%dT%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    frame = _pusht_overlay_calibration_frame(ctx)
    snapshot_path = out_dir / "snapshots" / f"{sample_id}.jpg"
    if frame is not None:
        cv2.imwrite(str(snapshot_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    sample = {
        "id": sample_id,
        "label": label,
        "snapshot": str(snapshot_path),
        **projection,
    }
    jsonl_path = _pusht_overlay_sample_file(ctx)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, sort_keys=True) + "\n")
    latest_path = out_dir / "latest_sample.json"
    latest_path.write_text(json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8")
    if hasattr(ctx, "pusht_overlay_table_cache"):
        delattr(ctx, "pusht_overlay_table_cache")
    new_table = _pusht_overlay_table_prediction(ctx, projection["tip_world"])
    saved_offset = np.asarray(projection["offset_px"], dtype=np.float64)
    learned_offset = np.asarray(new_table["offset_px"], dtype=np.float64)
    _set_pusht_overlay_adjust_px(
        ctx,
        saved_offset - learned_offset,
        projection["tip_world"],
    )
    count = _pusht_overlay_sample_count(ctx)
    print(
        f"[PushT] overlay calibration saved sample {sample_id} count={count} "
        f"offset_px={sample['offset_px']}",
        flush=True,
    )
    return {
        "success": True,
        "count": count,
        "sample": sample,
        "jsonl": str(jsonl_path),
    }


_PUSHT_OVERLAY_CALIBRATION_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PushT Overlay Calibration</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; width: 100vw; height: 100dvh; overflow: hidden; }
    #imagePane { min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; background: #000; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; }
    .view { position: relative; min-width: 0; min-height: 0; overflow: hidden; background: #000; }
    .view img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .view span { position: absolute; top: 8px; left: 8px; padding: 4px 7px; background: rgba(0,0,0,0.62); color: #fff; font-size: 12px; border-radius: 4px; pointer-events: none; }
    #frameView { touch-action: none; }
    #frame { cursor: crosshair; user-select: none; }
    #dragDot {
      position: absolute;
      left: 0;
      top: 0;
      width: 30px;
      height: 30px;
      margin: -15px 0 0 -15px;
      border: 2px solid #fff;
      border-radius: 50%;
      background: rgba(0,255,0,0.62);
      box-shadow: 0 0 0 2px rgba(0,0,0,0.55), 0 0 12px rgba(0,255,0,0.9);
      box-sizing: border-box;
      cursor: grab;
      display: none;
      pointer-events: auto;
      touch-action: none;
    }
    #dragDot.dragging { cursor: grabbing; background: rgba(0,255,0,0.85); }
    aside { min-width: 0; height: 100dvh; box-sizing: border-box; overflow-y: auto; overscroll-behavior: contain; padding: 14px; border-left: 1px solid #333; background: #181818; }
    button, input, select { font: inherit; }
    button { margin: 3px; padding: 8px 10px; background: #2a2a2a; color: #eee; border: 1px solid #555; border-radius: 4px; }
    button:hover { background: #3a3a3a; }
    input, select { width: 100%; box-sizing: border-box; padding: 8px; margin: 8px 0; background: #0c0c0c; color: #eee; border: 1px solid #555; }
    pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; color: #b8f5c8; }
    .pad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; max-width: 180px; }
  </style>
</head>
<body>
<main>
  <section id="imagePane">
    <div class="view" id="frameView"><img id="frame" src="/frame.jpg"><div id="dragDot" title="Drag to align final dot"></div><span>current</span></div>
    <div class="view"><img id="coverage" src="/coverage.jpg"><span>coverage</span></div>
  </section>
  <aside>
    <h2>Overlay Calibration</h2>
    <div>Cyan is raw projection, blue is table prediction, green is final. Drag the green dot, click the real gripper tip, or nudge with arrow keys.</div>
    <select id="sampleFile" title="Calibration file"></select>
    <input id="label" placeholder="label">
    <div class="pad">
      <span></span><button onclick="nudge(0,-1)">Up</button><span></span>
      <button onclick="nudge(-1,0)">Left</button><button onclick="save()">Save</button><button onclick="nudge(1,0)">Right</button>
      <span></span><button onclick="nudge(0,1)">Down</button><span></span>
    </div>
    <div>
      <button onclick="nudge(-5,0)">-5 x</button>
      <button onclick="nudge(5,0)">+5 x</button>
      <button onclick="nudge(0,-5)">-5 y</button>
      <button onclick="nudge(0,5)">+5 y</button>
    </div>
    <pre id="state"></pre>
  </aside>
</main>
<script>
const img = document.getElementById('frame');
const frameView = document.getElementById('frameView');
const dragDot = document.getElementById('dragDot');
const coverage = document.getElementById('coverage');
const stateEl = document.getElementById('state');
const sampleFile = document.getElementById('sampleFile');
let updatingSampleFile = false;
let latestState = {};
let draggingDot = false;
let pendingDragUv = null;
let dragCommitTimer = 0;
let dragCommitInFlight = false;
let dragGrabOffset = {u: 0, v: 0};
let refreshAfterDragCommit = false;
async function post(path, payload) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload || {})});
  return await r.json();
}
function imageMetrics() {
  if (!img.naturalWidth || !img.naturalHeight) return null;
  const r = img.getBoundingClientRect();
  const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const shownW = img.naturalWidth * scale;
  const shownH = img.naturalHeight * scale;
  return {
    scale,
    x0: r.left + (r.width - shownW) / 2,
    y0: r.top + (r.height - shownH) / 2,
    shownW,
    shownH,
  };
}
function pointerToImageUv(ev, clamp) {
  const m = imageMetrics();
  if (!m) return null;
  let u = (ev.clientX - m.x0) / m.scale;
  let v = (ev.clientY - m.y0) / m.scale;
  if (clamp) {
    return clampImageUv(u, v);
  } else if (u < 0 || v < 0 || u >= img.naturalWidth || v >= img.naturalHeight) {
    return null;
  }
  return {u, v};
}
function clampImageUv(u, v) {
  return {
    u: Math.max(0, Math.min(img.naturalWidth - 1, u)),
    v: Math.max(0, Math.min(img.naturalHeight - 1, v)),
  };
}
function positionDragDot() {
  const uv = Array.isArray(latestState.corrected_uv) ? latestState.corrected_uv : null;
  const m = imageMetrics();
  if (!uv || !m) {
    dragDot.style.display = 'none';
    return;
  }
  const {u, v} = clampImageUv(uv[0], uv[1]);
  const viewRect = frameView.getBoundingClientRect();
  const x = m.x0 - viewRect.left + u * m.scale;
  const y = m.y0 - viewRect.top + v * m.scale;
  dragDot.style.transform = `translate(${x}px, ${y}px)`;
  dragDot.style.display = 'block';
}
function renderState(s) {
  latestState = s || {};
  updateSampleFileSelect(latestState);
  stateEl.textContent = JSON.stringify(latestState, null, 2);
  positionDragDot();
}
function dragCommitActive() {
  return draggingDot || dragCommitInFlight || pendingDragUv || dragCommitTimer || refreshAfterDragCommit;
}
function updateSampleFileSelect(s) {
  const files = Array.isArray(s.sample_files) ? s.sample_files : [];
  const selected = s.sample_file_name || '';
  const signature = files.map(f => `${f.name}:${f.sample_count}:${f.exists}:${f.selected}`).join('|');
  if (sampleFile.dataset.signature === signature) {
    sampleFile.value = selected;
    return;
  }
  updatingSampleFile = true;
  sampleFile.innerHTML = '';
  for (const f of files) {
    const opt = document.createElement('option');
    opt.value = f.name;
    opt.textContent = `${f.name} (${f.sample_count || 0})${f.exists ? '' : ' new'}`;
    sampleFile.appendChild(opt);
  }
  sampleFile.value = selected;
  sampleFile.dataset.signature = signature;
  updatingSampleFile = false;
}
async function refresh() {
  if (dragCommitActive()) return;
  img.src = '/frame.jpg?t=' + Date.now();
  coverage.src = '/coverage.jpg?t=' + Date.now();
  const s = await fetch('/state').then(r => r.json()).catch(e => ({error: String(e)}));
  renderState(s);
}
async function nudge(dx, dy) { await post('/nudge', {dx, dy}); refresh(); }
async function save() { await post('/save', {label: document.getElementById('label').value}); refresh(); }
sampleFile.addEventListener('change', async () => {
  if (updatingSampleFile) return;
  await post('/select_sample_file', {file: sampleFile.value});
  refresh();
});
async function commitPendingDrag() {
  if (dragCommitTimer) {
    clearTimeout(dragCommitTimer);
    dragCommitTimer = 0;
  }
  if (dragCommitInFlight || !pendingDragUv) return;
  const uv = pendingDragUv;
  pendingDragUv = null;
  dragCommitInFlight = true;
  const s = await post('/set_observed_uv', {u: uv.u, v: uv.v}).catch(e => ({error: String(e)}));
  dragCommitInFlight = false;
  if (pendingDragUv) {
    commitPendingDrag();
    return;
  }
  if (!draggingDot) renderState(s);
  if (refreshAfterDragCommit && !draggingDot) {
    refreshAfterDragCommit = false;
    refresh();
  }
}
function queueDraggedUv(uv) {
  pendingDragUv = uv;
  latestState = {...latestState, corrected_uv: [uv.u, uv.v]};
  positionDragDot();
  if (!dragCommitTimer) dragCommitTimer = setTimeout(commitPendingDrag, 75);
}
function startDotDrag(ev) {
  const uv = pointerToImageUv(ev, true);
  if (!uv) return;
  ev.preventDefault();
  ev.stopPropagation();
  draggingDot = true;
  dragDot.classList.add('dragging');
  dragDot.setPointerCapture(ev.pointerId);
  const currentUv = Array.isArray(latestState.corrected_uv) ? latestState.corrected_uv : [uv.u, uv.v];
  const current = clampImageUv(currentUv[0], currentUv[1]);
  dragGrabOffset = {u: uv.u - current.u, v: uv.v - current.v};
}
function moveDotDrag(ev) {
  if (!draggingDot) return;
  const uv = pointerToImageUv(ev, true);
  if (!uv) return;
  ev.preventDefault();
  queueDraggedUv(clampImageUv(uv.u - dragGrabOffset.u, uv.v - dragGrabOffset.v));
}
function finishDotDrag(ev) {
  if (!draggingDot) return;
  const uv = pointerToImageUv(ev, true);
  if (uv) queueDraggedUv(clampImageUv(uv.u - dragGrabOffset.u, uv.v - dragGrabOffset.v));
  draggingDot = false;
  dragDot.classList.remove('dragging');
  try { dragDot.releasePointerCapture(ev.pointerId); } catch (_) {}
  refreshAfterDragCommit = true;
  commitPendingDrag();
}
dragDot.addEventListener('pointerdown', startDotDrag);
dragDot.addEventListener('pointermove', moveDotDrag);
dragDot.addEventListener('pointerup', finishDotDrag);
dragDot.addEventListener('pointercancel', finishDotDrag);
img.addEventListener('click', async ev => {
  const uv = pointerToImageUv(ev, false);
  if (!uv) return;
  const {u, v} = uv;
  await post('/set_observed_uv', {u, v});
  refresh();
});
window.addEventListener('keydown', ev => {
  const step = ev.shiftKey ? 5 : 1;
  if (ev.key === 'ArrowLeft') nudge(-step, 0);
  if (ev.key === 'ArrowRight') nudge(step, 0);
  if (ev.key === 'ArrowUp') nudge(0, -step);
  if (ev.key === 'ArrowDown') nudge(0, step);
  if (ev.key === 's') save();
});
img.addEventListener('load', positionDragDot);
window.addEventListener('resize', positionDragDot);
setInterval(refresh, 750);
refresh();
</script>
</body>
</html>
"""


def _resize_pusht_viewer_tile(
    frame: np.ndarray, tile_w: int, tile_h: int
) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 viewer frame, got shape={arr.shape}")

    h, w = arr.shape[:2]
    scale = min(float(tile_w) / max(1, w), float(tile_h) / max(1, h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (resized_w, resized_h), interpolation=interpolation)
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    x0 = (tile_w - resized_w) // 2
    y0 = (tile_h - resized_h) // 2
    tile[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return tile


def score_pusht_reward(ctx: RLContext) -> dict[str, Any] | None:
    top_rgb = ctx.obs.get("top_camera_image")
    if top_rgb is None:
        return None
    try:
        success_rect = _pusht_reward_success_rect(ctx, np.asarray(top_rgb).shape)
        bgr = _as_bgr(np.asarray(top_rgb))
        cur_mask, cur_info = _dominant_component(_red_mask(bgr))
        score = _score_success(cur_mask, cur_info, success_rect)
        gate_score_by_avoid(
            score,
            avoid_active=getattr(ctx, "pusht_pre_avoid_joint_state", None) is not None,
        )
        setattr(ctx, "pusht_reward_current_bgr", bgr)
        setattr(ctx, "pusht_reward_current_mask", cur_mask)
        setattr(ctx, "pusht_last_reward", float(score["score"]))
        setattr(ctx, "pusht_last_reward_info", _json_float_dict(score))
        return score
    except Exception as exc:
        if not getattr(ctx, "pusht_reward_warned", False):
            print(f"[PushT] reward display disabled: {exc}", flush=True)
            setattr(ctx, "pusht_reward_warned", True)
        return None


def _pusht_reward_success_rect(
    ctx: RLContext,
    current_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    goal_path = Path(ctx.cfg.pusht_reward_goal_image)
    cache_key = (str(goal_path), tuple(current_shape[:2]))
    cache = getattr(ctx, "pusht_reward_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["success_rect"]

    goal_bgr_full = cv2.imread(str(goal_path), cv2.IMREAD_COLOR)
    if goal_bgr_full is None:
        raise FileNotFoundError(f"goal image not found or unreadable: {goal_path}")
    meta = _load_meta(goal_path.with_name("goal_top_meta.json"))
    crop = _top_camera_crop_from_cfg(ctx.cfg)
    success_rect = meta.get(SUCCESS_RECT_KEY)
    if success_rect is None:
        raise RuntimeError(
            f"missing {SUCCESS_RECT_KEY} in {goal_path.with_name('goal_top_meta.json')}"
        )
    source_shape = goal_bgr_full.shape[:2]
    if crop is not None:
        success_rect = _rect_full_to_crop(success_rect, crop)
        source_shape = _crop(goal_bgr_full, crop).shape[:2]
    if success_rect is None:
        raise RuntimeError(
            f"{SUCCESS_RECT_KEY} does not intersect configured top-camera crop"
        )
    cur_h, cur_w = current_shape[:2]
    if source_shape != (cur_h, cur_w):
        success_rect = _scale_rect(success_rect, source_shape, (cur_h, cur_w))
    if success_rect is None:
        raise RuntimeError(
            f"failed to scale {SUCCESS_RECT_KEY} to current top-camera image shape"
        )
    success_rect = tuple(int(v) for v in success_rect)
    cache = {"key": cache_key, "success_rect": success_rect}
    setattr(ctx, "pusht_reward_cache", cache)
    return success_rect


def build_pusht_reward_panel(
    ctx: RLContext,
    reward: dict[str, float] | None,
) -> np.ndarray | None:
    if reward is None:
        return None
    cur_bgr = getattr(ctx, "pusht_reward_current_bgr", None)
    cur_mask = getattr(ctx, "pusht_reward_current_mask", None)
    if cur_bgr is None or cur_mask is None:
        return None
    overlay_view = _draw_reward_overlay(cur_bgr, cur_mask, reward)
    tile_h = int(ctx.cfg.pusht_viewer_reward_tile_height)
    panel = cv2.resize(
        overlay_view,
        (int(overlay_view.shape[1] * tile_h / overlay_view.shape[0]), tile_h),
        interpolation=cv2.INTER_AREA,
    )
    cv2.putText(
        panel,
        (
            f"reward {reward['score']:.1f} orient={int(reward.get('orientation_ok', False))} "
            f"range={int(reward.get('range_ok', False))} "
            f"parallel_err={reward.get('crossbar_parallel_error_deg', 0.0):.1f}deg "
            f"below={int(reward.get('crossbar_below_stem_ok', False))}"
        ),
        (8, panel.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _top_camera_crop_from_cfg(cfg: PushTConfig) -> tuple[int, int, int, int] | None:
    for name, region in zip(cfg.crop_camera_names, cfg.crop_region):
        if name != "top":
            continue
        text = str(region)
        if text == "center":
            return None
        if ":" in text:
            _, text = text.split(":", 1)
        parts = [int(float(part.strip())) for part in text.split(",")]
        if len(parts) != 4:
            return None
        return tuple(parts)
    return None


def run(ctx: RLContext) -> bool:
    should_auto_reset = False
    try:
        while True:
            ctx.action_clipped = False
            fello_action, fello_info, _ = ctx.policy_router.get_fello_action(ctx.obs)
            _, keyboard_info = ctx.keyboard_policy.get_action()
            if ctx.cfg.pusht_keyboard_shortcuts_enabled:
                dispatch_pusht_keyboard_events(ctx, keyboard_info)

            event, payload = ctx.event_router.get_event(
                fello_info,
                ctx.obs,
                is_learning=(ctx.state_machine.state == "learn"),
            )
            if event in ("next_pose", "prev_pose", "set_initial_position_index"):
                ctx.pending_pose_event = event
                ctx.pending_pose_payload = payload
            if event in TERMINAL_EVENTS:
                ctx.terminal_event = event
                ctx.last_terminal_event = event
                notify_pusht_policy_done(ctx, event)
                reward_info = getattr(ctx, "pusht_last_reward_info", None)
                if reward_info is None:
                    score = score_pusht_reward(ctx)
                    reward_info = None if score is None else _json_safe_dict(score)
                terminal_success = event == "success" and _pusht_reward_meets_success(
                    reward_info,
                    float(ctx.cfg.pusht_auto_reset_threshold),
                )
                finalize_pusht_episode(ctx, success=terminal_success)
                write_pusht_reset_request_for_terminal(ctx, event)
                should_auto_reset = True
                break
            if event == "restart":
                handle_restart(ctx, payload)

            prev_state = ctx.state_machine.state
            ctx.state_machine.transition(event)
            state = ctx.state_machine.state
            if state != prev_state:
                if prev_state == "author":
                    exit_author(ctx, write_positions=(event == "home"))
                if state == "author":
                    enter_author(ctx)

            if state == "home":
                do_home(ctx)
            elif state == "change_pose":
                do_change_pose(ctx)
            elif state == "parking":
                do_parking(ctx, event)
            elif state == "author":
                do_author(ctx, fello_action, keyboard_info)
            elif state == "hover":
                do_pusht_hover(ctx)
            elif state == "learn":
                do_pusht_learn(ctx, fello_action, fello_info, keyboard_info)

            render_pusht(ctx)
            if should_trigger_pusht_auto_reset(ctx):
                finalize_pusht_episode(ctx, success=True)
                reward = float(getattr(ctx, "pusht_last_reward", 0.0))
                write_pusht_reset_request(
                    ctx,
                    reward,
                    float(ctx.cfg.pusht_auto_reset_threshold),
                    reason="reward_threshold",
                    success=True,
                )
                notify_pusht_policy_done(ctx, "success")
                should_auto_reset = True
                break
    finally:
        ctx.env.close()
    return should_auto_reset


def write_pusht_reset_request_for_terminal(ctx: RLContext, event: str) -> None:
    score = score_pusht_reward(ctx)
    reward = float((score or {}).get("score", 0.0))
    threshold = float(ctx.cfg.pusht_auto_reset_threshold)
    success = event == "success" and _pusht_reward_meets_success(score, threshold)
    write_pusht_reset_request(
        ctx,
        reward,
        threshold,
        reason=f"terminal_{event}",
        success=success,
    )


def notify_pusht_policy_done(ctx: RLContext, reason: str) -> None:
    rl_policy = getattr(ctx.policy_router, "rl_policy", None)
    done = getattr(rl_policy, "done", None)
    if not callable(done):
        return
    info = {
        "reason": reason,
        "state": ctx.state_machine.state,
        "terminal_event": ctx.terminal_event,
        "last_reward_info": getattr(ctx, "pusht_last_reward_info", None),
    }
    try:
        response = done(reason=reason, info=info)
        print(f"[PushT] policy done reason={reason} response={response}", flush=True)
    except Exception as exc:
        print(f"[PushT] policy done failed reason={reason}: {exc}", flush=True)


def finalize_pusht_episode(ctx: RLContext, *, success: bool) -> None:
    finalize = getattr(ctx.env, "finalize_episode", None)
    if finalize is None:
        return
    reward_info = getattr(ctx, "pusht_last_reward_info", None)
    if reward_info is None:
        score = score_pusht_reward(ctx)
        reward_info = None if score is None else _json_float_dict(score)
    reward = float((reward_info or {}).get("score", 1.0 if success else 0.0))
    finalize(
        discard_episode=False,
        episode_terminal_reward=reward,
        episode_terminal_done=True,
        episode_terminal_event="success" if success else "fail",
    )
    episode_dir = getattr(ctx.env, "last_episode_dir", None)
    if episode_dir is not None:
        setattr(ctx, "pusht_last_episode_dir", str(episode_dir))


def should_trigger_pusht_auto_reset(ctx: RLContext) -> bool:
    if not ctx.cfg.pusht_auto_reset_on_reward:
        return False
    if getattr(ctx, "pusht_auto_reset_triggered", False):
        return False
    if ctx.state_machine.state != "learn":
        return False

    reward_info = getattr(ctx, "pusht_last_reward_info", None)
    if reward_info is None:
        score = score_pusht_reward(ctx)
        reward_info = None if score is None else _json_safe_dict(score)
    if reward_info is None:
        return False

    threshold = float(ctx.cfg.pusht_auto_reset_threshold)
    reward = float(reward_info.get("score", 0.0))
    if reward <= threshold:
        return False
    if not _pusht_reward_meets_success(reward_info, threshold):
        range_ok = bool(reward_info.get("range_ok", False))
        orientation_ok = bool(reward_info.get("orientation_ok", False))
        parallel_ok = bool(reward_info.get("crossbar_parallel_ok", False))
        below_ok = bool(reward_info.get("crossbar_below_stem_ok", False))
        previous = getattr(ctx, "pusht_last_rejected_success_label", None)
        marker = (range_ok, orientation_ok, parallel_ok, below_ok, round(reward, 3))
        if previous != marker:
            print(
                f"[PushT] reward {reward:.3f} > {threshold:.3f} but "
                f"range_ok={range_ok} orientation_ok={orientation_ok} "
                f"crossbar_parallel_ok={parallel_ok} crossbar_below_stem_ok={below_ok}; "
                "waiting for downtop(horizontal+below-stem)+range",
                flush=True,
            )
            setattr(ctx, "pusht_last_rejected_success_label", marker)
        return False

    setattr(ctx, "pusht_auto_reset_triggered", True)
    print(
        f"[PushT] reward {reward:.3f} > {threshold:.3f} and downtop(horizontal+below-stem)+range; "
        "requesting supervisor reset",
        flush=True,
    )
    write_pusht_reset_request(
        ctx,
        reward,
        threshold,
        reason="reward_threshold",
        success=True,
    )
    return True


def write_pusht_reset_request(
    ctx: RLContext,
    reward: float,
    threshold: float,
    *,
    reason: str,
    success: bool,
) -> None:
    path_text = getattr(ctx.cfg, "pusht_reset_request_path", None)
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_info = getattr(ctx, "pusht_last_reward_info", None)
    if reward_info is None:
        score = score_pusht_reward(ctx)
        reward_info = None if score is None else _json_float_dict(score)
    payload = {
        "success": success,
        "is_success": success,
        "is_failure": not success,
        "reason": reason,
        "reward": float(reward),
        "score": float(reward),
        "reward_info": reward_info or {"score": float(reward)},
        "threshold": float(threshold),
        "state": ctx.state_machine.state,
        "time_s": time.time(),
    }
    episode_dir = getattr(ctx, "pusht_last_episode_dir", None)
    if episode_dir:
        payload["episode_dir"] = str(episode_dir)
    payload["goal_image"] = str(ctx.cfg.pusht_reward_goal_image)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[PushT] reset request written to {path}", flush=True)


def _pusht_reward_meets_success(
    reward_info: dict[str, Any] | None, threshold: float
) -> bool:
    if reward_info is None:
        return False
    reward = float(reward_info.get("score", 0.0))
    range_ok = bool(reward_info.get("range_ok", False))
    orientation_ok = bool(reward_info.get("orientation_ok", False))
    return bool(reward > float(threshold) and orientation_ok and range_ok)


def _json_safe_dict(values: dict[str, Any]) -> dict[str, Any]:
    def _safe(value: Any) -> Any:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _safe(item) for key, item in value.items()}
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    out: dict[str, Any] = {}
    for key, value in values.items():
        out[str(key)] = _safe(value)
    return out


def _json_float_dict(values: dict[str, Any]) -> dict[str, Any]:
    return _json_safe_dict(values)


def run_pusht_auto_reset(cfg: PushTConfig) -> int:
    if cfg.pusht_auto_reset_cleanup_stale_processes:
        cleanup_stale_pusht_reset_processes(
            stale_age_s=float(cfg.pusht_auto_reset_stale_process_age_s)
        )

    env = os.environ.copy()
    env.update(
        {
            "CAP_CUROBO_HOST": "127.0.0.1",
            "CAP_CUROBO_PORT": "8611",
            "CAP_CUROBO_START_SERVER": "0",
            "CAP_TOP_CAMERA_BACKEND": "realsense",
            "NAIL_PREGRASP_DEBUG_UI": "1",
            "NAIL_PREGRASP_DEBUG_UI_BLOCKING": "0",
        }
    )
    cmd = [
        "uv",
        "run",
        "python",
        "run_script.py",
        f"script_file={cfg.pusht_auto_reset_script_file}",
        f"skill_library_path={cfg.pusht_auto_reset_skill_library_path}",
        "env.name=yam-real",
        "robot=real_yam",
        "robot.dashboard=false",
        "robot.await_exit=false",
        "robot.go_home_on_exit=false",
        "execution.record=true",
        "debug_ui.enabled=false",
        "debug_ui.auto_open=false",
        "debug_ui.auto_exit_on_run_end=true",
    ]
    print(
        "[PushT] launching CAP reset after rl_pusht shutdown:",
        " ".join(cmd),
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        proc.communicate(timeout=float(cfg.pusht_auto_reset_timeout_s))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5.0)
        print(
            f"[PushT] CAP reset timed out after {cfg.pusht_auto_reset_timeout_s:.1f}s; "
            "killed reset process group",
            flush=True,
        )
        return 124
    finally:
        if cfg.pusht_auto_reset_cleanup_stale_processes:
            cleanup_stale_pusht_reset_processes(
                stale_age_s=float(cfg.pusht_auto_reset_stale_process_age_s)
            )
    print(f"[PushT] CAP reset exited with code {proc.returncode}", flush=True)
    return int(proc.returncode)


def cleanup_stale_pusht_reset_processes(stale_age_s: float) -> None:
    """Clean old PushT reset/debug-ui processes left by interrupted dashboard runs."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,stat=,etimes=,args="],
            text=True,
        )
    except Exception as exc:
        print(f"[PushT] stale reset cleanup skipped: ps failed: {exc}", flush=True)
        return

    current_pid = os.getpid()
    candidates: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid_s, _ppid_s, stat, etimes_s, args = parts
        try:
            pid = int(pid_s)
            etimes = float(etimes_s)
        except ValueError:
            continue
        if pid == current_pid or etimes < float(stale_age_s):
            continue
        is_pusht_reset = (
            "run_script.py" in args
            and "script_file=cap/saved_scripts/place_grasped_t_reset.py" in args
        )
        is_pusht_debug_ui = (
            "-m cap.debug_ui.app" in args and "place_grasped_t_reset_" in args
        )
        if is_pusht_reset or is_pusht_debug_ui:
            candidates.append(pid)

    if not candidates:
        return
    print(f"[PushT] cleaning stale PushT reset processes: {candidates}", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive: list[int] = []
        for pid in candidates:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(f"[PushT] cannot kill stale pid {pid}: {exc}", flush=True)
        if sig == signal.SIGTERM and alive:
            time.sleep(1.0)


def do_pusht_learn(
    ctx: RLContext,
    fello_action: dict,
    fello_info: dict,
    keyboard_info: dict,
) -> None:
    from enpire.policy.rl.handlers import apply_collision_filter

    update_pusht_gripper_dot_observation(ctx)
    action, _ = ctx.policy_router.route_action(
        ctx.obs,
        fello_action,
        fello_info,
        use_rl=True,
        keyboard_info=keyboard_info,
    )
    if handle_pusht_special_action(ctx, action):
        return
    if getattr(ctx, "pusht_pre_avoid_joint_state", None) is not None:
        if not getattr(ctx, "pusht_reject_control_until_back_warned", False):
            print(
                "[PushT] rejecting normal control action after avoid; "
                "policy must send special action='back' before resuming XY control",
                flush=True,
            )
            setattr(ctx, "pusht_reject_control_until_back_warned", True)
        return
    if (
        ctx.cfg.control_mode == "cartesian_position"
        and ctx.cfg.pusht_teleop_position_only_planning
    ):
        action = plan_pusht_position_only_action(ctx, action)
    action = apply_collision_filter(ctx, action)
    ctx.action_clipped = ctx.collision_filter.clipped
    if "left_ee_pos" in action and "left_ee_pos" in ctx.obs:
        left_action = np.asarray(action["left_ee_pos"], dtype=np.float32).reshape(3)
        current_left = np.asarray(ctx.obs["left_ee_pos"], dtype=np.float32).reshape(3)
        if ctx.cfg.control_mode in DELTA_EE_CONTROL_MODES:
            ctx.obs["left_target_eef"] = current_left + left_action
        else:
            ctx.obs["left_target_eef"] = left_action.copy()
    _inject_pusht_recording_observation(ctx)
    ctx.obs, _, _, _, _ = ctx.env.step(action)
    update_pusht_gripper_dot_observation(ctx)


def _inject_pusht_recording_observation(ctx: RLContext) -> None:
    current_joint = getattr(ctx.env.unwrapped, "_current_joint_pos", None)
    if isinstance(current_joint, dict):
        for key in ("left_joint_pos", "right_joint_pos"):
            if key in current_joint:
                ctx.obs[key] = (
                    np.asarray(current_joint[key], dtype=np.float32).reshape(6).copy()
                )


def plan_pusht_position_only_action(ctx: RLContext, action: dict) -> dict:
    if "left_ee_pos" not in action:
        return action
    env = ctx.env.unwrapped
    if not hasattr(ctx, "pusht_position_only_kinematics"):
        setattr(
            ctx,
            "pusht_position_only_kinematics",
            YamKinematics(
                position_cost=1.0,
                orientation_cost=float(ctx.cfg.pusht_teleop_orientation_cost),
            ),
        )
    kinematics = getattr(ctx, "pusht_position_only_kinematics")
    current = env._read_current_joint_state()
    left_xyz = np.asarray(action["left_ee_pos"], dtype=np.float64).reshape(3)
    current_left_joint = np.asarray(
        current["left_joint_pos"], dtype=np.float32
    ).reshape(6)
    current_right_joint = np.asarray(
        current["right_joint_pos"], dtype=np.float32
    ).reshape(6)
    current_left_xyz, current_left_quat, _, _ = kinematics.forward_kinematics(
        current_left_joint,
        current_right_joint,
    )

    target_delta_m = float(
        np.linalg.norm(
            np.asarray(left_xyz, dtype=np.float64)
            - np.asarray(current_left_xyz, dtype=np.float64)
        )
    )
    if target_delta_m <= float(ctx.cfg.pusht_teleop_replan_deadband_m):
        setattr(ctx, "pusht_last_teleop_left_joint_pos", current_left_joint.copy())
        return {
            "left_joint_pos": current_left_joint.copy(),
            "left_gripper_pos": np.asarray(
                action.get("left_gripper_pos", current.get("left_gripper_pos")),
                dtype=np.float32,
            ).reshape(1),
            "right_joint_pos": current_right_joint.copy(),
            "right_gripper_pos": np.asarray(
                current.get("right_gripper_pos", action.get("right_gripper_pos")),
                dtype=np.float32,
            ).reshape(1),
            "source": action.get("source", "keyboard_hold"),
        }

    seed_left_joint = np.asarray(
        getattr(ctx, "pusht_last_teleop_left_joint_pos", current_left_joint),
        dtype=np.float32,
    ).reshape(6)
    _, current_left_quat, _, _ = kinematics.forward_kinematics(
        seed_left_joint,
        current_right_joint,
    )
    left_joint_pos, _, _, _ = kinematics.inverse_kinematics_full(
        left_xyz,
        np.asarray(current_left_quat, dtype=np.float32).reshape(4),
        None,
        None,
        left_seed=seed_left_joint,
        right_seed=current_right_joint,
        max_iters=40,
    )
    fk_left_pos, fk_left_quat, _, _ = kinematics.forward_kinematics(
        np.asarray(left_joint_pos, dtype=np.float32).reshape(6),
        current_right_joint,
    )
    setattr(
        ctx,
        "pusht_last_teleop_left_joint_pos",
        np.asarray(left_joint_pos, dtype=np.float32).reshape(6).copy(),
    )
    orient_err = float("nan")
    planned = {
        "left_joint_pos": np.asarray(left_joint_pos, dtype=np.float32).reshape(6),
        "left_gripper_pos": np.asarray(
            action.get("left_gripper_pos", current.get("left_gripper_pos")),
            dtype=np.float32,
        ).reshape(1),
        "right_joint_pos": np.asarray(current_right_joint, dtype=np.float32).reshape(6),
        "right_gripper_pos": np.asarray(
            current.get("right_gripper_pos", action.get("right_gripper_pos")),
            dtype=np.float32,
        ).reshape(1),
        "source": action.get("source", "keyboard"),
    }
    if not np.allclose(fk_left_pos, left_xyz, atol=1e-3):
        print(
            f"[PushT] position-only teleop target {np.round(left_xyz, 4)} -> "
            f"fk={np.round(fk_left_pos, 4)} "
            f"teleop_point_down={ctx.cfg.pusht_teleop_point_down_after_position_only} "
            f"point_down_err={orient_err:.4f}",
            flush=True,
        )
    return planned


def load_pusht_yaml_defaults(path: str) -> PushTConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    known = {f.name for f in fields(PushTConfig)}
    cleaned = {
        k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in known
    }
    if "embodiment_tag" in cleaned and isinstance(cleaned["embodiment_tag"], str):
        cleaned["embodiment_tag"] = EmbodimentTag[cleaned["embodiment_tag"]]
    cfg = PushTConfig(**cleaned)
    cfg.config_file = str(Path(path).expanduser().resolve())
    if os.environ.get("ENPIRE_RL_INITIAL_POSITIONS"):
        cfg.initial_positions_file = os.environ["ENPIRE_RL_INITIAL_POSITIONS"]
    if os.environ.get("PUSHT_GOAL_IMAGE"):
        cfg.pusht_reward_goal_image = os.environ["PUSHT_GOAL_IMAGE"]
    if os.environ.get("ENPIRE_YAM_STATION"):
        cfg.station = os.environ["ENPIRE_YAM_STATION"]
    return cfg


def main(cfg: PushTConfig) -> None:
    ctx = build_pusht_context(cfg)
    if cfg.pusht_start_on_launch:
        ctx.external_event_queue.put(("start", {}))
        print("[PushT] start-on-launch queued 'start' event", flush=True)
    should_auto_reset = run(ctx)
    if not should_auto_reset:
        return
    raise SystemExit(PUSHT_EXIT_RESET_REQUESTED)


if __name__ == "__main__":
    _config_file = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config-file" and i + 1 < len(sys.argv):
            _config_file = sys.argv[i + 1]
            break
        if arg.startswith("--config-file="):
            _config_file = arg.split("=", 1)[1]
            break
    if _config_file:
        print(f"[INFO] Loading PushT config from: {_config_file}")
        _default = load_pusht_yaml_defaults(_config_file)
    else:
        _default = PushTConfig()
    main(tyro.cli(PushTConfig, default=_default))
