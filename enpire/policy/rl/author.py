# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from enpire.policy.rl.context import RLContext
from enpire.policy.rl.handlers import apply_collision_filter, record_action_delta
from enpire.env.forge.robot.yam.kinematics import _rot6d_to_rot_matrix


AUTHOR_INITIAL_POSITIONS_PATH = Path(
    "tmux/realworld_rl/tasks_config/pin_insertion/initial_position.yaml"
)


def enter_author(ctx: RLContext) -> None:
    ctx.initial_pose_manager.select_first()
    ctx.policy_router.rl_policy.reset()
    ctx.obs, _ = ctx.env.reset(
        options={
            "target_ee_pose": ctx.initial_pose_manager.build_center_pose(),
            "discard_episode": True,
        }
    )
    ctx.author_left_arm_mode = "gravity"
    ctx.author_saved_initial_positions = []
    _set_author_left_arm_mode(ctx, "gravity")
    print(
        "[INFO] Author mode: first initial position. "
        "Arrows/PgUp/PgDn=right arm, Tab toggles left position/gravity, "
        "Enter=save, Backspace=undo, H=write+home, Esc=discard+home.",
        flush=True,
    )


def exit_author(ctx: RLContext, *, write_positions: bool) -> None:
    _set_command_enabled_sides(ctx, {"left", "right"})
    if write_positions:
        write_author_initial_positions(ctx)
    else:
        clear_author_initial_positions(ctx)


def do_author(ctx: RLContext, fello_action: dict, keyboard_info: dict) -> None:
    just = keyboard_info.get("_just_pressed_keys", set())
    if "KEY_TAB" in just:
        next_mode = "position" if ctx.author_left_arm_mode == "gravity" else "gravity"
        _set_author_left_arm_mode(ctx, next_mode)

    if ctx.author_left_arm_mode == "gravity":
        _left_arm_gravity_comp(ctx)
        _set_command_enabled_sides(ctx, {"right"})
    else:
        _set_command_enabled_sides(ctx, {"left", "right"})

    action = dict(fello_action)
    if "right_ee_pos" in action:
        action["right_ee_pos"] = _right_arm_keyboard_delta(ctx, keyboard_info)
    action["source"] = "human"
    action = apply_collision_filter(ctx, action)
    ctx.action_clipped = ctx.collision_filter.clipped
    record_action_delta(ctx, action)
    ctx.obs, _, _, _, _ = ctx.env.step(action)

    if ctx.author_left_arm_mode == "gravity":
        _left_arm_gravity_comp(ctx)

    if "KEY_ENTER" in just:
        save_current_author_pose(ctx)
    if "KEY_BACKSPACE" in just:
        discard_last_author_pose(ctx)


def save_current_author_pose(ctx: RLContext) -> None:
    saved = _author_saved_positions(ctx)
    pose_entry: dict = {}
    for side in ("left", "right"):
        pos = ctx.obs.get(f"{side}_ee_pos")
        rot6d = ctx.obs.get(f"{side}_ee_rot6d")
        gripper = ctx.obs.get(f"{side}_gripper_pos")
        if pos is None:
            continue
        pose_entry[side] = {
            "position": np.asarray(pos, dtype=np.float64).reshape(3).tolist(),
            "rpy_deg": _author_rpy_deg(side, rot6d),
            "gripper_pos": np.asarray(
                [0.0] if gripper is None else gripper, dtype=np.float64
            )
            .reshape(-1)[:1]
            .tolist(),
        }
    if not pose_entry:
        print("[author] No EE pose in observation; nothing saved.", flush=True)
        return
    saved.append(pose_entry)
    n = len(saved)
    pos_strs = " ".join(
        f"{side}={np.round(pose_entry[side]['position'], 4)}"
        for side in ("left", "right")
        if side in pose_entry
    )
    print(f"\033[1;32m[author] Saved position {n}: {pos_strs}\033[0m", flush=True)


def discard_last_author_pose(ctx: RLContext) -> None:
    saved = _author_saved_positions(ctx)
    if saved:
        saved.pop()
        print(
            f"\033[1;33m[author] Discarded last position. {len(saved)} remaining.\033[0m",
            flush=True,
        )
    else:
        print("[author] No saved positions to discard.", flush=True)


def write_author_initial_positions(ctx: RLContext) -> None:
    saved = _author_saved_positions(ctx)
    if not saved:
        print("[author] No positions saved; yaml not written.", flush=True)
        return
    out_path = AUTHOR_INITIAL_POSITIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_entries = {f"position_{i + 1}": entry for i, entry in enumerate(saved)}
    station = ctx.cfg.station
    if station:
        existing: dict = {}
        if out_path.exists():
            with out_path.open() as f:
                existing = yaml.safe_load(f) or {}
        existing[station] = new_entries
        data = existing
    else:
        data = new_entries
    with out_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
    station_note = f" (station: {station})" if station else ""
    print(
        f"\033[1;36m[author] Wrote {len(saved)} position(s) to {out_path}{station_note}\033[0m",
        flush=True,
    )
    saved.clear()


def clear_author_initial_positions(ctx: RLContext) -> None:
    saved = _author_saved_positions(ctx)
    if saved:
        saved.clear()
        print("[author] Discarded saved positions; yaml not written.", flush=True)


def _author_saved_positions(ctx: RLContext) -> list[dict]:
    if ctx.author_saved_initial_positions is None:
        ctx.author_saved_initial_positions = []
    return ctx.author_saved_initial_positions


def _author_rpy_deg(side: str, rot6d: object) -> list[float]:
    if side == "right":
        return [180.0, 0.0, -90.0]
    if rot6d is None:
        return [0.0, 0.0, 0.0]
    return (
        Rotation.from_matrix(_rot6d_to_rot_matrix(rot6d))
        .as_euler("xyz", degrees=True)
        .tolist()
    )


def _right_arm_keyboard_delta(ctx: RLContext, keyboard_info: dict) -> np.ndarray:
    dx_max, dy_max, dz_max = ctx.cfg.delta_ee_translation_xyz_max
    ee_pos = np.zeros(3, dtype=np.float32)
    if keyboard_info["left_held"]:
        ee_pos[0] = -dx_max
    elif keyboard_info["right_held"]:
        ee_pos[0] = dx_max
    if keyboard_info["down_held"]:
        ee_pos[1] = -dy_max
    elif keyboard_info["up_held"]:
        ee_pos[1] = dy_max
    if keyboard_info["pagedown_held"]:
        ee_pos[2] = -dz_max
    elif keyboard_info["pageup_held"]:
        ee_pos[2] = dz_max
    return ee_pos


def _set_author_left_arm_mode(ctx: RLContext, mode: str) -> None:
    if mode not in ("gravity", "position"):
        raise ValueError(f"Unsupported author left arm mode: {mode!r}")
    ctx.author_left_arm_mode = mode
    if mode == "gravity":
        _set_command_enabled_sides(ctx, {"right"})
        _left_arm_gravity_comp(ctx)
    else:
        _set_command_enabled_sides(ctx, {"left", "right"})
    print(f"[INFO] Author left arm mode: {mode}", flush=True)


def _set_command_enabled_sides(ctx: RLContext, sides: set[str]) -> None:
    env = ctx.env.unwrapped
    if hasattr(env, "set_command_enabled_sides"):
        env.set_command_enabled_sides(sides)


def _left_arm_gravity_comp(ctx: RLContext) -> None:
    env = ctx.env.unwrapped
    follower_arms = getattr(env, "follower_arms", {})
    left_arm = follower_arms.get("left")
    if left_arm is None:
        return
    left_arm.command_joint_state(
        {
            "pos": np.zeros(7, dtype=np.float32),
            "vel": np.zeros(7, dtype=np.float32),
            "kp": np.zeros(7, dtype=np.float32),
            "kd": np.zeros(7, dtype=np.float32),
        }
    )

