# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from enpire.env.forge.paths import FORGE_ROOT
from enpire.env.forge.robot.constants import DEFAULT_RESET_JOINT_STATE
from enpire.policy.rl.context import RLContext
from enpire.policy.rl.display import render_frame
from enpire.policy.rl.gpu_slot_hover import (
    check_gpu_slot_hover_dependencies,
    move_to_gpu_slot_hover,
)
from enpire.policy.rl.reset_options import terminal_label_options

HOME_POSITION_ATOL = 0.2


def handle_restart(ctx: RLContext, payload: dict) -> None:
    new_path = Path(payload["path"])
    ctx.env.set_output_dir(new_path)
    if ctx.handshake_server is not None:
        ctx.handshake_server.set_path(new_path)
    ctx.timing_log.update_dir(new_path)
    ctx.demo_success_count = 0
    ctx.demo_total_count = 0
    ctx.demo_rolling_window.clear()


def do_home(ctx: RLContext) -> None:
    cfg = getattr(ctx, "cfg", None)
    if getattr(cfg, "home_event_reset_target", "home") == "hover":
        ctx.speech_announcer.speak("hover")
        ctx.policy_router.rl_policy.reset()
        if _uses_gpu_slot_hover_reset(ctx):
            check_gpu_slot_hover_dependencies(ctx.cfg)
        lifted = _maybe_lift_before_reset(
            ctx,
            reason="home-to-hover",
            first_reset_options={"discard_episode": True},
        )
        if _uses_gpu_slot_hover_reset(ctx):
            if not lifted:
                ctx.obs, _ = ctx.env.reset(
                    options={"alias": "current", "discard_episode": True}
                )
            move_to_gpu_slot_hover(ctx)
            ctx.obs, _ = ctx.env.reset(
                options={"alias": "current", "discard_episode": True}
            )
            _maybe_rebase_current_observation(ctx)
        else:
            ctx.obs, _ = ctx.env.reset(
                options={
                    "target_ee_pose": ctx.initial_pose_manager.build_center_pose(),
                    "discard_episode": True,
                }
            )
        ctx.event_router.reset_timer()
        ctx.terminal_event = None
        ctx.timing_log.log("hover_home_done")
        ctx.state_machine.state = "idle"
        return

    ctx.speech_announcer.speak("home")
    if not _is_observation_at_home(ctx.obs):
        ctx.obs, _ = ctx.env.reset(
            options={
                "target_ee_pose": ctx.initial_pose_manager.build_center_pose(),
                "discard_episode": True,
            }
        )
    ctx.obs, _ = ctx.env.reset(options={"alias": "home", "discard_episode": True})
    ctx.terminal_event = None
    ctx.timing_log.log("home_done")
    ctx.state_machine.state = "idle"


def _is_observation_at_home(obs: dict) -> bool:
    for key, target in DEFAULT_RESET_JOINT_STATE.items():
        if key not in obs:
            return False
        if not np.allclose(obs[key], target, atol=HOME_POSITION_ATOL):
            return False
    return True


def do_change_pose(ctx: RLContext) -> None:
    pause_pose = ctx.initial_pose_manager.build_center_pose()
    if ctx.pending_pose_event == "next_pose":
        ctx.initial_pose_manager.select_next()
    elif ctx.pending_pose_event == "prev_pose":
        ctx.initial_pose_manager.select_previous()
    elif ctx.pending_pose_event == "set_initial_position_index":
        ctx.initial_pose_manager.select_initial_position_index(
            int(ctx.pending_pose_payload["initial_position_index"])
        )
    ctx.pending_pose_event = None
    ctx.pending_pose_payload = {}

    print(f"[INFO] Change pose -> {ctx.initial_pose_manager.describe_current()}")
    ctx.policy_router.rl_policy.reset()
    ctx.obs, _ = ctx.env.reset(
        options={
            "target_ee_pose": pause_pose,
            "discard_episode": True,
        }
    )
    ctx.obs, _ = ctx.env.reset(options={"alias": "home", "discard_episode": True})
    ctx.obs, _ = ctx.env.reset(
        options={
            "target_ee_pose": ctx.initial_pose_manager.build_center_pose(),
            "discard_episode": True,
        }
    )
    ctx.terminal_event = None
    ctx.state_machine.state = "idle"


def do_parking(ctx: RLContext, event: str | None) -> None:
    if event == "next_pose":
        ctx.initial_pose_manager.select_next()
        _reset_to_parking_center(ctx)
    elif event == "prev_pose":
        ctx.initial_pose_manager.select_previous()
        _reset_to_parking_center(ctx)
    elif event == "set_initial_position_index":
        ctx.initial_pose_manager.select_initial_position_index(
            int(ctx.event_router.last_payload["initial_position_index"])
        )
        _reset_to_parking_center(ctx)
    elif event == "init_boundary":
        ctx.parking_navigator.select_initial_boundary()
        _reset_to_boundary_corner(ctx)
    elif event == "oor_boundary":
        ctx.parking_navigator.select_oor_boundary()
        _reset_to_boundary_corner(ctx)
    elif event == "z_high":
        ctx.parking_navigator.select_z_high()
        print("[INFO] Parking boundary z-high selected", flush=True)
    elif event == "z_low":
        ctx.parking_navigator.select_z_low()
        print("[INFO] Parking boundary z-low selected", flush=True)
    elif event == "parking":
        _reset_to_parking_center(ctx)
    ctx.terminal_event = None


def _reset_to_parking_center(ctx: RLContext) -> None:
    print(f"[INFO] Parking -> {ctx.initial_pose_manager.describe_current()}")
    ctx.policy_router.rl_policy.reset()
    ctx.obs, _ = ctx.env.reset(
        options={
            "target_ee_pose": ctx.initial_pose_manager.build_center_pose(),
            "discard_episode": True,
        }
    )


def _reset_to_boundary_corner(ctx: RLContext) -> None:
    offset = ctx.initial_pose_manager.boundary_corner_offset(
        kind=ctx.parking_navigator.boundary_kind,
        corner_idx=ctx.parking_navigator.corner_idx,
        z_idx=ctx.parking_navigator.z_idx,
    )
    pose, label = ctx.parking_navigator.next_corner_pose(ctx.initial_pose_manager)
    print(f"[INFO] Parking {label}", flush=True)
    print(f"[INFO] Parking offset {np.round(offset, 4)}", flush=True)
    for side in sorted(pose):
        print(
            f"[INFO] Parking reset {side} position "
            f"{np.round(np.asarray(pose[side]['position'], dtype=float), 4)}",
            flush=True,
        )
    ctx.policy_router.rl_policy.reset()
    ctx.obs, _ = ctx.env.reset(
        options={
            "target_ee_pose": pose,
            "discard_episode": True,
        }
    )


def _load_right_arm_hover_mover():
    """Load ``move_right_arm_to_hover`` from the canonical gpu saved script.

    The pose itself lives in ``cap/saved_scripts/gpu/right_arm_hover.py`` so that
    data collection (this runner) and full-loop inference (which runs the same
    runner) share one definition. Importing the module is a no-op on the robot.
    """
    import importlib.util

    script_path = (
        FORGE_ROOT
        / "cap"
        / "saved_scripts"
        / "gpu"
        / "right_arm_hover.py"
    )
    spec = importlib.util.spec_from_file_location("gpu_right_arm_hover", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.move_right_arm_to_hover


def _maybe_move_right_to_hover(ctx: RLContext) -> None:
    """Drive the right arm to the canonical GPU-insertion hover pose.

    No-op unless ``cfg.enable_right_arm_hover`` is set. The pose is defined once
    in ``cap/saved_scripts/gpu/right_arm_hover.py`` and applied via the env's
    direct joint-interpolation API, which bypasses the recording wrapper (so it
    never finalizes/discards the active episode).
    """
    if not getattr(ctx.cfg, "enable_right_arm_hover", False):
        return
    env = getattr(ctx.env, "unwrapped", ctx.env)
    move_right_arm_to_hover = _load_right_arm_hover_mover()
    move_right_arm_to_hover(env)


def do_hover(ctx: RLContext) -> None:
    ctx.timing_log.log("reset_start")
    _maybe_move_right_to_hover(ctx)
    ctx.policy_router.rl_policy.reset()
    terminal_options = terminal_label_options(ctx.terminal_event)
    if _uses_gpu_slot_hover_reset(ctx):
        check_gpu_slot_hover_dependencies(ctx.cfg)
    if ctx.terminal_event is not None and _maybe_lift_before_reset(
        ctx,
        reason=f"terminal-{ctx.terminal_event}",
        first_reset_options=terminal_options,
    ):
        terminal_options = {}
    if _uses_gpu_slot_hover_reset(ctx):
        if ctx.terminal_event is not None and terminal_options:
            ctx.obs, _ = ctx.env.reset(options={"alias": "current", **terminal_options})
            terminal_options = {}
        move_to_gpu_slot_hover(ctx)
        if ctx.cfg.randomize_initial_pose:
            ctx.obs, _ = ctx.env.reset(options={"alias": "current"})
            _maybe_rebase_current_observation(ctx)
            start_pose = ctx.initial_pose_manager.build_episode_start_pose()
            opts = {
                "target_ee_pose": start_pose,
                "task_name": ctx.cfg.task_name,
                "start_new_episode": True,
            }
            print(
                f"[INFO] GPU hover randomized start from "
                f"{ctx.initial_pose_manager.describe_current()} "
                f"offset={np.round(ctx.initial_pose_manager.last_offset, 4)}",
                flush=True,
            )
        else:
            opts = {
                "alias": "current",
                "task_name": ctx.cfg.task_name,
                "start_new_episode": True,
            }
        opts.update(terminal_options)
        ctx.obs, _ = ctx.env.reset(options=opts)
        _maybe_rebase_current_observation(ctx)
        _flush_pending_keyboard_terminal_labels(ctx)
        ctx.event_router.reset_timer()
        ctx.terminal_event = None
        ctx.speech_announcer.speak("learn")
        ctx.timing_log.log("learn_start")
        ctx.state_machine.state = "learn"
        return

    opts = {
        "target_ee_pose": ctx.initial_pose_manager.build_episode_start_pose(),
        "task_name": ctx.cfg.task_name,
        "start_new_episode": True,
    }
    opts.update(terminal_options)
    print(
        f"[INFO] Episode start from {ctx.initial_pose_manager.describe_current()} "
        f"offset={np.round(ctx.initial_pose_manager.last_offset, 4)}",
        flush=True,
    )
    ctx.obs, _ = ctx.env.reset(options=opts)
    _maybe_rebase_current_observation(ctx)
    _flush_pending_keyboard_terminal_labels(ctx)
    ctx.event_router.reset_timer()
    ctx.terminal_event = None
    ctx.speech_announcer.speak("learn")
    ctx.timing_log.log("learn_start")
    ctx.state_machine.state = "learn"


def _uses_gpu_slot_hover_reset(ctx: RLContext) -> bool:
    return getattr(ctx.cfg, "episode_reset_strategy", "target_pose") == "gpu_slot_hover"


def _maybe_rebase_current_observation(ctx: RLContext) -> None:
    if ctx.cfg.initial_pose_source != "current_observation":
        return
    ctx.initial_pose_manager.set_base_pose_from_observation(ctx.obs)
    print(
        "[INFO] Rebased episode hover anchor from current observation: "
        f"{ctx.initial_pose_manager.describe_current()}",
        flush=True,
    )


def _flush_pending_keyboard_terminal_labels(ctx: RLContext) -> None:
    keyboard = getattr(ctx, "keyboard_policy", None)
    discard = getattr(keyboard, "discard_just_pressed", None)
    if discard is None:
        return
    keys = {
        str(getattr(ctx.cfg, "keyboard_success_key", "KEY_ENTER") or "").strip(),
        str(getattr(ctx.cfg, "keyboard_fail_key", "KEY_BACKSPACE") or "").strip(),
    }
    keys.discard("")
    if not keys:
        return
    discarded = discard(keys)
    if discarded:
        print(
            "[keyboard] flushed stale terminal label key(s) before learn: "
            f"{sorted(discarded)}",
            flush=True,
        )


def _maybe_lift_before_reset(
    ctx: RLContext,
    *,
    reason: str,
    first_reset_options: dict,
) -> bool:
    lift_m = float(getattr(ctx.cfg, "episode_reset_lift_m", 0.0) or 0.0)
    if lift_m <= 0.0:
        return False
    lift_pose = ctx.initial_pose_manager.pose_from_observation(
        ctx.obs,
        z_offset_m=lift_m,
    )
    for side, side_pose in sorted(lift_pose.items()):
        print(
            f"[INFO] Reset lift ({reason}) {side}: "
            f"target_z={float(side_pose['position'][2]):.4f} "
            f"lift_m={lift_m:.4f}",
            flush=True,
        )
    options = {"target_ee_pose": lift_pose}
    options.update(first_reset_options)
    ctx.obs, _ = ctx.env.reset(options=options)
    return True


def do_learn(
    ctx: RLContext,
    fello_action: dict,
    fello_info: dict,
    keyboard_info: dict,
) -> None:
    action, _ = ctx.policy_router.route_action(
        ctx.obs,
        fello_action,
        fello_info,
        use_rl=True,
        keyboard_info=keyboard_info,
    )
    action = apply_collision_filter(ctx, action)
    ctx.action_clipped = ctx.collision_filter.clipped
    record_action_delta(ctx, action)
    ctx.obs, _, _, _, _ = ctx.env.step(action)


def record_action_delta(ctx: RLContext, action: dict) -> None:
    side = _primary_control_side(ctx)
    delta = action.get(f"{side}_ee_pos")
    if delta is None:
        return
    d = np.asarray(delta, dtype=float).reshape(-1)
    if d.size < 3:
        return
    now = time.perf_counter()
    duration = max(float(ctx.cfg.action_trail_duration_s), 0.0)
    ctx.action_trail.append((now, tuple(float(v) for v in d[:3])))
    cutoff = now - duration
    ctx.action_trail[:] = [(t, xyz) for t, xyz in ctx.action_trail if t >= cutoff]


def apply_collision_filter(ctx: RLContext, action: dict) -> dict:
    side = _primary_control_side(ctx)
    ee_pos = ctx.obs.get(f"{side}_ee_pos")
    if ee_pos is None:
        ctx.collision_filter.clipped = False
        return action
    z = float(np.asarray(ee_pos, dtype=float).reshape(3)[2])
    if z >= ctx.cfg.collision_filter_z_threshold:
        ctx.collision_filter.clipped = False
        return action
    return ctx.collision_filter.apply_filter(
        ctx.obs.get(f"{side}_eef_force"),
        ctx.cfg.right_arm_z_force_limit,
        action,
        side=side,
    )


def _primary_control_side(ctx: RLContext) -> str:
    if ctx.cfg.enabled_sides in ("left", "right"):
        return ctx.cfg.enabled_sides
    return "right"


def _pose_label(ctx: RLContext) -> str | None:
    label = ctx.initial_pose_manager.label
    if ctx.cfg.station:
        return f"{ctx.cfg.station}  {label}" if label else ctx.cfg.station
    return label


def render(ctx: RLContext) -> None:
    if not ctx.cfg.display_image:
        return
    demo_counter = (ctx.demo_success_count, ctx.demo_total_count)
    rolling = ctx.demo_rolling_window
    demo_rolling = (sum(rolling), rolling.maxlen)
    render_frame(
        ctx.img_queue,
        ctx.state_machine.state,
        ctx.obs,
        ctx.terminal_event,
        ctx.last_terminal_event,
        ctx.action_clipped,
        None,
        ctx.cfg,
        hover_pose_label=_pose_label(ctx),
        action_trail=(
            ctx.action_trail if ctx.state_machine.state in ("author", "learn") else None
        ),
        demo_counter=demo_counter,
        demo_rolling=demo_rolling,
    )
