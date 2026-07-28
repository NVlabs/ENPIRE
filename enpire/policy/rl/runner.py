# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402  (imports must follow the sys.path / bootstrap block)
from __future__ import annotations

import sys

from enpire.env.forge.paths import REPOSITORY_ROOT

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from enpire.env.forge.tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPOSITORY_ROOT, required_modules=["gymnasium", "tyro"])

sys.stdout.reconfigure(line_buffering=True)

import tyro

from enpire.policy.rl.author import do_author, enter_author, exit_author
from enpire.policy.rl.auto_eval import AutoEvalController
from enpire.policy.rl.config import DataCollectionConfig, load_yaml_defaults
from enpire.policy.rl.context import RLContext, build_context
from enpire.policy.rl.events import TERMINAL_EVENTS
from enpire.policy.rl.handlers import (
    do_change_pose,
    do_home,
    do_hover,
    do_learn,
    do_parking,
    handle_restart,
    render,
)
from enpire.policy.rl.keyboard_events import dispatch_keyboard_events
from enpire.policy.rl.speech_announcer import TERMINAL_EVENT_PHRASES


def run(ctx: RLContext) -> None:
    auto_eval = AutoEvalController()
    ctx.timing_log.log("runner_start")
    try:
        while True:
            ctx.action_clipped = False
            fello_action, fello_info, _ = ctx.policy_router.get_fello_action(ctx.obs)
            _, spacemouse_info = ctx.policy_router.refresh_spacemouse_action(ctx.obs)
            _, keyboard_info = ctx.keyboard_policy.get_action()
            dispatch_keyboard_events(ctx, keyboard_info)

            event, payload = ctx.event_router.get_event(
                fello_info,
                ctx.obs,
                is_learning=(ctx.state_machine.state == "learn"),
                episode_step_count=getattr(ctx.env, "frame_count", None),
                spacemouse_info=spacemouse_info,
            )
            if event == "home":
                print(
                    "[RL] home event accepted: "
                    f"state={ctx.state_machine.state} "
                    f"source={payload.get('source', 'unknown')} "
                    f"client={payload.get('client', '-')}",
                    flush=True,
                )
                ctx.timing_log.log(
                    "home_event",
                    source=payload.get("source", "unknown"),
                    client=payload.get("client", "-"),
                )
            if event == "start":
                print(
                    "[RL] start event accepted: "
                    f"state={ctx.state_machine.state} "
                    f"source={payload.get('source', 'unknown')} "
                    f"client={payload.get('client', '-')}",
                    flush=True,
                )
                ctx.timing_log.log(
                    "start_event",
                    source=payload.get("source", "unknown"),
                    client=payload.get("client", "-"),
                )
            if event in ("next_pose", "prev_pose", "set_initial_position_index"):
                ctx.pending_pose_event = event
                ctx.pending_pose_payload = payload
            if event in TERMINAL_EVENTS:
                ctx.terminal_event = event
                ctx.last_terminal_event = event
                if ctx.state_machine.state == "learn":
                    ctx.timing_log.log("terminal", signal=event)
                    ctx.speech_announcer.speak(TERMINAL_EVENT_PHRASES[event])
                    ctx.demo_total_count += 1
                    is_success = event == "success"
                    if is_success:
                        ctx.demo_success_count += 1
                    ctx.demo_rolling_window.append(is_success)
            if event == "restart":
                handle_restart(ctx, payload)
            if event == "auto_eval_start":
                auto_eval.start(ctx)

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
                do_hover(ctx)
            elif state == "learn":
                do_learn(ctx, fello_action, fello_info, keyboard_info)

            render(ctx)
            auto_eval.observe(ctx)
    finally:
        ctx.policy_router.close()
        ctx.env.close()


def main(cfg: DataCollectionConfig) -> None:
    run(build_context(cfg))


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
        print(f"[INFO] Loading config from: {_config_file}")
        _default = load_yaml_defaults(_config_file)
    else:
        _default = DataCollectionConfig()
    main(tyro.cli(DataCollectionConfig, default=_default))
