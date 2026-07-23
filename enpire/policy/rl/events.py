# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import time

import numpy as np

from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.initial_pose_manager import InitialPoseManager

LEFT_TAKEOVER_BUTTON = 1
RIGHT_TAKEOVER_BUTTON = 0
SUCCESS_EVENT_BUTTON = 2
FAIL_EVENT_BUTTON = 1
Z_UP_KEY = "z"

TERMINAL_REWARD_BY_EVENT = {
    "success": 1.0,
    "fail": 0.0,
    "timeout": 0.0,
    "out-of-range": 0.0,
}
TERMINAL_EVENTS = frozenset(TERMINAL_REWARD_BY_EVENT)
EXTERNAL_EVENTS = {
    "restart",
    "home",
    "start",
    "parking",
    "next_pose",
    "prev_pose",
    "set_initial_position_index",
    "init_boundary",
    "oor_boundary",
    "z_high",
    "z_low",
    "author",
    "discard_author",
    "auto_eval_start",
}


class RLEventRouter:
    """Converts external controls, Fello buttons, and checks into RL events."""

    def __init__(
        self,
        cfg: DataCollectionConfig,
        initial_pose_manager: InitialPoseManager,
        external_event_queue: "queue.Queue[tuple[str, dict]]",
    ):
        self.cfg = cfg
        self.initial_pose_manager = initial_pose_manager
        self.external_event_queue = external_event_queue
        self.start_time = time.perf_counter()
        self.last_payload: dict = {}
        self._terminal_min_steps_notice: str | None = None

    def reset_timer(self) -> None:
        self.start_time = time.perf_counter()
        self._terminal_min_steps_notice = None

    def get_event(
        self,
        fello_info: dict,
        obs: dict,
        *,
        is_learning: bool,
        episode_step_count: int | None = None,
        spacemouse_info: dict | None = None,
    ) -> tuple[str | None, dict]:
        self.last_payload = {}
        try:
            event, payload = self.external_event_queue.get_nowait()
        except queue.Empty:
            event, payload = None, {}
        if event in TERMINAL_EVENTS:
            if is_learning:
                if not self._terminal_events_allowed(episode_step_count, event):
                    return None, {}
                self.last_payload = payload or {}
                return event, self.last_payload
            return None, {}
        if event == "home" and not bool(getattr(self.cfg, "accept_home_events", True)):
            payload = payload or {}
            print(
                "[RL] Ignoring home event: "
                f"source={payload.get('source', 'unknown')} "
                f"client={payload.get('client', '-')}",
                flush=True,
            )
            return None, {}
        if event in EXTERNAL_EVENTS:
            self.last_payload = payload or {}
            return event, self.last_payload

        if not is_learning:
            return None, {}

        spacemouse_event = self._spacemouse_terminal_event(spacemouse_info)
        if spacemouse_event is not None:
            label, button = spacemouse_event
            if not self._terminal_events_allowed(episode_step_count, label):
                return None, {}
            if label == "success":
                print(
                    "\033[1;32mSuccessful episode saved by SpaceMouse\033[0m",
                    flush=True,
                )
            else:
                print(
                    "\033[1;38;5;88mFailed episode saved by SpaceMouse\033[0m",
                    flush=True,
                )
            return label, {"source": "spacemouse", "button": button}

        if not self._terminal_events_allowed(episode_step_count, "auto"):
            return None, {}

        fello_buttons = fello_info["right_buttons"]
        if fello_buttons[SUCCESS_EVENT_BUTTON]:
            print("\033[1;32mSuccessful episode saved\033[0m", flush=True)
            return "success", {}
        if self._check_auto_reward(obs):
            return "success", {}
        if fello_buttons[FAIL_EVENT_BUTTON]:
            print("\033[1;38;5;88mFailed episode saved\033[0m", flush=True)
            return "fail", {}
        if self.initial_pose_manager.is_out_of_range(obs):
            detail = getattr(self.initial_pose_manager, "last_out_of_range", None)
            if detail:
                print(
                    "\033[1;33mOut-of-range: "
                    f"{detail['side']} {detail['axis']}_delta={detail['delta']:+.4f} "
                    f"outside [{detail['lo']:+.4f}, {detail['hi']:+.4f}]; "
                    "resetting\033[0m",
                    flush=True,
                )
            else:
                print("\033[1;33mOut-of-range: resetting\033[0m", flush=True)
            return "out-of-range", {}
        if time.perf_counter() - self.start_time > self.cfg.episode_timeout_s:
            self.reset_timer()
            return "timeout", {}
        return None, {}

    def _spacemouse_terminal_event(
        self,
        spacemouse_info: dict | None,
    ) -> tuple[str, int] | None:
        if not spacemouse_info:
            return None
        just_pressed = set(int(i) for i in spacemouse_info.get("just_pressed_buttons", ()))
        success_button = getattr(self.cfg, "spacemouse_success_button", None)
        fail_button = getattr(self.cfg, "spacemouse_fail_button", None)
        if success_button is not None and int(success_button) in just_pressed:
            return "success", int(success_button)
        if fail_button is not None and int(fail_button) in just_pressed:
            return "fail", int(fail_button)
        return None

    def _terminal_events_allowed(
        self,
        episode_step_count: int | None,
        event: str,
    ) -> bool:
        min_steps = max(
            0,
            int(getattr(self.cfg, "terminal_min_recorded_steps", 0) or 0),
        )
        if min_steps <= 0 or episode_step_count is None:
            return True
        if int(episode_step_count) >= min_steps:
            return True
        if event != "auto":
            notice = f"{event}:{episode_step_count}:{min_steps}"
            if self._terminal_min_steps_notice != notice:
                print(
                    f"[RL] Ignoring terminal event {event!r}: "
                    f"recorded_steps={int(episode_step_count)} < "
                    f"terminal_min_recorded_steps={min_steps}",
                    flush=True,
                )
                self._terminal_min_steps_notice = notice
        return False

    def _check_auto_reward(self, obs: dict) -> bool:
        if not self.cfg.enable_auto_reward:
            return False
        relative_drop_m = float(getattr(self.cfg, "auto_reward_z_drop_m", 0.0) or 0.0)
        if relative_drop_m > 0.0:
            for side, delta in self.initial_pose_manager.delta_from_base(obs).items():
                z_drop = -float(np.asarray(delta, dtype=np.float64).reshape(3)[2])
                if z_drop >= relative_drop_m:
                    print(
                        f"\033[1;32mAuto-reward: {side} EE z_drop={z_drop:.4f} >= "
                        f"threshold={relative_drop_m:.4f}\033[0m",
                        flush=True,
                    )
                    return True
        enabled = (
            {"left", "right"}
            if self.cfg.enabled_sides == "both"
            else {self.cfg.enabled_sides}
        )
        for side in enabled:
            ee_pos = obs.get(f"{side}_ee_pos")
            if ee_pos is None:
                continue
            z = float(np.asarray(ee_pos, dtype=np.float64).reshape(3)[2])
            if z < self.cfg.auto_reward_z_threshold:
                print(
                    f"\033[1;32mAuto-reward: {side} EE z={z:.4f} < "
                    f"threshold={self.cfg.auto_reward_z_threshold:.4f}\033[0m",
                    flush=True,
                )
                return True
        return False


class CollisionFilter:
    def __init__(self):
        self.clipped = False

    def apply_filter(
        self,
        force,
        force_limit: float,
        action: dict,
        *,
        side: str = "right",
    ) -> dict:
        self.clipped = False
        pos_key = f"{side}_ee_pos"
        if force is None or pos_key not in action:
            return action
        force_z = float(np.asarray(force, dtype=float).reshape(3)[2])
        if force_z < force_limit and action[pos_key][2] < 0:
            action[pos_key][2] = 0
            self.clipped = True
        return action

