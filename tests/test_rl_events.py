# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import time

import numpy as np

from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.events import RLEventRouter


class _PoseManager:
    def __init__(self, out_of_range=False):
        self.out_of_range = out_of_range

    def is_out_of_range(self, obs):
        return self.out_of_range


def _fello_info(buttons=None):
    if buttons is None:
        buttons = [0.0, 0.0, 0.0]
    return {
        "right_buttons": np.asarray(buttons, dtype=np.float32),
        "left_buttons": np.zeros(3, dtype=np.float32),
    }


def test_external_event_has_priority():
    events = queue.Queue()
    events.put(("start", {"source": "test"}))
    router = RLEventRouter(DataCollectionConfig(), _PoseManager(True), events)

    assert router.get_event(_fello_info([0, 0, 1]), {}, is_learning=True) == (
        "start",
        {"source": "test"},
    )


def test_home_event_can_be_disabled():
    events = queue.Queue()
    events.put(("home", {"source": "fastapi", "client": "127.0.0.1"}))
    router = RLEventRouter(
        DataCollectionConfig(accept_home_events=False),
        _PoseManager(False),
        events,
    )

    assert router.get_event(_fello_info(), {}, is_learning=False) == (None, {})


def test_external_terminal_events_only_fire_while_learning():
    events = queue.Queue()
    events.put(("success", {"source": "keyboard"}))
    router = RLEventRouter(DataCollectionConfig(), _PoseManager(False), events)

    assert router.get_event(_fello_info(), {}, is_learning=False) == (None, {})

    events.put(("fail", {"source": "keyboard"}))
    assert router.get_event(_fello_info(), {}, is_learning=True) == (
        "fail",
        {"source": "keyboard"},
    )


def test_terminal_events_only_fire_while_learning():
    router = RLEventRouter(DataCollectionConfig(), _PoseManager(True), queue.Queue())

    assert router.get_event(_fello_info([0, 0, 1]), {}, is_learning=False) == (
        None,
        {},
    )
    assert router.get_event(_fello_info([0, 0, 1]), {}, is_learning=True) == (
        "success",
        {},
    )


def test_spacemouse_buttons_can_label_terminal_events():
    router = RLEventRouter(DataCollectionConfig(), _PoseManager(False), queue.Queue())

    assert router.get_event(
        _fello_info(),
        {},
        is_learning=True,
        spacemouse_info={"just_pressed_buttons": (0,)},
    ) == ("success", {"source": "spacemouse", "button": 0})

    assert router.get_event(
        _fello_info(),
        {},
        is_learning=True,
        spacemouse_info={"just_pressed_buttons": (1,)},
    ) == ("fail", {"source": "spacemouse", "button": 1})


def test_auto_reward_oor_and_timeout_events():
    cfg = DataCollectionConfig(
        enable_auto_reward=True,
        auto_reward_z_threshold=0.5,
        enabled_sides="right",
    )
    router = RLEventRouter(cfg, _PoseManager(False), queue.Queue())

    assert router.get_event(
        _fello_info(),
        {"right_ee_pos": np.array([0.0, 0.0, 0.4])},
        is_learning=True,
    ) == ("success", {})

    router = RLEventRouter(
        DataCollectionConfig(enable_oor_check=True),
        _PoseManager(True),
        queue.Queue(),
    )
    assert router.get_event(_fello_info(), {}, is_learning=True) == ("out-of-range", {})

    cfg = DataCollectionConfig(episode_timeout_s=1.0)
    router = RLEventRouter(cfg, _PoseManager(False), queue.Queue())
    router.start_time = time.perf_counter() - 2.0
    assert router.get_event(_fello_info(), {}, is_learning=True) == ("timeout", {})


def test_auto_reward_can_use_relative_z_drop():
    cfg = DataCollectionConfig(
        enable_auto_reward=True,
        auto_reward_z_drop_m=0.012,
        enabled_sides="left",
    )
    pose_manager = _PoseManager(False)
    pose_manager.delta_from_base = lambda obs: {
        "left": np.array([0.0, 0.0, -0.013], dtype=np.float32)
    }
    router = RLEventRouter(cfg, pose_manager, queue.Queue())

    assert router.get_event(_fello_info(), {}, is_learning=True) == ("success", {})


def test_terminal_events_wait_for_min_recorded_steps():
    cfg = DataCollectionConfig(
        enable_auto_reward=True,
        auto_reward_z_drop_m=0.012,
        enabled_sides="left",
        terminal_min_recorded_steps=2,
    )
    pose_manager = _PoseManager(False)
    pose_manager.delta_from_base = lambda obs: {
        "left": np.array([0.0, 0.0, -0.020], dtype=np.float32)
    }
    events = queue.Queue()
    events.put(("success", {"source": "keyboard"}))
    router = RLEventRouter(cfg, pose_manager, events)

    assert router.get_event(
        _fello_info(),
        {},
        is_learning=True,
        episode_step_count=1,
    ) == (None, {})
    assert router.get_event(
        _fello_info(),
        {},
        is_learning=True,
        episode_step_count=1,
    ) == (None, {})
    assert router.get_event(
        _fello_info(),
        {},
        is_learning=True,
        episode_step_count=2,
    ) == ("success", {})
