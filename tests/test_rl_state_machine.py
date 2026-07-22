# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enpire.policy.rl.state_machine import RLStateMachine


def test_start_hover_learn_terminal_cycle():
    sm = RLStateMachine()

    sm.transition("start")
    assert sm.state == "hover"

    sm.transition(None)
    assert sm.state == "learn"

    sm.transition("success")
    assert sm.state == "hover"


def test_pose_change_home_and_restart_events():
    sm = RLStateMachine("learn")

    sm.transition("next_pose")
    assert sm.state == "change_pose"

    sm = RLStateMachine("learn")
    sm.transition("set_initial_position_index")
    assert sm.state == "change_pose"

    sm = RLStateMachine("learn")
    sm.transition("home")
    assert sm.state == "home"

    sm = RLStateMachine("learn")
    sm.transition("restart")
    assert sm.state == "learn"


def test_parking_transitions():
    sm = RLStateMachine("learn")

    sm.transition("parking")
    assert sm.state == "parking"

    sm.transition("init_boundary")
    assert sm.state == "parking"

    sm.transition("next_pose")
    assert sm.state == "parking"

    sm.transition("start")
    assert sm.state == "hover"


def test_author_is_global_mode_and_home_exits():
    sm = RLStateMachine("learn")

    sm.transition("author")
    assert sm.state == "author"

    sm.transition("home")
    assert sm.state == "home"


def test_author_discard_exits_home():
    sm = RLStateMachine("learn")

    sm.transition("author")
    sm.transition("discard_author")

    assert sm.state == "home"
