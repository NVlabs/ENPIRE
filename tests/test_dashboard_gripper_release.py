# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TUI's manual gripper release.

Without this control an operator whose script has clamped an object has no way
to let go: SPACE only freezes motion and X goes home still holding it.
"""

from __future__ import annotations

import time

import pytest

from enpire.env.forge.cap.env.real_bimanual_yam import dashboard as dash_mod
from enpire.env.forge.cap.env.real_bimanual_yam.dashboard import YamDashboard
from enpire.env.forge.cap.env.real_bimanual_yam.skills import (
    _pause_requested,
    _stop_requested,
)


@pytest.fixture(autouse=True)
def _clean_flags():
    """These module-level Events are shared process-wide; reset around each test."""
    _pause_requested.clear()
    _stop_requested.clear()
    yield
    _pause_requested.clear()
    _stop_requested.clear()


class _RecordingEnv:
    """Stands in for RealYamEnv, capturing what set_gripper_direct is told."""

    def __init__(self, fail_side: str | None = None) -> None:
        self.calls: list[tuple[str, float]] = []
        self._fail_side = fail_side

    def set_gripper(self, side, pos, timeout, vel_limit, torque_limit):
        if side == self._fail_side:
            raise RuntimeError("CAN write failed")
        self.calls.append((side, pos))
        return {"success": True}


def _drain(dashboard: YamDashboard, timeout_s: float = 5.0) -> None:
    """Wait for the release worker thread to finish."""
    deadline = time.time() + timeout_s
    while dashboard._release_active.is_set() and time.time() < deadline:
        time.sleep(0.01)
    assert not dashboard._release_active.is_set(), "release thread did not finish"


def test_o_opens_both_grippers_and_pauses_mid_script():
    env = _RecordingEnv()
    dashboard = YamDashboard(env, lambda: None)

    dashboard._handle_key("o")
    _drain(dashboard)

    # Both arms released, fully open.
    assert dashboard._env.calls == [("left", 1.0), ("right", 1.0)]
    # Paused, or the script thread's next command frame would re-close the
    # gripper before the operator could take the object.
    assert _pause_requested.is_set()
    # Release is not an abort: the script must remain resumable with SPACE.
    assert not _stop_requested.is_set()


def test_o_after_done_opens_without_pausing():
    env = _RecordingEnv()
    dashboard = YamDashboard(env, lambda: None)
    dashboard._done_flag.set()

    dashboard._handle_key("o")
    _drain(dashboard)

    assert dashboard._env.calls == [("left", 1.0), ("right", 1.0)]
    assert not _pause_requested.is_set()
    # Must not exit; the operator may want to release before choosing ENTER/S.
    assert not dashboard._exit_event.is_set()


def test_release_survives_one_arm_failing():
    """A dead left arm must not stop the right one from letting go."""
    env = _RecordingEnv(fail_side="left")
    dashboard = YamDashboard(env, lambda: None)

    dashboard._handle_key("o")
    _drain(dashboard)

    assert ("right", 1.0) in dashboard._env.calls
    assert any("FAILED" in line for line in dashboard._stdout_lines)


def test_repeat_keypress_does_not_stack_release_threads(monkeypatch):
    """Holding O must not spawn a thread per repeat, all fighting the same motor."""
    started = []
    release_gate = {"open": False}

    def slow_set_gripper(env, side, pos, **kwargs):
        started.append(side)
        while not release_gate["open"]:
            time.sleep(0.01)
        return {"success": True}

    monkeypatch.setattr(dash_mod, "_pause_requested", _pause_requested)
    monkeypatch.setattr(
        "enpire.env.forge.cap.agent.tools.native.set_gripper_direct", slow_set_gripper
    )

    dashboard = YamDashboard(_RecordingEnv(), lambda: None)
    dashboard._handle_key("o")
    while not started:
        time.sleep(0.01)
    for _ in range(5):
        dashboard._handle_key("o")  # ignored while the first release runs

    release_gate["open"] = True
    _drain(dashboard)

    assert started == ["left", "right"], f"release ran more than once: {started}"


def test_space_still_toggles_pause():
    """The new key must not disturb the existing controls."""
    dashboard = YamDashboard(_RecordingEnv(), lambda: None)

    dashboard._handle_key(" ")
    assert _pause_requested.is_set()
    dashboard._handle_key(" ")
    assert not _pause_requested.is_set()


def test_x_still_stops():
    dashboard = YamDashboard(_RecordingEnv(), lambda: None)

    dashboard._handle_key("x")
    assert _stop_requested.is_set()
