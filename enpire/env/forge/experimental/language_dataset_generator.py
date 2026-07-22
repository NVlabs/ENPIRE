# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline language-command control dataset generator for the YAM bimanual robot.

Generates (language_command, action_trajectory) pairs **entirely offline** —
no robot hardware or real-time simulation required.  Each episode couples one
natural-language instruction with a smooth, kinematically feasible joint
trajectory driven by the same ``ScriptedPolicy`` FK/IK mechanics used in the
live Viser UI.

Output — three replay-ready parquet files written under ``output_dir/``:

    joint/episode_XXXX.parquet
        action         : (T, 14) joint_pos + gripper [l_jp(6), l_g(1), r_jp(6), r_g(1)]
        observation.state : (T, 14) same layout, current state before each action
        → replay: --control-mode joint_position

    delta_ee/episode_XXXX.parquet
        action         : (T, 16) delta EE pose [l_dpos(3), l_dq(4), l_g(1), r_dpos(3), r_dq(4), r_g(1)]
        observation.state : (T, 14) joint format (for initial-state calibration)
        → replay: --control-mode delta_ee_pose

    ee_pose/episode_XXXX.parquet
        observation.state : (T, 16) absolute EE pose [l_pos(3), l_q(4), l_g(1), r_pos(3), r_q(4), r_g(1)]
        action            : (T, 16) absolute EE pose (same layout, target for this step)
        → replay: --control-mode ee_pose

Coordinate frame convention (world / robot base frame):
    x = forward  (away from the robot)
    y = left      (robot's own left side)
    z = up

Usage:
    cd ~/Project/lecar-tbd
    uv run python -m experimental.language_dataset_generator
    uv run python -m experimental.language_dataset_generator --output-dir /tmp/lang_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_TBD_ROOT = Path(__file__).resolve().parents[1]
if str(_TBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TBD_ROOT))

from enpire.env.forge.robot.yam.kinematics import YamKinematics
from enpire.env.forge.experimental.scripted_policy import ScriptedPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Robot home configuration (radians, matches kinematics.py test values)
HOME_LEFT_JP = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25], dtype=np.float64)
HOME_RIGHT_JP = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25], dtype=np.float64)
HOME_GRIPPER_OPEN = 1.0  # fully open

# Simulation parameters
SMOOTHING_STEPS = 15      # ScriptedPolicy interpolation steps per nudge
PRE_HOLD = 3              # hold steps before first motion
INTER_HOLD = 3            # hold steps between primitives
POST_HOLD = 5             # hold steps after last motion
CONTROL_HZ = 30.0

# Motion magnitude limits (anti-jitter safety bounds)
MAX_TRANS_M = 0.08        # 8 cm maximum translation per nudge
MAX_ROT_DEG = 40.0        # 40° maximum rotation per nudge
IK_ERROR_THRESHOLD = 3e-3 # 3 mm position error → reject


# ---------------------------------------------------------------------------
# Offline simulator
# ---------------------------------------------------------------------------


class OfflineSim:
    """Wraps ScriptedPolicy for headless, synchronous simulation.

    The robot is modelled as a perfect position-control follower: at each
    time step the "robot state" advances to exactly what the policy commanded
    in the previous step.  This gives clean (observation.state, action) pairs
    suitable for training.
    """

    def __init__(self, smoothing_steps: int = SMOOTHING_STEPS) -> None:
        ScriptedPolicy.SMOOTHING_STEPS = smoothing_steps  # class-level override
        self.kin = YamKinematics()
        self._policy: ScriptedPolicy | None = None
        self._joint_pos = {"left": HOME_LEFT_JP.copy(), "right": HOME_RIGHT_JP.copy()}
        self._gripper = {"left": HOME_GRIPPER_OPEN, "right": HOME_GRIPPER_OPEN}
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        left_jp: np.ndarray | None = None,
        right_jp: np.ndarray | None = None,
    ) -> None:
        """Reset the simulator to the given (or home) joint configuration."""
        self._joint_pos = {
            "left": (left_jp if left_jp is not None else HOME_LEFT_JP).copy(),
            "right": (right_jp if right_jp is not None else HOME_RIGHT_JP).copy(),
        }
        self._gripper = {"left": HOME_GRIPPER_OPEN, "right": HOME_GRIPPER_OPEN}
        self._policy = ScriptedPolicy(control_hz=CONTROL_HZ)
        # Prime the policy with the current state so _target_joints is set
        obs = self._make_obs()
        self._policy.get_action(obs)  # one-time initialisation

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def apply_nudge(
        self,
        side: str,
        delta_pos: np.ndarray | None = None,
        delta_quat_xyzw: np.ndarray | None = None,
    ) -> bool:
        """Apply an EE delta nudge.  Returns True if kinematically feasible."""
        assert self._policy is not None
        return self._policy.apply_nudge(
            side=side,
            delta_pos=delta_pos,
            delta_quat_xyzw=delta_quat_xyzw,
            ik_error_threshold=IK_ERROR_THRESHOLD,
        )

    def move_to_planned(
        self,
        side: str,
        target_pos: np.ndarray | None = None,
        target_quat_xyzw: np.ndarray | None = None,
        target_left_pos: np.ndarray | None = None,
        target_left_quat_xyzw: np.ndarray | None = None,
        target_right_pos: np.ndarray | None = None,
        target_right_quat_xyzw: np.ndarray | None = None,
    ) -> bool:
        """Use the motion planner to generate a collision-free trajectory to a target pose.

        For single-arm (side="left"/"right"): use ``target_pos`` / ``target_quat_xyzw``.
        For dual-arm (side="both"): use ``target_left_*`` / ``target_right_*``.

        Returns True if planning succeeded and the trajectory was executed.
        """
        assert self._policy is not None
        # Lazy-init the planner
        if not hasattr(self, "_planner"):
            from enpire.env.forge.experimental.portal_motion_planner import PortalMotionPlanner
            self._planner = PortalMotionPlanner(backend="curobo")

        cur_left = self._joint_pos["left"]
        cur_right = self._joint_pos["right"]

        kwargs: dict[str, Any] = {
            "current_left_jp": cur_left,
            "current_right_jp": cur_right,
            "side": side,
        }
        if side == "left":
            if target_pos is not None:
                kwargs["target_left_pos"] = target_pos
            if target_quat_xyzw is not None:
                kwargs["target_left_quat_xyzw"] = target_quat_xyzw
        elif side == "right":
            if target_pos is not None:
                kwargs["target_right_pos"] = target_pos
            if target_quat_xyzw is not None:
                kwargs["target_right_quat_xyzw"] = target_quat_xyzw
        elif side == "both":
            tl_pos = target_left_pos if target_left_pos is not None else target_pos
            tl_quat = target_left_quat_xyzw if target_left_quat_xyzw is not None else target_quat_xyzw
            tr_pos = target_right_pos if target_right_pos is not None else target_pos
            tr_quat = target_right_quat_xyzw if target_right_quat_xyzw is not None else target_quat_xyzw
            if tl_pos is not None:
                kwargs["target_left_pos"] = tl_pos
            if tl_quat is not None:
                kwargs["target_left_quat_xyzw"] = tl_quat
            if tr_pos is not None:
                kwargs["target_right_pos"] = tr_pos
            if tr_quat is not None:
                kwargs["target_right_quat_xyzw"] = tr_quat

        result = self._planner.plan_to_pose(**kwargs)
        if result["status"] != "Success":
            return False

        # Execute the trajectory by feeding waypoints to the policy
        self._policy.execute_trajectory(
            left_positions=result["left_positions"],
            right_positions=result["right_positions"],
        )
        return True

    def set_gripper(self, side: str, value: float) -> None:
        assert self._policy is not None
        self._policy.set_gripper(side, value)
        self._gripper[side] = float(np.clip(value, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Execute one control step.

        Returns:
            (obs_before, action_commanded) — each is a dict with
            left_joint_pos, right_joint_pos, left_gripper_pos, right_gripper_pos.
            obs_before is the state *before* this step; action_commanded is what
            the policy emits (= state after this step, since control is perfect).
        """
        assert self._policy is not None
        obs = self._make_obs()
        action, _ = self._policy.get_action(obs)
        # Perfect position control: robot advances to commanded position
        self._joint_pos["left"] = np.asarray(
            action["left_joint_pos"], dtype=np.float64
        )
        self._joint_pos["right"] = np.asarray(
            action["right_joint_pos"], dtype=np.float64
        )
        self._gripper["left"] = float(action["left_gripper_pos"][0])
        self._gripper["right"] = float(action["right_gripper_pos"][0])
        return obs, action

    def run_hold(self, n: int) -> list[tuple[dict, dict]]:
        """Run *n* hold steps (no new command queued)."""
        return [self.step() for _ in range(n)]

    def run_until_stable(self, post_hold: int = INTER_HOLD) -> list[tuple[dict, dict]]:
        """Run until the policy finishes interpolating, then hold *post_hold* steps."""
        assert self._policy is not None
        frames = []
        while self._policy.is_moving:
            frames.append(self.step())
        for _ in range(post_hold):
            frames.append(self.step())
        return frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> dict[str, np.ndarray]:
        return {
            "left_joint_pos": self._joint_pos["left"].astype(np.float32),
            "right_joint_pos": self._joint_pos["right"].astype(np.float32),
            "left_gripper_pos": np.array([self._gripper["left"]], dtype=np.float32),
            "right_gripper_pos": np.array([self._gripper["right"]], dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Episode specification
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """A single motion or gripper command within an episode."""
    prim_type: str  # "nudge" | "gripper" | "noop" | "move_to"
    side: str | None = None  # "left" | "right" | None
    delta_pos: np.ndarray | None = None  # metres, world frame
    delta_quat_xyzw: np.ndarray | None = None  # unit quaternion
    gripper_value: float | None = None  # 0=closed … 1=open
    # For move_to primitives (absolute targets)
    target_pos: np.ndarray | None = None  # absolute EE position (world frame)
    target_quat_xyzw: np.ndarray | None = None  # absolute EE orientation (xyzw)


@dataclass
class EpisodeSpec:
    language: str
    primitives: list[Primitive]
    episode_type: str = "generic"
    start_left_jp: np.ndarray | None = None   # custom start config (default: HOME)
    start_right_jp: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Episode catalogue
# ---------------------------------------------------------------------------


def _rot_quat(axis: np.ndarray, deg: float) -> np.ndarray:
    """Quaternion (xyzw) for a rotation of *deg* degrees around *axis*."""
    return Rotation.from_rotvec(axis * np.deg2rad(deg)).as_quat()


def _x() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])


def _y() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0])


def _z() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0])


# UI-convention helpers (matching Viser slider / nudge-button semantics):
#   Roll  = +X world rotation
#   Pitch = -Y world rotation  (+pitch nudge = -Y direction)
#   Yaw   = -Z world rotation  (+yaw   nudge = -Z direction)
def _roll_quat(deg: float) -> np.ndarray:
    """Quaternion for *deg* degrees of roll (+X rotation)."""
    return _rot_quat(_x(), deg)


def _pitch_quat(deg: float) -> np.ndarray:
    """Quaternion for *deg* degrees of pitch (-Y rotation)."""
    return _rot_quat(_y(), -deg)


def _yaw_quat(deg: float) -> np.ndarray:
    """Quaternion for *deg* degrees of yaw (-Z rotation)."""
    return _rot_quat(_z(), -deg)


def build_episode_catalogue() -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []

    # ----------------------------------------------------------------
    # 1. No-operation / hold-position
    # ----------------------------------------------------------------
    noop_langs = [
        "Stay still",
        "Don't move",
        "Hold your current position",
        "No operation",
        "Remain stationary",
        "Maintain current pose",
        "Do nothing",
    ]
    for lang in noop_langs:
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[Primitive(prim_type="noop")],
                episode_type="noop",
            )
        )

    # ----------------------------------------------------------------
    # 2. Single-arm translations
    # ----------------------------------------------------------------
    _trans_cfg = [
        # (direction_label, axis_vec, sign, dist_m)
        ("forward",    _x(),  1.0, 0.03),
        ("forward",    _x(),  1.0, 0.06),
        ("backward",   _x(), -1.0, 0.03),
        ("backward",   _x(), -1.0, 0.06),
        ("to the left", _y(),  1.0, 0.03),
        ("to the left", _y(),  1.0, 0.05),
        ("to the right", _y(), -1.0, 0.03),
        ("to the right", _y(), -1.0, 0.05),
        ("upward",      _z(),  1.0, 0.03),
        ("upward",      _z(),  1.0, 0.05),
        ("downward",    _z(), -1.0, 0.02),
        ("downward",    _z(), -1.0, 0.04),
    ]
    _trans_tmpl = [
        "Move your {side} arm {direction} {cm:.0f} centimeters",
        "Shift your {side} arm {direction} by {cm:.0f} cm",
        "Extend your {side} arm {direction} by {cm:.0f} centimeters",
        "Move your {side} hand {direction} {cm:.0f} cm",
    ]
    for side in ("left", "right"):
        for i, (direction, axis, sign, dist) in enumerate(_trans_cfg):
            dp = axis * sign * dist
            lang = _trans_tmpl[i % len(_trans_tmpl)].format(
                side=side, direction=direction, cm=dist * 100
            )
            episodes.append(
                EpisodeSpec(
                    language=lang,
                    primitives=[Primitive(prim_type="nudge", side=side, delta_pos=dp)],
                    episode_type="single_arm_translation",
                )
            )

    # ----------------------------------------------------------------
    # 3. Single-arm rotations
    # ----------------------------------------------------------------
    _rot_cfg = [
        # (ui_name, quat_fn, deg_in_ui_convention)
        # Roll  = +X rotation, Pitch = -Y rotation, Yaw = -Z rotation
        ("roll",  _roll_quat,   20.0),
        ("roll",  _roll_quat,  -20.0),
        ("pitch", _pitch_quat,  20.0),
        ("pitch", _pitch_quat, -20.0),
        ("yaw",   _yaw_quat,    30.0),
        ("yaw",   _yaw_quat,   -30.0),
        ("yaw",   _yaw_quat,    15.0),
        ("yaw",   _yaw_quat,   -15.0),
    ]
    _rot_tmpl = [
        "{action} your {side} arm {deg:.0f} degrees",
        "Apply {deg:.0f} degrees of {axis} to your {side} arm",
        "{action} the {side} wrist {deg:.0f} degrees",
        "Apply a {deg:.0f}-degree {axis} to your {side} arm",
    ]
    for side in ("left", "right"):
        for i, (ax_label, quat_fn, deg) in enumerate(_rot_cfg):
            dq = quat_fn(deg)
            action = ax_label.title()  # "Roll", "Pitch", "Yaw"
            lang = _rot_tmpl[i % len(_rot_tmpl)].format(
                side=side, axis=ax_label, action=action, deg=abs(deg)
            )
            if deg < 0:
                lang += " (clockwise)" if ax_label == "yaw" else " (negative direction)"
            episodes.append(
                EpisodeSpec(
                    language=lang,
                    primitives=[Primitive(prim_type="nudge", side=side, delta_quat_xyzw=dq)],
                    episode_type="single_arm_rotation",
                )
            )

    # ----------------------------------------------------------------
    # 4. Gripper actions
    # ----------------------------------------------------------------
    gripper_episodes: list[tuple[str, str, float]] = [
        ("Close your left gripper fully",    "left",  0.0),
        ("Open your left gripper fully",     "left",  1.0),
        ("Close your right gripper fully",   "right", 0.0),
        ("Open your right gripper fully",    "right", 1.0),
        ("Partially close your left gripper to 50%",  "left",  0.5),
        ("Partially close your right gripper to 50%", "right", 0.5),
        ("Open your left gripper halfway",   "left",  0.5),
        ("Open your right gripper halfway",  "right", 0.5),
        ("Squeeze your left gripper shut",   "left",  0.0),
        ("Release your right gripper",       "right", 1.0),
    ]
    for lang, side, val in gripper_episodes:
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[Primitive(prim_type="gripper", side=side, gripper_value=val)],
                episode_type="gripper",
            )
        )

    # ----------------------------------------------------------------
    # 5. Bimanual translations (both arms move together)
    # ----------------------------------------------------------------
    _bi_cfg = [
        ("forward",    _x(),  1.0, 0.04),
        ("forward",    _x(),  1.0, 0.06),
        ("backward",   _x(), -1.0, 0.04),
        ("upward",     _z(),  1.0, 0.04),
        ("upward",     _z(),  1.0, 0.06),
        ("to the left", _y(),  1.0, 0.03),
        ("to the right", _y(), -1.0, 0.03),
        ("downward",   _z(), -1.0, 0.03),
    ]
    _bi_tmpl = [
        "Move both arms {direction} {cm:.0f} centimeters",
        "Shift both arms {direction} by {cm:.0f} cm",
        "Extend both arms {direction} {cm:.0f} cm",
        "Move both hands {direction} by {cm:.0f} centimeters",
    ]
    for i, (direction, axis, sign, dist) in enumerate(_bi_cfg):
        dp = axis * sign * dist
        lang = _bi_tmpl[i % len(_bi_tmpl)].format(direction=direction, cm=dist * 100)
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[
                    Primitive(prim_type="nudge", side="left",  delta_pos=dp.copy()),
                    Primitive(prim_type="nudge", side="right", delta_pos=dp.copy()),
                ],
                episode_type="bimanual_translation",
            )
        )

    # ----------------------------------------------------------------
    # 6. Bimanual asymmetric (arms move in opposite directions)
    # ----------------------------------------------------------------
    _asym_episodes = [
        (
            "Move your left arm forward and your right arm backward simultaneously",
            Primitive(prim_type="nudge", side="left",  delta_pos=_x() * 0.04),
            Primitive(prim_type="nudge", side="right", delta_pos=-_x() * 0.04),
        ),
        (
            "Move your left arm to the right while your right arm moves to the left",
            Primitive(prim_type="nudge", side="left",  delta_pos=-_y() * 0.04),
            Primitive(prim_type="nudge", side="right", delta_pos= _y() * 0.04),
        ),
        (
            "Raise your left arm and lower your right arm at the same time",
            Primitive(prim_type="nudge", side="left",  delta_pos= _z() * 0.04),
            Primitive(prim_type="nudge", side="right", delta_pos=-_z() * 0.04),
        ),
        (
            "Extend both arms apart from each other",
            Primitive(prim_type="nudge", side="left",  delta_pos= _y() * 0.04),
            Primitive(prim_type="nudge", side="right", delta_pos=-_y() * 0.04),
        ),
        (
            "Bring both arms closer together",
            Primitive(prim_type="nudge", side="left",  delta_pos=-_y() * 0.04),
            Primitive(prim_type="nudge", side="right", delta_pos= _y() * 0.04),
        ),
    ]
    for lang, p_left, p_right in _asym_episodes:
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[p_left, p_right],
                episode_type="bimanual_asymmetric",
            )
        )

    # ----------------------------------------------------------------
    # 7. Combined motion + gripper
    # ----------------------------------------------------------------
    _combo_episodes = [
        (
            "Move your left arm forward 4 cm and close your left gripper",
            [
                Primitive(prim_type="nudge",   side="left", delta_pos=_x() * 0.04),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
            ],
        ),
        (
            "Raise your right arm 4 cm and open your right gripper",
            [
                Primitive(prim_type="nudge",   side="right", delta_pos=_z() * 0.04),
                Primitive(prim_type="gripper", side="right", gripper_value=1.0),
            ],
        ),
        (
            "Extend your left arm forward while closing its gripper",
            [
                Primitive(prim_type="nudge",   side="left", delta_pos=_x() * 0.05),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
            ],
        ),
        (
            "Move your right arm to the right and close the right gripper halfway",
            [
                Primitive(prim_type="nudge",   side="right", delta_pos=-_y() * 0.04),
                Primitive(prim_type="gripper", side="right", gripper_value=0.5),
            ],
        ),
        (
            "Close both grippers at the same time",
            [
                Primitive(prim_type="gripper", side="left",  gripper_value=0.0),
                Primitive(prim_type="gripper", side="right", gripper_value=0.0),
            ],
        ),
        (
            "Open both grippers simultaneously",
            [
                Primitive(prim_type="gripper", side="left",  gripper_value=1.0),
                Primitive(prim_type="gripper", side="right", gripper_value=1.0),
            ],
        ),
        (
            "Move both arms forward 5 cm then close both grippers",
            [
                Primitive(prim_type="nudge",   side="left",  delta_pos=_x() * 0.05),
                Primitive(prim_type="nudge",   side="right", delta_pos=_x() * 0.05),
                Primitive(prim_type="gripper", side="left",  gripper_value=0.0),
                Primitive(prim_type="gripper", side="right", gripper_value=0.0),
            ],
        ),
        (
            "Lift your left arm up and close your right gripper",
            [
                Primitive(prim_type="nudge",   side="left",  delta_pos=_z() * 0.04),
                Primitive(prim_type="gripper", side="right", gripper_value=0.0),
            ],
        ),
        (
            "Yaw your left arm 20 degrees while closing your left gripper",
            [
                Primitive(prim_type="nudge",   side="left", delta_quat_xyzw=_yaw_quat(20.0)),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
            ],
        ),
        (
            "Extend your right arm backward 3 cm and release the right gripper",
            [
                Primitive(prim_type="nudge",   side="right", delta_pos=-_x() * 0.03),
                Primitive(prim_type="gripper", side="right", gripper_value=1.0),
            ],
        ),
    ]
    for lang, prims in _combo_episodes:
        episodes.append(
            EpisodeSpec(language=lang, primitives=prims, episode_type="combined_motion_gripper")
        )

    # ----------------------------------------------------------------
    # 8. Multi-step sequences (translate then rotate, or two translations)
    # ----------------------------------------------------------------
    _seq_episodes = [
        (
            "Move your left arm forward 3 cm, then yaw it 15 degrees",
            [
                Primitive(prim_type="nudge", side="left", delta_pos=_x() * 0.03),
                Primitive(prim_type="nudge", side="left", delta_quat_xyzw=_yaw_quat(15.0)),
            ],
        ),
        (
            "Raise your right arm 4 cm, then move it forward 3 cm",
            [
                Primitive(prim_type="nudge", side="right", delta_pos=_z() * 0.04),
                Primitive(prim_type="nudge", side="right", delta_pos=_x() * 0.03),
            ],
        ),
        (
            "Move your left arm to the left 3 cm, then up 3 cm",
            [
                Primitive(prim_type="nudge", side="left", delta_pos= _y() * 0.03),
                Primitive(prim_type="nudge", side="left", delta_pos= _z() * 0.03),
            ],
        ),
        (
            "Extend your right arm forward 5 cm, then roll the wrist 20 degrees",
            [
                Primitive(prim_type="nudge", side="right", delta_pos=_x() * 0.05),
                Primitive(prim_type="nudge", side="right", delta_quat_xyzw=_roll_quat(20.0)),
            ],
        ),
        (
            "Move both arms upward 3 cm, then close both grippers",
            [
                Primitive(prim_type="nudge",   side="left",  delta_pos=_z() * 0.03),
                Primitive(prim_type="nudge",   side="right", delta_pos=_z() * 0.03),
                Primitive(prim_type="gripper", side="left",  gripper_value=0.0),
                Primitive(prim_type="gripper", side="right", gripper_value=0.0),
            ],
        ),
        (
            "Move your left arm forward 4 cm, then your right arm forward 4 cm",
            [
                Primitive(prim_type="nudge", side="left",  delta_pos=_x() * 0.04),
                Primitive(prim_type="nudge", side="right", delta_pos=_x() * 0.04),
            ],
        ),
        (
            "Pitch your right arm 15 degrees, then move it upward 3 cm",
            [
                Primitive(prim_type="nudge", side="right", delta_quat_xyzw=_pitch_quat(15.0)),
                Primitive(prim_type="nudge", side="right", delta_pos=_z() * 0.03),
            ],
        ),
        (
            "Move your left arm to the right 3 cm, close its gripper, then move it back left",
            [
                Primitive(prim_type="nudge",   side="left", delta_pos=-_y() * 0.03),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
                Primitive(prim_type="nudge",   side="left", delta_pos= _y() * 0.03),
            ],
        ),
    ]
    for lang, prims in _seq_episodes:
        episodes.append(
            EpisodeSpec(language=lang, primitives=prims, episode_type="multi_step_sequence")
        )

    # ----------------------------------------------------------------
    # 9. Planned single-arm moves (collision-free motion planner)
    # ----------------------------------------------------------------
    # Home EE positions: Left=[0.607, 0.148, 1.107], Right=[0.607, -0.148, 1.107]
    # Home EE quaternions (xyzw) — arms facing forward/down:
    _HOME_L_QUAT = np.array([-0.542, 0.675, -0.485, 0.122])
    _HOME_R_QUAT = np.array([0.675, -0.542, 0.122, -0.485])

    # Height-similar waypoints: targets are grouped by height band so that
    # start and end heights differ by at most ~0.03 m.
    _planned_single = [
        # (lang, side, target_pos, target_quat_xyzw)
        # --- same-height (z ≈ 1.10, near home height) ---
        ("Move your left arm forward to x=0.65 while keeping its orientation",
         "left", np.array([0.65, 0.15, 1.10]), _HOME_L_QUAT),
        ("Move your left arm to the center of the workspace",
         "left", np.array([0.62, 0.05, 1.10]), _HOME_L_QUAT),
        ("Move your left arm left to y=0.22",
         "left", np.array([0.60, 0.22, 1.10]), _HOME_L_QUAT),
        ("Move your left arm forward-left",
         "left", np.array([0.66, 0.18, 1.10]), _HOME_L_QUAT),
        ("Move your left arm backward slightly",
         "left", np.array([0.55, 0.15, 1.10]), _HOME_L_QUAT),
        ("Move your right arm forward to x=0.65 while keeping its orientation",
         "right", np.array([0.65, -0.15, 1.10]), _HOME_R_QUAT),
        ("Move your right arm to the center of the workspace",
         "right", np.array([0.62, -0.05, 1.10]), _HOME_R_QUAT),
        ("Move your right arm right to y=-0.22",
         "right", np.array([0.60, -0.22, 1.10]), _HOME_R_QUAT),
        ("Move your right arm forward-right",
         "right", np.array([0.66, -0.18, 1.10]), _HOME_R_QUAT),
        ("Move your right arm backward slightly",
         "right", np.array([0.55, -0.15, 1.10]), _HOME_R_QUAT),
        # --- higher band (z ≈ 1.15-1.17) ---
        ("Move your left arm upward to z=1.17",
         "left", np.array([0.60, 0.15, 1.17]), _HOME_L_QUAT),
        ("Move your left arm up and forward",
         "left", np.array([0.64, 0.15, 1.15]), _HOME_L_QUAT),
        ("Move your right arm upward to z=1.17",
         "right", np.array([0.60, -0.15, 1.17]), _HOME_R_QUAT),
        ("Move your right arm up and forward",
         "right", np.array([0.64, -0.15, 1.15]), _HOME_R_QUAT),
        # --- lower band (z ≈ 1.03-1.05) ---
        ("Lower your left arm to z=1.05",
         "left", np.array([0.60, 0.15, 1.05]), _HOME_L_QUAT),
        ("Lower your left arm to z=1.03 forward",
         "left", np.array([0.64, 0.12, 1.03]), _HOME_L_QUAT),
        ("Lower your right arm to z=1.05",
         "right", np.array([0.60, -0.15, 1.05]), _HOME_R_QUAT),
        ("Lower your right arm to z=1.03 forward",
         "right", np.array([0.64, -0.12, 1.03]), _HOME_R_QUAT),
    ]
    for lang, side, tgt_pos, tgt_quat in _planned_single:
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[
                    Primitive(
                        prim_type="move_to",
                        side=side,
                        target_pos=tgt_pos,
                        target_quat_xyzw=tgt_quat,
                    ),
                ],
                episode_type="planned_single_arm_move",
            )
        )

    # ----------------------------------------------------------------
    # 10. Planned moves from diverse start configurations
    # ----------------------------------------------------------------
    # Use IK to compute start joint configs from various EE positions.
    # Targets are at similar heights to their starts (delta z ≤ 0.03 m).
    _diverse_starts = [
        # (lang, side, start_l_pos, start_r_pos, target_pos, target_quat)
        # Left arm starting forward
        ("Move your left arm from forward position to the left",
         "left",
         np.array([0.65, 0.15, 1.10]), None,   # start left EE, right stays home
         np.array([0.60, 0.22, 1.10]), _HOME_L_QUAT),
        ("Move your left arm from forward position back toward home",
         "left",
         np.array([0.65, 0.12, 1.10]), None,
         np.array([0.60, 0.15, 1.10]), _HOME_L_QUAT),
        # Left arm starting high
        ("Move your left arm from high position forward",
         "left",
         np.array([0.58, 0.15, 1.16]), None,
         np.array([0.64, 0.15, 1.15]), _HOME_L_QUAT),
        # Left arm starting low
        ("Move your left arm from low position to center-left",
         "left",
         np.array([0.62, 0.12, 1.04]), None,
         np.array([0.60, 0.05, 1.05]), _HOME_L_QUAT),
        # Right arm starting forward
        ("Move your right arm from forward position to the right",
         "right",
         None, np.array([0.65, -0.15, 1.10]),
         np.array([0.60, -0.22, 1.10]), _HOME_R_QUAT),
        ("Move your right arm from forward position back toward home",
         "right",
         None, np.array([0.65, -0.12, 1.10]),
         np.array([0.60, -0.15, 1.10]), _HOME_R_QUAT),
        # Right arm starting high
        ("Move your right arm from high position forward",
         "right",
         None, np.array([0.58, -0.15, 1.16]),
         np.array([0.64, -0.15, 1.15]), _HOME_R_QUAT),
        # Right arm starting low
        ("Move your right arm from low position to center-right",
         "right",
         None, np.array([0.62, -0.12, 1.04]),
         np.array([0.60, -0.05, 1.05]), _HOME_R_QUAT),
    ]
    _kin = YamKinematics()
    for lang, side, start_l_pos, start_r_pos, tgt_pos, tgt_quat in _diverse_starts:
        # Compute start joint positions via IK
        sl_pos = start_l_pos if start_l_pos is not None else np.array([0.607, 0.148, 1.107])
        sr_pos = start_r_pos if start_r_pos is not None else np.array([0.607, -0.148, 1.107])
        sl_quat = _HOME_L_QUAT
        sr_quat = _HOME_R_QUAT
        start_l_jp, start_r_jp = _kin.inverse_kinematics(
            sl_pos, sl_quat, sr_pos, sr_quat, seeded=True
        )
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[
                    Primitive(prim_type="move_to", side=side,
                              target_pos=tgt_pos, target_quat_xyzw=tgt_quat),
                ],
                episode_type="planned_diverse_start",
                start_left_jp=start_l_jp,
                start_right_jp=start_r_jp,
            )
        )

    # ----------------------------------------------------------------
    # 11. Planned pick-and-place sequences (approach → grasp → lift → place → release)
    # ----------------------------------------------------------------
    # Approach and place at same height; only grasp goes down and lift goes up.
    _planned_pick_place = [
        (
            "Pick up an object with your left arm from the front-left area and place it center",
            "left",
            np.array([0.65, 0.12, 1.10]),  # approach
            np.array([0.65, 0.12, 1.04]),  # grasp (lower)
            np.array([0.65, 0.12, 1.12]),  # lift
            np.array([0.62, 0.04, 1.12]),  # place (same as lift height)
        ),
        (
            "Pick up an object with your right arm from the front-right area and place it center",
            "right",
            np.array([0.65, -0.12, 1.10]),  # approach
            np.array([0.65, -0.12, 1.04]),  # grasp
            np.array([0.65, -0.12, 1.12]),  # lift
            np.array([0.62, -0.04, 1.12]),  # place
        ),
        (
            "Use your left arm to pick from center-left and place forward-left",
            "left",
            np.array([0.61, 0.08, 1.10]),  # approach
            np.array([0.61, 0.08, 1.05]),  # grasp
            np.array([0.61, 0.08, 1.13]),  # lift
            np.array([0.66, 0.14, 1.13]),  # place (same as lift height)
        ),
        (
            "Use your right arm to pick from center-right and place forward-right",
            "right",
            np.array([0.61, -0.08, 1.10]),  # approach
            np.array([0.61, -0.08, 1.05]),  # grasp
            np.array([0.61, -0.08, 1.13]),  # lift
            np.array([0.66, -0.14, 1.13]),  # place
        ),
    ]
    for lang, side, approach, grasp, lift, place in _planned_pick_place:
        quat = _HOME_L_QUAT if side == "left" else _HOME_R_QUAT
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[
                    Primitive(prim_type="move_to", side=side,
                              target_pos=approach, target_quat_xyzw=quat),
                    Primitive(prim_type="move_to", side=side,
                              target_pos=grasp, target_quat_xyzw=quat),
                    Primitive(prim_type="gripper", side=side, gripper_value=0.0),
                    Primitive(prim_type="move_to", side=side,
                              target_pos=lift, target_quat_xyzw=quat),
                    Primitive(prim_type="move_to", side=side,
                              target_pos=place, target_quat_xyzw=quat),
                    Primitive(prim_type="gripper", side=side, gripper_value=1.0),
                ],
                episode_type="planned_pick_place",
            )
        )

    # ----------------------------------------------------------------
    # 12. Bimanual planned moves (height-similar targets)
    # ----------------------------------------------------------------
    _planned_bimanual = [
        (
            "Move both arms forward to x=0.65",
            np.array([0.65, 0.15, 1.10]), _HOME_L_QUAT,
            np.array([0.65, -0.15, 1.10]), _HOME_R_QUAT,
        ),
        (
            "Move both arms upward to z=1.16",
            np.array([0.60, 0.15, 1.16]), _HOME_L_QUAT,
            np.array([0.60, -0.15, 1.16]), _HOME_R_QUAT,
        ),
        (
            "Bring both arms closer together toward the center",
            np.array([0.60, 0.06, 1.10]), _HOME_L_QUAT,
            np.array([0.60, -0.06, 1.10]), _HOME_R_QUAT,
        ),
        (
            "Spread both arms apart to the sides",
            np.array([0.60, 0.24, 1.10]), _HOME_L_QUAT,
            np.array([0.60, -0.24, 1.10]), _HOME_R_QUAT,
        ),
        (
            "Move both arms forward and apart",
            np.array([0.66, 0.20, 1.10]), _HOME_L_QUAT,
            np.array([0.66, -0.20, 1.10]), _HOME_R_QUAT,
        ),
    ]
    for lang, l_pos, l_quat, r_pos, r_quat in _planned_bimanual:
        episodes.append(
            EpisodeSpec(
                language=lang,
                primitives=[
                    Primitive(prim_type="move_to", side="left",
                              target_pos=l_pos, target_quat_xyzw=l_quat),
                    Primitive(prim_type="move_to", side="right",
                              target_pos=r_pos, target_quat_xyzw=r_quat),
                ],
                episode_type="planned_bimanual_move",
            )
        )

    # ----------------------------------------------------------------
    # 13. Planned move + gripper combined
    # ----------------------------------------------------------------
    _planned_combo = [
        (
            "Move your left arm forward and close its gripper",
            [
                Primitive(prim_type="move_to", side="left",
                          target_pos=np.array([0.65, 0.15, 1.10]),
                          target_quat_xyzw=_HOME_L_QUAT),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
            ],
        ),
        (
            "Move your right arm up, then open its gripper fully",
            [
                Primitive(prim_type="move_to", side="right",
                          target_pos=np.array([0.60, -0.15, 1.16]),
                          target_quat_xyzw=_HOME_R_QUAT),
                Primitive(prim_type="gripper", side="right", gripper_value=1.0),
            ],
        ),
        (
            "Move both arms to the center and close both grippers",
            [
                Primitive(prim_type="move_to", side="left",
                          target_pos=np.array([0.62, 0.06, 1.10]),
                          target_quat_xyzw=_HOME_L_QUAT),
                Primitive(prim_type="move_to", side="right",
                          target_pos=np.array([0.62, -0.06, 1.10]),
                          target_quat_xyzw=_HOME_R_QUAT),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
                Primitive(prim_type="gripper", side="right", gripper_value=0.0),
            ],
        ),
        (
            "Move your left arm down, close gripper, then lift back up and open",
            [
                Primitive(prim_type="move_to", side="left",
                          target_pos=np.array([0.60, 0.15, 1.04]),
                          target_quat_xyzw=_HOME_L_QUAT),
                Primitive(prim_type="gripper", side="left", gripper_value=0.0),
                Primitive(prim_type="move_to", side="left",
                          target_pos=np.array([0.60, 0.15, 1.14]),
                          target_quat_xyzw=_HOME_L_QUAT),
                Primitive(prim_type="gripper", side="left", gripper_value=1.0),
            ],
        ),
    ]
    for lang, prims in _planned_combo:
        episodes.append(
            EpisodeSpec(language=lang, primitives=prims,
                        episode_type="planned_combined")
        )

    # ----------------------------------------------------------------
    # 14. No-ops at diverse configurations (not just home)
    # ----------------------------------------------------------------
    # Generate hold-position episodes at various EE positions so the model
    # learns "do nothing" regardless of where the arms currently are.
    _noop_diverse_langs = [
        "Stay still",
        "Hold your current position",
        "Maintain current pose",
        "Do nothing",
        "Remain stationary",
    ]
    _noop_start_positions = [
        # (description_suffix, left_ee_pos, right_ee_pos)
        ("forward", np.array([0.65, 0.15, 1.10]), np.array([0.65, -0.15, 1.10])),
        ("high",    np.array([0.60, 0.15, 1.17]), np.array([0.60, -0.15, 1.17])),
        ("low",     np.array([0.60, 0.15, 1.04]), np.array([0.60, -0.15, 1.04])),
        ("apart",   np.array([0.60, 0.22, 1.10]), np.array([0.60, -0.22, 1.10])),
        ("center",  np.array([0.62, 0.06, 1.10]), np.array([0.62, -0.06, 1.10])),
    ]
    for desc_suffix, noop_l_pos, noop_r_pos in _noop_start_positions:
        start_l_jp, start_r_jp = _kin.inverse_kinematics(
            noop_l_pos, _HOME_L_QUAT, noop_r_pos, _HOME_R_QUAT, seeded=True
        )
        for lang in _noop_diverse_langs:
            episodes.append(
                EpisodeSpec(
                    language=lang,
                    primitives=[Primitive(prim_type="noop")],
                    episode_type="noop_diverse",
                    start_left_jp=start_l_jp,
                    start_right_jp=start_r_jp,
                )
            )

    return episodes


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(spec: EpisodeSpec, sim: OfflineSim) -> list[dict[str, Any]] | None:
    """Simulate one episode.

    Returns a list of frame dicts (one per control step), or None if all
    nudge primitives were rejected as kinematically infeasible.

    Each frame dict contains:
        state_14d   : np.ndarray(14)  current joint state before the action
        action_14d  : np.ndarray(14)  commanded joint target
    """
    sim.reset(left_jp=spec.start_left_jp, right_jp=spec.start_right_jp)
    frames: list[dict[str, Any]] = []

    def _collect(raw_frames: list[tuple[dict, dict]]) -> None:
        for obs, act in raw_frames:
            frames.append(
                {
                    "state_14d": _pack_14d(obs),
                    "action_14d": _pack_14d(act),
                }
            )

    # Pre-hold (robot stationary at home)
    _collect(sim.run_hold(PRE_HOLD))

    any_nudge_accepted = False

    for prim in spec.primitives:
        if prim.prim_type == "noop":
            # No command — just run hold steps so we have data
            _collect(sim.run_hold(POST_HOLD))

        elif prim.prim_type == "nudge":
            accepted = sim.apply_nudge(
                side=prim.side,  # type: ignore[arg-type]
                delta_pos=prim.delta_pos,
                delta_quat_xyzw=prim.delta_quat_xyzw,
            )
            if not accepted:
                print(
                    f"  [EpisodeRunner] Nudge rejected for '{spec.language}' "
                    f"(side={prim.side}). Skipping episode."
                )
                return None
            any_nudge_accepted = True
            _collect(sim.run_until_stable(post_hold=INTER_HOLD))

        elif prim.prim_type == "move_to":
            accepted = sim.move_to_planned(
                side=prim.side,  # type: ignore[arg-type]
                target_pos=prim.target_pos,
                target_quat_xyzw=prim.target_quat_xyzw,
            )
            if not accepted:
                print(
                    f"  [EpisodeRunner] move_to planning failed for '{spec.language}' "
                    f"(side={prim.side}). Skipping episode."
                )
                return None
            any_nudge_accepted = True
            _collect(sim.run_until_stable(post_hold=INTER_HOLD))

        elif prim.prim_type == "gripper":
            sim.set_gripper(prim.side, prim.gripper_value)  # type: ignore[arg-type]
            # Gripper changes take effect immediately; hold a few steps so the
            # dataset captures the transition clearly.
            _collect(sim.run_hold(INTER_HOLD + 2))

        else:
            raise ValueError(f"Unknown primitive type: {prim.prim_type!r}")

    # Post-hold
    _collect(sim.run_hold(POST_HOLD))

    # Return None only when the episode had nudges but ALL were rejected
    has_nudges = any(p.prim_type == "nudge" for p in spec.primitives)
    if has_nudges and not any_nudge_accepted:
        return None

    return frames


def _pack_14d(d: dict[str, np.ndarray]) -> np.ndarray:
    """Pack obs/action dict into 14D array [l_jp(6), l_g(1), r_jp(6), r_g(1)]."""
    return np.concatenate(
        [
            np.asarray(d["left_joint_pos"],   dtype=np.float32).reshape(6),
            np.asarray(d["left_gripper_pos"], dtype=np.float32).reshape(1),
            np.asarray(d["right_joint_pos"],  dtype=np.float32).reshape(6),
            np.asarray(d["right_gripper_pos"],dtype=np.float32).reshape(1),
        ]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Format conversions  (joint → delta-EE, joint → UMI)
# ---------------------------------------------------------------------------


def _fk(kin: YamKinematics, jp14d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run FK on a 14D joint vector. Returns (l_pos, l_q, r_pos, r_q)."""
    return kin.forward_kinematics(jp14d[0:6], jp14d[7:13])


def compute_delta_ee_actions(
    states_14d: np.ndarray, actions_14d: np.ndarray, kin: YamKinematics
) -> np.ndarray:
    """Convert joint (state, action) pairs to 16D delta EE pose actions.

    delta_pos  = FK(action_joints).pos  - FK(state_joints).pos
    delta_quat = FK(action_joints).quat * inv(FK(state_joints).quat)

    Layout: [l_dpos(3), l_dq_xyzw(4), l_grip(1), r_dpos(3), r_dq_xyzw(4), r_grip(1)]
    """
    T = len(states_14d)
    out = np.zeros((T, 16), dtype=np.float32)
    for i in range(T):
        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = _fk(kin, states_14d[i])
        tgt_l_pos, tgt_l_q, tgt_r_pos, tgt_r_q = _fk(kin, actions_14d[i])

        dl_pos = tgt_l_pos - cur_l_pos
        dr_pos = tgt_r_pos - cur_r_pos
        dl_q = (Rotation.from_quat(tgt_l_q) * Rotation.from_quat(cur_l_q).inv()).as_quat()
        dr_q = (Rotation.from_quat(tgt_r_q) * Rotation.from_quat(cur_r_q).inv()).as_quat()

        out[i] = np.concatenate([
            dl_pos, dl_q, actions_14d[i, 6:7],    # left  (gripper absolute)
            dr_pos, dr_q, actions_14d[i, 13:14],  # right (gripper absolute)
        ]).astype(np.float32)
    return out


def compute_ee_pose_16d(jp14d: np.ndarray, kin: YamKinematics) -> np.ndarray:
    """Convert a single 14D joint vector to 16D absolute EE pose.

    Layout: [l_pos(3), l_q_xyzw(4), l_grip(1), r_pos(3), r_q_xyzw(4), r_grip(1)]
    """
    l_pos, l_q, r_pos, r_q = _fk(kin, jp14d)
    return np.concatenate([
        l_pos, l_q, jp14d[6:7],    # left
        r_pos, r_q, jp14d[13:14],  # right
    ]).astype(np.float32)


def compute_ee_pose_states(states_14d: np.ndarray, kin: YamKinematics) -> np.ndarray:
    """Compute 16D absolute EE pose from 14D joint states.

    Layout: [l_pos(3), l_q_xyzw(4), l_grip(1), r_pos(3), r_q_xyzw(4), r_grip(1)]
    """
    T = len(states_14d)
    out = np.zeros((T, 16), dtype=np.float32)
    for i in range(T):
        out[i] = compute_ee_pose_16d(states_14d[i], kin)
    return out


def compute_ee_pose_actions(actions_14d: np.ndarray, kin: YamKinematics) -> np.ndarray:
    """Compute 16D absolute EE pose actions from 14D joint actions.

    Each action is the absolute EE pose target for that step.
    Layout: [l_pos(3), l_q_xyzw(4), l_grip(1), r_pos(3), r_q_xyzw(4), r_grip(1)]
    """
    T = len(actions_14d)
    out = np.zeros((T, 16), dtype=np.float32)
    for i in range(T):
        out[i] = compute_ee_pose_16d(actions_14d[i], kin)
    return out


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_joint_df(
    frames: list[dict],
    language: str,
    episode_idx: int,
    step_offset: int = 0,
) -> pd.DataFrame:
    states  = [f["state_14d"]  for f in frames]
    actions = [f["action_14d"] for f in frames]
    T = len(frames)
    return pd.DataFrame({
        "step_idx":          [step_offset + i for i in range(T)],
        "episode_idx":       [episode_idx] * T,
        "language_command":  [language] * T,
        "observation.state": states,
        "action":            actions,
    })


def build_delta_ee_df(
    frames: list[dict],
    language: str,
    episode_idx: int,
    kin: YamKinematics,
    step_offset: int = 0,
) -> pd.DataFrame:
    states_14d  = np.stack([f["state_14d"]  for f in frames])
    actions_14d = np.stack([f["action_14d"] for f in frames])
    delta_ee    = compute_delta_ee_actions(states_14d, actions_14d, kin)
    T = len(frames)
    return pd.DataFrame({
        "step_idx":          [step_offset + i for i in range(T)],
        "episode_idx":       [episode_idx] * T,
        "language_command":  [language] * T,
        "observation.state": list(states_14d),   # 14D joint (for calibration)
        "action":            list(delta_ee),      # 16D delta EE
    })


def build_ee_pose_df(
    frames: list[dict],
    language: str,
    episode_idx: int,
    kin: YamKinematics,
    step_offset: int = 0,
) -> pd.DataFrame:
    states_14d  = np.stack([f["state_14d"] for f in frames])
    actions_14d = np.stack([f["action_14d"] for f in frames])
    ee_states   = compute_ee_pose_states(states_14d, kin)
    ee_actions  = compute_ee_pose_actions(actions_14d, kin)
    T = len(frames)
    return pd.DataFrame({
        "step_idx":          [step_offset + i for i in range(T)],
        "episode_idx":       [episode_idx] * T,
        "language_command":  [language] * T,
        "observation.state": list(ee_states),    # 16D absolute EE pose
        "action":            list(ee_actions),   # 16D absolute EE pose target
    })


# ---------------------------------------------------------------------------
# Jitter / smoothness validation
# ---------------------------------------------------------------------------


def validate_episode_smoothness(
    frames: list[dict],
    max_joint_delta_rad: float = 0.15,
    max_ee_delta_m: float = 0.05,
    kin: YamKinematics | None = None,
) -> bool:
    """Return True if the trajectory passes basic anti-jitter checks.

    Checks:
    1. No single joint step > max_joint_delta_rad (default 0.15 rad ≈ 8.6°).
    2. No single EE position step > max_ee_delta_m (default 5 cm).
    """
    ok = True
    prev_action = None
    prev_ee = None

    for i, frame in enumerate(frames):
        act = frame["action_14d"]
        if prev_action is not None:
            max_jd = float(np.max(np.abs(act[0:6] - prev_action[0:6])))
            max_jd_r = float(np.max(np.abs(act[7:13] - prev_action[7:13])))
            if max(max_jd, max_jd_r) > max_joint_delta_rad:
                print(
                    f"  [Validate] Step {i}: joint delta {max(max_jd, max_jd_r):.4f} rad "
                    f"> threshold {max_joint_delta_rad:.4f}"
                )
                ok = False

        if kin is not None:
            l_pos, _, r_pos, _ = kin.forward_kinematics(act[0:6], act[7:13])
            ee = np.concatenate([l_pos, r_pos])
            if prev_ee is not None:
                max_ee_step = float(np.max(np.abs(ee - prev_ee)))
                if max_ee_step > max_ee_delta_m:
                    print(
                        f"  [Validate] Step {i}: EE delta {max_ee_step:.4f} m "
                        f"> threshold {max_ee_delta_m:.4f}"
                    )
                    ok = False
            prev_ee = ee

        prev_action = act

    return ok


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_dataset(
    output_dir: Path,
    smoothing_steps: int = SMOOTHING_STEPS,
    validate: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate the full language dataset and write parquet files.

    Returns a summary dict with statistics and the output paths.
    """
    output_dir = Path(output_dir)
    catalogue = build_episode_catalogue()

    sim = OfflineSim(smoothing_steps=smoothing_steps)
    kin = sim.kin  # reuse the same kinematics instance

    joint_dir     = output_dir / "joint"
    delta_ee_dir  = output_dir / "delta_ee"
    ee_pose_dir   = output_dir / "ee_pose"

    manifest: list[dict[str, Any]] = []
    global_step = 0
    episode_idx = 0
    skipped = 0

    for spec_idx, spec in enumerate(catalogue):
        if verbose:
            print(f"[{spec_idx+1}/{len(catalogue)}] {spec.episode_type!r:30s} — {spec.language!r}")

        frames = run_episode(spec, sim)

        if frames is None:
            skipped += 1
            if verbose:
                print(f"  → SKIPPED (IK infeasible)")
            continue

        # Optional smoothness validation
        if validate:
            if not validate_episode_smoothness(frames, kin=kin):
                print(f"  → WARNING: smoothness check failed for ep {episode_idx}")

        T = len(frames)

        # Build and save parquets
        ep_name = f"episode_{episode_idx:04d}.parquet"
        jdf = build_joint_df(frames, spec.language, episode_idx, step_offset=0)
        ddf = build_delta_ee_df(frames, spec.language, episode_idx, kin, step_offset=0)
        edf = build_ee_pose_df(frames, spec.language, episode_idx, kin, step_offset=0)

        _save_parquet(jdf, joint_dir     / ep_name)
        _save_parquet(ddf, delta_ee_dir  / ep_name)
        _save_parquet(edf, ee_pose_dir   / ep_name)

        manifest.append(
            {
                "episode_idx":    episode_idx,
                "language":       spec.language,
                "episode_type":   spec.episode_type,
                "n_steps":        T,
                "n_primitives":   len(spec.primitives),
                "joint_path":     str(joint_dir     / ep_name),
                "delta_ee_path":  str(delta_ee_dir  / ep_name),
                "ee_pose_path":   str(ee_pose_dir   / ep_name),
            }
        )

        global_step += T
        episode_idx += 1

        if verbose:
            print(f"  → {T} steps  [total {global_step} steps so far]")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Also write combined parquets for convenience (all episodes in one file)
    _write_combined(output_dir, manifest, verbose)

    summary = {
        "total_episodes":   episode_idx,
        "skipped_episodes": skipped,
        "total_steps":      global_step,
        "output_dir":       str(output_dir),
        "manifest_path":    str(manifest_path),
    }
    print("\n" + "=" * 60)
    print(f"Dataset generation complete.")
    print(f"  Episodes : {episode_idx}  (skipped: {skipped})")
    print(f"  Steps    : {global_step}")
    print(f"  Output   : {output_dir}")
    print(f"  Manifest : {manifest_path}")
    print()
    print("Replay commands:")
    print(f"  Joint:     uv run python -m experimental.yam_control_loop --use-replay-policy --control-mode joint_position       --replay-dataset-path {joint_dir}/episode_0000.parquet")
    print(f"  DeltaEE:   uv run python -m experimental.yam_control_loop --use-replay-policy --control-mode delta_ee_pose        --replay-dataset-path {delta_ee_dir}/episode_0000.parquet")
    print(f"  EE Pose:   uv run python -m experimental.yam_control_loop --use-replay-policy --control-mode ee_pose              --replay-dataset-path {ee_pose_dir}/episode_0000.parquet")
    print("=" * 60)
    return summary


def _write_combined(
    output_dir: Path, manifest: list[dict], verbose: bool = True
) -> None:
    """Concatenate all per-episode parquets into three combined files."""
    for fmt, key in [("joint", "joint_path"), ("delta_ee", "delta_ee_path"), ("ee_pose", "ee_pose_path")]:
        dfs = []
        global_step = 0
        for entry in manifest:
            df = pd.read_parquet(entry[key])
            # Re-index steps globally across all episodes
            df["step_idx"] = list(range(global_step, global_step + len(df)))
            global_step += len(df)
            dfs.append(df)
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined_path = output_dir / fmt / "combined.parquet"
            _save_parquet(combined, combined_path)
            if verbose:
                print(f"  Combined {fmt}: {combined_path}  ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline language-command control dataset for YAM robot."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_TBD_ROOT / "data" / "language_dataset",
        help="Directory to write parquet files into.",
    )
    parser.add_argument(
        "--smoothing-steps",
        type=int,
        default=SMOOTHING_STEPS,
        help="Number of interpolation steps per nudge command (default: %(default)s).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip smoothness validation checks.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress output.",
    )
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        smoothing_steps=args.smoothing_steps,
        validate=not args.no_validate,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
