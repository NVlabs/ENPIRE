# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive motion planner test suite for the YAM bimanual robot.

Procedurally generates ~1000 test cases by systematically scanning the
reachable SE(3) configuration space:
  - XY grid at multiple Z heights (position)
  - Euler angle grid for orientation (SO(3))
  - Continuous gripper width sweep [0, 1]
  - Bug-regression cases (floating-point joint-limit drift)
  - User-reported failures and screenshot configurations

Run:
    python -m experimental.test_motion_planner           # full ~1000 tests
    python -m experimental.test_motion_planner --quick   # regression-only (~20)
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_LEFT = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25])
HOME_RIGHT = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25])

# ViserUI slider ranges (from the UI definition)
X_RANGE = (0.20, 0.85)
Y_RANGE = (-0.55, 0.55)  # right arm negative, left arm positive
Z_RANGE = (0.75, 1.45)
GRIPPER_RANGE = (0.0, 1.0)
RPY_RANGE = (-180.0, 180.0)  # each of roll, pitch, yaw


def display_rpy_to_quat_xyzw(
    roll_deg: float, pitch_deg: float, yaw_deg: float,
) -> np.ndarray:
    """Convert ViserUI display RPY (degrees) to quaternion (xyzw).

    Mirrors the conversion in start_stop_play_policy.py ViserUI._make_move_cb:
        euler_deg = [-pitch, roll, -yaw - 90.0]
        quat = R.from_euler('xyz', euler_deg, degrees=True).as_quat()
    """
    euler_deg = [-pitch_deg, roll_deg, -yaw_deg - 90.0]
    return R.from_euler("xyz", euler_deg, degrees=True).as_quat()


@dataclass
class TestCase:
    name: str
    side: str                           # "left" | "right"
    target_pos: list[float]             # [x, y, z]
    display_rpy: list[float]            # [roll, pitch, yaw] display values
    gripper: float = 1.0                # 0=closed, 1=open
    start_left: np.ndarray | None = None
    start_right: np.ndarray | None = None
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Procedural generation helpers
# ---------------------------------------------------------------------------

def _linspace_inclusive(lo: float, hi: float, n: int) -> list[float]:
    """Like np.linspace but returns a plain list."""
    return np.linspace(lo, hi, n).tolist()


def _build_regression_cases() -> list[TestCase]:
    """Hand-picked regression & bug-fix cases (always run)."""
    cases: list[TestCase] = []

    # --- Floating-point boundary bug ---
    for drift, label in [(-0.0001, "tiny"), (-0.001, "medium"), (-0.005, "large")]:
        for side, home, idx_name in [
            ("right", HOME_RIGHT, "right"),
            ("left", HOME_LEFT, "left"),
        ]:
            buggy = home.copy()
            buggy[1] = drift
            y = -0.31 if side == "right" else 0.31
            start_kw = {"start_right": buggy} if side == "right" else {"start_left": buggy}
            cases.append(TestCase(
                name=f"BUGFIX: {side} j2={drift} drift",
                side=side,
                target_pos=[0.6, y, 0.9125],
                display_rpy=[0.0, 90.4, 0.0],
                tags=["regression", "bugfix"],
                **start_kw,
            ))

    # --- User's exact failing case ---
    cases.append(TestCase(
        name="USER-CASE: right [0.6,-0.31,0.9125] RPY=(0,90.4,0)",
        side="right",
        target_pos=[0.6, -0.31, 0.9125],
        display_rpy=[0.0, 90.4, 0.0],
        tags=["regression", "user"],
    ))

    # --- Screenshot configuration ---
    cases.append(TestCase(
        name="SCREENSHOT: right [0.87,-0.11,0.91] RPY=(0.5,115.7,-49.9)",
        side="right",
        target_pos=[0.8655, -0.1104, 0.9105],
        display_rpy=[0.5, 115.7, -49.9],
        gripper=0.999,
        tags=["regression", "screenshot"],
    ))

    return cases


def _build_se3_scan_cases() -> list[TestCase]:
    """Procedurally generated SE(3) scan with gripper sweep.

    Grid design (approximate counts):
      Position:   X(8) x Y(6) x Z(5)  = 240 points per arm  (x2 arms)
      Orientation: roll(3) x pitch(5) x yaw(3) = 45 combos
      Gripper:    5 levels [0.0, 0.25, 0.5, 0.75, 1.0]

    We combine them strategically to hit ~1000 total:
      - Dense XYZ grid at default orientation + default gripper   → ~480
      - Orientation sweep at a few representative positions        → ~270
      - Gripper sweep at a few representative (pos, orient) combos → ~100
      - Cross-product spot-checks                                  → ~100+
    """
    cases: list[TestCase] = []
    default_rpy = [0.0, 90.0, 0.0]

    # ======================== POSITION GRID ========================
    # X: 8 values spanning the slider range, biased toward reachable center
    xs = _linspace_inclusive(0.35, 0.82, 8)
    # Z: 5 heights spanning the reachable range
    zs = _linspace_inclusive(0.88, 1.35, 5)
    # Y: 6 values — for right arm we use negative, for left positive
    y_abs = _linspace_inclusive(0.0, 0.45, 6)

    for side in ("right", "left"):
        y_sign = -1.0 if side == "right" else 1.0
        for x in xs:
            for y_a in y_abs:
                y = round(y_a * y_sign, 4)
                for z in zs:
                    cases.append(TestCase(
                        name=f"POS-{side[0].upper()}: ({x:.2f},{y:+.2f},{z:.2f}) g=1.0",
                        side=side,
                        target_pos=[round(x, 4), y, round(z, 4)],
                        display_rpy=default_rpy,
                        gripper=1.0,
                        tags=["position"],
                    ))

    # ======================== ORIENTATION GRID ========================
    # Test at 3 representative positions per arm
    pos_right_repr = [
        [0.55, -0.20, 0.95],
        [0.65, -0.10, 1.05],
        [0.50, -0.30, 0.91],
    ]
    pos_left_repr = [
        [0.55, 0.20, 0.95],
        [0.65, 0.10, 1.05],
        [0.50, 0.30, 0.91],
    ]

    rolls  = _linspace_inclusive(-90.0, 90.0, 3)   # 3 values
    pitches = _linspace_inclusive(60.0, 150.0, 5)   # 5 values (reachable range)
    yaws   = _linspace_inclusive(-60.0, 60.0, 3)    # 3 values

    for side, positions in [("right", pos_right_repr), ("left", pos_left_repr)]:
        for pos in positions:
            for roll, pitch, yaw in itertools.product(rolls, pitches, yaws):
                cases.append(TestCase(
                    name=(
                        f"ROT-{side[0].upper()}: "
                        f"({pos[0]:.2f},{pos[1]:+.2f},{pos[2]:.2f}) "
                        f"R={roll:.0f} P={pitch:.0f} Y={yaw:.0f}"
                    ),
                    side=side,
                    target_pos=pos,
                    display_rpy=[roll, pitch, yaw],
                    gripper=1.0,
                    tags=["orientation"],
                ))

    # ======================== GRIPPER SWEEP ========================
    grippers = _linspace_inclusive(0.0, 1.0, 5)  # [0, 0.25, 0.5, 0.75, 1.0]
    # Test at 2 positions x 2 orientations per arm
    grip_configs = [
        # (pos, rpy)
        ([0.55, -0.20, 0.95], [0.0, 90.0, 0.0]),
        ([0.65, -0.25, 1.00], [0.0, 120.0, -30.0]),
    ]
    grip_configs_left = [
        ([0.55, 0.20, 0.95], [0.0, 90.0, 0.0]),
        ([0.65, 0.25, 1.00], [0.0, 120.0, -30.0]),
    ]

    for side, cfgs in [("right", grip_configs), ("left", grip_configs_left)]:
        for pos, rpy in cfgs:
            for g in grippers:
                cases.append(TestCase(
                    name=(
                        f"GRIP-{side[0].upper()}: "
                        f"({pos[0]:.2f},{pos[1]:+.2f},{pos[2]:.2f}) "
                        f"P={rpy[1]:.0f} g={g:.2f}"
                    ),
                    side=side,
                    target_pos=pos,
                    display_rpy=rpy,
                    gripper=round(g, 2),
                    tags=["gripper"],
                ))

    # ======================== CROSS-PRODUCT SPOT CHECKS ========================
    # Full SE(3)+gripper combos at a few carefully chosen configs
    spot_xs = [0.50, 0.65, 0.75]
    spot_zs = [0.92, 1.10]
    spot_pitches = [80.0, 110.0, 140.0]
    spot_rolls = [-45.0, 0.0, 45.0]
    spot_grippers = [0.0, 0.5, 1.0]

    for side in ("right", "left"):
        y_val = -0.20 if side == "right" else 0.20
        for x, z, pitch, roll, g in itertools.product(
            spot_xs, spot_zs, spot_pitches, spot_rolls, spot_grippers,
        ):
            cases.append(TestCase(
                name=(
                    f"SE3-{side[0].upper()}: "
                    f"({x:.2f},{y_val:+.2f},{z:.2f}) "
                    f"R={roll:.0f} P={pitch:.0f} g={g:.1f}"
                ),
                side=side,
                target_pos=[x, y_val, z],
                display_rpy=[roll, pitch, 0.0],
                gripper=round(g, 2),
                tags=["se3_cross"],
            ))

    return cases


def build_test_cases(quick: bool = False) -> list[TestCase]:
    """Build the full test suite."""
    cases = _build_regression_cases()
    if not quick:
        cases += _build_se3_scan_cases()
    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(quick: bool = False) -> None:
    from enpire.env.forge.experimental.motion_planner import YamMotionPlanner

    planner = YamMotionPlanner()

    home_l_pos, home_l_q, home_r_pos, home_r_q = planner._kin.forward_kinematics(
        HOME_LEFT, HOME_RIGHT,
    )
    print(f"Home left  EE: pos={home_l_pos}, quat={home_l_q}")
    print(f"Home right EE: pos={home_r_pos}, quat={home_r_q}")

    cases = build_test_cases(quick=quick)
    total = len(cases)
    print(f"\nRunning {total} test cases {'(quick mode)' if quick else ''}\n")

    # Counters
    n_pass = n_fail = n_warn = 0
    failed: list[tuple[str, str]] = []
    warned: list[str] = []
    tag_stats: dict[str, dict[str, int]] = {}
    total_plan_time = 0.0

    for i, tc in enumerate(cases, 1):
        start_left = tc.start_left if tc.start_left is not None else HOME_LEFT.copy()
        start_right = tc.start_right if tc.start_right is not None else HOME_RIGHT.copy()
        quat_xyzw = display_rpy_to_quat_xyzw(*tc.display_rpy)
        tgt_pos = np.array(tc.target_pos)

        kwargs: dict = dict(
            current_left_jp=start_left,
            current_right_jp=start_right,
            side=tc.side,
        )
        if tc.side == "left":
            kwargs["target_left_pos"] = tgt_pos
            kwargs["target_left_quat_xyzw"] = quat_xyzw
            kwargs["left_gripper"] = tc.gripper
        else:
            kwargs["target_right_pos"] = tgt_pos
            kwargs["target_right_quat_xyzw"] = quat_xyzw
            kwargs["right_gripper"] = tc.gripper

        planner.set_gripper_qpos(left_gripper=1.0, right_gripper=tc.gripper)

        t0 = time.time()
        result = planner.plan_to_pose(**kwargs)
        dt = time.time() - t0
        total_plan_time += dt
        status = result["status"]

        if status == "Success":
            n_steps = len(result["position"])
            collisions = 0
            for s in range(len(result["left_positions"])):
                if planner.check_collision(
                    result["left_positions"][s],
                    result["right_positions"][s],
                ):
                    collisions += 1
            if collisions > 0:
                tag = "FAIL"
                detail = f"COLLISION {collisions}/{n_steps} steps"
                n_fail += 1
                failed.append((tc.name, detail))
            else:
                tag = "PASS"
                detail = f"{n_steps} steps ok"
                n_pass += 1
        elif status == "IK_Failed":
            tag = "WARN"
            detail = "IK unreachable"
            n_warn += 1
            warned.append(tc.name)
        else:
            tag = "FAIL"
            detail = "Planning_Failed"
            n_fail += 1
            failed.append((tc.name, detail))

        # Per-tag stats
        for t_tag in (tc.tags or ["other"]):
            if t_tag not in tag_stats:
                tag_stats[t_tag] = {"pass": 0, "fail": 0, "warn": 0}
            tag_stats[t_tag][tag.lower()] = tag_stats[t_tag].get(tag.lower(), 0) + 1

        sym = {"PASS": "+", "FAIL": "X", "WARN": "?"}[tag]
        # Print every failure/warn, and periodic progress for passes
        if tag != "PASS" or i % 50 == 0 or i == total:
            print(f"  [{sym}] [{i:4d}/{total}] {tc.name}  → {detail} ({dt:.2f}s)")

    # ======================== SUMMARY ========================
    print("\n" + "=" * 80)
    print(f"SUMMARY — {total} tests in {total_plan_time:.1f}s")
    print("=" * 80)
    print(f"  PASS: {n_pass:4d}/{total}  ({100*n_pass/total:.1f}%)")
    print(f"  FAIL: {n_fail:4d}/{total}  ({100*n_fail/total:.1f}%)")
    print(f"  WARN: {n_warn:4d}/{total}  ({100*n_warn/total:.1f}%)  (IK unreachable)")

    print("\n  Per-category breakdown:")
    for t_tag in sorted(tag_stats):
        s = tag_stats[t_tag]
        cat_total = s["pass"] + s["fail"] + s["warn"]
        print(
            f"    {t_tag:15s}: {cat_total:4d} total | "
            f"{s['pass']:3d} pass | {s['fail']:3d} fail | {s['warn']:3d} warn"
        )

    if failed:
        print(f"\n  FAILED TESTS ({len(failed)}):")
        for name, detail in failed:
            print(f"    [X] {name}  — {detail}")

    print()
    if n_fail == 0:
        print("ALL PLANNING TESTS PASSED (0 failures)")
    else:
        print(f"!!! {n_fail} TEST(S) FAILED !!!")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_tests(quick=quick)
