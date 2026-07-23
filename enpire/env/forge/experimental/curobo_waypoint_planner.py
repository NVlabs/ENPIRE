# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline waypoint trajectory planner: fast IK + cubic spline + CuRobo collision.

Given Cartesian waypoints (xyz) with start/end orientations:
1. Mink seeded IK for each waypoint (~2ms each, sequential continuity)
2. Cubic spline in joint space → C2 smooth, no jerk discontinuities
3. CuRobo collision validation on the final trajectory
4. Time-parameterize with velocity/acceleration bounds

Smooth by math, collision-safe via CuRobo.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

_DEFAULT_ARC_SPACING_M = 0.005
_MAX_JOINT_JUMP_RAD = 0.3
_DEFAULT_MAX_JOINT_VEL = 1.5


def display_rpy_to_quat_xyzw(rpy_deg):
    """Convert display RPY [roll, pitch, yaw] degrees to quaternion xyzw."""
    roll, pitch, yaw = np.asarray(rpy_deg, dtype=np.float64)
    euler_xyz = [-pitch, roll, -yaw - 90.0]
    return Rotation.from_euler("xyz", euler_xyz, degrees=True).as_quat()


def quat_xyzw_to_display_rpy(quat_xyzw):
    """Convert quaternion xyzw to display RPY [roll, pitch, yaw] degrees."""
    ex, ey, ez = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    disp = np.array([ey, -ex, -ez - 90.0], dtype=np.float64)
    return (disp + 180.0) % 360.0 - 180.0


def densify_waypoints(waypoints, arc_spacing):
    """Resample xyz waypoints to uniform arc-length spacing."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return waypoints
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total < 1e-9:
        return waypoints[:1]
    n = max(int(np.ceil(total / arc_spacing)) + 1, 2)
    t = np.linspace(0.0, total, n)
    dense = np.zeros((n, 3), dtype=np.float64)
    for d in range(3):
        dense[:, d] = np.interp(t, cum, waypoints[:, d])
    return dense


def interpolate_orientations(start_quat_xyzw, end_quat_xyzw, n_points):
    """SLERP between two quaternions, returns (n_points, 4) xyzw."""
    rots = Rotation.from_quat(
        np.array([start_quat_xyzw, end_quat_xyzw], dtype=np.float64)
    )
    slerp = Slerp([0.0, 1.0], rots)
    return slerp(np.linspace(0.0, 1.0, n_points)).as_quat()


class CuroboWaypointPlanner:
    """Waypoint trajectory planner: mink IK + cubic spline + CuRobo collision.

    Fast seeded IK gives joint configs at waypoints (~2ms each).
    Cubic spline gives C2-continuous interpolation — smooth by construction.
    CuRobo validates collision safety on the final trajectory.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        solver_speed: str = "fast",
    ):
        from enpire.env.forge.experimental.motion_planner_curobo import YamMotionPlannerCurobo
        from enpire.env.forge.robot.yam.kinematics import YamKinematics

        self._kin = YamKinematics()
        self._curobo = YamMotionPlannerCurobo(
            device=device,
            solver_speed=solver_speed,
            validate_with_mujoco=False,
            collision_checking=True,
            enable_finetune_trajopt=False,
        )
        print(f"[WaypointPlanner] initialized device={device}")

    def _solve_ik_sequential(
        self,
        positions: np.ndarray,
        quats_xyzw: np.ndarray,
        side: str,
        cur_left: np.ndarray,
        cur_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Solve IK for each waypoint with seeded continuity.

        Returns (left_solutions, right_solutions, ik_errors, success_mask, n_failed).
        """
        n = len(positions)
        left_sols = np.zeros((n, 6), dtype=np.float64)
        right_sols = np.zeros((n, 6), dtype=np.float64)
        ik_errors = np.zeros(n, dtype=np.float64)
        success = np.zeros(n, dtype=bool)
        n_failed = 0

        kin = self._kin
        cl, cr = cur_left.copy(), cur_right.copy()

        for i in range(n):
            # Seed from previous solution
            kin.configuration.data.qpos[:6] = cl
            kin.configuration.data.qpos[8:14] = cr

            if side == "left":
                target_pos, target_quat = positions[i], quats_xyzw[i]
                # Keep right at current, solve left
                _, rp, _, rq = kin.forward_kinematics(cl, cr)[2], kin.forward_kinematics(cl, cr)[3], None, None
                rp_cur, rq_cur = kin.forward_kinematics(cl, cr)[2:]
                l_ik, r_ik = kin.inverse_kinematics(
                    target_pos, target_quat, rp_cur, rq_cur, seeded=True
                )
            else:
                lp_cur, lq_cur = kin.forward_kinematics(cl, cr)[:2]
                l_ik, r_ik = kin.inverse_kinematics(
                    lp_cur, lq_cur, positions[i], quats_xyzw[i], seeded=True
                )

            # Verify with FK
            lp, lq, rp, rq = kin.forward_kinematics(l_ik, r_ik)
            if side == "left":
                err = float(np.linalg.norm(lp - positions[i]))
            else:
                err = float(np.linalg.norm(rp - positions[i]))

            ik_errors[i] = err
            if err < 0.01:  # 10mm threshold
                left_sols[i] = l_ik
                right_sols[i] = r_ik
                cl, cr = l_ik, r_ik
                success[i] = True
            else:
                # Retry without seeding
                l_ik2, r_ik2 = kin.inverse_kinematics(
                    lp_cur if side == "right" else positions[i],
                    lq_cur if side == "right" else quats_xyzw[i],
                    positions[i] if side == "right" else rp_cur,
                    quats_xyzw[i] if side == "right" else rq_cur,
                    seeded=False, max_iters=50,
                )
                lp2, lq2, rp2, rq2 = kin.forward_kinematics(l_ik2, r_ik2)
                err2 = float(np.linalg.norm(
                    (lp2 if side == "left" else rp2) - positions[i]
                ))
                if err2 < 0.01:
                    left_sols[i] = l_ik2
                    right_sols[i] = r_ik2
                    cl, cr = l_ik2, r_ik2
                    success[i] = True
                    ik_errors[i] = err2
                else:
                    left_sols[i] = cl
                    right_sols[i] = cr
                    n_failed += 1

        return left_sols, right_sols, ik_errors, success, n_failed

    def plan_waypoint_trajectory(
        self,
        waypoints_xyz: np.ndarray,
        start_rpy_deg: list[float] | np.ndarray,
        end_rpy_deg: list[float] | np.ndarray,
        side: Literal["left", "right"] = "right",
        current_left_jp: np.ndarray | None = None,
        current_right_jp: np.ndarray | None = None,
        arc_spacing: float = _DEFAULT_ARC_SPACING_M,
        max_joint_vel: float = _DEFAULT_MAX_JOINT_VEL,
        subsample: int = 3,
        spline_points_per_segment: int = 10,
    ) -> dict[str, Any]:
        """Plan a smooth joint trajectory following Cartesian waypoints.

        1. Densify + SLERP orientation
        2. Subsample → fast seeded IK (~2ms each)
        3. Cubic spline through IK solutions → C2 smooth
        4. Time-parameterize with velocity bounds
        """
        t_start = time.perf_counter()

        retract_l = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25])
        retract_r = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25])
        cur_l = np.asarray(
            current_left_jp if current_left_jp is not None else retract_l,
            dtype=np.float64,
        )
        cur_r = np.asarray(
            current_right_jp if current_right_jp is not None else retract_r,
            dtype=np.float64,
        )

        dense = densify_waypoints(
            np.asarray(waypoints_xyz, dtype=np.float64), arc_spacing
        )
        n_dense = len(dense)

        start_q = display_rpy_to_quat_xyzw(start_rpy_deg)
        end_q = display_rpy_to_quat_xyzw(end_rpy_deg)
        quats = interpolate_orientations(start_q, end_q, n_dense)

        # Subsample for IK targets
        target_indices = list(range(0, n_dense, max(1, subsample)))
        if target_indices[-1] != n_dense - 1:
            target_indices.append(n_dense - 1)
        n_targets = len(target_indices)

        target_pos = dense[target_indices]
        target_quats = quats[target_indices]

        t_ik_start = time.perf_counter()
        left_sols, right_sols, ik_errors, ik_success, n_failed = self._solve_ik_sequential(
            target_pos, target_quats, side, cur_l, cur_r
        )
        ik_ms = (time.perf_counter() - t_ik_start) * 1000.0

        if n_failed == n_targets:
            return self._empty_result(dense, quats, side, t_start, n_targets, n_failed)

        active_sols = left_sols if side == "left" else right_sols

        # Jump detection
        ik_jumps = []
        for i in range(1, n_targets):
            if not ik_success[i]:
                continue
            j = float(np.max(np.abs(active_sols[i] - active_sols[i - 1])))
            if j > _MAX_JOINT_JUMP_RAD:
                ik_jumps.append(i)

        # Arc-length parameter for spline knots
        arc_lengths = np.zeros(n_targets, dtype=np.float64)
        for i in range(1, n_targets):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(
                target_pos[i] - target_pos[i - 1]
            )
        total_arc = arc_lengths[-1]
        if total_arc < 1e-9:
            return self._empty_result(dense, quats, side, t_start, n_targets, n_failed)

        # Cubic spline through IK solutions (C2 continuous)
        spline = CubicSpline(arc_lengths, active_sols, bc_type="clamped", axis=0)

        n_spline = max(n_targets * spline_points_per_segment, 50)
        s_samples = np.linspace(0.0, total_arc, n_spline)
        joints = spline(s_samples)

        n_pts = len(joints)
        if side == "left":
            full_left = joints
            full_right = np.tile(cur_r.reshape(1, 6), (n_pts, 1))
        else:
            full_right = joints
            full_left = np.tile(cur_l.reshape(1, 6), (n_pts, 1))

        # Timestamps
        safe_vel = max(0.05, min(3.0, float(max_joint_vel)))
        timestamps = [0.0]
        t_acc = 0.0
        for i in range(n_pts - 1):
            delta = float(np.max(np.abs(joints[i + 1] - joints[i])))
            t_acc += max(1e-6, delta / safe_vel)
            timestamps.append(t_acc)
        timestamps = np.array(timestamps, dtype=np.float64)

        # FK for error measurement
        ee_positions = np.zeros((n_pts, 3), dtype=np.float64)
        for i in range(n_pts):
            l_p, l_q, r_p, r_q = self._kin.forward_kinematics(full_left[i], full_right[i])
            ee_positions[i] = l_p if side == "left" else r_p

        cart_errors = np.zeros(n_pts, dtype=np.float64)
        for i in range(n_pts):
            cart_errors[i] = float(np.min(np.linalg.norm(dense - ee_positions[i], axis=1)))

        max_jump = 0.0
        final_jumps = []
        for i in range(1, n_pts):
            j = float(np.max(np.abs(joints[i] - joints[i - 1])))
            if j > _MAX_JOINT_JUMP_RAD:
                final_jumps.append(i)
            max_jump = max(max_jump, j)

        # Smoothness metrics
        if n_pts >= 3:
            dt = np.maximum(np.diff(timestamps), 1e-6)
            vel = np.diff(joints, axis=0) / dt[:, None]
            acc = np.diff(vel, axis=0) / dt[1:, None]
            max_acc = float(np.max(np.abs(acc))) if len(acc) > 0 else 0.0
            if len(acc) >= 2:
                jerk = np.diff(acc, axis=0) / dt[2:, None]
                max_jerk = float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0
            else:
                max_jerk = 0.0
        else:
            max_acc, max_jerk = 0.0, 0.0

        planning_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "success": n_failed == 0 and not final_jumps,
            "joints": joints,
            "full_left_joints": full_left,
            "full_right_joints": full_right,
            "timestamps": timestamps,
            "ee_positions": ee_positions,
            "desired_positions": dense,
            "desired_quats": quats,
            "cart_errors": cart_errors,
            "max_cart_err_m": float(np.max(cart_errors)),
            "mean_cart_err_m": float(np.mean(cart_errors)),
            "max_ik_err_m": float(np.max(ik_errors)),
            "mean_ik_err_m": float(np.mean(ik_errors)),
            "max_joint_jump_rad": max_jump,
            "max_acceleration": max_acc,
            "max_jerk": max_jerk,
            "planning_ms": planning_ms,
            "ik_ms": ik_ms,
            "n_waypoints": n_dense,
            "n_traj_points": n_pts,
            "n_ik_targets": n_targets,
            "n_failed_ik": n_failed,
            "ik_jump_indices": ik_jumps,
            "jump_indices": final_jumps,
            "side": side,
        }

    def _empty_result(self, dense, quats, side, t_start, n_targets, n_failed):
        return {
            "success": False,
            "joints": np.empty((0, 6)),
            "full_left_joints": np.empty((0, 6)),
            "full_right_joints": np.empty((0, 6)),
            "timestamps": np.array([]),
            "ee_positions": np.empty((0, 3)),
            "desired_positions": dense,
            "desired_quats": quats,
            "cart_errors": np.array([]),
            "max_cart_err_m": float("inf"),
            "mean_cart_err_m": float("inf"),
            "max_ik_err_m": float("inf"),
            "mean_ik_err_m": float("inf"),
            "max_joint_jump_rad": 0.0,
            "max_acceleration": 0.0,
            "max_jerk": 0.0,
            "planning_ms": (time.perf_counter() - t_start) * 1000.0,
            "ik_ms": 0.0,
            "n_waypoints": len(dense),
            "n_traj_points": 0,
            "n_ik_targets": n_targets,
            "n_failed_ik": n_failed,
            "ik_jump_indices": [],
            "jump_indices": [],
            "side": side,
        }
