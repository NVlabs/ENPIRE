# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical orientation constants + display-RPY ↔ quaternion conversion.

Moved out of cap/constants/planning.py on 2026-05-20 so planning.py stays focused
on the cuRobo IK / planner threshold constants and the orientation configuration
+ helper functions live in their own file.

All callers should import from `cap.constants` (e.g.
`from cap.constants import TOPDOWN_RPY, display_rpy_to_quat_xyzw`) — the
re-export there is the stable public surface. Importing from
`cap.constants.configuration` directly works too but pins to this file path.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Top-down grasp orientation in the freespace_move / planner RPY convention
# (roll, pitch, yaw in degrees). Pitch = 180° puts the gripper pointing -Z
# in world frame; yaw rotates the fingers freely about the down axis without
# losing the top-down character. Mirrors the planner_rpy emitted by
# sample_grasp_pose_2d so freespace_move and the 2D grasp path agree.
TOPDOWN_RPY: list[float] = [0.0, 180.0, 0.0]


# -----------------------------------------------------------------------------
# Display-RPY conversion — single source of truth for the entire repo.
#
# "Display RPY" is the [roll, pitch, yaw] convention exposed to scripts via
# get_robot_state().*_ee_rpy and consumed by freespace_move's *_target_rpy
# arguments. It is NOT raw scipy "xyz" Euler — it applies a remap so that
# the canonical top-down pose lands at [0, 180, yaw_about_z]:
#
#     display_roll  =  scipy_pitch
#     display_pitch = -scipy_roll
#     display_yaw   = -(scipy_yaw + 90°)
#
# Both functions below NORMALIZE the result to [-180°, +180°] so callers
# never see values like -225.8° (which is equivalent to +134.2° but visually
# confusing). Every conversion in the repo should go through these — do NOT
# inline the remap. See git history (2026-05-20 refactor) for the previous
# 5+ inconsistent copies that prompted this consolidation.
# -----------------------------------------------------------------------------


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to the half-open interval [-180°, +180°)."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def quat_xyzw_to_display_rpy(quat_xyzw) -> list[float]:
    """Quaternion (xyzw) → display RPY [roll, pitch, yaw] in degrees, wrapped
    to [-180°, +180°]. Returns plain list[float] (always 3 elements)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)[:4]
    e = Rotation.from_quat(q).as_euler("xyz", degrees=True)
    disp = np.array([e[1], -e[0], -(e[2] + 90.0)], dtype=np.float64)
    disp = (disp + 180.0) % 360.0 - 180.0
    return [float(disp[0]), float(disp[1]), float(disp[2])]


def display_rpy_to_quat_xyzw(rpy_deg) -> list[float]:
    """Display RPY [roll, pitch, yaw] in degrees → quaternion (xyzw). Inverse
    of quat_xyzw_to_display_rpy. Returns plain list[float] (always 4 elements,
    xyzw order). Input is wrapped to [-180°, +180°] first — same orientation
    either way, but keeps internals canonical."""
    arr = np.asarray(rpy_deg, dtype=np.float64).reshape(-1)[:3]
    roll, pitch, yaw = (_wrap_deg(v) for v in arr)
    quat = Rotation.from_euler(
        "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
    ).as_quat()
    return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


# -----------------------------------------------------------------------------
# Handedness transform — reflect an EE pose across YAM's left/right symmetry
# plane (world Y=0, the XZ plane). Use this whenever you have a pose authored
# for one arm and want the equivalent pose for the other arm.
#
# This is the single source of truth for "swap arm roles" mirroring across
# the repo. Use it in handover scripts, role-swap test poses, and anywhere
# you've captured a pose with one arm via gravity-comp dragging and want to
# play it back on the other. See git history (2026-05-20) for the iterative
# discovery that pinned down the roll/pitch/yaw flip rule.
# -----------------------------------------------------------------------------


def handedness_transform_ee(xyz, rpy):
    """Reflect an end-effector pose (xyz, rpy) across the XZ plane (y=0), the
    plane that separates YAM's left arm from its right arm. Equivalent to
    swapping the arm that should reach the pose.

    Mirror rule (empirically validated on YAM, display-rpy convention):
        xyz: (x, y, z) → (x, -y, z)
        rpy: (roll, pitch, yaw) → (-roll, pitch, -yaw)

    Why roll and yaw flip but pitch doesn't:
      - roll  rotates the gripper about its open-close axis. Swapping arms
              flips that axis → roll sign flips.
      - pitch is the gripper down/up tilt. Same physical motion on either
              arm (display-rpy is symmetric across the mount) → unchanged.
      - yaw   rotates about world Z. Arm-swap reverses chirality → flips.

    Returns:
        (xyz_mirrored, rpy_mirrored), both as plain list[float].

    See also: handedness_transform_joint() for the joint-space equivalent
    used when mirroring a sequence of rotate_joint deltas.
    """
    return (
        [float(xyz[0]), float(-xyz[1]), float(xyz[2])],
        [float(-rpy[0]), float(rpy[1]), float(-rpy[2])],
    )


# Joints whose rotation-axis sign flips under YAM's XZ-plane handedness swap.
# Empirically: joints 1, 5, 6 reverse, joints 2, 3, 4 stay. This matches the
# observation that the EE-rpy roll & yaw flip while pitch stays — j1 is the
# arm yaw, j5/j6 are the wrist pitch/roll that compose into EE roll/yaw.
_HANDEDNESS_FLIP_JOINTS: frozenset[int] = frozenset({1, 5, 6})


def handedness_transform_joint(rotations, *, mirror: bool):
    """Joint-space companion to handedness_transform_ee — mirror a sequence of
    rotate_joint(rotations=[{joint, side, delta_deg, ...}, ...]) entries across
    YAM's XZ symmetry plane.

    Rule:
        For each rotation dict, if mirror=True AND the joint index is in
        {1, 5, 6} (the joints whose rotation axis flips under the mirror),
        negate its delta_deg. Other fields (side, joint, anything else)
        pass through unchanged.

    Use this when authoring rotate_joint deltas for ONE handedness (typically
    holding=left), then sign-flip them automatically for the other (holding=right):

        rotate_joint(rotations=handedness_transform_joint([
            {"side": other,   "joint": 5, "delta_deg":   2.0},
            {"side": other,   "joint": 6, "delta_deg": -164.0},
            {"side": holding, "joint": 2, "delta_deg":  10.0},   # j2 stays
        ], mirror=(holding == "right")))

    Returns a NEW list[dict]; input is not mutated.
    """
    rotations = list(rotations)
    if not mirror:
        return [dict(r) for r in rotations]
    out = []
    for r in rotations:
        rr = dict(r)
        if int(rr.get("joint", -1)) in _HANDEDNESS_FLIP_JOINTS and "delta_deg" in rr:
            rr["delta_deg"] = -float(rr["delta_deg"])
        out.append(rr)
    return out
