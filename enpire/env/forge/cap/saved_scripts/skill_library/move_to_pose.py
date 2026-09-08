"""Generic measured-pose wrapper around the registered CaP freespace_move tool.

Positions are world-frame metres. Quaternions are XYZW; RPY uses the existing
CaP display convention in degrees (not conventional XYZ Euler angles).
Omitted orientation preserves the measured orientation. Grippers are untouched.
No target relaxation, direct motor commands, or retries against an obstruction.
"""

from __future__ import annotations

import math
import time

import numpy as np


class PoseNotReachedError(RuntimeError):
    """The planner finished, but measured pose did not settle at the target."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            f"{report['side']} pose not reached: "
            f"position error={report['position_error_m']:.4f} m, "
            f"rotation error={report['rotation_error_deg']:.2f} deg; "
            "no recovery motion or gripper command issued"
        )


def _vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _quaternion(value):
    quat = _vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must have finite, nonzero length")
    return quat / norm


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _read_pose(tools, side):
    arm = _field(tools.get_robot_state(), "arms")[side]
    return (
        _vector(_field(arm, "ee_pos"), 3, "measured position"),
        _quaternion(_field(arm, "ee_quat")),
    )


def move_to_pose(
    side,
    target_pos=None,
    target_rpy=None,
    target_quat=None,
    *,
    planning_speed=None,
    position_tolerance_m=0.005,
    rotation_tolerance_deg=3.0,
    settle_timeout_s=1.5,
    stable_samples=3,
    preview_only=False,
):
    """Plan and execute an exact pose, then require stable measured convergence.

    Uses only the harness's freespace_move and get_robot_state tools. A failed
    plan propagates its original exception. A measured miss raises
    PoseNotReachedError with a structured report; callers must not close or
    release a gripper after that failure. Preview plans without moving and
    explicitly reports pose_verified=False.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right, as supported by freespace_move")
    if target_pos is None and target_rpy is None and target_quat is None:
        raise ValueError("at least one pose component is required")
    if target_rpy is not None and target_quat is not None:
        raise ValueError("supply target_rpy or target_quat, not both")
    for name, value in (
        ("position_tolerance_m", position_tolerance_m),
        ("rotation_tolerance_deg", rotation_tolerance_deg),
        ("settle_timeout_s", settle_timeout_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if planning_speed is not None and (not math.isfinite(planning_speed) or planning_speed <= 0):
        raise ValueError("planning_speed must be positive and finite")
    if type(stable_samples) is not int or not 1 <= stable_samples <= 20:
        raise ValueError("stable_samples must be an integer from 1 to 20")
    if settle_timeout_s < (stable_samples - 1) * 0.05:
        raise ValueError("settle_timeout_s is too short for stable_samples")
    if type(preview_only) is not bool:
        raise ValueError("preview_only must be a boolean")
    position = None if target_pos is None else _vector(target_pos, 3, "target_pos")
    rpy = None if target_rpy is None else _vector(target_rpy, 3, "target_rpy")
    quat = None if target_quat is None else _quaternion(target_quat)

    # Resolve the live harness namespace at call time; importing this script
    # neither creates an environment nor connects to a robot.
    import skill_library.namespace as tools
    from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool

    start_pos, start_quat = _read_pose(tools, side)
    position = start_pos if position is None else position
    if quat is None:
        quat = start_quat if rpy is None else FreespaceMoveTool._display_rpy_to_quat(rpy)
    kwargs = {
        f"{side}_target_pos": position.tolist(),
        f"{side}_target_quat": quat.tolist(),
        "preview_only": preview_only,
    }
    if planning_speed is not None:
        kwargs["planning_speed"] = float(planning_speed)
    # Single-target freespace_move retains its existing planner/collision path.
    # Never pass gripper targets, cached paths, or collision overrides here.
    motion = tools.freespace_move(**kwargs)
    if _field(motion, "status") != "Success":
        raise RuntimeError(f"freespace_move did not succeed: {motion}")
    report = {
        "side": side,
        "target_pos": position.tolist(),
        "target_quat": quat.tolist(),
        "pose_verified": False,
        "preview_only": preview_only,
    }
    if preview_only:
        return report

    deadline = time.monotonic() + settle_timeout_s
    consecutive = 0
    while True:
        actual_pos, actual_quat = _read_pose(tools, side)
        pos_error = float(np.linalg.norm(actual_pos - position))
        rot_error = FreespaceMoveTool._quat_error_deg(quat, actual_quat)
        reached = pos_error <= position_tolerance_m and rot_error <= rotation_tolerance_deg
        consecutive = consecutive + 1 if reached else 0
        report.update(
            actual_pos=actual_pos.tolist(),
            actual_quat=actual_quat.tolist(),
            position_error_m=pos_error,
            rotation_error_deg=float(rot_error),
        )
        if consecutive >= stable_samples:
            report["pose_verified"] = True
            return report
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PoseNotReachedError(report)
        time.sleep(min(0.05, remaining))
