"""Generic measured-pose wrapper around the registered CaP freespace_move tool.

Positions are world-frame metres. Quaternions are XYZW; RPY uses the existing
CaP display convention in degrees (not conventional XYZ Euler angles).
Omitted orientation preserves the measured orientation. Optional gripper actions
run before planning or after verified motion (0=closed, 1=open).
Optional bounded tracking corrections preserve the original measured-pose goal
and replan every correction through the registered collision planner.
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
            "no recovery motion or after-motion gripper command issued"
        )


class GripperCommandError(RuntimeError):
    """A requested gripper command failed or a before-action is unfinished."""

    def __init__(self, phase, report):
        self.report = report
        super().__init__(f"{phase} command failed or did not settle: {report[phase]}")


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
    gripper_before=None,
    gripper_after=None,
    gripper_vel_limit=None,
    gripper_torque_limit=None,
    planning_speed=None,
    position_tolerance_m=0.005,
    rotation_tolerance_deg=3.0,
    settle_timeout_s=1.5,
    stable_samples=3,
    preview_only=False,
    max_corrections=0,
    max_position_correction_m=0.03,
    max_rotation_correction_deg=6.0,
):
    """Plan and execute an exact pose, then require stable measured convergence.

    Uses the harness's freespace_move, get_robot_state and set_gripper tools. A failed
    plan propagates its original exception. A measured miss raises
    PoseNotReachedError with a structured report; callers must not close or
    release a gripper after that failure. Before-actions execute before planning,
    so a subsequent planning failure does not undo them. After-actions require
    stable measured pose convergence. Preview never actuates either gripper;
    its plan uses the current gripper geometry. Feedback is reported unchanged
    and does not establish object retention for a UMI gripper.

    max_corrections opts into bounded compensation for small tracking offsets
    in observed free space. The original tolerances are never relaxed. A
    correction must improve the measured error; a stagnant correction stops.
    This is not a contact-search mode. Check scene/payload clearance first.
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
        ("max_position_correction_m", max_position_correction_m),
        ("max_rotation_correction_deg", max_rotation_correction_deg),
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
    if type(max_corrections) is not int or not 0 <= max_corrections <= 4:
        raise ValueError("max_corrections must be an integer from 0 to 4")
    if max_position_correction_m > 0.03 or max_rotation_correction_deg > 6.0:
        raise ValueError("tracking corrections are limited to 0.03 m and 6 degrees")
    position = None if target_pos is None else _vector(target_pos, 3, "target_pos")
    rpy = None if target_rpy is None else _vector(target_rpy, 3, "target_rpy")
    quat = None if target_quat is None else _quaternion(target_quat)
    from enpire.env.forge.cap.saved_scripts.skill_library.set_gripper import gripper_kwargs

    # Validate both commands before any read, gripper command, or arm motion.
    limits = {"vel_limit": gripper_vel_limit, "torque_limit": gripper_torque_limit}
    gripper_kwargs(side, 0.0, **limits)
    before = None if gripper_before is None else gripper_kwargs(side, gripper_before, **limits)
    after = None if gripper_after is None else gripper_kwargs(side, gripper_after, **limits)

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
    report = {
        "side": side,
        "start_pos": start_pos.tolist(),
        "start_quat": start_quat.tolist(),
        "target_pos": position.tolist(),
        "target_quat": quat.tolist(),
        "pose_verified": False,
        "preview_only": preview_only,
        "gripper_before": {"requested": gripper_before, "executed": False},
        "gripper_after": {"requested": gripper_after, "executed": False},
        "attempts": [],
    }

    def command_gripper(phase, command):
        feedback = tools.set_gripper(**command)
        report[phase].update(executed=True, feedback=feedback)
        success = feedback.get("success") if isinstance(feedback, dict) else getattr(feedback, "success", None)
        settled = feedback.get("settled") if isinstance(feedback, dict) else getattr(feedback, "settled", None)
        if success is not True or (phase == "gripper_before" and settled is False):
            raise GripperCommandError(phase, report)

    if before is not None and not preview_only:
        command_gripper("gripper_before", before)
    command_pos, command_quat = position.copy(), quat.copy()
    previous_error = None
    for attempt in range(max_corrections + 1):
        kwargs[f"{side}_target_pos"] = command_pos.tolist()
        kwargs[f"{side}_target_quat"] = command_quat.tolist()
        # Every correction takes the ordinary planner/collision path. No
        # cached path, motor command, gain change, or collision override.
        motion = tools.freespace_move(**kwargs)
        if _field(motion, "status") != "Success":
            raise RuntimeError(f"freespace_move did not succeed: {motion}")
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
                actual_pos=actual_pos.tolist(), actual_quat=actual_quat.tolist(),
                position_error_m=pos_error, rotation_error_deg=float(rot_error),
            )
            if consecutive >= stable_samples or time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, deadline - time.monotonic()))
        report["attempts"].append({
            "command_pos": command_pos.tolist(), "command_quat": command_quat.tolist(),
            "position_error_m": pos_error, "rotation_error_deg": float(rot_error),
        })
        if consecutive >= stable_samples:
            report["pose_verified"] = True
            if after is not None:
                command_gripper("gripper_after", after)
            return report
        error = max(pos_error / position_tolerance_m, rot_error / rotation_tolerance_deg)
        if attempt >= max_corrections or (previous_error is not None and error >= previous_error * 0.9):
            raise PoseNotReachedError(report)
        from scipy.spatial.transform import Rotation

        next_pos = command_pos + position - actual_pos
        next_quat = (
            Rotation.from_quat(quat) * Rotation.from_quat(actual_quat).inv()
            * Rotation.from_quat(command_quat)
        ).as_quat()
        if (
            np.linalg.norm(next_pos - position) > max_position_correction_m
            or FreespaceMoveTool._quat_error_deg(next_quat, quat) > max_rotation_correction_deg
        ):
            raise PoseNotReachedError(report)
        previous_error = error
        command_pos, command_quat = next_pos, next_quat
