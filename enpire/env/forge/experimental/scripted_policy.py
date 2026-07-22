# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted policy for interactive EE control via Viser UI.

Controlled entirely through the Viser web panel — no CLI / keyboard input.
The Viser UI sends commands via Portal RPC. Three control modes:

1. **6D Gizmo teleop** (primary): draggable transform controls in the 3D
   scene send absolute EE targets at ~30 Hz via ``set_target_pose()`` /
   ``set_target_pose_bimanual()``.
2. **Move-To panel**: absolute Cartesian target sliders with optional RRT
   motion planning via ``move_to()`` / ``execute_trajectory()``.
3. **Button nudge** (legacy): axis-aligned delta buttons via ``apply_nudge()``.

**Key design**: the policy operates in ``joint_position`` mode.  It keeps an
internal absolute joint-position buffer that is updated when a new target
arrives (IK -> new joint target).  Between updates, ``get_action()`` returns
the **exact same** joint command every step, so the motor servos lock the
robot firmly in place.

**Motion smoothing**: when a new target is computed the policy interpolates
from the current commanded joints to the new target over a configurable
number of steps in joint space.  Once the interpolation finishes the command
is constant until the next target.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import termcolor
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

from enpire.env.forge.robot.yam.kinematics import YamKinematics

SCRIPTED_IK_POSITION_COST = 1.0
SCRIPTED_IK_ORIENTATION_COST = 0.05


# ---------------------------------------------------------------------------
# Safety configuration (kept for API compatibility with yam_control_loop)
# ---------------------------------------------------------------------------
from enpire.env.forge.robot.constants import MAX_JOINT_VELOCITY_RAD_S

@dataclass
class SafetyLimits:
    """Per-step safety clamp.

    ``max_joint_velocity_rad_s`` is the velocity limit in **rad/s**.
    The per-step delta is computed at runtime as:
        max_joint_step = max_joint_velocity_rad_s / control_hz

    Example at 30 Hz:  6 / 30 = 0.2 rad/step  →  6 rad/s  (~344 deg/s)
    """

    max_joint_velocity_rad_s: float = MAX_JOINT_VELOCITY_RAD_S  # rad/s — max safe joint speed


# ---------------------------------------------------------------------------
# ScriptedPolicy
# ---------------------------------------------------------------------------


class ScriptedPolicy:
    """Viser-UI-driven policy that emits absolute ``joint_position`` actions.

    Commands arrive from the Viser UI through methods called by
    ``StartStopPlayPolicyWrapper._apply_scripted_command``:

    * ``set_target_pose(side, pos, quat_xyzw)`` -- absolute EE target (gizmo)
    * ``set_target_pose_bimanual(...)`` -- both arms at once (gizmo)
    * ``move_to(...)`` -- absolute EE target with per-arm speed (Move-To panel)
    * ``execute_trajectory(...)`` -- pre-planned joint trajectory (RRT planner)
    * ``apply_nudge(side, delta_pos, delta_quat_xyzw)`` -- delta nudge (legacy)
    * ``set_gripper(side, value)``

    Between commands, ``get_action()`` returns the same joint command every
    call, so the robot's servos hold position firmly.
    """

    HOME_GRIPPER: float = 1.0  # open
    SMOOTHING_STEPS: int = 15  # steps to interpolate a nudge (~0.5 s @ 30 Hz)
    GIZMO_SMOOTHING_STEPS: int = 3  # fewer steps for continuous gizmo tracking
    # Default distance per step (m) for move_to when auto-scaling by L2 distance.
    # steps = distance / DISTANCE_PER_STEP, clamped to [MIN_SMOOTHING_STEPS, MAX_SMOOTHING_STEPS].
    DEFAULT_DISTANCE_PER_STEP: float = 0.01  # 1 cm/step @ 30 Hz ~ 0.3 m/s
    MIN_SMOOTHING_STEPS: int = 5
    MAX_SMOOTHING_STEPS: int = 150
    DEFAULT_JOINT_RECORD_PATH: Path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "yam"
        / "ScriptedJointRecord.parquet"
    )

    def __init__(
        self,
        action_space: Any = None,
        control_hz: float = 30.0,
        safety: SafetyLimits | None = None,
    ) -> None:
        self._action_space = action_space
        self._control_hz = control_hz
        self._safety = safety or SafetyLimits()

        # Internal kinematics for FK + IK
        self._kin = YamKinematics(
            position_cost=SCRIPTED_IK_POSITION_COST,
            orientation_cost=SCRIPTED_IK_ORIENTATION_COST,
        )

        # Absolute gripper targets (0 = closed, 1 = open)
        self._gripper: dict[str, float] = {
            "left": self.HOME_GRIPPER,
            "right": self.HOME_GRIPPER,
        }

        # --- Joint-space command state (protected by _lock) ---
        self._lock = threading.Lock()

        # The final target joints.
        self._target_joints: dict[str, np.ndarray] | None = None

        # What we're currently commanding (interpolated toward _target_joints).
        self._command_joints: dict[str, np.ndarray] | None = None

        # How many interpolation steps remain per arm (0 = idle / holding).
        self._steps_left_left: int = 0
        self._steps_left_right: int = 0

        # Buffered commanded joint trajectory (recorded at every get_action call).
        self._joint_record: list[dict[str, Any]] = []
        self._record_step_idx: int = 0

        # Queued trajectory waypoints from motion planner (consumed in get_action).
        self._trajectory_queue: list[dict[str, np.ndarray]] | None = None
        self._trajectory_idx: int = 0

    # ------------------------------------------------------------------
    # Public API called by StartStopPlayPolicyWrapper
    # ------------------------------------------------------------------

    def apply_nudge(
        self,
        side: str,
        delta_pos: np.ndarray | None = None,
        delta_quat_xyzw: np.ndarray | None = None,
        ik_error_threshold: float = 3e-3,
    ) -> bool:
        """Compute a new joint target by applying an EE delta to the current target.

        Returns True if IK converged, False if infeasible (target unchanged).
        """
        if side not in ("left", "right"):
            print(f"[ScriptedPolicy] Unknown side '{side}', ignoring nudge")
            return False

        with self._lock:
            if self._target_joints is None:
                print("[ScriptedPolicy] Not yet initialized, ignoring nudge")
                return False

            left_jp = self._target_joints["left_joint_pos"]
            right_jp = self._target_joints["right_joint_pos"]
            cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kin.forward_kinematics(
                left_jp, right_jp
            )

            if side == "left":
                cur_pos, cur_q = cur_l_pos.copy(), cur_l_q.copy()
            else:
                cur_pos, cur_q = cur_r_pos.copy(), cur_r_q.copy()

            new_pos = cur_pos
            new_rot = Rotation.from_quat(cur_q)
            if delta_pos is not None:
                new_pos = new_pos + np.asarray(delta_pos, dtype=np.float64)
            if delta_quat_xyzw is not None:
                delta_rot = Rotation.from_quat(
                    np.asarray(delta_quat_xyzw, dtype=np.float64)
                )
                new_rot = delta_rot * new_rot
            new_q = new_rot.as_quat()

            if side == "left":
                new_left_jp, new_right_jp = self._kin.inverse_kinematics(
                    new_pos,
                    new_q,
                    cur_r_pos,
                    cur_r_q,
                    seeded=True,
                )
            else:
                new_left_jp, new_right_jp = self._kin.inverse_kinematics(
                    cur_l_pos,
                    cur_l_q,
                    new_pos,
                    new_q,
                    seeded=True,
                )

            got_l_pos, _, got_r_pos, _ = self._kin.forward_kinematics(
                new_left_jp, new_right_jp
            )
            pos_err = float(
                np.linalg.norm(got_l_pos - new_pos)
                if side == "left"
                else np.linalg.norm(got_r_pos - new_pos)
            )
            if pos_err > ik_error_threshold:
                print(
                    f"[ScriptedPolicy] apply_nudge rejected (IK pos error {pos_err:.4f} m)"
                )
                return False

            self._target_joints = {
                "left_joint_pos": new_left_jp.copy(),
                "right_joint_pos": new_right_jp.copy(),
            }
            self._steps_left_left = self.SMOOTHING_STEPS
            self._steps_left_right = self.SMOOTHING_STEPS
            return True

    def set_gripper(self, side: str, value: float) -> None:
        """Set the absolute gripper target for *side* (0 = closed, 1 = open)."""
        if side not in ("left", "right"):
            return
        with self._lock:
            self._gripper[side] = float(np.clip(value, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 6D Gizmo teleop
    # ------------------------------------------------------------------

    def set_target_pose(
        self,
        side: str,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
    ) -> None:
        """Set an absolute EE target for one arm. IK is solved immediately."""
        if side not in ("left", "right"):
            return

        with self._lock:
            if self._target_joints is None:
                return

            left_jp = self._target_joints["left_joint_pos"]
            right_jp = self._target_joints["right_joint_pos"]
            cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kin.forward_kinematics(
                left_jp, right_jp
            )

            target_pos = np.asarray(pos, dtype=np.float64)
            target_q = np.asarray(quat_xyzw, dtype=np.float64)

            if side == "left":
                new_left_jp, new_right_jp = self._kin.inverse_kinematics(
                    target_pos,
                    target_q,
                    cur_r_pos,
                    cur_r_q,
                    seeded=True,
                )
            else:
                new_left_jp, new_right_jp = self._kin.inverse_kinematics(
                    cur_l_pos,
                    cur_l_q,
                    target_pos,
                    target_q,
                    seeded=True,
                )

            self._target_joints = {
                "left_joint_pos": new_left_jp.copy(),
                "right_joint_pos": new_right_jp.copy(),
            }
            self._steps_left_left = max(
                self._steps_left_left, self.GIZMO_SMOOTHING_STEPS
            )
            self._steps_left_right = max(
                self._steps_left_right, self.GIZMO_SMOOTHING_STEPS
            )

    def set_target_pose_bimanual(
        self,
        left_pos: np.ndarray,
        left_quat_xyzw: np.ndarray,
        right_pos: np.ndarray,
        right_quat_xyzw: np.ndarray,
    ) -> None:
        """Set absolute EE targets for both arms. Single bimanual IK solve."""
        with self._lock:
            if self._target_joints is None:
                return

            new_left_jp, new_right_jp = self._kin.inverse_kinematics(
                np.asarray(left_pos, dtype=np.float64),
                np.asarray(left_quat_xyzw, dtype=np.float64),
                np.asarray(right_pos, dtype=np.float64),
                np.asarray(right_quat_xyzw, dtype=np.float64),
                seeded=True,
            )
            self._target_joints = {
                "left_joint_pos": new_left_jp.copy(),
                "right_joint_pos": new_right_jp.copy(),
            }
            self._steps_left_left = max(
                self._steps_left_left, self.GIZMO_SMOOTHING_STEPS
            )
            self._steps_left_right = max(
                self._steps_left_right, self.GIZMO_SMOOTHING_STEPS
            )

    def get_current_ee_poses(self) -> dict | None:
        """FK current target joints and return EE poses. Used for gizmo snapping."""
        with self._lock:
            if self._target_joints is None:
                return None
            l_pos, l_q, r_pos, r_q = self._kin.forward_kinematics(
                self._target_joints["left_joint_pos"],
                self._target_joints["right_joint_pos"],
            )
            return {
                "left_pos": l_pos.tolist(),
                "left_quat_xyzw": l_q.tolist(),
                "right_pos": r_pos.tolist(),
                "right_quat_xyzw": r_q.tolist(),
            }

    def get_current_state(self) -> dict | None:
        """Return the best available current commanded state for planning.

        Used by the motion planner to get the starting configuration.
        """
        with self._lock:
            if self._target_joints is None and self._command_joints is None:
                return None
            joints = self._command_joints or self._target_joints
            return {
                "left_joint_pos": joints["left_joint_pos"].copy(),
                "right_joint_pos": joints["right_joint_pos"].copy(),
                "left_gripper": self._gripper.get("left", 1.0),
                "right_gripper": self._gripper.get("right", 1.0),
            }

    # ------------------------------------------------------------------
    # Move-To (absolute target with per-arm speed)
    # ------------------------------------------------------------------

    def _solve_ik_multi_seed(
        self,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        tgt_l_pos: np.ndarray,
        tgt_l_q: np.ndarray,
        tgt_r_pos: np.ndarray,
        tgt_r_q: np.ndarray,
        threshold: float,
        n_random_seeds: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Multi-seed IK: try seeded from current config, then random seeds."""
        self._kin.forward_kinematics(current_left_jp, current_right_jp)
        best_l, best_r = self._kin.inverse_kinematics(
            tgt_l_pos,
            tgt_l_q,
            tgt_r_pos,
            tgt_r_q,
            seeded=True,
            max_iters=100,
        )
        got_l, _, got_r, _ = self._kin.forward_kinematics(best_l, best_r)
        best_l_err = float(np.linalg.norm(got_l - tgt_l_pos))
        best_r_err = float(np.linalg.norm(got_r - tgt_r_pos))

        if best_l_err <= threshold and best_r_err <= threshold:
            return best_l, best_r, best_l_err, best_r_err

        for _ in range(n_random_seeds):
            gl, gr = self._kin.inverse_kinematics(
                tgt_l_pos,
                tgt_l_q,
                tgt_r_pos,
                tgt_r_q,
                seeded=False,
                max_iters=100,
            )
            got_l, _, got_r, _ = self._kin.forward_kinematics(gl, gr)
            l_err = float(np.linalg.norm(got_l - tgt_l_pos))
            r_err = float(np.linalg.norm(got_r - tgt_r_pos))
            if l_err <= threshold and r_err <= threshold:
                return gl, gr, l_err, r_err
            if max(l_err, r_err) < max(best_l_err, best_r_err):
                best_l, best_r = gl, gr
                best_l_err, best_r_err = l_err, r_err

        return best_l, best_r, best_l_err, best_r_err

    def move_to(
        self,
        left_target_pos: np.ndarray | None = None,
        left_target_quat_xyzw: np.ndarray | None = None,
        left_target_gripper: float | None = None,
        right_target_pos: np.ndarray | None = None,
        right_target_quat_xyzw: np.ndarray | None = None,
        right_target_gripper: float | None = None,
        left_distance_per_step: float | None = None,
        right_distance_per_step: float | None = None,
        ik_error_threshold: float = 0.005,
    ) -> bool:
        """Move to absolute EE target. Per-arm speed from distance_per_step."""
        with self._lock:
            if self._target_joints is None:
                print("[ScriptedPolicy] Not yet initialized, ignoring move_to")
                return False

            left_jp = self._target_joints["left_joint_pos"]
            right_jp = self._target_joints["right_joint_pos"]
            cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kin.forward_kinematics(
                left_jp, right_jp
            )

            tgt_l_pos = (
                np.asarray(left_target_pos, dtype=np.float64)
                if left_target_pos is not None
                else cur_l_pos.copy()
            )
            tgt_l_q = (
                np.asarray(left_target_quat_xyzw, dtype=np.float64)
                if left_target_quat_xyzw is not None
                else cur_l_q.copy()
            )
            tgt_r_pos = (
                np.asarray(right_target_pos, dtype=np.float64)
                if right_target_pos is not None
                else cur_r_pos.copy()
            )
            tgt_r_q = (
                np.asarray(right_target_quat_xyzw, dtype=np.float64)
                if right_target_quat_xyzw is not None
                else cur_r_q.copy()
            )

            tgt_l_q = tgt_l_q / (np.linalg.norm(tgt_l_q) + 1e-12)
            tgt_r_q = tgt_r_q / (np.linalg.norm(tgt_r_q) + 1e-12)

            new_left_jp, new_right_jp, l_err, r_err = self._solve_ik_multi_seed(
                left_jp,
                right_jp,
                tgt_l_pos,
                tgt_l_q,
                tgt_r_pos,
                tgt_r_q,
                ik_error_threshold,
            )
            if l_err > ik_error_threshold or r_err > ik_error_threshold:
                print(
                    termcolor.colored(
                        f"[ScriptedPolicy] move_to rejected (IK infeasible): "
                        f"left pos_err={l_err:.4f} m, right pos_err={r_err:.4f} m "
                        f"(threshold={ik_error_threshold:.4f} m)",
                        "red",
                    )
                )
                return False

            self._target_joints = {
                "left_joint_pos": new_left_jp.copy(),
                "right_joint_pos": new_right_jp.copy(),
            }

            if left_target_gripper is not None:
                self._gripper["left"] = float(np.clip(left_target_gripper, 0.0, 1.0))
            if right_target_gripper is not None:
                self._gripper["right"] = float(np.clip(right_target_gripper, 0.0, 1.0))

            def _steps_for_arm(cur_pos, tgt_pos, dist_per_step):
                dist = float(np.linalg.norm(tgt_pos - cur_pos))
                d = (
                    dist_per_step
                    if dist_per_step is not None
                    else self.DEFAULT_DISTANCE_PER_STEP
                )
                d = max(d, 1e-6)
                return max(
                    self.MIN_SMOOTHING_STEPS,
                    min(self.MAX_SMOOTHING_STEPS, int(round(dist / d))),
                )

            self._steps_left_left = (
                _steps_for_arm(cur_l_pos, tgt_l_pos, left_distance_per_step)
                if left_target_pos is not None
                else 0
            )
            self._steps_left_right = (
                _steps_for_arm(cur_r_pos, tgt_r_pos, right_distance_per_step)
                if right_target_pos is not None
                else 0
            )
            return True

    @property
    def is_moving(self) -> bool:
        """True while the policy is still interpolating toward a target."""
        with self._lock:
            if self._trajectory_queue is not None:
                return True
            return self._steps_left_left > 0 or self._steps_left_right > 0

    def _retime_joint_trajectory(
        self,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        *,
        max_joint_vel: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample a joint trajectory to satisfy a max joint velocity."""
        if left_positions.shape[0] == 0:
            return left_positions, right_positions

        full_path = np.concatenate([left_positions, right_positions], axis=1)
        if full_path.shape[0] <= 1:
            return full_path[:, :6], full_path[:, 6:]

        dt = 1.0 / max(float(self._control_hz), 1e-6)
        safe_vel = (
            max(0.05, float(max_joint_vel))
            if max_joint_vel is not None
            else max(0.05, float(self._safety.max_joint_velocity_rad_s))
        )

        positions: list[np.ndarray] = []
        for k in range(full_path.shape[0] - 1):
            diff = full_path[k + 1] - full_path[k]
            max_delta = float(np.max(np.abs(diff)))
            seg_time = max(max_delta / safe_vel, dt)
            n_steps = max(1, int(np.ceil(seg_time / dt)))
            for s in range(n_steps):
                t = s / n_steps
                positions.append(full_path[k] + t * diff)
        positions.append(full_path[-1].copy())

        retimed = np.asarray(positions, dtype=np.float64)
        return retimed[:, :6], retimed[:, 6:]

    def execute_trajectory(
        self,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
        max_joint_vel: float | None = None,
        current_left_joint_pos: np.ndarray | None = None,
        current_right_joint_pos: np.ndarray | None = None,
    ) -> None:
        """Queue a pre-planned joint trajectory for execution."""
        assert left_positions.shape[0] == right_positions.shape[0]
        with self._lock:
            if self._target_joints is None:
                print(
                    "[ScriptedPolicy] Not yet initialized, ignoring execute_trajectory"
                )
                return

            left_positions = np.asarray(left_positions, dtype=np.float64).reshape(-1, 6)
            right_positions = np.asarray(right_positions, dtype=np.float64).reshape(
                -1, 6
            )
            if left_positions.shape[0] == 0:
                print("[ScriptedPolicy] Empty trajectory, ignoring execute_trajectory")
                return

            if (
                current_left_joint_pos is not None
                and current_right_joint_pos is not None
            ):
                current_left = np.asarray(
                    current_left_joint_pos, dtype=np.float64
                ).reshape(1, 6)
                current_right = np.asarray(
                    current_right_joint_pos, dtype=np.float64
                ).reshape(1, 6)
            else:
                current_joints = self._command_joints or self._target_joints
                current_left = np.asarray(
                    current_joints["left_joint_pos"], dtype=np.float64
                ).reshape(1, 6)
                current_right = np.asarray(
                    current_joints["right_joint_pos"], dtype=np.float64
                ).reshape(1, 6)

            self._command_joints = {
                "left_joint_pos": current_left[0].copy(),
                "right_joint_pos": current_right[0].copy(),
            }
            self._target_joints = {
                "left_joint_pos": current_left[0].copy(),
                "right_joint_pos": current_right[0].copy(),
            }

            needs_bridge = not (
                np.allclose(left_positions[0], current_left[0], atol=1e-6)
                and np.allclose(right_positions[0], current_right[0], atol=1e-6)
            )
            if needs_bridge:
                left_positions = np.concatenate([current_left, left_positions], axis=0)
                right_positions = np.concatenate(
                    [current_right, right_positions], axis=0
                )

            left_positions, right_positions = self._retime_joint_trajectory(
                left_positions,
                right_positions,
                max_joint_vel=max_joint_vel,
            )

            waypoints = []
            for i in range(left_positions.shape[0]):
                waypoints.append(
                    {
                        "left_joint_pos": np.asarray(
                            left_positions[i], dtype=np.float64
                        ),
                        "right_joint_pos": np.asarray(
                            right_positions[i], dtype=np.float64
                        ),
                    }
                )
            self._trajectory_queue = waypoints
            self._trajectory_idx = 0
            self._steps_left_left = 0
            self._steps_left_right = 0
            if left_gripper is not None:
                self._gripper["left"] = float(np.clip(left_gripper, 0.0, 1.0))
            if right_gripper is not None:
                self._gripper["right"] = float(np.clip(right_gripper, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def ensure_initialized(self, observation: dict[str, np.ndarray]) -> None:
        """Initialize target joints from observation if not yet set."""
        with self._lock:
            if self._target_joints is not None:
                return
            self._target_joints = {
                "left_joint_pos": np.asarray(
                    observation["left_joint_pos"], dtype=np.float64
                ).copy(),
                "right_joint_pos": np.asarray(
                    observation["right_joint_pos"], dtype=np.float64
                ).copy(),
            }
            self._command_joints = {
                "left_joint_pos": self._target_joints["left_joint_pos"].copy(),
                "right_joint_pos": self._target_joints["right_joint_pos"].copy(),
            }

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_action(
        self, observation: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Return an absolute ``joint_position`` action."""
        info: dict[str, Any] = {}

        with self._lock:
            if self._target_joints is None:
                self._target_joints = {
                    "left_joint_pos": np.asarray(
                        observation["left_joint_pos"], dtype=np.float64
                    ).copy(),
                    "right_joint_pos": np.asarray(
                        observation["right_joint_pos"], dtype=np.float64
                    ).copy(),
                }
                self._command_joints = {
                    "left_joint_pos": self._target_joints["left_joint_pos"].copy(),
                    "right_joint_pos": self._target_joints["right_joint_pos"].copy(),
                }

            # Consume trajectory waypoints (if any)
            if self._trajectory_queue is not None:
                wp = self._trajectory_queue[self._trajectory_idx]
                self._command_joints["left_joint_pos"] = wp["left_joint_pos"].copy()
                self._command_joints["right_joint_pos"] = wp["right_joint_pos"].copy()
                self._trajectory_idx += 1
                if self._trajectory_idx >= len(self._trajectory_queue):
                    self._target_joints["left_joint_pos"] = wp["left_joint_pos"].copy()
                    self._target_joints["right_joint_pos"] = wp[
                        "right_joint_pos"
                    ].copy()
                    self._trajectory_queue = None
                    self._trajectory_idx = 0
            else:
                # Interpolate toward target in joint space (per-arm)
                for side, key, steps_left in (
                    ("left", "left_joint_pos", self._steps_left_left),
                    ("right", "right_joint_pos", self._steps_left_right),
                ):
                    if steps_left > 0:
                        frac = 1.0 / steps_left
                        diff = self._target_joints[key] - self._command_joints[key]
                        self._command_joints[key] = (
                            self._command_joints[key] + diff * frac
                        )
                        if side == "left":
                            self._steps_left_left -= 1
                            if self._steps_left_left == 0:
                                self._command_joints[key] = self._target_joints[
                                    key
                                ].copy()
                        else:
                            self._steps_left_right -= 1
                            if self._steps_left_right == 0:
                                self._command_joints[key] = self._target_joints[
                                    key
                                ].copy()

            gripper_snap = dict(self._gripper)

        action: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            action[f"{side}_joint_pos"] = self._command_joints[
                f"{side}_joint_pos"
            ].astype(np.float32)
            action[f"{side}_gripper_pos"] = np.array(
                [gripper_snap[side]], dtype=np.float32
            )

        # ---- Safety: clamp per-step joint delta against actual observation ----
        # max_step = max_joint_velocity_rad_s (rad/s) / control_hz (Hz)
        max_step = self._safety.max_joint_velocity_rad_s / max(self._control_hz, 1.0)
        for side in ("left", "right"):
            obs_key = f"{side}_joint_pos"
            if obs_key in observation:
                obs_jp = np.asarray(observation[obs_key], dtype=np.float64).reshape(-1)
                cmd_jp = action[f"{side}_joint_pos"]
                delta = cmd_jp - obs_jp
                if np.any(np.abs(delta) > max_step):
                    exceeded = np.where(np.abs(delta) > max_step)[0]
                    logger.warning(
                        "[ScriptedPolicy] Joint delta clamp on %s: "
                        "max_delta=%.4f rad (limit=%.4f rad = %.1f rad/s / %.0f Hz). "
                        "Exceeded joints: %s (deltas: %s)",
                        side,
                        float(np.max(np.abs(delta))),
                        max_step,
                        self._safety.max_joint_velocity_rad_s,
                        self._control_hz,
                        exceeded.tolist(),
                        np.round(delta[exceeded], 4).tolist(),
                    )
                    delta_clamped = np.clip(delta, -max_step, max_step)
                    cmd_jp_clamped = obs_jp + delta_clamped
                    action[f"{side}_joint_pos"] = cmd_jp_clamped.astype(np.float32)
                    # Update _command_joints so the interpolation tracks correctly
                    self._command_joints[f"{side}_joint_pos"] = cmd_jp_clamped

        self._record_joint_positions(action)
        return action, info

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record_joint_positions(self, action: dict[str, np.ndarray]) -> None:
        action_14d = np.concatenate(
            [
                np.asarray(action["left_joint_pos"], dtype=np.float32).reshape(6),
                np.asarray(action["left_gripper_pos"], dtype=np.float32).reshape(1),
                np.asarray(action["right_joint_pos"], dtype=np.float32).reshape(6),
                np.asarray(action["right_gripper_pos"], dtype=np.float32).reshape(1),
            ]
        ).astype(np.float32)

        with self._lock:
            self._joint_record.append(
                {"step_idx": self._record_step_idx, "action": action_14d.copy()}
            )
            self._record_step_idx += 1

    def record_scripted_trajectory_to_parquet(
        self,
        output_path: str | Path = DEFAULT_JOINT_RECORD_PATH,
        clear_buffer: bool = False,
        append_if_exists: bool = True,
    ) -> Path:
        """Write buffered joint-position trajectory to a parquet file."""
        with self._lock:
            if not self._joint_record:
                print("[ScriptedPolicy] No joint trajectory samples to save.")
                return Path(output_path)
            rows = list(self._joint_record)

        import pandas as pd

        df = pd.DataFrame(
            {
                "step_idx": [r["step_idx"] for r in rows],
                "action": [r["action"] for r in rows],
            }
        )
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if append_if_exists and save_path.exists():
            try:
                existing_df = pd.read_parquet(save_path)
                if "step_idx" in existing_df.columns and not existing_df.empty:
                    df["step_idx"] = (
                        df["step_idx"] + int(existing_df["step_idx"].max()) + 1
                    )
                if "action" in existing_df.columns:
                    df = pd.concat(
                        [existing_df[["step_idx", "action"]], df], ignore_index=True
                    )
            except Exception as exc:
                print(
                    f"[ScriptedPolicy] Failed to append existing parquet, overwriting: {exc}"
                )
        df.to_parquet(save_path, index=False)
        print(
            f"[ScriptedPolicy] Saved {len(rows)} joint trajectory samples to {save_path}"
        )

        if clear_buffer:
            with self._lock:
                self._joint_record.clear()
                self._record_step_idx = 0
        return save_path

    def reset(self) -> dict[str, Any]:
        """Reset targets and command buffer."""
        with self._lock:
            for side in ("left", "right"):
                self._gripper[side] = self.HOME_GRIPPER
            self._target_joints = None
            self._command_joints = None
            self._steps_left_left = 0
            self._steps_left_right = 0
            self._trajectory_queue = None
            self._trajectory_idx = 0
        return {"task_name": "scripted"}
