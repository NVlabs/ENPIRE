from typing import Any

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _rpy_display_to_quat_xyzw(rpy_display: Any) -> np.ndarray:
    """Convert displayed/global XYZ RPY degrees to quaternion in xyzw order."""
    rpy = np.asarray(rpy_display, dtype=np.float64).reshape(-1)
    if rpy.size != 3:
        raise ValueError(f"Expected RPY with 3 values, got shape {np.asarray(rpy_display).shape}")
    return Rotation.from_euler("xyz", rpy, degrees=True).as_quat().astype(np.float32)


def _quat_xyzw_to_rpy_display(q_xyzw: Any) -> np.ndarray:
    """Convert quaternion in xyzw order to displayed/global XYZ RPY degrees."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {np.asarray(q_xyzw).shape}")
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        raise ValueError("Expected quaternion with non-zero norm")
    return Rotation.from_quat(q / n).as_euler("xyz", degrees=True).astype(np.float32)


def _quat_xyzw_to_rot6d(q_xyzw: Any) -> np.ndarray:
    """
    Convert quaternion to rotation matrix's first two columns
    """
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)[:4]
    if q.size != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {np.asarray(q_xyzw).shape}")
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        q = q / n
    mat = Rotation.from_quat(q).as_matrix()
    return np.concatenate([mat[:, 0], mat[:, 1]], axis=0).astype(np.float32)


def _rot6d_to_rot_matrix(rot6d: Any) -> np.ndarray:
    if rot6d is None:
        return None
    v = np.asarray(rot6d, dtype=np.float64).reshape(-1)[:6]
    if v.size != 6:
        raise ValueError(f"Expected rot6d with 6 values, got shape {np.asarray(rot6d).shape}")

    a1, a2 = v[:3], v[3:6]
    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-8:
        a1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n1 = 1.0
    b1 = a1 / n1

    a2 = a2 - np.dot(b1, a2) * b1
    n2 = float(np.linalg.norm(a2))
    if n2 < 1e-8:
        a2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(b1, a2))) > 0.9:
            a2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        a2 = a2 - np.dot(b1, a2) * b1
        n2 = float(np.linalg.norm(a2))
    b2 = a2 / n2
    b3 = np.cross(b1, b2)

    mat = np.stack([b1, b2, b3], axis=1)
    return mat


def _rot6d_to_quat_xyzw(rot6d: Any) -> np.ndarray:
    """
    Convert rotation matrix's first two columns to a quaternion in xyzw order.
    """
    mat = _rot6d_to_rot_matrix(rot6d)
    return Rotation.from_matrix(mat).as_quat().astype(np.float32)


class YamKinematics:
    def __init__(
        self,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 1.0,
    ):
        from enpire.env.forge.robot.models.station.paths import get_station_xml

        model_path = get_station_xml()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        self._model = model
        self._lm_damping = lm_damping
        self.configuration = mink.Configuration(model)
        self.tasks = [
            mink.FrameTask(
                frame_name=f"{side}_grasp_site",
                frame_type="site",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=lm_damping,
            )
            for side in ["left", "right"]
        ]
        self.left_end_effector_task, self.right_end_effector_task = self.tasks

        # Orientation-only tasks and velocity limit for translation-only mode:
        # joints 1-3 (arm base) and gripper fingers are frozen while joints 4-6
        # (wrist) solve for a target EE orientation.
        self._orientation_only_tasks: list[mink.FrameTask] | None = None
        self._frozen_base_limit: mink.VelocityLimit | None = None
        self._joint_limit: mink.ConfigurationLimit | None = None

    def forward_kinematics(
        self, left_joint_pos: np.ndarray, right_joint_pos: np.ndarray
    ) -> np.ndarray:
        # Update qpos
        self.configuration.data.qpos[:6] = left_joint_pos
        self.configuration.data.qpos[8:14] = right_joint_pos
        self.configuration.update()

        # Compute forward kinematics
        left_ee_pose = self.configuration.get_transform_frame_to_world("left_grasp_site", "site")
        right_ee_pose = self.configuration.get_transform_frame_to_world("right_grasp_site", "site")

        # Extract end effector pos and quat
        left_ee_pos = left_ee_pose.translation()
        left_ee_quat_xyzw = left_ee_pose.rotation().wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw
        right_ee_pos = right_ee_pose.translation()
        right_ee_quat_xyzw = right_ee_pose.rotation().wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw

        return left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw

    def inverse_kinematics(
        self,
        left_ee_pos: np.ndarray,
        left_ee_quat_xyzw: np.ndarray,
        right_ee_pos: np.ndarray,
        right_ee_quat_xyzw: np.ndarray,
        seeded=False,
        dt=0.01,
        solver="daqp",
        damping=1e-3,
        err_threshold=1e-4,
        max_iters=20,
    ) -> np.ndarray:
        left_ee_quat = left_ee_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
        right_ee_quat = right_ee_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz

        # Initialize qpos to zero if not seeding from previous joint configuration
        if not seeded:
            self.configuration.update(np.zeros_like(self.configuration.data.qpos))

        # Update targets
        T_wt_left = mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=left_ee_quat), left_ee_pos)
        T_wt_right = mink.SE3.from_rotation_and_translation(
            mink.SO3(wxyz=right_ee_quat), right_ee_pos
        )
        self.left_end_effector_task.set_target(T_wt_left)
        self.right_end_effector_task.set_target(T_wt_right)

        # Solve IK
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks, dt, solver, damping)
            self.configuration.integrate_inplace(vel, dt)
            err_left = self.left_end_effector_task.compute_error(self.configuration)
            err_right = self.right_end_effector_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err_left) <= err_threshold
                and np.linalg.norm(err_right) <= err_threshold
            ):
                break

        # Extract joint positions
        qpos = self.configuration.q
        left_joint_pos = qpos[:6]
        right_joint_pos = qpos[8:14]

        return left_joint_pos, right_joint_pos

    def inverse_kinematics_full(
        self,
        left_ee_pos: np.ndarray | None,
        left_ee_quat_xyzw: np.ndarray | None,
        right_ee_pos: np.ndarray | None,
        right_ee_quat_xyzw: np.ndarray | None,
        left_seed: np.ndarray,
        right_seed: np.ndarray,
        dt: float = 0.01,
        solver: str = "daqp",
        damping: float = 1e-3,
        err_threshold: float = 1e-4,
        max_iters: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Full 6-DOF IK (position + orientation) with per-side skipping and
        explicit seeds. Pass both ``*_ee_pos`` and ``*_ee_quat_xyzw`` to
        include that side; pass ``None`` for both to leave that side at its
        seed. Errors are quat double-cover distances (same metric as
        ``inverse_kinematics_orientation_only``), so callers can reuse the
        existing orientation-tolerance constants.
        """
        # Seed configuration from the provided joint values for both arms.
        qpos = self.configuration.data.qpos.copy()
        qpos[0:6] = left_seed
        qpos[8:14] = right_seed
        self.configuration.update(qpos)

        tasks: list = []
        left_active = left_ee_pos is not None and left_ee_quat_xyzw is not None
        right_active = right_ee_pos is not None and right_ee_quat_xyzw is not None
        if left_active:
            left_wxyz = left_ee_quat_xyzw[[3, 0, 1, 2]]
            self.left_end_effector_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=left_wxyz), left_ee_pos)
            )
            tasks.append(self.left_end_effector_task)
        if right_active:
            right_wxyz = right_ee_quat_xyzw[[3, 0, 1, 2]]
            self.right_end_effector_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=right_wxyz), right_ee_pos)
            )
            tasks.append(self.right_end_effector_task)

        if tasks:
            for itr in range(max_iters):
                vel = mink.solve_ik(self.configuration, tasks, dt, solver, damping)
                self.configuration.integrate_inplace(vel, dt)
                # for t in tasks:
                #     solve_err = np.linalg.norm(t.compute_error(self.configuration))
                #     print(f"DEBUG-IK:: itr={itr}/max_iters={max_iters} | {solve_err} | err_thoreshold {err_threshold}")
                if all(
                    np.linalg.norm(t.compute_error(self.configuration)) <= err_threshold
                    for t in tasks
                ):
                    break

        qpos = self.configuration.q
        left_out = qpos[:6].copy()
        right_out = qpos[8:14].copy()

        # Orientation error via quat double-cover distance (consistent with
        # inverse_kinematics_orientation_only so the caller's tolerance
        # constants carry over). Sides that were skipped report 0.0.
        err_left_norm = 0.0
        err_right_norm = 0.0
        if left_active:
            left_final_q = (
                self.configuration.get_transform_frame_to_world("left_grasp_site", "site")
                .rotation()
                .wxyz[[1, 2, 3, 0]]
            )
            err_left_norm = min(
                float(np.linalg.norm(left_final_q - left_ee_quat_xyzw)),
                float(np.linalg.norm(left_final_q + left_ee_quat_xyzw)),
            )
        if right_active:
            right_final_q = (
                self.configuration.get_transform_frame_to_world("right_grasp_site", "site")
                .rotation()
                .wxyz[[1, 2, 3, 0]]
            )
            err_right_norm = min(
                float(np.linalg.norm(right_final_q - right_ee_quat_xyzw)),
                float(np.linalg.norm(right_final_q + right_ee_quat_xyzw)),
            )
        return left_out, right_out, err_left_norm, err_right_norm

    def _ensure_orientation_only_setup(self) -> None:
        if self._orientation_only_tasks is not None:
            return
        self._orientation_only_tasks = [
            mink.FrameTask(
                frame_name=f"{side}_grasp_site",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=1.0,
                lm_damping=self._lm_damping,
            )
            for side in ["left", "right"]
        ]
        self._frozen_base_limit = mink.VelocityLimit(
            self._model,
            {
                "left_joint1": 0.0,
                "left_joint2": 0.0,
                "left_joint3": 0.0,
                "left_left_finger": 0.0,
                "left_right_finger": 0.0,
                "right_joint1": 0.0,
                "right_joint2": 0.0,
                "right_joint3": 0.0,
                "right_left_finger": 0.0,
                "right_right_finger": 0.0,
            },
        )
        # Joint limits so the wrist solver can't propose angles the real robot
        # would clip. Without this, passing an explicit `limits=[...]` to
        # mink.solve_ik suppresses mink's default ConfigurationLimit.
        self._joint_limit = mink.ConfigurationLimit(self._model)

    def inverse_kinematics_orientation_only(
        self,
        left_joint_pos: np.ndarray,
        left_ee_quat_xyzw: np.ndarray | None,
        right_joint_pos: np.ndarray,
        right_ee_quat_xyzw: np.ndarray | None,
        dt: float = 0.01,
        solver: str = "daqp",
        damping: float = 1e-3,
        err_threshold: float = 1e-4,
        max_iters: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Solve IK for wrist joints (4, 5, 6) only, with base joints (1, 2, 3)
        and grippers pinned to the provided leader joint values. Pass
        ``left_ee_quat_xyzw`` or ``right_ee_quat_xyzw`` as ``None`` to skip
        that arm entirely — its joints are returned unchanged from the input
        and its error is reported as 0.0. Returns the full 6-DOF joint
        solution per arm plus the final orientation errors.
        """
        self._ensure_orientation_only_setup()
        assert self._orientation_only_tasks is not None
        assert self._frozen_base_limit is not None

        # Seed configuration with leader joints; base joints are pinned by the
        # zero-velocity limit, wrist joints act as the IK initial guess.
        qpos = self.configuration.data.qpos.copy()
        qpos[0:6] = left_joint_pos
        qpos[8:14] = right_joint_pos
        self.configuration.update(qpos)

        left_task, right_task = self._orientation_only_tasks
        # Build the active task list based on which sides requested solves.
        tasks: list = []
        if left_ee_quat_xyzw is not None:
            left_pose = self.configuration.get_transform_frame_to_world("left_grasp_site", "site")
            left_wxyz = left_ee_quat_xyzw[[3, 0, 1, 2]]
            left_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(wxyz=left_wxyz), left_pose.translation()
                )
            )
            tasks.append(left_task)
        if right_ee_quat_xyzw is not None:
            right_pose = self.configuration.get_transform_frame_to_world("right_grasp_site", "site")
            right_wxyz = right_ee_quat_xyzw[[3, 0, 1, 2]]
            right_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(wxyz=right_wxyz), right_pose.translation()
                )
            )
            tasks.append(right_task)

        if tasks:
            limits = [self._frozen_base_limit, self._joint_limit]
            for _ in range(max_iters):
                vel = mink.solve_ik(self.configuration, tasks, dt, solver, damping, limits=limits)
                self.configuration.integrate_inplace(vel, dt)
                if all(
                    np.linalg.norm(t.compute_error(self.configuration)) <= err_threshold
                    for t in tasks
                ):
                    break

        qpos = self.configuration.q
        left_out = qpos[:6].copy()
        right_out = qpos[8:14].copy()
        # Orientation-only error: min(||q-target||, ||q+target||) to handle the
        # quaternion double-cover (q and -q represent the same rotation).
        err_left_norm = 0.0
        err_right_norm = 0.0
        if left_ee_quat_xyzw is not None:
            left_final_q = (
                self.configuration.get_transform_frame_to_world("left_grasp_site", "site")
                .rotation()
                .wxyz[[1, 2, 3, 0]]
            )
            err_left_norm = min(
                float(np.linalg.norm(left_final_q - left_ee_quat_xyzw)),
                float(np.linalg.norm(left_final_q + left_ee_quat_xyzw)),
            )
        if right_ee_quat_xyzw is not None:
            right_final_q = (
                self.configuration.get_transform_frame_to_world("right_grasp_site", "site")
                .rotation()
                .wxyz[[1, 2, 3, 0]]
            )
            err_right_norm = min(
                float(np.linalg.norm(right_final_q - right_ee_quat_xyzw)),
                float(np.linalg.norm(right_final_q + right_ee_quat_xyzw)),
            )
        return left_out, right_out, err_left_norm, err_right_norm


def main():
    import time

    kinematics = YamKinematics()
    left_joint_pos = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25])
    right_joint_pos = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25])

    # Forward kinematics
    left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
        kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
    )

    # Inverse kinematics
    left_joint_pos_ik, right_joint_pos_ik = kinematics.inverse_kinematics(
        left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw
    )

    # Check consistency
    np.testing.assert_allclose(left_joint_pos, left_joint_pos_ik, atol=1e-3)
    np.testing.assert_allclose(right_joint_pos, right_joint_pos_ik, atol=1e-3)

    # Benchmark FK
    num_iters = 1000
    start_time = time.time()
    for _ in range(num_iters):
        # 0.015 ms (gear-ax8-max-05)
        kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
    end_time = time.time()
    print(f"FK time: {1000 * (end_time - start_time) / num_iters:.3f} ms")

    # Benchmark IK
    start_time = time.time()
    for _ in range(num_iters):
        # 4.7 ms starting from zero joint configuration (gear-ax8-max-05)
        # kinematics.inverse_kinematics(left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw)

        # 0.37 ms starting from previous joint configuration (gear-ax8-max-05)
        kinematics.inverse_kinematics(
            left_ee_pos,
            left_ee_quat_xyzw,
            right_ee_pos,
            right_ee_quat_xyzw,
            seeded=True,
        )
    end_time = time.time()
    print(f"IK time: {1000 * (end_time - start_time) / num_iters:.3f} ms")


if __name__ == "__main__":
    main()
