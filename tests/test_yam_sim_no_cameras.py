from __future__ import annotations

import numpy as np

from enpire.env.forge.robot.yam.yam_sim_env import YamSimEnv


def test_yam_sim_env_disable_cameras_returns_blank_frames() -> None:
    env = YamSimEnv(control_mode="joint_position", enable_cameras=False)
    try:
        obs, _ = env.reset(
            options={
                "target_joint_position": {
                    "left_joint_pos": np.zeros(6, dtype=np.float32),
                    "left_gripper_pos": np.ones(1, dtype=np.float32),
                    "right_joint_pos": np.zeros(6, dtype=np.float32),
                    "right_gripper_pos": np.ones(1, dtype=np.float32),
                }
            }
        )
        assert env.camera_ids == {}
        for name in ("top", "left", "right"):
            image = obs[f"{name}_camera_image"]
            assert image.shape == (env.CAMERA_HEIGHT, env.CAMERA_WIDTH, 3)
            assert np.count_nonzero(image) == 0
    finally:
        env.close()


def test_delta_ee_positive_z_resets_right_z_reference_to_observed_pose() -> None:
    env = YamSimEnv(control_mode="delta_ee_pose", enable_cameras=False)
    try:
        obs, _ = env.reset(
            options={
                "target_joint_position": {
                    "left_joint_pos": np.zeros(6, dtype=np.float32),
                    "left_gripper_pos": np.ones(1, dtype=np.float32),
                    "right_joint_pos": np.zeros(6, dtype=np.float32),
                    "right_gripper_pos": np.ones(1, dtype=np.float32),
                }
            }
        )
        assert env.last_commanded_ee_pose is not None
        observed_right_z = float(obs["right_ee_pos"][2])
        env.last_commanded_ee_pose["right"]["pos"][2] = observed_right_z - 0.2
        env._last_delta_ee_pos["right"] = np.array([0.0, 0.0, -0.01], dtype=np.float32)

        def fail_forward_kinematics(*_, **__):
            raise AssertionError(
                "delta conversion should use cached EE observation"
            )

        env._kinematics.forward_kinematics = fail_forward_kinematics

        identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        action = {
            "left_ee_pos": np.zeros(3, dtype=np.float32),
            "left_ee_rot6d": identity_rot6d.copy(),
            "left_gripper_pos": np.ones(1, dtype=np.float32),
            "right_ee_pos": np.array([0.0, 0.0, 0.03], dtype=np.float32),
            "right_ee_rot6d": identity_rot6d.copy(),
            "right_gripper_pos": np.ones(1, dtype=np.float32),
        }

        converted = env._convert_from_delta_ee_to_cartesian(action)

        np.testing.assert_allclose(
            converted["right_ee_pos"][2],
            observed_right_z + 0.03,
            atol=1e-6,
        )

        env.last_commanded_ee_pose["right"]["pos"][2] = observed_right_z - 0.2
        converted = env._convert_from_delta_ee_to_cartesian(action)
        np.testing.assert_allclose(
            converted["right_ee_pos"][2],
            observed_right_z - 0.2 + 0.03,
            atol=1e-6,
        )
    finally:
        env.close()
