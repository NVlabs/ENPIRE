from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from enpire.env.forge.experimental.lerobot_replay_policy import LerobotReplayPolicy


class TestRawFolderInitialState(unittest.TestCase):
    def test_raw_folder_populates_initial_state_from_state_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            np.save(root / "action-left-pos.npy", np.zeros((2, 7), dtype=np.float32))
            np.save(root / "action-right-pos.npy", np.zeros((2, 7), dtype=np.float32))

            left_joint = np.array(
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 1, 1, 1, 1, 1]],
                dtype=np.float32,
            )
            left_gripper = np.array([[0.7], [0.8]], dtype=np.float32)
            right_joint = np.array(
                [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], [2, 2, 2, 2, 2, 2]],
                dtype=np.float32,
            )
            right_gripper = np.array([[0.9], [1.0]], dtype=np.float32)

            np.save(root / "left-joint_pos.npy", left_joint)
            np.save(root / "left-gripper_pos.npy", left_gripper)
            np.save(root / "right-joint_pos.npy", right_joint)
            np.save(root / "right-gripper_pos.npy", right_gripper)

            policy = LerobotReplayPolicy(dataset_path=root, control_mode="joint_position")

            self.assertIsNotNone(policy.initial_state)
            assert policy.initial_state is not None
            np.testing.assert_allclose(policy.initial_state["left_joint_pos"], left_joint[0])
            np.testing.assert_allclose(policy.initial_state["left_gripper_pos"], left_gripper[0])
            np.testing.assert_allclose(policy.initial_state["right_joint_pos"], right_joint[0])
            np.testing.assert_allclose(policy.initial_state["right_gripper_pos"], right_gripper[0])


if __name__ == "__main__":
    unittest.main()
