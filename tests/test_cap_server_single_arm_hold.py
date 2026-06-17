from __future__ import annotations

import threading

import numpy as np

from cap.server.cap_server import CapServer


class _Safety:
    def is_estopped(self) -> bool:
        return False


class _StubServer:
    def __init__(self) -> None:
        self._motion_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._safety = _Safety()
        self._cmd_left_jp = np.zeros(6, dtype=np.float64)
        self._cmd_left_gp = np.zeros(1, dtype=np.float64)
        self._cmd_right_jp = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        self._cmd_right_gp = np.array([0.9], dtype=np.float64)
        self.left_joint_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        self.left_gripper_pos = np.array([0.1], dtype=np.float64)
        self.right_joint_pos = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64)
        self.right_gripper_pos = np.array([0.2], dtype=np.float64)
        self._dirty_calls: list[str] = []

    @staticmethod
    def _sample_joint_keypoints(timestamps, joint_positions, t_now):  # noqa: ANN001
        return CapServer._sample_joint_keypoints(timestamps, joint_positions, t_now)

    def _mark_policy_output_dirty(self, source: str) -> None:
        self._dirty_calls.append(source)


def test_single_arm_joint_keypoints_hold_opposite_arm_at_measured_state() -> None:
    server = _StubServer()

    result = CapServer.move_joint_keypoints(
        server,
        "left",
        [0.0],
        [np.array([9, 9, 9, 9, 9, 9], dtype=np.float64)],
    )

    assert result == {"success": True, "reason": "ok"}
    np.testing.assert_allclose(server._cmd_left_jp, np.full(6, 9.0))
    np.testing.assert_allclose(server._cmd_right_jp, server.right_joint_pos)
    np.testing.assert_allclose(server._cmd_right_gp, server.right_gripper_pos)
    assert server._dirty_calls == ["move_joint_keypoints"]
