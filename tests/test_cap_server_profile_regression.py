# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-refactor regression tests for CAP server and sim backend.

These tests capture the golden baseline behavior of the sim backend and
CapServer state management BEFORE the modular robot profile refactor.
Run them after every refactor phase to ensure no regressions.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SimBackend-level tests (no CapServer, no Portal, no pinocchio)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sim_backend():
    from enpire.env.forge.cap.server.sim_backend import SimBackend

    backend = SimBackend(viewer=False)
    yield backend
    backend.close()


class TestSimBackendRoundtrip:
    """Verify SimBackend command→step→observe produces consistent results."""

    def test_arm_observation_shapes(self, sim_backend):
        """get_arm_observation returns correct keys and shapes for both arms."""
        for side in ("left", "right"):
            obs = sim_backend.get_arm_observation(side)
            assert "joint_pos" in obs
            assert "gripper_pos" in obs
            assert obs["joint_pos"].shape == (6,), f"{side} joint_pos shape mismatch"
            assert obs["gripper_pos"].shape == (1,), f"{side} gripper_pos shape mismatch"

    def test_command_step_observe_roundtrip(self, sim_backend):
        """command_arm → step → get_arm_observation reflects commanded position."""
        target_jp = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        target_gp = 0.5
        cmd = {"pos": np.concatenate([target_jp, [target_gp]])}
        sim_backend.command_arm("left", cmd)

        # Step multiple times so physics converges toward target
        for _ in range(200):
            sim_backend.step()

        obs = sim_backend.get_arm_observation("left")
        np.testing.assert_allclose(obs["joint_pos"], target_jp, atol=0.05)

    def test_initial_gripper_open(self, sim_backend):
        """Grippers start in the open position."""
        for side in ("left", "right"):
            obs = sim_backend.get_arm_observation(side)
            # Gripper should be near 1.0 (open) after init
            assert obs["gripper_pos"][0] > 0.5, f"{side} gripper not open at init"

    def test_render_rgb_shape(self, sim_backend):
        """render_rgb returns correct shape and dtype."""
        for camera in ("top", "left", "right"):
            img = sim_backend.render_rgb(camera)
            assert img.shape == (480, 640, 3), f"{camera} RGB shape mismatch"
            assert img.dtype == np.uint8

    def test_render_depth_shape(self, sim_backend):
        """render_depth returns correct shape and dtype."""
        for camera in ("top", "left", "right"):
            depth = sim_backend.render_depth(camera)
            assert depth.shape == (480, 640), f"{camera} depth shape mismatch"
            assert depth.dtype == np.float32

    def test_camera_intrinsics_format(self, sim_backend):
        """get_camera_intrinsics returns [fx, fy, cx, cy]."""
        for camera in ("top", "left", "right"):
            intr = sim_backend.get_camera_intrinsics(camera)
            assert isinstance(intr, list)
            assert len(intr) == 4
            fx, fy, cx, cy = intr
            # fx/fy may be near-zero depending on MuJoCo camera config
            assert isinstance(fx, float) and isinstance(fy, float)
            assert abs(cx - 320.0) < 1.0  # should be ~ width/2
            assert abs(cy - 240.0) < 1.0  # should be ~ height/2

    def test_camera_extrinsics_format(self, sim_backend):
        """get_camera_extrinsics returns dict with position and rotation."""
        for camera in ("top", "left", "right"):
            ext = sim_backend.get_camera_extrinsics(camera)
            assert "position" in ext
            assert "rotation" in ext
            assert len(ext["position"]) == 3
            assert len(ext["rotation"]) == 3  # 3x3 rotation matrix as list of rows
            assert len(ext["rotation"][0]) == 3

    def test_scene_management_roundtrip(self, sim_backend):
        """setup_scene → get_object_positions → clear_table cycle."""
        scenes = sim_backend.get_scenes()
        assert scenes["ok"] is True
        assert isinstance(scenes["scenes"], list)
        assert scenes["active"] is None

        # If scenes are available, test the full cycle
        if scenes["scenes"]:
            scene_name = scenes["scenes"][0]
            result = sim_backend.setup_scene(scene_name)
            assert result["ok"] is True
            assert len(result["objects"]) > 0

            positions = sim_backend.get_object_positions()
            assert positions["ok"] is True

            cleared = sim_backend.clear_table()
            assert cleared["ok"] is True


# ---------------------------------------------------------------------------
# SimArmClient / SimCameraClient wrapper tests
# ---------------------------------------------------------------------------


class TestSimClientWrappers:
    """Verify SimArmClient and SimCameraClient match _ArmClient / _CameraClient interface."""

    def test_sim_arm_client_interface(self, sim_backend):
        from enpire.env.forge.cap.server.sim_backend import SimArmClient

        client = SimArmClient(sim_backend, "left")

        # get_observations
        obs = client.get_observations()
        assert "joint_pos" in obs
        assert "gripper_pos" in obs
        assert obs["joint_pos"].shape == (6,)

        # get_joint_pos
        jp = client.get_joint_pos()
        assert jp.shape == (6,)
        np.testing.assert_array_equal(jp, obs["joint_pos"])

        # command_joint_state (should not raise)
        client.command_joint_state({"pos": np.zeros(7)})

    def test_sim_camera_client_interface(self, sim_backend):
        import time

        from enpire.env.forge.cap.server.sim_backend import SimCameraClient

        client = SimCameraClient(sim_backend, "top")
        time.sleep(0.2)  # let background render thread produce a frame

        try:
            rgb = client.get_rgb()
            assert rgb.shape == (480, 640, 3)
            assert rgb.dtype == np.uint8

            depth = client.get_depth()
            assert depth.shape == (480, 640)
            assert depth.dtype == np.float32

            intr = client.get_intrinsics()
            assert len(intr) == 4
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Config constants test (baseline before profile migration)
# ---------------------------------------------------------------------------


class TestConfigConstants:
    """Verify cap/config.py constants have expected shapes and values."""

    def test_joint_limits_shape(self):
        from enpire.env.forge.cap.config import JOINT_LIMITS_HIGH, JOINT_LIMITS_LOW

        assert JOINT_LIMITS_LOW.shape == (12,)  # 6 left + 6 right
        assert JOINT_LIMITS_HIGH.shape == (12,)
        # Left and right halves have identical limits
        np.testing.assert_array_equal(JOINT_LIMITS_LOW[:6], JOINT_LIMITS_LOW[6:])
        np.testing.assert_array_equal(JOINT_LIMITS_HIGH[:6], JOINT_LIMITS_HIGH[6:])

    def test_motor_gains_shape(self):
        from enpire.env.forge.cap.config import INTERP_KD, INTERP_KP

        assert INTERP_KP.shape == (7,)  # 6 arm + 1 gripper
        assert INTERP_KD.shape == (7,)

    def test_home_state_structure(self):
        from enpire.env.forge.cap.config import HOME_JOINT_STATE

        assert set(HOME_JOINT_STATE.keys()) == {
            "left_joint_pos",
            "left_gripper_pos",
            "right_joint_pos",
            "right_gripper_pos",
        }
        assert HOME_JOINT_STATE["left_joint_pos"].shape == (6,)
        assert HOME_JOINT_STATE["left_gripper_pos"].shape == (1,)
        assert HOME_JOINT_STATE["right_joint_pos"].shape == (6,)
        assert HOME_JOINT_STATE["right_gripper_pos"].shape == (1,)
        # Home is all zeros
        for v in HOME_JOINT_STATE.values():
            np.testing.assert_array_equal(v, np.zeros_like(v))

    def test_camera_names(self):
        from enpire.env.forge.cap.config import CAMERA_NAMES

        assert isinstance(CAMERA_NAMES, tuple)
        # Should contain top, left, right (may vary by station profile)
        assert len(CAMERA_NAMES) >= 1

    def test_control_frequencies(self):
        from enpire.env.forge.cap.config import (
            CONTROL_FREQ_HZ,
            CONTROL_PERIOD_S,
            POLICY_FREQ_HZ,
            POLICY_PERIOD_S,
        )

        assert CONTROL_FREQ_HZ == 60.0
        assert abs(CONTROL_PERIOD_S - 1.0 / 60.0) < 1e-10
        assert POLICY_FREQ_HZ == 30.0
        assert abs(POLICY_PERIOD_S - 1.0 / 30.0) < 1e-10

    def test_gripper_range(self):
        from enpire.env.forge.cap.config import GRIPPER_MAX, GRIPPER_MIN

        assert GRIPPER_MIN == 0.0
        assert GRIPPER_MAX == 1.0


# ---------------------------------------------------------------------------
# CapServer get_state key format test (uses _StubServer pattern)
# ---------------------------------------------------------------------------


class TestProfileEquivalence:
    """Verify robot profiles produce correct values."""

    def test_yam_profile_matches_config_constants(self):
        from enpire.env.forge.cap.config import (
            CAMERA_NAMES,
            CONTROL_FREQ_HZ,
            GRIPPER_MAX,
            GRIPPER_MIN,
            HOME_JOINT_STATE,
            INTERP_KD,
            INTERP_KP,
            JOINT_LIMITS_HIGH,
            JOINT_LIMITS_LOW,
            POLICY_FREQ_HZ,
        )
        from enpire.env.forge.cap.env.profile import yam_profile

        profile = yam_profile()

        assert profile.name == "yam"
        assert profile.is_bimanual is True
        assert profile.arm_names == ("left", "right")
        assert profile.control_freq_hz == CONTROL_FREQ_HZ
        assert profile.policy_freq_hz == POLICY_FREQ_HZ
        assert profile.camera_names == CAMERA_NAMES

        for side, start in (("left", 0), ("right", 6)):
            arm = profile.arms[side]
            assert arm.dof == 6
            np.testing.assert_array_equal(arm.joint_limits_low, JOINT_LIMITS_LOW[start : start + 6])
            np.testing.assert_array_equal(arm.joint_limits_high, JOINT_LIMITS_HIGH[start : start + 6])
            assert arm.gripper_min == GRIPPER_MIN
            assert arm.gripper_max == GRIPPER_MAX
            np.testing.assert_array_equal(arm.interp_kp, INTERP_KP)
            np.testing.assert_array_equal(arm.interp_kd, INTERP_KD)
            np.testing.assert_array_equal(arm.home_joint_pos, HOME_JOINT_STATE[f"{side}_joint_pos"])
            np.testing.assert_array_equal(arm.home_gripper_pos, HOME_JOINT_STATE[f"{side}_gripper_pos"])

        assert profile.arms["left"].q_slice == slice(0, 6)
        assert profile.arms["right"].q_slice == slice(8, 14)
        assert profile.arms["left"].ee_frame_name == "left_grasp"
        assert profile.arms["right"].ee_frame_name == "right_grasp"

    def test_robocasa_panda_omron_profile(self):
        from enpire.env.forge.cap.env.profile import robocasa_panda_omron_profile

        profile = robocasa_panda_omron_profile()
        assert profile.name == "panda_omron"
        assert profile.is_bimanual is False
        assert profile.arm_names == ("right",)
        assert profile.arms["right"].dof == 7
        assert profile.control_freq_hz == 20.0
        # urdf_path may be set if robosuite is available on disk
        assert profile.camera_obs_key_map is not None
        assert "top" in profile.camera_obs_key_map

    def test_protocol_conformance_sim_backend(self):
        from enpire.env.forge.cap.env.base import SceneProtocol, EnvProtocol
        from enpire.env.forge.cap.server.sim_backend import SimBackend

        backend = SimBackend(viewer=False)
        try:
            assert isinstance(backend, EnvProtocol)
            assert isinstance(backend, SceneProtocol)
        finally:
            backend.close()


class _FakeArm:
    """Minimal arm client returning fixed observations."""

    def __init__(self, jp: np.ndarray, gp: np.ndarray):
        self._jp = jp
        self._gp = gp

    def get_observations(self):
        return {"joint_pos": self._jp.copy(), "gripper_pos": self._gp.copy()}

    def command_joint_state(self, cmd):
        pass

    def get_joint_pos(self):
        return self._jp.copy()


class TestGetStateKeyFormat:
    """Verify get_state() returns the exact expected key set and shapes."""

    def test_get_state_keys_and_shapes(self):
        """get_state() must return these exact keys with these exact shapes."""
        from enpire.env.forge.cap.server.cap_server import CapServer

        # Directly check what get_state returns by setting up minimal state
        # Use the _StubServer pattern from existing tests
        import threading

        class _MinimalServer:
            def __init__(self):
                self._state_lock = threading.Lock()
                # Dict-based state (new)
                self._arm_jp = {
                    "left": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                    "right": np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                }
                self._arm_gp = {
                    "left": np.array([0.8]),
                    "right": np.array([0.3]),
                }
                self._ee_pos = {
                    "left": np.array([0.1, 0.2, 0.3]),
                    "right": np.array([0.4, 0.5, 0.6]),
                }
                self._ee_quat = {
                    "left": np.array([0.0, 0.0, 0.0, 1.0]),
                    "right": np.array([0.0, 0.0, 0.7071, 0.7071]),
                }
                self._sim_backend = None

        server = _MinimalServer()
        state = CapServer.get_state(server)

        # Exact key set
        expected_keys = {
            "left_joint_pos",
            "left_gripper_pos",
            "right_joint_pos",
            "right_gripper_pos",
            "left_ee_pos",
            "left_ee_quat_xyzw",
            "right_ee_pos",
            "right_ee_quat_xyzw",
        }
        assert set(state.keys()) == expected_keys

        # Shapes
        assert state["left_joint_pos"].shape == (6,)
        assert state["left_gripper_pos"].shape == (1,)
        assert state["right_joint_pos"].shape == (6,)
        assert state["right_gripper_pos"].shape == (1,)
        assert state["left_ee_pos"].shape == (3,)
        assert state["left_ee_quat_xyzw"].shape == (4,)
        assert state["right_ee_pos"].shape == (3,)
        assert state["right_ee_quat_xyzw"].shape == (4,)

        # Values are copies (not references)
        assert state["left_joint_pos"] is not server._arm_jp["left"]
        np.testing.assert_array_equal(state["left_joint_pos"], server._arm_jp["left"])
        np.testing.assert_array_equal(state["right_ee_pos"], server._ee_pos["right"])
