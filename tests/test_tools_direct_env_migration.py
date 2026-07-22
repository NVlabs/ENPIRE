# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np

from enpire.env.forge.cap.debug_ui.app import LogMonitor
from enpire.env.forge.cap.agent.tools.base import FreespaceResult, ToolResult
from enpire.env.forge.cap.agent.tools.camera import GetCameraExtrinsicsTool, GetCameraIntrinsicsTool
from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool
from enpire.env.forge.cap.agent.tools.native import GetCameraImageTool, GetRobotStateTool, SetGripperTool
from enpire.env.forge.cap.agent.tools.nudge import NudgeTool


class _ArmProfile:
    interp_kp = np.ones(7)
    interp_kd = np.ones(7) * 0.1
    home_joint_pos = np.zeros(6)
    home_gripper_pos = np.ones(1)


class _Profile:
    arms = {"left": _ArmProfile(), "right": _ArmProfile()}


class _DirectEnv:
    _profile = _Profile()

    def __init__(self) -> None:
        self.commands: list[tuple[str, dict]] = []
        self.bimanual_calls: list[tuple] = []

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        sign = 1.0 if side == "left" else -1.0
        return {
            "joint_pos": np.zeros(6, dtype=np.float64),
            "gripper_pos": np.asarray([0.4], dtype=np.float64),
            "ee_pos": np.asarray([0.4, 0.2 * sign, 0.8], dtype=np.float64),
            "ee_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        }

    def command_joint_state(self, side: str, state: dict) -> None:
        self.commands.append((side, state))

    def render_rgb(self, camera: str) -> np.ndarray:
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def render_depth(self, camera: str) -> np.ndarray:
        return np.ones((2, 3), dtype=np.float32)

    def get_camera_intrinsics(self, camera: str) -> list[float]:
        return [100.0, 101.0, 1.5, 2.5]

    def get_camera_extrinsics(self, camera: str) -> dict:
        return {
            "position": [1.0, 2.0, 3.0],
            "rotation": np.eye(3).tolist(),
            "needs_optical_flip": True,
        }

    def move_bimanual_joint_keypoints(
        self,
        timestamps,
        left_positions,
        right_positions,
        left_gripper_positions=None,
        right_gripper_positions=None,
    ) -> dict:
        self.bimanual_calls.append(
            (
                np.asarray(timestamps),
                np.asarray(left_positions),
                np.asarray(right_positions),
                left_gripper_positions,
                right_gripper_positions,
            )
        )
        return {"success": True, "reason": "ok"}


def test_real_yam_skills_shim_has_no_cap_server_or_portal_rpc() -> None:
    text = Path("cap/env/real_bimanual_yam/skills.py").read_text()
    banned = [
        "import portal",
        "portal.Client",
        "CAP_SERVER_PORT",
        "cap_server",
        "move_eef_pose_precise",
    ]
    assert [term for term in banned if term in text] == []


def test_camera_tools_direct_env_include_k_and_t_cam_world() -> None:
    env = _DirectEnv()

    intr = GetCameraIntrinsicsTool(env=env).execute(camera="top")
    assert intr.success is True
    assert intr.data["intrinsics"] == [100.0, 101.0, 1.5, 2.5]
    assert intr.data["K"] == [[100.0, 0.0, 1.5], [0.0, 101.0, 2.5], [0.0, 0.0, 1.0]]

    extr = GetCameraExtrinsicsTool(env=env).execute(camera="top")
    assert extr.success is True
    assert extr.data["needs_optical_flip"] is True
    assert np.allclose(
        np.asarray(extr.data["T_cam_world"])[:3, :3],
        np.diag([-1.0, -1.0, 1.0]),
    )
    assert np.allclose(np.asarray(extr.data["T_cam_world"])[:3, 3], [1.0, 2.0, 3.0])


def test_set_gripper_direct_env_falls_back_to_command_joint_state() -> None:
    env = _DirectEnv()
    result = SetGripperTool(env=env).execute(side="left", pos=0.25)

    assert result.success is True
    assert env.commands
    side, state = env.commands[-1]
    assert side == "left"
    assert np.asarray(state["pos"]).shape == (7,)
    assert float(state["pos"][-1]) == 0.25


def test_native_state_and_camera_image_direct_env() -> None:
    env = _DirectEnv()

    state = GetRobotStateTool(env=env).execute()
    assert state.success is True
    assert state.data.left_joint_pos == [0.0] * 6
    assert state.data.right_ee_pos == [0.4, -0.2, 0.8]

    image = GetCameraImageTool(env=env).execute(camera="top")
    assert image.success is True
    assert image.data.shape == (2, 3, 3)


def test_freespace_execute_trajectory_direct_uses_bimanual_env_method() -> None:
    env = _DirectEnv()
    tool = FreespaceMoveTool(env=env)
    err = tool._execute_trajectory(
        "left",
        [0.0, 0.5],
        np.zeros((2, 6), dtype=np.float64),
        np.ones((2, 6), dtype=np.float64),
    )

    assert err is None
    assert len(env.bimanual_calls) == 1


def test_nudge_direct_env_delegates_to_freespace(monkeypatch) -> None:
    env = _DirectEnv()
    calls: list[dict] = []

    def _fake_execute(self, **kwargs):
        calls.append(kwargs)
        return ToolResult(
            success=True,
            data=FreespaceResult(status="Success", executed=False),
        )

    monkeypatch.setattr(FreespaceMoveTool, "execute", _fake_execute)

    result = NudgeTool(env=env).execute(
        side="right",
        delta_pos=[0.0, 0.0, 0.01],
        preview_only=True,
    )

    assert result.success is True
    assert calls
    assert calls[-1]["right_target_pos"] == [0.4, -0.2, 0.81]
    assert calls[-1]["preview_only"] is True
    assert "right_target_quat" in calls[-1]


def test_nudge_world_pose_preserving_is_freespace_compat_wrapper(monkeypatch) -> None:
    from enpire.env.forge.cap.env.real_bimanual_yam import skills

    env = _DirectEnv()
    calls: list[dict] = []

    def _fake_execute(self, **kwargs):
        calls.append(kwargs)
        return ToolResult(
            success=True,
            data=FreespaceResult(
                status="Success",
                executed=False,
                final_pos_error_m=0.001,
                final_rot_error_deg=0.5,
                trajectory_steps=4,
            ),
        )

    monkeypatch.setattr(FreespaceMoveTool, "execute", _fake_execute)

    namespace = skills.make_namespace(env)
    result = namespace["nudge_world_pose_preserving"](
        side="left",
        world_delta_m=[0.01, -0.02, 0.03],
        initial_pos_m=[0.5, 0.1, 0.8],
        initial_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        preview_only=True,
    )

    assert result["success"] is True
    assert result["target_pos"] == [0.51, 0.08, 0.8300000000000001]
    assert calls
    assert calls[-1]["left_target_pos"] == [0.51, 0.08, 0.8300000000000001]
    assert calls[-1]["left_target_quat"] == [0.0, 0.0, 0.0, 1.0]
    assert calls[-1]["preview_only"] is True


def test_debug_ui_scans_nested_vis_artifact_images(tmp_path) -> None:
    vis_img = tmp_path / "vis" / "mask.png"
    vlm_img = tmp_path / "vis" / "vlm" / "001_query" / "camera_top.png"
    vis_img.parent.mkdir(parents=True)
    vlm_img.parent.mkdir(parents=True)
    vis_img.write_bytes(b"vis")
    vlm_img.write_bytes(b"vlm")

    monitor = LogMonitor(tmp_path, "debug_events.jsonl", 0.25)
    monitor._scan_images()
    monitor._scan_images()

    paths = {image["path"] for image in monitor.snapshot()["images"]}
    assert paths == {"vis/mask.png", "vis/vlm/001_query/camera_top.png"}
