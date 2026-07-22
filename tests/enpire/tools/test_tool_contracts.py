# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

import numpy as np

from enpire.env.forge.cap.agent.tools import segmentation, vlm_query
from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool
from enpire.env.forge.cap.agent.tools.native import GetRobotStateTool, SetGripperTool
from enpire.env.forge.cap.agent.tools.segmentation import SegmentObjectTool
from enpire.env.forge.cap.agent.tools.vlm_query import VlmQueryTool


class _Env:
    def __init__(self) -> None:
        self.gripper_calls: list[tuple] = []

    def render_rgb(self, camera: str) -> np.ndarray:
        assert camera == "top"
        return np.full((16, 16, 3), 127, dtype=np.uint8)

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        offset = 0.1 if side == "left" else -0.1
        return {
            "joint_pos": np.arange(6, dtype=np.float32) + offset,
            "gripper_pos": np.array([0.5], dtype=np.float32),
            "ee_pos": np.array([offset, 0.0, 1.0], dtype=np.float32),
            "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def set_gripper(self, *args):
        self.gripper_calls.append(args)
        return {"success": True, "side": args[0], "gripper": args[1]}


def _encoded_mask(mask: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, mask)
    return base64.b64encode(stream.getvalue()).decode()


def test_vision_segmentation_uses_original_service_payload(monkeypatch) -> None:
    mask = np.ones((16, 16), dtype=bool)
    captured: dict = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "score": 0.9,
                "mask_b64": _encoded_mask(mask),
                "bbox_xywh": [0, 0, 16, 16],
                "mask_area": 256,
            }

    def post(url, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(segmentation.requests, "post", post)
    monkeypatch.setattr(segmentation, "log_mask", lambda *args, **kwargs: None)

    result = SegmentObjectTool(env=_Env()).execute(
        query="red cube", media="camera:top", score_thresh=0.5
    )

    assert result.success is True
    assert result.data.score == 0.9
    assert captured["payload"]["text"] == "red cube"
    assert captured["url"].endswith("/segment")


def test_control_state_and_gripper_run_against_direct_env() -> None:
    env = _Env()

    state = GetRobotStateTool(env=env).execute()
    grip = SetGripperTool(env=env).execute(side="left", pos=0.25)

    assert state.success is True
    assert set(state.data.arms) == {"left", "right"}
    assert grip.success is True
    assert env.gripper_calls[0][0:2] == ("left", 0.25)


def test_planning_rejects_ambiguous_cached_trajectory_without_motion() -> None:
    result = FreespaceMoveTool(env=_Env()).execute(
        trajectory_cache_key="preview-1",
        left_target_pos=[0.4, 0.0, 0.8],
    )

    assert result.success is False
    assert result.data.status == "Invalid"
    assert "cannot be combined" in result.error


def test_vlm_tool_routes_images_and_backend_without_credentials(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        vlm_query,
        "_transport_query",
        lambda **kwargs: calls.append(kwargs) or "cube is centered",
    )
    import enpire.env.forge.cap.agent.tools._artifact_log as artifact_log

    monkeypatch.setattr(artifact_log, "log_vlm_query", lambda **kwargs: None)

    result = VlmQueryTool(env=_Env()).execute(
        text="Where is the cube?", backend="qwen", media=["camera:top"]
    )

    assert result.success is True
    assert result.data == "cube is centered"
    assert calls[0]["backend"] == "qwen"
    assert calls[0]["images"][0].shape == (16, 16, 3)
