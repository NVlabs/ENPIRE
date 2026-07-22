# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from fastapi.testclient import TestClient

from enpire.env.forge.cap.agent.tools import ToolRegistry
from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult


class _DummyVisualizer:
    def __init__(self, port: int):
        self.port = port

    def clear_prediction(self):
        pass

    def clear_grasp_poses(self):
        pass

    def clear_scene_objects(self):
        pass

    def update_detections(self, *_args, **_kwargs):
        pass

    def set_robot_callbacks(self, **_kwargs):
        pass


class _FakeFreespaceMoveTool(Tool):
    name = "freespace_move"
    description = "fake freespace planner"
    parameters = [
        ToolParameter("left_target_pos", "list[float]", "Left target position", required=False),
        ToolParameter("left_target_rpy", "list[float]", "Left target RPY", required=False),
        ToolParameter("right_target_pos", "list[float]", "Right target position", required=False),
        ToolParameter("right_target_rpy", "list[float]", "Right target RPY", required=False),
        ToolParameter("planner_backend", "str", "Planner backend", required=False, default="curobo"),
    ]

    def execute(self, **kwargs):
        backend = kwargs.get("planner_backend", "curobo")
        return ToolResult(
            success=False,
            data={"status": "Error", "reason": "planner bootstrap failed"},
            error=f"Failed to import cuRobo for backend={backend}",
        )


class _ImmediateResult:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakePortalClient:
    def __init__(self):
        self.estop_calls = 0
        self.release_estop_calls = 0

    def estop(self):
        self.estop_calls += 1
        return _ImmediateResult(True)

    def release_estop(self):
        self.release_estop_calls += 1
        return _ImmediateResult(True)


class _FakeGetRobotStateTool(Tool):
    name = "get_robot_state"
    description = "fake get_robot_state"
    parameters = []

    def __init__(self, client: _FakePortalClient):
        self._host = "localhost"
        self._port = 8300
        self._client = client

    def _get_client(self, host, port):
        assert host == self._host
        assert port == self._port
        return self._client

    def execute(self, **kwargs):
        return ToolResult(success=True, data={"ok": True})


def test_freespace_move_failure_broadcasts_runtime_error(monkeypatch):
    import enpire.env.forge.cap.agent.cap_agent as cap_agent_module

    registry = ToolRegistry()
    registry.register(_FakeFreespaceMoveTool())

    monkeypatch.setattr(cap_agent_module, "CapVisualizer", _DummyVisualizer)
    monkeypatch.setattr(cap_agent_module, "create_default_registry", lambda **_kwargs: registry)

    app = cap_agent_module.create_app(viser_port=18080)
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            initial = json.loads(ws.receive_text())
            assert initial["type"] == "status_change"

            resp = client.post(
                "/api/execute",
                json={
                    "code": (
                        "freespace_move("
                        "left_target_pos=[0.22, 0.42, 1.02], "
                        "left_target_rpy=[0, 90, 0], "
                        "planner_backend='curobo'"
                        ")"
                    )
                },
            )
            assert resp.status_code == 200
            assert resp.json()["ok"] is False

            seen_error = None
            for _ in range(10):
                message = json.loads(ws.receive_text())
                if message["type"] == "error":
                    seen_error = message["data"]
                    break

            assert seen_error is not None
            assert seen_error["source"] == "freespace_move[curobo]"
            assert "Failed to import cuRobo" in seen_error["detail"]


def test_execute_auto_releases_estop(monkeypatch):
    import enpire.env.forge.cap.agent.cap_agent as cap_agent_module

    client_stub = _FakePortalClient()
    registry = ToolRegistry()
    registry.register(_FakeGetRobotStateTool(client_stub))

    monkeypatch.setattr(cap_agent_module, "CapVisualizer", _DummyVisualizer)
    monkeypatch.setattr(cap_agent_module, "create_default_registry", lambda **_kwargs: registry)

    app = cap_agent_module.create_app(viser_port=18081)
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    with TestClient(app) as client:
        estop_resp = client.post("/api/estop")
        assert estop_resp.status_code == 200
        assert estop_resp.json()["ok"] is True
        assert client_stub.estop_calls == 1

        execute_resp = client.post("/api/execute", json={"code": "x = 1"})
        assert execute_resp.status_code == 200
        assert execute_resp.json()["ok"] is True
        assert client_stub.release_estop_calls == 1
