import queue
from pathlib import Path

from fastapi.testclient import TestClient

from enpire.policy.rl.fastapi_server import build_app


def _client(tmp_path):
    events = queue.Queue()
    app = build_app(event_queue=events, data_root=tmp_path, config_files=None)
    return TestClient(app), events


def test_control_routes_enqueue_expected_events(tmp_path):
    client, events = _client(tmp_path)

    assert client.post("/home").status_code == 200
    event, payload = events.get_nowait()
    assert event == "home"
    assert payload["source"] == "fastapi"

    assert client.post("/resume").status_code == 200
    event, payload = events.get_nowait()
    assert event == "start"
    assert payload["source"] == "fastapi"

    assert client.post("/pause").status_code == 200
    event, payload = events.get_nowait()
    assert event == "parking"
    assert payload["source"] == "fastapi"


def test_restart_creates_new_output_dir_and_enqueues_path(tmp_path):
    client, events = _client(tmp_path)

    response = client.post("/restart")

    assert response.status_code == 200
    path = response.json()["path"]
    assert Path(path).is_dir()
    assert events.get_nowait() == ("restart", {"path": path, "source": "fastapi"})


def test_help_only_lists_exposed_routes(tmp_path):
    client, _ = _client(tmp_path)

    response = client.get("/help")

    assert response.status_code == 200
    endpoints = response.json()["endpoints"]
    assert "POST /home" in endpoints
    assert "POST /pause" in endpoints
    assert "POST /resume" in endpoints
    assert "POST /restart" in endpoints
    assert "POST /set_initial_position_index" not in endpoints
    assert "GET  /initial_position_index" not in endpoints


def test_initial_position_routes_are_not_exposed(tmp_path):
    client, events = _client(tmp_path)

    assert client.get("/initial_position_index").status_code == 404
    assert client.post("/set_initial_position_index").status_code == 404
    assert events.empty()
