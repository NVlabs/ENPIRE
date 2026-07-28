# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for GET /api/tasks/{task_id}/episodes/{idx}/component_timestamps.

Tests cover happy paths, error paths, caching, concurrent access, malformed
data, and response schema validation.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# The overlay_viz package lives under third_party/ which is excluded from
# the main project's setuptools packages.  We need the repo root on
# sys.path so that ``from experimental.task_options_config import ...``
# (used inside app.py) resolves correctly.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also add the overlay_viz parent so relative imports inside the package work
# when pytest runs from the repo root.
_OVERLAY_VIZ_PARENT = str(Path(__file__).resolve().parents[2])
if _OVERLAY_VIZ_PARENT not in sys.path:
    sys.path.insert(0, _OVERLAY_VIZ_PARENT)

from fastapi.testclient import TestClient  # noqa: E402

from overlay_viz.app import create_app  # noqa: E402
from overlay_viz.scanner import EpisodeScanner  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode_dir(
    base: Path,
    idx: int = 0,
    *,
    component_timestamps: Any | None = "DEFAULT",
    action_source_json: Any | None = None,
    action_source_npy: Any | None = None,
    timestamp_npy: np.ndarray | None = None,
    video: bool = True,
) -> Path:
    """Create a fake episode directory with the requested files.

    ``component_timestamps="DEFAULT"`` writes a small valid payload.
    Pass ``None`` to omit the file entirely, or a raw string to write
    malformed content.
    """
    folder = base / f"episode_{idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)

    # The scanner expects a video file to exist for an entry to be valid
    if video:
        (folder / "top.mp4").write_bytes(b"\x00" * 64)

    if component_timestamps == "DEFAULT":
        data = [
            {
                "left_state": 1000.0 + i * 0.05,
                "right_state": 1000.0 + i * 0.05 + 0.001,
                "left_action": 1000.0 + i * 0.05 + 0.002,
                "right_action": 1000.0 + i * 0.05 + 0.003,
            }
            for i in range(10)
        ]
        (folder / "component_timestamps.json").write_text(json.dumps(data))
    elif component_timestamps is not None:
        # Allow writing raw string (e.g. malformed JSON)
        (folder / "component_timestamps.json").write_text(str(component_timestamps))

    if action_source_json is not None:
        (folder / "action-source.json").write_text(json.dumps(action_source_json))

    if action_source_npy is not None:
        np.save(str(folder / "action-source.npy"), np.array(action_source_npy, dtype=object))

    if timestamp_npy is not None:
        np.save(str(folder / "timestamp.npy"), timestamp_npy)

    return folder


def _make_scanner(data_root: Path, n_episodes: int = 1) -> EpisodeScanner:
    """Build an EpisodeScanner whose internals point at pre-built episode dirs.

    We bypass the real scanning (which requires actual video files and
    OpenCV) and directly inject entries.
    """
    scanner = EpisodeScanner(data_root)
    entries = []
    for i in range(n_episodes):
        folder = data_root / f"episode_{i:04d}"
        video = folder / "top.mp4"
        if folder.exists():
            entries.append((folder, video))
    scanner._entries = entries
    scanner._ready = True
    return scanner


def _client(
    task_id: str,
    scanner: EpisodeScanner,
) -> TestClient:
    """Build a TestClient with a pre-seeded scanner for the given task."""
    app = create_app(initial_scanners={task_id: scanner})
    # Clear the module-level cache that persists across calls inside
    # ``create_app``'s closure.  We access it via the endpoint's own
    # ``__self__`` — but since it's a closure var we need a different
    # approach: just create a fresh app each time.
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


# ---------------------------------------------------------------------------
# 1. Happy path -- valid episode with all files present
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_200_with_all_files(self, data_root: Path) -> None:
        _make_episode_dir(
            data_root,
            0,
            action_source_json=["human"] * 10,
            timestamp_npy=np.linspace(1000.0, 1000.5, 10),
        )
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 200
        body = resp.json()

        assert "timestamps" in body
        assert "action_sources" in body
        assert "record_timestamps" in body
        assert len(body["timestamps"]) == 10
        assert len(body["action_sources"]) == 10
        assert len(body["record_timestamps"]) == 10

    def test_timestamps_contain_expected_keys(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        step = resp.json()["timestamps"][0]
        assert "left_state" in step
        assert "right_state" in step
        assert "left_action" in step
        assert "right_action" in step

    def test_values_are_floats(self, data_root: Path) -> None:
        _make_episode_dir(
            data_root,
            0,
            timestamp_npy=np.array([1.0, 2.0, 3.0]),
        )
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        for ts in body["record_timestamps"]:
            assert isinstance(ts, float)


# ---------------------------------------------------------------------------
# 2. Missing component_timestamps.json -> 404
# ---------------------------------------------------------------------------


class TestMissingComponentTimestamps:
    def test_returns_404_when_json_missing(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0, component_timestamps=None)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 404
        assert "component_timestamps.json" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Missing episode index -> 404
# ---------------------------------------------------------------------------


class TestMissingEpisode:
    def test_out_of_range_idx(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/999/component_timestamps")
        assert resp.status_code == 404
        assert "Episode 999 not found" in resp.json()["detail"]

    def test_negative_idx(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/-1/component_timestamps")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. Scanner not ready -> 404
# ---------------------------------------------------------------------------


class TestScannerNotReady:
    def test_scanner_not_ready(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        scanner._ready = False  # force not-ready
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 404
        assert "Scan not ready" in resp.json()["detail"]

    def test_scanner_none(self, data_root: Path) -> None:
        """Task registered but scanner never created (only path registered)."""
        # Build app with no scanners then manually register a path
        app = create_app()
        client = TestClient(app)
        # Accessing a task that doesn't exist at all
        resp = client.get("/api/tasks/nonexistent/episodes/0/component_timestamps")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. Invalid task_id
# ---------------------------------------------------------------------------


class TestInvalidTaskId:
    def test_unknown_task(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/BOGUS/episodes/0/component_timestamps")
        assert resp.status_code == 404
        assert "Unknown task" in resp.json()["detail"]

    def test_empty_task_id(self, data_root: Path) -> None:
        """Empty string task id should not match any registered task."""
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        # FastAPI treats this as a different route, likely 404 or 405
        resp = client.get("/api/tasks//episodes/0/component_timestamps")
        assert resp.status_code in (404, 405, 422)

    def test_task_id_with_slashes(self, data_root: Path) -> None:
        """Path traversal attempt via task_id."""
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/../../etc/passwd/episodes/0/component_timestamps")
        assert resp.status_code in (404, 422)

    def test_task_id_with_special_chars(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get(
            "/api/tasks/%00null_byte/episodes/0/component_timestamps"
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 6. Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_second_call_returns_same_object(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        r1 = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        r2 = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()

    def test_cache_does_not_reload_from_disk(self, data_root: Path) -> None:
        """After the first load, modifying the file on disk should not change
        the response (because the result is cached)."""
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        r1 = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert r1.status_code == 200
        original_body = r1.json()

        # Mutate the file on disk
        ts_path = data_root / "episode_0000" / "component_timestamps.json"
        ts_path.write_text(json.dumps([{"mutated": True}]))

        r2 = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert r2.json() == original_body  # still cached

    def test_different_episodes_cache_independently(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        _make_episode_dir(data_root, 1)
        scanner = _make_scanner(data_root, 2)
        client = _client("test_task", scanner)

        r0 = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        r1 = client.get("/api/tasks/test_task/episodes/1/component_timestamps")
        assert r0.status_code == 200
        assert r1.status_code == 200
        # Both valid but independent
        assert r0.json()["timestamps"] == r1.json()["timestamps"]  # same default data

    def test_different_tasks_cache_independently(self, data_root: Path) -> None:
        """Two tasks with the same episode idx should not share cache."""
        root_a = data_root / "task_a_data"
        root_b = data_root / "task_b_data"
        _make_episode_dir(root_a, 0)
        _make_episode_dir(root_b, 0)

        scanner_a = _make_scanner(root_a, 1)
        scanner_b = _make_scanner(root_b, 1)

        app = create_app(initial_scanners={"task_a": scanner_a, "task_b": scanner_b})
        client = TestClient(app)

        ra = client.get("/api/tasks/task_a/episodes/0/component_timestamps")
        rb = client.get("/api/tasks/task_b/episodes/0/component_timestamps")
        assert ra.status_code == 200
        assert rb.status_code == 200


# ---------------------------------------------------------------------------
# 7. action-source.json present
# ---------------------------------------------------------------------------


class TestActionSourceJson:
    def test_action_source_json_loaded(self, data_root: Path) -> None:
        sources = ["human", "policy", "human", "policy", "human",
                    "policy", "human", "policy", "human", "policy"]
        _make_episode_dir(data_root, 0, action_source_json=sources)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["action_sources"] == sources

    def test_action_source_json_empty_list(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0, action_source_json=[])
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["action_sources"] == []


# ---------------------------------------------------------------------------
# 8. action-source.npy instead of json
# ---------------------------------------------------------------------------


class TestActionSourceNpy:
    def test_npy_used_when_json_absent(self, data_root: Path) -> None:
        sources = ["human", "policy", "human"]
        _make_episode_dir(data_root, 0, action_source_npy=sources)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["action_sources"] == sources

    def test_json_preferred_over_npy(self, data_root: Path) -> None:
        """When both files exist, json wins."""
        json_sources = ["json_source"] * 10
        npy_sources = ["npy_source"] * 10
        _make_episode_dir(
            data_root, 0,
            action_source_json=json_sources,
            action_source_npy=npy_sources,
        )
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["action_sources"] == json_sources


# ---------------------------------------------------------------------------
# 9. Neither action source file
# ---------------------------------------------------------------------------


class TestNoActionSource:
    def test_returns_empty_list(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["action_sources"] == []


# ---------------------------------------------------------------------------
# 10. timestamp.npy present vs missing
# ---------------------------------------------------------------------------


class TestRecordTimestamps:
    def test_timestamp_npy_loaded(self, data_root: Path) -> None:
        ts = np.array([100.0, 100.05, 100.10])
        _make_episode_dir(data_root, 0, timestamp_npy=ts)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert len(body["record_timestamps"]) == 3
        assert pytest.approx(body["record_timestamps"], abs=1e-6) == ts.tolist()

    def test_missing_timestamp_npy_returns_empty(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["record_timestamps"] == []


# ---------------------------------------------------------------------------
# 11. Malformed JSON in component_timestamps.json
# ---------------------------------------------------------------------------


class TestMalformedJson:
    def test_invalid_json_raises_error(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0, component_timestamps="{not valid json!!!")
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        # The endpoint does not catch json.JSONDecodeError, so the
        # exception propagates.  TestClient with raise_server_exceptions=False
        # will return a 500; with the default (True) it raises.
        with pytest.raises(json.JSONDecodeError):
            client.get("/api/tasks/test_task/episodes/0/component_timestamps")

    def test_invalid_json_returns_500_when_exceptions_suppressed(
        self, data_root: Path
    ) -> None:
        """Same scenario, but verify the HTTP 500 when exceptions are suppressed."""
        _make_episode_dir(data_root, 0, component_timestamps="{not valid json!!!")
        scanner = _make_scanner(data_root, 1)
        app = create_app(initial_scanners={"test_task": scanner})
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 500

    def test_truncated_json(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0, component_timestamps='[{"a": 1}, {"b":')
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        with pytest.raises(json.JSONDecodeError):
            client.get("/api/tasks/test_task/episodes/0/component_timestamps")

    def test_json_is_string_not_list(self, data_root: Path) -> None:
        """JSON is valid but has wrong type (string instead of list) -- rejected by validation."""
        _make_episode_dir(data_root, 0, component_timestamps='"just a string"')
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 404

    def test_json_is_null(self, data_root: Path) -> None:
        """null is valid JSON but not a list -- rejected by validation."""
        _make_episode_dir(data_root, 0, component_timestamps="null")
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 12. Empty timestamps array
# ---------------------------------------------------------------------------


class TestEmptyTimestamps:
    def test_empty_list(self, data_root: Path) -> None:
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")
        (folder / "component_timestamps.json").write_text("[]")

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["timestamps"] == []
        assert body["action_sources"] == []
        assert body["record_timestamps"] == []


# ---------------------------------------------------------------------------
# 13. Large episode (many steps)
# ---------------------------------------------------------------------------


class TestLargeEpisode:
    def test_1000_steps(self, data_root: Path) -> None:
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")

        big_ts = [{"comp": float(i)} for i in range(1000)]
        (folder / "component_timestamps.json").write_text(json.dumps(big_ts))
        np.save(str(folder / "timestamp.npy"), np.arange(1000, dtype=np.float64))

        sources = ["human"] * 500 + ["policy"] * 500
        (folder / "action-source.json").write_text(json.dumps(sources))

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["timestamps"]) == 1000
        assert len(body["action_sources"]) == 1000
        assert len(body["record_timestamps"]) == 1000


# ---------------------------------------------------------------------------
# 14. Concurrent requests to same endpoint
# ---------------------------------------------------------------------------


class TestConcurrentRequests:
    def test_concurrent_same_episode(self, data_root: Path) -> None:
        """Multiple threads hitting the same episode should not corrupt cache."""
        _make_episode_dir(
            data_root, 0,
            action_source_json=["human"] * 10,
            timestamp_npy=np.linspace(0, 1, 10),
        )
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        results: list[dict] = []
        errors: list[Exception] = []

        def _fetch() -> None:
            try:
                r = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
                assert r.status_code == 200
                results.append(r.json())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent fetch: {errors}"
        assert len(results) == 10
        # All results should be identical
        for r in results[1:]:
            assert r == results[0]

    def test_concurrent_different_episodes(self, data_root: Path) -> None:
        for i in range(5):
            _make_episode_dir(data_root, i)
        scanner = _make_scanner(data_root, 5)
        client = _client("test_task", scanner)

        results: dict[int, dict] = {}
        errors: list[Exception] = []

        def _fetch(ep_idx: int) -> None:
            try:
                r = client.get(
                    f"/api/tasks/test_task/episodes/{ep_idx}/component_timestamps"
                )
                assert r.status_code == 200
                results[ep_idx] = r.json()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_fetch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 5


# ---------------------------------------------------------------------------
# 15. Response schema validation
# ---------------------------------------------------------------------------


class TestResponseSchema:
    """Verify the response object matches the ComponentTimestampData interface."""

    def test_top_level_keys_only(self, data_root: Path) -> None:
        _make_episode_dir(
            data_root, 0,
            action_source_json=["human"],
            timestamp_npy=np.array([1.0]),
        )
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert set(body.keys()) == {"timestamps", "action_sources", "record_timestamps"}

    def test_timestamps_is_list(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert isinstance(body["timestamps"], list)

    def test_action_sources_is_list_of_strings(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0, action_source_json=["human", "policy"])
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert isinstance(body["action_sources"], list)
        for item in body["action_sources"]:
            assert isinstance(item, str)

    def test_record_timestamps_is_list_of_numbers(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0, timestamp_npy=np.array([1.0, 2.0]))
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert isinstance(body["record_timestamps"], list)
        for item in body["record_timestamps"]:
            assert isinstance(item, (int, float))

    def test_content_type_is_json(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Edge cases and adversarial inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_episode_idx_zero(self, data_root: Path) -> None:
        """Idx 0 is valid and should work."""
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 200

    def test_idx_not_integer_returns_422(self, data_root: Path) -> None:
        """Non-integer idx should be rejected by FastAPI's path validation."""
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/abc/component_timestamps")
        assert resp.status_code == 422

    def test_very_large_idx(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/999999999/component_timestamps")
        assert resp.status_code == 404

    def test_float_idx_rejected(self, data_root: Path) -> None:
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0.5/component_timestamps")
        assert resp.status_code == 422

    def test_component_timestamps_with_extra_keys(self, data_root: Path) -> None:
        """Extra unexpected keys in the step dicts should pass through."""
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")
        data = [{"left_state": 1.0, "extra_weird_key": "surprise"}]
        (folder / "component_timestamps.json").write_text(json.dumps(data))

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["timestamps"][0]["extra_weird_key"] == "surprise"

    def test_action_source_npy_with_non_string_values(self, data_root: Path) -> None:
        """npy file containing integers instead of strings."""
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")
        data = [{"step": 0}]
        (folder / "component_timestamps.json").write_text(json.dumps(data))
        np.save(str(folder / "action-source.npy"), np.array([1, 2, 3]))

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 200
        # Values come through as integers
        body = resp.json()
        assert isinstance(body["action_sources"], list)

    def test_head_request_works(self, data_root: Path) -> None:
        """The app has HEAD->GET middleware for /api/ paths."""
        _make_episode_dir(data_root, 0)
        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.head("/api/tasks/test_task/episodes/0/component_timestamps")
        # Should return 200 with empty body due to HEAD middleware
        assert resp.status_code == 200

    def test_unicode_in_component_timestamps(self, data_root: Path) -> None:
        """Unicode strings in timestamps JSON should serialize fine."""
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")
        data = [{"label": "\u2603 snowman", "val": 1.0}]
        (folder / "component_timestamps.json").write_text(json.dumps(data, ensure_ascii=False))

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["timestamps"][0]["label"] == "\u2603 snowman"

    def test_deeply_nested_timestamps(self, data_root: Path) -> None:
        """Nested objects in timestamps JSON are passed through verbatim."""
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")
        data = [{"nested": {"deep": {"value": 42}}}]
        (folder / "component_timestamps.json").write_text(json.dumps(data))

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        body = client.get("/api/tasks/test_task/episodes/0/component_timestamps").json()
        assert body["timestamps"][0]["nested"]["deep"]["value"] == 42

    def test_empty_episode_folder(self, data_root: Path) -> None:
        """Episode folder exists but has no files at all."""
        folder = data_root / "episode_0000"
        folder.mkdir(parents=True)
        (folder / "top.mp4").write_bytes(b"\x00")  # video needed for scanner

        scanner = _make_scanner(data_root, 1)
        client = _client("test_task", scanner)

        resp = client.get("/api/tasks/test_task/episodes/0/component_timestamps")
        assert resp.status_code == 404
        assert "component_timestamps.json" in resp.json()["detail"]
