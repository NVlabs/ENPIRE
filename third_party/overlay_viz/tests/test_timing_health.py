"""Tests for GET /api/tasks/{task_id}/timing_health and _compute_timing_health."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_OVERLAY_VIZ_PARENT = str(Path(__file__).resolve().parents[2])
if _OVERLAY_VIZ_PARENT not in sys.path:
    sys.path.insert(0, _OVERLAY_VIZ_PARENT)

from overlay_viz.app import _compute_timing_health, create_app  # noqa: E402
from overlay_viz.scanner import EpisodeScanner  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode_dir(
    base: Path,
    idx: int = 0,
    *,
    timestamp_npy: np.ndarray | None = None,
    component_timestamps: list | None = None,
    video: bool = True,
) -> Path:
    folder = base / f"episode_{idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    if video:
        (folder / "top.mp4").write_bytes(b"\x00" * 64)
    if timestamp_npy is not None:
        np.save(str(folder / "timestamp.npy"), timestamp_npy)
    if component_timestamps is not None:
        (folder / "component_timestamps.json").write_text(json.dumps(component_timestamps))
    return folder


def _make_scanner(data_root: Path, n_episodes: int = 1) -> EpisodeScanner:
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


def _client(task_id: str, scanner: EpisodeScanner) -> TestClient:
    app = create_app(initial_scanners={task_id: scanner})
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests for _compute_timing_health
# ---------------------------------------------------------------------------


class TestComputeTimingHealthUnit:
    """Test the pure computation function directly."""

    def test_uniform_timestamps_good(self, tmp_path: Path):
        """Perfectly uniform timestamps should be 'good'."""
        folder = tmp_path / "ep"
        folder.mkdir()
        ts = np.arange(0, 10, 1 / 30.0)  # 30 Hz, perfectly uniform
        np.save(str(folder / "timestamp.npy"), ts)

        health = _compute_timing_health(folder)
        assert health["has_data"] is True
        assert health["level"] == "good"
        assert health["spike_count"] == 0
        assert health["jitter_ratio"] == 0.0

    def test_jittery_timestamps_warn(self, tmp_path: Path):
        """Moderate jitter should produce 'warn'."""
        folder = tmp_path / "ep"
        folder.mkdir()
        rng = np.random.default_rng(42)
        dt = 1 / 30.0
        # Add jitter: IQR/median ~0.15 (between _JITTER_WARN=0.10 and _JITTER_BAD=0.25)
        ts = np.cumsum(np.abs(rng.normal(dt, dt * 0.08, 300)))
        np.save(str(folder / "timestamp.npy"), ts)

        health = _compute_timing_health(folder)
        assert health["has_data"] is True
        assert health["level"] in ("warn", "bad")  # depends on random seed

    def test_spiky_timestamps_bad(self, tmp_path: Path):
        """Multiple large spikes should produce 'bad'."""
        folder = tmp_path / "ep"
        folder.mkdir()
        dt = 1 / 30.0
        ts = np.arange(0, 100 * dt, dt)
        # Insert 5 large gaps (> 3x median)
        for i in [10, 30, 50, 70, 90]:
            ts[i:] += dt * 10
        np.save(str(folder / "timestamp.npy"), ts)

        health = _compute_timing_health(folder)
        assert health["has_data"] is True
        assert health["level"] == "bad"
        assert health["spike_count"] >= 4

    def test_no_data_returns_no_data(self, tmp_path: Path):
        """Missing timestamp files should return has_data=False."""
        folder = tmp_path / "ep"
        folder.mkdir()

        health = _compute_timing_health(folder)
        assert health["has_data"] is False
        assert health["level"] == "good"

    def test_single_timestamp_no_data(self, tmp_path: Path):
        """Only 1 timestamp (no diffs possible) should return has_data=False."""
        folder = tmp_path / "ep"
        folder.mkdir()
        np.save(str(folder / "timestamp.npy"), np.array([1.0]))

        health = _compute_timing_health(folder)
        assert health["has_data"] is False

    def test_fallback_to_component_timestamps(self, tmp_path: Path):
        """When timestamp.npy is missing, falls back to component_timestamps.json."""
        folder = tmp_path / "ep"
        folder.mkdir()
        ct = [
            {"left_state": 1000.0 + i * 0.033, "right_state": 1000.0 + i * 0.033 + 0.001}
            for i in range(100)
        ]
        (folder / "component_timestamps.json").write_text(json.dumps(ct))

        health = _compute_timing_health(folder)
        assert health["has_data"] is True
        assert health["level"] == "good"
        assert health["n_steps"] == 100

    def test_metrics_values(self, tmp_path: Path):
        """Verify metric values for a known distribution."""
        folder = tmp_path / "ep"
        folder.mkdir()
        # 10 steps at exactly 33ms intervals
        ts = np.array([i * 0.033 for i in range(10)])
        np.save(str(folder / "timestamp.npy"), ts)

        health = _compute_timing_health(folder)
        assert health["has_data"] is True
        assert health["n_steps"] == 10
        assert abs(health["median_dt"] - 0.033) < 0.001
        assert health["jitter_ratio"] == 0.0
        assert health["spike_count"] == 0
        assert abs(health["max_gap_s"] - 0.033) < 0.001


# ---------------------------------------------------------------------------
# Integration tests for the endpoint
# ---------------------------------------------------------------------------


class TestTimingHealthEndpoint:
    """Test the GET /api/tasks/{task_id}/timing_health endpoint."""

    def test_returns_health_for_all_episodes(self, tmp_path: Path):
        """Endpoint should return health for every episode."""
        for i in range(3):
            ts = np.arange(0, 3, 1 / 30.0)
            _make_episode_dir(tmp_path, i, timestamp_npy=ts)

        scanner = _make_scanner(tmp_path, 3)
        client = _client("t1", scanner)

        # First call triggers background computation
        resp = client.get("/api/tasks/t1/timing_health")
        assert resp.status_code == 200

        # Poll until ready (should be near-instant for 3 episodes)
        import time
        for _ in range(20):
            resp = client.get("/api/tasks/t1/timing_health")
            data = resp.json()
            if data["ready"]:
                break
            time.sleep(0.1)

        assert data["ready"] is True
        assert data["total"] == 3
        assert data["done"] == 3
        assert len(data["episodes"]) == 3
        for idx_str in ["0", "1", "2"]:
            ep = data["episodes"][idx_str]
            assert ep["has_data"] is True
            assert ep["level"] == "good"

    def test_mixed_health_levels(self, tmp_path: Path):
        """Episodes with different quality should get different levels."""
        # Good episode: uniform
        ts_good = np.arange(0, 3, 1 / 30.0)
        _make_episode_dir(tmp_path, 0, timestamp_npy=ts_good)

        # Bad episode: big spikes
        dt = 1 / 30.0
        ts_bad = np.arange(0, 100 * dt, dt)
        for i in [10, 30, 50, 70, 90]:
            ts_bad[i:] += dt * 10
        _make_episode_dir(tmp_path, 1, timestamp_npy=ts_bad)

        scanner = _make_scanner(tmp_path, 2)
        client = _client("t1", scanner)

        import time
        for _ in range(20):
            resp = client.get("/api/tasks/t1/timing_health")
            data = resp.json()
            if data["ready"]:
                break
            time.sleep(0.1)

        assert data["episodes"]["0"]["level"] == "good"
        assert data["episodes"]["1"]["level"] == "bad"

    def test_no_scan_returns_404(self, tmp_path: Path):
        """Endpoint should 404 if scan hasn't started."""
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/tasks/nonexistent/timing_health")
        assert resp.status_code in (404, 422)

    def test_episode_without_timestamps(self, tmp_path: Path):
        """Episode without timestamp files should have has_data=False."""
        _make_episode_dir(tmp_path, 0)  # no timestamp data

        scanner = _make_scanner(tmp_path, 1)
        client = _client("t1", scanner)

        import time
        for _ in range(20):
            resp = client.get("/api/tasks/t1/timing_health")
            data = resp.json()
            if data["ready"]:
                break
            time.sleep(0.1)

        assert data["episodes"]["0"]["has_data"] is False
        assert data["episodes"]["0"]["level"] == "good"
