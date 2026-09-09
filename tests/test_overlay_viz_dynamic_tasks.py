# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from third_party.overlay_viz.app import (
    build_value_prediction_cache_payload,
    create_app,
    fetch_value_predictions_from_server,
    normalize_value_server_url,
    read_local_value_prediction_cache,
    value_prediction_cache_matches_episode,
    value_prediction_cache_matches_server,
    write_local_value_prediction_cache,
)
from third_party.overlay_viz.scanner import CAMERA_FILENAMES, VIDEO_FILENAME, EpisodeScanner


class _ScannerStub:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.ready = False
        self.progress = (0, 0)

    def total_episodes(self) -> int:
        return 0


class TestDynamicStartupTasks(unittest.TestCase):
    def test_dynamic_startup_task_is_usable_across_task_routes(self) -> None:
        task_id = "20260316T230251114130"
        data_path = "/tmp/datasets/unknown_grasp/20260316T230251114130"
        app = create_app(
            initial_scanners={task_id: _ScannerStub(data_path)},
            default_task=task_id,
        )

        with TestClient(app) as client:
            default_task = client.get("/api/default_task")
            self.assertEqual(default_task.status_code, 200)
            self.assertEqual(default_task.json(), {"task_id": task_id})

            tasks = client.get("/api/tasks")
            self.assertEqual(tasks.status_code, 200)
            self.assertIn(
                {"id": task_id, "data_path": data_path},
                tasks.json(),
            )

            status = client.get(f"/api/tasks/{task_id}/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(
                status.json(),
                {"done": 0, "total": 0, "ready": False, "episodes": 0},
            )

            scan = client.post(f"/api/tasks/{task_id}/scan")
            self.assertEqual(scan.status_code, 200)
            self.assertEqual(scan.json()["status"], "already_started")


class TestEpisodeScannerDiscovery(unittest.TestCase):
    def test_single_episode_root_is_discovered(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / VIDEO_FILENAME).touch()

            scanner = EpisodeScanner(root)
            scanner._bg_scan()

            self.assertTrue(scanner.ready)
            self.assertEqual(scanner.progress, (1, 1))
            self.assertEqual(scanner.total_episodes(), 1)
            self.assertEqual(scanner.episodes, [{"idx": 0, "folder": root.name}])
            self.assertEqual(scanner.get_episode_path(0), root)


class TestValuePredictionHelpers(unittest.TestCase):
    def test_normalize_value_server_url(self) -> None:
        self.assertEqual(normalize_value_server_url("localhost:8765"), "http://localhost:8765")
        self.assertEqual(
            normalize_value_server_url("https://example.com/path/"),
            "https://example.com/path",
        )
        self.assertIsNone(normalize_value_server_url(""))
        self.assertIsNone(normalize_value_server_url(None))

    @patch("third_party.overlay_viz.app.urllib_request.urlopen")
    def test_fetch_value_predictions_posts_expected_payload(self, mock_urlopen) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "frame_idx": [0, 4, 8],
                "absolute_value": [0.1, 0.4, 0.9],
                "absolute_advantage": [0.0, 0.3, 0.8],
                "mode": "1step",
                "cached": False,
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = fetch_value_predictions_from_server(
            "http://localhost:8765",
            "/tmp/episode_000",
            {"force": True, "relative_interval": 25},
        )

        self.assertEqual(result["mode"], "1step")
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:8765/infer")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "episode_path": "/tmp/episode_000",
                "relative_interval": 25,
                "force": True,
            },
        )

    def test_local_value_prediction_cache_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            episode_path = Path(tmp_dir)
            for filename in CAMERA_FILENAMES.values():
                (episode_path / filename).touch()

            server_health = {
                "mode": "1step",
                "ckpt_dir": "/tmp/checkpoints/30000",
                "model_cam_names": ["base", "left_wrist", "right_wrist"],
                "device": "cuda:0",
            }
            payload = build_value_prediction_cache_payload(
                {
                    "frame_idx": [0, 1, 2],
                    "absolute_value": [0.1, 0.2, 0.3],
                    "absolute_advantage": [0.0, 0.1, 0.2],
                    "mode": "1step",
                    "created_at": "2026-04-04T00:00:00Z",
                },
                episode_path=episode_path,
                task_id="pickup_cutter_d2",
                episode_idx=4,
                server_health=server_health,
            )

            cache_path = write_local_value_prediction_cache(episode_path, payload)
            loaded = read_local_value_prediction_cache(episode_path)

            self.assertEqual(cache_path.name, "value_predictions.npz")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["frame_idx"], [0, 1, 2])
            self.assertEqual(len(loaded["absolute_value"]), 3)
            self.assertAlmostEqual(loaded["absolute_value"][0], 0.1, places=6)
            self.assertAlmostEqual(loaded["absolute_value"][1], 0.2, places=6)
            self.assertAlmostEqual(loaded["absolute_value"][2], 0.3, places=6)
            self.assertEqual(loaded["episode_idx"], 4)
            self.assertEqual(loaded["task_id"], "pickup_cutter_d2")
            self.assertEqual(loaded["cache_format"], "npz")
            self.assertTrue(value_prediction_cache_matches_episode(loaded, episode_path))
            self.assertTrue(value_prediction_cache_matches_server(loaded, server_health))

    def test_value_prediction_cache_detects_stale_server_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            episode_path = Path(tmp_dir)
            for filename in CAMERA_FILENAMES.values():
                (episode_path / filename).touch()

            payload = build_value_prediction_cache_payload(
                {
                    "frame_idx": [0],
                    "absolute_value": [0.5],
                    "absolute_advantage": [0.0],
                    "mode": "1step",
                    "ckpt_dir": "/tmp/checkpoints/30000",
                },
                episode_path=episode_path,
                task_id="pickup_cutter_d2",
                episode_idx=0,
                server_health={
                    "mode": "1step",
                    "ckpt_dir": "/tmp/checkpoints/30000",
                    "model_cam_names": ["base", "left_wrist", "right_wrist"],
                },
            )

            self.assertFalse(
                value_prediction_cache_matches_server(
                    payload,
                    {
                        "mode": "2step",
                        "ckpt_dir": "/tmp/checkpoints/50000",
                        "model_cam_names": ["base", "left_wrist", "right_wrist"],
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
