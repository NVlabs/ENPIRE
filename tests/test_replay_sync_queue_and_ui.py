# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest


class TestQueuedSyncToInit(unittest.TestCase):
    def setUp(self) -> None:
        self.src = open("experimental/start_stop_play_policy.py").read()

    def test_wrapper_tracks_pending_sync_requests(self) -> None:
        self.assertIn("_pending_sync_to_init = False", self.src)

    def test_portal_sync_queues_request(self) -> None:
        idx = self.src.find("def _portal_sync_to_init")
        self.assertGreater(idx, 0)
        snippet = self.src[idx:idx + 700]
        self.assertIn("_pending_sync_to_init = True", snippet)
        self.assertIn("Queued sync_to_init", snippet)

    def test_get_action_processes_queued_sync_after_pause(self) -> None:
        self.assertIn(
            'if self._pending_sync_to_init and self.execution_state == "pause":',
            self.src,
        )
        idx = self.src.find('if self._pending_sync_to_init and self.execution_state == "pause":')
        snippet = self.src[idx:idx + 320]
        self.assertIn('self.enter_state("sync_to_init")', snippet)


class TestReplayUiFeedback(unittest.TestCase):
    def setUp(self) -> None:
        self.html = open("third_party/overlay_viz/static/index.html").read()

    def test_ui_has_visible_replay_notice(self) -> None:
        self.assertIn("replayNotice", self.html)
        self.assertIn("alert-error", self.html)

    def test_load_episode_checks_http_status_before_marking_loaded(self) -> None:
        idx = self.html.find("async loadEpisodeForReplay()")
        self.assertGreater(idx, 0)
        snippet = self.html[idx:idx + 700]
        self.assertIn("await this.requireReplayOk(res", snippet)
        self.assertIn("this.replayLoadedEpisode = this.viewerIdx", snippet)

    def test_sync_to_init_sets_user_visible_notice(self) -> None:
        idx = self.html.find("async replaySyncToInit()")
        self.assertGreater(idx, 0)
        snippet = self.html[idx:idx + 700]
        self.assertIn("await this.requireReplayOk(res", snippet)
        self.assertIn("setReplayNotice('Sync to Init requested.", snippet)

    def test_task_switch_clears_loaded_episode_state(self) -> None:
        idx = self.html.find("async onTaskSelected()")
        self.assertGreater(idx, 0)
        snippet = self.html[idx:idx + 500]
        self.assertIn("this.replayLoadedEpisode = null", snippet)


if __name__ == "__main__":
    unittest.main()
