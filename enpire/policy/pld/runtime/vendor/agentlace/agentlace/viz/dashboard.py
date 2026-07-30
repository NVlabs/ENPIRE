# SPDX-FileCopyrightText: Copyright (c) 2023 You Liang Tan
# SPDX-License-Identifier: MIT

"""
VizDashboard: drop-in replacement for TrainerServer that collects metrics
and serves a live web dashboard.

Usage:
    from agentlace.viz import VizDashboard

    server = VizDashboard(config, request_callback=my_cb, viz_port=8080)
    server.register_data_store("replay", replay_buffer)
    server.start(threaded=True)
    server.publish_network(params)
    # Open http://localhost:8080
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Dict, Optional, Set

import numpy as np

from agentlace.data.data_store import DataStoreBase
from agentlace.trainer import TrainerConfig, TrainerServer

from .web_server import VizWebServer

logger = logging.getLogger(__name__)


class VizDashboard:
    """Drop-in wrapper around TrainerServer that adds a web dashboard."""

    def __init__(
        self,
        config: TrainerConfig,
        data_callback=None,
        request_callback=None,
        log_level=logging.INFO,
        viz_port: int = 8585,
    ):
        self._lock = threading.Lock()
        self._viz_port = viz_port

        # Metric time series (capped ring buffers)
        self._max_history = 300
        self._timestamps: deque = deque(maxlen=self._max_history)
        self._buffer_sizes: Dict[str, deque] = {}
        self._episode_returns: deque = deque(maxlen=self._max_history)
        self._success_rates: deque = deque(maxlen=self._max_history)
        self._timer_stats: Dict[str, float] = {}
        self._network_publish_count: int = 0
        self._network_publish_log: deque = deque(maxlen=50)
        self._latest_actions: list = []
        self._data_insert_count: int = 0

        # Wrap user callbacks
        wrapped_data_cb = self._wrap_data_callback(data_callback)
        wrapped_req_cb = self._wrap_request_callback(request_callback)

        # Create the real TrainerServer
        self._server = TrainerServer(
            config,
            data_callback=wrapped_data_cb,
            request_callback=wrapped_req_cb,
            log_level=log_level,
        )

        # Start web server
        self._web_server = VizWebServer(port=viz_port, snapshot_fn=self._build_snapshot)
        self._web_server.start()
        logger.info(f"Viz dashboard at http://localhost:{viz_port}")

    # ------------------------------------------------------------------
    # Delegated TrainerServer methods
    # ------------------------------------------------------------------

    def register_data_store(self, name: str, data_store: DataStoreBase):
        with self._lock:
            self._buffer_sizes[name] = deque(maxlen=self._max_history)
        self._server.register_data_store(name, data_store)

    def data_store(self, name: str) -> Optional[DataStoreBase]:
        return self._server.data_store(name)

    def store_names(self) -> Set[str]:
        return self._server.store_names()

    def publish_network(self, payload: dict):
        with self._lock:
            self._network_publish_count += 1
            self._network_publish_log.append({"t": time.time(), "keys": list(payload.keys())})
        self._server.publish_network(payload)

    def start(self, threaded: bool = False):
        self._server.start(threaded=threaded)

    def stop(self):
        self._web_server.stop()
        self._server.stop()

    # ------------------------------------------------------------------
    # Callback wrappers
    # ------------------------------------------------------------------

    def _wrap_data_callback(self, user_cb):
        """Wrap user's data_callback to collect metrics on data insertions."""

        def _cb(store_name: str, payload: dict):
            batch = payload.get("data", [])
            with self._lock:
                self._data_insert_count += len(batch)
                # Try to extract image and actions from latest item
                if batch:
                    latest = batch[-1]
                    self._extract_obs(latest)

            if user_cb:
                return user_cb(store_name, payload)

        return _cb

    def _wrap_request_callback(self, user_cb):
        """Wrap user's request_callback to intercept stats payloads."""

        def _cb(type: str, payload: dict) -> dict:
            with self._lock:
                self._extract_stats(payload)
            if user_cb:
                return user_cb(type, payload)
            return {}

        return _cb

    # ------------------------------------------------------------------
    # Metric extraction helpers
    # ------------------------------------------------------------------

    def _extract_obs(self, transition: dict):
        """Pull actions from a transition dict (called under lock)."""
        actions = transition.get("actions")
        if actions is not None:
            if isinstance(actions, np.ndarray):
                self._latest_actions = actions.tolist()
            elif isinstance(actions, (list, tuple)):
                self._latest_actions = list(actions)

    def _extract_stats(self, payload: dict):
        """Pull episode return, success rate, timers from stats dict (under lock)."""
        # Nested format: {"environment": {"episode": {"return": ..., "success": ...}}}
        env = payload.get("environment", {})
        ep = env.get("episode", {})
        ret = ep.get("return")
        if ret is not None:
            self._episode_returns.append(float(ret))
        succ = ep.get("success")
        if succ is not None:
            self._success_rates.append(float(succ))

        # Flat format: {"episode_return": ..., "success_rate": ...}
        if ret is None:
            ret = payload.get("episode_return")
            if ret is not None:
                self._episode_returns.append(float(ret))
        if succ is None:
            succ = payload.get("success_rate")
            if succ is not None:
                self._success_rates.append(float(succ))

        # Timer stats
        timers = payload.get("timer", payload.get("timers", {}))
        if isinstance(timers, dict):
            self._timer_stats.update(timers)

    # ------------------------------------------------------------------
    # Snapshot builder
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> dict:
        """Build a JSON-serializable snapshot of all collected metrics."""
        with self._lock:
            # Sample current buffer sizes from registered data stores
            now = time.time()
            buf_sizes = {}
            for name in list(self._buffer_sizes.keys()):
                ds = self._server.data_store(name)
                if ds is not None:
                    size = len(ds)
                    self._buffer_sizes[name].append(size)
                    buf_sizes[name] = {
                        "current": size,
                        "history": list(self._buffer_sizes[name]),
                    }

            return {
                "timestamp": now,
                "buffer_sizes": buf_sizes,
                "episode_returns": list(self._episode_returns),
                "success_rates": list(self._success_rates),
                "timer_stats": dict(self._timer_stats),
                "network_publish_count": self._network_publish_count,
                "network_publish_log": list(self._network_publish_log),
                "latest_actions": self._latest_actions,
                "data_insert_count": self._data_insert_count,
            }

