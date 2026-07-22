# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote Deployment Environment - ZMQ REP SERVER for RL data collection.

Speaks the raw-style msgpack REQ/REP protocol that
forge.experimental.rl_interface.RLInterface uses (see
forge/tmux/realworld_rl/rl_gear.sh: --zmq_wire_format msgpack
--zmq_request_style raw --zmq_payload_mode payload).

Direction:
  RLInterface (robot client)  → step payload      → RemoteDeploymentEnv (us)
  RemoteDeploymentEnv         → {"action": dict}  → RLInterface

Wire protocol (raw msgpack):
  request {"__reset__": True[, "obs": payload]}  → response {}
  request payload {images, states, text, ...}    → response {"action": action_dict}
  exception during handling                       → response {"error": ..., "detail": ...}

action_dict matches the bimanual format expected by
forge/run_data_collection_for_rl.py (e.g. {joint_pos_action_left, ...} or
{left_ee_pos, left_ee_rot6d, ...}).

Gym API for the RL training loop:
  reset()       → obs, info        waits for robot's first step() call
  step(action)  → obs, r, d, t, i  sends action_dict to robot, waits for next obs

Threading model:
  ZMQ worker thread blocks in _handle_step_from_robot waiting for an action
  from the RL thread (via _action_queue). The RL thread blocks in env.step()
  waiting for the next obs from the ZMQ thread (via _obs_queue).
  They interleave in lockstep.
"""

from __future__ import annotations

import collections
import io
import logging
import queue
import threading
import time
from typing import Any

import gymnasium as gym
import msgpack
import numpy as np
import zmq
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Dict

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Key mapping constants
# ---------------------------------------------------------------------------

# Maps robot/payload state keys → env proprio_keys
_STATE_KEY_MAP: dict[str, str] = {
    # lecar-tbd XDOF format (from PROPRIO_KEY_MAP_xdof)
    "joint_pos_obs_left": "left_joint_pos",
    "gripper_pos_obs_left": "left_gripper_pos",
    "joint_pos_obs_right": "right_joint_pos",
    "gripper_pos_obs_right": "right_gripper_pos",
    # Direct mappings
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
    # EE pose format
    "ee_pos_obs_left": "left_ee_pos",
    "ee_quat_obs_left": "left_ee_quat",
    "ee_pos_obs_right": "right_ee_pos",
    "ee_quat_obs_right": "right_ee_quat",
    "left_ee_pos": "left_ee_pos",
    "left_ee_quat": "left_ee_quat",
    "right_ee_pos": "right_ee_pos",
    "right_ee_quat": "right_ee_quat",
    # Flat dual-arm EEF rot6d proprio:
    # [L xyz(3), L rot6d(6), L grip(1), R xyz(3), R rot6d(6), R grip(1)]
    "state_eef_rot6d": "state_eef_rot6d",
}

# Known proprioceptive key dimensions for Yam bimanual
_PROPRIO_KEY_DIMS: dict[str, int] = {
    "left_joint_pos": 6,
    "left_gripper_pos": 1,
    "right_joint_pos": 6,
    "right_gripper_pos": 1,
    "left_ee_pos": 3,
    "left_ee_quat": 4,
    "right_ee_pos": 3,
    "right_ee_quat": 4,
    "state_eef_rot6d": 20,
}

_STATE_EEF_ROT6D_COMPONENTS: dict[str, list[int]] = {
    "L_x": [0],
    "L_y": [1],
    "L_z": [2],
    "L_rot6d": [3, 4, 5, 6, 7, 8],
    "L_grip": [9],
    "R_x": [10],
    "R_y": [11],
    "R_z": [12],
    "R_rot6d": [13, 14, 15, 16, 17, 18],
    "R_grip": [19],
}

_PROPRIO_COMPONENT_MAPS: dict[str, dict[str, list[int]]] = {
    "state_eef_rot6d": _STATE_EEF_ROT6D_COMPONENTS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_quat(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _validate_action_repr(action_repr: str | None, action_dim: int) -> str:
    """Validate the flat action representation instead of inferring silently."""
    if action_repr is None or str(action_repr).strip() == "":
        raise ValueError(
            "remote_deployment requires explicit env.action_repr: "
            "'joint', 'delta_eef_quat', 'delta_eef_rot6d', or 'delta_eef_pos'."
        )
    repr_name = str(action_repr)
    expected_dims = {
        "joint": {7, 14},
        "delta_eef_quat": {8, 16},
        "delta_eef_rot6d": {10, 20},
        "delta_eef_pos": {3, 6},
    }
    if repr_name not in expected_dims:
        raise ValueError(
            f"Unsupported env.action_repr={repr_name!r}; expected one of "
            f"{sorted(expected_dims)}"
        )
    if int(action_dim) not in expected_dims[repr_name]:
        raise ValueError(
            f"env.action_repr={repr_name!r} is incompatible with "
            f"action_dim={action_dim}; expected one of {sorted(expected_dims[repr_name])}"
        )
    return repr_name


def _resolve_proprio_filter(proprio_filter: Any) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Parse proprio filter config into key -> (mode, excluded raw indices)."""
    if not proprio_filter:
        return {}
    if not hasattr(proprio_filter, "items"):
        raise TypeError("proprio_filter must be a mapping from proprio key to config")

    resolved: dict[str, tuple[str, tuple[int, ...]]] = {}
    for key, cfg in proprio_filter.items():
        if key not in _PROPRIO_COMPONENT_MAPS:
            raise ValueError(
                f"Unsupported proprio_filter key {key!r}; expected one of "
                f"{sorted(_PROPRIO_COMPONENT_MAPS)}"
            )
        if cfg is None:
            continue
        if not hasattr(cfg, "get"):
            raise TypeError(
                f"proprio_filter[{key!r}] must be a mapping with mode/include or "
                "mode/exclude fields"
            )

        mode = str(cfg.get("mode", "zero"))
        if mode not in {"zero", "drop"}:
            raise ValueError(
                f"proprio_filter[{key!r}].mode={mode!r}; expected 'zero' or 'drop'"
            )

        component_map = _PROPRIO_COMPONENT_MAPS[key]
        has_include = "include" in cfg and cfg.get("include") is not None
        has_exclude = "exclude" in cfg and cfg.get("exclude") is not None
        if has_include and has_exclude:
            raise ValueError(
                f"proprio_filter[{key!r}] must specify only one of include or exclude"
            )

        def component_indices(components: Any, *, field: str) -> list[int]:
            if isinstance(components, str):
                components = [components]
            indices: list[int] = []
            for component in components or []:
                component_name = str(component)
                if component_name not in component_map:
                    raise ValueError(
                        f"Unknown proprio_filter component {component_name!r} for "
                        f"{key!r}.{field}; expected one of {sorted(component_map)}"
                    )
                indices.extend(component_map[component_name])
            return indices

        if has_include:
            included_indices = set(component_indices(cfg.get("include"), field="include"))
            all_indices = {idx for indices in component_map.values() for idx in indices}
            excluded_indices = sorted(all_indices - included_indices)
        else:
            excluded_indices = component_indices(cfg.get("exclude", []), field="exclude")
        resolved[key] = (mode, tuple(sorted(set(excluded_indices))))
    return resolved


def _make_arrays_contiguous(obj: Any, memo: dict | None = None) -> Any:
    """Recursively ensure numpy arrays are contiguous for ZMQ serialization."""
    if memo is None:
        memo = {}
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    if isinstance(obj, dict):
        new_obj: dict = {}
        memo[obj_id] = new_obj
        for key, value in obj.items():
            new_obj[key] = _make_arrays_contiguous(value, memo)
        return new_obj
    if isinstance(obj, (list, tuple)):
        new_list: list = []
        memo[obj_id] = new_list
        for item in obj:
            new_list.append(_make_arrays_contiguous(item, memo))
        return new_list
    return obj


def _normalize_camera_name(key: str) -> str:
    """Reduce a camera key to a canonical prefix for fuzzy matching.

    Examples:
      "top_camera-images-rgb"  -> "top_camera"
      "top_camera_image"       -> "top_camera"
      "observation.images.left_camera-images-rgb_320_240" -> "left_camera"
    """
    key = key.lower()
    # Strip "observation.images." prefix
    for pfx in ("observation.images.", "observation/images/"):
        if key.startswith(pfx):
            key = key[len(pfx) :]
    # Strip resolution suffix like "_320_240"
    import re

    key = re.sub(r"_\d+_\d+$", "", key)
    # Strip image/rgb suffixes
    for suffix in (
        "-images-rgb",
        "_images_rgb",
        "-image-rgb",
        "_image_rgb",
        "-images",
        "_images",
        "-image",
        "_image",
        "-rgb",
        "_rgb",
    ):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key.replace("-", "_").strip("_")


def _match_image(images: dict[str, Any], target_key: str) -> Any | None:
    """Find the image in payload images dict that best matches target_key."""
    norm_target = _normalize_camera_name(target_key)
    # Exact canonical match
    for k, v in images.items():
        if _normalize_camera_name(k) == norm_target:
            return v
    # Prefix match: "top_camera" matches "top_camera-images-rgb"
    for k, v in images.items():
        if _normalize_camera_name(k).startswith(norm_target):
            return v
    return None


def _zmq_msgpack_default(obj: Any) -> Any:
    """Serialize ndarrays to msgpack — matches rl_interface._msgpack_default."""
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, np.ascontiguousarray(obj), allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    raise TypeError(f"Cannot serialize {type(obj).__name__} over ZMQ msgpack")


def _zmq_msgpack_object_hook(obj: dict[str, Any]) -> Any:
    """Inverse of _zmq_msgpack_default — matches rl_interface._msgpack_object_hook."""
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class RemoteDeploymentEnv(gym.Env[ObsType, ActType]):
    """Gymnasium env that acts as a ZMQ REP SERVER for forge's RLInterface client.

    The remote robot (forge/run_data_collection_for_rl.py with
    --server_address=<host>:<port>) connects as the ZMQ REQ CLIENT and drives the
    timing. This env bridges the robot-driven async REQ/REP loop to the RL actor's
    synchronous Gym step/reset interface.

    Usage:
      - Start this env (it binds the ZMQ REP server on `host:port`)
      - Launch the forge robot side with rl_gear.sh / rl.sh pointing here
      - Call env.reset() to wait for the robot's first obs; call env.step(action)
        to return an action and receive the next obs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str | None = "0.0.0.0",
        port: int = 8964,
        image_keys: list[str] | None = None,
        proprio_keys: list[str] | None = None,
        action_dim: int = 14,
        action_exec_horizon: int = 20,
        image_size: tuple[int, int] = (480, 640),
        instruction: str = "",
        step_timeout: float | None = None,
        control_mode: str = "both",
        action_repr: str | None = None,
        proprio_filter: dict[str, Any] | None = None,
    ):
        """
        Args:
            host: ZMQ server bind host. Use the actor machine LAN hostname/IP
                for multi-machine deployments, or "0.0.0.0" to bind all IPv4
                interfaces.
            port: ZMQ server port (must match lecar-tbd's --server-address port).
            image_keys: Camera obs keys, e.g. ["top_camera_image", "left_camera_image"].
                Used for observation_space and obs parsing. Should match the cameras
                that lecar-tbd sends.
            proprio_keys: Proprioceptive keys. Dimension is inferred from _PROPRIO_KEY_DIMS.
                Must match what lecar-tbd sends (after key remapping).
            action_dim: Flat action dimension. Joint actions are 14-D bimanual
                or 7-D single-arm; delta-EEF actions are 16/8-D quaternion,
                20/10-D rot6d, or 6/3-D position-only.
            action_exec_horizon: Number of action steps per policy call. Should match
                lecar-tbd's --action-horizon. Actions are tiled to fill the horizon.
            image_size: (H, W) placeholder image size for observation_space before
                SERLObsWrapper applies the configured RL image resolution.
            instruction: Language instruction returned by get_language_instruction().
            step_timeout: Seconds to wait for action/obs before raising TimeoutError.
                None = block indefinitely.
            control_mode: "left", "right", or "both". Determines how the flat action
                vector is unpacked into per-arm action dicts.
            proprio_filter: Optional mapping that masks configured proprio components.
        """
        super().__init__()
        self._host = host or "0.0.0.0"
        self._port = port
        self._server_address = f"{self._host}:{self._port}"
        self._image_keys = list(image_keys or ["top_camera_image", "left_camera_image"])
        self._proprio_keys = list(
            proprio_keys
            or ["left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"]
        )
        self._action_dim = action_dim
        self._action_exec_horizon = action_exec_horizon
        self._image_size = image_size
        self._instruction = instruction
        self._step_timeout = None if step_timeout is None else float(step_timeout)
        self._control_mode = control_mode
        self._action_repr = _validate_action_repr(action_repr, self._action_dim)
        self._proprio_filter = _resolve_proprio_filter(proprio_filter)

        # Thread-safe queues for obs/action handoff between ZMQ and RL threads
        self._obs_queue: queue.Queue = queue.Queue()
        self._action_queue: queue.Queue = queue.Queue()
        self._last_obs: dict | None = None
        self._step_count = 0
        self._last_robot_step_t: float | None = None
        self._last_robot_reset_t: float | None = None
        self._last_action_sent_t: float | None = None
        self._last_obs_to_action_s: float | None = None
        self._total_obs_to_action_s = 0.0
        self._obs_to_action_count = 0

        # Per-step latency profiling.
        # The ZMQ thread sets _last_obs_queued_t before each _obs_queue.put so
        # _real_step can compute dequeue_lag (time obs waited before RL picked it up).
        # _real_step sets the _last_* fields; the ZMQ thread reads them when building
        # the profile entry for that same step.  Access is safe in CPython because the
        # two threads are lockstep-synchronised through the queues.
        self._last_obs_queued_t: float | None = None
        self._last_obs_parse_ms: float | None = None
        self._last_dequeue_lag_ms: float | None = None
        self._last_format_action_ms: float | None = None
        self._last_next_obs_wait_ms: float | None = None
        self._profile_history: collections.deque = collections.deque(maxlen=2000)

        # Build observation/action spaces from config before robot connects
        self._build_spaces_from_config()

        # Start ZMQ REP server in a worker thread. Matches the raw-style msgpack
        # REQ client in forge.experimental.rl_interface.RLInterface.
        self._zmq_ctx = zmq.Context.instance()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REP)
        self._zmq_sock.bind(f"tcp://{self._host}:{self._port}")
        self._zmq_running = True
        self._zmq_thread = threading.Thread(
            target=self._zmq_serve_loop,
            name="RemoteDeploymentEnv-zmq",
            daemon=True,
        )
        self._zmq_thread.start()

        print(
            f"[RemoteDeploymentEnv] ZMQ REP server bound at "
            f"tcp://{self._host}:{self._port}. Waiting for robot to connect..."
        )

    # ------------------------------------------------------------------
    # ZMQ RPC handlers (called in ZMQ's worker thread)
    # ------------------------------------------------------------------

    def _handle_step_from_robot(self, payload: dict) -> dict:
        """Robot sends {images, states, ...} → we queue obs and wait for RL action.

        This call BLOCKS until the RL actor calls env.step(action) and puts an
        action in _action_queue. This is intentional: ZMQ's worker thread stays
        alive while the RL thread processes the obs and computes the next action.
        """
        recv_t = time.monotonic()
        obs = self._parse_payload(payload)
        self._last_robot_step_t = recv_t

        # Record timestamp before put so _real_step can compute dequeue_lag for this obs.
        obs_queued_t = time.monotonic()
        self._last_obs_queued_t = obs_queued_t
        self._obs_queue.put(obs)

        try:
            action_dict = self._action_queue.get(timeout=self._step_timeout)
        except queue.Empty:
            raise TimeoutError(
                f"[RemoteDeploymentEnv] Timed out after {self._step_timeout}s "
                "waiting for RL actor to call env.step(action)."
            )
        sent_t = time.monotonic()
        self._last_action_sent_t = sent_t
        obs_to_action_s = sent_t - recv_t
        self._last_obs_to_action_s = obs_to_action_s
        self._total_obs_to_action_s += obs_to_action_s
        self._obs_to_action_count += 1

        obs_parse_ms = (obs_queued_t - recv_t) * 1000
        self._last_obs_parse_ms = obs_parse_ms
        # _last_format_action_ms / _last_dequeue_lag_ms / _last_next_obs_wait_ms were
        # written by _real_step for this step before it put action_dict in the queue.
        self._profile_history.append(
            {
                "obs_to_action_ms": obs_to_action_s * 1000,
                "obs_parse_ms": obs_parse_ms,
                "rl_wait_ms": (sent_t - obs_queued_t) * 1000,
                "format_action_ms": self._last_format_action_ms,
                "dequeue_lag_ms": self._last_dequeue_lag_ms,
                "next_obs_wait_ms": self._last_next_obs_wait_ms,
            }
        )
        return _make_arrays_contiguous(action_dict)

    def _handle_reset_from_robot(self) -> None:
        """Robot signals episode boundary. Puts a sentinel None in obs_queue."""
        self._last_robot_reset_t = time.monotonic()
        self._obs_queue.put(None)
        return None

    def _handle_health_check(self) -> bool:
        return True

    def _handle_get_config(self) -> dict[str, Any]:
        """Return ZMQ server metadata in the same spirit as serve_policy.py."""
        return {
            "server": "pld_lite_remote_deployment_env",
            "host": self._host,
            "port": self._port,
            "server_address": self._server_address,
            "image_keys": list(self._image_keys),
            "proprio_keys": list(self._proprio_keys),
            "action_dim": int(self._action_dim),
            "action_exec_horizon": int(self._action_exec_horizon),
            "control_mode": self._control_mode,
            "action_repr": self._action_repr,
            "instruction": self._instruction or None,
        }

    def _handle_get_info(self) -> dict[str, Any]:
        """Return policy metadata expected by Forge run setup.

        Forge's YAM control loop queries policy servers with get_info() before
        constructing its recording directory. PLD does not serve a checkpoint in
        this process, so expose stable descriptive metadata instead.
        """
        mode = self._control_mode or "both"
        repr_name = self._action_repr or "unknown"
        return {
            "server_address": self._server_address,
            "ckpt_path": None,
            "ckpt_name": None,
            "policy_name": f"pld_lite_{mode}_{repr_name}",
            "step": 0,
        }

    # ------------------------------------------------------------------
    # ZMQ REP server loop
    # ------------------------------------------------------------------

    def _zmq_serve_loop(self) -> None:
        """Read msgpack requests on the REP socket and dispatch to handlers."""
        while self._zmq_running:
            try:
                msg = self._zmq_sock.recv()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError:
                if not self._zmq_running:
                    return
                continue
            try:
                request = msgpack.unpackb(msg, object_hook=_zmq_msgpack_object_hook, raw=False)
                response = self._dispatch_zmq_request(request)
            except Exception as exc:
                logging.exception("[RemoteDeploymentEnv] ZMQ request failed")
                response = {"error": type(exc).__name__, "detail": str(exc)}
            try:
                self._zmq_sock.send(
                    msgpack.packb(response, default=_zmq_msgpack_default, use_bin_type=True)
                )
            except zmq.ZMQError:
                if not self._zmq_running:
                    return

    def _dispatch_zmq_request(self, request: Any) -> Any:
        """Route a raw-style request to the reset or step handler.

        rl_interface.py raw-style REQ payloads:
          {"__reset__": True[, "obs": payload]}  → reset, return {}
          step payload {images, states, ...}     → step, return {"action": action_dict}
        """
        if not isinstance(request, dict):
            raise TypeError(
                f"RemoteDeploymentEnv expected dict request, got {type(request).__name__}"
            )
        if request.get("__reset__"):
            self._handle_reset_from_robot()
            return {}
        action_dict = self._handle_step_from_robot(request)
        return {"action": action_dict}

    def get_remote_status(self) -> dict[str, Any]:
        """Return live ZMQ/queue status for actor-side dashboards."""
        avg_obs_to_action_s = (
            self._total_obs_to_action_s / self._obs_to_action_count
            if self._obs_to_action_count
            else None
        )
        return {
            "host": self._host,
            "port": self._port,
            "server_address": self._server_address,
            "control_mode": self._control_mode,
            "action_repr": self._action_repr,
            "action_dim": int(self._action_dim),
            "action_exec_horizon": int(self._action_exec_horizon),
            "obs_queue_size": self._obs_queue.qsize(),
            "action_queue_size": self._action_queue.qsize(),
            "step_count": int(self._step_count),
            "last_obs_seen": self._last_obs is not None,
            "last_robot_step_t": self._last_robot_step_t,
            "last_robot_reset_t": self._last_robot_reset_t,
            "last_action_sent_t": self._last_action_sent_t,
            "last_obs_to_action_s": self._last_obs_to_action_s,
            "avg_obs_to_action_s": avg_obs_to_action_s,
            "obs_to_action_count": int(self._obs_to_action_count),
            # Per-step profiling breakdowns (ms)
            "last_obs_parse_ms": self._last_obs_parse_ms,
            "last_dequeue_lag_ms": self._last_dequeue_lag_ms,
            "last_format_action_ms": self._last_format_action_ms,
            "last_next_obs_wait_ms": self._last_next_obs_wait_ms,
        }

    def get_profile_history(self) -> list[dict]:
        """Return a snapshot of recent per-step ZMQ timing records."""
        return list(self._profile_history)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def _real_reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[ObsType, dict]:
        """Wait for the robot to start a new episode (its first step() call).

        Blocks until a real obs arrives. Episode-boundary sentinels from
        a {"__reset__": True} request are skipped below.
        """
        self._step_count = 0

        # Block until fresh obs arrives
        obs = None
        while obs is None:
            try:
                item = self._obs_queue.get(timeout=self._step_timeout)
            except queue.Empty:
                raise TimeoutError(
                    "[RemoteDeploymentEnv] Timed out waiting for robot to connect "
                    "and send first observation. "
                    "Make sure lecar-tbd is running with --server-address pointing here."
                )
            # Skip episode-boundary sentinels from a previous episode
            if item is not None:
                obs = item

        self._last_obs = obs
        self._update_spaces_from_obs(obs)
        return obs, {}

    def _real_step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """Send action to robot (completing the pending RPC), wait for next obs.

        action: flat numpy array [action_dim], already action-scaled.
        Returns: (obs, reward, terminated, truncated, info)
          - reward is always 0.0 (no reward signal from robot)
          - terminated/truncated True when robot sends {"__reset__": True}
        """
        t_format_start = time.monotonic()
        action_dict = self._format_action_dict(np.asarray(action))
        t_formatted = time.monotonic()

        # Set format time BEFORE queuing so ZMQ can read it when building the
        # profile entry for this step (ZMQ reads _last_format_action_ms after
        # _action_queue.get() returns, i.e. after this put() unblocks it).
        self._last_format_action_ms = (t_formatted - t_format_start) * 1000

        # Snapshot the current obs_queued_t so we can detect when ZMQ updates it
        # for the NEXT obs (used to compute dequeue_lag for that next obs).
        obs_queued_snap = self._last_obs_queued_t

        self._action_queue.put(action_dict)
        t_action_queued = time.monotonic()

        # Wait for next obs
        try:
            item = self._obs_queue.get(timeout=self._step_timeout)
        except queue.Empty:
            raise TimeoutError(
                "[RemoteDeploymentEnv] Timed out waiting for next obs from robot."
            )

        t_obs_got = time.monotonic()
        # next_obs_wait_ms = robot round-trip time (not part of obs→action latency)
        self._last_next_obs_wait_ms = (t_obs_got - t_action_queued) * 1000

        # Compute dequeue_lag for the obs we just received.
        # ZMQ sets _last_obs_queued_t BEFORE putting the new obs in the queue,
        # so by the time _obs_queue.get() returns the new value is already written.
        new_queued_t = self._last_obs_queued_t
        if new_queued_t is not None and (
            obs_queued_snap is None or new_queued_t > obs_queued_snap
        ):
            self._last_dequeue_lag_ms = (t_obs_got - new_queued_t) * 1000
        else:
            self._last_dequeue_lag_ms = None

        if item is None:
            # Robot sent {"__reset__": True} → episode boundary
            # Return last obs with truncated=True
            obs = self._last_obs if self._last_obs is not None else self._zero_obs()
            return obs, 0.0, False, True, {}

        obs = item
        self._last_obs = obs
        self._step_count += 1
        return obs, 0.0, False, False, {}

    def get_language_instruction(self) -> str:
        """Return task instruction for base agents and logging."""
        return self._instruction

    def close(self) -> None:
        # _make_fake skips ZMQ setup, so guard on attribute presence.
        if getattr(self, "_zmq_sock", None) is None:
            return
        self._zmq_running = False
        try:
            self._zmq_sock.close(linger=0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def _make_fake(
        cls,
        image_keys: list[str],
        proprio_keys: list[str],
        action_dim: int = 14,
        action_exec_horizon: int = 20,
        image_size: tuple[int, int] = (480, 640),
        instruction: str = "",
        control_mode: str = "both",
        action_repr: str | None = None,
        host: str | None = "0.0.0.0",
        proprio_filter: dict[str, Any] | None = None,
    ) -> "RemoteDeploymentEnv":
        """Create a placeholder env with correct spaces but no ZMQ server.

        Used by the LEARNER which needs observation/action spaces to build the
        SAC agent architecture but never actually steps through the environment.
        """
        obj = object.__new__(cls)
        gym.Env.__init__(obj)
        obj._host = host or "0.0.0.0"
        obj._port = None
        obj._server_address = f"{obj._host}:fake"
        obj._image_keys = image_keys
        obj._proprio_keys = proprio_keys
        obj._action_dim = action_dim
        obj._action_exec_horizon = action_exec_horizon
        obj._image_size = image_size
        obj._instruction = instruction
        obj._step_timeout = None
        obj._control_mode = control_mode
        obj._action_repr = _validate_action_repr(action_repr, obj._action_dim)
        obj._proprio_filter = _resolve_proprio_filter(proprio_filter)
        obj._obs_queue = queue.Queue()
        obj._action_queue = queue.Queue()
        obj._last_obs = None
        obj._step_count = 0
        obj._last_robot_step_t = None
        obj._last_robot_reset_t = None
        obj._last_action_sent_t = None
        obj._last_obs_to_action_s = None
        obj._total_obs_to_action_s = 0.0
        obj._obs_to_action_count = 0
        obj._last_obs_queued_t = None
        obj._last_obs_parse_ms = None
        obj._last_dequeue_lag_ms = None
        obj._last_format_action_ms = None
        obj._last_next_obs_wait_ms = None
        obj._profile_history: collections.deque = collections.deque(maxlen=2000)
        obj._build_spaces_from_config()
        print(
            f"[RemoteDeploymentEnv] Fake env created for learner (control_mode={control_mode})."
        )
        return obj

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict]:
        if self._port is None:
            return self._zero_obs(), {}
        return self._real_reset(seed=seed, options=options)

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:  # type: ignore[override]
        if self._port is None:
            return self._zero_obs(), 0.0, False, True, {}
        return self._real_step(action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_spaces_from_config(self) -> None:
        """Build placeholder observation/action spaces from config (before robot connects)."""
        H, W = self._image_size
        spaces: dict = {}
        for key in self._image_keys:
            spaces[key] = Box(0, 255, (H, W, 3), np.uint8)
        for key in self._proprio_keys:
            dim = self._effective_proprio_dim(key)
            spaces[key] = Box(-np.inf, np.inf, (dim,), np.float32)
        self.observation_space = Dict(spaces)
        self.action_space = Box(-1.0, 1.0, (self._action_dim,), np.float32)

    def _effective_proprio_dim(self, key: str) -> int:
        """Return the observation dim after any configured proprio filter."""
        base_dim = _PROPRIO_KEY_DIMS.get(key, 1)
        filter_spec = self._proprio_filter.get(key)
        if filter_spec is None:
            return base_dim
        mode, excluded_indices = filter_spec
        if mode == "drop":
            return base_dim - len(excluded_indices)
        return base_dim

    def _apply_proprio_filter(self, key: str, arr: np.ndarray) -> np.ndarray:
        """Apply configured zero/drop filtering to a raw proprio vector."""
        values = np.asarray(arr, dtype=np.float32).reshape(-1)
        filter_spec = self._proprio_filter.get(key)
        if filter_spec is None:
            return values

        mode, excluded_indices = filter_spec
        if not excluded_indices:
            return values
        if mode == "zero":
            filtered = values.copy()
            filtered[list(excluded_indices)] = 0.0
            return filtered
        return np.delete(values, list(excluded_indices)).astype(np.float32, copy=False)

    def _filtered_proprio_index(self, key: str, raw_idx: int) -> int | None:
        """Map a raw proprio index to its filtered index, or None if dropped."""
        filter_spec = self._proprio_filter.get(key)
        if filter_spec is None:
            return raw_idx

        mode, excluded_indices = filter_spec
        if raw_idx in excluded_indices:
            return None
        if mode == "drop":
            return raw_idx - sum(idx < raw_idx for idx in excluded_indices)
        return raw_idx

    def _update_spaces_from_obs(self, obs: dict) -> None:
        """Optionally update observation_space if real obs shapes differ from config."""
        try:
            spaces: dict = {}
            for key, value in obs.items():
                arr = np.asarray(value)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    h, w, c = arr.shape
                    spaces[key] = Box(0, 255, (h, w, c), np.uint8)
                else:
                    dim = int(arr.reshape(-1).size)
                    spaces[key] = Box(-np.inf, np.inf, (dim,), np.float32)
            if all(k in spaces for k in self._image_keys + self._proprio_keys):
                self.observation_space = Dict(spaces)
        except Exception:
            pass  # Keep existing spaces on error

    def _parse_payload(self, payload: dict) -> dict[str, Any]:
        """Convert robot step payload to SERL obs dict with configured keys."""
        images = payload.get("images", {}) or {}
        states = payload.get("states", {}) or {}
        obs: dict[str, Any] = {}

        # --- Images ---
        for key in self._image_keys:
            value = _match_image(images, key)
            if value is not None:
                arr = np.asarray(value)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                obs[key] = arr

        # --- States ---
        if isinstance(states, dict):
            for robot_key, robot_val in states.items():
                env_key = _STATE_KEY_MAP.get(robot_key, robot_key)
                if env_key in self._proprio_keys:
                    obs[env_key] = np.asarray(robot_val, dtype=np.float32).reshape(-1)
        elif isinstance(states, np.ndarray):
            # Flat state vector – split by proprio_key dims
            flat = np.asarray(states).reshape(-1).astype(np.float32)
            offset = 0
            for key in self._proprio_keys:
                dim = _PROPRIO_KEY_DIMS.get(key, 1)
                obs[key] = flat[offset : offset + dim]
                offset += dim

        missing_proprio = [key for key in self._proprio_keys if key not in obs]
        if missing_proprio:
            raise KeyError(
                "RemoteDeploymentEnv missing required proprio keys "
                f"{missing_proprio}; configured proprio_keys={self._proprio_keys}. "
                "For remote_yam rot6d PLD, Forge must send states['state_eef_rot6d']."
            )
        for key in self._proprio_keys:
            expected_dim = _PROPRIO_KEY_DIMS.get(key)
            if (
                expected_dim is not None
                and np.asarray(obs[key]).reshape(-1).size != expected_dim
            ):
                raise ValueError(
                    f"RemoteDeploymentEnv proprio key {key!r} has shape "
                    f"{np.asarray(obs[key]).shape}; expected flat dim {expected_dim}"
                )
            obs[key] = self._apply_proprio_filter(key, obs[key])

        return obs

    def _format_action_dict(self, action: np.ndarray) -> dict:
        """Convert flat action [action_dim] to action_dict chunk for the robot.

        Output format matches serve_policy._split_actions:
          joint_pos_action_left:     [H, 6]
          gripper_pos_action_left:   [H, 1]
          joint_pos_action_right:    [H, 6]
          gripper_pos_action_right:  [H, 1]

        For single-arm control_mode, only the active arm's motion keys are
        included; the robot holds the other arm.
        """
        ac = action.flatten().astype(np.float32)
        H = self._action_exec_horizon

        if self._action_repr == "delta_eef_pos":
            identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            closed_grip = np.array([0.0], dtype=np.float32)
            if self._control_mode in ("left", "right"):
                if ac.shape[0] != 3:
                    raise ValueError(
                        f"single-arm delta_eef_pos expects 3-D action, got {ac.shape}"
                    )
                return self._format_single_arm_delta_eef_action(
                    side=self._control_mode,
                    pos=ac[0:3],
                    quat_xyzw=identity_quat,
                    grip=closed_grip,
                    horizon=H,
                )
            if ac.shape[0] != 6:
                raise ValueError(f"bimanual delta_eef_pos expects 6-D action, got {ac.shape}")
            return {
                "left_ee_pos": np.tile(ac[0:3][None], (H, 1)),
                "left_ee_quat_xyzw": np.tile(identity_quat[None], (H, 1)),
                "left_gripper_pos": np.tile(closed_grip[None], (H, 1)),
                "right_ee_pos": np.tile(ac[3:6][None], (H, 1)),
                "right_ee_quat_xyzw": np.tile(identity_quat[None], (H, 1)),
                "right_gripper_pos": np.tile(closed_grip[None], (H, 1)),
            }

        if self._action_repr == "delta_eef_rot6d":
            # Send 6-D Zhou rot6d directly to Forge under {side}_ee_rot6d.
            # Forge's delta_ee_pose path now consumes rot6d natively, so we no
            # longer collapse to a 4-D quaternion here.
            if self._control_mode in ("left", "right"):
                if ac.shape[0] != 10:
                    raise ValueError(
                        f"single-arm delta_eef_rot6d expects 10-D action, got {ac.shape}"
                    )
                return self._format_single_arm_delta_eef_rot6d_action(
                    side=self._control_mode,
                    pos=ac[0:3],
                    rot6d=ac[3:9],
                    grip=ac[9:10],
                    horizon=H,
                )
            if ac.shape[0] != 20:
                raise ValueError(
                    f"bimanual delta_eef_rot6d expects 20-D action, got {ac.shape}"
                )
            return {
                "left_ee_pos": np.tile(ac[0:3][None], (H, 1)),
                "left_ee_rot6d": np.tile(ac[3:9][None], (H, 1)),
                "left_gripper_pos": np.tile(ac[9:10][None], (H, 1)),
                "right_ee_pos": np.tile(ac[10:13][None], (H, 1)),
                "right_ee_rot6d": np.tile(ac[13:19][None], (H, 1)),
                "right_gripper_pos": np.tile(ac[19:20][None], (H, 1)),
            }

        if self._action_repr == "delta_eef_quat":
            if self._control_mode in ("left", "right"):
                if ac.shape[0] != 8:
                    raise ValueError(
                        f"single-arm delta_eef_quat expects 8-D action, got {ac.shape}"
                    )
                return self._format_single_arm_delta_eef_action(
                    side=self._control_mode,
                    pos=ac[0:3],
                    quat_xyzw=_normalise_quat(ac[3:7]),
                    grip=ac[7:8],
                    horizon=H,
                )
            if ac.shape[0] != 16:
                raise ValueError(
                    f"bimanual delta_eef_quat expects 16-D action, got {ac.shape}"
                )
            return {
                "left_ee_pos": np.tile(ac[0:3][None], (H, 1)),
                "left_ee_quat_xyzw": np.tile(_normalise_quat(ac[3:7])[None], (H, 1)),
                "left_gripper_pos": np.tile(ac[7:8][None], (H, 1)),
                "right_ee_pos": np.tile(ac[8:11][None], (H, 1)),
                "right_ee_quat_xyzw": np.tile(_normalise_quat(ac[11:15])[None], (H, 1)),
                "right_gripper_pos": np.tile(ac[15:16][None], (H, 1)),
            }

        if self._action_repr != "joint":
            raise ValueError(
                f"env.action_repr={self._action_repr!r} is not implemented by "
                "this action formatter"
            )

        if self._control_mode == "left":
            # 7-dim: [joint(6), grip(1)] → left arm only
            return {
                "joint_pos_action_left": np.tile(ac[0:6][None], (H, 1)),
                "gripper_pos_action_left": np.tile(ac[6:7][None], (H, 1)),
            }
        elif self._control_mode == "right":
            # 7-dim: [joint(6), grip(1)] → right arm only
            return {
                "joint_pos_action_right": np.tile(ac[0:6][None], (H, 1)),
                "gripper_pos_action_right": np.tile(ac[6:7][None], (H, 1)),
            }
        else:
            # 14-dim bimanual: [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
            return {
                "joint_pos_action_left": np.tile(ac[0:6][None], (H, 1)),
                "gripper_pos_action_left": np.tile(ac[6:7][None], (H, 1)),
                "joint_pos_action_right": np.tile(ac[7:13][None], (H, 1)),
                "gripper_pos_action_right": np.tile(ac[13:14][None], (H, 1)),
            }

    def _format_single_arm_delta_eef_action(
        self,
        *,
        side: str,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        grip: np.ndarray,
        horizon: int,
    ) -> dict:
        """Format a single-arm delta-EE action and hold the inactive gripper.

        Use explicit zero/identity delta for the inactive arm instead of
        omitting its EE keys. Forge/YamRealEnv will hold the inactive side in
        delta-EE mode, and RecordEpisodeWrapper will still save a full
        bimanual delta-EEF action file that the PLD disk ingestor can slice
        back down to the configured single-arm action space.
        """
        inactive = "left" if side == "right" else "right"
        action = {
            f"{side}_ee_pos": np.tile(np.asarray(pos, dtype=np.float32)[None], (horizon, 1)),
            f"{side}_ee_quat_xyzw": np.tile(
                np.asarray(quat_xyzw, dtype=np.float32)[None], (horizon, 1)
            ),
            f"{side}_gripper_pos": np.tile(
                np.asarray(grip, dtype=np.float32).reshape(1, 1), (horizon, 1)
            ),
        }
        inactive_grip = self._inactive_gripper_value(inactive)
        action[f"{inactive}_ee_pos"] = np.zeros((horizon, 3), dtype=np.float32)
        action[f"{inactive}_ee_quat_xyzw"] = np.tile(
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (horizon, 1)
        )
        action[f"{inactive}_gripper_pos"] = np.tile(inactive_grip[None], (horizon, 1))
        return action

    def _format_single_arm_delta_eef_rot6d_action(
        self,
        *,
        side: str,
        pos: np.ndarray,
        rot6d: np.ndarray,
        grip: np.ndarray,
        horizon: int,
    ) -> dict:
        """Format a single-arm delta-EE action and hold the inactive gripper.

        Use explicit zero/identity delta for the inactive arm instead of
        omitting its EE keys. Forge/YamRealEnv will hold the inactive side in
        delta-EE mode, and RecordEpisodeWrapper will still save a full
        bimanual delta-EEF action file that the PLD disk ingestor can slice
        back down to the configured single-arm action space.
        """
        inactive = "left" if side == "right" else "right"
        action = {
            f"{side}_ee_pos": np.tile(np.asarray(pos, dtype=np.float32)[None], (horizon, 1)),
            f"{side}_ee_rot6d": np.tile(
                np.asarray(rot6d, dtype=np.float32)[None], (horizon, 1)
            ),
            f"{side}_gripper_pos": np.tile(
                np.asarray(grip, dtype=np.float32).reshape(1, 1), (horizon, 1)
            ),
        }
        inactive_grip = self._inactive_gripper_value(inactive)
        action[f"{inactive}_ee_pos"] = np.zeros((horizon, 3), dtype=np.float32)
        action[f"{inactive}_ee_rot6d"] = np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32), (horizon, 1)
        )
        action[f"{inactive}_gripper_pos"] = np.tile(inactive_grip[None], (horizon, 1))
        return action

    def _inactive_gripper_value(self, side: str) -> np.ndarray:
        if self._last_obs is not None and "state_eef_rot6d" in self._last_obs:
            state = np.asarray(self._last_obs["state_eef_rot6d"], dtype=np.float32).reshape(-1)
            raw_idx = 9 if side == "left" else 19
            idx = self._filtered_proprio_index("state_eef_rot6d", raw_idx)
            if idx is not None and state.size > idx:
                return state[idx : idx + 1].astype(np.float32)
        return np.array([0.5], dtype=np.float32)

    def _drain_obs_queue(self) -> None:
        """Remove all pending items from obs_queue."""
        while not self._obs_queue.empty():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                break

    def _zero_obs(self) -> dict:
        """Return a zero observation (used when no obs is available yet)."""
        H, W = self._image_size
        obs: dict = {}
        for key in self._image_keys:
            obs[key] = np.zeros((H, W, 3), dtype=np.uint8)
        for key in self._proprio_keys:
            dim = self._effective_proprio_dim(key)
            obs[key] = np.zeros(dim, dtype=np.float32)
        return obs

