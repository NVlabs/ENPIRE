from __future__ import annotations

import io
import json
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import msgpack
import numpy as np

from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.key_remapping_utils import (
    hold_action_from_proprio,
    map_action,
    map_observation,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is optional at runtime
    Image = None


@dataclass
class PolicyAdapters:
    map_observation: (
        Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None
    ) = None
    map_action: Callable[[dict[str, Any]], dict[str, Any]] | None = None


DEFAULT_REQUEST_TIMEOUT_SECONDS = 1.0
TIMEOUT_WARNING_INTERVAL_SECONDS = 5.0
IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
ZMQWireFormat = Literal["msgpack", "pickle"]
ZMQRequestStyle = Literal["endpoint", "raw"]
ZMQPayloadMode = Literal["observation", "payload"]
DEFAULT_ZMQ_WIRE_FORMAT: ZMQWireFormat = "msgpack"
DEFAULT_ZMQ_REQUEST_STYLE: ZMQRequestStyle = "raw"
DEFAULT_ZMQ_PAYLOAD_MODE: ZMQPayloadMode = "payload"


class RLInterface:
    """Minimal RobotInterface-shaped client for non-VLA RL policy servers."""

    def __init__(
        self,
        embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF_WRISTONLY,
        server_address: str = "localhost:8965",
        startup_timeout: float = 120.0,
        adapters: PolicyAdapters | None = None,
        resolution: Literal[240, 256, 480] = 256,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        zmq_wire_format: ZMQWireFormat = DEFAULT_ZMQ_WIRE_FORMAT,
        zmq_request_style: ZMQRequestStyle = DEFAULT_ZMQ_REQUEST_STYLE,
        zmq_payload_mode: ZMQPayloadMode = DEFAULT_ZMQ_PAYLOAD_MODE,
        zmq_endpoint: str = "get_action",
        zmq_reset_endpoint: str = "reset",
        verify_connection: bool = False,
        timeout_fallback_action: Literal["hold", "raise"] = "hold",
        fallback_control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        ]
        | None = None,
        timeout_warning_interval: float = TIMEOUT_WARNING_INTERVAL_SECONDS,
        **_: Any,
    ) -> None:
        import zmq

        if server_address is None:
            server_address = "localhost:8965"

        self.embodiment_tag = embodiment_tag
        self.resolution = resolution
        self.adapters = adapters or PolicyAdapters(
            map_observation=lambda obs: map_observation(obs, embodiment_tag, resolution),
            map_action=lambda act: map_action(act, embodiment_tag),
        )
        self.server_address = server_address
        self._address = _zmq_address(server_address)
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock: Any | None = None
        self._request_timeout_s = float(request_timeout)
        self._request_timeout_ms = int(self._request_timeout_s * 1000)
        self._wire_format = zmq_wire_format
        self._request_style = zmq_request_style
        self._payload_mode = zmq_payload_mode
        self._endpoint = zmq_endpoint
        self._reset_endpoint = zmq_reset_endpoint
        self._timeout_fallback_action = timeout_fallback_action
        self._fallback_control_mode = fallback_control_mode
        self._timeout_warning_interval = float(timeout_warning_interval)
        self._last_timeout_warning_s = 0.0
        self._open_socket()
        if verify_connection:
            self._connect(startup_timeout)

    def reset(self, observation: dict[str, Any] | None = None) -> Any:
        try:
            if self._request_style == "raw":
                if observation is None:
                    return self._request({"__reset__": True})
                return self._request(
                    {"__reset__": True, "obs": self._build_payload(observation)}
                )

            if observation is None:
                return self._request_endpoint(
                    self._reset_endpoint, data=self._reset_request_data(None)
                )
            return self._request_endpoint(
                self._reset_endpoint,
                data=self._reset_request_data(self._build_payload(observation)),
            )
        except TimeoutError as exc:
            if self._timeout_fallback_action == "raise":
                raise
            self._warn_timeout("reset", exc)
            return self._timeout_info("reset", exc)

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._request_style == "raw":
            return self._request(_to_numpy(payload))
        return self._request_endpoint(
            self._endpoint, data=self._action_request_data(payload)
        )

    def done(self, reason: str = "done", info: dict[str, Any] | None = None) -> Any:
        payload: dict[str, Any] = {"__done__": True, "reason": reason}
        if info:
            payload["info"] = _to_numpy(info)
        try:
            if self._request_style == "raw":
                return self._request(payload)
            return self._request_endpoint("done", data=payload)
        except TimeoutError as exc:
            if self._timeout_fallback_action == "raise":
                raise
            self._warn_timeout("done", exc)
            return self._timeout_info("done", exc)

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.adapters.map_observation is None:
            raise RuntimeError("RLInterface requires adapters.map_observation.")

        try:
            response = self.step(self._build_payload(observation))
        except TimeoutError as exc:
            if self._timeout_fallback_action == "raise":
                raise
            self._warn_timeout("get_action", exc)
            return self._timeout_hold_action(observation), self._timeout_info(
                "get_action", exc
            )

        action, info = _extract_action(response)

        if self.adapters.map_action is not None:
            action = self.adapters.map_action(action)

        return _single_step_action(action), info

    def _timeout_hold_action(
        self, observation: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        if self._fallback_control_mode == "delta_joint_position":
            return _single_step_action(
                {
                    "left_joint_pos": np.zeros(6, dtype=np.float32),
                    "left_gripper_pos": _gripper_hold(observation, "left"),
                    "right_joint_pos": np.zeros(6, dtype=np.float32),
                    "right_gripper_pos": _gripper_hold(observation, "right"),
                }
            )
        if self._fallback_control_mode in (
            "delta_ee_pose",
            "delta_ee_pose_translation",
        ):
            return _single_step_action(
                {
                    "left_ee_pos": np.zeros(3, dtype=np.float32),
                    "left_ee_rot6d": IDENTITY_ROT6D.copy(),
                    "left_gripper_pos": _gripper_hold(observation, "left"),
                    "right_ee_pos": np.zeros(3, dtype=np.float32),
                    "right_ee_rot6d": IDENTITY_ROT6D.copy(),
                    "right_gripper_pos": _gripper_hold(observation, "right"),
                }
            )

        hold_action = hold_action_from_proprio(observation)
        hold_action.pop("source", None)
        return _single_step_action(hold_action)

    def _timeout_info(self, operation: str, exc: TimeoutError) -> dict[str, Any]:
        return {
            "rl_timeout": True,
            "timeout": True,
            "timeout_operation": operation,
            "timeout_error": str(exc),
            "action_source": "human",
        }

    def _warn_timeout(self, operation: str, exc: TimeoutError) -> None:
        now = time.monotonic()
        if now - self._last_timeout_warning_s < self._timeout_warning_interval:
            return
        self._last_timeout_warning_s = now
        print(
            "[WARN] RLInterface "
            f"{operation} timed out after {self._request_timeout_s:.1f}s; "
            "returning a hold action until the policy server responds. "
            f"({exc})"
        )

    def _build_payload(self, observation: dict[str, Any]) -> dict[str, Any]:
        assert self.adapters.map_observation is not None
        images, proprio = self.adapters.map_observation(observation)
        return {
            "images": {
                key.replace("observation.images.", ""): _to_numpy(value)
                for key, value in images.items()
            },
            "states": _to_numpy(proprio),
            "text": observation.get("annotation.task"),
            "embodiment": _embodiment_value(self.embodiment_tag),
        }

    def health_check(self) -> bool:
        if self._request_style == "raw":
            return self._sock is not None
        response = self._request_endpoint("ping", requires_input=False)
        if isinstance(response, dict):
            return "error" not in response
        return True

    def dry_run(self, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        """Exercise the configured ZMQ path without stepping the robot environment."""

        started = time.perf_counter()
        result: dict[str, Any] = {
            "server_address": self.server_address,
            "wire_format": self._wire_format,
            "request_style": self._request_style,
            "payload_mode": self._payload_mode,
        }
        result["health_check"] = self.health_check()
        if observation is None:
            result["latency_ms"] = _ms(time.perf_counter() - started)
            return result

        payload = self._build_payload(observation)
        response = self.step(payload)
        action, info = _extract_action(response)
        if self.adapters.map_action is not None:
            action = self.adapters.map_action(action)
        action = _single_step_action(action)
        result.update(
            {
                "latency_ms": _ms(time.perf_counter() - started),
                "payload": _summarize_tree(payload),
                "action": _summarize_tree(action),
                "info_keys": sorted(info.keys()),
            }
        )
        return result

    def _connect(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                if self.health_check():
                    return
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
        detail = f": {last_error}" if last_error is not None else ""
        raise TimeoutError(
            f"RLInterface could not complete ZMQ health check with "
            f"{self.server_address} after {timeout}s{detail}"
        )

    def _open_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = self._ctx.socket(self._zmq.REQ)
        linger = getattr(self._zmq, "LINGER", None)
        if linger is not None:
            self._sock.setsockopt(linger, 0)
        self._sock.setsockopt(self._zmq.RCVTIMEO, self._request_timeout_ms)
        self._sock.setsockopt(self._zmq.SNDTIMEO, self._request_timeout_ms)
        self._sock.connect(self._address)

    def _request_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = _to_numpy(data or {})
        response = self._request(request)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(
                f"RLInterface ZMQ server error at endpoint {endpoint}: "
                f"{response['error']}"
            )
        return response

    def _action_request_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._payload_mode == "observation":
            return {"observation": _to_numpy(payload), "options": None}
        return _to_numpy(payload)

    def _reset_request_data(
        self, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        if self._payload_mode == "observation":
            options: dict[str, Any] = {}
            if payload is not None:
                options["observation"] = _to_numpy(payload)
            return {"options": options or None}
        return _to_numpy(payload or {})

    def _request(self, payload: dict[str, Any]) -> Any:
        assert self._sock is not None
        try:
            self._sock.send(_serialize_message(_to_numpy(payload), self._wire_format))
            response = _deserialize_message(self._sock.recv(), self._wire_format)
            if isinstance(response, dict) and "error" in response:
                detail = response.get("detail")
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"RLInterface ZMQ server error: {response['error']}{suffix}"
                )
            return response
        except self._zmq.Again as exc:
            self._open_socket()
            raise TimeoutError("RLInterface ZMQ request timed out") from exc
        except self._zmq.ZMQError as exc:
            self._open_socket()
            raise RuntimeError(f"RLInterface ZMQ request failed: {exc}") from exc


def _extract_action(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {}
    if isinstance(response, (tuple, list)):
        if not response:
            raise ValueError("empty policy response")
        info["raw_response_tail"] = tuple(response[1:])
        response = response[0]
    if not isinstance(response, dict):
        response_type = type(response).__name__
        raise TypeError(f"policy response must be a dict, got {response_type}")
    action_key = "action" if "action" in response else "actions"
    if action_key not in response:
        return response, info
    info.update(response)
    action = info.pop(action_key)
    return action, info


def _single_step_action(action: dict[str, Any]) -> dict[str, np.ndarray]:
    single_action: dict[str, np.ndarray] = {}
    for key, value in action.items():
        if isinstance(value, dict):
            continue
        arr = np.asarray(value)
        if arr.ndim == 3:
            assert arr.shape[0] == 1, "Should only have batch size 1"
            arr = arr[0, 0]
        elif arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim == 0:
            arr = arr.reshape(1)
        single_action[key] = np.ascontiguousarray(arr)
    return single_action


def _gripper_hold(observation: dict[str, Any], side: Literal["left", "right"]) -> np.ndarray:
    return np.asarray(
        observation.get(f"{side}_gripper_pos", np.zeros(1, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)[:1]


def _to_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    if Image is not None and isinstance(obj, Image.Image):
        return np.ascontiguousarray(np.asarray(obj))
    if isinstance(obj, EmbodimentTag):
        return obj.value
    if isinstance(obj, dict):
        return {key: _to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_numpy(value) for value in obj]
    return obj


def _embodiment_value(embodiment: EmbodimentTag | str) -> str:
    if isinstance(embodiment, EmbodimentTag):
        return embodiment.value
    return str(embodiment)


def _zmq_address(server_address: str) -> str:
    if "://" in server_address:
        return server_address
    return f"tcp://{server_address}"


def _serialize_message(
    payload: dict[str, Any], wire_format: Literal["msgpack", "pickle"]
) -> bytes:
    if wire_format == "pickle":
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return msgpack.packb(payload, default=_msgpack_default, use_bin_type=True)


def _deserialize_message(
    message: bytes, wire_format: Literal["msgpack", "pickle"]
) -> Any:
    if wire_format == "pickle":
        return pickle.loads(message)
    return msgpack.unpackb(message, object_hook=_msgpack_object_hook, raw=False)


def _msgpack_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, np.ascontiguousarray(obj), allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    if isinstance(obj, EmbodimentTag):
        return obj.value
    raise TypeError(f"Cannot serialize {type(obj).__name__} over ZMQ msgpack")


def _msgpack_object_hook(obj: dict[str, Any]) -> Any:
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if "__ModalityConfig_class__" in obj:
        return json.loads(obj["as_json"])
    return obj


def _summarize_tree(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {key: _summarize_tree(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_summarize_tree(value) for value in obj]
    return obj


def _ms(seconds: float) -> float:
    return seconds * 1000.0


to_numpy = _to_numpy

