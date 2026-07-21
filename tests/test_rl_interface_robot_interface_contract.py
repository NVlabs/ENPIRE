from __future__ import annotations

import io
import pickle
import sys
import types
from typing import Any

import msgpack
import numpy as np

from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.key_remapping_utils import map_action, map_observation
from enpire.env.forge.experimental.rl_interface import PolicyAdapters, RLInterface
from enpire.env.forge.experimental.robot_interface import RobotInterface


SERVER_ADDRESS = "192.0.2.88:8965"
TASK_NAME = "Plug the pin into the socket"
RESOLUTION = 256
EMBODIMENT = EmbodimentTag.XDOF_WRISTONLY


class _Future:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> Any:
        return self._value


class _FakePortalClient:
    instances: list["_FakePortalClient"] = []

    def __init__(self, server_address: str) -> None:
        self.server_address = server_address
        self.step_payloads: list[dict[str, Any]] = []
        self.reset_payloads: list[tuple[Any, ...]] = []
        _FakePortalClient.instances.append(self)

    def health_check(self) -> _Future:
        return _Future(True)

    def reset(self, *args: Any) -> _Future:
        self.reset_payloads.append(args)
        return _Future({"ok": True})

    def step(self, payload: dict[str, Any]) -> _Future:
        self.step_payloads.append(payload)
        action = {
            "joint_pos_action_left": np.arange(6, dtype=np.float32)[None, :],
            "gripper_pos_action_left": np.array([[0.25]], dtype=np.float32),
            "joint_pos_action_right": (np.arange(6, dtype=np.float32) + 10.0)[None, :],
            "gripper_pos_action_right": np.array([[0.75]], dtype=np.float32),
        }
        if "actions" in payload:
            return _Future(action)
        return _Future({"action": action, "action_type": "joint_angle"})


class _FakeZMQAgain(Exception):
    pass


class _FakeZMQError(Exception):
    pass


class _FakeZMQSocket:
    instances: list["_FakeZMQSocket"] = []
    recv_exception: type[Exception] | None = None

    def __init__(self) -> None:
        self.connected_address: str | None = None
        self.options: dict[int, int] = {}
        self.sent_payloads: list[dict[str, Any]] = []
        self._reply: bytes = b""
        self._wire_format = "msgpack"
        _FakeZMQSocket.instances.append(self)

    def setsockopt(self, option: int, value: int) -> None:
        self.options[option] = value

    def connect(self, address: str) -> None:
        self.connected_address = address

    def send(self, data: bytes) -> None:
        try:
            request = pickle.loads(data)
            self._wire_format = "pickle"
        except Exception:
            request = msgpack.unpackb(data, object_hook=_msgpack_object_hook, raw=False)
            self._wire_format = "msgpack"
        self.sent_payloads.append(request)
        action = {
            "joint_pos_action_left": np.arange(6, dtype=np.float32)[None, :],
            "gripper_pos_action_left": np.array([[0.25]], dtype=np.float32),
            "joint_pos_action_right": (np.arange(6, dtype=np.float32) + 10.0)[None, :],
            "gripper_pos_action_right": np.array([[0.75]], dtype=np.float32),
        }
        endpoint = request.get("endpoint")
        if endpoint == "ping":
            response = {"status": "ok"}
        elif endpoint == "reset":
            response = {"ok": True}
        elif endpoint == "get_action":
            response = {"action": action, "action_type": "joint_angle"}
        elif endpoint is None:
            response = {"action": action, "action_type": "joint_angle"}
        else:
            response = {"error": f"unknown endpoint {endpoint}"}
        if self._wire_format == "pickle":
            self._reply = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self._reply = msgpack.packb(
                response, default=_msgpack_default, use_bin_type=True
            )

    def recv(self) -> bytes:
        if self.recv_exception is not None:
            raise self.recv_exception()
        return self._reply


class _FakeZMQContext:
    def socket(self, socket_type: int) -> _FakeZMQSocket:
        assert socket_type == 1
        return _FakeZMQSocket()


def _install_fake_transports(monkeypatch: Any) -> None:
    _FakePortalClient.instances = []
    _FakeZMQSocket.instances = []
    _FakeZMQSocket.recv_exception = None
    monkeypatch.setitem(sys.modules, "portal", types.SimpleNamespace(Client=_FakePortalClient))
    monkeypatch.setitem(
        sys.modules,
        "zmq",
        types.SimpleNamespace(
            Context=_FakeZMQContext,
            REQ=1,
            RCVTIMEO=2,
            SNDTIMEO=3,
            LINGER=4,
            Again=_FakeZMQAgain,
            ZMQError=_FakeZMQError,
        ),
    )


def _msgpack_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, np.ascontiguousarray(obj), allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    raise TypeError


def _msgpack_object_hook(obj: dict[str, Any]) -> Any:
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _adapters() -> PolicyAdapters:
    return PolicyAdapters(
        map_observation=lambda obs: map_observation(obs, EMBODIMENT, RESOLUTION),
        map_action=lambda act: map_action(act, EMBODIMENT),
    )


def _observation() -> dict[str, Any]:
    return {
        "left_camera_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "right_camera_image": np.full((480, 640, 3), 127, dtype=np.uint8),
        "left_joint_pos": np.linspace(0.0, 0.5, 6, dtype=np.float32),
        "left_gripper_pos": np.array([0.1], dtype=np.float32),
        "right_joint_pos": np.linspace(1.0, 1.5, 6, dtype=np.float32),
        "right_gripper_pos": np.array([0.9], dtype=np.float32),
        "annotation.task": TASK_NAME,
    }


def test_rl_interface_matches_robot_interface_io_contract_for_realworld_rl_constructor(
    monkeypatch: Any,
) -> None:
    _install_fake_transports(monkeypatch)
    adapters = _adapters()

    robot_policy = RobotInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=adapters,
    )
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=adapters,
    )

    robot_action, robot_info = robot_policy.get_action(_observation())
    rl_action, rl_info = rl_policy.get_action(_observation())

    assert len(_FakePortalClient.instances) == 1
    assert len(_FakeZMQSocket.instances) == 1
    robot_client = _FakePortalClient.instances[0]
    rl_socket = _FakeZMQSocket.instances[0]
    assert robot_client.server_address == SERVER_ADDRESS
    assert rl_socket.connected_address == f"tcp://{SERVER_ADDRESS}"

    robot_payload = robot_client.step_payloads[-1]
    rl_payload = rl_socket.sent_payloads[-1]

    assert robot_payload["text"] == rl_payload["text"] == TASK_NAME
    assert robot_payload["embodiment"] == rl_payload["embodiment"] == "xdof_wristonly"
    assert sorted(robot_payload["images"]) == sorted(rl_payload["images"])
    assert sorted(robot_payload["states"]) == sorted(rl_payload["states"])

    for key in robot_payload["images"]:
        assert robot_payload["images"][key].shape == rl_payload["images"][key].shape
        assert robot_payload["images"][key].dtype == rl_payload["images"][key].dtype

    expected_state_shapes = {
        "joint_pos_obs_left": (6,),
        "gripper_pos_obs_left": (1,),
        "joint_pos_obs_right": (6,),
        "gripper_pos_obs_right": (1,),
        "state_eef_rot6d": (20,),
    }
    for key, shape in expected_state_shapes.items():
        assert robot_payload["states"][key].shape == shape
        assert rl_payload["states"][key].shape == shape

    assert sorted(robot_action) == sorted(rl_action)
    for key in ("left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"):
        np.testing.assert_array_equal(robot_action[key], rl_action[key])
        assert robot_action[key].ndim == 1
        assert rl_action[key].ndim == 1
    assert "action_chunk" in robot_info
    assert "action_chunk" not in rl_info


def test_rl_interface_dry_run_uses_real_zmq_request_path(monkeypatch: Any) -> None:
    _install_fake_transports(monkeypatch)
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=_adapters(),
    )

    result = rl_policy.dry_run(_observation())

    assert result["health_check"] is True
    assert result["request_style"] == "raw"
    assert result["payload_mode"] == "payload"
    assert result["wire_format"] == "msgpack"
    assert result["payload"]["text"] == TASK_NAME
    assert result["payload"]["images"]["left_camera-images-rgb_256_256"]["shape"] == (
        256,
        256,
        3,
    )
    assert result["action"]["left_joint_pos"]["shape"] == (6,)


def test_rl_interface_can_use_msgpack_endpoint_servers(monkeypatch: Any) -> None:
    _install_fake_transports(monkeypatch)
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=_adapters(),
        zmq_wire_format="msgpack",
        zmq_request_style="endpoint",
        zmq_payload_mode="observation",
    )

    rl_policy.get_action(_observation())

    rl_socket = _FakeZMQSocket.instances[0]
    rl_payload = rl_socket.sent_payloads[-1]["data"]["observation"]
    assert rl_payload["text"] == TASK_NAME
    assert rl_payload["embodiment"] == "xdof_wristonly"


def test_rl_interface_defaults_to_one_second_request_timeout(monkeypatch: Any) -> None:
    _install_fake_transports(monkeypatch)
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=_adapters(),
    )

    assert rl_policy._request_timeout_ms == 1000
    socket = _FakeZMQSocket.instances[0]
    assert socket.options[2] == 1000
    assert socket.options[3] == 1000


def test_rl_interface_returns_hold_action_when_get_action_times_out(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    _install_fake_transports(monkeypatch)
    _FakeZMQSocket.recv_exception = _FakeZMQAgain
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=_adapters(),
        timeout_warning_interval=0.0,
    )
    observation = _observation()

    action, info = rl_policy.get_action(observation)

    assert info["rl_timeout"] is True
    assert info["timeout_operation"] == "get_action"
    assert info["action_source"] == "rl_timeout"
    np.testing.assert_array_equal(action["left_joint_pos"], observation["left_joint_pos"])
    np.testing.assert_array_equal(action["right_joint_pos"], observation["right_joint_pos"])
    assert "RLInterface get_action timed out after 1.0s" in capsys.readouterr().out


def test_rl_interface_reset_timeout_is_nonfatal_by_default(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    _install_fake_transports(monkeypatch)
    _FakeZMQSocket.recv_exception = _FakeZMQAgain
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=_adapters(),
        timeout_warning_interval=0.0,
    )

    result = rl_policy.reset()

    assert result["rl_timeout"] is True
    assert result["timeout_operation"] == "reset"
    assert result["action_source"] == "rl_timeout"
    assert "RLInterface reset timed out after 1.0s" in capsys.readouterr().out


def test_rl_interface_timeout_hold_respects_delta_ee_control_mode(
    monkeypatch: Any,
) -> None:
    _install_fake_transports(monkeypatch)
    _FakeZMQSocket.recv_exception = _FakeZMQAgain
    rl_policy = RLInterface(
        server_address=SERVER_ADDRESS,
        checkpoint_dir=None,
        embodiment_tag=EMBODIMENT,
        adapters=_adapters(),
        fallback_control_mode="delta_ee_pose",
        timeout_warning_interval=0.0,
    )

    action, info = rl_policy.get_action(_observation())

    assert info["action_source"] == "rl_timeout"
    np.testing.assert_array_equal(action["left_ee_pos"], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(action["right_ee_pos"], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(
        action["left_ee_rot6d"], np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        action["right_ee_rot6d"], np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    )
