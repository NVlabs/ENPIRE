# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools that communicate via Portal RPC or a direct env.

These wrap the low-level robot control primitives: reading state, moving
end-effectors, commanding grippers, and going home.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.base import (
    MoveResult,
    RobotState,
    Tool,
    ToolParameter,
    ToolResult,
)
from enpire.env.forge.cap.config import (
    CAP_SERVER_PORT,
    GRIPPER_POLL_S,
    GRIPPER_SETTLE_THRESH,
    GRIPPER_SETTLE_TIMEOUT_S,
)


class _PortalMixin:
    """Lazy Portal client for native tools.

    Each tool instance gets its own client (stored on the instance, not the class).
    """

    def _get_client(self, host: str, port: int) -> Any:
        if not hasattr(self, "_client") or self._client is None:
            import portal

            self._client = portal.Client(f"{host}:{port}")
        return self._client


def _quat_xyzw_to_rpy(q: list[float] | np.ndarray) -> list[float]:
    """Convert quaternion [x, y, z, w] to display RPY degrees."""
    from scipy.spatial.transform import Rotation

    e = Rotation.from_quat(q).as_euler("xyz", degrees=True)
    return [float(e[1]), float(-e[0]), float(-(e[2] + 90.0))]


def _env_obs(env: Any, side: str) -> dict[str, np.ndarray]:
    if hasattr(env, "get_observations"):
        return env.get_observations(side)
    if hasattr(env, "get_arm_observation"):
        return env.get_arm_observation(side)
    raise AttributeError(
        "env must expose get_observations(side) or get_arm_observation(side)"
    )


def _result_success(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", result.get("ok", True)))
    return bool(result)


def _result_reason(result: Any, default: str = "") -> str:
    if isinstance(result, dict):
        return str(result.get("reason") or result.get("error") or default)
    return default


def _arm_profile(env: Any, side: str) -> Any:
    return getattr(getattr(env, "_profile", None), "arms", {})[side]


def _command_joint7(
    env: Any,
    side: str,
    joint7: np.ndarray,
    *,
    vel_limit: float | None = None,
    torque_limit: float | None = None,
) -> None:
    profile = _arm_profile(env, side)
    target = np.asarray(joint7, dtype=np.float64).reshape(7)
    env.command_joint_state(
        side,
        {
            "pos": target,
            "vel": np.zeros_like(target),
            "kp": profile.interp_kp,
            "kd": profile.interp_kd,
            "gripper_vel_limit": vel_limit,
            "gripper_torque_limit_nm": torque_limit,
        },
    )


def set_gripper_direct(
    env: Any,
    side: str,
    pos: float,
    *,
    timeout: float = GRIPPER_SETTLE_TIMEOUT_S,
    vel_limit: float | None = None,
    torque_limit: float | None = None,
) -> Any:
    """Set a RealYamEnv gripper using env.set_gripper when present, else command_joint_state."""
    if hasattr(env, "set_gripper"):
        return env.set_gripper(side, pos, timeout, vel_limit, torque_limit)
    if hasattr(env, "_set_gripper"):
        return env._set_gripper(side, pos, timeout, vel_limit, torque_limit)

    value = float(np.clip(float(pos), 0.0, 1.0))
    timeout = max(0.0, float(timeout))
    start = time.time()
    initial_obs = _env_obs(env, side)
    hold_joint_pos = (
        np.asarray(initial_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6].copy()
    )
    actual: float | None = None
    settled = False
    allow_early_settle = not (torque_limit is not None and value <= GRIPPER_SETTLE_THRESH)

    while True:
        obs = _env_obs(env, side)
        actual = float(np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        _command_joint7(
            env,
            side,
            np.concatenate([hold_joint_pos, [value]]),
            vel_limit=vel_limit,
            torque_limit=torque_limit,
        )

        settled = abs(actual - value) < GRIPPER_SETTLE_THRESH
        if (settled and allow_early_settle) or time.time() - start >= timeout:
            break
        time.sleep(min(GRIPPER_POLL_S, max(0.0, timeout - (time.time() - start))))

    return {
        "success": True,
        "side": side,
        "gripper": value,
        "final_gripper_pos": actual,
        "settled": settled,
        "stall_held": False,
        "vel_limit": vel_limit,
        "torque_limit": torque_limit,
    }


def go_home_direct(env: Any, *, duration: float = 3.0) -> Any:
    """Move both RealYamEnv arms to profile home using direct joint commands."""
    if hasattr(env, "go_home"):
        return env.go_home()
    if hasattr(env, "_go_home"):
        return env._go_home()

    left_p = _arm_profile(env, "left")
    right_p = _arm_profile(env, "right")
    obs_l = _env_obs(env, "left")
    obs_r = _env_obs(env, "right")
    initial = np.concatenate(
        [
            np.asarray(obs_l["joint_pos"], dtype=np.float64).reshape(-1)[:6],
            np.asarray(obs_l["gripper_pos"], dtype=np.float64).reshape(-1)[:1],
            np.asarray(obs_r["joint_pos"], dtype=np.float64).reshape(-1)[:6],
            np.asarray(obs_r["gripper_pos"], dtype=np.float64).reshape(-1)[:1],
        ]
    )
    target = np.concatenate(
        [
            np.asarray(left_p.home_joint_pos, dtype=np.float64).reshape(-1)[:6],
            np.asarray(left_p.home_gripper_pos, dtype=np.float64).reshape(-1)[:1],
            np.asarray(right_p.home_joint_pos, dtype=np.float64).reshape(-1)[:6],
            np.asarray(right_p.home_gripper_pos, dtype=np.float64).reshape(-1)[:1],
        ]
    )

    dt = 0.02
    t0 = time.time()
    duration = max(dt, float(duration))
    while time.time() - t0 < duration:
        alpha = float(np.clip((time.time() - t0) / duration, 0.0, 1.0))
        interp = (1.0 - alpha) * initial + alpha * target
        _command_joint7(env, "left", interp[:7])
        _command_joint7(env, "right", interp[7:14])
        time.sleep(dt)

    settle_t0 = time.time()
    while time.time() - settle_t0 < 0.5:
        _command_joint7(env, "left", target[:7])
        _command_joint7(env, "right", target[7:14])
        time.sleep(dt)
    return {"success": True, "status": "done"}


def _sample_keypoints(timestamps: np.ndarray, values: np.ndarray, t_now: float) -> np.ndarray:
    if t_now <= float(timestamps[0]):
        return values[0]
    if t_now >= float(timestamps[-1]):
        return values[-1]
    idx = int(np.searchsorted(timestamps, t_now, side="right") - 1)
    idx = max(0, min(idx, len(timestamps) - 2))
    t0 = float(timestamps[idx])
    t1 = float(timestamps[idx + 1])
    alpha = 1.0 if t1 <= t0 + 1e-9 else (float(t_now) - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * values[idx] + alpha * values[idx + 1]


def _prepare_joint7_waypoints(
    env: Any,
    side: str,
    joint_positions: Any,
    gripper_positions: Any,
    n: int,
) -> np.ndarray:
    arr = np.asarray(joint_positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 6:
        raise ValueError(
            f"{side}_joint_positions must have shape (N,6) or (N,7); got {arr.shape}"
        )
    joints = arr[:, :6]
    if gripper_positions is None:
        if arr.shape[1] >= 7:
            gripper = arr[:, 6]
        else:
            cur = _env_obs(env, side)["gripper_pos"]
            gripper = np.full(
                n, float(np.asarray(cur).reshape(-1)[0]), dtype=np.float64
            )
    else:
        gripper = np.asarray(gripper_positions, dtype=np.float64).reshape(n, -1)[:, 0]
    return np.column_stack([joints, np.clip(gripper, 0.0, 1.0)]).astype(np.float64)


def execute_bimanual_joint_keypoints_direct(
    env: Any,
    timestamps: Any,
    left_joint_positions: Any,
    right_joint_positions: Any,
    left_gripper_positions: Any = None,
    right_gripper_positions: Any = None,
    *,
    playback_speed: float = 1.0,
    command_hz: float = 60.0,
    start_interp_s: float = 0.0,
) -> dict[str, Any]:
    """Execute bimanual waypoints directly against RealYamEnv-style command APIs."""
    # Keep commanding the checked endpoint before a subsequent gripper command
    # snapshots the measured arm pose. Some stations need more than 0.2 seconds.
    settle_s = float(os.environ.get("ENPIRE_TRAJECTORY_SETTLE_S", "0.2"))
    if not np.isfinite(settle_s) or not 0.0 <= settle_s <= 5.0:
        raise ValueError("ENPIRE_TRAJECTORY_SETTLE_S must be finite and between 0 and 5")
    if hasattr(env, "move_bimanual_joint_keypoints"):
        return env.move_bimanual_joint_keypoints(
            timestamps,
            left_joint_positions,
            right_joint_positions,
            left_gripper_positions,
            right_gripper_positions,
            playback_speed=playback_speed,
            command_hz=command_hz,
            start_interp_s=start_interp_s,
        )
    if hasattr(env, "_move_bimanual_joint_keypoints"):
        return env._move_bimanual_joint_keypoints(
            timestamps,
            left_joint_positions,
            right_joint_positions,
            left_gripper_positions,
            right_gripper_positions,
            playback_speed=playback_speed,
            command_hz=command_hz,
            start_interp_s=start_interp_s,
        )

    ts_original = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts_original.size < 1:
        return {"success": False, "reason": "empty timestamps"}
    if not np.all(np.isfinite(ts_original)):
        return {"success": False, "reason": "timestamps contain non-finite values"}
    ts = ts_original - float(ts_original[0])
    if np.any(np.diff(ts) < -1e-9):
        return {
            "success": False,
            "reason": "timestamps must be monotonically increasing",
        }

    keep = np.ones(ts.shape[0], dtype=bool)
    keep[1:] = np.diff(ts) > 1e-9
    ts = ts[keep]
    try:
        left7 = _prepare_joint7_waypoints(
            env,
            "left",
            left_joint_positions,
            left_gripper_positions,
            len(ts_original),
        )[keep]
        right7 = _prepare_joint7_waypoints(
            env,
            "right",
            right_joint_positions,
            right_gripper_positions,
            len(ts_original),
        )[keep]
    except Exception as exc:
        return {"success": False, "reason": str(exc)}

    speed = max(0.05, float(playback_speed))
    ts = ts / speed
    duration_s = float(ts[-1]) if ts.size else 0.0
    dt = 1.0 / max(1.0, float(command_hz))

    obs_l = _env_obs(env, "left")
    obs_r = _env_obs(env, "right")
    cur_left7 = np.concatenate(
        [
            np.asarray(obs_l["joint_pos"], dtype=np.float64).reshape(-1)[:6],
            np.asarray(obs_l["gripper_pos"], dtype=np.float64).reshape(-1)[:1],
        ]
    )
    cur_right7 = np.concatenate(
        [
            np.asarray(obs_r["joint_pos"], dtype=np.float64).reshape(-1)[:6],
            np.asarray(obs_r["gripper_pos"], dtype=np.float64).reshape(-1)[:1],
        ]
    )

    interp_steps = int(np.ceil(max(0.0, float(start_interp_s)) / dt))
    for step in range(1, interp_steps + 1):
        alpha = float(step) / float(max(interp_steps, 1))
        _command_joint7(env, "left", (1.0 - alpha) * cur_left7 + alpha * left7[0])
        _command_joint7(env, "right", (1.0 - alpha) * cur_right7 + alpha * right7[0])
        time.sleep(dt)

    t0 = time.time()
    command_count = 0
    while True:
        t_now = time.time() - t0
        _command_joint7(env, "left", _sample_keypoints(ts, left7, t_now))
        _command_joint7(env, "right", _sample_keypoints(ts, right7, t_now))
        command_count += 1
        if t_now >= duration_s:
            break
        time.sleep(dt)

    settle_steps = max(1, int(round(settle_s / dt)))
    for _ in range(settle_steps):
        _command_joint7(env, "left", left7[-1])
        _command_joint7(env, "right", right7[-1])
        time.sleep(dt)

    return {
        "success": True,
        "reason": "ok",
        "waypoints": int(ts.size),
        "duration_s": round(duration_s, 4),
        "playback_speed": float(speed),
        "command_hz": float(command_hz),
        "command_count": int(command_count),
        "start_interp_s": float(start_interp_s),
        "final_left_gripper": float(left7[-1, 6]),
        "final_right_gripper": float(right7[-1, 6]),
    }


# ---------------------------------------------------------------------------
# get_robot_state
# ---------------------------------------------------------------------------


class GetRobotStateTool(_PortalMixin, Tool):
    name = "get_robot_state"
    description = (
        "Get current joint positions, gripper states, and EE poses for both arms."
    )
    parameters: list[ToolParameter] = []

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            from enpire.env.forge.cap.agent.tools.base import ArmState

            if self._env is not None:
                arms: dict[str, ArmState] = {}
                for side in ("left", "right"):
                    obs = _env_obs(self._env, side)
                    quat = list(
                        np.asarray(obs["ee_quat"], dtype=np.float64).reshape(-1)
                    )
                    arms[side] = ArmState(
                        joint_pos=list(
                            np.asarray(obs["joint_pos"], dtype=np.float64).reshape(-1)
                        ),
                        gripper_pos=float(
                            np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(
                                -1
                            )[0]
                        ),
                        ee_pos=list(
                            np.asarray(obs["ee_pos"], dtype=np.float64).reshape(-1)
                        ),
                        ee_quat=quat,
                        ee_rpy=_quat_xyzw_to_rpy(quat),
                    )
                return ToolResult(success=True, data=RobotState(arms=arms))

            client = self._get_client(self._host, self._port)
            state = client.get_state().result()

            arms: dict[str, ArmState] = {}
            for key in state:
                if key.endswith("_joint_pos"):
                    side = key[: -len("_joint_pos")]
                    # Skip non-arm keys like "base_pos"
                    if f"{side}_ee_pos" not in state:
                        continue
                    quat = list(state[f"{side}_ee_quat_xyzw"])
                    arms[side] = ArmState(
                        joint_pos=list(state[f"{side}_joint_pos"]),
                        gripper_pos=float(state[f"{side}_gripper_pos"][0]),
                        ee_pos=list(state[f"{side}_ee_pos"]),
                        ee_quat=quat,
                        ee_rpy=_quat_xyzw_to_rpy(quat),
                    )
            robot_state = RobotState(arms=arms)
            return ToolResult(success=True, data=robot_state)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# set_gripper / open_gripper / close_gripper
# ---------------------------------------------------------------------------


class SetGripperTool(_PortalMixin, Tool):
    name = "set_gripper"
    description = (
        "Set gripper position on the specified side. "
        "pos=1.0 fully open, pos=-1.0 fully closed. Blocking until settled."
    )
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "pos",
            "float",
            "Gripper target position in [0.0, 1.0]. 1.0=open, 0.0=closed.",
        ),
        ToolParameter(
            "vel_limit",
            "float",
            "Gripper speed limit in rad/s. Lower = slower. Defaults to the robot's configured limit.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "torque_limit",
            "float",
            "Gripper torque limit in Nm. Lower = gentler grip. Defaults to robot configured limit.",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if self._env is not None:
                result = set_gripper_direct(
                    self._env,
                    kwargs["side"],
                    kwargs["pos"],
                    timeout=GRIPPER_SETTLE_TIMEOUT_S,
                    vel_limit=kwargs.get("vel_limit"),
                    torque_limit=kwargs.get("torque_limit"),
                )
                ok = _result_success(result)
                return ToolResult(
                    success=ok,
                    data=result,
                    error=None if ok else _result_reason(result),
                )

            client = self._get_client(self._host, self._port)
            ok = client.set_gripper(
                kwargs["side"],
                kwargs["pos"],
                GRIPPER_SETTLE_TIMEOUT_S,
                kwargs.get("vel_limit"),
                kwargs.get("torque_limit"),
            ).result()
            return ToolResult(success=bool(ok))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class OpenGripperTool(_PortalMixin, Tool):
    name = "open_gripper"
    description = "Open the gripper on the specified side ('left' or 'right')."
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "vel_limit",
            "float",
            "Gripper speed limit in rad/s. Lower = slower. Defaults to the robot's configured limit.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "torque_limit",
            "float",
            "Gripper torque limit in Nm. Lower = gentler grip. Defaults to robot configured limit.",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if self._env is not None:
                result = set_gripper_direct(
                    self._env,
                    kwargs["side"],
                    1.0,
                    timeout=GRIPPER_SETTLE_TIMEOUT_S,
                    vel_limit=kwargs.get("vel_limit"),
                    torque_limit=kwargs.get("torque_limit"),
                )
                ok = _result_success(result)
                return ToolResult(
                    success=ok,
                    data=result,
                    error=None if ok else _result_reason(result),
                )

            client = self._get_client(self._host, self._port)
            ok = client.set_gripper(
                kwargs["side"],
                1.0,
                GRIPPER_SETTLE_TIMEOUT_S,
                kwargs.get("vel_limit"),
                kwargs.get("torque_limit"),
            ).result()
            return ToolResult(success=bool(ok))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class OpenGripperFastTool(_PortalMixin, Tool):
    name = "open_gripper_fast"
    description = "Open the gripper quickly with a short blocking timeout."
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "vel_limit",
            "float",
            "Gripper speed limit in rad/s. Defaults to 30.0.",
            required=False,
            default=30.0,
        ),
        ToolParameter(
            "torque_limit",
            "float",
            "Gripper torque limit in Nm. Defaults to robot configured limit.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "timeout",
            "float",
            "Maximum seconds to block waiting for the gripper command.",
            required=False,
            default=0.25,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            timeout = float(kwargs.get("timeout", 0.25))
            vel_limit = kwargs.get("vel_limit", 30.0)
            if self._env is not None:
                result = set_gripper_direct(
                    self._env,
                    kwargs["side"],
                    1.0,
                    timeout=timeout,
                    vel_limit=vel_limit,
                    torque_limit=kwargs.get("torque_limit"),
                )
                ok = _result_success(result)
                return ToolResult(
                    success=ok,
                    data=result,
                    error=None if ok else _result_reason(result),
                )

            client = self._get_client(self._host, self._port)
            ok = client.set_gripper(
                kwargs["side"],
                1.0,
                timeout,
                vel_limit,
                kwargs.get("torque_limit"),
            ).result()
            return ToolResult(success=bool(ok))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CloseGripperTool(_PortalMixin, Tool):
    name = "close_gripper"
    description = "Close the gripper on the specified side ('left' or 'right')."
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "vel_limit",
            "float",
            "Gripper speed limit in rad/s. Lower = slower. Defaults to the robot's configured limit.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "torque_limit",
            "float",
            "Gripper torque limit in Nm. Lower = gentler grip. Defaults to robot configured limit.",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if self._env is not None:
                result = set_gripper_direct(
                    self._env,
                    kwargs["side"],
                    0.0,
                    timeout=GRIPPER_SETTLE_TIMEOUT_S,
                    vel_limit=kwargs.get("vel_limit"),
                    torque_limit=kwargs.get("torque_limit"),
                )
                ok = _result_success(result)
                return ToolResult(
                    success=ok,
                    data=result,
                    error=None if ok else _result_reason(result),
                )

            client = self._get_client(self._host, self._port)
            ok = client.set_gripper(
                kwargs["side"],
                0.0,
                GRIPPER_SETTLE_TIMEOUT_S,
                kwargs.get("vel_limit"),
                kwargs.get("torque_limit"),
            ).result()
            return ToolResult(success=bool(ok))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# go_home
# ---------------------------------------------------------------------------


class GoHomeTool(_PortalMixin, Tool):
    name = "go_home"
    description = "Move both arms to zero/home configuration."
    parameters: list[ToolParameter] = []

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if self._env is not None:
                result = go_home_direct(self._env)
                ok = _result_success(result)
                return ToolResult(
                    success=ok,
                    data=result,
                    error=None if ok else _result_reason(result),
                )

            client = self._get_client(self._host, self._port)
            ok = client.go_home().result()
            return ToolResult(success=bool(ok))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# get_camera_image
# ---------------------------------------------------------------------------


class MoveJointKeypointsTool(_PortalMixin, Tool):
    name = "move_joint_keypoints"
    description = (
        "Execute joint-space trajectory via timestamped waypoints. "
        "timestamps: seconds from call time (monotonically increasing). "
        "joint_positions: one joint position array per timestamp. "
        "gripper_positions: optional gripper waypoint per timestamp. "
        "Linearly interpolates. Blocking."
    )
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter(
            "timestamps",
            "list[float]",
            "Waypoint arrival times in seconds from call time.",
        ),
        ToolParameter(
            "joint_positions",
            "list",
            "One joint position array per timestamp.",
        ),
        ToolParameter(
            "gripper_positions",
            "list | None",
            "Optional gripper position per timestamp (0=closed, 1=open).",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        side = kwargs["side"]
        timestamps = np.asarray(kwargs["timestamps"], dtype=np.float64)
        joint_positions = np.asarray(kwargs["joint_positions"], dtype=np.float64)
        gripper_positions = kwargs.get("gripper_positions")
        try:
            if self._env is not None:
                if hasattr(self._env, "move_joint_keypoints"):
                    result = self._env.move_joint_keypoints(
                        side, timestamps, joint_positions, gripper_positions
                    )
                elif hasattr(self._env, "_move_joint_keypoints"):
                    result = self._env._move_joint_keypoints(
                        side, timestamps, joint_positions, gripper_positions
                    )
                else:
                    other_side = "right" if side == "left" else "left"
                    other_obs = _env_obs(self._env, other_side)
                    other_joint = np.asarray(
                        other_obs["joint_pos"], dtype=np.float64
                    ).reshape(1, -1)[:, :6]
                    other_positions = np.repeat(
                        other_joint, joint_positions.shape[0], axis=0
                    )
                    if side == "left":
                        result = execute_bimanual_joint_keypoints_direct(
                            self._env,
                            timestamps,
                            joint_positions,
                            other_positions,
                            gripper_positions,
                            None,
                        )
                    else:
                        result = execute_bimanual_joint_keypoints_direct(
                            self._env,
                            timestamps,
                            other_positions,
                            joint_positions,
                            None,
                            gripper_positions,
                        )
                success = _result_success(result)
                move_result = MoveResult(reached=success)
                if not success:
                    return ToolResult(
                        success=False,
                        data=move_result,
                        error=_result_reason(result, "joint trajectory execution failed"),
                    )
                return ToolResult(success=True, data=move_result)

            client = self._get_client(self._host, self._port)
            result = client.move_joint_keypoints(
                side, timestamps, joint_positions, gripper_positions
            ).result()
            success = bool(result.get("success", False))
            reason = result.get("reason", "")
            move_result = MoveResult(reached=success)
            if not success:
                return ToolResult(success=False, data=move_result, error=reason)
            return ToolResult(success=True, data=move_result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetCameraImageTool(_PortalMixin, Tool):
    name = "get_camera_image"
    description = (
        "Get latest RGB image from specified camera ('top', 'left', or 'right')."
    )
    parameters = [
        ToolParameter("camera", "str", "Camera name: 'top', 'left', or 'right'.")
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            if self._env is not None:
                image = self._env.render_rgb(kwargs["camera"])
                return ToolResult(success=True, data=image)

            client = self._get_client(self._host, self._port)
            image = client.get_camera_image(kwargs["camera"]).result()
            return ToolResult(success=True, data=image)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Scene management (sim-only)
# ---------------------------------------------------------------------------


class SetupSceneTool(_PortalMixin, Tool):
    name = "setup_scene"
    description = (
        "Load a named scene into the simulation, placing objects on the table. "
        "Use list_scenes() first to see available scenes. Replaces any current scene."
    )
    parameters = [ToolParameter("name", "str", "Scene name (e.g. 'block_plate').")]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            result = client.setup_scene(kwargs["name"]).result()
            if result.get("ok"):
                return ToolResult(success=True, data=result)
            return ToolResult(success=False, error=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ClearTableTool(_PortalMixin, Tool):
    name = "clear_table"
    description = "Remove all scene objects from the simulation table, restoring the bare station."
    parameters: list[ToolParameter] = []

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            result = client.clear_table().result()
            if result.get("ok"):
                return ToolResult(success=True, data=result)
            return ToolResult(success=False, error=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListScenesTool(_PortalMixin, Tool):
    name = "list_scenes"
    description = "List available scene files and the currently active scene."
    parameters: list[ToolParameter] = []

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            result = client.list_scenes().result()
            if result.get("ok"):
                return ToolResult(success=True, data=result)
            return ToolResult(success=False, error=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetTaskInfoTool(_PortalMixin, Tool):
    name = "get_task_info"
    description = "Get task info (reward, success, done, object positions) from the environment."
    parameters: list[ToolParameter] = []

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            result = client.get_task_info().result()
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LoadTaskTool(_PortalMixin, Tool):
    name = "load_task"
    description = "Load a new task/scene (e.g. RoboCasa task). Resets the environment."
    parameters: list[ToolParameter] = [
        ToolParameter(name="task_name", type="string", description="Task name to load", required=True),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            result = client.load_task(kwargs["task_name"]).result()
            if result.get("ok"):
                return ToolResult(success=True, data=result)
            return ToolResult(success=False, error=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetObjectPositionsTool(_PortalMixin, Tool):
    name = "get_object_positions"
    description = (
        "Get positions and orientations of all scene objects (sim-only, for debugging)."
    )
    parameters: list[ToolParameter] = []

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            result = client.get_object_positions().result()
            if result.get("ok"):
                return ToolResult(success=True, data=result)
            return ToolResult(success=False, error=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SetBodyPoseTool(_PortalMixin, Tool):
    name = "set_body_pose"
    description = (
        "Set a scene body's pose in the simulation. "
        "quat_wxyz is [w,x,y,z]. Enables gravity compensation by default so the body floats."
    )
    parameters = [
        ToolParameter("name", "str", "Body name (must be in the active scene)."),
        ToolParameter("pos", "list", "[x, y, z] world position."),
        ToolParameter(
            "quat_wxyz", "list", "[w, x, y, z] quaternion (MuJoCo convention)."
        ),
        ToolParameter(
            "gravity_comp", "bool",
            "Enable gravity compensation so the body floats (default True).",
            required=False,
        ),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = self._get_client(self._host, self._port)
            gravity_comp = kwargs.get("gravity_comp", True)
            result = client.set_body_pose(
                kwargs["name"],
                kwargs["pos"],
                kwargs["quat_wxyz"],
                gravity_comp,
            ).result()
            if result.get("ok"):
                return ToolResult(success=True, data=result)
            return ToolResult(success=False, error=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SetupRewardTool(_PortalMixin, Tool):
    """Set the reward function used by learn_skill / learn_skill_policy.

    Calls cap_server.set_reward_mode(mode) which stores the reward function
    in-process.  On every learn_skill step, the local fn is called instead of
    the external reward server (if one is running, it is ignored while a local
    fn is active).

    Pass mode="" to clear the local fn and fall back to the external server.

    Available modes:
        constant-0  Always 0.0
        constant-1  Always 1.0
        random      Random 0 or 1 each step
        insert_usb  1.0 when USB drive is mounted (real hardware)
        gemini      VLM binary reward via Gemini API (needs GEMINI_API_KEY)
    """

    name = "setup_reward"
    description = (
        "Set the reward function for the RL training loop. "
        "Call this before learn_skill to choose how reward is computed each step. "
        "Available modes: constant-0, constant-1, random, insert_usb, gemini. "
        "Pass mode='' to clear and fall back to the external reward server."
    )
    parameters = [
        ToolParameter(
            "mode",
            "str",
            "Reward mode name (e.g. 'peg_insertion', 'gemini', 'constant-1'). "
            "Pass '' to clear.",
        ),
    ]

    def __init__(self, host: str = "localhost", port: int = CAP_SERVER_PORT):
        self._host = host
        self._port = port

    def execute(self, **kwargs: Any) -> ToolResult:
        mode: str = kwargs.get("mode", "")
        try:
            client = self._get_client(self._host, self._port)
            result = client.set_reward_mode(mode).result()
            if result.get("ok"):
                active = result.get("mode")
                msg = (
                    f"Reward mode set to {active!r}"
                    if active
                    else "Reward cleared (using external server)"
                )
                return ToolResult(success=True, data=msg)
            return ToolResult(success=False, error=result.get("error", str(result)))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
