# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from evdev import InputDevice, ecodes, list_devices

IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


@dataclass(frozen=True)
class SpaceMouseState:
    axes: dict[str, float]
    buttons: tuple[int, ...]
    age_s: float


class EvdevSpaceMouseReader:
    """Read a 3Dconnexion SpaceMouse from Linux evdev."""

    _AXIS_CODES = {
        ecodes.REL_X: "x",
        ecodes.REL_Y: "y",
        ecodes.REL_Z: "z",
        ecodes.REL_RX: "roll",
        ecodes.REL_RY: "pitch",
        ecodes.REL_RZ: "yaw",
    }

    def __init__(
        self,
        *,
        device_path: str | None,
        device_name_substring: str,
        axis_scale: float,
        stale_timeout_s: float,
    ):
        self.device_path = self._resolve_device_path(device_path, device_name_substring)
        self.device = InputDevice(self.device_path)
        self.axis_scale = float(axis_scale)
        if self.axis_scale <= 0.0:
            raise ValueError("spacemouse_axis_scale must be positive")
        self.stale_timeout_s = float(stale_timeout_s)
        self.description = f"{self.device.name} ({self.device_path})"

        self._axis_raw = {name: 0.0 for name in self._AXIS_CODES.values()}
        self._buttons = [0] * self._button_count(self.device)
        self._button_indices = self._button_indices_by_code(self.device)
        self._last_report_s = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _button_indices_by_code(device: InputDevice) -> dict[int, int]:
        key_codes = sorted(device.capabilities().get(ecodes.EV_KEY, []))
        return {int(code): i for i, code in enumerate(key_codes)}

    @classmethod
    def _button_count(cls, device: InputDevice) -> int:
        return max(1, len(cls._button_indices_by_code(device)))

    @classmethod
    def _has_required_axes(cls, device: InputDevice) -> bool:
        rel_codes = set(device.capabilities().get(ecodes.EV_REL, []))
        return set(cls._AXIS_CODES).issubset(rel_codes)

    @classmethod
    def _matches_device(cls, path: str, name_substring: str) -> bool:
        try:
            device = InputDevice(path)
        except OSError:
            return False
        try:
            name_matches = name_substring.lower() in device.name.lower()
            return name_matches and cls._has_required_axes(device)
        finally:
            device.close()

    @classmethod
    def _resolve_device_path(
        cls,
        device_path: str | None,
        device_name_substring: str,
    ) -> str:
        if device_path:
            path = str(Path(device_path).expanduser())
            if not Path(path).exists():
                raise FileNotFoundError(f"SpaceMouse device path does not exist: {path}")
            return path

        name = device_name_substring.strip() or "SpaceMouse"
        candidates: list[str] = []
        by_id = Path("/dev/input/by-id")
        if by_id.exists():
            candidates.extend(str(path) for path in sorted(by_id.glob("*event*")))
        candidates.extend(str(path) for path in list_devices())

        seen: set[str] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if cls._matches_device(path, name):
                return path

        raise FileNotFoundError(
            "No SpaceMouse evdev device found. Set spacemouse_device_path or "
            f"adjust spacemouse_device_name_substring (current: {name!r})."
        )

    def _read_loop(self) -> None:
        try:
            for event in self.device.read_loop():
                if not self._running:
                    break
                if event.type == ecodes.EV_REL and event.code in self._AXIS_CODES:
                    with self._lock:
                        self._axis_raw[self._AXIS_CODES[event.code]] = float(event.value)
                elif event.type == ecodes.EV_KEY and event.code in self._button_indices:
                    with self._lock:
                        self._buttons[self._button_indices[event.code]] = int(event.value > 0)
                elif event.type == ecodes.EV_SYN and event.code == ecodes.SYN_REPORT:
                    with self._lock:
                        self._last_report_s = time.monotonic()
        except OSError as exc:
            if self._running:
                self._error = exc

    def get_state(self) -> SpaceMouseState:
        if self._error is not None:
            raise RuntimeError(f"SpaceMouse read failed: {self._error}") from self._error
        now = time.monotonic()
        with self._lock:
            age_s = now - self._last_report_s if self._last_report_s > 0.0 else float("inf")
            if age_s > self.stale_timeout_s:
                axes = {name: 0.0 for name in self._axis_raw}
            else:
                axes = {
                    name: float(np.clip(value / self.axis_scale, -1.0, 1.0))
                    for name, value in self._axis_raw.items()
                }
            buttons = tuple(int(v) for v in self._buttons)
        return SpaceMouseState(axes=axes, buttons=buttons, age_s=age_s)

    def close(self) -> None:
        self._running = False
        self.device.close()


class SpaceMouseTeleopPolicy:
    """Map SpaceMouse motion to Forge's bimanual delta-EE action schema."""

    def __init__(
        self,
        *,
        enabled_sides: str,
        control_side: str,
        delta_ee_translation_xyz_max: Sequence[float],
        xyz_scale: Sequence[float],
        deadzone: float,
        axis_order: Sequence[str],
        axis_signs: Sequence[float],
        require_takeover_button: bool,
        takeover_button: int | None,
        open_gripper_button: int | None,
        close_gripper_button: int | None,
        gripper_open_pos: float,
        gripper_close_pos: float,
        reader: object | None = None,
        device_path: str | None = None,
        device_name_substring: str = "SpaceMouse",
        axis_scale: float = 350.0,
        stale_timeout_s: float = 0.25,
    ):
        self.enabled_sides = (
            ("left", "right") if enabled_sides == "both" else (str(enabled_sides),)
        )
        self.control_side = self._resolve_control_side(control_side, self.enabled_sides)
        self.delta_max = np.asarray(delta_ee_translation_xyz_max, dtype=np.float32).reshape(3)
        self.xyz_scale = np.asarray(xyz_scale, dtype=np.float32).reshape(3)
        self.deadzone = float(deadzone)
        self.axis_order = tuple(str(axis) for axis in axis_order)
        self.axis_signs = np.asarray(axis_signs, dtype=np.float32).reshape(-1)
        if len(self.axis_order) != 6 or self.axis_signs.shape != (6,):
            raise ValueError("SpaceMouse axis_order and axis_signs must each have 6 values")
        self.require_takeover_button = bool(require_takeover_button)
        self.takeover_button = takeover_button
        self.open_gripper_button = open_gripper_button
        self.close_gripper_button = close_gripper_button
        self.gripper_open_pos = float(gripper_open_pos)
        self.gripper_close_pos = float(gripper_close_pos)
        self._last_buttons: tuple[int, ...] = ()
        self.reader = reader or EvdevSpaceMouseReader(
            device_path=device_path,
            device_name_substring=device_name_substring,
            axis_scale=axis_scale,
            stale_timeout_s=stale_timeout_s,
        )

    @staticmethod
    def _resolve_control_side(control_side: str, enabled_sides: tuple[str, ...]) -> str:
        side = str(control_side).strip().lower()
        if side == "auto":
            if len(enabled_sides) != 1:
                raise ValueError(
                    "spacemouse_control_side='auto' is ambiguous when enabled_sides='both'"
                )
            return enabled_sides[0]
        if side not in {"left", "right"}:
            raise ValueError(f"Invalid spacemouse_control_side: {control_side!r}")
        if side not in enabled_sides:
            raise ValueError(
                f"spacemouse_control_side={side!r} is not in enabled_sides={enabled_sides!r}"
            )
        return side

    @staticmethod
    def _gripper_from_obs(obs: dict, side: str) -> np.ndarray:
        val = obs.get(f"{side}_gripper_pos")
        if val is None:
            return np.array([1.0], dtype=np.float32)
        return np.asarray(val, dtype=np.float32).reshape(-1)[:1].copy()

    @staticmethod
    def _button_pressed(buttons: Sequence[int], index: int | None) -> bool:
        if index is None or int(index) < 0:
            return False
        idx = int(index)
        return idx < len(buttons) and bool(buttons[idx])

    def _base_action(self, obs: dict) -> dict:
        return {
            "left_ee_pos": np.zeros(3, dtype=np.float32),
            "left_ee_rot6d": IDENTITY_ROT6D.copy(),
            "left_gripper_pos": self._gripper_from_obs(obs, "left"),
            "right_ee_pos": np.zeros(3, dtype=np.float32),
            "right_ee_rot6d": IDENTITY_ROT6D.copy(),
            "right_gripper_pos": self._gripper_from_obs(obs, "right"),
        }

    def get_action(self, obs: dict) -> tuple[dict, dict]:
        state = self.reader.get_state()
        last_buttons = self._last_buttons
        just_pressed = tuple(
            i
            for i, pressed in enumerate(state.buttons)
            if pressed and (i >= len(last_buttons) or not last_buttons[i])
        )
        self._last_buttons = state.buttons
        raw = np.array(
            [float(state.axes.get(axis, 0.0)) for axis in self.axis_order],
            dtype=np.float32,
        )
        normalized = np.clip(raw * self.axis_signs, -1.0, 1.0)
        normalized[np.abs(normalized) < self.deadzone] = 0.0
        xyz = normalized[:3]
        takeover = self._button_pressed(state.buttons, self.takeover_button)
        gripper_open = self._button_pressed(state.buttons, self.open_gripper_button)
        gripper_close = self._button_pressed(state.buttons, self.close_gripper_button)
        moving = bool(np.linalg.norm(xyz) > 0.0)
        active = takeover or gripper_open or gripper_close
        if self.require_takeover_button:
            active = active and takeover
            xyz = xyz if takeover else np.zeros(3, dtype=np.float32)
        else:
            active = active or moving

        action = self._base_action(obs)
        action[f"{self.control_side}_ee_pos"] = (xyz * self.delta_max * self.xyz_scale).astype(
            np.float32
        )
        if gripper_open:
            action[f"{self.control_side}_gripper_pos"] = np.array(
                [self.gripper_open_pos], dtype=np.float32
            )
        elif gripper_close:
            action[f"{self.control_side}_gripper_pos"] = np.array(
                [self.gripper_close_pos], dtype=np.float32
            )
        action["__action_t"] = time.time()

        info = {
            "source": "human",
            "active": active,
            "takeover": takeover,
            "buttons": np.asarray(state.buttons, dtype=np.float32),
            "just_pressed_buttons": just_pressed,
            "motion_norm": float(np.linalg.norm(xyz)),
            "age_s": float(state.age_s),
            "control_side": self.control_side,
            "normalized_action": normalized.copy(),
        }
        return action, info

    def close(self) -> None:
        close = getattr(self.reader, "close", None)
        if close is not None:
            close()

