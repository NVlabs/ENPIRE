# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import struct
import threading
import time
import fcntl
from pathlib import Path
from select import select
from typing import Protocol, runtime_checkable

EV_KEY = 0x01
_EVENT_FORMAT = "llHHI"
_EVENT_SIZE = struct.calcsize(_EVENT_FORMAT)
_EVIOCGRAB = 0x40044590


@runtime_checkable
class ButtonMonitor(Protocol):
    """Common interface for all footswitch / button monitor backends."""

    def is_pressed(self) -> bool: ...

    @property
    def device_path(self) -> str: ...

    def close(self) -> None: ...


class FootSwitchMonitor:
    def __init__(
        self,
        device_path: str | None = None,
        vendor_id: str = "3553",
        product_id: str = "b001",
        grab_device: bool = True,
        min_hold_seconds: float = 0.1,
    ) -> None:
        vendor_id = vendor_id.lower().replace("0x", "")
        product_id = product_id.lower().replace("0x", "")
        self._device_path = device_path or self._find_device(vendor_id, product_id)
        if self._device_path is None:
            raise FileNotFoundError(
                "Footswitch input device not found. "
                "Pass device_path or ensure /dev/input/by-id has the footswitch."
            )

        self._grab_fds: list[int] = []
        self._fd = os.open(self._device_path, os.O_RDONLY | os.O_NONBLOCK)
        self._grabbed = False
        if grab_device:
            try:
                fcntl.ioctl(self._fd, _EVIOCGRAB, 1)
                self._grabbed = True
            except OSError:
                print(
                    f"[footswitch] WARNING: failed to grab {self._device_path}. "
                    "Keypresses may reach the terminal; try running with sudo or adjust udev rules."
                )
            self._grab_related_devices()
        self._pressed_codes: set[int] = set()
        self._pressed = False
        self._min_hold_seconds = min_hold_seconds
        self._press_time = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _find_device(self, vendor_id: str, product_id: str) -> str | None:
        by_id = Path("/dev/input/by-id")
        if by_id.exists():
            preferred = None
            keyboard_fallback = None
            for entry in sorted(by_id.iterdir()):
                name = entry.name.lower()
                if "event" not in name:
                    continue
                if vendor_id in name and product_id in name:
                    resolved = str(entry.resolve())
                    if "footswitch" in name and "keyboard" not in name and "mouse" not in name:
                        return resolved
                    if "keyboard" in name:
                        keyboard_fallback = resolved
                    elif preferred is None:
                        preferred = resolved
            if preferred is not None:
                return preferred
            if keyboard_fallback is not None:
                return keyboard_fallback

        sys_input = Path("/sys/class/input")
        for event in sorted(sys_input.glob("event*")):
            try:
                vendor = (event / "device/id/vendor").read_text().strip().lower()
                product = (event / "device/id/product").read_text().strip().lower()
            except FileNotFoundError:
                continue
            if vendor == vendor_id and product == product_id:
                return str(Path("/dev/input") / event.name)

        return None

    def _grab_related_devices(self) -> None:
        try:
            target = Path(self._device_path)
            event_name = target.name
        except Exception:
            return

        sys_input = Path("/sys/class/input")
        id_dir = sys_input / event_name / "device/id"
        try:
            vendor = (id_dir / "vendor").read_text().strip().lower()
            product = (id_dir / "product").read_text().strip().lower()
        except FileNotFoundError:
            return

        for event in sorted(sys_input.glob("event*")):
            try:
                ev_vendor = (event / "device/id/vendor").read_text().strip().lower()
                ev_product = (event / "device/id/product").read_text().strip().lower()
            except FileNotFoundError:
                continue
            if ev_vendor != vendor or ev_product != product:
                continue
            dev_path = Path("/dev/input") / event.name
            if str(dev_path) == self._device_path:
                continue
            try:
                fd = os.open(str(dev_path), os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                fcntl.ioctl(fd, _EVIOCGRAB, 1)
                self._grab_fds.append(fd)
            except OSError:
                os.close(fd)

    @property
    def device_path(self) -> str:
        return self._device_path

    def _poll_loop(self) -> None:
        while self._running:
            # Expire the latch if the minimum hold time has passed and no
            # physical key is still held down.
            with self._lock:
                if (
                    self._pressed
                    and not self._pressed_codes
                    and time.monotonic() - self._press_time >= self._min_hold_seconds
                ):
                    self._pressed = False

            rlist, _, _ = select([self._fd], [], [], 0.01)
            if not rlist:
                continue
            try:
                data = os.read(self._fd, _EVENT_SIZE * 64)
            except BlockingIOError:
                continue
            for offset in range(0, len(data), _EVENT_SIZE):
                chunk = data[offset : offset + _EVENT_SIZE]
                if len(chunk) < _EVENT_SIZE:
                    continue
                _, _, ev_type, code, value = struct.unpack(_EVENT_FORMAT, chunk)
                if ev_type != EV_KEY:
                    continue
                if value == 1:
                    self._pressed_codes.add(code)
                elif value == 0:
                    self._pressed_codes.discard(code)
                with self._lock:
                    if self._pressed_codes:
                        if not self._pressed:
                            self._press_time = time.monotonic()
                        self._pressed = True
                    # Don't clear _pressed here; let the latch expiry above handle it
            time.sleep(0.001)

    def is_pressed(self) -> bool:
        with self._lock:
            return self._pressed

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=0.5)
        finally:
            for fd in self._grab_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._grab_fds.clear()
            os.close(self._fd)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class KeyboardFootSwitchMonitor:
    """Monitors a specific keyboard key globally as a footswitch.

    Uses ``pynput`` to capture key events system-wide (regardless of window
    focus).  Key-down latches ``is_pressed()`` to ``True``; key-up clears it.
    A minimum hold time ensures brief taps are visible to the control loop.

    On macOS you must grant Accessibility / Input Monitoring permission to the
    terminal or Python binary the first time.
    """

    _instances: list["KeyboardFootSwitchMonitor"] = []
    _listener_started = False
    _listener_lock = threading.Lock()

    def __init__(self, key: str, min_hold_seconds: float = 0.15) -> None:
        self._key = key.lower()
        self._pressed = False
        self._physically_held = False
        self._press_time = 0.0
        self._min_hold_seconds = min_hold_seconds
        self._lock = threading.Lock()
        KeyboardFootSwitchMonitor._instances.append(self)
        KeyboardFootSwitchMonitor._start_listener()

    @classmethod
    def _start_listener(cls) -> None:
        with cls._listener_lock:
            if cls._listener_started:
                return
            cls._listener_started = True
            from pynput import keyboard

            def _on_press(key):
                try:
                    ch = key.char.lower() if hasattr(key, "char") and key.char else None
                except AttributeError:
                    ch = None
                if ch is None:
                    return
                for monitor in cls._instances:
                    if ch == monitor._key:
                        with monitor._lock:
                            monitor._physically_held = True
                            if not monitor._pressed:
                                monitor._press_time = time.monotonic()
                            monitor._pressed = True

            def _on_release(key):
                try:
                    ch = key.char.lower() if hasattr(key, "char") and key.char else None
                except AttributeError:
                    ch = None
                if ch is None:
                    return
                for monitor in cls._instances:
                    if ch == monitor._key:
                        with monitor._lock:
                            monitor._physically_held = False

            listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
            listener.daemon = True
            listener.start()

    def is_pressed(self) -> bool:
        with self._lock:
            if (
                self._pressed
                and not self._physically_held
                and time.monotonic() - self._press_time >= self._min_hold_seconds
            ):
                self._pressed = False
            return self._pressed

    @property
    def device_path(self) -> str:
        return f"keyboard:{self._key}"

    def close(self) -> None:
        if self in KeyboardFootSwitchMonitor._instances:
            KeyboardFootSwitchMonitor._instances.remove(self)


class SerialButtonHub:
    """Reads a serial device that streams ``[0, 1, 0]``-style JSON lines.

    A single hub is shared by multiple :class:`SerialButtonMonitor` instances
    (one per button index).  The hub runs a daemon thread that continuously
    reads lines, parses them, and updates per-button pressed state with a
    minimum-hold latch so brief taps are visible to the 100 Hz control loop.
    """

    _RP2040_VENDOR = "2e8a"
    _RP2040_PRODUCT = "101f"

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        num_buttons: int = 3,
        min_hold_seconds: float = 0.15,
    ) -> None:
        import serial as _serial

        if port is None:
            port = self._find_serial_device()
        if port is None:
            raise FileNotFoundError(
                "Serial button device not found. "
                "Pass port explicitly or ensure the Waveshare RP2040-Zero is connected."
            )
        self._port = port
        self._num_buttons = num_buttons
        self._min_hold_seconds = min_hold_seconds

        self._pressed = [False] * num_buttons
        self._physically_held = [False] * num_buttons
        self._press_times = [0.0] * num_buttons
        self._lock = threading.Lock()
        self._running = True
        self._ref_count = 0

        self._ser = _serial.Serial(port, baudrate, timeout=1)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    @classmethod
    def _find_serial_device(cls) -> str | None:
        by_id = Path("/dev/serial/by-id")
        if by_id.exists():
            for entry in sorted(by_id.iterdir()):
                name = entry.name.lower()
                if cls._RP2040_VENDOR in name or "rp2040" in name:
                    return str(entry.resolve())

        # Fallback: scan sysfs for the vendor/product pair
        for tty in sorted(Path("/sys/class/tty").glob("ttyACM*")):
            device_dir = tty / "device"
            if not device_dir.exists():
                continue
            try:
                uevent = (device_dir / "../uevent").read_text()
            except (FileNotFoundError, OSError):
                continue
            if cls._RP2040_VENDOR in uevent.lower() and cls._RP2040_PRODUCT in uevent.lower():
                return f"/dev/{tty.name}"
        return None

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline()
            except Exception:
                time.sleep(0.05)
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            try:
                values = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(values, list) or len(values) < self._num_buttons:
                continue

            now = time.monotonic()
            with self._lock:
                for i in range(self._num_buttons):
                    held = bool(values[i])
                    was_held = self._physically_held[i]
                    self._physically_held[i] = held
                    if held and not was_held:
                        self._press_times[i] = now
                        self._pressed[i] = True
                    elif held:
                        self._pressed[i] = True
                    elif not held and was_held:
                        pass  # let latch expiry handle it

        self._ser.close()

    def get_button(self, index: int) -> bool:
        with self._lock:
            if (
                self._pressed[index]
                and not self._physically_held[index]
                and time.monotonic() - self._press_times[index] >= self._min_hold_seconds
            ):
                self._pressed[index] = False
            return self._pressed[index]

    def acquire(self) -> None:
        self._ref_count += 1

    def release(self) -> None:
        self._ref_count -= 1
        if self._ref_count <= 0:
            self.close()

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass

    @property
    def port(self) -> str:
        return self._port


class SerialButtonMonitor:
    """Per-button view into a shared :class:`SerialButtonHub`.

    Implements the same ``is_pressed()`` / ``device_path`` / ``close()``
    interface as :class:`FootSwitchMonitor` and :class:`KeyboardFootSwitchMonitor`.
    """

    def __init__(self, hub: SerialButtonHub, index: int = 0) -> None:
        self._hub = hub
        self._index = index
        hub.acquire()

    def is_pressed(self) -> bool:
        return self._hub.get_button(self._index)

    @property
    def device_path(self) -> str:
        return f"serial:{self._hub.port}[{self._index}]"

    def close(self) -> None:
        self._hub.release()

