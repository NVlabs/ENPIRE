# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time

from evdev import InputDevice, categorize, ecodes


class KeyboardPolicy:
    def __init__(self, device_path=None):
        if device_path is None:
            device_path = os.environ.get("RL_KEYBOARD_DEVICE")
        if not device_path:
            raise RuntimeError(
                "Set RL_KEYBOARD_DEVICE to a station-local logical input alias"
            )
        self.device = InputDevice(device_path)

        self._pressed = set()
        self._just_pressed = set()
        self._lock = threading.Lock()
        self._running = True

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        for event in self.device.read_loop():
            if not self._running:
                break
            if event.type != ecodes.EV_KEY:
                continue
            key_event = categorize(event)
            key_name = key_event.keycode
            if isinstance(key_name, list):
                key_name = key_name[0]
            with self._lock:
                if event.value == 1:  # press
                    self._pressed.add(key_name)
                    self._just_pressed.add(key_name)
                elif event.value == 0:  # release
                    self._pressed.discard(key_name)

    def _is_pressed(self, key_name):
        with self._lock:
            return key_name in self._pressed

    def _consume_just_pressed(self):
        with self._lock:
            just = set(self._just_pressed)
            self._just_pressed.clear()
        return just

    def discard_just_pressed(self, key_names=None):
        """Drop pending edge-triggered key presses without changing held state."""
        with self._lock:
            if key_names is None:
                discarded = set(self._just_pressed)
                self._just_pressed.clear()
                return discarded
            keys = set(key_names)
            discarded = self._just_pressed.intersection(keys)
            self._just_pressed.difference_update(keys)
            return set(discarded)

    def get_action(self):
        just = self._consume_just_pressed()
        info = {
            # held keys
            "s": self._is_pressed("KEY_S"),
            "h": self._is_pressed("KEY_H"),
            "r": self._is_pressed("KEY_R"),
            "z": self._is_pressed("KEY_Z"),
            # held arrow keys (for continuous manual control)
            "up_held": self._is_pressed("KEY_UP"),
            "down_held": self._is_pressed("KEY_DOWN"),
            "left_held": self._is_pressed("KEY_LEFT"),
            "right_held": self._is_pressed("KEY_RIGHT"),
            "pageup_held": self._is_pressed("KEY_PAGEUP"),
            "pagedown_held": self._is_pressed("KEY_PAGEDOWN"),
            # held WASD/RF keys (for init_pose control)
            "w_held": self._is_pressed("KEY_W"),
            "a_held": self._is_pressed("KEY_A"),
            "d_held": self._is_pressed("KEY_D"),
            "f_held": self._is_pressed("KEY_F"),
            # edge-triggered keys
            "i": "KEY_I" in just,
            "t": "KEY_T" in just,
            "enter": "KEY_ENTER" in just,
            "esc": "KEY_ESC" in just,
            "m": "KEY_M" in just,
            "-": "KEY_MINUS" in just,
            "up": "KEY_UP" in just,
            "down": "KEY_DOWN" in just,
            "left": "KEY_LEFT" in just,
            "right": "KEY_RIGHT" in just,
            "pageup": "KEY_PAGEUP" in just,
            "pagedown": "KEY_PAGEDOWN" in just,
            # modifier
            "shift_held": self._is_pressed("KEY_LEFTSHIFT") or self._is_pressed("KEY_RIGHTSHIFT"),
            # Raw set of just-pressed evdev key codes (e.g. {"KEY_H", "KEY_S", "KEY_F5"}).
            # Consumers can do edge-triggered checks for keys not in the fixed dict above.
            "_just_pressed_keys": just,
        }
        return None, info

    def close(self):
        self._running = False
        self.device.close()


if __name__ == "__main__":
    # Replace this with your actual keyboard event device
    policy = KeyboardPolicy()

    try:
        while True:
            action, info = policy.get_action()
            print(f"action={action}, info={info}")
            time.sleep(0.02)  # 50 Hz

    except KeyboardInterrupt:
        policy.close()
        print("KeyboardPolicy stopped.")
