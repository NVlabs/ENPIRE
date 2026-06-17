from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot.yam.yam_real_env import NonBlockingZed


class TestNonBlockingZedResolver(unittest.TestCase):
    def test_resolve_device_matches_symlink_target(self) -> None:
        devices = [
            SimpleNamespace(path="/dev/video8", serial_number=111, camera_state="AVAILABLE"),
            SimpleNamespace(path="/dev/video12", serial_number=38531109, camera_state="AVAILABLE"),
        ]

        def fake_realpath(path: str) -> str:
            return {
                "/dev/video_top_zed2i": "/dev/video12",
                "/dev/video8": "/dev/video8",
                "/dev/video12": "/dev/video12",
            }.get(path, path)

        with patch("robot.yam.yam_real_env.os.path.exists", return_value=True), patch(
            "robot.yam.yam_real_env.os.path.realpath",
            side_effect=fake_realpath,
        ):
            device = NonBlockingZed._resolve_device("/dev/video_top_zed2i", devices)

        self.assertEqual(device.serial_number, 38531109)

    def test_resolve_device_falls_back_when_only_one_camera_exists(self) -> None:
        devices = [
            SimpleNamespace(path="/dev/video12", serial_number=38531109, camera_state="AVAILABLE"),
        ]

        with patch("robot.yam.yam_real_env.os.path.exists", return_value=False):
            device = NonBlockingZed._resolve_device("/dev/video_top_zed2i", devices)

        self.assertEqual(device.serial_number, 38531109)


if __name__ == "__main__":
    unittest.main()
