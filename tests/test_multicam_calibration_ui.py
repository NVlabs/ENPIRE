from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from raiden.calibration.multicam import MultiCameraCalibrationUI


class _FakeCamera:
    def __init__(self, grab_results: list[bool], image_size: tuple[int, int] = (640, 480)) -> None:
        self._grab_results = list(grab_results)
        self._image_size = image_size
        self.closed = False

    def open(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def get_intrinsics(self):
        return np.eye(3, dtype=np.float64), np.zeros(5, dtype=np.float64), self._image_size

    def grab(self) -> bool:
        if self._grab_results:
            return self._grab_results.pop(0)
        return False

    def get_frame(self):
        raise AssertionError("get_frame() should not be called in this test")


class TestMultiCameraCalibrationUI(unittest.TestCase):
    def _make_ui(self) -> MultiCameraCalibrationUI:
        ui = object.__new__(MultiCameraCalibrationUI)
        ui.camera_names = ["left_d405", "top_zed"]
        ui.camera_config = MagicMock()
        ui.output_file = None
        ui.reference_camera = "top_zed"
        ui.board_config = None
        ui.calibrator = None
        ui.warmup_frames = 2
        ui.min_captures = 8
        ui.panel_width = 640
        ui.cameras = {}
        ui.intrinsics = {}
        ui.captures = []
        ui.capture_metrics = []

        camera_types = {"left_d405": "realsense", "top_zed": "zed"}
        serials = {"left_d405": "000000000021", "top_zed": 10000001}
        ui.camera_config.get_camera_type.side_effect = camera_types.__getitem__
        ui.camera_config.get_serial_by_name.side_effect = serials.__getitem__
        return ui

    def test_initialize_cameras_fails_fast_when_camera_never_warms_up(self) -> None:
        ui = self._make_ui()
        left_camera = _FakeCamera([False, False])
        top_camera = _FakeCamera([True, True])
        ui.camera_config.create_camera.side_effect = {
            "left_d405": left_camera,
            "top_zed": top_camera,
        }.__getitem__

        with self.assertRaisesRegex(RuntimeError, "No frames received during warmup from: left_d405"):
            ui.initialize_cameras()

        self.assertTrue(left_camera.closed)
        self.assertTrue(top_camera.closed)
        self.assertEqual(ui.cameras, {})

    def test_detect_frame_returns_error_panel_when_grab_fails(self) -> None:
        ui = self._make_ui()
        ui.cameras = {"left_d405": _FakeCamera([False])}
        ui.intrinsics = {
            "left_d405": (
                np.eye(3, dtype=np.float64),
                np.zeros(5, dtype=np.float64),
                (640, 480),
            )
        }

        detection = ui._detect_frame("left_d405")

        self.assertIsNone(detection.board_to_camera)
        self.assertIn("Failed to grab frame from left_d405", detection.error or "")
        self.assertEqual(detection.image.shape, (480, 640, 3))
        self.assertEqual(detection.num_charuco, 0)
        self.assertEqual(detection.num_markers, 0)


if __name__ == "__main__":
    unittest.main()
