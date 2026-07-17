from __future__ import annotations

import json

import numpy as np
import pytest

from enpire.cli import main as cli_main
from enpire.env.forge.yam.calibration import calibrator, server


def test_calibrator_requires_motion_confirmation_before_hardware() -> None:
    with pytest.raises(RuntimeError, match="--confirm-motion"):
        calibrator.main(["--camera", "top", "--no-interactive"])


def test_calibration_server_requires_confirmation_before_connecting() -> None:
    with pytest.raises(RuntimeError, match="--confirm-motion"):
        server.main(["--side", "left"])


def test_validate_calibration_cli_is_hardware_free(tmp_path, capsys) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "camera_name": "top_camera",
                "camera_serial": "external-device-id",
                "hand_eye": {
                    "T_base_from_camera": np.eye(4).tolist(),
                    "translation_rms_mm": 1.0,
                    "rotation_rms_deg": 0.1,
                },
                "intrinsics": {
                    "camera_matrix": np.eye(3).tolist(),
                    "dist_coeffs": [0.0] * 5,
                    "image_size": [640, 480],
                },
            }
        )
    )

    assert cli_main(["station", "validate-calibration", str(calibration)]) == 0
    assert "PASS top_camera" in capsys.readouterr().out
