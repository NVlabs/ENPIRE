# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_board_argv_omits_unset_overrides() -> None:
    from enpire.env.forge.yam.calibration.run import board_argv

    assert board_argv() == []
    assert board_argv(square_length=0.020, marker_length=0.015) == [
        "--square-length",
        "0.02",
        "--marker-length",
        "0.015",
    ]
    assert board_argv(squares_x=7, squares_y=5) == [
        "--squares-x",
        "7",
        "--squares-y",
        "5",
    ]


def test_board_overrides_reach_the_calibrator(monkeypatch) -> None:
    """`station calibrate` must forward board geometry, not silently drop it."""
    from enpire.env.forge.yam.calibration import run as run_module

    seen: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(run_module, "_port_open", lambda port: True)
    monkeypatch.setitem(
        __import__("sys").modules,
        "enpire.env.forge.yam.calibration.calibrator",
        type(
            "M",
            (),
            {"main": staticmethod(fake_main), "__spec__": None, "__name__": "calibrator"},
        ),
    )

    assert (
        cli_main(
            [
                "station",
                "calibrate",
                "--station",
                "unit-test",
                "--camera",
                "top",
                "--confirm-motion",
                "--square-length",
                "0.020",
                "--marker-length",
                "0.015",
            ]
        )
        == 0
    )
    argv = seen["argv"]
    assert "--square-length" in argv and argv[argv.index("--square-length") + 1] == "0.02"
    assert "--marker-length" in argv and argv[argv.index("--marker-length") + 1] == "0.015"


def test_board_overrides_default_to_config(monkeypatch) -> None:
    """Without flags the calibrator keeps its own config defaults."""
    from enpire.env.forge.yam.calibration import run as run_module

    seen: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(run_module, "_port_open", lambda port: True)
    monkeypatch.setitem(
        __import__("sys").modules,
        "enpire.env.forge.yam.calibration.calibrator",
        type(
            "M",
            (),
            {"main": staticmethod(fake_main), "__spec__": None, "__name__": "calibrator"},
        ),
    )

    assert (
        cli_main(
            [
                "station",
                "calibrate",
                "--station",
                "unit-test",
                "--camera",
                "left_wrist",
                "--confirm-motion",
            ]
        )
        == 0
    )
    assert "--square-length" not in seen["argv"]
    assert "--marker-length" not in seen["argv"]


def test_calibrate_all_forwards_board_geometry(monkeypatch) -> None:
    """calibrate-all must pass geometry through to the tmux sequence too."""
    from enpire.env.forge.yam.calibration import launch as launch_module

    seen: dict[str, object] = {}

    def fake_launch(**kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(launch_module, "launch", fake_launch)

    assert (
        cli_main(
            [
                "station",
                "calibrate-all",
                "--station",
                "unit-test",
                "--confirm-motion",
                "--square-length",
                "0.020",
                "--marker-length",
                "0.015",
            ]
        )
        == 0
    )
    assert seen["square_length"] == 0.020
    assert seen["marker_length"] == 0.015
