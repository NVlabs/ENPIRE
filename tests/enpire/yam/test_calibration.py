from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from enpire.env.forge.yam.calibration.bundle import (
    CalibrationRecord,
    load_calibration_record,
    validate_calibration_record,
)
from enpire.env.forge.yam.calibration.xml_writer import (
    inject_camera,
    update_body_extrinsic,
)


def _record(**overrides) -> CalibrationRecord:
    values = {
        "camera_name": "top",
        "camera_serial": "TEST-CAMERA",
        "transform": np.eye(4),
        "camera_matrix": np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]]),
        "distortion": np.zeros(5),
        "image_size": (640, 480),
        "translation_rms_mm": 1.0,
        "rotation_rms_deg": 0.1,
    }
    values.update(overrides)
    return CalibrationRecord(**values)


def test_valid_calibration_record():
    assert validate_calibration_record(_record()) == ()


def test_calibration_validation_reports_geometry_and_quality_failures():
    bad_transform = np.eye(4)
    bad_transform[0, 0] = 2.0
    errors = validate_calibration_record(
        _record(
            transform=bad_transform,
            translation_rms_mm=21.0,
            rotation_rms_deg=6.0,
        )
    )
    assert "transform rotation is not orthonormal" in errors
    assert "transform rotation determinant is not 1" in errors
    assert "translation residual exceeds threshold" in errors
    assert "rotation residual exceeds threshold" in errors


def test_original_xml_writer_updates_existing_body(tmp_path):
    source = tmp_path / "station.xml"
    target = tmp_path / "calibrated.xml"
    source.write_text("<mujoco><asset/><worldbody><body name='top_camera'/></worldbody></mujoco>")
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]

    update_body_extrinsic(str(source), "top_camera", transform, str(target))

    body = ET.parse(target).find("worldbody/body")
    assert body is not None
    assert body.get("pos") == "0.10000000 -0.20000000 0.30000000"
    assert body.get("quat") == "1.00000000 0.00000000 0.00000000 0.00000000"


def test_original_xml_writer_injects_camera(tmp_path):
    source = tmp_path / "base.xml"
    target = tmp_path / "calibrated.xml"
    source.write_text("<mujoco><asset/><worldbody/></mujoco>")
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]])

    inject_camera(
        str(source),
        "top_camera",
        np.eye(4),
        intrinsics,
        np.zeros(5),
        (640, 480),
        str(target),
        camera_mesh_file="d405.stl",
    )

    root = ET.parse(target).getroot()
    body = root.find("worldbody/body")
    camera = body.find("camera") if body is not None else None
    assert body is not None and body.get("name") == "top_camera"
    assert camera is not None and camera.get("resolution") == "640 480"


def test_original_xml_writer_fails_for_unknown_body(tmp_path):
    source = tmp_path / "station.xml"
    source.write_text("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="not found"):
        update_body_extrinsic(str(source), "missing", np.eye(4), str(tmp_path / "out.xml"))


def test_loads_original_calibration_json_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        """{
          "camera_name": "top_camera",
          "camera_serial": "external-device-id",
          "hand_eye": {
            "T_base_from_camera": [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]],
            "translation_rms_mm": 1.2,
            "rotation_rms_deg": 0.3
          },
          "intrinsics": {
            "camera_matrix": [[500,0,320],[0,500,240],[0,0,1]],
            "dist_coeffs": [0,0,0,0,0],
            "image_size": [640,480]
          },
          "timestamp": "2026-01-01T00:00:00"
        }"""
    )

    record = load_calibration_record(path)

    assert record.camera_name == "top_camera"
    assert record.transform[2, 3] == pytest.approx(0.3)
    assert record.metadata["timestamp"] == "2026-01-01T00:00:00"
