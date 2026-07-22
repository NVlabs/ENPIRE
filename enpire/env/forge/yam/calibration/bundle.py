# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation and packaging around the original calibration implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationRecord:
    camera_name: str
    camera_serial: str
    transform: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    translation_rms_mm: float
    rotation_rms_deg: float
    metadata: dict[str, Any] = field(default_factory=dict)


def load_calibration_record(path: str | Path) -> CalibrationRecord:
    """Load the JSON schema emitted by the original yam-calibration pipeline."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    hand_eye = payload["hand_eye"]
    intrinsics = payload["intrinsics"]
    known = {"camera_name", "camera_serial", "hand_eye", "intrinsics"}
    return CalibrationRecord(
        camera_name=str(payload["camera_name"]),
        camera_serial=str(payload["camera_serial"]),
        transform=np.asarray(hand_eye["T_base_from_camera"], dtype=np.float64),
        camera_matrix=np.asarray(intrinsics["camera_matrix"], dtype=np.float64),
        distortion=np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64),
        image_size=tuple(int(value) for value in intrinsics["image_size"]),
        translation_rms_mm=float(hand_eye["translation_rms_mm"]),
        rotation_rms_deg=float(hand_eye["rotation_rms_deg"]),
        metadata={key: value for key, value in payload.items() if key not in known},
    )


def validate_calibration_record(
    record: CalibrationRecord,
    *,
    max_translation_rms_mm: float = 20.0,
    max_rotation_rms_deg: float = 5.0,
) -> tuple[str, ...]:
    """Return deterministic validation errors without touching hardware."""

    errors: list[str] = []
    transform = np.asarray(record.transform, dtype=np.float64)
    matrix = np.asarray(record.camera_matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        errors.append("transform must be 4x4")
    elif not np.all(np.isfinite(transform)):
        errors.append("transform contains non-finite values")
    else:
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            errors.append("transform rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            errors.append("transform rotation determinant is not 1")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            errors.append("transform bottom row is invalid")
    if matrix.shape != (3, 3):
        errors.append("camera_matrix must be 3x3")
    elif matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        errors.append("camera focal lengths must be positive")
    if record.image_size[0] <= 0 or record.image_size[1] <= 0:
        errors.append("image_size must be positive")
    if record.translation_rms_mm > max_translation_rms_mm:
        errors.append("translation residual exceeds threshold")
    if record.rotation_rms_deg > max_rotation_rms_deg:
        errors.append("rotation residual exceeds threshold")
    return tuple(errors)
