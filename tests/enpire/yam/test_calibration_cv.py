# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="install the calibration extra")

import enpire.env.forge.yam.calibration.hand_eye as hand_eye
from enpire.env.forge.yam.calibration.charuco import CharucoDetector
from enpire.env.forge.yam.calibration.hand_eye import Sample, run


def _sample() -> Sample:
    identity = np.eye(4)
    return Sample(
        q=np.zeros(6),
        T_base_from_ee=identity.copy(),
        T_ee_from_base=identity.copy(),
        T_cam_from_board=identity.copy(),
    )


def test_original_hand_eye_result_contract(monkeypatch):
    monkeypatch.setattr(hand_eye, "METHODS", {"characterized": 7})
    monkeypatch.setattr(
        cv2,
        "calibrateHandEye",
        lambda *args, **kwargs: (np.eye(3), np.zeros((3, 1))),
        raising=False,
    )

    result = run([_sample() for _ in range(4)])

    assert result["best"] == "characterized"
    assert np.array_equal(result["all"]["characterized"]["T"], np.eye(4))
    assert result["all"]["characterized"]["trans_rms_mm"] == 0.0
    assert result["all"]["characterized"]["rot_rms_deg"] == 0.0


def test_original_charuco_rejects_too_few_correspondences():
    detector = CharucoDetector(5, 5, 0.04, 0.03)
    corners = np.zeros((3, 1, 2), dtype=np.float32)
    ids = np.arange(3, dtype=np.int32).reshape(-1, 1)

    rvec, tvec = detector.estimate_pose(corners, ids, np.eye(3), np.zeros(5))

    assert rvec is None
    assert tvec is None
