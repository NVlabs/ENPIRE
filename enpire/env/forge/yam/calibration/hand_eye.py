# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Sample:
    q: np.ndarray
    T_base_from_ee: np.ndarray
    T_ee_from_base: np.ndarray
    T_cam_from_board: np.ndarray


METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def run(samples: list[Sample], eye_in_hand: bool = False) -> dict:
    """Run all hand-eye methods, return best by translation consistency.

    eye_in_hand=False (default): eye-to-hand — camera fixed, board on gripper.
      Passes T_ee_from_base; returns T_base_from_cam.
    eye_in_hand=True: eye-in-hand — camera on gripper, board fixed.
      Passes T_base_from_ee; returns T_ee_from_cam.
    """
    if eye_in_hand:
        R_g2b = [s.T_base_from_ee[:3, :3] for s in samples]
        t_g2b = [s.T_base_from_ee[:3, 3].reshape(3, 1) for s in samples]
    else:
        R_g2b = [s.T_ee_from_base[:3, :3] for s in samples]
        t_g2b = [s.T_ee_from_base[:3, 3].reshape(3, 1) for s in samples]
    R_t2c = [s.T_cam_from_board[:3, :3] for s in samples]
    t_t2c = [s.T_cam_from_board[:3, 3].reshape(3, 1) for s in samples]

    results = {}
    for name, method in METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R.reshape(3, 3)
            T[:3, 3] = t.ravel()

            # eye_to_hand: T_ee_from_base @ T_base_from_cam @ T_cam_from_board = T_ee_from_board (const)
            # eye_in_hand: T_base_from_ee @ T_ee_from_cam  @ T_cam_from_board = T_base_from_board (const)
            if eye_in_hand:
                probe = [s.T_base_from_ee @ T @ s.T_cam_from_board for s in samples]
            else:
                probe = [s.T_ee_from_base @ T @ s.T_cam_from_board for s in samples]
            pos = np.array([M[:3, 3] for M in probe])
            trans_rms = float(np.sqrt(np.mean(np.var(pos, axis=0)))) * 1000

            rots = Rotation.from_matrix([M[:3, :3] for M in probe])
            mean_r = rots.mean()
            rot_rms = float(
                np.sqrt(np.mean([np.degrees((r * mean_r.inv()).magnitude()) ** 2 for r in rots]))
            )

            results[name] = {"T": T, "trans_rms_mm": trans_rms, "rot_rms_deg": rot_rms}
        except cv2.error as e:
            print(f"  [{name}] failed: {e}")

    if not results:
        raise RuntimeError("All hand-eye methods failed.")

    best = min(results, key=lambda k: results[k]["trans_rms_mm"])
    return {"best": best, "all": results}
