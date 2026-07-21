from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CharucoDetector:
    def __init__(
        self,
        squares_x: int,
        squares_y: int,
        square_length: float,
        marker_length: float,
        dictionary: str = "DICT_4X4_50",
    ):
        dict_id = getattr(cv2.aruco, dictionary)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, self.aruco_dict
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        pts = self.board.getChessboardCorners()
        center = np.array(
            [squares_x * square_length / 2, squares_y * square_length / 2, 0.0], np.float32
        )
        self.obj_pts = pts - center

    def detect(self, image: np.ndarray):
        corners, ids, mc, mi = self.detector.detectBoard(image)
        if corners is not None and len(corners) == 0:
            corners = ids = None
        if mc is not None and len(mc) == 0:
            mc = mi = None
        return corners, ids, mc, mi

    def estimate_pose(
        self, corners, ids, K: np.ndarray, dist: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        obj_pts = self.obj_pts[ids.ravel()]
        img_pts = corners.reshape(-1, 1, 2)
        if len(obj_pts) < 4:
            return None, None
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None, None
        return (rvec, tvec) if ok else (None, None)
