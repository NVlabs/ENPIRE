# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Station model path resolution based on top camera backend.

Usage::

    from enpire.env.forge.robot.models.station.paths import get_station_xml, get_station_urdf
"""

from __future__ import annotations

import functools
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_STATION_DIR = Path(
    os.environ.get("ENPIRE_YAM_MODEL_ROOT", Path(__file__).resolve().parent)
).expanduser()

_XML_MAP = {
    # "realsense": "station.xml",
    "realsense": "station_fello_gripper_with_top_camera.xml",
    # station_zed2itop_calibrated.{xml,urdf} = station_zed2itop.{xml,urdf} with
    # the top_camera_zed2i body's pos+quat replaced by the latest hand-eye
    # calibration. Re-patch via tools/calibration/apply_zed_calibration.py
    # after each calibration run; everything else is identical to the parent.
    "zed": "station_zed2itop_calibrated.xml",
}

_URDF_MAP = {
    # "realsense": "station.urdf",
    "realsense": "station_fello_gripper_with_top_camera.urdf",
    "zed": "station_zed2itop_calibrated.urdf",
}

_CAMERA_FRAME_MAP = {
    "realsense": "top_camera_d405",
    "zed": "top_camera_zed2i",
}

_OPENCV_CALIBRATED_FRAMES = {
    "top_camera_d405",
    "top_camera_left_d405",
    "top_camera_left_d435",
}
_LEFT_FIXED_CAMERA_FRAME_CANDIDATES = (
    "top_camera_left_d435",
    "top_camera_left_d405",
)

_REALSENSE_XML_OVERRIDE_ENV_VARS = (
    "YAM_STATION_CALIBRATED_XML_PATH",
    "YAM_STATION_CALIBRATED_XML",
    "FORGE_STATION_XML",
)
_REALSENSE_URDF_OVERRIDE_ENV_VARS = (
    "FORGE_STATION_URDF",
    "YAM_STATION_CALIBRATED_URDF",
)


def _get_top_camera_backend() -> str:
    """Return normalized top camera backend (``'realsense'`` or ``'zed'``)."""
    raw = os.environ.get("CAP_TOP_CAMERA_BACKEND", "").strip().lower()
    if raw in _XML_MAP:
        return raw

    mapping_str = os.environ.get("CAP_CAMERA_BACKENDS", "").strip()
    if mapping_str:
        for item in mapping_str.split(","):
            if "=" in item:
                name, value = item.split("=", 1)
                if name.strip().lower() == "top":
                    v = value.strip().lower()
                    if v in _XML_MAP:
                        return v

    if _env_path(*_REALSENSE_XML_OVERRIDE_ENV_VARS) is not None:
        return "realsense"

    return "zed"


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser()
    return None


def _get_station_xml_override() -> Path | None:
    if _get_top_camera_backend() != "realsense":
        return None
    override = _env_path(*_REALSENSE_XML_OVERRIDE_ENV_VARS)
    if override is not None:
        return override
    return None


def _get_station_urdf_override() -> Path | None:
    if _get_top_camera_backend() != "realsense":
        return None
    override = _env_path(*_REALSENSE_URDF_OVERRIDE_ENV_VARS)
    if override is not None:
        return override
    xml_override = _get_station_xml_override()
    if xml_override is not None:
        sibling_urdf = xml_override.with_suffix(".urdf")
        if sibling_urdf.is_file():
            return sibling_urdf
    return None


def get_station_xml() -> Path:
    """Return the station MuJoCo XML path for the current top camera backend."""
    override = _get_station_xml_override()
    if override is not None:
        return override
    return _STATION_DIR / _XML_MAP[_get_top_camera_backend()]


def get_station_urdf() -> Path:
    """Return the station URDF path for the current top camera backend."""
    override = _get_station_urdf_override()
    if override is not None:
        return override
    return _STATION_DIR / _URDF_MAP[_get_top_camera_backend()]


def get_top_camera_frame() -> str:
    """Return the top camera frame/body name for the current backend."""
    return _CAMERA_FRAME_MAP[_get_top_camera_backend()]


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_vec(attr: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not attr:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in attr.split()], dtype=np.float64)


def _find_body_world_pose(
    body: ET.Element,
    target_name: str,
    parent_pos: np.ndarray,
    parent_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    local_pos = _parse_vec(body.get("pos"), (0.0, 0.0, 0.0))
    local_quat = _parse_vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    local_rot = _quat_wxyz_to_rotmat(local_quat)
    world_pos = parent_pos + parent_rot @ local_pos
    world_rot = parent_rot @ local_rot
    if body.get("name") == target_name:
        return world_pos, world_rot
    for child in body.findall("body"):
        found = _find_body_world_pose(child, target_name, world_pos, world_rot)
        if found is not None:
            return found
    return None


@functools.lru_cache(maxsize=8)
def _xml_body_world_pose(xml_path_str: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    xml_path = Path(xml_path_str)
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"Station XML has no <worldbody>: {xml_path}")
    parent_pos = np.zeros(3, dtype=np.float64)
    parent_rot = np.eye(3, dtype=np.float64)
    for body in worldbody.findall("body"):
        found = _find_body_world_pose(body, body_name, parent_pos, parent_rot)
        if found is not None:
            return found
    raise RuntimeError(f"Body {body_name!r} not found in station XML: {xml_path}")


def get_camera_extrinsics_override(camera: str) -> dict | None:
    """Return calibrated static extrinsics override when an XML override is active."""
    if _get_top_camera_backend() != "realsense":
        return None
    xml_override = _get_station_xml_override()
    if xml_override is None:
        return None

    body_names: tuple[str, ...]
    camera = str(camera).strip().lower()

    if camera == "top":
        body_names = (os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame()),)
    elif camera in {"left_fixed", "left_third"}:
        env_body_name = os.environ.get("CAP_LEFT_THIRD_CAMERA_FRAME", "").strip()
        if env_body_name:
            if env_body_name not in _OPENCV_CALIBRATED_FRAMES:
                return None
            body_names = (env_body_name,)
        else:
            body_names = _LEFT_FIXED_CAMERA_FRAME_CANDIDATES
    else:
        return None

    for body_name in body_names:
        try:
            position, rotation = _xml_body_world_pose(str(xml_override.resolve()), body_name)
        except RuntimeError:
            if camera == "top":
                raise
            continue

        return {
            "position": position.tolist(),
            "rotation": rotation.tolist(),
            "needs_optical_flip": body_name not in _OPENCV_CALIBRATED_FRAMES,
        }

    return None


def needs_optical_flip(camera: str) -> bool:
    """Whether the pinocchio frame for *camera* needs an OpenCV ↔ Pinocchio flip.

    The D405 URDF body frame uses Pinocchio convention (+X left, +Y up,
    +Z forward).  Converting to OpenCV (+X right, +Y down, +Z forward)
    requires ``R @ diag(-1, -1, 1)``.

    The ZED 2i URDF frame was calibrated directly in OpenCV convention,
    so no flip is needed.

    Wrist cameras (left/right) are D405 Pinocchio frames and need the flip.
    Fixed left-side calibration frames are written directly in OpenCV
    convention, matching the top-camera calibration.
    """
    camera = str(camera).strip().lower()
    if camera == "top":
        # Both public station XML variants use calibrated OpenCV convention.
        return False
    if camera in {"left_fixed", "left_third"}:
        default_frame = "top_camera_left_d435"
        frame_name = os.environ.get(
            "CAP_LEFT_THIRD_CAMERA_FRAME",
            default_frame,
        )
        if frame_name in _OPENCV_CALIBRATED_FRAMES:
            return False
    return True
