"""Runtime configuration for the preserved YAM ChArUco pipeline.

Reusable, uncalibrated mechanical models ship with ENPIRE. Calibration
observations and the generated station-specific XML are written beneath
``ENPIRE_CALIBRATION_OUTPUT_ROOT`` or ENPIRE's external data directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from enpire.env.forge.paths import forge_path
from enpire.env.forge.yam.station import enpire_data_home
from enpire.env.forge.robot.constants import (
    LEFT_FOLLOWER_CAN_INTERFACE,
    LEFT_FOLLOWER_PORT,
    RIGHT_FOLLOWER_CAN_INTERFACE,
    RIGHT_FOLLOWER_PORT,
)

CAN_INTERFACE = LEFT_FOLLOWER_CAN_INTERFACE
CAN_INTERFACE_RIGHT = RIGHT_FOLLOWER_CAN_INTERFACE
MOTOR_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
MOTOR_TYPES = ["4340", "4340", "4340", "4310", "4310", "4310"]

ARM_KP = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0]
ARM_KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
GRAVITY_KP = [0.0] * 6
GRAVITY_KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]

ARM_SERVER_PORT = LEFT_FOLLOWER_PORT
ARM_SERVER_PORT_RIGHT = RIGHT_FOLLOWER_PORT

MODEL_ROOT = Path(
    os.environ.get(
        "ENPIRE_YAM_MODEL_ROOT", forge_path("robot", "models", "station")
    )
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("ENPIRE_CALIBRATION_OUTPUT_ROOT", enpire_data_home() / "calibration")
).expanduser()
FK_XML = str(MODEL_ROOT / "station_fello_gripper_without_top_camera.xml")
GRAVITY_COMP_XML = str(MODEL_ROOT / "yam_fk.xml")
FK_BODY = "left_link_6"
FK_JOINT_NAMES = [f"left_joint{i}" for i in range(1, 7)]
FK_BODY_RIGHT = "right_link_6"
FK_JOINT_NAMES_RIGHT = [f"right_joint{i}" for i in range(1, 7)]

CAMERA_SERIAL = os.environ.get("ENPIRE_TOP_CAMERA_DEVICE", "video_top")
WRIST_LEFT_CAMERA_SERIAL = os.environ.get("ENPIRE_LEFT_CAMERA_DEVICE", "video_left")
WRIST_RIGHT_CAMERA_SERIAL = os.environ.get("ENPIRE_RIGHT_CAMERA_DEVICE", "video_right")
CAMERA_FPS = 15
CALIBRATION_RESOLUTION = (640, 480)
TOP_CALIBRATION_RESOLUTION = (1280, 720)

SQUARES_X = 5
SQUARES_Y = 5
SQUARE_LENGTH = 0.040
MARKER_LENGTH = 0.030
DICTIONARY = "DICT_4X4_50"
MIN_SAMPLES = 12

CAMERA_NAME = "top_camera"
CALIB_BODY_NAME = "top_camera_d405"
BASE_XML = str(MODEL_ROOT / "station_fello_gripper_without_top_camera.xml")
OUTPUT_XML = os.environ.get(
    "ENPIRE_YAM_CALIBRATED_XML_OUTPUT",
    str(OUTPUT_ROOT / "station_fello_gripper_with_top_camera.xml"),
)
WRIST_LEFT_BODY_NAME = "left_camera_d405"
WRIST_RIGHT_BODY_NAME = "right_camera_d405"

MAX_VEL = 0.6
INTERP_HZ = 50
SETTLE_TIME = 0.8
HOME_VEL = 0.3

CALIBRATION_POSE_OFFSETS = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.30, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.30, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.15, 0.00, 0.00, 0.00, 0.00],
    [0.00, -0.15, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.15, 0.00, 0.00, 0.00],
    [0.00, 0.00, -0.15, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.30, 0.00, 0.00],
    [0.00, 0.00, 0.00, -0.30, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.30, 0.00],
    [0.00, 0.00, 0.00, 0.00, -0.30, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.60],
    [0.00, 0.00, 0.00, 0.00, 0.00, -0.60],
    [-0.25, 0.00, 0.00, 0.00, 0.00, 0.50],
    [0.25, 0.00, 0.00, 0.00, 0.00, -0.50],
]


def required_model_paths() -> tuple[Path, ...]:
    return (Path(FK_XML), Path(GRAVITY_COMP_XML), Path(BASE_XML))
