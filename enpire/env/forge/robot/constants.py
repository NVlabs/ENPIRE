import sys

import numpy as np
from scipy.spatial.transform import Rotation

from enpire.env.forge.robot.station_profiles import active_station_can

# ---------------------------------------------------------------------------
# CAN bus type: gs_usb on macOS (USB-CAN adapters), socketcan on Linux
# ---------------------------------------------------------------------------
CAN_BUSTYPE = "gs_usb" if sys.platform == "darwin" else "socketcan"

# ---------------------------------------------------------------------------
# Arm CAN interfaces
# Linux (socketcan): kernel interface names like "can_leader_l"
# macOS (gs_usb): USB serial number strings — discover with:
#   python -c "from gs_usb.gs_usb import GsUsb
#   for i,d in enumerate(GsUsb.scan()): print(i, d.serial_number)"
# ---------------------------------------------------------------------------
_STATION_CAN = active_station_can()
LEFT_FOLLOWER_CAN_INTERFACE = _STATION_CAN.left_follower
RIGHT_FOLLOWER_CAN_INTERFACE = _STATION_CAN.right_follower
LEFT_LEADER_CAN_INTERFACE = _STATION_CAN.left_leader
RIGHT_LEADER_CAN_INTERFACE = _STATION_CAN.right_leader

# Arm server ports
LEFT_FOLLOWER_PORT = 11333
RIGHT_FOLLOWER_PORT = 11334
LEFT_LEADER_PORT = 11335
RIGHT_LEADER_PORT = 11336
PICO_PORT = 8963

# Fello arm CAN interfaces (reuse leader CAN interfaces)
LEFT_FELLO_CAN_INTERFACE = LEFT_LEADER_CAN_INTERFACE
RIGHT_FELLO_CAN_INTERFACE = RIGHT_LEADER_CAN_INTERFACE

# YAM arm motor configuration
YAM_ARM_MOTOR_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
YAM_ARM_MOTOR_TYPES = ["4340", "4340", "4340", "4310", "4310", "4310"]
YAM_GRIPPER_MOTOR_ID = 0x07
YAM_GRIPPER_MOTOR_TYPE = "4310"  # linear_4310 gripper

# Default PD gains (from i2rt/get_yam_robot)
YAM_ARM_KP = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0]
YAM_ARM_KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
YAM_GRIPPER_KP = 20.0
YAM_GRIPPER_KD = 0.5

# Gripper FORCE_POS control parameters
YAM_GRIPPER_VEL_LIMIT = 30.0  # rad/s (0-100)
YAM_GRIPPER_TORQUE_LIMIT_NM = 3.75  # Nm (4310 T_max=10 Nm)
YAM_GRIPPER_GRAVCOMP_TORQUE_LIMIT_NM = 0.5  # Nm, soft hold without relaxing GPU grasp

# Gripper motor direction: motor_pos → env_pos = SIGN * motor_pos
YAM_GRIPPER_SIGN = -1

# ---------------------------------------------------------------------------
# Safety: max joint velocity (rad/s)
#
# Single limit enforced at both the policy layer and the env layer.
# Per-step delta is computed at runtime:
#   max_delta_per_step = MAX_JOINT_VELOCITY_RAD_S / control_hz
#
# At 30 Hz:  6 / 30 = 0.2 rad/step
# At 60 Hz:  6 / 60 = 0.1 rad/step
# ---------------------------------------------------------------------------
MAX_JOINT_VELOCITY_RAD_S = 6  # rad/s — max safe joint speed for all layers. 6 rad/s  (~344 deg/s)

# Camera resolutions
WRIST_CAM_RESOLUTION = (640, 480)
TOP_CAM_RESOLUTION = (1280, 720)  # it was (640, 480)

# Teleop/RL reset tolerance
DEFAULT_RESET_JOINT_STATE = {
    "left_joint_pos": np.zeros(6, dtype=np.float32),
    "left_gripper_pos": np.zeros(1, dtype=np.float32),
    "right_joint_pos": np.zeros(6, dtype=np.float32),
    "right_gripper_pos": np.zeros(1, dtype=np.float32),
}
RESET_TARGET_EE_POSE_DEFAULT_POSITION_TOLERANCE_M = 0.03
RESET_TARGET_EE_POSE_DEFAULT_QUAT_TOLERANCE = 0.20
RESET_TARGET_EE_POSE_DEFAULT_MAX_IK_ITERS = 200

POINTING_DOWN_EULER_XYZ_DEG = (180.0, 0.0, -90.0)
POINTING_DOWN_QUAT_XYZW = (
    Rotation.from_euler("xyz", POINTING_DOWN_EULER_XYZ_DEG, degrees=True)
    .as_quat()
    .astype(np.float32)
)

# Hover pose
HOVER_EE_POSE = {
    "left": {
        "position": [0.420, -0.057, 0.85],
        "rpy_deg": [-150.94987, -60.692646, -137.13707],
        "gripper_pos": [0.0],
    },
    "right": {
        "position": [0.45, -0.1, 0.88],
        "rpy_deg": [180.0, 0.0, -90.0],
        "gripper_pos": [0.0],
    },
}

HOVER_JOINT_POSE = {
    "left_joint_pos": np.array(
        [-0.8642, 1.6211, 0.5941, 0.5278, 0.6838, -1.0813], dtype=np.float32
    ),
    "left_gripper_pos": np.zeros(1, dtype=np.float32),
    "right_joint_pos": np.array([0.8623, 1.5463, 1.3117, -1.2695, 0.064, 1.0366], dtype=np.float32),
    "right_gripper_pos": np.zeros(1, dtype=np.float32),
}

# Video compression
DEFAULT_COMPRESSED_VIDEO_SHAPE = (256, 256)
