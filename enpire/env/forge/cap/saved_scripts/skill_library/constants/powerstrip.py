# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Powerstrip/adapter task constants preserved from reset_powerstrip scripts."""

ADAPTER_OBJ = "dark adapter"
ADAPTER_BLOCK_OBJ = "dark adapter block"
ADAPTER_QUERIES = (
    "dark adapter",
    "adapter block",
    "dark and yellow adapter",
    "yellow adapter",
    "yellow block",
)
PRONG_QUERY = "prongs on the adapter"
STRIP_OBJ = "white power strip"
CORD_OBJ = "white power cable on power strip"

CENTER_XY = [0.50, 0.00]
LEFT_COMFORT = {"x": (0.60, 0.70), "y": (0.05, 0.30)}
RIGHT_COMFORT = {"x": (0.60, 0.70), "y": (-0.30, -0.05)}
LAND_Z = 0.80

STRIP_GRASP_Z_M = 0.80
ADAPTER_GRASP_Z_M = 0.85
ADAPTER_TCP_OFFSET_Z_M = 0.0
STRIP_TILT_DEG = 30.0
STRIP_TILT_DIR = +1
STRIP_TILT_FALLBACKS = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
STRIP_HALF_LEN_M = 0.10
STRIP_MAX_GRASPS = 32
STRIP_REORIENT_FRACTIONS = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, -0.20]
STRIP_UNPLUG_FRACTIONS = [-0.45, -0.35, -0.25, -0.15, -0.05, -0.25, -0.20, -0.30]
MIN_ASPECT_RATIO = 1.5

BATCH_TOP_K = 16
ADAPTER_BATCH_K = 32
COMFORT_BATCH_TOP_K = 32
REORIENT_BATCH_K = 32
HOVER_BATCH_TOP_K = 64
TILT_BATCH_TOP_K = 64
DEFAULT_MAX_RETRIES = 999
PLANNING_SPEED = 1.5
UNPLUG_PLANNING_SPEED = 1.0
ALIGN_PLANNING_SPEED = 1.0
IK_ERROR_THRESHOLD_M = 0.01
IK_XYZ_WEIGHT = 1.0
IK_RPY_WEIGHT = 0.3
PLANNER_BACKEND = "curobo"

BATCH_BEST_PROFILE = {
    "batch_top_k": BATCH_TOP_K,
    "planning_speed": PLANNING_SPEED,
    "ik_error_threshold_m": IK_ERROR_THRESHOLD_M,
    "ik_xyz_weight": IK_XYZ_WEIGHT,
    "ik_rpy_weight": IK_RPY_WEIGHT,
}
COMFORT_BATCH_BEST_PROFILE = {**BATCH_BEST_PROFILE, "batch_top_k": COMFORT_BATCH_TOP_K}
ALIGN_BATCH_BEST_PROFILE = {**BATCH_BEST_PROFILE, "planning_speed": ALIGN_PLANNING_SPEED}
UNPLUG_BATCH_BEST_PROFILE = {**BATCH_BEST_PROFILE, "planning_speed": UNPLUG_PLANNING_SPEED}
BATCH_CHUNK_PROFILE = {
    "planning_speed": PLANNING_SPEED,
    "ik_error_threshold_m": IK_ERROR_THRESHOLD_M,
    "ik_xyz_weight": IK_XYZ_WEIGHT,
    "ik_rpy_weight": IK_RPY_WEIGHT,
}
FREESPACE_PROFILE = {
    "planning_speed": PLANNING_SPEED,
    "ik_error_threshold": IK_ERROR_THRESHOLD_M,
    "ik_xyz_weight": IK_XYZ_WEIGHT,
    "ik_rpy_weight": IK_RPY_WEIGHT,
    "planner_backend": PLANNER_BACKEND,
}
ALIGN_FREESPACE_PROFILE = {**FREESPACE_PROFILE, "planning_speed": ALIGN_PLANNING_SPEED}
UNPLUG_FREESPACE_PROFILE = {**FREESPACE_PROFILE, "planning_speed": UNPLUG_PLANNING_SPEED}
BEV_OPEN_GRIPPERS_PROFILE = {
    "planning_speed": PLANNING_SPEED,
    "left_gripper_width": 1.0,
    "right_gripper_width": 1.0,
}

GRIPPER_WIDTH_M = 0.08
GRIP_TIGHT_M = 0.005
GRIP_POLL_SECS = 0.6
GRIP_POLL_STEPS = 6
GRIP_PROFILE = {
    "grip_tight_m": GRIP_TIGHT_M,
    "poll_secs": GRIP_POLL_SECS,
    "poll_steps": GRIP_POLL_STEPS,
}
GRIP_TIGHT_PROFILE = {
    "threshold": GRIP_TIGHT_M,
    "poll_secs": GRIP_POLL_SECS,
    "poll_steps": GRIP_POLL_STEPS,
}

ADAPTER_TOPDOWN_MAX_GRASPS = 16
ADAPTER_ANYGRASP_MAX_GRASPS = 16
PRONG_MAX_GRASPS = 10
ADAPTER_DEBUG_MAX_GRASPS = 4
CORD_MAX_GRASPS = 8

HOVER_CLEARANCE_M = 0.15
HOVER_CLEARANCE_CANDIDATES_M = (0.10, 0.15, 0.20, 0.25)
LOCAL_REORIENT_LIMIT = 3
LIFT_HEIGHT_M = 0.10
J6_SPEED_DEG_S = 50.0
J6_LIMIT_RAD = 2.094
KEYPOINT_SPACING_DEG = 1.0
TARGET_LO = 160.0
TARGET_HI = 180.0
TOLERANCE_DEG = 5.0
SAFE_REL_TOL = 0.12
SAFE_ABS_TOL_M = 0.004
SAFE_FLIP_MAX_ATTEMPTS = 2

COMFORT_CENTER_ZONE = {"x": (0.40, 0.60), "y": (-0.10, 0.10)}
PLACE_POS_LEFT = [0.55, 0.05, 0.82]
PLACE_POS_RIGHT = [0.55, -0.05, 0.82]
UNPLUG_PLACE_POS_LEFT = [0.60, -0.05, 0.82]
UNPLUG_PLACE_POS_RIGHT = [0.60, 0.05, 0.82]
PLACE_YAW_LEFT = 90.0
PLACE_YAW_RIGHT = -90.0
PLACE_PITCH = 0.0
PLACE_ROLL_LEFT = -45.0
PLACE_ROLL_RIGHT = 45.0

HOVER_Z_M = 0.90
SOCKET_HOVER_CLEARANCE_M = 0.08
SOCKET_HOVER_MAX_GAP_M = 0.085
HANDOVER_X = 0.50
HANDOVER_Z = 1.05
HANDOVER_Y_OFFSET = 0.04
LEFT_HANDOVER_RPY = [180.0, -90.0, -90.0]
RIGHT_HANDOVER_RPY = [0.0, 90.0, -90.0]

WOBBLE_DURATION_S = 4.0
WOBBLE_STEPS = 120
WOBBLE_CYCLES = 12
WOBBLE_AMPLITUDE_M = 0.010
WOBBLE_PITCH_DEG = 2.5
WOBBLE_LIFT_M = 0.20
UNPLUG_WOBBLE_PROFILE = {
    "duration_s": WOBBLE_DURATION_S,
    "steps": WOBBLE_STEPS,
    "cycles": WOBBLE_CYCLES,
    "amplitude_m": WOBBLE_AMPLITUDE_M,
    "lift_m": WOBBLE_LIFT_M,
    "pitch_deg": WOBBLE_PITCH_DEG,
}
MAX_WOBBLES_PER_ROUND = 5
DEFAULT_GRIPPER_TORQUE_NM = 0.75
UNPLUG_TORQUE_NM = 1.50
ADAPTER_TORQUE_NM = 1.00
UNPLUG_LIFT_THRESHOLD_M = 0.05
RETREAT_M = 0.10
ADAPTER_GRASP_XY_OFFSETS_M = (
    (0.0, 0.0),
    (0.003, 0.0),
    (-0.003, 0.0),
    (0.0, 0.003),
    (0.0, -0.003),
    (0.003, 0.003),
    (-0.003, -0.003),
    (-0.003, 0.003),
)
STRIP_UNPLUG_SCORE_TARGET = -0.25
STRIP_UNPLUG_GRASP_PROFILE = {
    "half_len": STRIP_HALF_LEN_M,
    "fractions": STRIP_UNPLUG_FRACTIONS,
    "tilt_dir": STRIP_TILT_DIR,
    "primary_tilt": STRIP_TILT_DEG,
    "fallback_tilts": STRIP_TILT_FALLBACKS,
    "score_target": STRIP_UNPLUG_SCORE_TARGET,
    "grasp_z": STRIP_GRASP_Z_M,
    "gripper_width_m": GRIPPER_WIDTH_M,
}
UNPLUG_STRIP_REORIENT_PROFILE = {
    "place_pos_left": UNPLUG_PLACE_POS_LEFT,
    "place_pos_right": UNPLUG_PLACE_POS_RIGHT,
    "place_yaw_left": PLACE_YAW_LEFT,
    "place_yaw_right": PLACE_YAW_RIGHT,
    "place_pitch": PLACE_PITCH,
    "place_roll_left": PLACE_ROLL_LEFT,
    "place_roll_right": PLACE_ROLL_RIGHT,
    "reorient_batch_k": REORIENT_BATCH_K,
    "gripper_width_m": GRIPPER_WIDTH_M,
    "nonfatal": True,
}
UNPLUG_ADAPTER_GRASP_PROFILE = {
    "adapter_obj": ADAPTER_OBJ,
    "tcp_offset_z_m": ADAPTER_TCP_OFFSET_Z_M,
    "gripper_width_m": GRIPPER_WIDTH_M,
    "xy_offsets": ADAPTER_GRASP_XY_OFFSETS_M,
    "adapter_batch_k": ADAPTER_BATCH_K,
}

MAX_ROUNDS = DEFAULT_MAX_RETRIES
