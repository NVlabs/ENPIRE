"""Run profiles for saved sorting scripts."""

from .manipulation import (
    MAX_GRASP_ATTEMPTS,
    TWO_D_TOP_DOWN_Z_M,
)
from .robot import (
    HOME_VIEW_Z_OFFSET,
    LOCAL_LEFT_TARGET_POS,
    LOCAL_LEFT_TARGET_RPY,
    LOCAL_PLANNING_SPEED,
    LOCAL_RIGHT_TARGET_POS,
    LOCAL_RIGHT_TARGET_RPY,
    LOCAL_IK_ERROR_THRESHOLD_M,
    LOCAL_IK_RPY_WEIGHT,
    LOCAL_IK_XYZ_WEIGHT,
    LEFT_BIRDEYE_VIEW_RPY,
    LEFT_HOME_XYZ,
    RIGHT_BIRDEYE_VIEW_RPY,
    RIGHT_HOME_XYZ,
)
from .planning import (
    BATCH_SOLVER_SPEED,
    BATCH_TOP_K,
    BATCH_VALIDATE_TRAJECTORY,
    IK_ERROR_THRESHOLD_M,
    IK_RPY_WEIGHT,
    IK_XYZ_WEIGHT,
    MOTION_PLANNER_BACKEND,
    PLANNING_SPEED,
)
from .vision import (
    ANYGRASP_DISABLE_PLANNER_Z_CLIPPING,
    ANYGRASP_TCP_OFFSET_Z_M,
    BUNDLESDF_CAMERA,
    DEFAULT_VLM_CAMERAS,
    MAJORITY,
    NUM_VOTES,
    SMALL_OBJECT_TARGET_DROP_Z_OFFSETS,
    TABLE_TARGET_DROP_Z_OFFSETS,
    TOP_GRASP_MAX,
    TOP_GRASP_TRY,
    VLM_BACKEND,
    VLM_MODEL,
)

TABLE_SORT_RUN_CONFIG = dict(
    max_grasps=TOP_GRASP_MAX,
    top_grasp_try=TOP_GRASP_TRY,
    max_attempts=MAX_GRASP_ATTEMPTS,
    batch_top_k=BATCH_TOP_K,
    solver_speed=BATCH_SOLVER_SPEED,
    batch_validate_trajectory=BATCH_VALIDATE_TRAJECTORY,
    planning_speed=PLANNING_SPEED,
    ik_error_threshold=IK_ERROR_THRESHOLD_M,
    ik_xyz_weight=IK_XYZ_WEIGHT,
    ik_rpy_weight=IK_RPY_WEIGHT,
    planner_backend=MOTION_PLANNER_BACKEND,
    bundlesdf_camera=BUNDLESDF_CAMERA,
    target_drop_z_offsets=TABLE_TARGET_DROP_Z_OFFSETS,
    tcp_offset_z_m=ANYGRASP_TCP_OFFSET_Z_M,
    disable_planner_z_clipping=ANYGRASP_DISABLE_PLANNER_Z_CLIPPING,
    left_home_xyz=LEFT_HOME_XYZ,
    right_home_xyz=RIGHT_HOME_XYZ,
    home_view_z_offset=HOME_VIEW_Z_OFFSET,
    left_birdeye_view_rpy=LEFT_BIRDEYE_VIEW_RPY,
    right_birdeye_view_rpy=RIGHT_BIRDEYE_VIEW_RPY,
)

SMALL_OBJECT_SORT_RUN_CONFIG = dict(
    TABLE_SORT_RUN_CONFIG,
    planning_speed=0.5,
    target_drop_z_offsets=SMALL_OBJECT_TARGET_DROP_Z_OFFSETS,
    tcp_offset_z_m=0.0,
    grasp_z_m=TWO_D_TOP_DOWN_Z_M,
    local_left_target_pos=LOCAL_LEFT_TARGET_POS,
    local_left_target_rpy=LOCAL_LEFT_TARGET_RPY,
    local_right_target_pos=LOCAL_RIGHT_TARGET_POS,
    local_right_target_rpy=LOCAL_RIGHT_TARGET_RPY,
    local_planning_speed=LOCAL_PLANNING_SPEED,
    local_ik_error_threshold=LOCAL_IK_ERROR_THRESHOLD_M,
    local_ik_xyz_weight=LOCAL_IK_XYZ_WEIGHT,
    local_ik_rpy_weight=LOCAL_IK_RPY_WEIGHT,
)

SORT_VLM_CONFIG = dict(
    backend=VLM_BACKEND,
    model=VLM_MODEL,
    cameras=DEFAULT_VLM_CAMERAS,
    num_votes=NUM_VOTES,
    majority=MAJORITY,
)
