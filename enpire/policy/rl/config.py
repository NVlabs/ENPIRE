import os
import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.rl_interface import DEFAULT_REQUEST_TIMEOUT_SECONDS


@dataclass
class DataCollectionConfig:
    display_image: bool = True
    data_saving_path: str | None = None
    run_id: str | None = None
    mode: Literal["learn", "eval"] = "eval"
    handshake_bind_address: str = "tcp://*:1986"
    learner_broadcast_port: int | None = None
    server_address: str | None = None
    control_mode: Literal[
        "joint_position",
        "cartesian_position",
        "delta_joint_position",
        "delta_ee_pose",
        "delta_ee_pose_translation",
    ] = "joint_position"
    policy_control_freq: float = 30.0
    enable_cameras: bool = True
    enabled_camera_names: Tuple[str, ...] = ("left", "right")
    enabled_depth_camera_names: Tuple[str, ...] = ()
    crop_camera_names: Tuple[str, ...] = ()
    crop_region: Tuple[str, ...] = ("center",)
    enabled_sides: str = "right"
    enable_eef_force_observation: bool = False
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF_WRISTONLY
    resolution: Literal[240, 256, 480] = 256
    task_name: str = ""
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    randomize_initial_pose: bool = False
    x_init_lim: Tuple[float, float] = (0.0, 0.0)
    y_init_lim: Tuple[float, float] = (0.0, 0.0)
    z_init_lim: Tuple[float, float] = (0.0, 0.0)
    enable_oor_check: bool = False
    x_oor_lim: Tuple[float, float] = (0.0, 0.0)
    y_oor_lim: Tuple[float, float] = (0.0, 0.0)
    z_oor_lim: Tuple[float, float] = (0.0, 0.0)
    enable_auto_reward: bool = False
    auto_reward_z_threshold: float = 0.0
    visualize_force_axis: Tuple[str, ...] = ("z",)
    visualize_position_axis: Tuple[str, ...] = ("x", "y", "z")
    config_file: str | None = None
    initial_positions_file: str | None = None
    reward_config_file: str | None = None
    station: str | None = None
    fastapi_server_host: str = "127.0.0.1"
    fastapi_server_port: int = 8203
    enable_fastapi_server: bool = True
    restart_key: str = "KEY_F5"
    keyboard_start_key: str = "KEY_S"
    keyboard_home_key: str = "KEY_H"
    keyboard_parking_key: str = "KEY_P"
    keyboard_success_key: str = "KEY_ENTER"
    keyboard_fail_key: str = "KEY_BACKSPACE"
    accept_home_events: bool = True
    enable_author_mode: bool = True
    enable_eval_mode: bool = True
    auto_eval_key: str = "KEY_E"
    auto_eval_episodes_per_hole: int = 50
    auto_eval_num_holes: int | None = None  # None = all holes
    auto_eval_output_yaml: str | None = None
    use_fello: bool = False
    use_spacemouse: bool = False
    spacemouse_device_path: str | None = None
    spacemouse_device_name_substring: str = "SpaceMouse"
    spacemouse_control_side: Literal["auto", "left", "right"] = "auto"
    spacemouse_axis_scale: float = 350.0
    spacemouse_stale_timeout_s: float = 0.25
    spacemouse_deadzone: float = 0.05
    spacemouse_axis_order: Tuple[str, ...] = (
        "y",
        "x",
        "z",
        "pitch",
        "roll",
        "yaw",
    )
    spacemouse_axis_signs: Tuple[float, ...] = (1.0, 1.0, -1.0, 1.0, -1.0, -1.0)
    spacemouse_xyz_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    spacemouse_require_takeover_button: bool = False
    spacemouse_takeover_button: int | None = None
    spacemouse_open_gripper_button: int | None = None
    spacemouse_close_gripper_button: int | None = None
    spacemouse_success_button: int | None = 0
    spacemouse_fail_button: int | None = 1
    spacemouse_gripper_open_pos: float = 1.0
    spacemouse_gripper_close_pos: float = 0.0
    scaled_control_xyz_scale: Tuple[float, float, float] = (0.25, 0.25, 1.0)
    delta_ee_translation_xyz_max: Tuple[float, float, float] = (0.0003, 0.0003, 0.0006)
    right_arm_z_force_limit: float = -0.5
    collision_filter_z_threshold: float = float("inf")
    episode_timeout_s: float = 8.0
    terminal_min_recorded_steps: int = 2
    action_trail_duration_s: float = 0.5
    enable_reset_telemetry: bool = False
    reset_max_joint_velocity: float = 0.5
    demo_collection: bool = False
    startup_reset_alias: Literal["home", "hover", "current"] = "home"
    initial_pose_source: Literal["config", "current_observation"] = "config"
    home_event_reset_target: Literal["home", "hover"] = "home"
    auto_reward_z_drop_m: float = 0.0
    episode_reset_lift_m: float = 0.0
    episode_reset_strategy: Literal["target_pose", "gpu_slot_hover"] = "target_pose"
    # When true, drive the right arm to the canonical GPU-insertion hover pose
    # on every hover reset (i.e. when 's'/start is pressed). The pose is defined
    # in cap/saved_scripts/gpu/right_arm_hover.py and shared by data collection
    # and full-loop inference so the policy sees the same right-arm pose in both.
    enable_right_arm_hover: bool = False
    gpu_slot_hover_camera: str = "top"
    gpu_slot_hover_aux_camera: str = "left_third"
    gpu_slot_hover_aux_prefer_world_pose: bool = True
    gpu_slot_hover_aux_required: bool = False
    gpu_slot_hover_save_artifacts: bool = True
    gpu_success_full_cycle_enabled: bool = False
    gpu_success_full_cycle_request_path: str | None = None


def load_yaml_defaults(path: str) -> DataCollectionConfig:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    known = {f.name for f in fields(DataCollectionConfig)}
    cleaned = {}
    for k, v in data.items():
        if k not in known:
            continue
        cleaned[k] = tuple(v) if isinstance(v, list) else v
    if "embodiment_tag" in cleaned and isinstance(cleaned["embodiment_tag"], str):
        cleaned["embodiment_tag"] = EmbodimentTag[cleaned["embodiment_tag"]]
    cfg = DataCollectionConfig(**cleaned)
    cfg.config_file = str(Path(path).expanduser().resolve())
    if os.environ.get("ENPIRE_RL_INITIAL_POSITIONS"):
        cfg.initial_positions_file = os.environ["ENPIRE_RL_INITIAL_POSITIONS"]
    if os.environ.get("ENPIRE_RL_REWARD_CONFIG"):
        cfg.reward_config_file = os.environ["ENPIRE_RL_REWARD_CONFIG"]
        cfg.enable_auto_reward = True
    if os.environ.get("ENPIRE_YAM_STATION"):
        cfg.station = os.environ["ENPIRE_YAM_STATION"]
    return cfg


def sanitize_for_path(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("._-")
    return cleaned or "unknown"


def get_output_dir(cfg: DataCollectionConfig) -> Path:
    if cfg.data_saving_path is not None:
        base = Path(cfg.data_saving_path).expanduser()
    elif "YAM_RAW_PATH" in os.environ:
        base = Path(os.environ["YAM_RAW_PATH"]).expanduser()
    else:
        raise RuntimeError("Set --data-saving-path or the YAM_RAW_PATH environment variable")
    timestamp = cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M")
    return base / sanitize_for_path(cfg.task_name) / timestamp


def resolve_config_relative_path(path: str | Path, config_file: str | None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and config_file:
        resolved = Path(config_file).expanduser().parent / resolved
    return resolved


def apply_station_reward_config(cfg: DataCollectionConfig) -> None:
    if not cfg.reward_config_file:
        return

    import yaml

    path = resolve_config_relative_path(cfg.reward_config_file, cfg.config_file)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if cfg.station and cfg.station in data:
        station_data = data[cfg.station]
    elif None in data or not cfg.station:
        station_data = data
    else:
        print(
            f"\033[1;33m[WARN] Station '{cfg.station}' not found in {path},"
            f" using yaml-level auto_reward_z_threshold\033[0m"
        )
        return

    if "auto_reward_z_threshold" in station_data:
        cfg.auto_reward_z_threshold = float(station_data["auto_reward_z_threshold"])
        print(
            f"[INFO] Reward config loaded from: {path}"
            + (f" (station: {cfg.station})" if cfg.station else "")
        )
        print(f"[INFO] auto_reward_z_threshold = {cfg.auto_reward_z_threshold}")


def provenance_config_files(cfg: DataCollectionConfig) -> list[Path]:
    paths: list[Path] = []
    if cfg.config_file is not None:
        paths.append(Path(cfg.config_file).expanduser())
    if cfg.initial_positions_file is not None:
        paths.append(resolve_config_relative_path(cfg.initial_positions_file, cfg.config_file))
    if cfg.reward_config_file is not None:
        paths.append(resolve_config_relative_path(cfg.reward_config_file, cfg.config_file))
    return paths
