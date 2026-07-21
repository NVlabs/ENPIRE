from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import queue
import shutil

import gymnasium as gym
from gymnasium.envs.registration import register

from enpire.env.forge.experimental.rl_interface import RLInterface
from enpire.policy.rl.record_episode_wrapper import RecordEpisodeWrapper
from enpire.env.forge.robot.fello.fello_teleop_policy import DualFelloPolicy
from enpire.env.forge.robot.keyboard.keyboard_policy import KeyboardPolicy
from enpire.env.forge.robot.spacemouse.spacemouse_policy import SpaceMouseTeleopPolicy
from enpire.env.forge.display_utils import ImageDisplayer

from enpire.policy.rl import fastapi_server
from enpire.policy.rl.timing_log import TimingLogger
from enpire.policy.rl.config import (
    DataCollectionConfig,
    apply_station_reward_config,
    get_output_dir,
    provenance_config_files,
)
from enpire.policy.rl.events import CollisionFilter, RLEventRouter
from enpire.policy.rl.initial_pose_manager import InitialPoseManager
from enpire.policy.rl.learner import OnlineBufferHandshakeServer
from enpire.policy.rl.parking import ParkingNavigator
from enpire.policy.rl.policy import PolicyRouter, build_policy_adapters
from enpire.policy.rl.speech_announcer import SpeechAnnouncer
from enpire.policy.rl.state_machine import RLStateMachine


@dataclass
class RLContext:
    cfg: DataCollectionConfig
    env: object
    policy_router: PolicyRouter
    keyboard_policy: KeyboardPolicy
    event_router: RLEventRouter
    state_machine: RLStateMachine
    initial_pose_manager: InitialPoseManager
    parking_navigator: ParkingNavigator
    collision_filter: CollisionFilter
    speech_announcer: SpeechAnnouncer
    img_queue: queue.Queue | None
    obs: dict
    external_event_queue: "queue.Queue[tuple[str, dict]]"
    fastapi_base_url: str
    timing_log: TimingLogger
    handshake_server: OnlineBufferHandshakeServer | None = None
    terminal_event: str | None = None
    last_terminal_event: str | None = None
    action_clipped: bool = False
    action_trail: list[tuple[float, tuple[float, float, float]]] = field(
        default_factory=list
    )
    pending_pose_event: str | None = None
    pending_pose_payload: dict = field(default_factory=dict)
    author_left_arm_mode: str = "gravity"
    author_saved_initial_positions: list[dict] | None = None
    demo_success_count: int = 0
    demo_total_count: int = 0
    demo_rolling_window: deque[bool] = field(default_factory=lambda: deque(maxlen=20))


def build_context(cfg: DataCollectionConfig) -> RLContext:
    apply_station_reward_config(cfg)
    initial_pose_manager = InitialPoseManager(cfg)
    print(f"[INFO] Active initial pose: {initial_pose_manager.describe_current()}")

    output_dir = get_output_dir(cfg)
    print(f"[INFO] Local output directory: {output_dir}")
    provenance_files = provenance_config_files(cfg)
    if provenance_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in provenance_files:
            if not src.exists():
                print(f"[WARN] Config file not found, skipping save: {src}")
                continue
            dest = output_dir / src.name
            shutil.copy(src, dest)
            print(f"[INFO] Config saved to: {dest}")

    timing_log = TimingLogger(output_dir)

    handshake_server: OnlineBufferHandshakeServer | None = None
    if cfg.mode == "learn":
        handshake_server = OnlineBufferHandshakeServer(
            bind_address=cfg.handshake_bind_address,
            initial_path=output_dir,
        )

    external_event_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
    fastapi_base_url = f"http://{cfg.fastapi_server_host}:{cfg.fastapi_server_port}"
    if getattr(cfg, "enable_fastapi_server", True):
        fastapi_server.start_in_thread(
            host=cfg.fastapi_server_host,
            port=cfg.fastapi_server_port,
            event_queue=external_event_queue,
            data_root=output_dir.parent,
            config_files=provenance_files,
        )

    register(id="YamReal-v0", entry_point="enpire.env.forge.robot.yam.yam_real_env:YamRealEnv")
    print(f"[INFO] Using policy control frequency: {cfg.policy_control_freq:.1f} Hz")
    env = gym.make(
        "YamReal-v0",
        control_mode=cfg.control_mode,
        policy_control_freq=cfg.policy_control_freq,
        enable_cameras=cfg.enable_cameras,
        enabled_camera_names=cfg.enabled_camera_names,
        enabled_depth_camera_names=cfg.enabled_depth_camera_names,
        enabled_sides=cfg.enabled_sides,
        crop_camera_names=cfg.crop_camera_names,
        crop_region=cfg.crop_region,
        enable_eef_force_observation=cfg.enable_eef_force_observation,
        delta_ee_translation_xyz_max=cfg.delta_ee_translation_xyz_max,
        enable_reset_telemetry=cfg.enable_reset_telemetry,
        reset_max_joint_velocity=cfg.reset_max_joint_velocity,
    )

    img_queue = None
    if cfg.display_image:
        img_queue = queue.Queue(maxsize=1)
        ImageDisplayer(img_queue, "RL", size=256).start()

    env = RecordEpisodeWrapper(env, output_dir=str(output_dir))

    fello_policy = (
        DualFelloPolicy(
            action_type="delta_eef",
            scaled_control=True,
            scaled_control_xyz_scale=cfg.scaled_control_xyz_scale,
            delta_ee_translation_xyz_max=cfg.delta_ee_translation_xyz_max,
            enabled_sides=cfg.enabled_sides,
            decouple_translation=True,
            takeover_button=(1, 0),
        )
        if cfg.use_fello
        else None
    )
    spacemouse_policy = None
    if cfg.use_spacemouse:
        spacemouse_policy = SpaceMouseTeleopPolicy(
            enabled_sides=cfg.enabled_sides,
            control_side=cfg.spacemouse_control_side,
            delta_ee_translation_xyz_max=cfg.delta_ee_translation_xyz_max,
            xyz_scale=cfg.spacemouse_xyz_scale,
            deadzone=cfg.spacemouse_deadzone,
            axis_order=cfg.spacemouse_axis_order,
            axis_signs=cfg.spacemouse_axis_signs,
            require_takeover_button=cfg.spacemouse_require_takeover_button,
            takeover_button=cfg.spacemouse_takeover_button,
            open_gripper_button=cfg.spacemouse_open_gripper_button,
            close_gripper_button=cfg.spacemouse_close_gripper_button,
            gripper_open_pos=cfg.spacemouse_gripper_open_pos,
            gripper_close_pos=cfg.spacemouse_gripper_close_pos,
            device_path=cfg.spacemouse_device_path,
            device_name_substring=cfg.spacemouse_device_name_substring,
            axis_scale=cfg.spacemouse_axis_scale,
            stale_timeout_s=cfg.spacemouse_stale_timeout_s,
        )
        print(
            "[INFO] SpaceMouse teleop enabled: "
            f"device={getattr(spacemouse_policy.reader, 'description', 'unknown')} "
            f"side={spacemouse_policy.control_side}",
            flush=True,
        )
    rl_policy = RLInterface(
        server_address=cfg.server_address,
        embodiment_tag=cfg.embodiment_tag,
        adapters=build_policy_adapters(cfg),
        request_timeout=cfg.request_timeout,
        fallback_control_mode=cfg.control_mode,
    )

    obs, _ = env.reset(
        options={"alias": cfg.startup_reset_alias, "discard_episode": True}
    )
    if cfg.initial_pose_source == "current_observation":
        initial_pose_manager.set_base_pose_from_observation(obs)
        print(
            "[INFO] Captured current observation as initial pose: "
            f"{initial_pose_manager.describe_current()}"
        )

    return RLContext(
        cfg=cfg,
        env=env,
        policy_router=PolicyRouter(
            fello_policy=fello_policy,
            spacemouse_policy=spacemouse_policy,
            rl_policy=rl_policy,
            enabled_sides=cfg.enabled_sides,
            z_up_step_m=cfg.delta_ee_translation_xyz_max[0],
            demo_collection=cfg.demo_collection,
        ),
        keyboard_policy=KeyboardPolicy(),
        event_router=RLEventRouter(cfg, initial_pose_manager, external_event_queue),
        state_machine=RLStateMachine(),
        initial_pose_manager=initial_pose_manager,
        parking_navigator=ParkingNavigator(),
        collision_filter=CollisionFilter(),
        speech_announcer=SpeechAnnouncer.for_mode(cfg.mode),
        img_queue=img_queue,
        obs=obs,
        external_event_queue=external_event_queue,
        fastapi_base_url=fastapi_base_url,
        timing_log=timing_log,
        handshake_server=handshake_server,
    )
