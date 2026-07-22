# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MuJoCo simulation environment for YAM bimanual robot station.

Mostly adapted from `yam_env.py` in the `xdof_samples` starter code.
"""

import os
import re
import signal
import shutil
import subprocess

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_HUB_CACHE"] = "/mnt/amlfs-02/shared/ckpts"
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import socket
import threading
import time
from typing import Any, Literal

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import portal
from scipy.spatial.transform import Rotation
import tyro

from enpire.env.forge.experimental.async_chunking_policy import AsyncChunkingPolicy
from enpire.env.forge.experimental.realtime_rtc_chunking_policy import RealtimeRTCChunkingPolicy

from enpire.env.forge.experimental.filter_utils import PeriodicAverageAccumulator
from enpire.env.forge.experimental.key_remapping_utils import map_action, map_observation
from enpire.env.forge.experimental.start_stop_play_policy import (
    StartStopPlayPolicyWrapper,
    run_viser_subprocess,
)
from enpire.env.forge.experimental.sync_chunking_policy import SyncChunkingPolicy
from enpire.env.forge.experimental.viser_policy import PolicyAdapters
from enpire.env.forge.experimental.lerobot_replay_policy import LerobotReplayPolicy
from enpire.policy.rl.record_episode_wrapper import RecordEpisodeWrapper
from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.robot_interface import RobotInterface
from enpire.env.forge.experimental.hil_policy import HILPolicyWrapper
from enpire.env.forge.experimental.pico_policy import PicoPolicy
from enpire.env.forge.robot.constants import LEFT_LEADER_PORT, RIGHT_LEADER_PORT
from enpire.env.forge.robot.yam.kinematics import YamKinematics
from enpire.env.forge.robot.fello.fello_teleop_policy import DualFelloPolicy as DualFelloTeleopPolicy, FelloTeleopPolicy
from enpire.env.forge.experimental.scripted_policy import ScriptedPolicy, SafetyLimits

_SS_PID_RE = re.compile(r"pid=(\d+)")


def _is_tcp_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _listening_pids_for_tcp_port(port: int) -> set[int]:
    ss_bin = shutil.which("ss")
    if not ss_bin:
        return set()
    proc = subprocess.run(
        [ss_bin, "-ltnp", f"( sport = :{int(port)} )"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {int(match.group(1)) for match in _SS_PID_RE.finditer(proc.stdout)}


def _pid_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_text().replace("\x00", " ").strip()
    except Exception:
        return ""


def _terminate_pids(pids: set[int], *, timeout_s: float = 1.0) -> None:
    current_pid = os.getpid()
    parent_pid = os.getppid()
    victims = [pid for pid in sorted(pids) if pid not in {current_pid, parent_pid}]
    if not victims:
        return
    for pid in victims:
        cmd = _pid_cmdline(pid)
        print(f"[Main] Reclaiming local port from pid={pid} cmd='{cmd or '?'}'")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[Main] No permission to terminate pid={pid}")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        alive = []
        for pid in victims:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                alive.append(pid)
            else:
                alive.append(pid)
        if not alive:
            return
        time.sleep(0.05)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _prepare_control_port(
    port: int,
    *,
    label: str,
    reclaim: bool,
    remap_if_busy: bool,
) -> int:
    port = int(port)
    if not _is_tcp_port_in_use(port):
        return port
    if reclaim:
        _terminate_pids(_listening_pids_for_tcp_port(port))
        if not _is_tcp_port_in_use(port):
            print(f"[Main] Reclaimed {label} port {port}")
            return port
    if remap_if_busy:
        new_port = _find_free_tcp_port()
        print(f"[Main] {label} port {port} busy; using free port {new_port} instead")
        return new_port
    return port


@dataclass
class EvalConfig:
    # Path to the model checkpoint (not required if using replay mode)
    ckpt_path: str = ""
    # Whether to use the real robot (vs simulation).
    use_real_robot: bool = False
    # Task description for the policy.
    task_description: str = "Do something useful"
    # Whether to use VLLM for inference. Only applies to RobotInterface
    use_vllm: bool = True
    # Embodiment tag. Shouldn't need to be changed
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF

    # Policy server address (for remote deployment)
    server_address: str = "localhost:8964"

    # Speed to run control loop at. Will cause robot to move faster or slower.
    policy_control_freq: int = 30

    use_pico: bool = False
    use_fello: bool = False
    fello_mode: Literal["absolute", "delta"] = "absolute"
    fello_host: str = "localhost"
    fello_side: Literal["left", "right", "both"] = "both"
    fello_left_host: str = ""
    fello_right_host: str = ""
    fello_left_port: int | None = None
    fello_right_port: int | None = None
    footswitch_device: str | None = None
    use_footswitch: bool = True

    # Length of action chunk to execute (max is 50).
    # Keep this at 50 to match UMI training defaults.
    action_horizon: int = 50
    # Replan the next chunk when the current chunk reaches this step index.
    # Async chunking and realtime RTC require replan_horizon < action_horizon.
    replan_horizon: int | None = None
    # Number of the station
    station: int | None = 1
    # Name of the operator
    operator: str | None = None
    # Video resolution
    resolution: Literal[240, 480] = 480
    # Amount of time reserved for evaluating the policy (in steps).
    # Legacy async chunking knob. When replan_horizon is set, async chunking
    # derives this automatically as action_horizon - replan_horizon - 1.
    policy_latency_steps: int = 4
    # Bootstrap overlap-delay estimate for realtime RTC, in control steps.
    # After the first measured request, realtime RTC switches to runtime
    # latency estimation automatically.
    rtc_initial_delay_steps: int = 4
    # Whether to use async policy evaluation
    use_async: bool = False
    # Enable client-owned realtime RTC overlap. This uses asynchronous chunk
    # execution with replan_horizon < action_horizon and sends the overlap
    # prefix from the currently-active chunk back to the policy server.
    use_realtime_rtc: bool = False
    # Optional explicit cap for RTC overlap in action steps. Defaults to
    # action_horizon - replan_horizon when unset.
    rtc_max_delay_steps: int | None = None
    # Linear overlap blending at chunk boundaries (kai0-style smoothing).
    # For realtime RTC this blends the old unexecuted tail with the incoming
    # successor chunk at the switch point. For async chunking this blends the
    # currently-queued old tail with the incoming successor chunk when it lands.
    use_chunk_smoothing: bool = False
    # Minimum overlap length for chunk-wise smoothing.
    min_smooth_steps: int = 8
    # Enable RTC JSONL logging. Logging stays fully disabled unless this is on.
    log_rtc: bool = False
    # Optional JSONL diagnostics path for client-side realtime RTC logging.
    # Only used when log_rtc is enabled.
    rtc_debug_log_path: str = ""
    # Applies the low-pass filter to policy actions
    use_low_pass_filter: bool = True
    # Policy action type used for visualization
    action_type: Literal["absolute", "relative"] = "absolute"
    # Portal server port for policy (Viser commands)
    policy_port: int = 8009
    # Portal server port for Viser UI (policy updates)
    viser_port: int = 8010
    # Viser web UI port
    viser_web_port: int = 8080
    # Automatically reclaim local control/UI ports from older YAM control-loop processes.
    reclaim_control_ports: bool = True
    # If reclaim fails and a port is still occupied, automatically use a free replacement port.
    remap_busy_control_ports: bool = True
    # Replay mode (local policy from dataset instead of server/GPU)
    use_replay_policy: bool = False
    # Scripted policy (Viser-driven SE(3) teleop via IK)
    use_scripted_policy: bool = False
    # Default Move-To behavior in scripted UI: use the motion planner.
    scripted_use_rrt: bool = True
    # Default RRT execution speed for scripted Move-To motions.
    scripted_planner_max_joint_vel: float = 2.0
    # cuRobo solver preset for scripted planning in YAM UI.
    scripted_planner_solver_speed: Literal["slow", "fast"] = "fast"
    # Show the motion planner feasible-region point cloud in the Viser UI.
    show_mp_feasible_region: bool = False
    # Force a fresh feasible-region recompute instead of loading a cached result.
    recompute_mp: bool = False
    # Precompute the motion-planner feasible-region cache headlessly, then exit.
    precompute_mp_feasible_region: bool = False
    # Motion-planner backend used by scripted Move-To when planner mode is enabled in the UI.
    motion_planner_backend: Literal["rrtconnect", "curobo"] = "curobo"
    # (Removed — joint velocity limit is now solely defined by
    # robot.constants.MAX_JOINT_VELOCITY_RAD_S. No per-config override.)
    # Replay dataset path (parquet/npy/npz file or folder with action files). Empty uses default.
    replay_dataset_path: str = ""
    # Number of steps to advance per get_action() in replay
    replan_replan_horizon: int = 1
    # Control mode: "joint_position" for 14D joint actions, "cartesian_position" for 16D ee_pose actions,
    # "delta_joint_position" for delta joint actions, "delta_ee_pose" for 16D delta EE pose actions,
    # "umi_ee_pose" for UMI volatile delta replay generated from observation.state chunks.
    control_mode: Literal[
        "joint_position",
        "cartesian_position",
        "delta_joint_position",
        "delta_ee_pose",
        "umi_ee_pose",
    ] = "joint_position"
    # Path to norm_stats.json for 20D UMI training-pipeline replay (required for umi_ee_pose with 20D data).
    norm_stats_path: str = ""
    # Enable microphone speech-to-text prompts.
    use_voice_prompt: bool = False
    # Voice mode: continuous streaming or push-to-talk only.
    voice_mode: Literal["continuous", "push_to_talk"] = "continuous"
    # Whether voice listening starts enabled.
    voice_start_enabled: bool = True
    # WebRTC VAD sensitivity (higher = more sensitive).
    voice_webrtc_sensitivity: int = 35
    # Stop recording after this many seconds of silence.
    voice_post_speech_silence: float = 0.7
    # Minimum seconds between prompt updates.
    voice_min_update_interval: float = 0.6
    # Minimum characters required to accept a transcript.
    voice_min_chars: int = 1
    # Disable throttling/length filters for transcripts.
    voice_disable_filtering: bool = False
    # Record Viser UI streams to disk.
    save_videos: bool = False
    # Target FPS for Viser video writer.
    video_fps: int = 30
    # Queue size for video writer.
    video_queue_size: int = 512
    # Repeat frames to match wall-clock timing.
    video_realtime: bool = True
    record_episode: bool = False
    # Run name label pre-filled in the Viser episode-upload panel.
    eval_run_name: str = ""


def prompt_missing_fields(cfg: EvalConfig) -> EvalConfig:
    # Operator username
    if cfg.operator is None and _should_prompt_for_operator(cfg):
        cfg.operator = input("Enter operator username: ")

    # Station number
    if cfg.station is None:
        hostname = socket.gethostname()
        if hostname == "maxf-desktop":
            cfg.station = 1
        else:
            assert hostname.startswith("gear-yam-desktop-"), (
                "If not using gear-yam-desktop-*, station number must be specified."
            )
            cfg.station = int(hostname.split("-")[-1])

    return cfg


def _should_prompt_for_operator(cfg: EvalConfig) -> bool:
    return not (
        cfg.use_replay_policy
        or cfg.use_scripted_policy
        or cfg.precompute_mp_feasible_region
        or bool(cfg.replay_dataset_path)
    )


def _parse_env_bool(value: str) -> bool | None:
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if cleaned in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _parse_env_camera_names(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    items = tuple(x.strip().lower() for x in value.split(",") if x.strip())
    return items or default


def _precompute_motion_planner_feasible_region(
    observation: dict[str, np.ndarray],
    *,
    force_recompute: bool = False,
) -> None:
    from enpire.env.forge.experimental.motion_planner import YamMotionPlanner
    from enpire.env.forge.experimental.motion_planner_feasible_region import (
        DEFAULT_FEASIBLE_REGION_CONFIG,
        compute_feasible_region_cache_key,
        compute_side_feasible_region,
        feasible_region_cache_path,
        load_feasible_region_cache,
        save_feasible_region_cache,
    )

    left_jp = (
        np.asarray(observation.get("left_joint_pos", np.zeros(6)), dtype=np.float64)
        .reshape(-1)[:6]
        .copy()
    )
    right_jp = (
        np.asarray(observation.get("right_joint_pos", np.zeros(6)), dtype=np.float64)
        .reshape(-1)[:6]
        .copy()
    )
    left_gripper = float(
        np.asarray(
            observation.get("left_gripper_pos", np.ones(1)), dtype=np.float64
        ).reshape(-1)[0]
    )
    right_gripper = float(
        np.asarray(
            observation.get("right_gripper_pos", np.ones(1)), dtype=np.float64
        ).reshape(-1)[0]
    )

    planner = YamMotionPlanner()
    _, left_quat, _, right_quat = planner._kin.forward_kinematics(left_jp, right_jp)
    config = DEFAULT_FEASIBLE_REGION_CONFIG
    cache_key = compute_feasible_region_cache_key(
        current_left_jp=left_jp,
        current_right_jp=right_jp,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        left_target_quat_xyzw=left_quat,
        right_target_quat_xyzw=right_quat,
        config=config,
    )
    cache_path = feasible_region_cache_path(cache_key)

    if not force_recompute:
        cached = load_feasible_region_cache(cache_key)
        if cached is not None:
            created_at = cached.get("created_at", "unknown time")
            print(
                f"[Main] Motion-planner feasible-region cache already exists: "
                f"{cache_path} (created {created_at})"
            )
            print(
                "[Main] Launch later with --show-mp-feasible-region to load it in Viser."
            )
            return

    print("[Main] Precomputing motion-planner feasible region headlessly...")
    print(f"[Main] Cache key: {cache_key}")
    print(
        f"[Main] Grid: {config.positions_per_side} points/arm, "
        f"{config.num_workers} worker threads"
    )
    progress_last_ts = {"left": 0.0, "right": 0.0}
    results: dict[str, Any] = {}
    scan_start = time.time()

    def _progress(side: str, done: int, total: int) -> None:
        now = time.time()
        if done < total and (now - progress_last_ts[side]) < 0.5:
            return
        progress_last_ts[side] = now
        pct = 100.0 * float(done) / float(total)
        print(
            f"[Main] {side.title()} feasible-region scan: {done}/{total} ({pct:.0f}%)",
            flush=True,
        )

    for side, target_quat in (("left", left_quat), ("right", right_quat)):
        side_start = time.time()
        print(f"[Main] Sampling {side} arm...")
        result = compute_side_feasible_region(
            planner,
            side=side,
            current_left_jp=left_jp,
            current_right_jp=right_jp,
            target_quat_xyzw=target_quat,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            config=config,
            progress_callback=lambda done, total, side=side: _progress(
                side, done, total
            ),
        )
        results[side] = result
        reachable = int(np.count_nonzero(result.scores > 0.0))
        print(
            f"[Main] {side.title()} reachable: {reachable}/{len(result.scores)} "
            f"in {time.time() - side_start:.1f}s"
        )

    save_feasible_region_cache(
        cache_key=cache_key,
        config=config,
        start_left_jp=left_jp,
        start_right_jp=right_jp,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        left_target_quat_xyzw=left_quat,
        right_target_quat_xyzw=right_quat,
        results=results,
    )
    print(
        f"[Main] Saved motion-planner feasible-region cache to {cache_path} "
        f"in {time.time() - scan_start:.1f}s"
    )
    print("[Main] Launch later with --show-mp-feasible-region to load it in Viser.")


def _waypoint_ramp_to_joint_state(
    env: gym.Env,
    observation: dict[str, np.ndarray],
    target_state: dict[str, np.ndarray],
    policy_control_freq: float,
    viser_push: Any = None,
    max_joint_vel_rad_s: float = 0.1,
    min_duration_s: float = 5.0,
    label: str = "ramp",
) -> dict[str, np.ndarray]:
    """Smoothly drive the robot to ``target_state`` via env.step waypoints.

    Uniformly safe for every replay mode (joint_position, delta_*, cartesian_*,
    umi_ee_pose) on both sim and real: the env is temporarily forced into
    ``joint_position`` mode for the ramp, absolute-joint actions are sent at
    ``policy_control_freq``, and the original control mode is restored at the
    end. Speed is capped at ``max_joint_vel_rad_s`` with a ``min_duration_s``
    floor. Pushes every intermediate obs to Viser when ``viser_push`` is
    provided so the motion is visible in the UI.
    """
    start_left = np.asarray(
        observation.get("left_joint_pos", np.zeros(6)), dtype=np.float64
    )
    start_right = np.asarray(
        observation.get("right_joint_pos", np.zeros(6)), dtype=np.float64
    )
    target_left = np.asarray(target_state["left_joint_pos"], dtype=np.float64)
    target_right = np.asarray(target_state["right_joint_pos"], dtype=np.float64)
    left_grip = np.asarray(target_state["left_gripper_pos"], dtype=np.float32)
    right_grip = np.asarray(target_state["right_gripper_pos"], dtype=np.float32)

    pre_dist = float(
        max(
            np.max(np.abs(target_left - start_left)),
            np.max(np.abs(target_right - start_right)),
        )
    )
    if pre_dist < 0.005:
        print(f"  [{label}] already at target (max err {pre_dist:.4f} rad); skipping.")
        return observation

    duration_s = max(min_duration_s, pre_dist / max(1e-6, max_joint_vel_rad_s))
    n_waypoints = max(2, int(duration_s * max(1.0, policy_control_freq)))
    print(
        f"  [{label}] ramping {n_waypoints} waypoints "
        f"(~{duration_s:.1f}s @ {policy_control_freq:.0f} Hz, "
        f"max {max_joint_vel_rad_s:.2f} rad/s, dist {pre_dist:.3f} rad)"
    )

    # Force env into joint_position mode for the ramp so absolute-joint
    # actions are accepted regardless of what the replay uses. Restore after.
    target_env = getattr(env, "unwrapped", env)
    prev_control_mode = getattr(target_env, "control_mode", None)
    if (
        prev_control_mode is not None
        and prev_control_mode != "joint_position"
        and hasattr(target_env, "set_control_mode")
    ):
        target_env.set_control_mode("joint_position", observation=observation)
        print(
            f"  [{label}] env mode: {prev_control_mode} -> joint_position "
            f"(temporary for ramp)"
        )

    try:
        for i in range(1, n_waypoints + 1):
            alpha = i / n_waypoints
            interp_left = (
                (1.0 - alpha) * start_left + alpha * target_left
            ).astype(np.float32)
            interp_right = (
                (1.0 - alpha) * start_right + alpha * target_right
            ).astype(np.float32)

            action = {
                "left_joint_pos": interp_left,
                "left_gripper_pos": left_grip,
                "right_joint_pos": interp_right,
                "right_gripper_pos": right_grip,
            }

            observation, _, _, _, _ = env.step(action)
            if viser_push is not None:
                try:
                    viser_push(observation)
                except Exception as exc:
                    print(f"  [{label}] viser_push failed (non-fatal): {exc}")
    finally:
        if (
            prev_control_mode is not None
            and prev_control_mode != "joint_position"
            and hasattr(target_env, "set_control_mode")
        ):
            target_env.set_control_mode(prev_control_mode, observation=observation)
            print(f"  [{label}] env mode restored to {prev_control_mode}")

    obs_left = np.asarray(
        observation.get("left_joint_pos", np.zeros(6)), dtype=np.float64
    )
    obs_right = np.asarray(
        observation.get("right_joint_pos", np.zeros(6)), dtype=np.float64
    )
    final_err = float(
        max(
            np.max(np.abs(obs_left - target_left)),
            np.max(np.abs(obs_right - target_right)),
        )
    )
    print(f"  [{label}] done, max joint err {final_err:.4f} rad")
    return observation


def _calibrate_for_delta_replay(
    env: gym.Env,
    policy: Any,
    control_mode: str,
    observation: dict[str, np.ndarray],
    zero_state: dict[str, np.ndarray],
    cfg: EvalConfig,
    settle_extra_s: float = 2.0,
    settle_tol: float = 0.02,
    settle_poll_hz: float = 10.0,
    viser_push: Any = None,
) -> dict[str, np.ndarray]:
    """Calibrate the env to the recording's initial state before delta replay.

    1. Validates the replay policy has a valid ``initial_state``.
    2. Calls ``env.reset(initial_state=...)`` to interpolate the robot there.
    3. Waits for the robot to settle, printing progress.
    4. Reports final calibration error.
    5. Returns the post-calibration observation.

    Falls back to the current observation (zero_state) if calibration data is
    missing or malformed, with clear warnings.
    """
    replay_initial = getattr(policy, "initial_state", None)

    # --- No initial_state available ---
    if replay_initial is None:
        raise ValueError(
            f"No initial_state found in replay data for {control_mode} mode. Add that, or your replay will not match original joint positions. "
        )

    # --- Validate keys ---
    required_keys = {
        "left_joint_pos",
        "left_gripper_pos",
        "right_joint_pos",
        "right_gripper_pos",
    }
    missing = required_keys - set(replay_initial.keys())
    if missing:
        print(
            f"\033[33m[Main] WARNING: initial_state is missing keys {missing}. "
            f"Cannot calibrate — using zero_state instead.\033[0m"
        )
        return observation

    # --- Compute how far we need to move ---
    target_jp = np.concatenate(
        [
            replay_initial["left_joint_pos"],
            replay_initial["right_joint_pos"],
        ]
    )
    current_jp = np.concatenate(
        [
            observation.get("left_joint_pos", np.zeros(6)),
            observation.get("right_joint_pos", np.zeros(6)),
        ]
    )
    pre_dist = float(np.max(np.abs(target_jp - current_jp)))

    is_already_home = pre_dist < 0.005  # < 0.3 degrees — effectively at target

    print()
    print("=" * 64)
    print(f"  CALIBRATING for {control_mode} replay")
    print("=" * 64)
    print(f"  Target left_joint_pos  = {replay_initial['left_joint_pos']}")
    print(f"  Target left_gripper    = {replay_initial['left_gripper_pos']}")
    print(f"  Target right_joint_pos = {replay_initial['right_joint_pos']}")
    print(f"  Target right_gripper   = {replay_initial['right_gripper_pos']}")
    print(f"  Max joint distance from current: {pre_dist:.4f} rad")

    if is_already_home:
        print("  Robot is already at the target state. No motion needed.")
        print("=" * 64)
        print()
        return observation

    # Delegate the ramp to the shared helper so initial calibration and
    # post-replay home use one identical code path. Calibration is meant to
    # be snappier than the home ramp — 3 s floor, speed high enough that
    # the floor always dominates for typical intra-workspace distances.
    observation = _waypoint_ramp_to_joint_state(
        env,
        observation,
        replay_initial,
        cfg.policy_control_freq,
        viser_push=viser_push,
        max_joint_vel_rad_s=2.0,
        min_duration_s=3.0,
        label="calibration",
    )

    cur_jp = np.concatenate(
        [
            observation.get("left_joint_pos", np.zeros(6)),
            observation.get("right_joint_pos", np.zeros(6)),
        ]
    )
    final_err = float(np.max(np.abs(cur_jp - target_jp)))
    if final_err < settle_tol:
        print(
            f"\033[32m  Calibration COMPLETE: max joint error = "
            f"{final_err:.4f} rad (< {settle_tol} rad tolerance)\033[0m"
        )
    else:
        print(
            f"\033[33m  Calibration WARNING: max joint error = "
            f"{final_err:.4f} rad (tolerance was {settle_tol} rad). "
            f"Replay will proceed.\033[0m"
        )

    print(f"  Starting {control_mode} replay NOW.")
    print("=" * 64)
    print()

    return observation


def main(cfg: EvalConfig) -> None:
    """Run the environment with random actions in the MuJoCo viewer.

    This function demonstrates the environment by running it with the policy
    and allowing the user to reset with the spacebar.

    Args:
        ckpt_path: Path to the model checkpoint.
        use_real_robot: Whether to use the real robot (vs simulation).
        task_description: Task description for the policy.
        use_vllm: Whether to use VLLM for inference.
        policy_control_freq: Control frequency for the policy.
        action_horizon: Action horizon for the policy.
        use_robot_interface: Whether to use RobotInterface (vs OmniDiffusionPolicy).
        embodiment_tag: Embodiment tag for the policy.
    """
    cfg = prompt_missing_fields(cfg)
    env_save_videos = os.getenv("YAM_SAVE_VIDEOS")
    if env_save_videos is not None:
        parsed = _parse_env_bool(env_save_videos)
        if parsed is not None:
            cfg.save_videos = parsed
    if cfg.replay_dataset_path and not cfg.use_replay_policy:
        cfg.use_replay_policy = True

    # Determine control mode based on replay action space
    control_mode = cfg.control_mode
    if cfg.use_scripted_policy:
        # ScriptedPolicy emits absolute joint commands (FK+IK only when a
        # nudge arrives; identical command every idle step → motors locked).
        control_mode = "joint_position"

    # UMI replay converts deltas to joint commands via IK before env.step,
    # so the env itself must run in joint_position mode.
    env_control_mode = (
        "joint_position"
        if control_mode in ("umi_ee_pose", "delta_ee_pose")
        else control_mode
    )

    if cfg.use_real_robot:
        register(id="YamReal-v0", entry_point="enpire.env.forge.robot.yam.yam_real_env:YamRealEnv")
        enabled_camera_names = _parse_env_camera_names(
            os.environ.get("YAM_ENABLED_CAMERAS"),
            ("top",),
        )
        env = gym.make(
            "YamReal-v0",
            policy_control_freq=cfg.policy_control_freq,
            control_mode=env_control_mode,
            enable_cameras=not cfg.use_scripted_policy,
        )
    else:
        register(id="YamSim-v0", entry_point="enpire.env.forge.robot.yam.yam_sim_env:YamSimEnv")
        env = gym.make(
            "YamSim-v0",
            control_mode=env_control_mode,
            enable_cameras=not cfg.use_scripted_policy,
        )
    embodiment_tag = cfg.embodiment_tag

    if cfg.record_episode:
        # Fetch policy metadata from the server for the run directory name.
        _policy_name = ""
        _policy_step = 0
        try:
            import portal as _portal

            _info_client = _portal.Client(cfg.server_address)
            _server_info = _info_client.get_info().result(timeout=5.0)
            if isinstance(_server_info, dict):
                _policy_name = _server_info.get("policy_name", "")
                _policy_step = int(_server_info.get("step", 0))
                print(f"[Main] Policy info from server: name={_policy_name}, step={_policy_step}")
        except Exception as exc:
            print(f"[Main] Could not query policy server get_info: {exc}")

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_dir_name = f"{timestamp}-YAM-{cfg.station:02d}-eval"
        # Append policy name and step to match the gdrive naming convention:
        #   2026-04-06-16-57-56-YAM-01-eval_pi05_yam_pickup_cutter_step018000
        if _policy_name:
            run_dir_name += f"_{_policy_name}"
        if _policy_step > 0:
            run_dir_name += f"_step{_policy_step:06d}"
        output_dir = Path(os.environ["YAM_RAW_PATH"]) / "eval" / run_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        operator = cfg.operator
        if operator is None and _should_prompt_for_operator(cfg):
            operator = input("Enter operator username: ")
        env = RecordEpisodeWrapper(env, output_dir=str(output_dir), operator=operator)
        # Auto-fill eval_run_name so Viser UI shows it and uploads use it.
        if not cfg.eval_run_name:
            cfg.eval_run_name = run_dir_name
        print(f"\033[34m[Main] Recording episodes to: {output_dir}\033[0m")

    zero_state = {
        "left_joint_pos": np.zeros(6),
        "left_gripper_pos": np.ones(1),
        "right_joint_pos": np.zeros(6),
        "right_gripper_pos": np.ones(1),
    }
    # We discard the next episode because we don't have the task name yet
    observation, info = env.reset(
        options=dict(target_joint_position=deepcopy(zero_state), start_new_episode=False)
    )

    if cfg.precompute_mp_feasible_region:
        try:
            _precompute_motion_planner_feasible_region(
                observation,
                force_recompute=cfg.recompute_mp,
            )
        finally:
            env.close()
        return

    # Keep adapters for StartStopPlayPolicyWrapper and ViserUI
    adapters = PolicyAdapters(
        # If env and URDF joint orders differ, provide explicit ordering here
        map_observation=lambda _observation: map_observation(
            _observation, embodiment_tag, cfg.resolution
        ),
        map_action=lambda _action: map_action(_action, embodiment_tag),
    )

    if cfg.use_scripted_policy:
        safety = SafetyLimits()
        policy = ScriptedPolicy(
            action_space=env.action_space,
            control_hz=cfg.policy_control_freq,
            safety=safety,
        )
    elif cfg.use_replay_policy:
        replay_kwargs = dict(
            replan_horizon=cfg.replan_replan_horizon,
            control_mode=cfg.control_mode,
            action_horizon=cfg.action_horizon,
            norm_stats_path=cfg.norm_stats_path,
        )
        if cfg.replay_dataset_path:
            replay_kwargs["dataset_path"] = cfg.replay_dataset_path
        else:
            # Force empty path so UI can provide it later.
            replay_kwargs["dataset_path"] = ""
        policy = LerobotReplayPolicy(**replay_kwargs)
    else:
        policy = RobotInterface(
            checkpoint_dir=cfg.ckpt_path,
            embodiment_tag=embodiment_tag,
            device="cuda:0",
            freq_bins=20,
            use_vllm=cfg.use_vllm,
            server_address=cfg.server_address,
            adapters=adapters,
        )

        replan_horizon = (
            cfg.replan_horizon if cfg.replan_horizon is not None else cfg.action_horizon
        )
        if cfg.use_realtime_rtc:
            if replan_horizon >= cfg.action_horizon:
                raise ValueError(
                    "Realtime RTC requires --replan-horizon < --action-horizon, "
                    f"got {replan_horizon} >= {cfg.action_horizon}"
                )
            max_delay_steps = cfg.rtc_max_delay_steps
            if max_delay_steps is None:
                max_delay_steps = cfg.action_horizon - replan_horizon
            rtc_log_path = None
            if cfg.log_rtc:
                rtc_log_path = (
                    cfg.rtc_debug_log_path
                    or f"/tmp/yam_control_loop_rtc_{int(time.time())}.jsonl"
                )
                print(f"[Main] RTC logging enabled: {rtc_log_path}")
            policy = RealtimeRTCChunkingPolicy(
                policy=policy,
                action_horizon=cfg.action_horizon,
                replan_horizon=replan_horizon,
                bootstrap_delay_steps=cfg.rtc_initial_delay_steps,
                max_delay_steps=max_delay_steps,
                control_hz=cfg.policy_control_freq,
                # Spend at most half of the control loop time waiting for an action
                max_get_action_seconds=0.5 * 1 / cfg.policy_control_freq,
                require_prev_observation=(control_mode == "umi_ee_pose"),
                use_chunk_smoothing=cfg.use_chunk_smoothing,
                min_smooth_steps=cfg.min_smooth_steps,
                debug_log_path=rtc_log_path,
            )
        # Pico data collect requires use of AsyncChunkingPolicy
        elif cfg.use_async:
            policy = AsyncChunkingPolicy(
                policy=policy,
                action_exec_horizon=cfg.action_horizon,
                policy_latency_steps=cfg.policy_latency_steps,
                replan_horizon=cfg.replan_horizon,
                # Spend at most half of the control loop time waiting for an action
                max_get_action_seconds=0.5 * 1 / cfg.policy_control_freq,
                use_chunk_smoothing=cfg.use_chunk_smoothing,
                min_smooth_steps=cfg.min_smooth_steps,
                require_prev_observation=(control_mode == "umi_ee_pose"),
            )
        else:
            policy = SyncChunkingPolicy(
                policy=policy,
                action_exec_horizon=cfg.action_horizon,
                require_prev_observation=(control_mode == "umi_ee_pose"),
            )

    # Before any replay begins, smoothly drive the robot to the recording's
    # first-frame joint state. Required for delta modes (deltas are relative
    # to the recorded start pose) and also matters for joint_position replay
    # (otherwise the first env.step jumps straight to the first recorded
    # joint target, which looks violent and is unsafe on the real robot).
    if cfg.use_replay_policy and control_mode in (
        "joint_position",
        "delta_joint_position",
        "cartesian_position",
        "delta_ee_pose",
        "umi_ee_pose",
    ):
        # Unwrap to find the inner replay policy that holds initial_state
        _inner = policy
        while True:
            if hasattr(_inner, "policy"):
                _inner = _inner.policy
            elif hasattr(_inner, "base_policy"):
                _inner = _inner.base_policy
            else:
                break
        # Only calibrate at startup if a dataset has already populated
        # initial_state. When launched without --dataset-path the user loads
        # the dataset later via the UI, which triggers the mode-change
        # calibration path on its own.
        if getattr(_inner, "initial_state", None) is None:
            print(
                "[Main] Skipping startup calibration — replay dataset not loaded yet; "
                "will calibrate on mode change / sync_to_init once data is set."
            )
        else:
            observation = _calibrate_for_delta_replay(
                env,
                _inner,
                control_mode,
                observation,
                zero_state,
                cfg,
            )
            # The wrapper doesn't exist yet at this point; its hold_action
            # will be lazily computed from the current (post-calibration)
            # observation on first get_action call, so no invalidation needed.

    if cfg.use_pico and cfg.use_fello:
        raise ValueError("use_pico and use_fello are mutually exclusive")

    if cfg.use_pico:
        pico_policy = PicoPolicy(env=env)
        policy = HILPolicyWrapper(base_policy=policy, hil_policy=pico_policy)
    if cfg.use_fello:

        def _default_fello_port(side: Literal["left", "right"]) -> int:
            return LEFT_LEADER_PORT if side == "left" else RIGHT_LEADER_PORT

        if cfg.fello_side == "both":
            fello_policy = DualFelloTeleopPolicy(
                left=FelloTeleopPolicy(
                    host=cfg.fello_left_host or cfg.fello_host,
                    target_side="left",
                    use_footswitch=cfg.use_footswitch,
                    port=(
                        cfg.fello_left_port
                        if cfg.fello_left_port is not None
                        else _default_fello_port("left")
                    ),
                ),
                right=FelloTeleopPolicy(
                    host=cfg.fello_right_host or cfg.fello_host,
                    target_side="right",
                    use_footswitch=cfg.use_footswitch,
                    port=(
                        cfg.fello_right_port
                        if cfg.fello_right_port is not None
                        else _default_fello_port("right")
                    ),
                ),
            )
        else:
            fello_policy = FelloTeleopPolicy(
                host=cfg.fello_host,
                target_side=cfg.fello_side,
                use_footswitch=cfg.use_footswitch,
                port=(
                    cfg.fello_left_port
                    if cfg.fello_side == "left" and cfg.fello_left_port is not None
                    else cfg.fello_right_port
                    if cfg.fello_side == "right" and cfg.fello_right_port is not None
                    else _default_fello_port(cfg.fello_side)
                ),
            )
        policy = HILPolicyWrapper(
            base_policy=policy,
            hil_policy=fello_policy,
            delta_mode=(cfg.fello_mode == "delta"),
        )
    # If this policy wrapper is not last, it is likely to result in an
    # unresponsive ViserUI
    original_ports = (cfg.policy_port, cfg.viser_port, cfg.viser_web_port)
    cfg.policy_port = _prepare_control_port(
        cfg.policy_port,
        label="policy IPC",
        reclaim=cfg.reclaim_control_ports,
        remap_if_busy=cfg.remap_busy_control_ports,
    )
    cfg.viser_port = _prepare_control_port(
        cfg.viser_port,
        label="viser IPC",
        reclaim=cfg.reclaim_control_ports,
        remap_if_busy=cfg.remap_busy_control_ports,
    )
    cfg.viser_web_port = _prepare_control_port(
        cfg.viser_web_port,
        label="viser web",
        reclaim=cfg.reclaim_control_ports,
        remap_if_busy=cfg.remap_busy_control_ports,
    )
    updated_ports = (cfg.policy_port, cfg.viser_port, cfg.viser_web_port)
    if updated_ports != original_ports:
        print(
            "[Main] Control/UI ports configured as "
            f"policy={cfg.policy_port}, viser={cfg.viser_port}, web={cfg.viser_web_port}",
            flush=True,
        )
    policy = StartStopPlayPolicyWrapper(
        policy=policy,
        adapters=adapters,
        embodiment_tag=embodiment_tag,
        policy_port=cfg.policy_port,
        viser_port=cfg.viser_port,
        motion_planner_backend=cfg.motion_planner_backend,
        default_motion_planner_solver_speed=cfg.scripted_planner_solver_speed,
        preload_motion_planner=(
            bool(cfg.use_scripted_policy)
            and bool(cfg.scripted_use_rrt)
            and cfg.motion_planner_backend == "curobo"
        ),
    )
    # Make sure the wrapper's pause hold_action is lazily recomputed from
    # the current (possibly post-calibration) observation on its first pause
    # get_action call, rather than any stale cached value from init.
    if hasattr(policy, "_hold_action"):
        policy._hold_action = None
    shared_motion_planner_port = getattr(policy, "shared_motion_planner_port", None)

    # Optional voice prompt loop (runs in background thread).
    if cfg.use_voice_prompt:
        if hasattr(policy, "set_voice_enabled"):
            policy.set_voice_enabled(cfg.voice_start_enabled)
        AudioToTextRecorder = None
        try:
            from RealtimeSTT import AudioToTextRecorder as _Recorder

            AudioToTextRecorder = _Recorder
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[Main] RealtimeSTT import failed: {exc}")
            # Fallback to local repo if present
            try:
                import sys as _sys

                _rt_root = Path(__file__).resolve().parents[2] / "RealtimeSTT"
                if _rt_root.exists():
                    _sys.path.insert(0, str(_rt_root))
                    from RealtimeSTT import AudioToTextRecorder as _Recorder  # type: ignore[assignment]

                    AudioToTextRecorder = _Recorder
                    print(f"[Main] RealtimeSTT loaded from: {_rt_root}")
            except Exception as exc2:  # pragma: no cover - optional dependency
                print(
                    f"[Main] Voice prompt disabled (RealtimeSTT import failed): {exc2}"
                )
        if AudioToTextRecorder is not None:
            stop_event = threading.Event()
            last_update = 0.0
            last_text = ""

            def _should_update(cleaned: str) -> bool:
                nonlocal last_update, last_text
                if cfg.voice_disable_filtering:
                    return True
                now = time.monotonic()
                if (now - last_update) < cfg.voice_min_update_interval:
                    print("[Voice] Throttling transcript update")
                    return False
                if cleaned == last_text:
                    print("[Voice] Skipping duplicate transcript")
                    return False
                last_update = now
                last_text = cleaned
                return True

            def _on_transcription(text: str) -> None:
                cleaned = text.strip()
                if not cleaned:
                    print("[Voice] Ignored empty transcript")
                    return
                if (
                    not cfg.voice_disable_filtering
                    and len(cleaned) < cfg.voice_min_chars
                ):
                    print(f"[Voice] Ignored short transcript: {cleaned}")
                    return
                print(f"[Voice] Transcript: {cleaned}")
                if not _should_update(cleaned):
                    return
                policy.set_task_command(cleaned)
                print(f"[Voice] Task command updated: {cleaned}")

            def _voice_loop() -> None:
                print("[Main] Initializing voice prompt (WebRTC)...")
                try:
                    recorder = AudioToTextRecorder(
                        model="tiny.en",
                        webrtc_sensitivity=cfg.voice_webrtc_sensitivity,
                        spinner=True,
                        post_speech_silence_duration=cfg.voice_post_speech_silence,
                    )
                except Exception as exc:
                    print(
                        f"[Main] Voice prompt disabled (AudioToTextRecorder init failed): {exc}"
                    )
                    return
                print("[Main] 🎤 Voice prompt ready.")
                try:
                    while not stop_event.is_set():
                        should_capture = False
                        if (
                            hasattr(policy, "consume_voice_once")
                            and policy.consume_voice_once()
                        ):
                            should_capture = True
                        elif cfg.voice_mode == "continuous":
                            if hasattr(policy, "get_voice_enabled"):
                                should_capture = bool(policy.get_voice_enabled())
                            else:
                                should_capture = True
                        if should_capture:
                            recorder.text(_on_transcription)
                        else:
                            time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
                finally:
                    recorder.shutdown()

            voice_thread = threading.Thread(target=_voice_loop, daemon=True)
            voice_thread.start()

    # Start ViserUI in subprocess using portal.Process
    # Portal.Process doesn't support args, so we need to create a wrapper function
    def start_viser():
        run_viser_subprocess(
            adapters,
            cfg.task_description,
            cfg.action_horizon,
            embodiment_tag,
            action_type=cfg.action_type,
            policy_port=cfg.policy_port,
            viser_port=cfg.viser_port,
            viser_web_port=cfg.viser_web_port,
            video_enabled=cfg.save_videos,
            video_fps=cfg.video_fps,
            video_realtime=cfg.video_realtime,
            video_queue_size=cfg.video_queue_size,
            show_scripted_controls=cfg.use_scripted_policy,
            scripted_use_planner_default=cfg.scripted_use_rrt,
            scripted_planner_solver_speed=cfg.scripted_planner_solver_speed,
            scripted_planner_max_joint_vel=min(
                3.0,
                max(0.0, float(cfg.scripted_planner_max_joint_vel)),
            ),
            show_mp_feasible_region=cfg.show_mp_feasible_region,
            recompute_mp=cfg.recompute_mp,
            default_motion_planner_backend=cfg.motion_planner_backend,
            shared_motion_planner_port=shared_motion_planner_port,
            eval_run_name=cfg.eval_run_name,
        )

    portal.Process(start_viser, start=True)
    print("[Main] ViserUI subprocess started", flush=True)
    print("[Main] DBG: past Viser subprocess launch", flush=True)

    terminated = False
    truncated = False
    forward_time_accumulator = PeriodicAverageAccumulator(window_seconds=1.0)
    is_recording: bool = False  # Manual recording state (pedal 2=start, pedal 0=save)

    # Debug: record converted joint actions during delta_ee_pose replay
    # so we can compare them against the original joint-space parquet.
    _DEBUG_PARQUET_PATH = Path(
        "data/yam/CutterBuxJointConvertedFromEEDeltaPose_000000.parquet"
    )
    _debug_actions: list[np.ndarray] = []  # 14D post-IK joint actions
    _debug_states: list[np.ndarray] = []  # 14D pre-step observation state
    _debug_episode_done = False  # stop recording after replay ends

    def _save_debug_parquet() -> None:
        """Save accumulated debug data to parquet and clear buffers."""
        nonlocal _debug_episode_done
        if not _debug_actions:
            _debug_episode_done = False
            return
        import pandas as pd

        actions_arr = np.stack(_debug_actions)  # (T, 14)
        states_arr = np.stack(_debug_states)  # (T, 14)
        df = pd.DataFrame(
            {
                "action": list(actions_arr),
                "observation.state": list(states_arr),
            }
        )
        _DEBUG_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_DEBUG_PARQUET_PATH, index=False)
        print(
            f"\033[34m[Debug] Saved {len(_debug_actions)} frames to "
            f"{_DEBUG_PARQUET_PATH}\033[0m"
        )
        _debug_actions.clear()
        _debug_states.clear()
        _debug_episode_done = False

    # Main control loop
    step_count = 0
    delta_replay_kinematics = (
        YamKinematics()
        if cfg.use_replay_policy and control_mode in ("delta_ee_pose", "umi_ee_pose")
        else None
    )
    delta_replay_joint_seed: dict[str, np.ndarray] | None = None
    umi_chunk_start_step: int | None = None
    umi_chunk_base_ee: dict[str, np.ndarray] | None = None

    def _seed_delta_replay_joint_state(
        obs: dict[str, Any], reset_umi: bool = False
    ) -> None:
        nonlocal delta_replay_joint_seed, umi_chunk_start_step, umi_chunk_base_ee
        if delta_replay_kinematics is None:
            return
        delta_replay_joint_seed = {
            "left_joint_pos": np.asarray(
                obs.get("left_joint_pos", np.zeros(6)), dtype=np.float64
            ).copy(),
            "right_joint_pos": np.asarray(
                obs.get("right_joint_pos", np.zeros(6)), dtype=np.float64
            ).copy(),
        }
        if reset_umi:
            umi_chunk_start_step = None
            umi_chunk_base_ee = None

    def _convert_delta_ee_replay_action_to_joint(
        delta_action: dict[str, Any],
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal delta_replay_joint_seed, umi_chunk_start_step, umi_chunk_base_ee
        if delta_replay_kinematics is None:
            return delta_action
        if not (
            "left_ee_pos" in delta_action
            and "left_ee_quat_xyzw" in delta_action
            and "right_ee_pos" in delta_action
            and "right_ee_quat_xyzw" in delta_action
        ):
            return delta_action

        if delta_replay_joint_seed is None:
            _seed_delta_replay_joint_state(obs, reset_umi=True)
        assert delta_replay_joint_seed is not None

        if "_recorded_state_left_jp" in delta_action:
            cur_left = np.asarray(delta_action["_recorded_state_left_jp"], dtype=np.float64)
            cur_right = np.asarray(delta_action["_recorded_state_right_jp"], dtype=np.float64)
        else:
            cur_left = delta_replay_joint_seed["left_joint_pos"]
            cur_right = delta_replay_joint_seed["right_joint_pos"]

        is_absolute = bool(delta_action.get("umi_absolute_target", False))

        ee_l_pos = np.asarray(delta_action["left_ee_pos"], dtype=np.float64)
        ee_l_q = np.asarray(delta_action["left_ee_quat_xyzw"], dtype=np.float64)
        ee_r_pos = np.asarray(delta_action["right_ee_pos"], dtype=np.float64)
        ee_r_q = np.asarray(delta_action["right_ee_quat_xyzw"], dtype=np.float64)

        if is_absolute:
            # 20D training-pipeline path: pos/quat are already global-frame absolute targets.
            tgt_l_pos, tgt_l_q = ee_l_pos, ee_l_q
            tgt_r_pos, tgt_r_q = ee_r_pos, ee_r_q
        else:
            # Legacy 16D delta path
            cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = (
                delta_replay_kinematics.forward_kinematics(cur_left, cur_right)
            )

            # --- UMI: establish chunk base BEFORE anything else ---
            if control_mode == "umi_ee_pose":
                chunk_idx = int(delta_action.get("umi_chunk_start_step", -1))
                if chunk_idx < 0:
                    chunk_idx = (
                        0 if umi_chunk_start_step is None else int(umi_chunk_start_step)
                    )
                if umi_chunk_base_ee is None or umi_chunk_start_step != chunk_idx:
                    umi_chunk_start_step = chunk_idx
                    umi_chunk_base_ee = {
                        "left_pos": np.asarray(cur_l_pos, dtype=np.float64).copy(),
                        "left_quat": np.asarray(cur_l_q, dtype=np.float64).copy(),
                        "right_pos": np.asarray(cur_r_pos, dtype=np.float64).copy(),
                        "right_quat": np.asarray(cur_r_q, dtype=np.float64).copy(),
                    }

            # Zero-delta fast path: skip IK when delta is identity.
            if (
                np.linalg.norm(ee_l_pos) < 1e-7
                and np.linalg.norm(ee_r_pos) < 1e-7
                and np.linalg.norm(ee_l_q - np.array([0.0, 0.0, 0.0, 1.0])) < 1e-6
                and np.linalg.norm(ee_r_q - np.array([0.0, 0.0, 0.0, 1.0])) < 1e-6
            ):
                out_action = {
                    "left_joint_pos": cur_left.astype(np.float32),
                    "right_joint_pos": cur_right.astype(np.float32),
                    "left_gripper_pos": np.asarray(
                        delta_action["left_gripper_pos"], dtype=np.float32
                    ),
                    "right_gripper_pos": np.asarray(
                        delta_action["right_gripper_pos"], dtype=np.float32
                    ),
                }
                if "source" in delta_action:
                    out_action["source"] = delta_action["source"]
                return out_action

            # Compute absolute target EE pose from deltas.
            if control_mode == "umi_ee_pose":
                assert umi_chunk_base_ee is not None
                base_l_pos = umi_chunk_base_ee["left_pos"]
                base_l_q = umi_chunk_base_ee["left_quat"]
                base_r_pos = umi_chunk_base_ee["right_pos"]
                base_r_q = umi_chunk_base_ee["right_quat"]
                tgt_l_pos = base_l_pos + ee_l_pos
                tgt_r_pos = base_r_pos + ee_r_pos
                tgt_l_q = (
                    Rotation.from_quat(ee_l_q) * Rotation.from_quat(base_l_q)
                ).as_quat()
                tgt_r_q = (
                    Rotation.from_quat(ee_r_q) * Rotation.from_quat(base_r_q)
                ).as_quat()
            else:
                tgt_l_pos = cur_l_pos + ee_l_pos
                tgt_r_pos = cur_r_pos + ee_r_pos
                tgt_l_q = (
                    Rotation.from_quat(ee_l_q) * Rotation.from_quat(cur_l_q)
                ).as_quat()
                tgt_r_q = (
                    Rotation.from_quat(ee_r_q) * Rotation.from_quat(cur_r_q)
                ).as_quat()

        delta_replay_kinematics.forward_kinematics(cur_left, cur_right)
        new_left, new_right = delta_replay_kinematics.inverse_kinematics(
            tgt_l_pos,
            tgt_l_q,
            tgt_r_pos,
            tgt_r_q,
            seeded=True,
            max_iters=60,
            err_threshold=1e-5,
        )

        max_step = 0.12
        new_left = cur_left + np.clip(new_left - cur_left, -max_step, max_step)
        new_right = cur_right + np.clip(new_right - cur_right, -max_step, max_step)

        delta_replay_joint_seed = {
            "left_joint_pos": new_left.copy(),
            "right_joint_pos": new_right.copy(),
        }

        out_action: dict[str, Any] = {
            "left_joint_pos": new_left.astype(np.float32),
            "right_joint_pos": new_right.astype(np.float32),
            "left_gripper_pos": np.asarray(
                delta_action["left_gripper_pos"], dtype=np.float32
            ),
            "right_gripper_pos": np.asarray(
                delta_action["right_gripper_pos"], dtype=np.float32
            ),
        }
        if "source" in delta_action:
            out_action["source"] = delta_action["source"]
        return out_action

    print("[Main] DBG: about to seed delta replay joint state", flush=True)
    _seed_delta_replay_joint_state(observation, reset_umi=True)
    print("[Main] DBG: seed done", flush=True)

    # For UMI mode, initialize joint seed via IK from first frame's EE pose.
    # Without this, the seed would be zeros (from env.reset zero_state),
    # causing FK(zeros) ≠ s_0 and every chunk-relative target to be wrong.
    if control_mode == "umi_ee_pose" and delta_replay_kinematics is not None:
        _inner_rp = policy
        while True:
            if hasattr(_inner_rp, "policy"):
                _inner_rp = _inner_rp.policy
            elif hasattr(_inner_rp, "base_policy"):
                _inner_rp = _inner_rp.base_policy
            else:
                break
        _umi_ee0 = getattr(_inner_rp, "umi_initial_ee", None)
        if _umi_ee0 is not None:
            _nom = np.zeros(6, dtype=np.float64)
            delta_replay_kinematics.forward_kinematics(_nom, _nom)
            _init_left, _init_right = delta_replay_kinematics.inverse_kinematics(
                np.asarray(_umi_ee0[0:3], dtype=np.float64),
                np.asarray(_umi_ee0[3:7], dtype=np.float64),
                np.asarray(_umi_ee0[8:11], dtype=np.float64),
                np.asarray(_umi_ee0[11:15], dtype=np.float64),
                seeded=True,
                max_iters=200,
                err_threshold=1e-6,
            )
            delta_replay_joint_seed = {
                "left_joint_pos": _init_left.copy(),
                "right_joint_pos": _init_right.copy(),
            }
            print("[Main] UMI: Initialized joint seed via IK from first frame EE")
            print(f"  left_joints:  {_init_left.round(4)}")
            print(f"  right_joints: {_init_right.round(4)}")

    def _save_scripted_joint_recording(clear_buffer: bool = True) -> None:
        """Persist scripted-policy joint trajectory if supported."""
        inner = policy
        while True:
            if hasattr(inner, "record_scripted_trajectory_to_parquet"):
                try:
                    inner.record_scripted_trajectory_to_parquet(
                        clear_buffer=clear_buffer, append_if_exists=True
                    )
                except Exception as exc:
                    print(f"[Main] WARNING: scripted joint record save failed: {exc}")
                return
            if hasattr(inner, "policy"):
                inner = inner.policy
            elif hasattr(inner, "base_policy"):
                inner = inner.base_policy
            else:
                return

    def _maybe_update_env_mode_from_action(
        env_handle: gym.Env,
        action: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        target_env = getattr(env_handle, "unwrapped", env_handle)
        current_mode = getattr(target_env, "control_mode", None)
        # Never override delta_ee_pose — its actions also contain ee_pos keys
        # but must be interpreted as deltas, not absolute cartesian.
        if current_mode == "delta_ee_pose":
            return
        if "left_ee_pos" in action and "right_ee_pos" in action:
            target_mode = "cartesian_position"
        else:
            return
        if current_mode == target_mode:
            return
        if hasattr(target_env, "set_control_mode"):
            target_env.set_control_mode(target_mode, observation=observation)
            print(f"[Main] Control mode updated to: {target_mode}")
        else:
            print("[Main] WARNING: env does not support set_control_mode")

    _dbg_loop_iter = 0
    _dbg_replay_started = False
    _dbg_replay_ended = False
    _dbg_replay_t0 = 0.0
    # Verbose-log the first 20 iterations unconditionally so we can see what
    # the state machine / replay policy is doing right after launch.
    _dbg_verbose_iters = 20

    # Replay-trajectory recorder: captures pre-step observation.state (14D
    # joint) and post-conversion joint action for every replay step the
    # wrapper is actually in "start" state. Saved to the path in
    # $YAM_DBG_REPLAY_TRACE_PATH if set.
    _dbg_trace_path = os.environ.get("YAM_DBG_REPLAY_TRACE_PATH", "")
    _dbg_trace_obs: list[np.ndarray] = []
    _dbg_trace_act: list[np.ndarray] = []
    _dbg_trace_step: list[int] = []

    def _dbg_flush_trace() -> None:
        if not _dbg_trace_path or not _dbg_trace_obs:
            return
        import pandas as pd
        obs_arr = np.stack(_dbg_trace_obs)
        act_arr = np.stack(_dbg_trace_act)
        df = pd.DataFrame({
            "step": _dbg_trace_step,
            "observation.state": list(obs_arr),
            "action": list(act_arr),
        })
        Path(_dbg_trace_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_dbg_trace_path, index=False)
        print(
            f"[DBG] Wrote replay trace: {_dbg_trace_path} ({len(df)} rows)",
            flush=True,
        )
    print(
        f"[Main] === Entering main control loop === "
        f"control_mode={control_mode!r} "
        f"env.control_mode={getattr(getattr(env, 'unwrapped', env), 'control_mode', '?')!r} "
        f"obs.left_jp[:3]={np.round(np.asarray(observation.get('left_joint_pos', np.zeros(6)))[:3], 3)}",
        flush=True,
    )

    # DBG headless: auto-start replay after a short pause so we can reproduce
    # the "bumps back home" issue without a browser. Remove after debugging.
    if os.environ.get("YAM_DBG_AUTOSTART_REPLAY") == "1":
        def _dbg_autostart():
            time.sleep(3.0)
            print("[DBG] Auto-enter_state('start')", flush=True)
            try:
                policy.enter_state("start")
            except Exception as exc:
                print(f"[DBG] auto-start failed: {exc}", flush=True)
        threading.Thread(target=_dbg_autostart, daemon=True).start()
    try:
        while True:
            observation.setdefault("annotation.task", cfg.task_description)
            action, policy_info = policy.get_action(observation)
            if action is None:
                continue
            action["__action_t"] = time.time()
            _dbg_loop_iter += 1
            _wrap_exec = getattr(policy, "execution_state", "?")
            # Emit one-shot boundary markers when replay actually starts/ends.
            if _wrap_exec == "start" and not _dbg_replay_started:
                _dbg_replay_started = True
                _dbg_replay_t0 = time.time()
                print(
                    f"[REPLAY STARTED t={_dbg_replay_t0:.3f}] "
                    f"first step={policy_info.get('current_step')}",
                    flush=True,
                )
            if policy_info.get("episode_done", False) and not _dbg_replay_ended:
                _dbg_replay_ended = True
                print(
                    f"[REPLAY ENDED t={time.time():.3f} "
                    f"elapsed={time.time() - _dbg_replay_t0:.2f}s] "
                    f"final step={policy_info.get('current_step')}",
                    flush=True,
                )
            _dbg_force = (
                _wrap_exec == "start"
                or policy_info.get("event") in ("home", "sync_to_init")
                or policy_info.get("episode_done", False)
            )
            if _dbg_verbose_iters > 0 or _dbg_force:
                if _dbg_verbose_iters > 0:
                    _dbg_verbose_iters -= 1
                _src = action.get("source", None)
                _oleft = np.asarray(
                    observation.get("left_joint_pos", np.zeros(6)), dtype=np.float64
                )
                _oright = np.asarray(
                    observation.get("right_joint_pos", np.zeros(6)), dtype=np.float64
                )
                _ee_str = ""
                if delta_replay_kinematics is not None:
                    try:
                        lp, lq, rp, rq = delta_replay_kinematics.forward_kinematics(
                            _oleft, _oright
                        )
                        _ee_str = (
                            f" | EE_L pos={np.round(lp, 3)} quat={np.round(lq, 3)} "
                            f"EE_R pos={np.round(rp, 3)} quat={np.round(rq, 3)}"
                        )
                    except Exception as exc:
                        _ee_str = f" | FK failed: {exc}"
                print(
                    f"[DBG iter={_dbg_loop_iter} t={time.time():.3f}] "
                    f"wrap={_wrap_exec!r} step={policy_info.get('current_step')} "
                    f"done={policy_info.get('episode_done', False)} "
                    f"src={_src!r} "
                    f"joint_L={np.round(_oleft, 3)} joint_R={np.round(_oright, 3)}"
                    f"{_ee_str}",
                    flush=True,
                )
            if "control_mode_changed" in policy_info:
                new_mode = policy_info["control_mode_changed"]
                try:
                    target_env = getattr(env, "unwrapped", env)
                    _env_mode = (
                        "joint_position"
                        if new_mode in ("umi_ee_pose", "delta_ee_pose")
                        else new_mode
                    )
                    if hasattr(target_env, "set_control_mode"):
                        target_env.set_control_mode(_env_mode, observation=observation)
                        print(
                            f"[Main] Control mode updated to: {new_mode} (env: {_env_mode})"
                        )
                        control_mode = str(new_mode)
                    else:
                        print("[Main] WARNING: env does not support set_control_mode")
                    # When switching to a delta mode mid-run, calibrate the env to
                    # the replay's initial state so deltas are applied correctly.
                    if new_mode in (
                        "joint_position",
                        "delta_joint_position",
                        "cartesian_position",
                        "delta_ee_pose",
                        "umi_ee_pose",
                    ):
                        if new_mode == "delta_ee_pose" and delta_replay_kinematics is None:
                            delta_replay_kinematics = YamKinematics()
                        # Reach through the wrapper stack to find the replay policy.
                        # Wrappers use either .policy or .base_policy for the inner policy.
                        inner = policy
                        while True:
                            if hasattr(inner, "policy"):
                                inner = inner.policy
                            elif hasattr(inner, "base_policy"):
                                inner = inner.base_policy
                            else:
                                break
                        if getattr(inner, "initial_state", None) is None:
                            print(
                                "[Main] Skipping calibration on mode change — "
                                "replay dataset not loaded yet."
                            )
                        else:
                            observation = _calibrate_for_delta_replay(
                                env,
                                inner,
                                new_mode,
                                observation,
                                zero_state,
                                cfg,
                                viser_push=getattr(policy, "_send_viser_message", None),
                            )
                            _dbg_verbose_iters = 15
                            print(
                                f"[DBG] post-calibration env.control_mode="
                                f"{getattr(getattr(env, 'unwrapped', env), 'control_mode', '?')!r}, "
                                f"inner.current_step="
                                f"{getattr(inner, 'current_step', '?')}, "
                                f"inner.num_steps="
                                f"{getattr(inner, 'num_steps', '?')}"
                            )
                            # Invalidate any cached pause hold_action — it was
                            # computed from the pre-calibration observation
                            # (arm near zero). If we don't clear it, the very
                            # next pause iteration commands the arm back to
                            # zero and the robot bumps home right after
                            # calibration completes.
                            if hasattr(policy, "_hold_action"):
                                policy._hold_action = None
                                print("[Main] Cleared stale hold_action after calibration")
                        _seed_delta_replay_joint_state(observation, reset_umi=True)
                        # Clear debug buffers so calibration settle frames aren't recorded.
                        _debug_actions.clear()
                        _debug_states.clear()
                        # IMPORTANT: Skip the rest of this iteration.  The `action`
                        # variable still holds the *stale* hold-action from BEFORE
                        # calibration (recorded when the robot was at home/zero).
                        # If we let env.step(action) run, it would command the
                        # robot back to zero, undoing the calibration.
                        # By continuing, the next iteration's get_action() will
                        # produce a fresh hold-action (or replay action) based on
                        # the post-calibration observation.
                        continue
                    if new_mode == "umi_ee_pose":
                        if delta_replay_kinematics is None:
                            delta_replay_kinematics = YamKinematics()
                        _seed_delta_replay_joint_state(observation, reset_umi=True)
                        _umi_inner = policy
                        while True:
                            if hasattr(_umi_inner, "policy"):
                                _umi_inner = _umi_inner.policy
                            elif hasattr(_umi_inner, "base_policy"):
                                _umi_inner = _umi_inner.base_policy
                            else:
                                break
                        _umi_ee0 = getattr(_umi_inner, "umi_initial_ee", None)
                        if _umi_ee0 is not None:
                            _nom = np.zeros(6, dtype=np.float64)
                            delta_replay_kinematics.forward_kinematics(_nom, _nom)
                            _il, _ir = delta_replay_kinematics.inverse_kinematics(
                                np.asarray(_umi_ee0[0:3], dtype=np.float64),
                                np.asarray(_umi_ee0[3:7], dtype=np.float64),
                                np.asarray(_umi_ee0[8:11], dtype=np.float64),
                                np.asarray(_umi_ee0[11:15], dtype=np.float64),
                                seeded=True,
                                max_iters=200,
                                err_threshold=1e-6,
                            )
                            delta_replay_joint_seed = {
                                "left_joint_pos": _il.copy(),
                                "right_joint_pos": _ir.copy(),
                            }
                            print("[Main] UMI mode switch: IK seed from first frame EE")
                        _debug_actions.clear()
                        _debug_states.clear()
                        continue
                except Exception as exc:
                    print(f"[Main] Failed to update control mode to {new_mode}: {exc}")
                    raise exc

            if "forward_time_ms" in policy_info:
                avg = forward_time_accumulator.add(policy_info["forward_time_ms"])
                if avg is not None:
                    print(f"[Main] Policy forward pass (1s avg): {avg:.1f}ms")
            policy_wants_reset = policy_info.get("event", None) == "home"
            replay_episode_done = bool(policy_info.get("episode_done", False))
            execution_state = str(policy_info.get("execution_state", ""))

            # sync_to_init: interpolate robot to the episode's first-frame state.
            # Works for all replay modes (joint_position, delta_*, umi_ee_pose).
            if policy_info.get("event") == "sync_to_init" and cfg.use_replay_policy:
                inner = policy
                while True:
                    if hasattr(inner, "policy"):
                        inner = inner.policy
                    elif hasattr(inner, "base_policy"):
                        inner = inner.base_policy
                    else:
                        break
                try:
                    observation = _calibrate_for_delta_replay(
                        env,
                        inner,
                        control_mode,
                        observation,
                        zero_state,
                        cfg,
                        viser_push=getattr(policy, "_send_viser_message", None),
                    )
                    if hasattr(policy, "_hold_action"):
                        policy._hold_action = None
                    print(
                        "[Main] sync_to_init complete — robot is at episode start pose."
                    )
                except Exception as exc:
                    print(f"[Main] sync_to_init failed: {exc}")
                continue
            is_policy_action = action.get("source", None) == "policy"
            if not policy_wants_reset:
                # In replay mode, do not send pause/hold actions to the env.
                # This guarantees we only execute actual dataset actions.
                if cfg.use_replay_policy and execution_state == "pause":
                    time.sleep(1.0 / max(1, cfg.policy_control_freq))
                    continue
                if cfg.use_replay_policy and control_mode in (
                    "delta_ee_pose",
                    "umi_ee_pose",
                ):
                    _pre_conv = {
                        k: np.asarray(v).copy() if isinstance(v, np.ndarray) else v
                        for k, v in action.items()
                        if k
                        in (
                            "left_ee_pos",
                            "left_ee_quat_xyzw",
                            "right_ee_pos",
                            "right_ee_quat_xyzw",
                            "_recorded_state_left_jp",
                            "_recorded_state_right_jp",
                        )
                    }
                    action = _convert_delta_ee_replay_action_to_joint(
                        action, observation
                    )
                    if _dbg_loop_iter < 200 and (
                        policy_info.get("current_step", 0) in (1, 2, 3, 50, 100)
                    ):
                        print(
                            f"[DBG conv step={policy_info.get('current_step')}] "
                            f"pre.left_ee_pos={_pre_conv.get('left_ee_pos')} "
                            f"pre.left_ee_quat={_pre_conv.get('left_ee_quat_xyzw')} "
                            f"pre.recorded_left_jp={_pre_conv.get('_recorded_state_left_jp')} "
                            f"post.left_joint_pos="
                            f"{np.asarray(action.get('left_joint_pos', np.zeros(6)))} "
                            f"post.right_joint_pos="
                            f"{np.asarray(action.get('right_joint_pos', np.zeros(6)))}",
                            flush=True,
                        )
                # Capture pre-step state for debug logging (before env.step mutates observation)
                _pre_step_state = np.concatenate(
                    [
                        observation.get("left_joint_pos", np.zeros(6)),
                        observation.get("left_gripper_pos", np.zeros(1)),
                        observation.get("right_joint_pos", np.zeros(6)),
                        observation.get("right_gripper_pos", np.zeros(1)),
                    ]
                ).astype(np.float64)

                _maybe_update_env_mode_from_action(env, action, observation)
                # Trace: record whenever the replay policy is actually producing
                # a replay frame (source=='policy' with a real current_step).
                # Covers the last frame too, where the wrapper auto-transitions
                # to pause but the action is still a replay action.
                _dbg_record = (
                    _dbg_trace_path
                    and action.get("source") == "policy"
                    and "left_joint_pos" in action
                    and "right_joint_pos" in action
                    and policy_info.get("current_step") is not None
                )
                if _dbg_record:
                    _dbg_trace_obs.append(
                        np.concatenate(
                            [
                                np.asarray(observation.get("left_joint_pos", np.zeros(6))),
                                np.asarray(observation.get("left_gripper_pos", np.zeros(1))),
                                np.asarray(observation.get("right_joint_pos", np.zeros(6))),
                                np.asarray(observation.get("right_gripper_pos", np.zeros(1))),
                            ]
                        ).astype(np.float64)
                    )
                    _dbg_trace_act.append(
                        np.concatenate(
                            [
                                np.asarray(action["left_joint_pos"]),
                                np.asarray(action.get("left_gripper_pos", np.zeros(1))),
                                np.asarray(action["right_joint_pos"]),
                                np.asarray(action.get("right_gripper_pos", np.zeros(1))),
                            ]
                        ).astype(np.float64)
                    )
                    _dbg_trace_step.append(int(policy_info.get("current_step", -1)))
                observation, reward, terminated, truncated, env_info = env.step(action)
                if _dbg_record and policy_info.get("episode_done", False):
                    _dbg_flush_trace()
                # UMI uses feed-forward seeding (IK output → next seed) to
                # match the verified offline script.  Observation re-seeding
                # introduces jitter from tracking error, causing shaking.
                if control_mode != "umi_ee_pose":
                    _seed_delta_replay_joint_state(observation)

                # Collect debug data: post-IK joint action + pre-step state.
                # Only record actual replay steps — skip once episode is done
                # (otherwise the last delta action keeps being applied, causing drift).
                if (
                    "_debug_joint_action" in env_info
                    and not _debug_episode_done
                    and is_policy_action
                ):
                    _debug_actions.append(env_info["_debug_joint_action"])
                    _debug_states.append(_pre_step_state)
                    if policy_info.get("episode_done", False):
                        _debug_episode_done = True
                        _save_debug_parquet()
                        print(
                            f"\033[34m[Debug] Replay episode done — "
                            f"recorded {len(_debug_actions)} frames. "
                            f"Debug parquet saved automatically.\033[0m"
                        )

                # Manual recording via footswitch (pedal 2=start, pedal 0=save)
                if cfg.record_episode:
                    if policy_info.get("start_pressed") and not is_recording:
                        observation, _ = env.reset(
                            options={"task_name": cfg.task_description}
                        )
                        is_recording = True
                        policy._is_recording = True
                        print(
                            f"\033[32m[Main] Recording started (task: {cfg.task_description})\033[0m"
                        )
                    elif policy_info.get("save_pressed") and is_recording:
                        observation, _ = env.reset(options={"start_new_episode": False})
                        is_recording = False
                        policy._is_recording = False
                        last_ep = getattr(env, "last_episode_dir", None)
                        policy._last_episode_dir = str(last_ep) if last_ep else ""
                        print(f"\033[34m[Main] Recording saved → {last_ep}\033[0m")

            if policy_wants_reset or terminated or truncated:
                _save_debug_parquet()
                if is_recording:
                    print(
                        "\033[33m[Main] Home during recording → discarding episode\033[0m"
                    )

            # For replay mode, force reset immediately after the final valid frame.
            # This guarantees exactly N executed replay steps for an N-row dataset.
            # NOTE: `replay_episode_done` used to trigger auto-reset-to-home here,
            # which made the arm swing back to zero the moment the replay
            # finished. That was jarring and unsafe. Home now only fires on an
            # explicit user-initiated `home` event (UI button / footswitch) or a
            # genuine env termination. The replay policy already auto-pauses on
            # episode_done, so the arm holds at the final pose until the user
            # decides what to do.
            if policy_wants_reset or terminated or truncated:
                print(
                    f"[Main] Reset triggered: policy_wants_reset={policy_wants_reset} "
                    f"terminated={terminated} truncated={truncated} "
                    f"replay_episode_done={replay_episode_done} "
                    f"execution_state={execution_state!r}"
                )
                # Save debug recording before reset clears the episode
                _save_debug_parquet()
                _save_scripted_joint_recording(clear_buffer=True)
                policy_info = policy.reset()
                task_name = policy_info["task_name"]
                assert isinstance(task_name, str)
                # Slow ramp to zero_state BEFORE env.reset teleports the sim
                # (or fast-ramps the real robot). This guarantees the home
                # motion is visible and safe on both backends.
                observation = _waypoint_ramp_to_joint_state(
                    env,
                    observation,
                    zero_state,
                    cfg.policy_control_freq,
                    viser_push=getattr(policy, "_send_viser_message", None),
                    label="home",
                )
                print("Starting env reset")
                observation, env_info = env.reset(
                    options=dict(
                        target_joint_position=deepcopy(zero_state),
                        # Used by RecordEpisodeWrapper
                        task_name=task_name,
                        # Only explicit Start/footswitch actions should begin a
                        # new eval episode. Resetting for Home, termination, or
                        # replay episode rollover must not silently open a fresh
                        # recording window, otherwise the home-between-evals
                        # segment gets saved as an extra episode on the next Start.
                        start_new_episode=is_recording,
                        # Used by RecordEpisodeWrapper
                        # We need this to actually cause the inner env to reset
                        force_reset=True,
                        # Used by RecordEpisodeWrapper, set in StartStopPlayPolicyWrapper
                        discard_episode=policy_info.get("discard_episode", False)
                        or is_recording,
                    )
                )
                is_recording = False
                policy._is_recording = False
                if cfg.use_fello:  # Home Fello
                    fello_policy.slow_home(zero_state)
                _seed_delta_replay_joint_state(observation, reset_umi=True)
                print("Finished env reset")
    except KeyboardInterrupt:
        print(
            "\n[Main] Ctrl+C received — attempting safe home to zero_state before exit..."
        )
        try:
            _save_debug_parquet()
            _save_scripted_joint_recording(clear_buffer=True)
            env.reset(
                options=dict(
                    target_joint_position=deepcopy(zero_state),
                    force_reset=True,
                    start_new_episode=False,
                    discard_episode=True,
                )
            )
            if cfg.use_fello:
                fello_policy.slow_home(zero_state)
                print("[Main] Fello home complete.")
            print("[Main] Safe home complete.")
        except Exception as exc:
            print(f"[Main] WARNING: Safe home on Ctrl+C failed: {exc}")
    finally:
        _save_debug_parquet()
        _save_scripted_joint_recording(clear_buffer=True)
        env.close()


if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
