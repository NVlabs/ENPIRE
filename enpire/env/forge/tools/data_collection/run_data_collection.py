"""
# Run data collection:
uv run launch.py --mode=data_collection --use-fello

# Run data collection with full-arm force feedback:
uv run launch.py --mode=data_collection --use-fello --force-feedback

# Ruu data collection with voice annotations:
uv run launch.py --mode=data_collection --use-fello --use-voice

# Or directly:
uv run python tools/data_collection/run_data_collection.py --station=1 --use-voice

"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enpire.env.forge.tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPO_ROOT, required_modules=["gymnasium", "tyro"])

# Ensure prints show up immediately over SSH/uv
sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym
from gymnasium.envs.registration import register
from prompt_toolkit.shortcuts import radiolist_dialog
import tyro

from enpire.policy.rl.record_episode_wrapper import RecordEpisodeWrapper
from teleop_policy import DEFAULT_FORCE_FEEDBACK_RATIOS, TeleopPolicy
from timing_jsonl import TimingJsonlLogger
from viser_env_wrapper import ViserEnvWrapper
from enpire.env.forge.display_utils import ImageDisplayer, put_latest_image
from enpire.env.forge.tools.teleop_voice_annotate.voice_annotation import start_voice_annotation_thread
import queue
import numpy as np


@dataclass
class DataCollectionConfig:
    operator: str | None = None
    """Username of operator."""

    task_list_path: str | None = None
    """Path to .txt task list file containing one task per line."""

    station: int | None = None
    """Station number. If not provided, will get station number from hostname."""

    display_image: bool = False
    """Whether to display the image in a window."""

    visualize: bool = False
    """Whether to visualize the environment using Viser."""

    use_voice: bool = False
    """Enable microphone speech-to-text for on-the-fly voice annotations."""

    voice_trigger_mode: bool = True
    """Enable trigger word detection (e.g., 'start', 'begin', 'end', 'done') for precise temporal alignment."""

    voice_silence_duration: float = 0.5
    """Seconds of silence before finalizing speech. Increase if your sentences get cut off mid-pause."""

    voice_input_device_index: int | None = None
    """Optional audio input device index for the microphone."""

    save_annotation: bool = False
    """Auto-generate annotated video after saving an episode."""

    use_fello: bool = False
    """Use Fello arms (footswitch-based) instead of YAM leader arms."""

    force_feedback: bool = False
    """Mirror YAM torque-minus-gravity back to Fello as extra feedforward torque."""

    no_translation_mode: bool = False
    """Disable translation-only mode; left-pedal buttons are unbound."""

    force_feedback_ratios: tuple[float, float, float, float, float, float, float] = (
        DEFAULT_FORCE_FEEDBACK_RATIOS
    )
    """Per-joint force-feedback intensity ratios. Each value must be <= 1/3."""

    policy_control_freq: float = 30.0
    """Follower command/update frequency in Hz for data collection."""

    timing_debug: bool = False
    """Print slow-step timing diagnostics and periodic summaries."""

    timing_warn_ms: float = 50.0
    """Warn when any timed phase exceeds this threshold in milliseconds."""

    timing_summary_every: int = 300
    """Print aggregated timing summary every N loop iterations when timing_debug is enabled."""

    timing_log_dir: str = "/tmp/yam_timing"
    """Directory for persisted timing JSONL logs when timing_debug is enabled."""

    data_saving_path: str | None = None
    """Override for the base directory where episodes are saved. Defaults to $YAM_RAW_PATH."""

    no_top: bool = False
    """Disable the top camera (use only left/right wrist cameras)."""


def prompt_missing_fields(cfg: DataCollectionConfig) -> DataCollectionConfig:
    # Operator username
    if cfg.operator is None:
        cfg.operator = input("Enter operator username: ")

    # Station number
    if cfg.station is None:
        hostname = socket.gethostname()
        # assert hostname.startswith(
        #     "gear-ax8-max-"
        # ), "If not using gear-ax8-max-*, station number must be specified using --station."
        cfg.station = int(hostname.split("-")[-1])

    return cfg


def get_task_list(cfg: DataCollectionConfig) -> list[str]:
    # Load task list from file if provided
    if cfg.task_list_path is not None:
        print(f"[INFO] Loading task list from {cfg.task_list_path}")
        with open(cfg.task_list_path, "r") as f:
            task_list = [line.strip() for line in f if len(line.strip()) > 0]
        assert len(task_list) > 0, "Task list should not be empty"
        for i, task in enumerate(task_list, start=1):
            print(f"{i}. {task}")

    # Prompt operator for one task name to use for all episodes
    else:
        task_name = input("Enter task name: ").strip()
        print(f"[INFO] Using task name: {task_name}")
        task_list = [task_name]

    return task_list


def prompt_task_name(task_list: list[str]) -> str:
    if len(task_list) == 1:
        return task_list[0]

    task_name = None
    while task_name is None:
        # Shuffle task list
        shuffled_task_list = task_list.copy()
        random.shuffle(shuffled_task_list)

        # Operator should select the first valid task from the shuffled list
        task_name = radiolist_dialog(
            title="Task selection",
            text="Please select the first valid task:",
            values=[(task, task) for task in shuffled_task_list],
        ).run()

    print(f"[INFO] Using task name: {task_name}")
    return task_name


def sanitize_for_path(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def main(cfg: DataCollectionConfig):
    loop_jsonl_logger = TimingJsonlLogger(
        enabled=cfg.timing_debug,
        component="run_data_collection",
        label="main_loop",
        warn_ms=0.0,
        summary_every=0,
        log_dir=cfg.timing_log_dir,
    )

    def _record_loop_timing(samples: dict[str, float]) -> None:
        if not cfg.timing_debug:
            return

        loop_jsonl_logger.record("loop_step", samples)

        timing_state["count"] += 1
        for name, value in samples.items():
            timing_state["sum_ms"][name] = timing_state["sum_ms"].get(name, 0.0) + value
            timing_state["max_ms"][name] = max(
                timing_state["max_ms"].get(name, 0.0), value
            )

        if cfg.timing_warn_ms > 0:
            slow = {
                name: value
                for name, value in samples.items()
                if value >= cfg.timing_warn_ms
            }
            if slow:
                slow_str = ", ".join(
                    f"{name}={value:.1f}ms" for name, value in sorted(slow.items())
                )
                print(f"[Timing][Loop] slow step {timing_state['count']}: {slow_str}")

        if (
            cfg.timing_summary_every > 0
            and timing_state["count"] % cfg.timing_summary_every == 0
        ):
            avg_str = ", ".join(
                f"{name}={timing_state['sum_ms'][name] / timing_state['count']:.1f}ms"
                for name in sorted(timing_state["sum_ms"])
            )
            max_str = ", ".join(
                f"{name}={timing_state['max_ms'][name]:.1f}ms"
                for name in sorted(timing_state["max_ms"])
            )
            print(
                f"[Timing][Loop] {timing_state['count']} samples | avg {avg_str} | max {max_str}"
            )

    def _render_annotations_async(episode_dir: Path) -> None:
        script_path = (
            REPO_ROOT / "tools" / "teleop_voice_annotate" / "visualize_annotations.py"
        )
        if not script_path.exists():
            print(
                f"[WARN] tools/teleop_voice_annotate/visualize_annotations.py not found at {script_path}",
                flush=True,
            )
            return

        def _worker() -> None:
            print(f"[INFO] Rendering annotated video for {episode_dir}", flush=True)
            cmd = [
                sys.executable,
                str(script_path),
                "--episode-dir",
                str(episode_dir),
                "--output",
                "annotated.mp4",
            ]
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout.strip(), flush=True)
                if result.stderr:
                    print(result.stderr.strip(), flush=True)
                print(f"[INFO] Annotated video saved in {episode_dir}", flush=True)
            except subprocess.CalledProcessError as exc:
                err = exc.stderr.strip() if exc.stderr else str(exc)
                print(f"[WARN] Annotation render failed: {err}", flush=True)

        threading.Thread(target=_worker, daemon=True).start()

    # Prompt for missing fields
    cfg = prompt_missing_fields(cfg)

    # Get task list
    task_list = get_task_list(cfg)
    task_name = prompt_task_name(task_list)

    # Data collection settings
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if cfg.data_saving_path is not None:
        base_output_dir = Path(cfg.data_saving_path).expanduser()
    elif "YAM_RAW_PATH" in os.environ:
        base_output_dir = Path(os.environ["YAM_RAW_PATH"])
    else:
        raise RuntimeError(
            "Set --data-saving-path or the YAM_RAW_PATH environment variable"
        )
    operator_prefix = sanitize_for_path(cfg.operator or "unknown")
    task_prefix = sanitize_for_path(task_name)
    output_dir = (
        base_output_dir
        / f"{operator_prefix}_{task_prefix}_{timestamp}-YAM-{cfg.station:02d}"
    )
    print(f"[INFO] Starting data collection for operator: {cfg.operator}")
    print(f"[INFO] Local output directory: {output_dir}")

    # Environment
    register(id="YamReal-v0", entry_point="enpire.env.forge.robot.yam.yam_real_env:YamRealEnv")
    print(f"[INFO] Using policy control frequency: {cfg.policy_control_freq:.1f} Hz")
    enabled_camera_names = ("left", "right") if cfg.no_top else ("top", "left", "right")
    env = gym.make(
        "YamReal-v0",
        policy_control_freq=cfg.policy_control_freq,
        enabled_camera_names=enabled_camera_names,
    )

    # Visualization (very slow, disable if not needed)
    if cfg.visualize:
        env = ViserEnvWrapper(env)

    display_image = cfg.display_image
    if display_image:
        img_queue = queue.Queue(maxsize=1)
        displayer = ImageDisplayer(img_queue, "Record Demos", size=256)
        displayer.start()  # Record episodes to disk

    env = RecordEpisodeWrapper(
        env,
        output_dir=str(output_dir),
        operator=cfg.operator,
    )

    # Optional voice annotation thread
    voice_thread = None
    voice_stop_event = threading.Event()
    apply_cached_voice_label = None
    if cfg.use_voice:
        voice_thread, voice_stop_event, apply_cached_voice_label = (
            start_voice_annotation_thread(
                env,
                trigger_mode=cfg.voice_trigger_mode,
                silence_duration=cfg.voice_silence_duration,
                input_device_index=cfg.voice_input_device_index,
            )
        )

    # Policy
    if not cfg.use_fello:
        raise RuntimeError(
            "run_data_collection now expects right-arm buttons [0,1,2] for save/discard/start. "
            "Run with --use-fello."
        )
    policy = TeleopPolicy(
        use_fello=cfg.use_fello,
        force_feedback=cfg.force_feedback,
        force_feedback_ratios=cfg.force_feedback_ratios,
        disable_translation_mode=cfg.no_translation_mode,
        timing_debug=cfg.timing_debug,
        timing_warn_ms=cfg.timing_warn_ms,
        timing_summary_every=cfg.timing_summary_every,
        timing_log_dir=cfg.timing_log_dir,
    )
    print("[INFO] Leader arms are ready. Sampling current pose to sync follower arms.")
    initial_action, _ = policy.get_action(None)

    # Main loop
    is_recording = False
    timing_state = {"count": 0, "sum_ms": {}, "max_ms": {}}
    obs, _ = env.reset(
        options={"alias": "home", "start_new_episode": False}
    )
    print(
        "[INFO] Follower arms are now in sync. "
        "Use right-arm buttons: 2=start, 0=save, 1=discard."
    )
    try:
        while True:
            loop_t0 = time.perf_counter()
            policy_t0 = time.perf_counter()
            action, policy_info = policy.get_action(obs)
            # print(action)
            policy_dt_ms = (time.perf_counter() - policy_t0) * 1000.0
            action["__action_t"] = time.time()
            env_step_t0 = time.perf_counter()
            obs, _, _, _, step_info = env.step(action)
            env_step_dt_ms = (time.perf_counter() - env_step_t0) * 1000.0

            post_step_ui_t0 = time.perf_counter()
            if display_image:
                image_keys = [
                    "left_camera_image",
                    "right_camera_image",
                ]
                concat_img = np.concatenate([obs[key] for key in image_keys], axis=1)
                padded = np.zeros(
                    (concat_img.shape[0] + 8, concat_img.shape[1] + 8, 3),
                    dtype=concat_img.dtype,
                )
                if is_recording:
                    padded[:, :, 1] = 255
                padded[4:-4, 4:-4] = concat_img
                concat_img = padded
                put_latest_image(img_queue, concat_img)
            post_step_ui_dt_ms = (time.perf_counter() - post_step_ui_t0) * 1000.0

            timing_samples = {
                "policy_call_ms": policy_dt_ms,
                "env_step_ms": env_step_dt_ms,
                "post_step_ui_ms": post_step_ui_dt_ms,
                "loop_total_ms": (time.perf_counter() - loop_t0) * 1000.0,
            }
            policy_timing = policy_info.get("__timing")
            if isinstance(policy_timing, dict):
                for name, value in policy_timing.items():
                    if isinstance(value, (int, float)):
                        timing_samples[name] = float(value)
            env_timing = step_info.get("__timing")
            if isinstance(env_timing, dict):
                for name, value in env_timing.items():
                    if isinstance(value, (int, float)):
                        timing_samples[name] = float(value)
            _record_loop_timing(timing_samples)

            # Start new recording
            if "start" in policy_info:
                if not is_recording:
                    obs, _ = env.reset(options={"task_name": task_name, "start_new_episode": True})
                    is_recording = True
                    if apply_cached_voice_label is not None:
                        apply_cached_voice_label()
                    print(f"[INFO] Recording started for task: {task_name}")
                    print("[INFO] Use right-arm buttons: 0=save, 1=discard.")
                else:
                    print(
                        "[INFO] Recording already in progress. Did not start new recording."
                    )

            # Discard current recording
            elif "discard" in policy_info:
                if is_recording:
                    obs, _ = env.reset(
                        options={"discard_episode": True, "start_new_episode": False}
                    )
                    is_recording = False
                    print(
                        "[INFO] Recording discarded. Press right-arm button 2 to start new recording."
                    )
                else:
                    print("[INFO] No recording in progress.")

            # Save current recording
            elif "save" in policy_info:
                if is_recording:
                    print("[INFO] Saving current recording...")
                    obs, _ = env.reset(options={"start_new_episode": False})
                    is_recording = False
                    if cfg.save_annotation:
                        episode_dir = getattr(env, "last_episode_dir", None)
                        if episode_dir is not None:
                            _render_annotations_async(Path(episode_dir))
                        else:
                            print(
                                "[WARN] No saved episode directory found.", flush=True
                            )
                    print(
                        f"[INFO] Recording saved to {getattr(env, 'last_episode_dir', None)}. Press right-arm button 2 to start new recording."
                    )
                else:
                    print("[INFO] No recording in progress.")
            else:
                pass

    finally:
        voice_stop_event.set()
        env.close()


if __name__ == "__main__":
    main(tyro.cli(DataCollectionConfig))
