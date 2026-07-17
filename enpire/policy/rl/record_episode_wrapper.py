from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import queue
import shutil
import socket
import tempfile
import threading
import time
from typing import Any
import uuid
import warnings
import multiprocessing as mp
import gymnasium as gym
import numpy as np
from enpire.env.forge.tools.data_collection.async_video_compression import _video_worker_loop
from enpire.env.forge.robot.constants import DEFAULT_COMPRESSED_VIDEO_SHAPE


@dataclass
class _PendingEpisode:
    """Immutable snapshot of one finished episode handed to the background saver thread.

    reset() flushes the video writers + captures all per-step buffers and the env-derived
    metadata SYNCHRONOUSLY (the saver thread must never touch self.env — that's the live
    robot), then rebinds the wrapper's buffers to fresh objects so the next episode records
    independently while this snapshot is written to disk off the control thread.
    """

    tmp_dir_obj: Any  # tempfile.TemporaryDirectory — kept alive until moved
    tmp_dir: str
    timestamps: list
    observations: dict
    actions: dict
    action_sources: list
    component_timestamps: list
    rewards: list
    dones: list
    metadata: dict  # precomputed (reads env) on the control thread
    control_period: float
    episode_dir: Path


def _append_dictionaries(dest: dict[str, list[Any]], src: dict[str, Any]):
    """Appends a dictionary of items to a list of dictionaries.

    Ensures sure that the set of keys does not change after the first append.
    """
    if dest:
        assert set(dest.keys()) == set(src.keys()), f"Set of keys changed from {dest.keys()} to {src.keys()}"
    else:
        for k, v in src.items():
            dest[k] = []
    for k, v in src.items():
        dest[k].append(v)


class RecordEpisodeWrapper(gym.Wrapper):
    """Gym wrapper that records YAM environment episodes to disk.

    Each call to `reset()` finalizes the ongoing episode and starts recording a new one.
    By default, every recorded episode is saved to disk.

    Available reset options:
    - discard_episode: Discard the ongoing episode without saving to disk.
    - start_new_episode: After reset, the next episode will be recorded.
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str = "data/debug",
        operator: str | None = None,
        policy_config: dict | None = None,
        record: bool = True,
    ):
        super().__init__(env)
        # record=False -> transparent passthrough: never start recording, never save to disk,
        # never spawn the async H.264 writer. Used by --eval-only (policy inference, no data
        # collection -> no per-reset save/encode -> no control-loop overruns).
        self._record = record
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n\033[92mDirectory created/verified: {self.output_dir}\033[0m")
        except OSError as e:
            raise PermissionError(f"\n\033[91mDirectory {self.output_dir} is not writable: {e}\033[0m")

        self.task_name = None
        self.operator = operator
        self.policy_config = policy_config or {}
        self.last_episode_dir: Path | None = None

        # Recording state
        self.is_recording = False

        # Episode data
        self.tmp_episode_dir = None
        self.timestamps: list[float] = []
        self.observations: dict[str, list[np.ndarray]] = {}
        self.actions: dict[str, list[np.ndarray]] = {}
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.terminal_event: str | None = None

        # Per-step action source labels ("human" / "policy" / "unknown")
        self.action_sources: list[str] = []

        # Per-step per-component creation timestamps
        self.component_timestamps: list[dict[str, float]] = []
        self._prev_obs_timestamps: dict[str, float] | None = None

        # Previous step data
        self.prev_obs = None
        self.prev_action = None

        # Async video writer/converter process.
        self._video_queue: mp.Queue = mp.Queue()
        self._video_ack_queue: mp.Queue = mp.Queue()
        self._video_process = None
        # Background episode saver: the npy writes + shutil.move + convert-enqueue for a finished
        # episode run HERE, off the control thread, so env.reset (which starts the next episode's
        # hover + learn) never blocks on the previous episode's disk I/O. A single worker thread
        # keeps episodes saved in finish order; the wrapper hands it _PendingEpisode snapshots.
        self._save_queue: queue.Queue = queue.Queue()
        self._save_thread = None
        # Serializes flushes so a control-thread discard flush and the saver's per-episode flush
        # never both wait on the shared ack queue at once (which could cross their ack tokens).
        self._flush_lock = threading.Lock()
        if self._record:
            self._video_process = mp.Process(
                target=_video_worker_loop,
                args=(
                    self._video_queue,
                    self._video_ack_queue,
                    DEFAULT_COMPRESSED_VIDEO_SHAPE,
                ),
                daemon=True,
                name="video-writer",
            )
            self._video_process.start()
            self._save_thread = threading.Thread(target=self._saver_loop, daemon=True, name="episode-saver")
            self._save_thread.start()

    @property
    def frame_count(self) -> int:
        """Return the number of frames recorded so far in the current episode."""
        return len(self.timestamps)

    def set_output_dir(self, new_dir: Path | str) -> None:
        """Rotate the on-disk output directory.

        If an episode is in flight, it is discarded (matching the user-confirmed
        behavior for /restart mid-learn). Safe because ``output_dir`` is only
        consulted inside ``_save_episode_data`` at the ``shutil.move`` line; the
        async video worker uses ``tmp_episode_dir``, not ``output_dir``.
        """
        if self.is_recording:
            self._enqueue_save(discard_episode=True)
            self.is_recording = False
        new_dir = Path(new_dir)
        new_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = new_dir
        print(f"\n\033[92m[Record] Output directory rotated to: {new_dir}\033[0m")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.is_recording:
            # Record timestamp
            self.timestamps.append(time.time())

            # Record observations
            assert self.prev_obs is not None
            non_video_obs = {}
            for k, v in self.prev_obs.items():
                if getattr(v, "ndim", None) == 3:
                    self._write_video_frame(k, v)
                else:
                    non_video_obs[k] = v
            _append_dictionaries(self.observations, non_video_obs)

            # Record actions and per-step source label
            self.action_sources.append(action.get("source", "unknown"))
            # Strip non-numeric keys before recording (source and timestamps tracked separately)
            action_to_record = {k: v for k, v in action.items() if k not in ("source", "__action_t")}
            _append_dictionaries(self.actions, action_to_record)

            self.rewards.append(float(reward))
            self.dones.append(bool(terminated or truncated))

            # Collect per-component creation timestamps
            step_ts: dict[str, float] = {}
            if isinstance(self._prev_obs_timestamps, dict):
                step_ts.update(self._prev_obs_timestamps)
            action_t = action.get("__action_t")
            if action_t is not None:
                step_ts["action"] = float(action_t)
            self.component_timestamps.append(step_ts)

        self.prev_obs = obs
        self.prev_action = action
        self._prev_obs_timestamps = info.get("__timestamps")

        return obs, reward, terminated, truncated, info

    def _write_video_frame(self, obs_key: str, frame: np.ndarray):
        assert self.tmp_episode_dir is not None
        self._video_queue.put(
            (
                "frame",
                obs_key,
                frame.copy(),  # copy: frame buffer may be reused by camera driver
                self.tmp_episode_dir.name,
                self.env.unwrapped.policy_control_freq,
            )
        )

    def _flush_video_writers(self, tmp_dir: str | None = None) -> None:
        """Block until queued frames are written and writers released. tmp_dir=None releases
        all writers (shutdown); a tmp_dir releases only that episode's writers, so this can run
        in the background saver while the next episode keeps recording."""
        with self._flush_lock:
            token = uuid.uuid4().hex
            self._video_queue.put(("flush", token, tmp_dir))
            while self._video_ack_queue.get() != token:
                pass

    def reset(self, seed=None, options=None):
        options = options if options is not None else {}

        # Finalize ongoing episode
        if self.is_recording:
            # If discard, then discard. Otherwise, save (off the control thread).
            discard_episode = options.get("discard_episode")
            self._apply_terminal_label(options)
            self._enqueue_save(discard_episode=discard_episode)
            self.is_recording = False

        # Start recording new episode. record=False (eval-only) -> never start, so is_recording
        # stays False and step()/save are all no-ops (transparent passthrough).
        start_new_episode = options.get("start_new_episode")
        if start_new_episode and self._record:
            self._start_new_episode()  # clear old data buffer
            self.is_recording = True
            assert "task_name" in options, "task_name is required for new episode"
            self.task_name = options["task_name"]
        else:
            self.task_name = None

        # Note: We only call call super().reset() for the very first episode in teleoperation,
        # or each time an RL episode need a reset.
        # For subsequent episodes in teleoperation or the in-episode steps,
        # we don't reset the super class which contains the gym environment, and the reset here only means saving or discarding episodic data
        # according to the teleoperator or human in the RL loop.
        if (
            self.prev_action is None  # for the start of episode in teleoperation
            or options.get("force_reset", False)
            or any(
                k in options for k in ("alias", "target_joint_position", "target_ee_pose")
            )  # for RL reset between episodes
        ):
            obs, info = super().reset(seed=seed, options=options)
        else:  # for continuing teleoperation after the first step
            obs, _, _, _, info = super().step(self.prev_action)

        self.prev_obs = obs
        self._prev_obs_timestamps = info.get("__timestamps")

        return obs, info

    def _start_new_episode(self):
        print("\033[32m[Record] Started recording episode\033[0m")
        self.tmp_episode_dir = tempfile.TemporaryDirectory()
        self.timestamps.clear()
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.terminal_event = None
        self.action_sources.clear()
        self.component_timestamps.clear()

    def _apply_terminal_label(self, options: dict[str, Any]) -> None:
        if not self.rewards or not self.dones:
            return
        if "episode_terminal_reward" in options:
            self.rewards[-1] = float(options["episode_terminal_reward"])
        if "episode_terminal_done" in options:
            self.dones[-1] = bool(options["episode_terminal_done"])
        terminal_event = options.get("episode_terminal_event")
        if terminal_event is not None:
            self.terminal_event = str(terminal_event)

    def truncate_in_memory(self, keep: int) -> None:
        """Slice every per-step in-memory buffer down to ``keep`` rows.

        Used by late-success terminators (e.g. the auto-reward worker, which
        detects success ~300 ms after it happens — so by the time the runner
        calls env.reset the in-memory buffer holds 3-4 extra post-success
        steps that would otherwise be written to disk and then have to be
        trimmed off again). Calling this BEFORE env.reset's
        ``_apply_terminal_label`` + ``_save_episode_data`` makes the next
        on-disk episode dir come out at the correct length, with the
        terminal stamp landing at the SAM3-success step rather than the
        whatever-current step env happens to be on.

        Per-frame mp4 frames are NOT touched — they live in the async
        ``_video_queue`` writer pipeline, not in the in-memory buffers this
        method owns. Callers wanting the mp4 trimmed too still need to
        post-process the videos (e.g. via ``rl/auto_reward.py``'s
        background ``_trim_mp4s_in_dir``, which waits for the convert
        worker's ``.videos_converted`` marker and does a two-phase commit
        across every mp4 in the episode dir).

        No-op outside an active recording or when ``keep`` already covers
        the full buffer length.
        """
        if not self.is_recording or keep < 0:
            return
        keep = int(keep)
        if keep >= len(self.timestamps):
            return
        # Per-step LIST buffers — slice in-place so any caller-held reference
        # to the list stays valid.
        for buf in (self.timestamps, self.rewards, self.dones, self.action_sources, self.component_timestamps):
            del buf[keep:]
        # Per-key DICT-of-lists buffers. Both ``self.observations`` and
        # ``self.actions`` are ``{component_name: [per-step value, ...]}``
        # (see _append_dictionaries in step()), so we trim every inner list.
        for d in (self.observations, self.actions):
            for lst in d.values():
                if isinstance(lst, list):
                    del lst[keep:]

    def finalize_episode(self, discard_episode: bool = False, **terminal_options: Any) -> None:
        """Finalize the active recording without stepping or resetting the robot."""
        if not self.is_recording:
            return
        self._apply_terminal_label(terminal_options)
        self._enqueue_save(discard_episode=discard_episode)
        self.is_recording = False
        self.task_name = None

    def _reset_episode_buffers(self) -> None:
        """Rebind the per-episode buffers to FRESH objects so a snapshot handed to the saver
        thread is never mutated by the next episode's step() (the old objects go with the
        snapshot; the new episode appends to these)."""
        self.tmp_episode_dir = None
        self.timestamps = []
        self.observations = {}
        self.actions = {}
        self.rewards = []
        self.dones = []
        self.action_sources = []
        self.component_timestamps = []
        self.terminal_event = None

    def _enqueue_save(self, discard_episode: bool = False):
        """Finalize the in-flight episode on the CONTROL thread as cheaply as possible, then hand
        EVERYTHING heavy — video flush, npy writes, move, H.264 convert — to the background saver
        so the next episode's reset/learn isn't blocked. The control thread only: captures
        last_episode_dir + env-derived metadata (auto_eval/auto_reward read last_episode_dir
        immediately; the saver thread must never touch self.env) and snapshots the buffers.
        A [Record][timing] line reports the control-thread cost so overruns can be attributed."""
        assert self.is_recording
        assert self.tmp_episode_dir is not None
        t0 = time.perf_counter()

        if not self.observations or not self.actions:
            print("No observations or actions recorded")
            discard_episode = True

        if discard_episode:
            # Rare (park / restart / empty). Release this episode's writers + drop the tmp dir
            # synchronously — not on the per-episode-end hot path, so cost here doesn't matter.
            self._flush_video_writers(self.tmp_episode_dir.name)
            print(f"\033[33m[Record] Discarded episode ({len(self.timestamps)} steps)\033[0m")
            self.last_episode_dir = None
            self.tmp_episode_dir.cleanup()
            self._reset_episode_buffers()
            return

        # Final on-disk dir + env-derived metadata captured NOW (control thread); files land async.
        episode_dir = self.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        self.last_episode_dir = episode_dir
        pending = _PendingEpisode(
            tmp_dir_obj=self.tmp_episode_dir,
            tmp_dir=self.tmp_episode_dir.name,
            timestamps=self.timestamps,
            observations=self.observations,
            actions=self.actions,
            action_sources=self.action_sources,
            component_timestamps=self.component_timestamps,
            rewards=self.rewards,
            dones=self.dones,
            metadata=self._get_metadata(),
            control_period=self.env.unwrapped.control_period,
            episode_dir=episode_dir,
        )
        self._reset_episode_buffers()
        self._save_queue.put(pending)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # This is the ONLY recording cost the control thread now pays per episode end. If overruns
        # persist while this stays small, the culprit is elsewhere (policy server / cuRobo hover).
        print(
            f"\033[34m[Record][timing] save-handoff {dt_ms:.1f}ms on control thread (saver depth={self._save_queue.qsize()})\033[0m"
        )

    def _saver_loop(self) -> None:
        """Background thread: write queued episode snapshots to disk in finish order."""
        while True:
            pending = self._save_queue.get()
            try:
                if pending is None:
                    return
                self._write_pending(pending)
            except Exception as exc:  # never let one bad save kill the saver
                print(f"\033[91m[Record] background episode save failed: {exc}\033[0m")
            finally:
                self._save_queue.task_done()

    def wait_for_pending_saves(self) -> None:
        """Block until every queued episode has been fully written to disk. The saver is
        asynchronous (so env.reset never blocks on disk I/O); callers that must read the
        on-disk episode immediately — tests, end-of-run flushes — wait here first."""
        if self._save_thread is not None:
            self._save_queue.join()

    def _write_pending(self, p: "_PendingEpisode") -> None:
        t0 = time.perf_counter()
        # Finalize THIS episode's mp4 writers (per-dir, so it never touches the next episode's
        # still-open writers) — now off the control thread.
        self._flush_video_writers(p.tmp_dir)
        t_flush = time.perf_counter()
        self._save_timestamps(p)
        self._save_component_timestamps(p)
        self._save_observations(p)
        self._save_reward(p)
        self._save_actions(p)
        self._save_metadata(p)
        shutil.move(p.tmp_dir, p.episode_dir)
        # Queue the H.264 convert BEFORE the tmp cleanup so a cleanup hiccup can't skip it.
        self._video_queue.put(("convert", str(p.episode_dir)))
        flush_ms = (t_flush - t0) * 1000.0
        write_ms = (time.perf_counter() - t_flush) * 1000.0
        num_episodes = len([x for x in self.output_dir.iterdir() if x.is_dir()])
        print(
            f"\033[34m[Record] Saved episode to {p.episode_dir} ({len(p.timestamps)} steps, "
            f"{num_episodes} total) [background: flush={flush_ms:.0f}ms write={write_ms:.0f}ms]\033[0m"
        )
        try:
            p.tmp_dir_obj.cleanup()
        except OSError:
            pass  # dir already moved away; nothing to clean

    def close(self):
        if self.is_recording:
            self._enqueue_save(discard_episode=True)
            self.is_recording = False

        # Drain the background saver first (its converts get queued onto the video process),
        # then shut the video process so it finalizes every pending H.264 encode.
        if self._save_thread is not None:
            self._save_queue.put(None)
            self._save_thread.join()
        if self._video_process is not None:
            self._video_queue.put(None)
            self._video_process.join()
        super().close()

    def _save_timestamps(self, p: "_PendingEpisode"):
        np.save(Path(p.tmp_dir) / "timestamp.npy", p.timestamps)

        # Show warning for unusually long episode
        duration_mins = len(p.timestamps) * p.control_period / 60.0
        if duration_mins > 10:  # 10 mins
            print(f"Warning: Unusually long episode ({duration_mins:.1f} mins)")

    def _save_component_timestamps(self, p: "_PendingEpisode"):
        if not p.component_timestamps:
            return
        path = Path(p.tmp_dir) / "component_timestamps.json"
        with open(path, "w") as f:
            json.dump(p.component_timestamps, f)

    def _save_observations(self, p: "_PendingEpisode"):
        observations = p.observations
        if "left_joint_pos" in observations.keys() and "left_ee_pos" not in observations.keys():
            # For joint position control
            for k, v in observations.items():
                k = k.replace("left_", "left-")
                k = k.replace("right_", "right-")
                try:
                    v = np.array(v, dtype=np.float64)
                    np.save(Path(p.tmp_dir) / f"{k}.npy", v)
                except ValueError:
                    warnings.warn(f"Could not record key {k}")
            return

        if "left_ee_pos" in observations.keys():
            # For cartesian, delta_ee_pose, delta_ee_pose_translation modes
            bimanual_ee_pose_proprioception = np.hstack(
                [
                    observations["left_ee_pos"],
                    observations["left_ee_rot6d"],
                    observations["left_gripper_pos"],
                    observations["right_ee_pos"],
                    observations["right_ee_rot6d"],
                    observations["right_gripper_pos"],
                ]
            )
            np.save(Path(p.tmp_dir) / "state-eef-rot6d.npy", bimanual_ee_pose_proprioception)

        for key in ("left_joint_pos", "right_joint_pos", "left_target_eef", "right_target_eef"):
            if key not in observations:
                continue
            try:
                value = np.array(observations[key], dtype=np.float32)
                out_key = key.replace("left_", "left-").replace("right_", "right-")
                np.save(Path(p.tmp_dir) / f"{out_key}.npy", value)
            except ValueError:
                warnings.warn(f"Could not record key {key}")

    def _save_actions(self, p: "_PendingEpisode"):
        actions = p.actions
        if "left_joint_pos" in actions.keys():
            left_action = np.hstack([actions["left_joint_pos"], actions["left_gripper_pos"]])
            right_action = np.hstack([actions["right_joint_pos"], actions["right_gripper_pos"]])
            np.save(Path(p.tmp_dir) / "action-left-pos.npy", left_action)
            np.save(Path(p.tmp_dir) / "action-right-pos.npy", right_action)
        elif "left_ee_pos" in actions.keys():
            # Delta ee_pose.
            # TODO: disambiguate eepose or delta eepose cmd
            bimanual_delta_ee_pose_action = np.hstack(
                [
                    actions["left_ee_pos"],
                    actions["left_ee_rot6d"],
                    actions["left_gripper_pos"],
                    actions["right_ee_pos"],
                    actions["right_ee_rot6d"],
                    actions["right_gripper_pos"],
                ]
            )
            np.save(Path(p.tmp_dir) / "action-delta-eef-rot6d.npy", bimanual_delta_ee_pose_action)
        else:
            raise NotImplementedError(f"Action format not supported: {actions.keys()}")

        # Per-step source labels (e.g. "human" / "policy")
        source_array = np.array(p.action_sources)
        np.save(Path(p.tmp_dir) / "action-source.npy", source_array)
        # Backward-compatible JSON (list of per-step labels)
        with open(Path(p.tmp_dir) / "action-source.json", "w") as f:
            json.dump(p.action_sources, f)

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _save_reward(self, p: "_PendingEpisode"):
        step_count = len(p.timestamps)
        rewards = np.asarray(p.rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(p.dones, dtype=np.bool_).reshape(-1)
        if rewards.shape[0] != step_count or dones.shape[0] != step_count:
            raise ValueError(
                "Reward/done labels must match recorded steps; got "
                f"steps={step_count}, rewards={rewards.shape}, dones={dones.shape}"
            )
        if step_count > 0:
            dones[-1] = True

        path = Path(p.tmp_dir)
        np.save(path / "reward.npy", rewards)
        np.save(path / "dones.npy", dones)
        np.savez(path / "reward.npz", rewards=rewards, dones=dones)

    def _save_metadata(self, p: "_PendingEpisode"):
        # metadata was captured on the control thread (it reads self.env); just serialize it.
        with open(Path(p.tmp_dir) / "metadata.json", "w") as f:
            json.dump(p.metadata, f, indent=2, default=self._json_default)

    def _get_metadata(self):
        # Reference episode: episode_OAV_oKrUJ1ZU4zqYX1bJ8kA-3L-EzJnmzbmhgBipcHE
        motion = self.task_name  # Placeholder
        motion_object = self.task_name  # Placeholder
        station_metadata = {
            "arm_type": "yam",
            "world_frame": "left_arm",
            "extrinsics": {
                "right_arm_extrinsic": {
                    "position": [0.0, -0.61, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                }
            },
        }
        attributes = {
            "manipulation_surface": "white",
            "object": "object",  # Placeholder
        }
        camera_info = self._get_camera_metadata()
        metadata = {
            "task_name": self.task_name,
            "motion": motion,
            "motion_object": motion_object,
            "env_loop_frequency": self.env.unwrapped.policy_control_freq,
            "duration": self.timestamps[-1] - self.timestamps[0],
            "station_metadata": station_metadata,
            "attributes": attributes,
            "camera_info": camera_info,
        }

        # Custom metadata
        metadata["operator"] = self.operator
        metadata["hostname"] = socket.gethostname()
        metadata["policy_config"] = self.policy_config
        metadata["terminal_event"] = self.terminal_event

        # # Save control mode for more convenient visualization and replay
        # metadata["control_mode"] = getattr(self, "control_mode", None)
        return metadata

    def _get_camera_metadata(self):
        camera_info = {}
        if not hasattr(self.env.unwrapped, "cameras"):
            return camera_info

        for camera_name, camera in self.env.unwrapped.cameras.items():
            name = f"{camera_name}_camera"
            cam_impl = getattr(camera, "camera", camera)
            resolution = tuple(getattr(cam_impl, "resolution", (None, None)))
            camera_info[name] = {
                "camera_type": getattr(cam_impl, "camera_type", cam_impl.__class__.__name__),
                "device_id": getattr(cam_impl, "device_id", None),
                "width": resolution[0] if len(resolution) >= 1 else None,
                "height": resolution[1] if len(resolution) >= 2 else None,
                "polling_fps": getattr(cam_impl, "fps", None),
                "name": name,
                "image_transfer_time_offset_ms": None,
                "exposure_value": None,
                "auto_exposure": getattr(cam_impl, "auto_exposure", None),
                "intrinsic_data": None,
                "extrinsics": None,
                "concat_image": False,
            }
        return camera_info


def main():
    from gymnasium.envs.registration import register

    from groot.control.envs.yam.yam_sim_env import SimDummyPolicy

    # Environment
    register(id="YamSim-v0", entry_point="groot.control.envs.yam.yam_sim_env:YamSimEnv")
    env = gym.make("YamSim-v0")

    env = RecordEpisodeWrapper(env)

    # Policy
    policy = SimDummyPolicy(env.action_space)

    # Do not record scene reset
    obs, _ = env.reset(options={"start_new_episode": False})
    for _ in range(30):
        action, _ = policy.get_action(obs)
        obs, _, _, _, _ = env.step(action)

    # Record actual task execution
    obs, _ = env.reset(options={"task_name": "task"})
    for i in range(1, 90 + 1):
        action = policy.step(obs)
        obs, _, _, _, _ = env.step(action)

        if i % 30 == 0:
            obs, _ = env.reset(options={"task_name": "task"})

    env.close()


if __name__ == "__main__":
    main()

