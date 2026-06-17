# TBD Data Studio

End-to-end system for collecting, recording, browsing, replaying, labeling, and visualizing robot demonstration episodes on the YAM bimanual platform.

**Related docs:**
- `docs/frequency_relationship.md` -- Control loop frequency behavior (30 Hz outer loop, 50 Hz motor sender)
- `docs/MULTI_CAMERA_CONFIG.md` -- Multi-source camera configuration (RealSense, ZED, etc.)
- `docs/VOICE_INPUT.md` -- Voice input and RealtimeSTT integration
- `docs/SERIAL_FOOTSWITCH.md` -- Fello footswitch hardware for data collection triggers
- `docs/debug_shit_data_collection_infra.md` -- Data collection debugging notes
- `docs/data_summary_new.md` -- Dataset summary and statistics
- `docs/RL_PIPELINE_DESIGN.md` -- RL training pipeline that consumes collected data
- `docs/SAFETY_ZONE_DESIGN.md` -- Task-aware EE safety zones for RL exploration

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                              │
│                                                                  │
│  run_data_collection.py  ←── tools/vision/run_data_collection.py │
│       │                                                          │
│       ├── TeleopPolicy (teleop_policy.py)                        │
│       │     └── LeaderRobotClient / FollowerRobotClient          │
│       │         (Portal RPC to arm servers)                      │
│       │                                                          │
│       ├── RecordEpisodeWrapper (record_episode_wrapper.py)       │
│       │     └── mp4v → H.264 conversion (ffmpeg)                │
│       │                                                          │
│       ├── TimingJsonlLogger (timing_jsonl.py)                    │
│       │                                                          │
│       └── VoiceAnnotation (tools/teleop_voice_annotate/)         │
│             └── start_voice_annotation_thread()                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               BROWSE / REPLAY / LABEL                            │
│                                                                  │
│  tbd data <subcommand>             Go CLI (Cobra)                │
│      │                                                           │
│      ├── inspect / label / replay   cli/cmd/data*.go, replay.go  │
│      │                                                           │
│      v                                                           │
│  tools/data_vis/launch_overlay_viz.py   Python launcher          │
│      │                                                           │
│      v                                                           │
│  third_party/overlay_viz/              FastAPI server + React UI  │
│      │  HTTP :8888                                               │
│      v                                                           │
│  Browser (operator)                    Episode browser, charts   │
│      │                                                           │
│      │  Portal IPC :8009                                         │
│      v                                                           │
│  experimental/yam_control_loop.py      Control loop (replay)     │
│      │                                                           │
│      v                                                           │
│  experimental/lerobot_replay_policy.py Step-by-step replay       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│             OFFLINE VISUALIZATION TOOLS                           │
│                                                                  │
│  tools/data_vis/cook_data_vis.py           3D trajectory viz     │
│  tools/data_vis/cook_data_timestamp_vis.py Timestamp charts      │
│  tools/data_vis/cook_action_diff.py        Action diff norms     │
│  tools/data_vis/launch_data_visualizer.py  Rerun-based viewer    │
│  tools/data_vis/play_video.py              Video-to-GIF tool     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Collection

### Entry Point (`run_data_collection.py`)

The primary data collection script. Also mirrored at `tools/vision/run_data_collection.py`.

**Launch:**
```bash
uv run launch.py --mode=data_collection --use-fello
uv run launch.py --mode=data_collection --use-fello --force-feedback
uv run launch.py --mode=data_collection --use-fello --use-voice
```

**Configuration** (`run_data_collection.py:54`):
```python
@dataclass
class DataCollectionConfig:
    operator: str | None = None            # Username of operator
    task_list_path: str | None = None      # Path to .txt task list (one task per line)
    station: int | None = None             # Station number (auto-detected from hostname)
    display_image: bool = False            # Display live camera feed in window
    visualize: bool = False                # Viser 3D visualization
    use_voice: bool = False                # Enable voice annotations (speech-to-text)
    voice_trigger_mode: bool = True        # Trigger word detection (start/begin/end/done)
    voice_silence_duration: float = 0.5    # Seconds of silence before finalizing speech
    voice_input_device_index: int | None   # Audio input device index
    save_annotation: bool = False          # Auto-render annotated video after saving
    use_fello: bool = False                # Use Fello arms (footswitch-based)
    force_feedback: bool = False           # Mirror follower torque back to Fello leader
    force_feedback_ratios: tuple = DEFAULT_FORCE_FEEDBACK_RATIOS  # Per-joint (7 values, each <= 1/3)
    policy_control_freq: float = 30.0      # Follower command frequency (Hz)
    timing_debug: bool = False             # Print slow-step diagnostics
    timing_warn_ms: float = 50.0           # Warn threshold per phase (ms)
    timing_summary_every: int = 300        # Aggregated summary interval (steps)
    timing_log_dir: str = "/tmp/yam_timing" # JSONL timing log directory
```

**Operator controls** (Fello right-arm footswitch buttons, `run_data_collection.py:382-423`):
- Button 2: Start recording
- Button 0: Save episode
- Button 1: Discard episode

**Data flow** (`run_data_collection.py:174`):
1. Prompts for operator and task name (or loads task list from file)
2. Creates output directory under `$YAM_RAW_PATH/{operator}_{task}_{timestamp}-YAM-{station}`
3. Registers `YamReal-v0` gym env at `policy_control_freq` Hz
4. Wraps env with `RecordEpisodeWrapper`
5. Creates `TeleopPolicy` with optional force feedback
6. Main loop: reads leader joints, sends to follower, records obs/actions
7. Optional voice annotation thread runs in parallel

### Teleop Policy (`teleop_policy.py`)

Mirrors leader arm positions to follower arms via Portal RPC.

**Key classes** (`teleop_policy.py:108-251`):

| Class | Role | File:Line |
|-------|------|-----------|
| `LeaderRobotClient` | Read leader arm joint positions + buttons | `teleop_policy.py:108` |
| `FollowerRobotClient` | Read follower observations (for force feedback) | `teleop_policy.py:221` |
| `TeleopPolicy` | Coordinate leaders/followers, detect button events | `teleop_policy.py:251` |

**Safety limits** (`teleop_policy.py:119-122`):
```python
MAX_KP = np.array([80, 80, 80, 40, 15, 15])   # Max proportional gains
MAX_KD = np.array([10, 10, 10, 5, 5, 5])       # Max derivative gains
MAX_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # Max velocity (rad/s)
```

**Force feedback** (`teleop_policy.py:443-504`):
- Reads `force_feedback_torque` from follower observations
- Multiplies by per-joint ratios (capped at 1/3) and sign configuration
- Gripper feedback has deadband filtering on position and velocity
- Configurable via `robot/fello/fello_config.py`

**Button events** (`teleop_policy.py:347-360`):
- Fello mode: right arm only -- `[2=start, 1=discard, 0=save]`
- YAM mode: left `[0=reset, 1=discard]`, right `[0=start, 1=save]`
- Edge-detection prevents repeated triggers from held buttons

### Timing Logger (`timing_jsonl.py`)

Persists per-step timing samples to JSONL files for latency debugging.

**Class** (`timing_jsonl.py:9`):
```python
class TimingJsonlLogger:
    def __init__(self, *, enabled, component, label, warn_ms, summary_every, log_dir, flush_every=25): ...
    def record(self, event: str, samples_ms: dict[str, float], *, extra: dict | None = None): ...
```

**JSONL record format** (`timing_jsonl.py:71-82`):
```json
{"event":"loop_step","count":42,"unix_time_ns":1234567890000,"monotonic_ns":9876543210,"samples_ms":{"policy_call_ms":2.1,"env_step_ms":15.3}}
```

**Output location:** `{timing_log_dir}/{component}_{label}_pid{PID}.jsonl`

Used by `run_data_collection.py`, `TeleopPolicy`, `LeaderRobotClient`, and `FollowerRobotClient` -- each creates its own logger instance to track different phases of the control loop.

### Recording Wrapper (`record_episode_wrapper.py`)

Gym wrapper that records episodes during data collection or evaluation.

**Class** (`record_episode_wrapper.py:35`):
```python
class RecordEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, output_dir: str, operator: str | None = None, policy_config: dict | None = None): ...
```

**Reset options** (`record_episode_wrapper.py:162`):
- `task_name` (required for new episode)
- `start_new_episode` -- start new
- `discard_episode` -- finalize and delete
- `force_reset` -- call inner `env.reset()` even if not first episode

**What gets recorded per step** (`record_episode_wrapper.py:110-146`):
- `timestamps`: wall-clock `time.time()` per step
- `observations`: non-image obs as `.npy`, images as `.mp4` video
- `actions`: joint/gripper positions per arm
- `action_sources`: per-step `"human"` / `"policy"` / `"unknown"` labels
- `component_timestamps`: per-step per-component creation times (state reads, camera captures, action computation)

**Voice annotation support** (`record_episode_wrapper.py:90-108`):
- `add_voice_annotation(frame_idx, text)` records segment-based annotations
- Previous segment's `to_frame` is auto-closed when a new one starts
- Saved as `top_camera-images-rgb_annotation.json`

**Video pipeline** (`record_episode_wrapper.py:319-364`):
1. Records as mp4v via OpenCV VideoWriter
2. Converts to H.264 (libx264, preset=fast, crf=23, yuv420p) via ffmpeg subprocess
3. Even-dimension scaling filter: `scale=trunc(iw/2)*2:trunc(ih/2)*2`
4. Falls back to original format if ffmpeg is missing or conversion fails

### Voice Annotation (`tools/teleop_voice_annotate/`)

| File | Role |
|------|------|
| `tools/teleop_voice_annotate/voice_annotation.py` | `start_voice_annotation_thread()` -- background speech-to-text |
| `tools/teleop_voice_annotate/visualize_annotations.py` | Render annotated overlay video post-hoc |

---

## 2. CLI Commands

All commands are subcommands of `tbd data`, implemented in Go with Cobra.

### `tbd data inspect`

Browse and analyze recorded episodes without a robot.

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8888 | Web UI port |
| `--host` | 0.0.0.0 | Bind host |
| `--task` | -- | Auto-scan a predefined task on startup |
| `--policy-port` | 8009 | Portal IPC port (for optional policy connection) |

**File:** `cli/cmd/data_inspect.go`

**Execution flow** (`data_inspect.go:45-96`):
1. Finds project root by walking up from cwd looking for `tools/data_vis/launch_overlay_viz.py`
2. Runs `uv run tools/data_vis/launch_overlay_viz.py` with forwarded flags
3. Forwards SIGINT/SIGTERM to child process

### `tbd data label`

Keyframe-based progress labeling UI. Saves `progress_labels.json` per episode.

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | (required) | Dataset directory |
| `--port` | 8888 | Web UI port |
| `--host` | 0.0.0.0 | Bind host |

**File:** `cli/cmd/data_label.go`

**Known inconsistency:** `data_label.go:49` (and `replay.go` RTG-only mode) reference `root/launch_overlay_viz.py`, but **no such file exists at repo root** — only `tools/data_vis/launch_overlay_viz.py` is present. Both `tbd data label` and `tbd data replay --rtg` (without `--real`) will error with "launch script not found" until either the Go callers are updated to point at `tools/data_vis/launch_overlay_viz.py` or a root shim is added.

### `tbd data replay`

Interactive replay on a real or simulated robot with synchronized visualization.

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | -- | Dataset directory (opens folder picker if omitted) |
| `--port` | 8888 | Web UI port |
| `--host` | 0.0.0.0 | Bind host |
| `--policy-port` | 8009 | Portal IPC port for control loop |
| `--real` | false | Connect to real robot hardware |
| `--rtg` | false | RTG (reward-to-go) labeling mode |
| `--value-server` | -- | Optional value prediction server URL |

**File:** `cli/cmd/replay.go`

**Execution flow** (`replay.go:59-258`):
1. Opens native folder picker (zenity/kdialog/AppleScript) if `--path` not provided (`replay.go:282-338`)
2. **RTG-only mode** (`--rtg` without `--real`): launches only the overlay viz with `--mode rtg`, no control loop
3. **Full replay mode**: launches control loop: `experimental/yam_control_loop.py --use-replay-policy`
4. Waits up to 45s for control loop readiness (TCP port probe on `policy-port`, `replay.go:261-278`)
5. Launches overlay viz: `tools/data_vis/launch_overlay_viz.py` with `--path` and `--policy-port`
6. Waits up to 30s for overlay UI readiness
7. Manages both processes, forwarding SIGINT/SIGTERM
8. If either process exits, kills the other

**Dataset path detection** (`replay.go:355-362`): checks for `left-joint_pos.npy`, `action-left-pos.npy`, or `timestamp.npy` to determine if the path is an episode directory (vs. a session directory with episode subdirs).

### Supporting Go files

| File | Role |
|------|------|
| `cli/cmd/root.go` | `tbd` root command |
| `cli/cmd/data.go` | `tbd data` parent command (`data.go:7-15`) |
| `cli/cmd/data_inspect.go` | `tbd data inspect` |
| `cli/cmd/data_label.go` | `tbd data label` |
| `cli/cmd/replay.go` | `tbd data replay` |

---

## 3. Episode Data Format

Episodes are recorded by `RecordEpisodeWrapper` and stored as timestamped directories under a task root.

### Directory Structure

```
$YAM_RAW_PATH/{operator}_{task}_{timestamp}-YAM-{station}/
  └── {episode_timestamp}/
        ├── top_camera-images-rgb.mp4       # H.264 video (libx264, crf=23, yuv420p)
        ├── left_camera-images-rgb.mp4      # (optional)
        ├── right_camera-images-rgb.mp4     # (optional)
        ├── action-left-pos.npy             # (T, 7) float32 -- 6 joints + 1 gripper
        ├── action-right-pos.npy            # (T, 7) float32
        ├── action-source.npy               # (T,) str -- "human" | "policy" | "unknown"
        ├── action-source.json              # same data, JSON list (backward compat)
        ├── left-joint_pos.npy              # (T, 6) float64 -- observation
        ├── left-gripper_pos.npy            # (T, 1) float64
        ├── right-joint_pos.npy             # (T, 6) float64
        ├── right-gripper_pos.npy           # (T, 1) float64
        ├── timestamp.npy                   # (T,) float64 -- UNIX wall-clock seconds
        ├── component_timestamps.json       # per-step per-component creation times
        ├── metadata.json                   # task, operator, camera info, policy config
        ├── progress_labels.json            # keyframe-based progress labels (from labeling)
        ├── value_predictions.npz           # cached value predictions (RTG mode)
        ├── value_predictions.json          # legacy value prediction cache format
        └── *_annotation.json               # voice annotations (optional)
```

### Action Formats

| Control Mode | Dims | Layout |
|-------------|------|--------|
| `joint_position` | 14 | `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]` |
| `delta_joint_position` | 14 | deltas from current state, same layout |
| `cartesian_position` | 16 | `[left_pos(3), left_quat_xyzw(4), left_grip(1), right_pos(3), right_quat_xyzw(4), right_grip(1)]` |
| `delta_ee_pose` | 16 | delta EE pose, same layout |
| `umi_ee_pose` | 16/20 | UMI-specific encoding (20D parquet with observation.state, converted to 16D) |

### Supported Input Formats (Replay)

The replay policy auto-detects format (`experimental/lerobot_replay_policy.py:270-388`):

1. **Raw folder** (priority 1): separate `action-left-pos.npy` + `action-right-pos.npy`
2. **Parquet** (priority 2): `df["action"]` column with numpy arrays
3. **Single NPY** (priority 3): `actions.npy` with shape `(T, D)`
4. **NPZ archive** (priority 4): `archive["action"]`

### Metadata (`metadata.json`)

Written by `RecordEpisodeWrapper._get_metadata()` (`record_episode_wrapper.py:366-401`):

```json
{
  "task_name": "cutter_buss_01",
  "motion": "cutter_buss_01",
  "motion_object": "cutter_buss_01",
  "env_loop_frequency": 30,
  "duration": 10.5,
  "operator": "username",
  "hostname": "machine-name",
  "station_metadata": {
    "arm_type": "yam",
    "world_frame": "left_arm",
    "extrinsics": { "right_arm_extrinsic": { "position": [0.0, -0.61, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0] } }
  },
  "attributes": { "manipulation_surface": "white", "object": "object" },
  "camera_info": {
    "top_camera": { "camera_type": "...", "width": 1280, "height": 720, "polling_fps": 30, ... }
  },
  "policy_config": { ... }
}
```

**Additional metadata fields** (written by overlay viz labeling endpoints):
- `trim_start_frame`, `trim_end_frame`, `trim_segments` -- episode trim ranges
- `value_trim_start`, `value_trim_end` -- value-learning trim bounds
- `rtg_start`, `rtg_end`, `rtg_status`, `rtg_marker` -- reward-to-go labels
- `discarded` -- boolean, marks episode for exclusion

### Component Timestamps (`component_timestamps.json`)

Per-step dict recording when each component produced its data (`record_episode_wrapper.py:134-141`):

```json
[
  {"left_state": 1712345678.123, "right_state": 1712345678.125, "top_camera": 1712345678.130, "action": 1712345678.135, "action_source": 1712345678.135},
  ...
]
```

Used by the TimestampChart in the overlay viz UI and by `tools/data_vis/cook_data_timestamp_vis.py`.

---

## 4. Backend Components

### Overlay Viz Launcher (`tools/data_vis/launch_overlay_viz.py`)

Python launcher that creates and runs the FastAPI app via uvicorn.

**Arguments** (`launch_overlay_viz.py:31-63`):

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | 0.0.0.0 | Bind host |
| `--port` | 8888 | Bind port |
| `--task` | -- | Auto-scan a predefined task |
| `--path` | -- | Auto-scan an arbitrary dataset directory |
| `--policy-port` | 8009 | Portal IPC port |
| `--mode` | inspect | UI mode: `inspect`, `label`, or `rtg` |
| `--value-server` | -- | Value prediction server URL |

**Startup flow** (`launch_overlay_viz.py:76-109`):
1. If `--path`: creates `EpisodeScanner` for the given directory, uses dir name as `task_id`
2. If `--task`: looks up path from `TASK_DATA_PATHS` in `experimental/task_options_config.py`
3. Passes initial scanners and config to `create_app()`
4. Runs uvicorn

### Overlay Viz Server (`third_party/overlay_viz/`)

FastAPI application serving the web UI and bridging operator controls to the replay policy.

| File | Role | Key Lines |
|------|------|-----------|
| `app.py` | FastAPI app factory, 50+ REST endpoints | `app.py:635` (`create_app()`) |
| `scanner.py` | Episode discovery, video frame extraction with multi-level caching | `scanner.py:34` (`EpisodeScanner`) |
| `pusher.py` | Non-blocking live camera frame pusher (single-slot queue, drop-on-backpressure) | `pusher.py:26` (`FramePusher`) |
| `__main__.py` | Direct entry point (`python -m third_party.overlay_viz`) | `__main__.py:1` |
| `ui/` | React + TypeScript frontend (Vite + Tailwind + Shadcn) | -- |
| `tests/` | Component timestamp endpoint tests | -- |

**App factory signature** (`app.py:635`):
```python
def create_app(
    *,
    initial_scanners: dict[str, EpisodeScanner] | None = None,
    policy_port: int = 8009,
    default_task: str | None = None,
    mode: str = "inspect",       # "inspect" | "label" | "rtg"
    value_server_url: str | None = None,
) -> FastAPI: ...
```

**Internal classes** (`app.py`):

| Class | Role | Line |
|-------|------|------|
| `CameraStream` | Live camera frame buffer (push mode + direct RealSense capture) | `app.py:378` |
| `PolicyConnection` | Thin wrapper around replay-control Portal RPCs (fresh client per call) | `app.py:498` |
| `ValueServerError` | Custom exception for value server communication | `app.py:40` |

**Scanner caching strategy** (`scanner.py`):
- First-frame BGR cache: LRU, 64 entries (`scanner.py:53`)
- JPEG cache: LRU, 200 entries, keyed by `(idx, camera, frame_num, quality)` (`scanner.py:66-68`)
- VideoCapture handle pool: LRU, 12 open handles (`scanner.py:57-59`)
- Episode pre-decode cache: 1 episode at a time, low-res JPEGs (`scanner.py:74-78`)
- Thread pool: 3 workers for parallel camera reads (`scanner.py:71`)

**Frame read optimization** (`scanner.py:601-660`):
- Sequential reads (frame N+1 after N): skip seeking, just `cap.read()`
- Small forward jumps (<=5 frames): read-and-discard instead of seek
- Large jumps or backward: `cap.set(CAP_PROP_POS_FRAMES, frame_num)`
- Per-camera locks allow left/top/right to seek in parallel

### Key API Endpoints

**App configuration:**
- `GET /api/default_task` -- task auto-loaded at startup (`app.py:855`)
- `GET /api/mode` -- UI mode, value server status (`app.py:860`)

**Task and episode browsing:**
- `GET /api/tasks` -- list all configured tasks (`app.py:869`)
- `POST /api/tasks/{id}/scan` -- start background scan (`app.py:874`)
- `GET /api/tasks/{id}/status` -- scan progress (`app.py:901`)
- `GET /api/tasks/{id}/episodes` -- list episodes with discarded status (`app.py:916`)
- `GET /api/tasks/{id}/episodes/{idx}/frame` -- first frame JPEG (`app.py:950`)
- `GET /api/tasks/{id}/episodes/{idx}/info` -- video metadata, trim/RTG state, value cache status (`app.py:970`)

**Camera frames (scrubbing):**
- `GET /api/tasks/{id}/episodes/{idx}/camera_video` -- stream source MP4 for native browser playback (`app.py:1523`)
- `GET /api/tasks/{id}/episodes/{idx}/video_frame` -- single frame JPEG by frame number (`app.py:1542`)
- `GET /api/tasks/{id}/episodes/{idx}/camera_frame` -- single camera frame JPEG (top/left/right) (`app.py:1568`)
- `GET /api/tasks/{id}/episodes/{idx}/camera_frames` -- all 3 cameras as base64 JSON in one request (`app.py:1602`)
- `POST /api/tasks/{id}/episodes/{idx}/preload` -- trigger background pre-decode of all frames (`app.py:1645`)
- `GET /api/tasks/{id}/episodes/{idx}/preload_status` -- preload progress (`app.py:1659`)

**Data for charts:**
- `GET /api/tasks/{id}/episodes/{idx}/actions` -- windowed action trajectory `[t-h, t+h]` (`app.py:1675`)
- `GET /api/tasks/{id}/episodes/{idx}/states` -- full state trajectory `(T, 14)` (`app.py:1755`)
- `GET /api/tasks/{id}/episodes/{idx}/action_source` -- per-step source labels + contiguous segments (`app.py:1815`)
- `GET /api/tasks/{id}/episodes/{idx}/frequency` -- FFT frequency analysis per action dimension (`app.py:1878`)
- `GET /api/tasks/{id}/episodes/{idx}/component_timestamps` -- per-component creation times (`app.py:1992`)

**Replay control (Portal IPC to control loop):**
- `POST /api/replay/connect` / `disconnect` / `status` (`app.py:2227-2243`)
- `POST /api/replay/play` / `pause` / `step` / `home` (`app.py:2245-2283`)
- `POST /api/replay/load` -- load episode and set control mode (`app.py:2285`)
- `POST /api/replay/sync_to_init` -- move robot to episode start pose (`app.py:2336`)
- `POST /api/replay/task_command` -- set task command string (`app.py:2318`)

**Labeling:**
- `GET/POST /api/tasks/{id}/episodes/{idx}/labels` -- progress labels (`app.py:1416-1472`)
- `POST /api/tasks/{id}/episodes/{idx}/trim` -- trim episode to frame range(s), supports multi-segment (`app.py:1225`)
- `POST /api/tasks/{id}/episodes/{idx}/auto_clip` -- suggest segments by dropping idle runs; body `{threshold, min_idle_steps, start_threshold, end_threshold, min_segment_steps}` (`start_threshold` / `end_threshold` independently control leading- and trailing-edge skips; segments shorter than `min_segment_steps` are filtered). Arm vs gripper channels are decoupled (dim==14 → ch 6 / ch 13 are grippers, matching `findFirstMovementStep` in `chart-utils.ts`); the gripper threshold is `0.2x` the supplied arm threshold. Leading/trailing trim OR-combines state and action (falls back to actions when states are absent): a frame is considered motion when either state or action moves — consistent with the mid-episode idle definition (idle = both quiet ⇔ motion = either loud). Mid-episode idle detection AND-combines state and action: a step is idle only when BOTH channels are quiet in BOTH arm and grip groups, avoiding splits at contact pauses where the state is momentarily still but the operator is pushing against resistance. Returns `{segments, idle_runs, leading_skip, trailing_skip, dropped_short_segments, used_states}` without persisting.
- `POST /api/tasks/{id}/auto_clip_all` -- run auto-clip on every episode in the task and persist the result; body adds `force: bool` (default false). When `force=false`, episodes whose `metadata.json` already contains a `trim_segments` list are skipped. Returns `{total, processed, skipped, errors, processed_episodes, skipped_episodes, error_details, params, force}`.
- `POST /api/tasks/{id}/auto_screen_all` -- scan every episode in parallel (`ThreadPoolExecutor(max_workers=4)`) and auto-discard those that are too short, contain pure-color / featureless frames, or have sync gaps above threshold; body `{min_frames=64, latency_threshold_s=0.05, pure_color_std_max=5.0, pure_color_subsample_stride=4, pure_color_max_offenders=20, dry_run=false}` — per-frame streaming decode with spatial subsample (stride 4) for throughput, early-exits a given episode once `max_offenders` pure-color frames are found since one is already enough to flag it; when `dry_run=false`, merges `{"discarded": true, "auto_discard_flags": [...], "auto_discard_details": {...}}` into each flagged episode's `metadata.json` without touching existing keys like `trim_segments`. Returns `{total, flagged, by_reason: {too_short, latency, pure_color}, results: [{idx, folder, flags, details}], params, dry_run}` — only flagged episodes appear in `results`, unflagged ones are omitted to keep the response small.
- `POST /api/tasks/{id}/episodes/{idx}/value_trim` -- value-learning trim bounds (`app.py:1321`)
- `POST /api/tasks/{id}/episodes/{idx}/discard` -- toggle discarded flag (`app.py:1377`)
- `POST /api/tasks/{id}/episodes/{idx}/rtg_marker` -- save RTG range and success/failure status (`app.py:1478`)
- `POST /api/tasks/{id}/episodes/{idx}/clear_annotations` -- strip all annotation keys (`trim_*`, `value_trim_*`, `rtg_*`, `discarded`, `auto_discard_*`) from a single episode's `metadata.json` and delete `progress_labels.json`. Returns `{episode_dir, removed_keys, removed_progress_labels}`.
- `POST /api/tasks/{id}/clear_annotations_all` -- run the same clear across every episode in the task. Returns `{total, cleared, errors, cleared_episodes, error_details}`. Exposed in the UI as the "Clear" / "Clear All" buttons in `TaskSelector`, each fronted by a `window.confirm` since the operation is irreversible.

**Value predictions (requires `--value-server`):**
- `POST /api/tasks/{id}/episodes/{idx}/value_predictions` -- compute/cache with local+server validation (`app.py:1029`)
- `POST /api/tasks/{id}/value_predictions/precompute` -- batch all episodes (`app.py:1142`)
- `GET /api/tasks/{id}/value_predictions/precompute_status` -- batch progress (`app.py:1197`)

**Live camera:**
- `GET /api/cameras` -- camera status (RealSense device query) (`app.py:2076`)
- `POST /api/cameras/push` -- accept JPEG from external source (`app.py:2107`)
- `POST /api/cameras/start` / `stop` -- direct RealSense capture (`app.py:2127-2137`)
- `GET /api/cameras/frame` -- latest camera frame JPEG (`app.py:2139`)
- `GET /api/cameras/stream` -- MJPEG stream (~15 fps) (`app.py:2154`)

**Overlay blending:**
- `GET /api/tasks/{id}/episodes/{idx}/overlay` -- blended live+dataset frame with color modes (`app.py:2172`)

### Value Prediction Caching (`app.py:125-318`)

The value prediction system uses a two-tier cache:

1. **Local episode cache**: `value_predictions.npz` (current) or `value_predictions.json` (legacy) stored per episode
2. **Task-level index**: `value_predictions.index.json` at the task root, tracking all episode caches

Cache validation checks (`app.py:281-317`):
- Episode signature: SHA-256 hash of video file sizes + mtimes
- Server match: mode, ckpt_dir, model_cam_names must match between cache and server health

### Replay Policy (`experimental/lerobot_replay_policy.py`)

Loads episode data and feeds actions step-by-step to the control loop.

**Class** (`lerobot_replay_policy.py:28`):
```python
class LerobotReplayPolicy(Policy):
    CONTROL_MODE_DIMS = {
        "joint_position": 14,
        "delta_joint_position": 14,
        "cartesian_position": 16,
        "delta_ee_pose": 16,
        "umi_ee_pose": 16,
    }

    def __init__(
        self,
        dataset_path: str | Path,
        replan_horizon: int = 1,
        control_mode: str = "joint_position",
        action_horizon: int = 50,
        norm_stats_path: str | Path = "",
    ): ...

    def get_action(self, obs) -> (action_dict, info_dict): ...
    def peek_action(self, obs) -> (action_dict, info_dict): ...  # no side effects
    def reset(self) -> dict: ...
    def update_replay_config(...) -> None: ...  # hot-reload dataset at runtime
```

**Return format:**
```python
action = {
    "left_joint_pos": np.ndarray(6,),
    "left_gripper_pos": np.ndarray(1,),
    "right_joint_pos": np.ndarray(6,),
    "right_gripper_pos": np.ndarray(1,),
}
info = {
    "action_chunk": { ... },   # next action_horizon steps for UI visualization
    "episode_done": bool,
    "current_step": int,
    "num_steps": int,
}
```

**Initial state extraction** (`lerobot_replay_policy.py:89-247`):
- From raw teleop folders: reads `left-joint_pos.npy`, `left-gripper_pos.npy`, etc.
- From parquet: extracts `observation.state` column first frame (must be 14D joint format)
- Validates joint values in `[-2pi, 2pi]` and gripper in `[-0.1, 1.1]`
- Critical for delta replay modes and `Sync to Init`

At episode end: returns zero action (no repeat of last action to prevent drift in delta modes), sets `episode_done: True`.

---

## 5. Frontend (`third_party/overlay_viz/ui/`)

React + TypeScript + Vite application. Built with Tailwind CSS and Shadcn UI components.

### Build

```bash
cd third_party/overlay_viz/ui
npm install
npx vite build    # production build -> ../static/
npx vite dev      # dev server with HMR
```

### App Structure (`ui/src/App.tsx`)

The root `App` component orchestrates all hooks and renders:
- `Navbar` (top bar with camera status)
- `EpisodeSidebar` (slide-out panel, toggled with Cmd+\)
- `TaskSelector` (dropdown + scan progress)
- `RobotControlPanel` (hidden in label mode)
- `EpisodeViewer` (main content area)
- `StatusBar` (bottom bar with connection/camera status)

### Components (`ui/src/components/`)

| Component | Purpose | File |
|-----------|---------|------|
| `Navbar` | Top navigation bar, camera status indicator | `Navbar.tsx` |
| `StatusBar` | Bottom status bar (task, episode count, replay status) | `StatusBar.tsx` |
| `CameraPanels` | Multi-camera display (left/top/right), frame scrubbing | `CameraPanels.tsx` |
| `EpisodeSidebar` | Scrollable episode list with keyboard navigation (Cmd+\) | `EpisodeSidebar.tsx` |
| `EpisodeViewer` | Main viewer container, coordinates all panels. The Next button is always clickable; pressing it past the last episode swaps the viewer for `FinishedAnnotation` (source: `App.tsx:navigate`) | `EpisodeViewer.tsx` |
| `FinishedAnnotation` | End-of-task summary panel shown after the operator walks off the last episode — hides all panels and shows progress counts, the dataset folder path, and an "Open folder" button (wired to `api.openPath(tasks.dataPath)`) | `FinishedAnnotation.tsx` |
| `ExportLerobotPanel` | Collapsible panel that shells `uv run --no-sync scripts/data/convert_gearraw_to_lerobot.py` inside a configurable `minimal_policy` dir. Args (`--input-root`, `--output-dir`, `--task-name`, `--min-segment-length`, `--annotate-only`) are editable, and the server-side subprocess stdout is polled via `/api/tasks/{id}/export_lerobot/status` and shown live in a log pane | `ExportLerobotPanel.tsx` |
| `RobotControlPanel` | Play / Pause / Step / Home / Sync buttons | `RobotControlPanel.tsx` |
| `TaskSelector` | Task dropdown with scan controls | `TaskSelector.tsx` |
| `TimelineControls` | Frame scrubber bar, dataset Play/Stop/Reset/End, playback-speed selector (0.25x–4x), Auto Clip (drops idle runs into trim segments), "on load" toggle that auto-runs Auto Clip whenever the operator opens an episode with no saved trim segments | `TimelineControls.tsx` |
| `TimelineChart` | Action / state (joint pos) trajectory chart — actions and states render as separate, independently draggable panels (`charts` and `states`) | `TimelineChart.tsx` |
| `FrequencyChart` | FFT frequency visualization per action dimension | `FrequencyChart.tsx` |
| `TimestampChart` | Per-component timestamp visualization | `TimestampChart.tsx` |
| `ValueChart` | Value prediction chart (RTG mode) | `ValueChart.tsx` |
| `ProgressLabelPanel` | Keyframe-based progress labeling | `ProgressLabelPanel.tsx` |
| `DraggablePanel` | Resizable/draggable panel container | `DraggablePanel.tsx` |

### Hooks (`ui/src/hooks/`)

| Hook | Purpose | File |
|------|---------|------|
| `useTasks` | Load task list, trigger scans, manage episode list | `useTasks.ts` |
| `useEpisode` | Load and cache episode data (actions, states, predictions) | `useEpisode.ts` |
| `useCameras` | Live camera feed state (online, count, streaming) | `useCameras.ts` |
| `useReplay` | Connect/disconnect/control replay policy | `useReplay.ts` |
| `usePolling` | Periodic status polling (preload progress, connection) | `usePolling.ts` |
| `useProgressLabels` | Keyframe interpolation and label save/load | `useProgressLabels.ts` |
| `usePanelOrder` | Draggable panel layout persistence | `usePanelOrder.ts` |

### API Client (`ui/src/api/`)

| File | Role |
|------|------|
| `client.ts` | TypeScript API client wrapping all REST endpoints |
| `types.ts` | TypeScript type definitions for API responses |

### Chart Utilities (`ui/src/lib/`)

| File | Role |
|------|------|
| `chart-utils.ts` | Shared chart configuration and helpers |
| `utils.ts` | General utility functions |

---

## 6. Offline Visualization Tools (`tools/data_vis/`)

Standalone scripts for post-hoc dataset analysis. All use `tools/_bootstrap.py` for `uv` dependency management.

### 3D Trajectory Viewer (`cook_data_vis.py`)

Generates a self-contained HTML visualization from recorded episodes:
- Three.js 3D viewer with color-coded EE trajectories and URDF robot model
- 2D density heatmaps (XY / XZ / YZ planes) for Left and Right EE
- Toggleable 3D density planes and selection box
- SO(3) orientation density balls

```bash
uv run python tools/data_vis/cook_data_vis.py --data_path /path/to/dataset
uv run python tools/data_vis/cook_data_vis.py --data_path /path/to/dataset --max-episodes 50
```

**File:** `tools/data_vis/cook_data_vis.py`

### Timestamp Visualizer (`cook_data_timestamp_vis.py`)

Interactive HTML timeline chart showing per-component creation timestamps across episodes. Reads `component_timestamps.json` from each episode.

```bash
uv run python tools/data_vis/cook_data_timestamp_vis.py --data-path ep_dir1 ep_dir2 ...
```

**File:** `tools/data_vis/cook_data_timestamp_vis.py`

### Action Diff Plotter (`cook_action_diff.py`)

Plots `||action[t] - action[t-1]||` per timestep across all dimensions (6 joints + 1 gripper per arm) to identify idle segments.

```bash
uv run python tools/data_vis/cook_action_diff.py --data-path /path/to/dataset
uv run python tools/data_vis/cook_action_diff.py --data-path /path/to/dataset --arm left
```

**File:** `tools/data_vis/cook_action_diff.py`

### Rerun Data Visualizer (`launch_data_visualizer.py`)

YAM Robot Data Visualizer using Rerun SDK. Loads session directories and replays episodes with 3D visualization.

```bash
uv run tools/data_vis/launch_data_visualizer.py [<session_dir>] [--episode <idx>] [--all]
uv run tools/data_vis/launch_data_visualizer.py --browse --all
```

**File:** `tools/data_vis/launch_data_visualizer.py`

### Video-to-GIF Tool (`play_video.py`)

Converts video files to GIF using ffmpeg.

```bash
uv run python tools/data_vis/play_video.py --video /path/to/video.mp4 --fps 10 --width 640
```

**File:** `tools/data_vis/play_video.py`

---

## 7. Ports and IPC

| Component | Default Port | Protocol |
|-----------|-------------|----------|
| Overlay Viz Web UI | 8888 | HTTP |
| Replay Control Loop | 8009 | Portal IPC |
| Viser (debug viz, replay mode) | 8010 / 8889 | WebSocket |
| Value Server | (external) | HTTP |
| Camera Push Endpoint | 8888 | HTTP POST `/api/cameras/push` |

---

## 8. File Reference

### Data Collection

| Path | Description | Key Line |
|------|-------------|----------|
| `run_data_collection.py` | Main data collection script | `:54` (config), `:174` (main) |
| `tools/vision/run_data_collection.py` | Mirror of main data collection script | -- |
| `teleop_policy.py` | Teleop policy (leader/follower arms) | `:251` (TeleopPolicy class) |
| `timing_jsonl.py` | JSONL timing logger | `:9` (TimingJsonlLogger class) |
| `record_episode_wrapper.py` | Gym wrapper for episode recording | `:35` (RecordEpisodeWrapper class) |
| `viser_env_wrapper.py` | Viser 3D visualization wrapper | -- |
| `tools/teleop_voice_annotate/voice_annotation.py` | Voice annotation background thread | -- |
| `tools/teleop_voice_annotate/visualize_annotations.py` | Annotated video renderer | -- |

### CLI

| Path | Description | Key Line |
|------|-------------|----------|
| `cli/cmd/root.go` | `tbd` root command | -- |
| `cli/cmd/data.go` | `tbd data` parent command | `:7` |
| `cli/cmd/data_inspect.go` | `tbd data inspect` | `:21` |
| `cli/cmd/data_label.go` | `tbd data label` | `:20` |
| `cli/cmd/replay.go` | `tbd data replay` | `:28` |

### Backend

| Path | Description | Key Line |
|------|-------------|----------|
| `tools/data_vis/launch_overlay_viz.py` | Python launcher for overlay viz | `:31` |
| `third_party/overlay_viz/app.py` | FastAPI server (50+ endpoints) | `:635` (`create_app()`) |
| `third_party/overlay_viz/scanner.py` | Episode scanner and frame cache | `:34` (`EpisodeScanner`) |
| `third_party/overlay_viz/pusher.py` | Live frame pusher | `:26` (`FramePusher`) |
| `third_party/overlay_viz/__main__.py` | Direct entry point | -- |
| `third_party/overlay_viz/ui/` | React frontend | -- |
| `third_party/overlay_viz/tests/` | Backend tests | -- |
| `experimental/lerobot_replay_policy.py` | Step-by-step replay policy | `:28` |
| `experimental/dataset_replay_policy.py` | Simpler 46D replay (legacy) | -- |
| `experimental/yam_control_loop.py` | Control loop (replay + live) | -- |
| `experimental/task_options_config.py` | Task names and data path registry (`TASK_DATA_PATHS`) | -- |

### Offline Tools

| Path | Description |
|------|-------------|
| `tools/data_vis/cook_data_vis.py` | 3D trajectory + density visualization (HTML) |
| `tools/data_vis/cook_data_timestamp_vis.py` | Component timestamp timeline chart (HTML) |
| `tools/data_vis/cook_action_diff.py` | Action diff norm plots (PNG) |
| `tools/data_vis/launch_data_visualizer.py` | Rerun SDK episode viewer |
| `tools/data_vis/play_video.py` | Video-to-GIF converter (ffmpeg) |
