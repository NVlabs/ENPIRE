# Frequency & Timing Relationships

Comprehensive reference for all timing-critical loops, rates, and frequency relationships across the YAM bimanual robot platform. Covers both the **CAP server path** (production) and the **experimental control-loop path** (research/data-collection).

> **Last verified against codebase**: 2026-04-08

---

## Quick-Reference Table

| Component | Default Rate | Period | Source |
|---|---|---|---|
| CAP `CONTROL_FREQ_HZ` loop | **60 Hz** | 16.7 ms | `cap/config.py:107` |
| CAP `POLICY_FREQ_HZ` (learn_skill / Fello) | **30 Hz** | 33.3 ms | `cap/config.py:110` |
| HIL (human-in-the-loop) RL step rate | **10 Hz** | 100 ms | `cap/config.py:117` (`POLICY_FREQ_HZ / HIL_POLICY_SLOWDOWN`) |
| YAM motor sender (`send_rate_hz`) | **50 Hz** | 20 ms | `robot/yam/yam_controller.py:37` |
| Fello leader arm gravity-comp loop | **100 Hz** | 10 ms | `robot/models/fello/fello_config.yaml:70` |
| Serial footswitch USB output | **50 Hz** | 20 ms | `hardware/serial_footswitch_firmware/README.md:9` |
| Camera capture (RealSense / ZED default) | **60 fps** | 16.7 ms | `robot/camera_factory.py:169` |
| MuJoCo physics timestep (default) | **500 Hz** | 2.0 ms | MuJoCo default `opt.timestep = 0.002` |
| MuJoCo substeps per control tick (sim) | ~8 | N/A | `CONTROL_PERIOD_S / opt.timestep` — `cap/server/sim_backend.py:57` |
| Experimental outer loop (`policy_control_freq`) | **30 Hz** | 33.3 ms | `experimental/yam_control_loop.py:166` |
| Video recording FPS (learn_skill) | **30 fps** | 33.3 ms | `cap/server/cap_server.py:434` (`VIDEO_FPS = POLICY_FREQ_HZ`) |
| Gemini reward VLM query interval | **0.2 Hz** | 5.0 s | `cap/reward/gemini_reward.py:40` |
| Diagnostics UDP emit (every 20th tick) | **3 Hz** | ~333 ms | `cap/server/cap_server.py:1284` |

---

## 1. CAP Server Path (Production)

This is the primary production path. All hardware communication flows through `cap_server.py`.

### 1.1 CONTROL_FREQ_HZ — Hardware Control Loop

**Rate**: 60 Hz (16.7 ms period)

Defined in `cap/config.py:107-108`:

```python
CONTROL_FREQ_HZ = 60.0
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ
```

The `CapServer._control_loop()` (`cap/server/cap_server.py:664`) is the single owner of all hardware communication. Every tick it:

1. Reads joint positions from both follower arms (via Portal RPC to `arm_server`)
2. Computes FK for cached EE poses
3. Reads commanded targets under lock (`_state_lock`)
4. Applies joint limit clamping (`cap/server/cap_server.py:1240-1245`)
5. Sends commands to both arms via `command_joint_state()`
6. Steps the sim backend if in sim mode (`cap/server/cap_server.py:1263-1264`)
7. Sleeps until next tick with busy-wait for precision (`cap/server/cap_server.py:1266-1277`)

The busy-wait strategy sleeps coarsely until 2 ms before deadline, then spin-waits for precise timing:

```python
# cap/server/cap_server.py:1272-1277
_BUSYWAIT_S = 0.002  # busy-wait the last 2 ms
coarse = remaining - _BUSYWAIT_S
if coarse > 0:
    time.sleep(coarse)
while time.time() < deadline:
    pass  # spin
```

### 1.2 POLICY_FREQ_HZ — Skill Execution / RL Loop

**Rate**: 30 Hz (33.3 ms period)

Defined in `cap/config.py:110-111`:

```python
POLICY_FREQ_HZ = 30.0
POLICY_PERIOD_S = 1.0 / POLICY_FREQ_HZ
```

Used by:
- **`learn_skill` / `learn_skill_rl`** — the RL training loop inside cap_server (`cap/server/cap_server.py:3207-3211`). Each step builds observations, queries the policy server, evaluates reward, and writes targets under lock. The targets are then sent to hardware by the `CONTROL_FREQ_HZ` loop.
- **Fello loop** — `_fello_loop()` (`cap/server/cap_server.py:1287-1288`) reads Fello leader state and manages takeover at `POLICY_FREQ_HZ`.
- **Video recording** — `_SkillDataRecorder.VIDEO_FPS = POLICY_FREQ_HZ` (`cap/server/cap_server.py:434`).

### 1.3 HIL_POLICY_SLOWDOWN — Human-in-the-Loop Throttling

**Effective RL rate**: ~10 Hz (100 ms period)

Defined in `cap/config.py:117`:

```python
HIL_POLICY_SLOWDOWN: int = 3  # RL step every 3rd control step → ~10 Hz
```

During human takeover via Fello footswitch, the control loop still writes joint commands at `POLICY_FREQ_HZ` for smooth motion, but expensive operations (obs building, reward evaluation, RL server RPC) are skipped for `HIL_POLICY_SLOWDOWN - 1` out of every `HIL_POLICY_SLOWDOWN` steps (`cap/server/cap_server.py:3214-3223`).

This means:
- Joint writes to hardware: 30 Hz (no jitter)
- RL server RPC: 30 / 3 = 10 Hz

### 1.4 Two-Tier Layering: CONTROL_FREQ_HZ vs POLICY_FREQ_HZ

The CAP server has two concurrent timing domains:

```
Policy / RL loop (POLICY_FREQ_HZ = 30 Hz)
    │
    │  writes targets under _state_lock
    ▼
Control loop (CONTROL_FREQ_HZ = 60 Hz)
    │
    │  reads targets under _state_lock, sends to arm_server
    ▼
YAM motor sender (send_rate_hz = 50 Hz)
    │
    │  re-sends cached CAN command
    ▼
DaMiao motor CAN bus
```

The control loop runs 2x faster than the policy loop. Between policy updates, the control loop re-sends the same commanded joint positions. The arm_server motor sender independently re-sends at 50 Hz.

---

## 2. Experimental Control-Loop Path (Research / Data Collection)

This is the standalone path used by `experimental/yam_control_loop.py` and `run_data_collection.py`. It does **not** use `cap_server.py` — instead it directly constructs `YamRealEnv` or `YamSimEnv`.

### 2.1 Outer Loop — `policy_control_freq`

**Default rate**: 30 Hz (33.3 ms period)

Defined in `experimental/yam_control_loop.py:166`:

```python
policy_control_freq: int = 30
```

Passed into `YamRealEnv` (`robot/yam/yam_real_env.py:110-119`):

```python
self.policy_control_freq = policy_control_freq
self.control_period = 1.0 / policy_control_freq
```

`step()` sleeps until the next `control_period` boundary (`robot/yam/yam_real_env.py:224`).

The same default (30 Hz) is used by:
- `run_data_collection.py:96` — `policy_control_freq: float = 30.0`
- `scripts/eval_yam_sim.py:404` — `policy_control_freq: float = 30.0`
- `robot/yam/yam_sim_env.py:27` — `policy_control_freq: float = 30.0`

The viser-based variant uses a lower default:
- `experimental/yam_control_loop_viser_policy.py:91` — `policy_control_freq: int = 10`

### 2.2 Chunking Policies

Three chunking wrappers sit between the policy model and the outer loop:

#### SyncChunkingPolicy (default)

When `use_async: bool = False` (the default), the common path is `SyncChunkingPolicy`:

- If the action queue is empty, call the inner policy once to get an `action_chunk`
- Push the chunk into a local queue
- Pop exactly one action per outer-loop tick

With the default `action_horizon: int = 50` (`experimental/yam_control_loop.py:182`), one policy call provides up to 50 actions = `50 / 30 = 1.67 s` of execution before replan.

#### AsyncChunkingPolicy

Defined in `experimental/async_chunking_policy.py:21`. Runs the policy in a background thread. Supports:
- `replan_horizon` — triggers replan before chunk exhaustion
- `use_chunk_smoothing` — blends overlapping actions across chunk boundaries (`experimental/async_chunking_policy.py:56,182,278-290`)
- `min_smooth_steps: int = 8` — minimum blending window

#### RealtimeRTCChunkingPolicy

Defined in `experimental/realtime_rtc_chunking_policy.py:131`. Real-time receding-horizon controller that:
- Requires `replan_horizon < action_horizon`
- Tracks real inference latency via `control_hz` parameter
- Supports `use_chunk_smoothing` and `max_delay_steps`

### 2.3 Motor Sender — `send_rate_hz`

**Default rate**: 50 Hz (20 ms period)

Defined in `robot/yam/yam_controller.py:37`:

```python
send_rate_hz: float = 50.0
```

The background sender loop (`robot/yam/yam_controller.py:241-259`):

```python
def _send_loop(self) -> None:
    dt = 1.0 / self.send_rate_hz
    while not self._stop_event.is_set():
        # ... read cached target under lock ...
        self._do_send(pos, kp, kd, ...)
        time.sleep(dt)
```

`command_joint_pos()` (`robot/yam/yam_controller.py:395`) only updates the cached target — it does not send immediately. The background thread continuously streams the latest target to the motors.

---

## 3. Peripheral Rates

### 3.1 Camera Capture

**Default**: 60 fps

Set in `robot/camera_factory.py:169`:

```python
def create_camera(camera_name, *, resolution=(640, 480), fps=60, enable_depth=True):
```

Overridable per-camera via env vars `CAP_<NAME>_CAMERA_FPS` or `CAP_CAMERA_FPS` (`robot/camera_factory.py:91-96`).

### 3.2 Fello Leader Arm Gravity-Comp Loop

**Default**: 100 Hz (10 ms period)

Configured in `robot/models/fello/fello_config.yaml:70`:

```yaml
server:
  control_loop_hz: 100
```

Consumed by `robot/fello/fello_server.py:476`:

```python
control_thread = start_control_loop(robot, ..., freq_hz=args.control_loop_hz)
```

The loop (`robot/fello/fello_server.py:187-229`) sends gravity compensation or position commands every `1.0 / freq_hz` seconds.

### 3.3 Serial Footswitch

**Output rate**: 50 Hz over USB CDC serial (115200 baud)

Documented in `hardware/serial_footswitch_firmware/README.md:9`. Can be verified with `scripts/setup/measure_serial_rate.py`.

### 3.4 MuJoCo Simulation Timestep

**Default**: 0.002 s (500 Hz)

MuJoCo's default `opt.timestep` is 0.002 s (not overridden in the station XML). The number of physics substeps per control tick is computed dynamically:

- CAP sim backend: `round(CONTROL_PERIOD_S / model.opt.timestep)` = `round(0.01667 / 0.002)` = **8 substeps** (`cap/server/sim_backend.py:57`)
- Warp sim backend: same formula (`cap/server/warp_sim_backend.py:74`)
- Experimental `YamSimEnv`: `int(self.control_period // model.opt.timestep)` (`robot/yam/yam_sim_env.py:38`)

### 3.5 Reward Evaluation

- **Gemini VLM reward**: queries at most every **5.0 s** (default `GEMINI_REWARD_INTERVAL_S`), returning cached value between queries (`cap/reward/gemini_reward.py:40,99`).
- **Other reward functions**: called every RL step at `POLICY_FREQ_HZ` (or `POLICY_FREQ_HZ / HIL_POLICY_SLOWDOWN` during takeover).

### 3.6 Diagnostics UDP Emitter

`emit()` is called from the control loop every 20th tick (`cap/server/cap_server.py:1284-1285`):

```python
if tick_count % 20 == 0:
    emit("control_loop", "tick", dt_ms=...)
```

At 60 Hz, this is **3 Hz**. Individual events (step_start, obs_built, reward_end, etc.) fire per-step during `learn_skill`. The emitter is zero-overhead UDP — packets are silently dropped if no dashboard listens (`cap/diag/emitter.py:1-4`).

---

## 4. Frequency Mismatch Analysis

### 4.1 CAP Path: 60 Hz control vs 30 Hz policy vs 50 Hz motor

The 60 Hz control loop reads `POLICY_FREQ_HZ`-rate target updates. Between policy steps, the control loop re-sends the same commanded positions (2 sends per policy step on average).

The 50 Hz motor sender in `yam_controller.py` is an additional layer — on real hardware, the arm_server process runs the YAM motor sender, and the CAP server talks to arm_server via Portal RPC (not directly to the motor sender). The arm_server's motor sender continuously re-streams the latest target received from cap_server.

Effective command chain on real hardware:

```
Policy (30 Hz) → cap_server lock → control_loop (60 Hz) → Portal RPC → arm_server → motor_sender (50 Hz) → CAN bus
```

### 4.2 Experimental Path: 30 Hz outer vs 50 Hz motor

The outer loop at 30 Hz calls `env.step()` which writes a target to the arm server. The arm server's background motor sender re-sends at 50 Hz. The ratio is:

- `50 / 30 = 1.67` low-level sends per outer-loop action
- A typical pattern: `a0, a0, a1, a1, a2, a3, a3, ...` (not evenly interleaved because 50 is not an integer multiple of 30)

### 4.3 What "duplicates" actually mean

Repeated motor sends are **not** identical packets. Each send through `_do_send()` (`robot/yam/yam_controller.py:210-239`) recomputes:
- Gravity compensation feedforward torque from current joint state
- MIT mode control command with `(target_pos, stiffness, damping, feedforward_torque)`

So while the target position, kp, and kd stay the same across repeated sends, the feedforward torque may differ slightly based on the current robot state.

---

## 5. Benchmarking Tools

### 5.1 `scripts/bench_station_timing.py`

End-to-end station timing benchmark. Measures per-step latency for the full observation loop (leader reads, follower reads, camera captures) at a configurable target rate.

**Key parameters** (`scripts/bench_station_timing.py:116-128`):
- `target_hz: float = 30.0` — target loop rate
- `duration_s: float = 60.0` — benchmark duration
- `warmup_s: float = 3.0` — warmup period (excluded from stats)

**Metrics collected** (per step):
- `loop_total_ms` — wall time for entire step
- `policy_phase_ms` — leader arm reads (Portal RPC `get_info`)
- `env_obs_phase_ms` — follower reads + camera captures
- `pacing_overshoot_ms` — how much the sleep overshot the target period
- `obs_component_span_ms` — spread between earliest and latest observation component
- Per-arm Portal RPC latency (`left_leader_wait_ms`, etc.)
- Per-camera `get_image` latency

**Output**: JSON summary to `/tmp/station_timing_bench/` with percentile statistics (p50, p95, p99) and top-8 outlier steps.

**Usage**:
```bash
uv run scripts/bench_station_timing.py --target-hz 30 --duration-s 60 --save-raw
```

### 5.2 `scripts/bench_pi05_dummy_inference.py`

Benchmarks Pi05 policy server inference latency with dummy inputs. Reports latency in both milliseconds and outer-loop action-step units.

**Key parameters** (`scripts/bench_pi05_dummy_inference.py:248-284`):
- `--server` — Portal server address (default `localhost:8964`)
- `--control-hz` — outer-loop rate for step-unit conversion (default 30.0)
- `--runs` — number of timed inferences (default 50)

**Output**: JSONL log + PNG histogram of latency distribution.

---

## 6. Mental Models

### CAP Server (Production)

```
Observation enters learn_skill loop at POLICY_FREQ_HZ (30 Hz)
    │
    ├─ RL policy server returns action
    ├─ Reward is evaluated
    ├─ Targets written under _state_lock
    │
    ▼
CONTROL_FREQ_HZ loop (60 Hz) reads targets, sends to arm_server
    │
    ▼
arm_server motor sender (50 Hz) streams CAN commands
    │
    ▼
DaMiao motors execute
```

If you want one number for "new actions from the policy", use **30 Hz**.
If you want one number for "cap_server hardware writes", use **60 Hz**.
If you want one number for "motor CAN bus commands", use **50 Hz**.

### Experimental Path (Research)

```
Policy / chunk wrapper returns one action at policy_control_freq (30 Hz)
    │
    ├─ YamRealEnv.step() applies action
    │
    ▼
arm_server motor sender (50 Hz) re-sends cached target
    │
    ▼
DaMiao motors execute
```

If you want one number for "new actions from the experimental control loop", use **30 Hz**.
If you want one number for "host-to-motor command streaming", use **50 Hz**.

---

## 7. Key Configuration Knobs

| Knob | Location | Default | Effect |
|---|---|---|---|
| `CONTROL_FREQ_HZ` | `cap/config.py:107` | 60.0 | CAP server hardware loop rate |
| `POLICY_FREQ_HZ` | `cap/config.py:110` | 30.0 | RL / skill execution step rate |
| `HIL_POLICY_SLOWDOWN` | `cap/config.py:117` | 3 | RL step decimation during takeover |
| `send_rate_hz` | `robot/yam/yam_controller.py:37` | 50.0 | YAM motor CAN streaming rate |
| `policy_control_freq` | `experimental/yam_control_loop.py:166` | 30 | Experimental outer loop rate |
| `action_horizon` | `experimental/yam_control_loop.py:182` | 50 | Actions per chunk (replan interval) |
| `control_loop_hz` | `robot/models/fello/fello_config.yaml:70` | 100 | Fello gravity-comp loop rate |
| `fps` | `robot/camera_factory.py:169` | 60 | Camera capture frame rate |
| `GEMINI_REWARD_INTERVAL_S` | `cap/reward/gemini_reward.py:40` | 5.0 | Gemini VLM query cooldown |
| `target_hz` | `scripts/bench_station_timing.py:120` | 30.0 | Bench station target loop rate |

---

## Cross-References

- **CAP system architecture**: `docs/CAP_DESIGN.md` — layer stack, `CONTROL_FREQ_HZ` control loop design
- **RL training pipeline**: `docs/RL_PIPELINE_DESIGN.md` — `POLICY_FREQ_HZ` step loop, HIL timing diagrams, per-step event timeline
- **Safety zones**: `docs/SAFETY_ZONE_DESIGN.md` — joint clamping in the `CONTROL_FREQ_HZ` loop
- **Serial footswitch**: `docs/SERIAL_FOOTSWITCH.md` — 50 Hz USB CDC protocol, button mapping
- **Debug dashboard**: `docs/CAP_SYSTEM_DASHBOARD.md` — UDP diagnostics events and visualization
- **Camera configuration**: `docs/MULTI_CAMERA_CONFIG.md` — per-camera env var overrides (FPS, resolution, backend)
- **Data collection infra debug**: `docs/debug_shit_data_collection_infra.md` — references `scripts/bench_station_timing.py`
- **Viser IK teleop**: `docs/VISER_IK_TELEOP_DESIGN.md` — `POLICY_FREQ_HZ` control loop integration with IK targets
