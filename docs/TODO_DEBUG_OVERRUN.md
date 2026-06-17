# Debugging Control Loop Overruns

Guide for diagnosing and fixing timing overruns in the YAM bimanual robot control pipeline. Covers the CAP server control loop, the learn_skill RL loop, the standalone env loops, and the available profiling/benchmarking tools.

> **Related docs:**
> - [`docs/frequency_relationship.md`](frequency_relationship.md) -- frequency layering for the **experimental** (non-CAP) real-robot path (30 Hz outer / 50 Hz motor sender)
> - [`docs/debug_shit_data_collection_infra.md`](debug_shit_data_collection_infra.md) -- detailed investigation of data-collection timestamp non-uniformity (host-side jitter root cause)
> - [`docs/RL_PIPELINE_DESIGN.md`](RL_PIPELINE_DESIGN.md) -- RL training pipeline architecture (`learn_skill`, `serve_rl_policy`, diagnostics)
> - [`docs/SAFETY_ZONE_DESIGN.md`](SAFETY_ZONE_DESIGN.md) -- safety zone clamping inside the control loop
> - [`docs/CAP_DESIGN.md`](CAP_DESIGN.md) -- overall CAP system architecture

---

## 1. Control-Loop Architecture Overview

There are **three independent timing loops** that can produce overrun warnings. Each has its own cadence, pacing mechanism, and overrun detection.

### 1a. CAP Server CONTROL_FREQ_HZ Loop

The main control loop that owns all hardware communication. Runs in a dedicated daemon thread.

| Item | Value | Source |
|------|-------|--------|
| Frequency | 60 Hz (16.67 ms budget) | `cap/config.py:107` -- `CONTROL_FREQ_HZ = 60.0` |
| Period | `CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ` | `cap/config.py:108` |
| Loop entry | `CapServer._control_loop()` | `cap/server/cap_server.py:1098` |
| Thread start | `CapServer.start()` | `cap/server/cap_server.py:3784-3788` |

**What the loop does each tick** (`cap/server/cap_server.py:1111-1285`):

1. **Read hardware state** (lines 1114-1128) -- `get_observations()` for left and right arms via Portal RPC or sim backend; writes into `self.left_joint_pos`, `self.right_joint_pos`, etc. under `self._state_lock`.
2. **Forward kinematics** (lines 1130-1142) -- `self._forward_kinematics(ljp, rjp)` to cache EE poses.
3. **Estop check** (lines 1144-1147) -- if estopped, sleeps `CONTROL_PERIOD_S` and skips sending commands.
4. **Read commanded positions** (lines 1149-1158) -- copies `_cmd_left_jp`, `_cmd_right_jp`, etc. under lock. These are written by `_ik_servo`, `execute_skill`, `go_home`, or `learn_skill`.
5. **ALWAYS_TAKEOVERABLE Fello override** (lines 1161-1238) -- optional press-anchored incremental Fello human takeover.
6. **Joint clamping** (lines 1240-1245) -- `np.clip` to URDF limits.
7. **Send to hardware** (lines 1247-1261) -- `command_joint_state()` for each arm. In sim mode, also calls `self._sim_backend.step()` (line 1264).
8. **Pacing / sleep** (lines 1266-1277) -- hybrid coarse-sleep + busy-wait (see section 3).
9. **Overrun detection** (lines 1278-1281) -- logs warning if tick exceeded budget.
10. **Diagnostics emit** (lines 1283-1285) -- every 20th tick, fires a UDP `emit("control_loop", "tick", dt_ms=...)`.

### 1b. learn_skill RL Loop (POLICY_FREQ_HZ)

The RL training loop inside `learn_skill`. Runs **on the Portal RPC handler thread** (not the control-loop thread), writing commanded positions under `self._state_lock` which the control loop then sends at `CONTROL_FREQ_HZ`.

| Item | Value | Source |
|------|-------|--------|
| Frequency | 30 Hz (33.33 ms budget) | `cap/config.py:110` -- `POLICY_FREQ_HZ = 30.0` |
| Period | `POLICY_PERIOD_S = 1.0 / POLICY_FREQ_HZ` | `cap/config.py:111` |
| Loop entry | `CapServer.learn_skill()` | `cap/server/cap_server.py:2776` |
| Pacing | `cap/server/cap_server.py:3207-3212` -- busy-sleep `0.0001s` increments until `POLICY_PERIOD_S` boundary |
| HIL slowdown | `HIL_POLICY_SLOWDOWN = 3` | `cap/config.py:117` -- during human takeover, obs/reward/RPC skipped 2 of every 3 steps |

**Per-step breakdown** (`cap/server/cap_server.py:2929-3325`):

1. Estop check (line 2930)
2. `emit("cap_server", "step_start", ...)` (line 2938)
3. Read Fello state from cache (lines 2947-2965, timed as `_fello_ms`)
4. Determine action source (rl vs human) and compute action (lines ~2970-3094)
5. Safety zone enforcement via `_enforce_safety_zone()` (line 3117)
6. Gripper clamping (lines 3186-3189)
7. **Write targets under lock** (lines 3192-3197) -- `self._cmd_left_jp[:] = left_jp`, etc.
8. `emit("cap_server", "action_applied", ...)` (line 3199)
9. **Pacing sleep** to `POLICY_PERIOD_S` boundary (lines 3207-3212)
10. If `_do_rl_step`: build obs (`_build_skill_obs` + `_build_rl_obs`, line 3231-3232), get reward (line 3245), send transition to rl_policy_server via Portal RPC (line 3271)
11. Per-step timing log (lines 3283-3301): `total_ms`, `rpc_ms`, `local_ms`, `obs_ms`, `rew_ms`, `fello_ms`, `act_ms`
12. `emit("cap_server", "step_end", ...)` (line 3325)

### 1c. Standalone Env Loops (yam_sim_env / yam_real_env)

Used by the experimental control loop (`experimental/yam_control_loop.py`) and by `eval_yam_sim.py`, `run_data_collection.py`. These are **not** the CAP server path but share the same overrun pattern.

| Env | Overrun detection | Source |
|-----|-------------------|--------|
| Sim | `robot/yam/yam_sim_env.py:93-98` | Prints budget vs actual if `time.time() > sleep_end_time` |
| Real | `robot/yam/yam_real_env.py:225-230` | Same pattern, has `# TODO: Resolve latency issues` |

Both use `time.sleep(0.0001)` busy-sleep pacing (sim: line 99-100; real: line 231-232).

---

## 2. Overrun Detection — Where Warnings Come From

### CAP server (`cap/server/cap_server.py:1278-1281`)

```python
else:
    logger.warning(
        f"[CapServer] Loop overrun: {(time.time() - tick_start) * 1000:.1f}ms"
    )
```

Triggers when `remaining = deadline - time.time()` is negative (i.e., the tick body took longer than `CONTROL_PERIOD_S`). Logged via Python `logging.warning()`.

### Standalone envs (`robot/yam/yam_sim_env.py:94-98`, `robot/yam/yam_real_env.py:226-230`)

```python
print(
    f"Warning: Control loop timing overrun "
    f"(budget {self.control_period * 1000:.1f} ms, "
    f"actual {1000 * (time.time() - self.last_step_time):.1f} ms)"
)
```

Logged to stdout via `print()`.

### Diagnostics emitter (`cap/diag/emitter.py:31-43`)

Every 20th control-loop tick (`cap/server/cap_server.py:1284-1285`):
```python
emit("control_loop", "tick", dt_ms=(time.time() - tick_start) * 1000)
```

Zero-overhead UDP fire-and-forget to `127.0.0.1:9999`. Consumed by `cap/diag/dashboard.py` when running. Disabled with `DIAG_ENABLED=0`.

---

## 3. Pacing Mechanism

### CAP server: hybrid sleep + busy-wait (`cap/server/cap_server.py:1266-1277`)

```python
# Sleep until next tick.  On macOS, time.sleep() overshoots by
# ~2 ms due to coarse timer resolution, so we sleep only up to a
# threshold and busy-wait the remainder for precise timing.
deadline = tick_start + CONTROL_PERIOD_S
remaining = deadline - time.time()
if remaining > 0:
    _BUSYWAIT_S = 0.002  # busy-wait the last 2 ms
    coarse = remaining - _BUSYWAIT_S
    if coarse > 0:
        time.sleep(coarse)
    while time.time() < deadline:
        pass  # spin
```

**Why this matters:** On macOS, `time.sleep()` resolution is ~2 ms. On Linux, it is typically ~1 ms. The busy-wait tail compensates. However, the busy-wait burns CPU. If other threads or processes are competing for CPU, the `time.time()` poll can still overshoot.

### learn_skill loop (`cap/server/cap_server.py:3209-3212`)

```python
sleep_end = last_step_time + POLICY_PERIOD_S
while time.time() < sleep_end:
    time.sleep(0.0001)
last_step_time = time.time()
```

No hybrid busy-wait -- uses fine `0.1ms` sleep increments only. Less CPU than spinning but lower timing precision.

### Standalone envs (`robot/yam/yam_sim_env.py:99-101`, `robot/yam/yam_real_env.py:231-233`)

Same pattern as learn_skill: `time.sleep(0.0001)` busy-sleep.

---

## 4. Common Overrun Causes

### 4a. CAP Server Control Loop

| Suspect | Why it overruns | How to check |
|---------|----------------|--------------|
| **Arm get_observations() RPC** | Portal RPC round-trip to follower arm server. On real hardware, CAN bus reads can stall. | Check `left_follower_wait_ms` / `right_follower_wait_ms` in `bench_station_timing.py` output |
| **FK computation** | `_forward_kinematics()` calls pinocchio FK. Usually fast (<1 ms) but can spike under GC. | Add timing around lines 1134-1141 |
| **Sim backend step** | `self._sim_backend.step()` (line 1264) in sim mode. MuJoCo physics step cost depends on scene complexity. | Profile sim step separately |
| **GIL contention** | Python GIL held by other threads (learn_skill obs building, camera streaming, video encoding). | Check if overruns correlate with learn_skill activity |
| **OS scheduler** | Thread not scheduled in time. Worse on macOS. | Run on Linux with `SCHED_FIFO` or nice priority |

### 4b. learn_skill RL Loop

| Suspect | Why it overruns | How to check |
|---------|----------------|--------------|
| **Camera reads in _build_skill_obs()** | Camera `get_image()` can take 5-30+ ms, especially RealSense USB reads. | Watch `obs_ms` in learn_skill per-step log |
| **Reward computation** | VLM/vision reward can be slow. | Watch `rew_ms` in learn_skill per-step log |
| **RL server RPC** | `rl_client.step(transition).result()` blocks until the policy server responds. | Watch `rpc_ms` in learn_skill per-step log |
| **Video encoding** | Video encoding is offloaded to background thread (`cap/server/cap_server.py:449-457`, `_frame_writer_loop` at line 506) specifically to avoid overruns. If the queue backs up, `queue.put()` can block. | Check `_frame_queue.qsize()` or add timing around `record_step()` |
| **Episode save/reset** | Episode save triggers disk I/O. Known to cause ~580 ms overruns at episode boundaries (see `docs/debug_shit_data_collection_infra.md`, line 328). | These are expected; they occur between episodes, not during control |

### 4c. Standalone Env Loops

| Suspect | Why it overruns | How to check |
|---------|----------------|--------------|
| **Image rendering (sim)** | `_get_obs()` renders camera images. On Apple Silicon MBP, MuJoCo rendering is ~30+ ms (noted at `robot/yam/yam_sim_env.py:103`). | Reduce camera count or resolution |
| **Follower arm command** | Real env `command_joint_pos()` (line 217-221) goes through Portal RPC. | Use `bench_station_timing.py` |

---

## 5. Profiling and Benchmarking Tools

### 5a. CAP Agent Profiler (`cap/agent/profiler.py`)

Wraps every tool callable with timing that prints duration and idle gaps. Activated automatically by `run_script.py` (line 558-579).

**Key functions:**
- `enable_file_logging(log_dir, script_name)` (line 36) -- opens a real-time line-buffered log file
- `wrap_callables_with_timing(callables)` (line 162) -- wraps each callable with `_wrap_with_timing()`
- `set_state_fn(fn)` (line 62) -- registers `get_robot_state` for state snapshots after state-changing tools
- `enable_stdout_tee()` (line 288) -- tees stdout to log file via `StdoutTee` class

**Output format:**
```
[profile HH:MM:SS.fff] #N tool_name  (wait Xms)
[profile HH:MM:SS.fff] #N tool_name -> Xms
```

**Usage:**
```bash
uv run run_script.py --sim --file cap/saved_scripts/my_script.py
# Logs written to logs/ directory by default (--log-dir to override)
# Disable with --no-log
```

### 5b. Station Timing Benchmark (`scripts/bench_station_timing.py`)

Standalone benchmark that exercises the exact same hardware path as the control loop (Portal RPC to followers/leaders, camera `get_image()`). Does NOT require cap_server to be running.

**What it measures** (lines 163-180, summary function):
- `loop_total_ms` -- full iteration time
- `policy_phase_ms` -- leader arm RPC time
- `env_obs_phase_ms` -- follower arm RPC + camera reads
- `pacing_ms` -- time spent sleeping
- `pacing_overshoot_ms` -- how much sleep overshot
- `overrun_ms` -- how much the tick exceeded budget
- `obs_component_span_ms` -- wall-clock span across all obs components
- Per-component: `left_follower_wait_ms`, `right_follower_wait_ms`, `{camera}_get_image_ms`, etc.

**Statistics reported** (lines 163-180): mean, std, min, p50, p95, p99, max, plus count of samples exceeding 2/5/10/20 ms thresholds.

**Usage:**
```bash
uv run scripts/bench_station_timing.py \
    --target-hz 30 \
    --duration-s 60 \
    --warmup-s 3 \
    --include-cameras \
    --include-followers \
    --include-leaders \
    --save-raw \
    --label my_test
# Output: /tmp/station_timing_bench/<label>_<timestamp>_summary.json
```

**Args** (dataclass at line 115-128):
| Arg | Default | Description |
|-----|---------|-------------|
| `--target-hz` | 30.0 | Target loop frequency |
| `--duration-s` | 60.0 | Benchmark duration |
| `--warmup-s` | 3.0 | Warmup before recording |
| `--include-leaders` | True | Include leader arm RPC |
| `--include-followers` | True | Include follower arm RPC |
| `--include-cameras` | True | Include camera reads |
| `--save-raw` | False | Save per-step JSONL |
| `--output-dir` | /tmp/station_timing_bench | Output directory |

### 5c. Station Timing Comparison Visualizer

Compare two benchmark runs side-by-side to find regressions.

```bash
uv run scripts/station_timing_compare_vis.py <summary_a.json> <summary_b.json>
# Generates: docs/station_timing_compare_report.html
```

Also: `scripts/render_station_timing_compare.py` for alternative rendering.

### 5d. UDP Diagnostics Emitter (`cap/diag/emitter.py`)

Fire-and-forget UDP packets with ~1 us overhead. Consumed by `cap/diag/dashboard.py`.

```python
from cap.diag.emitter import emit
emit("cap_server", "step_start", ep=3, step=142)
emit("cap_server", "obs_built", ep=3, step=142, camera_read_ms=0.8)
```

**Key emit points in cap_server.py:**
| Line | Event | Payload |
|------|-------|---------|
| 1285 | `control_loop` / `tick` | `dt_ms` (every 20th tick) |
| 2938 | `cap_server` / `step_start` | `ep`, `step`, `max_steps`, `cumR` |
| 3199 | `cap_server` / `action_applied` | `ep`, `step`, `action_source` |
| 3234 | `cap_server` / `obs_built` | `ep`, `step`, `camera_read_ms` |
| 3243 | `cap_server` / `reward_start` | `ep`, `step` |
| 3248 | `cap_server` / `reward_end` | `ep`, `step`, `reward` |
| 3256 | `cap_server` / `record_done` | `ep`, `step` |
| 3269 | `cap_server` / `rpc_call_start` | `ep`, `step` |
| 3273 | `cap_server` / `rpc_call_end` | `ep`, `step` |
| 3325 | `cap_server` / `step_end` | `ep`, `step` |

### 5e. learn_skill Built-in Timing Log

Every step of `learn_skill` logs a breakdown at `cap/server/cap_server.py:3283-3301`:

```
[learn_skill] ep=0 step=42/500 src=rl rl_step=True r=0.010 cumR=0.420 |
  total=35ms  rpc=12ms local=23ms (obs=8 rew=3 fello=0 act=1)
```

Fields: `total_ms`, `rpc_ms` (RL server round-trip), `local_ms` (total minus RPC), `obs_ms` (camera + obs build), `rew_ms` (reward), `fello_ms` (Fello state read), `act_ms` (action application + write under lock).

---

## 6. Debugging Procedure

### Step 1: Identify which loop is overrunning

- **"[CapServer] Loop overrun: X.Xms"** -- CAP server 60 Hz control loop (`cap/server/cap_server.py:1280`)
- **"Warning: Control loop timing overrun (budget X ms, actual Y ms)"** -- standalone env (`yam_sim_env.py:95` or `yam_real_env.py:227`)
- **learn_skill per-step log shows `total_ms` >> 33.3 ms** -- RL loop overrun (no explicit warning printed)

### Step 2: Run the station timing benchmark

Isolates hardware timing from application logic:

```bash
# On the robot machine, with arm servers already running:
uv run scripts/bench_station_timing.py --target-hz 60 --duration-s 30 --save-raw
```

If `overrun_ms` is consistently > 0 here, the problem is at the hardware/RPC/camera level. If not, the problem is in the application logic (obs building, reward, RL RPC, etc.).

### Step 3: Use the profiler for script-level diagnosis

```bash
uv run run_script.py --sim --file cap/saved_scripts/<your_script>.py
# Check the log file in logs/ for per-tool timing
```

### Step 4: Add targeted timing to the control loop

If the station benchmark is clean but the control loop still overruns, add timing instrumentation inside `_control_loop()` around the suspects:

```python
# Example: time arm reads (around cap/server/cap_server.py:1114-1128)
_t0 = time.time()
obs = self._arms[side].get_observations()
_arm_read_ms = (time.time() - _t0) * 1000
if _arm_read_ms > 5:
    logger.warning(f"[CapServer] Slow {side} arm read: {_arm_read_ms:.1f}ms")
```

### Step 5: For learn_skill overruns, read the built-in log

The per-step breakdown (`total_ms`, `rpc_ms`, `obs_ms`, `rew_ms`, etc.) at `cap/server/cap_server.py:3283-3301` tells you exactly which phase is slow. Common culprits:
- `obs_ms` high: camera reads are slow (USB contention, frame drops)
- `rpc_ms` high: RL policy server is slow to respond (GPU inference time, network latency)
- `rew_ms` high: reward function (VLM query, vision processing) is slow

### Step 6: Check for GIL/thread contention

The CAP server runs multiple threads:
- Control loop thread (`_control_loop`, started at line 3786)
- Fello loop thread (`_fello_loop`, started at line 3793-3796)
- Camera streaming threads (per-camera `_CameraClient` background reader, lines ~260-314)
- Video writer thread (`_frame_writer_loop`, `cap/server/cap_server.py:506`)
- Portal RPC handler threads (learn_skill, _ik_servo, etc.)

All share the Python GIL. Heavy CPU work in any thread (numpy ops, image processing, FK) can delay other threads.

---

## 7. Known Issues and Mitigations

### Video encoding offloaded to background thread

Video frame encoding (`cvtColor` + `VideoWriter.write`) was moved to a dedicated background thread to prevent control loop overruns. See `_SkillRecorder._frame_writer_loop()` at `cap/server/cap_server.py:506-513` and the queue at line 451.

### macOS timer resolution

`time.sleep()` on macOS overshoots by ~2 ms. The CAP server control loop compensates with a hybrid coarse-sleep + busy-wait strategy (`cap/server/cap_server.py:1266-1277`). The standalone envs and learn_skill loop use `time.sleep(0.0001)` without this optimization -- they may have slightly worse timing precision.

### Episode save/reset spikes

Episode save to disk can cause ~580 ms overruns at episode boundaries. These are **between** episodes and do not affect in-episode control quality. See `docs/debug_shit_data_collection_infra.md`, "Save/reset overruns are separate" section.

### Host-specific timing non-uniformity

Per `docs/debug_shit_data_collection_infra.md`, the remaining timestamp non-uniformity in data collection is primarily a host-side issue:
- Client-side Portal/RPC wait jitter
- OS scheduler jitter
- Sequential host-side observation assembly
- Not the motor controller, camera, or visualizer

---

## 8. Quick-Reference: Key File Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| CONTROL_FREQ_HZ / CONTROL_PERIOD_S | `cap/config.py` | 107-108 |
| POLICY_FREQ_HZ / POLICY_PERIOD_S | `cap/config.py` | 110-111 |
| HIL_POLICY_SLOWDOWN | `cap/config.py` | 117 |
| Control loop body | `cap/server/cap_server.py` | 1098-1285 |
| Control loop overrun warning | `cap/server/cap_server.py` | 1279-1281 |
| Control loop pacing (hybrid) | `cap/server/cap_server.py` | 1266-1277 |
| Control loop start | `cap/server/cap_server.py` | 3784-3788 |
| learn_skill entry | `cap/server/cap_server.py` | 2776 |
| learn_skill step loop | `cap/server/cap_server.py` | 2929-3325 |
| learn_skill pacing | `cap/server/cap_server.py` | 3207-3212 |
| learn_skill per-step log | `cap/server/cap_server.py` | 3283-3301 |
| Sim env overrun warning | `robot/yam/yam_sim_env.py` | 93-98 |
| Real env overrun warning | `robot/yam/yam_real_env.py` | 225-230 |
| Video writer background thread | `cap/server/cap_server.py` | 449-457, 506-513 |
| UDP diagnostics emitter | `cap/diag/emitter.py` | 31-43 |
| CAP agent profiler | `cap/agent/profiler.py` | 1-301 |
| Station timing benchmark | `scripts/bench_station_timing.py` | 1-420 |
| Station timing comparison | `scripts/station_timing_compare_vis.py` | full file |
| Camera streaming thread | `cap/server/cap_server.py` | ~260-314 |

---

## 9. Original TODO Checklist

These were the original debugging steps. They remain valid as a workflow:

- [ ] Enable profiling: `uv run run_script.py --sim --file <script>.py` (profiler auto-enabled unless `--no-log`)
- [ ] Examine the log file in `logs/` for per-tool timing and idle gaps
- [ ] Run `bench_station_timing.py` to isolate hardware-level timing
- [ ] Fix any identified bottleneck (camera reads, RL RPC, reward computation, GIL contention)
- [ ] Verify: no overrun warnings and the run executes without error
- [ ] Repeat until clean