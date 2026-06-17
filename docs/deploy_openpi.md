
# Deploy OpenPI Policy Server (Pi0.5)

How the YAM bimanual robot consumes actions from an OpenPI-based policy server (Pi0.5 checkpoint). Covers the external server, the CAP-side client chain, and the tmux launch helpers.

**Related docs:**
- `docs/CAP_DESIGN.md` -- CAP system architecture and layer stack
- `docs/RL_PIPELINE_DESIGN.md` -- RL training pipeline (learn_skill, serve_rl_policy)
- `docs/remote_serving.md` -- Remote model serving, SSH tunnels, port map
- `docs/TABLE_BUSSING_SKILLS.md` -- Table bussing skill tools

---

## 1. Architecture Overview

```
 OpenPI repo (GPU machine)                   lecar-tbd (robot machine)
 ─────────────────────────                   ──────────────────────────
 openpi-main/                                cap/server/cap_server.py
   src/openpi/serving/yam/                       │
     launch_yam_policy_server*.sh                │ Portal RPC
     yam_policy_server_correct.py                │ (port 8964)
         │                                       │
         │ Portal RPC server                     ▼
         │ binds step(), reset(),           RobotInterfaceClient
         │ health_check()                   (experimental/robot_interface.py:228)
         │                                       │
         ▼                                       │ .step(payload) → action_chunk
     Pi0.5 model checkpoint                      ▼
                                            SyncChunkingPolicy
                                            (experimental/sync_chunking_policy.py:12)
                                                 │
                                                 │ .get_action(obs) → (action, info)
                                                 ▼
                                            cap_server._run_policy_steps()
                                            (cap/server/cap_server.py:2423)
                                                 │
                                                 │ writes _cmd_left_jp, _cmd_right_jp
                                                 ▼
                                            CONTROL_FREQ_HZ hardware loop
```

The OpenPI server is **external** to lecar-tbd. It lives in a separate repo (`openpi-main`) and exposes a Portal RPC `step(payload)` endpoint. CAP never imports OpenPI code directly -- it only sends serialized observations over Portal and receives action chunks back.

---

## 2. Configuration

All policy-server settings live in `cap/config.py`.

### 2.1 Ports and Addresses

| Constant | Value | File | Line | Purpose |
|---|---|---|---|---|
| `POLICY_SERVER_PORT` | `8964` | `cap/config.py` | 66 | Default Portal RPC port for the policy server |
| `PI05_POLICY_SERVER` | `localhost:8964` (env override) | `cap/config.py` | 70-73 | Address CAP uses to reach the Pi0.5 server |
| `PI05_POLICY_LAUNCH_SCRIPT` | `/home/lecar/Project/openpi-main/.../launch_yam_policy_server_functional_grasp.sh` (env override) | `cap/config.py` | 74-77 | Path to the shell script that starts the OpenPI server |
| `DEFAULT_POLICY_SERVER` | `localhost:8964` (env override) | `cap/config.py` | 294-296 | Fallback for `execute_skill` and `cap_agent.py` |

### 2.2 Named Model Config

`POLICY_MODEL_CONFIGS` (`cap/config.py:78-87`) defines named policy models that CAP can call on demand. Currently one entry:

```python
PI05_FUNCTIONAL_GRASP_MODEL = "Pi05-stateless-functional-grasp"

POLICY_MODEL_CONFIGS = {
    "Pi05-stateless-functional-grasp": {
        "server": PI05_POLICY_SERVER,           # "localhost:8964"
        "default_task_description": "functional grasp",
        "embodiment_tag": "xdof",
        "resolution": 480,
        "launch_script": PI05_POLICY_LAUNCH_SCRIPT,
        "reset_behavior": "noop",
    },
}
```

The `policy_output` tools (`cap/agent/tools/policy_output.py`) use this config to resolve the server address, embodiment, and reset behavior at runtime.

### 2.3 Skill-Name Routing

`SKILL_POLICY_SERVER` (`cap/config.py:92-95`) maps skill names to policy server addresses for `execute_skill`:

```python
SKILL_POLICY_SERVER = {
    "hold_still": "hold",                       # built-in, no server
    "pick up stick": f"localhost:{POLICY_SERVER_PORT}",  # OpenPI
}
```

### 2.4 Environment Variables

| Variable | Default | Where Used |
|---|---|---|
| `PI05_POLICY_SERVER` | `localhost:8964` | `cap/config.py:70` |
| `PI05_POLICY_LAUNCH_SCRIPT` | `...launch_yam_policy_server_functional_grasp.sh` | `cap/config.py:74` |
| `DEFAULT_POLICY_SERVER` | `localhost:8964` | `cap/config.py:294` |

Override these in your shell or in the tmux launch script to point at a remote GPU host.

---

## 3. Client-Side Call Chain (CAP)

When the agent calls `use_policy_output(model="Pi05-stateless-functional-grasp", ...)`, the following chain executes inside `cap_server.py`:

### 3.1 Tool Layer

`cap/agent/tools/policy_output.py` defines four Portal RPC tool wrappers, registered in `cap/agent/tools/__init__.py:138-205`:

| Tool | RPC | Purpose |
|---|---|---|
| `StartPolicyOutputTool` | `start_policy_output(model, replan_horizon, task_description, policy_server)` | Create a policy session |
| `StepPolicyOutputTool` | `step_policy_output(max_steps)` | Execute a bounded burst of policy steps |
| `StopPolicyOutputTool` | `stop_policy_output()` | Tear down the session, clear state |
| `UsePolicyOutputTool` | `use_policy_output(model, replan_horizon, max_steps, task_description, policy_server)` | One-shot: start + step + stop |

Additionally, `ExecuteSkillTool` (`cap/agent/tools/skill.py:22`) provides a lower-level `execute_skill` RPC that also creates a policy runtime internally.

### 3.2 cap_server: Session Management

`cap/server/cap_server.py` manages the policy session:

1. **`start_policy_output`** (line 2476) -- Resolves model config from `POLICY_MODEL_CONFIGS`, creates the policy runtime, stores it in `_policy_output_session`.

2. **`_create_policy_runtime`** (line 2298) -- Builds the inference stack:
   ```
   EmbodimentTag("xdof")
        ↓
   PolicyAdapters(map_observation, map_action)
        ↓
   RobotInterface(server_address="localhost:8964", adapters=...)
        ↓
   SyncChunkingPolicy(policy=robot_interface, action_exec_horizon=replan_horizon)
   ```

3. **`step_policy_output`** (line 2564) / **`_run_policy_steps`** (line 2423) -- Loops at `POLICY_PERIOD_S` (30 Hz): builds obs via `_build_skill_obs` (line 3629), calls `policy.get_action(obs)`, writes joint targets to `_cmd_left_jp` / `_cmd_right_jp` under `_state_lock`.

4. **`stop_policy_output`** (line 2639) -- Clears the session, resets policy state.

5. **Dirty tracking** -- Any concurrent robot command (`freespace_move`, `go_home`, `set_gripper`, etc.) calls `_mark_policy_output_dirty(reason)` (line 2361), which clears the action queue and sets `session.dirty = True`. On the next `step_policy_output`, the policy re-samples fresh.

### 3.3 Policy Runtime Internals

**`SyncChunkingPolicy`** (`experimental/sync_chunking_policy.py:12`):
- Wraps `RobotInterface` (the inner Portal client).
- Maintains an `action_queue` (deque). When empty, calls `self.policy.get_action(obs)` to get a new chunk from the server, truncates it to `action_exec_horizon`, and enqueues.
- `clear_local_state()` flushes the queue without resetting the server.

**`RobotInterface`** (`experimental/robot_interface.py:28`):
- Instantiates `RobotInterfaceClient(server_address)` on init (line 52).
- `get_action(obs)` (line 79): runs `adapters.map_observation(obs)` to produce `(images, proprio)`, builds `VLAStepData`, calls `self.step(vla_step_data)`.
- `step(vla_step_data)` delegates to `RobotInterfaceClient.step()`.

**`RobotInterfaceClient`** (`experimental/robot_interface.py:228`):
- Connects to the OpenPI server via `portal.Client(server_address)` with a 120s startup timeout and health check.
- `step(vla_step_data)` (line 260): serializes `VLAStepData` into a dict via `_serialize_vla_step_data()` (line 212), calls `self._client.step(payload).result()`, returns the action dict.
- Supports profiling via `ROBOT_INTERFACE_PROFILE=1` env var.

### 3.4 Observation and Action Key Mapping

`experimental/key_remapping_utils.py` maps between the env observation format (`left_joint_pos`, `top_camera_image`, ...) and the OpenPI wire format.

For `EmbodimentTag.XDOF` at resolution 480:

**Observation (proprio):**
| Env Key | Wire Key |
|---|---|
| `left_joint_pos` | `joint_pos_obs_left` |
| `left_gripper_pos` | `gripper_pos_obs_left` |
| `right_joint_pos` | `joint_pos_obs_right` |
| `right_gripper_pos` | `gripper_pos_obs_right` |

**Observation (cameras):**
| Env Key | Wire Key |
|---|---|
| `top_camera_image` | `observation.images.top_camera-images-rgb` |
| `left_camera_image` | `observation.images.left_camera-images-rgb` |
| `right_camera_image` | `observation.images.right_camera-images-rgb` |

**Action (server response -> env):**
| Wire Key | Env Key |
|---|---|
| `joint_pos_action_left` | `left_joint_pos` |
| `gripper_pos_action_left` | `left_gripper_pos` |
| `joint_pos_action_right` | `right_joint_pos` |
| `gripper_pos_action_right` | `right_gripper_pos` |

The server also supports EE-pose action format (`left_ee_pos`, `left_ee_quat_xyzw`, etc.) -- `_resolve_policy_action_targets` in `cap_server.py:2389` detects the format and runs IK if needed.

### 3.5 VLAStepData Wire Format

The Portal RPC payload sent to the OpenPI server is a dict with these keys (`experimental/robot_interface.py:212-225`):

```python
{
    "images": {"top_camera-images-rgb": np.ndarray(480,640,3), ...},
    "states": {"joint_pos_obs_left": np.ndarray(6), ...},
    "actions": {},
    "text": "functional grasp",
    "rl_info": None,
    "embodiment": "xdof",
    "is_demonstration": False,
    "metadata": {},
}
```

---

## 4. Observation Building

`cap_server._build_skill_obs()` (`cap/server/cap_server.py:3629`) builds the observation dict that feeds the policy:

```python
{
    "left_joint_pos":    np.ndarray(6),    # from _state_lock
    "left_gripper_pos":  np.ndarray(1),
    "right_joint_pos":   np.ndarray(6),
    "right_gripper_pos": np.ndarray(1),
    "top_camera_image":  np.ndarray(H,W,3),
    "left_camera_image": np.ndarray(H,W,3),
    "right_camera_image":np.ndarray(H,W,3),
    "annotation.task":   "functional grasp",
}
```

Camera names are resolved from `CAMERA_NAMES` (`cap/config.py:222`), which reads from the active station profile.

---

## 5. Launch Methods

### 5.1 Standalone tmux helper (local server)

```bash
# Launch the Pi05 policy server in its own tmux session.
# Reads PI05_POLICY_LAUNCH_SCRIPT env var or defaults to the functional-grasp script.
bash tmux/table_bussing/table_bussing_history/launch_pi05_policy.sh
bash tmux/table_bussing/table_bussing_history/launch_pi05_policy.sh --no-attach
```

**Source:** `tmux/table_bussing/table_bussing_history/launch_pi05_policy.sh`

This creates a tmux session `pi05-policy` and runs the OpenPI launch script inside it. The server loads the checkpoint and idles on port 8964.

### 5.2 Pi05 + Table Bussing (SSH tunnel to remote GPU)

```bash
# Full table bussing stack with Pi05 via SSH tunnel.
bash tmux/table_bussing/table_bussing_history/launch_table_bussing_pi05.sh
bash tmux/table_bussing/table_bussing_history/launch_table_bussing_pi05.sh --sim
bash tmux/table_bussing/table_bussing_history/launch_table_bussing_pi05.sh --pi05-server 172.26.43.77:8964
bash tmux/table_bussing/table_bussing_history/launch_table_bussing_pi05.sh --pi05-user tonghe
```

**Source:** `tmux/table_bussing/table_bussing_history/launch_table_bussing_pi05.sh`

Layout:
```
┌────────────────┬────────────────┐
│  cap_server    │  cap_agent     │
├────────────────┼────────────────┤
│  bundlesdf     │  pi05 tunnel   │
├────────────────┴────────────────┤
│  cap_ui (dev)                    │
└──────────────────────────────────┘
```

The `pi05 tunnel` pane runs `ssh -N -L 8964:localhost:8964 tonghe@<gpu-host>`, making the remote OpenPI server appear as `localhost:8964` on the robot machine. `cap_server` and `cap_agent` are started with `PI05_POLICY_SERVER=localhost:8964`.

### 5.3 Experimental: yam_control_loop (legacy)

```bash
# Terminal 1: evaluation control loop
cd ~/Project/lecar-tbd
uv run python launch.py --mode=evaluate
uv run python -m experimental.yam_control_loop --use-real-robot --action-horizon 64

# Terminal 2: OpenPI server
bash /home/lecar/Project/openpi-main/src/openpi/serving/yam/launch_yam_policy_server.sh
```

**Source:** `experimental/yam_control_loop.py`

This is the older standalone control loop. It uses `AsyncChunkingPolicy`, `RealtimeRTCChunkingPolicy`, or `SyncChunkingPolicy` directly (not through `cap_server`). The CAP-based `policy_output` tools (Section 3) are the preferred path.

---

## 6. Benchmarking

```bash
# Benchmark Pi05 inference latency with dummy inputs.
uv run python scripts/bench_pi05_dummy_inference.py \
    --server localhost:8964 \
    --height 480 --width 640 \
    --text "functional grasp" \
    --warmup 1 --runs 50 --control-hz 30

# Only compute stats from an existing log file:
uv run python scripts/bench_pi05_dummy_inference.py --stats-only
```

**Source:** `scripts/bench_pi05_dummy_inference.py`

Sends dummy RGB images and zero proprioception directly to the Portal `step(payload)` RPC. Produces:
- `scripts/pi05_dummy_inference_bench.jsonl` -- Per-run latency log
- `scripts/pi05_dummy_inference_bench.jsonl.stats.json` -- Summary statistics
- `scripts/pi05_dummy_inference_bench.png` -- Latency distribution histograms (ms and action-step units)

---

## 7. System Catalog Entry

The policy server is registered in `bringup/system_catalog.py`:

| Service ID | Port | Protocol | Probe |
|---|---|---|---|
| `policy_server` | 8964 | TCP | TCP connect |
| `rl_policy_server` | 8965 | TCP | TCP connect |

The dashboard edge `cap_server -> policy_server` (port 8964) monitors connectivity.

---

## 8. Quick Reference: File Index

| File | Purpose |
|---|---|
| `cap/config.py:66-87` | Port, address, and model config constants |
| `cap/config.py:92-95` | Skill-name to policy-server routing |
| `cap/config.py:294-296` | `DEFAULT_POLICY_SERVER` |
| `cap/server/cap_server.py:141-151` | `_PolicyOutputSession` dataclass |
| `cap/server/cap_server.py:2298-2327` | `_create_policy_runtime()` -- builds the inference stack |
| `cap/server/cap_server.py:2423-2474` | `_run_policy_steps()` -- 30 Hz step loop |
| `cap/server/cap_server.py:2476-2562` | `start_policy_output()` RPC |
| `cap/server/cap_server.py:2564-2637` | `step_policy_output()` RPC |
| `cap/server/cap_server.py:2639-2670` | `stop_policy_output()` RPC |
| `cap/server/cap_server.py:2672-2719` | `use_policy_output()` RPC (one-shot wrapper) |
| `cap/server/cap_server.py:2721-2774` | `execute_skill()` RPC (lower-level) |
| `cap/server/cap_server.py:3629-3643` | `_build_skill_obs()` -- observation builder |
| `cap/agent/tools/policy_output.py` | Agent-side tool wrappers for policy_output RPCs |
| `cap/agent/tools/skill.py:22-70` | `ExecuteSkillTool` |
| `cap/agent/tools/__init__.py:138-205` | Tool registration for policy_output tools |
| `experimental/robot_interface.py:28-166` | `RobotInterface` -- adapters + Portal client |
| `experimental/robot_interface.py:228-291` | `RobotInterfaceClient` -- Portal RPC to server |
| `experimental/robot_interface.py:212-225` | `_serialize_vla_step_data()` -- wire format |
| `experimental/sync_chunking_policy.py:12-78` | `SyncChunkingPolicy` -- action queue + chunking |
| `experimental/key_remapping_utils.py` | Obs/action key mapping for XDOF embodiment |
| `experimental/embodiment_tags.py:241-244` | `EmbodimentTag.XDOF` |
| `experimental/_types.py:7-32` | `VLAStepData` dataclass |
| `experimental/yam_control_loop.py` | Legacy standalone control loop (not CAP-based) |
| `scripts/bench_pi05_dummy_inference.py` | Latency benchmarking tool |
| `bringup/system_catalog.py:196-200` | System catalog entry for policy_server |
| `tmux/table_bussing/table_bussing_history/launch_pi05_policy.sh` | Standalone tmux launcher |
| `tmux/table_bussing/table_bussing_history/launch_table_bussing_pi05.sh` | Full stack + SSH tunnel launcher |

---

## 9. Troubleshooting

**Server not reachable:**
```bash
# Check if the Portal server is listening
python -c "import portal; c = portal.Client('localhost:8964'); print(c.health_check().result(timeout=5))"
```

**SSH tunnel to remote GPU:**
```bash
# Forward local 8964 to remote 8964
ssh -N -L 8964:localhost:8964 <user>@<gpu-host>
```

**Profiling RobotInterface latency:**
```bash
ROBOT_INTERFACE_PROFILE=1 uv run cap/server/cap_server.py
```

This prints per-call timings for observation mapping, VLA preparation, Portal RPC, and action mapping.

**Wrong embodiment tag:** The server and client must agree on the embodiment. The default is `"xdof"` (`cap/config.py:82`). If the OpenPI server expects a different tag, update `POLICY_MODEL_CONFIGS` or pass `embodiment_tag` in `execute_skill` params.

**Action format mismatch:** The server can return joint-space (`left_joint_pos`, `right_joint_pos`) or EE-pose (`left_ee_pos`, `left_ee_quat_xyzw`, ...) actions. `cap_server._resolve_policy_action_targets()` (line 2389) auto-detects the format and runs inverse kinematics for EE-pose actions.