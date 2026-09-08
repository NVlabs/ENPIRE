# Design Doc: CAP for RL Pipeline

> Paths written as `cap/…`, `robot/…`, `experimental/…`, `tmux/…` or
> `experiments/…` are relative to `enpire/env/forge/`.
> **Last updated**: 2026-04-08
> **Status**: Implemented and actively used

## Goal

Use the CAP skill infrastructure (`enpire/`) as the RL actor environment. The `learn_skill()` method on `CapServer` communicates with an external `rl_policy_server` over Portal RPC, receiving single-step actions and sending back (obs, action, reward, done) transitions. Human-in-the-loop takeover via Fello arms is fully supported: human actions are tagged as `action_source="human"` and routed to the intervention buffer on the RL server.

## Cross-References

| Document | Relevance |
|----------|-----------|
| [`docs/CAP_DESIGN.md`](CAP_DESIGN.md) | CAP server architecture, control loop, Portal RPC bindings |
| [`SKILL_LIBRARY.md`](SKILL_LIBRARY.md) | Table bussing skill tools used in RL scripts |

## Projects Involved

- **`enpire/`** — CAP server & agent: robot control loop (`CONTROL_FREQ_HZ`), skill execution (`POLICY_FREQ_HZ`), Fello human takeover, camera pipeline, reward server, diagnostics dashboard, safety zones
- **`bc_policy/`** — (Optional, for full RL training) SAC agent, agentlace actor-learner communication, replay buffers, training loop (`train_delta_rlpd.py`)
- **`scripts/serve_rl_policy.py`** — Dummy RL policy server for testing (random delta actions, no GPU needed)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        bc_policy (GPU machine)                       │
│                                                                      │
│  ┌─────────────────────┐        agentlace          ┌──────────────┐ │
│  │   rl_policy_server   │◄════════════════════════►│   learner     │ │
│  │  (serve_rl_policy.py)│   (port 8001/8002)       │(train_delta   │ │
│  │                      │   transitions ──────►    │ _rlpd.py)     │ │
│  │  - SAC agent         │   ◄────── params         │               │ │
│  │  - base agent        │                          │ - SAC update  │ │
│  │  - Portal RPC server │                          │ - replay buf  │ │
│  └──────────┬───────────┘                          │ - demo buf    │ │
│             │ Portal RPC (port 8965)                └──────────────┘ │
└─────────────┼────────────────────────────────────────────────────────┘
              │
              │  Network (LAN)
              │
┌─────────────┼────────────────────────────────────────────────────────┐
│             │              enpire (Robot machine)                  │
│  ┌──────────▼────────────┐                                           │
│  │     CAP Server         │                                          │
│  │   (cap_server.py)      │                                          │
│  │                        │                                          │
│  │  learn_skill()         │◄── Fello leader arms (human takeover)    │
│  │   - POLICY_FREQ_HZ    │                                          │
│  │     step loop          │──► Reward (local fn or server:8500)      │
│  │   - calls rl_policy    │                                          │
│  │     server.step()      │──► _SkillDataRecorder (local disk)       │
│  │   - detects takeover   │                                          │
│  │   - safety zone        │──► Diagnostics dashboard (:8888)         │
│  │     enforcement        │                                          │
│  └────────────────────────┘                                          │
│             │                                                        │
│  ┌──────────▼────────────┐                                           │
│  │   YAM Robot            │                                          │
│  │   CONTROL_FREQ_HZ     │                                          │
│  │   Arms + Cameras       │                                          │
│  └────────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. `learn_skill()` — `cap/server/cap_server.py`

The single unified RL training loop method on `CapServer`. Registered as Portal RPC at `cap/server/cap_server.py`:

```python
self._server.bind("learn_skill", self.learn_skill, workers=1)
```

**Signature** (`cap/server/cap_server.py`):
```python
def learn_skill(
    self,
    skill_name: str,
    params: dict | None = None,
) -> dict:
```

**Parameters (via `params` dict)**:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `task_description` | `str` | `skill_name` | Task string passed to `_build_skill_obs()` for policy conditioning |
| `max_steps` | `int` | `RL_EPISODE_MAX_STEPS` (1000) | Max steps per episode |
| `rl_host` | `str` | `self._rl_host` (from CLI `--rl-host` or `config.RL_POLICY_HOST`) | RL policy server hostname |
| `rl_port` | `int` | `config.RL_POLICY_PORT` (8965) | RL policy server Portal RPC port |
| `control_mode` | `str` | `"both"` | `"left"`, `"right"`, or `"both"` — which arm(s) the RL policy controls |

**Returns**:
```python
{"success": True, "steps_executed": int}
# or on error:
{"success": False, "steps_executed": int, "reason": str}
```

**Step loop logic** (`cap/server/cap_server.py`):
1. Check e-stop
2. Read Fello state from cache (under `_state_lock`)
3. Resolve RL action to joint targets via `_resolve_action_to_joints()`
4. Apply press-anchored incremental Fello takeover if footswitch pressed
5. Enforce safety zone constraints via `_enforce_safety_zone()`
6. Clamp grippers to `[GRIPPER_MIN, GRIPPER_MAX]`
7. Write targets under `_state_lock` — control loop sends at `CONTROL_FREQ_HZ`
8. Rate-limit to `POLICY_FREQ_HZ` (busy-wait loop)
9. During human takeover: skip expensive obs/reward/RPC for `HIL_POLICY_SLOWDOWN - 1` out of every `HIL_POLICY_SLOWDOWN` steps
10. Build next observation via `_build_skill_obs()` + `_build_rl_obs()`
11. Get reward (local fn or reward server)
12. Send transition to rl_policy_server, get next action
13. Emit diagnostics events via `cap.diag.emitter.emit()`
14. Update `_learn_skill_status` dict (consumed by UI via `get_learn_skill_status` RPC)

**Key design decisions**:
- **Single-step actions only** — no chunking. RL needs per-step reward attribution.
- **Multiple action types** supported: `joint_angle`, `delta_joint_angle`, `eef_pose`, `delta_eef_pose` (see Action Types section below)
- **Partial action dicts**: if left/right keys are missing, that arm holds its current commanded position (supports single-arm `control_mode`)
- **Fello takeover is transparent**: human actions are sent back to RL server with `action_source="human"`, server routes them to intervention buffer
- **No separate `learn_skill_rl()` method** — the single `learn_skill()` method serves both RL training and policy evaluation

---

### 2. `rl_policy_server` — Portal RPC Interface

The RL policy server runs on the GPU machine and exposes three Portal RPC endpoints:

**Portal RPC Interface (port 8965)**:
```python
class RLPolicyServer:
    def reset(self, obs: dict) -> dict:
        """Episode start. Returns first action.
        Returns: {"action": dict, "action_type": str}
        """

    def step(self, transition: dict) -> dict:
        """Each control step with transition from previous action.
        Args: {"obs": dict, "action_executed": dict, "action_source": str,
               "reward": float, "done": bool}
        Returns: {"action": dict, "action_type": str}
        """

    def get_stats(self) -> dict:
        """Returns current training stats."""
```

**Dummy server for testing** — `scripts/serve_rl_policy.py`:

The `RandomRLPolicyServer` class returns small random delta joint actions. No GPU needed.

```bash
uv run enpire/env/forge/scripts/serve_rl_policy.py --port 8965 --control-mode left --action-scale 0.02
```

CLI options (`scripts/serve_rl_policy.py`):
| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8965` | Portal RPC port |
| `--control-mode` | `left` | `left`, `right`, or `both` |
| `--action-scale` | `0.02` | Max delta per joint per step (radians) |

**Full RL policy server** — `bc_policy/scripts/pld_lite/serve_rl_policy.py`:

Holds SAC agent (JAX) + base agent (`minimal_policy`), exposes Portal RPC, runs agentlace `TrainerClient`.

**Internal flow per step** (full server):
1. Receive `transition` from CAP
2. Compute `base_action` from base agent using `transition.obs`
3. If `action_source == "rl"`: compute delta = `(action_executed - base_action) / gamma`
4. If `action_source == "human"`: store in intervention data store
5. Build agentlace transition dict and insert into `data_store`
6. Call `client.update()` to sync with learner
7. Sample next action: `delta = agent.sample_actions(obs, base_action)` → `action = delta * gamma + base_action`
8. Return action to CAP

---

### 3. Observation & Action Spaces

#### Observation (robot → rl_policy_server)

Built by `_build_skill_obs()` (`cap/server/cap_server.py`) then `_build_rl_obs()` (`cap/server/cap_server.py`):

```python
{
    # Images: uint8, resized to 256x256x3 (from _RL_IMAGE_SIZE at cap_server.py)
    "top_camera_image": ndarray(256, 256, 3),
    "left_camera_image": ndarray(256, 256, 3),
    "right_camera_image": ndarray(256, 256, 3),

    # Flat proprioception vector: float32
    # - control_mode="both": 14-dim (left_jp[6] + left_gp[1] + right_jp[6] + right_gp[1])
    # - control_mode="left":  7-dim (left_jp[6] + left_gp[1])
    # - control_mode="right": 7-dim (right_jp[6] + right_gp[1])
    "state": ndarray(14),  # or 7 for single-arm
}
```

Camera names are dynamically resolved from `config.CAMERA_NAMES` (`cap/config.py`), which reads from active station profile.

**Proprio key order** matches `remote_deployment.yaml` / `SERLObsWrapper` (`cap/server/cap_server.py`):
```python
_RL_PROPRIO_KEYS = [
    ("left_joint_pos", 6),
    ("left_gripper_pos", 1),
    ("right_joint_pos", 6),
    ("right_gripper_pos", 1),
]
```

The task description is passed in the raw obs as `obs["annotation.task"]` (`cap/server/cap_server.py`) for VLM reward functions.

#### Action Types — `_resolve_action_to_joints()` (`cap/server/cap_server.py`)

The RL policy server returns `{"action": dict, "action_type": str}`. The `action_type` determines how the action dict is interpreted:

| `action_type` | Method | Keys in `action` dict | Description |
|---------------|--------|-----------------------|-------------|
| `joint_angle` | `_resolve_joint_angle` :3410 | `left_joint_pos[6]`, `left_gripper_pos[1]`, `right_joint_pos[6]`, `right_gripper_pos[1]` | Absolute joint targets — write directly |
| `delta_joint_angle` | `_resolve_delta_joint_angle` :3434 | Same keys | Delta from current commanded position |
| `eef_pose` | `_resolve_eef_pose` :3458 | `left_ee_pos[3]`, `left_ee_quat_xyzw[4]`, `left_gripper_pos[1]`, ... | Absolute EE pose — IK from current seed |
| `delta_eef_pose` | `_resolve_delta_eef_pose` :3506 | `left_ee_delta_pos[3]`, `left_ee_delta_euler[3]`, `left_gripper_pos[1]`, ... | Delta EE pose — FK→delta→IK |

All resolvers support **partial action dicts**: if left/right keys are missing, that arm holds its current commanded position (`_cmd_*_jp`). This enables single-arm `control_mode`.

---

### 4. Human-in-the-Loop Takeover Protocol

#### Takeover Gating: `USE_FELLO` vs `ALWAYS_TAKEOVERABLE`

Two independent flags control Fello takeover scope (`cap/config.py`):

| Flag | Default | Effect |
|------|---------|--------|
| `USE_FELLO` | `True` | Enables per-step Fello takeover inside `learn_skill` only. Atomic skills (`freespace_move`, `go_home`, `set_gripper`) are **not** interruptible. |
| `ALWAYS_TAKEOVERABLE` | `False` | Enables control-loop-level Fello takeover for **all** commands (`CONTROL_FREQ_HZ` override). |

**CLI flags** on `cap_server.py` (`cap/server/cap_server.py`):
- `--use-fello` → sets `USE_FELLO=True`, `ALWAYS_TAKEOVERABLE=False`
- `--always-takeoverable` → sets both `USE_FELLO=True` and `ALWAYS_TAKEOVERABLE=True`

On real hardware (no `--env`), both flags use their `config.py` defaults.

**Why decoupled**: RL exploration scripts interleave atomic motions (`freespace_move` for approach/regrasp, `go_home` for recovery) with `learn_skill` RL episodes. The human should only be able to take over during `learn_skill`, not during the scripted motions — accidental footswitch presses during `freespace_move` caused unsafe interference.

#### HIL Step Throttle

During human takeover, joint writes continue at `POLICY_FREQ_HZ` for smooth Fello control (no jitter), but obs building, reward evaluation, and the RL server RPC are skipped for `HIL_POLICY_SLOWDOWN - 1` out of every `HIL_POLICY_SLOWDOWN` steps (`cap/config.py`):

```python
HIL_POLICY_SLOWDOWN: int = 3  # RL step every 3rd control step → ~10 Hz
```

This keeps the RL server at a lower effective rate (`POLICY_FREQ_HZ / HIL_POLICY_SLOWDOWN`) without stalling the control loop. When the human releases the footswitch, `_hil_substep` resets to 0 so the next step immediately sends a full RL transition (`cap/server/cap_server.py`).

#### Takeover Flow

```
                    ┌──────────────────────────────┐
                    │    Fello Button Pressed?     │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼──────────┐
              Yes   │   Use Fello joints   │   No
             ┌──────┤   as action          ├──────┐
             │      └─────────────────────┘      │
             │                                    │
    ┌────────▼─────────┐              ┌──────────▼────────┐
    │ action_source =   │              │ action_source =    │
    │ "human"           │              │ "rl"               │
    │ step @ 10 Hz      │              │ step @ POLICY_FREQ │
    └────────┬─────────┘              └──────────┬────────┘
             │                                    │
             │      ┌───────────────────┐         │
             └──────► rl_policy_server   ◄────────┘
                    │  .step(transition) │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │  Routes to:        │
                    │  human → intvn_buf │
                    │  rl    → online_buf│
                    └───────────────────┘
```

**On takeover onset — press-anchored incremental control** (`cap/server/cap_server.py`):
- **Arm joints**: anchor set from hardware feedback (`robot_ljp` / `robot_rjp`)
- **Gripper**: anchor set from **last commanded position** (`self._cmd_left_gp` / `self._cmd_right_gp`), not hardware feedback. When the gripper is stalled against an object, command and feedback diverge (e.g. command=0.5 close vs feedback=0.65 stalled). Using feedback would relax the grip.
- **Gripper settle period**: gripper stays locked at anchor for `FELLO_GRIP_SETTLE_TICKS` steps, then re-anchors and applies Fello gripper delta only after the deadband (`FELLO_GRIP_TAKEOVER_DEADBAND`) is exceeded.

**On takeover release** (`cap/server/cap_server.py`):
- Hold last human position for 1 control step (smooth transition)
- RL policy server receives `action_source="rl"` again on next step
- Policy continues from current state (no explicit reset — SAC is state-conditioned)

---

### 5. Episode Lifecycle

```
Agent calls learn_skill("task_name", {max_steps: 1000})
    │
    ├─ 1. CAP Server: build initial obs (_build_skill_obs + _build_rl_obs)
    ├─ 2. CAP Server → rl_policy_server.reset(obs)
    ├─ 3. rl_policy_server: reset episode, return first action + action_type
    ├─ 4. Return action to CAP
    │
    ├─ [LOOP at POLICY_FREQ_HZ]  (cap/server/cap_server.py)
    │   ├─ 5. Check e-stop
    │   ├─ 6. Read Fello state from cache
    │   ├─ 7. Resolve RL action → joint targets
    │   ├─ 8. Apply Fello takeover if active
    │   ├─ 9. Enforce safety zone
    │   ├─ 10. Write targets under _state_lock
    │   ├─ 11. Rate-limit to POLICY_FREQ_HZ
    │   ├─ 12. Build next_obs, get reward (if _do_rl_step)
    │   ├─ 13. CAP Server → rl_policy_server.step(transition)
    │   ├─ 14. Update progress bar + _learn_skill_status
    │   └─ 15. Emit diagnostics events
    │
    ├─ [EPISODE END: done=True or max_steps reached]
    │   ├─ 16. Stop progress bar
    │   └─ 17. Return result to agent
    │
    └─ Agent can call learn_skill() again for next episode
```

---

### 6. Reward System

#### Architecture Overview

Reward is obtained per-step inside `learn_skill()`. Two mechanisms are supported, with local functions taking priority over the external server:

1. **Local reward function** (in-process, set via `set_reward_mode` RPC)
2. **External reward server** (Portal RPC on port 8500)

The priority logic is at `cap/server/cap_server.py`:
```python
def _get_step_reward(obs: dict) -> float:
    if _local_fn is not None:        # set via set_reward_mode()
        return float(_local_fn(obs))
    if _reward_client is None:       # no external server available
        return 0.0
    return _reward_client.get_reward(obs)
```

#### `set_reward_mode()` — In-process Reward Switching (`cap/server/cap_server.py`)

Portal RPC endpoint that sets a local reward function, bypassing the external server:

```python
# Set mode (available: constant-0, constant-1, random, insert_usb, gemini, smolvlm)
client.set_reward_mode("gemini").result()

# Clear — fall back to external reward server
client.set_reward_mode("").result()
```

#### Reward Server — `cap/reward/reward_server.py`

The unified reward server supports multiple modes via `--mode`:

```bash
uv run -m enpire.env.forge.cap.reward.reward_server --mode <mode> [--port 8500]
```

**Available modes** (`cap/reward/reward_server.py`):

| Mode | Function | Description | File |
|------|----------|-------------|------|
| `constant-0` | `_constant_zero` | Always returns 0.0 | `cap/reward/reward_server.py` |
| `constant-1` | `_constant_one` | Always returns 1.0 | `cap/reward/reward_server.py` |
| `random` | `_random_reward` | Random {0.0, 1.0} each call | `cap/reward/reward_server.py` |
| `insert_usb` | `insert_usb_reward` | 1.0 if USB drive mounted, else 0.0 | `cap/reward/insert_usb/reward.py` |
| `gemini` | `vlm_reward` | VLM reward via Google Gemini API | `cap/reward/gemini_reward.py` |
| `smolvlm` | `smolvlm_reward` | VLM reward via local SmolVLM vLLM | `cap/reward/smolvlm_reward.py` |

**`RewardServer` class** (`cap/reward/reward_server.py`): Portal RPC server wrapping any `Callable[[dict], float]`. Exposes `get_reward(obs)` and runs a continuous 10 Hz print loop showing the current reward.

**`RewardClient` class** (`cap/reward/reward_client.py`): Portal RPC client that connects to reward server and calls `get_reward(obs)`.

**Alternative entry point** — `cap/reward/serve_reward.py`: Simpler launcher that loads reward functions by dotted import path. Currently only supports `insert_usb`.

#### VLM Reward Functions

Both VLM backends send camera images + task description to a model and parse YES/NO:

**Gemini** (`cap/reward/gemini_reward.py`):
- Requires `GEMINI_API_KEY` env var
- Rate-limited: `GEMINI_REWARD_INTERVAL_S` (default 5.0s)
- Uses prompt template from `cap/prompt/reward_gemini.json`
- Model: `GEMINI_REWARD_MODEL` env var (default `gemini-2.5-flash`)

**SmolVLM** (`cap/reward/smolvlm_reward.py`):
- Runs locally on lab GPU node via vLLM — no API key needed
- Rate-limited: `SMOLVLM_REWARD_INTERVAL_S` (default 0.5s)
- Uses prompt template from `cap/prompt/reward_smolvlm.json`
- Uses vLLM `guided_choice: ["YES", "NO"]` for constrained decoding

**Prompt templates** are JSON files in `cap/prompt/` with schema:
```json
{"description": "...", "backend": "gemini|smolvlm", "template": "...{task}..."}
```
Loaded by `cap/utils/prompt_loader.py`.

---

### 7. Safety Zone Enforcement

See `docs/SAFETY_ZONE_DESIGN.md` for full details.

Safety zones restrict end-effector exploration during RL training. They are set by the LLM agent before launching `learn_skill()` and are enforced every step inside the loop.

**Agent tools** (`cap/agent/tools/safety.py`):

| Tool | RPC | Description |
|------|-----|-------------|
| `SetSafetyZoneTool` :50 | `set_safety_zone(side, keyposes, pos_margin, ori_margin)` | Set per-arm zone as convex hull of 7D keyposes + margins |
| `ClearSafetyZoneTool` :108 | `clear_safety_zone([side])` | Clear one or both arms |
| `GetSafetyZoneTool` :141 | `get_safety_zone()` | Query current zone config |

**Enforcement** (`cap/server/cap_server.py`): `_enforce_safety_zone()` runs FK on proposed joints, checks EE pose against zone, and interpolates back toward current position if outside. Elastic attenuation smooths the boundary before hard clamping.

**Server-side safety module** (`cap/server/safety.py`): Implements convex hull distance computation, elastic/hard boundary enforcement, and orientation constraints.

**Typical RL script pattern:**
```python
# Set safety zone before RL loop
set_safety_zone("left", [hover_pose, insertion_pose], pos_margin=0.08, ori_margin=0.3)

for ep in range(NUM_EPISODES):
    learn_skill("insertion", params={"max_steps": 100, "control_mode": "left"})
    go_home()

clear_safety_zone()
```

---

### 8. Configuration

**`cap/config.py`** — RL-relevant entries:

```python
# Control frequencies (cap/config.py)
CONTROL_FREQ_HZ = 60.0     # Hardware control loop
POLICY_FREQ_HZ = 30.0      # RL policy step rate

# HIL throttle (cap/config.py)
HIL_POLICY_SLOWDOWN: int = 3  # RL step every 3rd step during takeover → ~10 Hz

# Fello (cap/config.py)
USE_FELLO = True
ALWAYS_TAKEOVERABLE = False

# RL policy server (cap/config.py)
RL_POLICY_HOST = "localhost"
RL_POLICY_PORT = 8965
RL_EPISODE_MAX_STEPS = 1000
RL_DATA_PATH = "/media/<user>/Extreme SSD/data/learn_skill_rl"

# Reward server (cap/config.py)
REWARD_SERVER_PORT = 8500

# Camera names (cap/config.py) — resolved from active station profile
CAMERA_NAMES: tuple[str, ...] = ("top", "left", "right")  # default

# Joint limits (cap/config.py) — from station.xml actuator ctrlrange
JOINT_LIMITS_LOW = np.array([...])   # 12-dim (6 per arm)
JOINT_LIMITS_HIGH = np.array([...])

# Gripper range (cap/config.py)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
```

**`cap_server.py` CLI arguments** (`cap/server/cap_server.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--arm-host` | `127.0.0.1` | Arm server host |
| `--port` | `8300` | Portal RPC port |
| `--rl-host` | `config.RL_POLICY_HOST` | RL policy server host |
| `--env` | `None` | Environment: `yam` (MuJoCo sim) or `yam-real` (hardware) |
| `--viewer` | `False` | Launch viewer window (sim envs only) |
| `--use-fello` | `False` (sim only) | Enable Fello HIL in sim |
| `--always-takeoverable` | `False` | Fello takeover for ALL commands |
| `--no-cameras` | `False` | Disable cameras |
| `--no-arms` | `False` | Stub arms (no hardware) |

---

### 9. Data Recording

> **Status note (2026-04-18 audit)**: `_SkillDataRecorder` is currently **not
> instantiated** inside `learn_skill()`. The class still exists (see line below)
> but the call site was removed — `learn_skill()` emits a `record_done` event
> but does not write per-episode data to disk. The section below describes the
> intended design; treat it as design-only until the call site is rewired.

#### `_SkillDataRecorder` — `cap/server/cap_server.py` (class defined; not currently called)

Per-episode recorder. Matches `RecordEpisodeWrapper` format for compatibility with offline training pipelines.

**`record_step()` signature** (`cap/server/cap_server.py`):
```python
def record_step(self, obs: dict, left_jp, left_grip, right_jp, right_grip,
                left_takeover: bool, right_takeover: bool, reward: float = 0.0)
```

**Video encoding**: done on a background thread (`_frame_writer_loop` at `cap/server/cap_server.py`) to keep `cvtColor` + `VideoWriter.write` off the control-loop thread.

**Saved episode directory layout** (`$LEARN_SKILL_DATA_PATH/<skill_name>/YYYYMMDDTHHMMSS######/`):

```
timestamp.npy              # float64 wall-clock timestamps per step
left-joint-pos.npy         # float64 (N, 6) — left arm joint observations
left-gripper-pos.npy       # float64 (N, 1) — left gripper observations
right-joint-pos.npy        # float64 (N, 6)
right-gripper-pos.npy      # float64 (N, 1)
action-left-pos.npy        # float64 (N, 7) — commanded left arm jp[6] + gp[1]
action-right-pos.npy       # float64 (N, 7)
action-source.npy          # str array — "human" or "policy" per step
is-human-action.npy        # float32 (N,) — 1.0 if human, 0.0 if policy
reward.npy                 # float32 (N,) — per-step reward
metadata.json              # task_name, env_loop_frequency, duration, station_metadata
top_camera-images-rgb.mp4  # H.264 video (converted from mp4v after recording)
left_camera-images-rgb.mp4
right_camera-images-rgb.mp4
```

---

### 10. Agent Tools

#### `LearnSkillTool` — `cap/agent/tools/skill.py`

The agent-facing tool that calls `cap_server.learn_skill()` via Portal RPC.

```python
class LearnSkillTool(Tool):
    name = "learn_skill"
    # Calls: client.learn_skill(skill_name, params).result()
```

**Parameters**:
- `skill_name` (str): Name/description of the RL skill
- `params` (dict, optional): Overrides — `rl_host`, `rl_port`, `max_steps`, `control_mode`, `task_description`

#### `ExecuteSkillTool` — `cap/agent/tools/skill.py`

For running learned flow-matching policies (non-RL). Calls `cap_server.execute_skill()`.

#### Safety Zone Tools — `cap/agent/tools/safety.py`

- `SetSafetyZoneTool` :50 — Set per-arm EE safety zone
- `ClearSafetyZoneTool` :108 — Clear zones
- `GetSafetyZoneTool` :141 — Query current config

#### Profiler — `cap/agent/profiler.py`

`learn_skill` calls are wrapped with timing by `wrap_callables_with_timing()` (`cap/agent/profiler.py`). Logs call duration, idle gaps, and post-call robot state snapshots (for state-changing tools listed in `_STATE_SNAPSHOT_TOOLS` at `cap/agent/profiler.py`).

---

### 11. UI Integration

#### LearnSkillPanel — `cap/ui/src/components/LearnSkillPanel.tsx`

A React component that displays live `learn_skill` status. Shows:
- Episode number, step progress with countdown
- Per-step reward, cumulative reward
- Action source badge (RL = blue, Human = orange)
- Expandable detail panel with progress bar

Consumes `LearnSkillStatus` from `useRobotState` hook, which reads from `get_learn_skill_status` RPC (`cap/server/cap_server.py`).

The status dict emitted by `learn_skill()` (`cap/server/cap_server.py`):
```python
self._learn_skill_status = {
    "active": True,
    "episode": int,
    "step": int,
    "max_steps": int,
    "reward": float,
    "cumulative_reward": float,
    "action_source": str,  # "rl" or "human"
}
```

---

## Diagnostics Dashboard

### Architecture — `cap/diag/`

```
 learn_skill loop (POLICY_FREQ_HZ)           rl_policy_server
 ┌──────────────────────┐             ┌──────────────────┐
 │  t0  step_start      │             │                  │
 │  t1  fello_read      │             │                  │
 │  t2  obs_built       │             │                  │
 │  t3  rpc_call_start  │── Portal ──►│ t4  step_recv    │
 │                      │             │ t5  base_action   │
 │                      │             │ t6  sac_sample    │
 │  t7  rpc_call_end  ◄─│── Portal ──│ t7  step_send    │
 │  t8  reward_start    │             └──────────────────┘
 │  t9  reward_end      │
 │  t10 action_applied  │
 │  t11 record_done     │
 │  t12 step_end        │
 └──────┬───────────────┘
        │ UDP (fire-and-forget)
        ▼
 ┌──────────────────────┐
 │  Diagnostics Server  │ port 8888
 │  (dashboard.py)      │
 │                      │
 │  ┌─ UDP Collector ─┐ │
 │  │ :9999           │ │
 │  │ ring buffer     │ │
 │  │ (10K raw /      │ │
 │  │  2K steps)      │ │
 │  └────────┬────────┘ │
 │           │           │
 │  ┌─ Motor Temp ────┐ │
 │  │ Collector       │ │  ← Portal RPC to arm servers
 │  │ (2 Hz polling)  │ │
 │  └────────┬────────┘ │
 │           │           │
 │  ┌────────▼────────┐ │
 │  │ FastAPI + WS    │ │
 │  │ /     (HTML)    │ │  ← browser
 │  │ /ws   (live)    │ │
 │  │ /api/stats      │ │
 │  └─────────────────┘ │
 └──────────────────────┘
```

### UDP Timestamp Emitter — `cap/diag/emitter.py`

Zero-overhead fire-and-forget UDP emitter (~44 lines). `sendto()` is non-blocking (~1us).

```python
from cap.diag.emitter import emit
emit("cap_server", "step_start", ep=3, step=142)
emit("cap_server", "obs_built", ep=3, step=142, camera_read_ms=0.8)
```

**Configuration** (env vars, `cap/diag/emitter.py`):
- `DIAG_ENABLED` — set to `"0"` to disable entirely
- `DIAG_HOST` — target host (default `127.0.0.1`)
- `DIAG_PORT` — target UDP port (default `9999`)

The socket is set to non-blocking (`cap/diag/emitter.py`). All exceptions are silently caught.

**Events emitted by `learn_skill()`**:

| Source | Event | When | Meta |
|--------|-------|------|------|
| cap_server | `step_start` | Top of loop | `max_steps`, `cumR` |
| cap_server | `fello_read` | After Fello state read | `takeover_left`, `takeover_right` |
| cap_server | `obs_built` | After `_build_rl_obs()` | `camera_read_ms` |
| cap_server | `rpc_call_start` | Before `rl_client.step()` | |
| cap_server | `rpc_call_end` | After `.result()` returns | |
| cap_server | `reward_start` | Before reward computation | |
| cap_server | `reward_end` | After reward received | `reward` |
| cap_server | `action_applied` | After writing targets | `action_source` |
| cap_server | `safety_enforced` | When safety zone clips | `side`, pos/ori info |
| cap_server | `record_done` | After recorder.record_step() | |
| cap_server | `step_end` | End of iteration | |
| control_loop | `tick` | Every Nth control tick | `dt_ms` |
| fello_loop | `tick` | Every Fello read | `dt_ms`, `takeover_left/right` |
| camera | `frame` | Every Nth camera frame | `camera_name`, `dt_ms` |
| reward_server | `reward_recv` | RPC handler entry | |
| reward_server | `reward_compute` | After reward fn | `ms`, `reward` |
| reward_server | `reward_send` | Before returning | |
| rl_policy_server | `step_recv` | Handler entry | |
| rl_policy_server | `base_action_done` | After base inference | `ms` |
| rl_policy_server | `sac_sample_done` | After SAC sampling | `ms` |
| rl_policy_server | `buffer_insert_done` | After data_store insert | `ms` |
| rl_policy_server | `step_send` | Before returning | |

### Dashboard Server — `cap/diag/dashboard.py`

Standalone FastAPI server with inline HTML dashboard (~1000 lines, no build step).

**Launch**:
```bash
uv run enpire/env/forge/cap/diag/dashboard.py [--udp-port 9999] [--http-port 8888] [--motor-host localhost]
```

**Components**:

1. **UDP Collector** (`cap/diag/dashboard.py`): Background thread receiving msgpack UDP packets, dispatching to `Store`.

2. **Motor Temp Collector** (`cap/diag/dashboard.py`): Background thread polling arm RPC servers for motor temperatures at configurable rate (default 2 Hz). Polls all four arms: follower_left/right (ports 11333/11334), leader_left/right (ports 11335/11336).

3. **Store** (`cap/diag/dashboard.py`): Thread-safe ring buffer holding raw events (10K), step data (2K), plus aux buffers for control ticks, fello ticks, camera frames, and motor temps.

4. **FastAPI app** (`cap/diag/dashboard.py`):
   - `GET /` → inline HTML+JS dashboard
   - `WS /ws` → pushes snapshot JSON every 200ms
   - `GET /api/stats` → aggregate stats

**Dashboard UI panels** (inline HTML at `cap/diag/dashboard.py`):

| Panel | Description |
|-------|-------------|
| **Health Panel** | Episode, step progress with countdown, reward, cumulative reward, loop Hz, action source, control loop Hz + jitter, Fello Hz, camera FPS per camera, motor temp max + per-joint details |
| **Live Step Timeline** | Horizontal bars per step with colored segments: fello_read, build_obs, rpc_call, reward, apply_action, record, idle. Red border on segments exceeding threshold. |
| **Latency Stats** | Rolling mean/P50/P95/max for all segments + RPC breakdown (base_action, sac_sample, buffer_insert, network_rtt) |
| **Episode Overview** | Bar chart of step loop times, colored by action source (blue=rl, orange=human), red top border if over target |

**Segment definitions** (`cap/diag/dashboard.py`):

| Segment | Start Event | End Event |
|---------|-------------|-----------|
| fello_read | step_start | fello_read |
| build_obs | fello_read | obs_built |
| rpc_call | rpc_call_start | rpc_call_end |
| reward | reward_start | reward_end |
| apply_action | reward_end | action_applied |
| record | action_applied | record_done |
| idle | record_done | step_end |
| total | step_start | step_end |

---

## Step Latency Breakdown

The `learn_skill` progress bar shows per-step timing (`cap/server/cap_server.py`):

```
[learn_skill] ep=0  ██ 42/1000  r=0.00 cumR=0.00 src=rl rl=7ms local=50ms (obs=12 rew=2 fello=0 act=1)
```

| Field | Meaning |
|-------|---------|
| `rl` | Time in `rl_client.step()` — network RTT + policy inference on GPU machine |
| `local` | Everything else minus RPC time (includes rate-limiting sleep to maintain `POLICY_FREQ_HZ`) |
| `obs` | `_build_skill_obs()` + `_build_rl_obs()` — camera grab + resize + proprio |
| `rew` | Reward computation (local fn or reward server RPC) |
| `fello` | Reading Fello state from cache (under `_state_lock`) |
| `act` | Action resolve + takeover computation + safety zone + gripper clamp + write targets |

`local` includes the ~33ms rate-limiting sleep (`POLICY_FREQ_HZ`), so `obs + rew + fello + act` won't sum to `local` — the gap is sleep + emit overhead.

---

## Script Helper Structure

RL training scripts live in `cap/saved_scripts/rl/` (not shipped in this
release). They are executed inside `cap_agent` in "Oracle mode" with access
to all registered tools.

**Common patterns across scripts**:
- Scene setup: `clear_table()` → `setup_scene("stick_plate")` (sim only)
- Object detection: `detect_object(name, camera, backend)`
- Approach: `freespace_move(side, pos, quat, max_vel)`
- Safety zone: `set_safety_zone(side, keyposes, pos_margin, ori_margin)`
- RL loop: `learn_skill(name, params={max_steps, control_mode, rl_host})`
- Cleanup: `clear_safety_zone()` → `go_home()`

**Example scripts**:

| Script | Description |
|--------|-------------|
| `test_learn_skill_with_dummy_rl_reward.py` | Smoke test: 3 episodes of 50 steps with dummy RL + constant-0 reward |
| `learn_insert_usb.py` | USB insertion with `insert_usb` reward, prepositioning both arms |
| `claude_plate_and_stick_learn_insert_safe.py` | Stick insertion with safety zones: scene setup, grasp, regrasp, safety zone, RL loop |
| `claude_plate_and_stick_learn_insert_vlm_regrasp.py` | Same + VLM-based regrasp validation |
| `sim_peg_insertion_rl.py` | Sim-mode peg insertion RL |

**Motion helpers** (in complex scripts like `claude_learn_peg_insertion.py`):
- `place_stick_on_table()` — releases stick away from plate
- `place_block_on_table()` — places block back on table
- `reorient_stick_to_stand()` — picks lying stick, places upright, re-detects
- `vertical_regrasp_in_place()` — face-aligned top-down regrasp
- `lift_after_grasp()` — post-grasp lift
- `move_to_hover_on_plate()` — detects plate, moves to insertion hover
- `rim_pinch_plate()` — helper arm grasps plate rim

---

## Launch Commands

### Test Mode (dummy RL, no GPU needed)

```bash
# Terminal 1: CAP Server (CONTROL_FREQ_HZ control loop)
uv run enpire/env/forge/cap/server/cap_server.py [--env yam]

# Terminal 2: Reward server (constant-0 or constant-1)
uv run -m enpire.env.forge.cap.reward.reward_server --mode constant-0

# Terminal 3: Dummy RL policy server (random delta actions)
uv run enpire/env/forge/scripts/serve_rl_policy.py --port 8965 --control-mode both

# Terminal 4: (Optional) Diagnostics dashboard
uv run enpire/env/forge/cap/diag/dashboard.py
```

Then trigger an RL episode from the CAP agent or via Portal RPC:

```python
import portal
client = portal.Client("localhost:8300")
result = client.learn_skill("test_hold", {"max_steps": 300}).result()
print(result)
```

Or use the test script:
```bash
# Run in Oracle mode in CAP UI (http://localhost:5173)
# Script: cap/saved_scripts/rl/test_learn_skill_with_dummy_rl_reward.py
```

### Full RL Training

```bash
# Robot machine (enpire/)
# Terminal 1: CAP Server
uv run enpire/env/forge/cap/server/cap_server.py

# Terminal 2: Reward server (placeholder or real)
uv run -m enpire.env.forge.cap.reward.reward_server --mode insert_usb

# GPU machine (bc_policy/)
# Terminal 3: RL Learner (trains SAC from replay buffer)
uv run scripts/pld_lite/train_delta_rlpd.py \
    --config-dir ../../configs --config-name config \
    train=delta_rlpd mode=learner

# Terminal 4: RL Policy Server (actor — connects to learner + CAP)
uv run scripts/pld_lite/serve_rl_policy.py \
    --config-dir ../../configs --config-name config \
    train=delta_rlpd

# Robot machine (enpire/)
# Terminal 5: Run episodes (loop or from agent)
import portal
client = portal.Client("localhost:8300")
for ep in range(100):
    result = client.learn_skill("cutter_handover", {
        "max_steps": 1000,
        "rl_host": "192.168.0.XXX",
        "rl_port": 8965,
    }).result()
    print(f"Episode {ep}: {result}")
    client.go_home().result()
```

### HIL Sim Evaluation (Fello + MuJoCo)

Run RL evaluation in simulation with real Fello arms for human-in-the-loop takeover.

**Key flags**:
- `cap_server.py --env yam --use-fello` — sim mode with Fello HIL for `learn_skill` only
- `cap_server.py --env yam --always-takeoverable` — sim mode with Fello for **all** commands
- `launch.py --mode=evaluation --fello-only` — only Fello leader servers (sim handles followers)

**Option A — Single command:**
```bash
# NOTE: tmux/launch_sim.sh is not part of this release
```

**Option B — Separate Fello servers:**
```bash
# Session 1: Fello leader servers only
uv run launch.py --mode=evaluation --fello-only

# Session 2: Sim stack
uv run enpire/env/forge/cap/server/cap_server.py --env yam --use-fello
```

> **Do not** combine `launch.py --fello-only` with `launch_sim.sh --use-fello` — both would try to launch Fello servers on the same ports.

**Option C — Manual:**
```bash
# Terminal 1: CAP Server (sim + Fello HIL)
uv run enpire/env/forge/cap/server/cap_server.py --env yam --use-fello

# Terminal 2: Fello left leader
uv run enpire/env/forge/robot/fello/fello_server.py --side left --can-interface can_leader_l --port 11335

# Terminal 3: Fello right leader
uv run enpire/env/forge/robot/fello/fello_server.py --side right --can-interface can_leader_r --port 11336

# Terminal 4: Reward server
uv run -m enpire.env.forge.cap.reward.reward_server --mode constant-0

# Terminal 5: RL Policy Server (GPU machine or local)
uv run enpire/env/forge/scripts/serve_rl_policy.py --port 8965 --control-mode both

# Terminal 6: Run evaluation episodes
import portal
client = portal.Client("localhost:8300")
for ep in range(50):
    result = client.learn_skill("eval_task", {
        "max_steps": 1000,
    }).result()
    print(f"Episode {ep}: {result}")
    client.go_home().result()
```

### Port Summary

| Service | Port | Machine | File |
|---------|------|---------|------|
| CAP Server | 8300 | Robot | `cap/config.py` |
| Reward Server | 8500 | Robot | `cap/config.py` |
| RL Policy Server | 8965 | GPU (or local for testing) | `cap/config.py` |
| Agentlace (data) | 8001 | GPU | `bc_policy/` config |
| Agentlace (params) | 8002 | GPU | `bc_policy/` config |
| Diagnostics Dashboard | 8888 | Robot | `cap/diag/dashboard.py` |
| Diagnostics UDP | 9999 | Robot | `cap/diag/emitter.py` |
| CAP Agent | 8200 | Robot | `cap/config.py` |
| Policy Server (flow matching) | 8964 | GPU | `cap/config.py` |
| Fello Left Leader | 11335 | Robot | `cap/config.py` |
| Fello Right Leader | 11336 | Robot | `cap/config.py` |
| Left Follower | 11333 | Robot | `cap/config.py` |
| Right Follower | 11334 | Robot | `cap/config.py` |

---

## File Reference

### Core RL Loop

| File | Line | What |
|------|------|------|
| `cap/server/cap_server.py` | :2776 | `learn_skill()` — main RL loop |
| `cap/server/cap_server.py` | :3392 | `_resolve_action_to_joints()` — action type dispatch |
| `cap/server/cap_server.py` | :3410 | `_resolve_joint_angle()` — absolute joint targets |
| `cap/server/cap_server.py` | :3434 | `_resolve_delta_joint_angle()` — delta from current |
| `cap/server/cap_server.py` | :3458 | `_resolve_eef_pose()` — absolute EE pose → IK |
| `cap/server/cap_server.py` | :3506 | `_resolve_delta_eef_pose()` — delta EE → FK+IK |
| `cap/server/cap_server.py` | :3599 | `_build_rl_obs()` — build RL observation dict |
| `cap/server/cap_server.py` | :3629 | `_build_skill_obs()` — build raw observation dict |
| `cap/server/cap_server.py` | :427 | `_SkillDataRecorder` — episode data recording |
| `cap/server/cap_server.py` | :3345 | `get_learn_skill_status()` — live status for UI |
| `cap/server/cap_server.py` | :3349 | `set_reward_mode()` — in-process reward switching |
| `cap/server/cap_server.py` | :3701 | `_enforce_safety_zone()` — per-step zone enforcement |

### Reward System

| File | Line | What |
|------|------|------|
| `cap/reward/reward_server.py` | :36 | `RewardServer` — Portal RPC wrapper for reward fn |
| `cap/reward/reward_server.py` | :91 | Built-in modes: constant-0, constant-1, random |
| `cap/reward/reward_server.py` | :103 | `MODES` dict — all reward modes |
| `cap/reward/reward_server.py` | :111 | `_get_vlm_modes()` — lazy VLM import |
| `cap/reward/reward_client.py` | :10 | `RewardClient` — Portal RPC client |
| `cap/reward/serve_reward.py` | :1 | Alternative entry point (dotted import) |
| `cap/reward/insert_usb/reward.py` | :13 | `insert_usb_reward()` — USB mount detection |
| `cap/reward/gemini_reward.py` | :69 | `vlm_reward()` — Gemini VLM reward |
| `cap/reward/smolvlm_reward.py` | :89 | `smolvlm_reward()` — SmolVLM reward |
| `cap/utils/prompt_loader.py` | :25 | `load_reward_prompt()` — load JSON prompt templates |
| `cap/prompt/reward_gemini.json` | — | Gemini reward prompt template |
| `cap/prompt/reward_smolvlm.json` | — | SmolVLM reward prompt template |

### Safety Zones

| File | Line | What |
|------|------|------|
| `cap/server/safety.py` | :1 | Safety module: e-stop, convex hull zones, enforcement |
| `cap/agent/tools/safety.py` | :50 | `SetSafetyZoneTool` |
| `cap/agent/tools/safety.py` | :108 | `ClearSafetyZoneTool` |
| `cap/agent/tools/safety.py` | :141 | `GetSafetyZoneTool` |

### Agent Tools

| File | Line | What |
|------|------|------|
| `cap/agent/tools/skill.py` | :73 | `LearnSkillTool` — agent-facing learn_skill |
| `cap/agent/tools/skill.py` | :22 | `ExecuteSkillTool` — policy execution |
| `cap/agent/profiler.py` | :162 | `wrap_callables_with_timing()` — tool call profiling |

### Diagnostics

| File | Line | What |
|------|------|------|
| `cap/diag/emitter.py` | :31 | `emit()` — UDP diagnostics emitter |
| `cap/diag/dashboard.py` | :1 | Standalone diagnostics dashboard server |
| `cap/diag/dashboard.py` | :84 | `Store` — ring buffer + step index |
| `cap/diag/dashboard.py` | :374 | `udp_collector()` — UDP receiver thread |
| `cap/diag/dashboard.py` | :407 | `motor_temp_collector()` — arm temp polling |
| `cap/diag/dashboard.py` | :476 | `DASHBOARD_HTML` — inline HTML+JS dashboard |

### RL Policy Servers

| File | Line | What |
|------|------|------|
| `scripts/serve_rl_policy.py` | :20 | `RandomRLPolicyServer` — dummy delta server for testing |

### UI

| File | What |
|------|------|
| `cap/ui/src/components/LearnSkillPanel.tsx` | Live RL status panel in React UI |
| `cap/ui/src/hooks/useRobotState.ts` | `LearnSkillStatus` type, WebSocket consumer |

### Configuration

| File | Line | What |
|------|------|------|
| `cap/config.py` | :107 | `CONTROL_FREQ_HZ`, `POLICY_FREQ_HZ` |
| `cap/config.py` | :117 | `HIL_POLICY_SLOWDOWN` |
| `cap/config.py` | :126 | `USE_FELLO`, `ALWAYS_TAKEOVERABLE` |
| `cap/config.py` | :149 | `RL_POLICY_HOST/PORT`, `RL_EPISODE_MAX_STEPS`, `RL_DATA_PATH` |
| `cap/config.py` | :98 | `REWARD_SERVER_PORT` |
| `cap/config.py` | :161 | `JOINT_LIMITS_LOW/HIGH` |
| `cap/config.py` | :173 | `GRIPPER_MIN/MAX` |

### Example RL Scripts

The `cap/saved_scripts/rl/` examples (USB insertion, plate-and-stick, peg
insertion, and the dummy-reward smoke test) are **not** part of this release.
The protocol and helper structure documented above are what you need to write
your own; `cap/saved_scripts/examples/pick_object.py` is the closest shipped
reference for the script conventions.

## Resolved Decisions

1. **Image compression** — No compression. Send raw over LAN to avoid latency. Images resized to 256x256 in `_build_rl_obs()`.
2. **Base agent** — `minimal_policy` (DIT flow matching + DINOv3 features). Trained via `bc_train.py` with `agent=dit_flow_256`, `agent/features=dino_v3`, `ac_chunk=50`.
3. **Bimanual vs single-arm** — Support both via `control_mode` parameter. Partial action dicts hold missing arm in place.
4. **Reward** — Dual mechanism: local in-process function (via `set_reward_mode`) or external Portal RPC server (port 8500). VLM backends (Gemini, SmolVLM) supported for vision-based task completion judgment.
5. **Go-home** — Agent script decides, not `learn_skill()`.
6. **Warmup** — `train_delta_rlpd` actor decides whether warmup episodes use pure base actions.
7. **HIL in sim** — `--env yam --use-fello` keeps Fello enabled in sim mode. `launch.py --fello-only` launches only Fello servers (no YAM hardware). MuJoCo handles followers; Fello handles human takeover.
8. **Action types** — Four types supported: joint_angle (absolute), delta_joint_angle, eef_pose (IK), delta_eef_pose (FK+delta+IK). Default from dummy server is `delta_joint_angle`.
9. **No separate `learn_skill_rl()`** — Single `learn_skill()` method handles all RL use cases. The name `learn_skill_rl` in early design was consolidated into `learn_skill`.
10. **Placeholder server** — No separate `placeholder_reward_server.py`. Built-in modes (constant-0, constant-1, random) are in `reward_server.py`.
