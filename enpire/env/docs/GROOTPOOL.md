# GR00T Pool — multi-script / multi-worker policy middleware

**Status**: v2 — N1.5 + N1.6 backend classes, single-node, sticky-per-episode,
reached from CAP via the `use_policy_output` tool.

Lets *n* CAP agent scripts share *m* GR00T worker subprocesses. Each script
opens a session against the pool; the session sticks to one worker until close.
Routing uses ZMQ `ROUTER`/`DEALER` identity frames — no HTTP cookies, no IP
tracking.

**What's different from v1**: CAP scripts no longer open grootpool sessions by
hand. The `use_policy_output` tool (`cap/env/robocasa/skills.py`) resolves a
backend class from `POLICY_MODEL_CONFIGS`, wraps `RoboCasaEnv` in a gym view
that shares the same robosuite sim (no duplicate env), and delegates to
`cap.policy.inference.run_episode`. Agent-generated scripts are one-liners:
`use_policy_output(model="grootpool/n15", max_episode_steps=500, …)`.

## Quick start (OSMO, real GR00T)

Three commands, run on the OSMO node (`<user>@<osmo-node>`), forge already rsynced to `/mnt/shared/<user>/forge/`.

```bash
cd /mnt/shared/<user>/forge

# 1) one-time GR00T env setup on this Lustre mount (idempotent; ~20-30 min first run, instant after)
bash scripts/setup_grootpool.sh

# 2) launch M GR00T workers + middleware in tmux (returns immediately; workers
#    start loading in the background)
bash tmux/launch_grootpool.sh 3 0,1,2
#                             M  GPU_IDS

# 3) wait until all 3 workers finish loading (~2-3 min each, loaded in parallel)
bash scripts/wait_for_grootpool.sh

# 4) one RoboCasa365 episode via the pool
uv run python scripts/run_grootpool_eval.py \
  --env-name robocasa/PickPlaceCounterToCabinet \
  --task-description "pick the object from the counter and place it in the cabinet" \
  --n-episodes 5 --max-steps 300
```

That's the whole happy path. Everything below is reference / drill-down.



## What each command does

### 1. `scripts/setup_grootpool.sh` — one-time env setup

Idempotent. Steps (skipping anything already done):

1. Clones `robocasa-benchmark/Isaac-GR00T` → `Isaac-GR00T-benchmark/` (~1 min).
2. Builds `Isaac-GR00T-benchmark/model_server_venv/` with **uv** + **Python
   3.11** — torch cu128, flash-attn cu128, gr00t, diffusers, transformers,
   pyzmq, msgpack, fastapi, uvicorn (~5–10 min). Python 3.11 (not 3.12) because
   gr00t's `onnx==1.15.0` transitive dep has no cp312 prebuilt wheel, and the
   source build fails on Ubuntu 24.04's broken `libprotobuf-dev` (missing
   `libutf8_validity.a`).
3. Downloads N1.5 `checkpoint-120000` → `PretrainedModels/gr00t_n1-5/` (~12 GB,
   10–20 min at typical HF bandwidth).
4. Writes `.env.grootpool` at the forge root for the tmux launcher.

All artifacts land on Lustre, so this runs **once per shared filesystem** — any other
OSMO node on the same mount re-runs the script in <1 s (every step no-ops).

Prereq: `uv` on the target node (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`).

Override paths with env vars before running: `OSMO_ROOT`, `GROOT_BENCHMARK_ROOT`, `MODEL_DIR`, `HF_HOME`.

### 2. `tmux/launch_grootpool.sh M GPU_IDS [BASE_PORT]` — bring up the pool

Creates tmux session `grootpool` with **M+2 windows**:

| Window | Runs | Purpose |
|--------|------|---------|
| `overview` | `watch curl .../status` | Live pool status (2 s refresh) |
| `worker-0..M-1` | `model_server_venv/bin/python .../_servers/n15.py --port 5555+i` with `CUDA_VISIBLE_DEVICES=gpu_i` | One GR00T worker per window — attach to inspect / restart |
| `middleware` | `.venv/bin/python -m cap.policy.grootpool.server --external-workers 127.0.0.1:5555,5556,...` | ROUTER (port 7070) + admin (7071) |

Worker logs also mirror to `logs/grootpool/worker_N_gpuM.log`.

Attach: `tmux attach -t grootpool`. Switch: `Ctrl-b <window-num>`. Detach: `Ctrl-b d`.

### 3. `scripts/run_grootpool_eval.py` — benchmark-style multi-episode eval

Standalone runner (outside the CAP tool boundary). Opens a RoboCasa365 gym env
directly, wraps a grootpool session as a `PolicyBackend`
(`cap/policy/grootpool/backend.py`), runs `cap.policy.inference_policy` the
same way the 365 benchmark does. Writes `summary.json` with success rate. Add
`--record-video` to save MP4s to `logs/grootpool_eval/videos/`.

Use this for large sweeps (many tasks × many seeds) where picklability for
`AsyncVectorEnv` matters. For a single experiment inside the CAP agent
pipeline, use the `use_policy_output` path below.

## CAP integration — `use_policy_output`

Grootpool is wired into the CAP tool system. LLM-generated scripts call one
tool; the tool handles backend resolution, controller matching, obs key
layout, session lifecycle, and env/gym dispatch.

### The script

`cap/saved_scripts/gr00t_oracle.py` is the reference — it represents what a
coding agent writes for a pick-and-place task:

```python
result = use_policy_output(
    model="grootpool/n15",
    max_episode_steps=500,
    action_horizon=16,
    replan_horizon=16,
)
print(f"success={result['success']}  steps={result['steps']}")
```

Only model + budget are hardcoded. Task prompt, seed, and endpoint come from
the env / config implicitly — see *Resolution order* below.

### Flow

```
run_script.py (or run_agent.py direct mode)
   │  builds RoboCasaEnv + skill namespace via create_runtime()
   ▼
script calls use_policy_output(model="grootpool/n15", …)
   │
   │  cap/env/robocasa/skills.py:use_policy_output
   ├── resolve backend class   POLICY_MODEL_CONFIGS["grootpool/n15"] → GrootpoolN15Backend
   ├── resolve task prompt     env.get_task_description() (robocasa get_ep_meta()["lang"])
   ├── resolve seed            cfg.env.seed
   ├── resolve endpoint        cfg.policy.endpoint (or backend default)
   ├── controller assert       backend.CONTROLLER_TYPE == "osc_pose" (env is pinned)
   ▼
GrootpoolN15Backend(task_description=…, endpoint=…) → GrootPoolClient.session()
   │
   ▼
ChunkingPolicy(backend, ChunkingConfig(action_horizon, replan_horizon, …))
   │
   ▼
cap.policy.inference.run_episode(env.as_gym_env(), policy, cfg, seed)
   │  gym view = _SharedRoboCasaGymView wrapping the same robosuite env
   │  (CAP skill tools + policy rollout share one sim, no duplicate env)
   ▼
loop: gym_view.step(action_dict)
   │  ↳ robocasa wrapper's unmap_action → robosuite action array
   │  ↳ robosuite env.step → advance sim
   │  ↳ _SharedRoboCasaGymView.get_basic_observation injects camera frames
   │     via RoboCasaEnv.render_rgb_by_mujoco_name (shared render thread)
   ▼
SkillResult(success, steps, task_description, model)
```

### Resolution order — what the tool pulls from where

| Parameter | Source | Notes |
|-----------|--------|-------|
| `model` / `backend` | script kwarg (agent's choice) | Format: `"grootpool/n15"`. Cfg fallback exists via `cfg.policy.{backend,model}` but the agent normally writes this inline. |
| `max_episode_steps`, `action_horizon`, `replan_horizon` | script kwarg | Rollout budget / chunking are task-dependent agent decisions. |
| `task_description` | `env.get_task_description()` → robocasa `get_ep_meta()["lang"]` | **Env is authoritative.** Cached at reset (robocasa re-samples paraphrases per `get_ep_meta()` call when `use_novel_instructions=True`). |
| `endpoint` | `cfg.policy.endpoint` | Infrastructure — not an agent decision. Defaults to the backend class's endpoint (`tcp://127.0.0.1:7070`). |
| `seed` | `cfg.env.seed` | Same seed the env was built with; keeps the fresh `gym.reset(seed)` deterministic. |
| Video recording | `ScriptRecorder` (session-level, `cfg.recording.enabled` + `execution.record` for subprocess mode) | **Not the tool's job.** Tool does not record its own videos. |

### Controller pinning

`RoboCasaEnv` is pinned to `OSC_POSE` — there is no controller-switch
mechanism. `use_policy_output` statically asserts
`backend_cls.CONTROLLER_TYPE == "osc_pose"` and raises otherwise
(`cap/env/robocasa/skills.py:1439-1451`). The old
`ensure_controller_type()` rebuild path was removed; the `controller_type`
kwarg is silently dropped by `cap/env/__init__.py:85-87`.

Joint-space skills (`freespace_move` via cuRobo) coexist with OSC rollouts
by FK'ing each planned waypoint to an EE target and driving the OSC
controller tick-by-tick (see `_execute_osc_trajectory` in `skills.py`).
No env rebuild, no re-seed, no cache invalidation.

### Gym view — sharing one robosuite env

`RoboCasaEnv.as_gym_env()` returns `_SharedRoboCasaGymView`, a subclass of
robocasa's upstream `RoboCasaGymEnv` that:

1. Skips `create_env` + `env.reset` in the parent `__init__` — injects the
   existing `RoboCasaEnv._env` directly.
2. Overrides `get_basic_observation` to inject camera frames from CAP's
   thread-safe `mujoco.Renderer` (robosuite env is built with
   `use_camera_obs=False` to avoid EGL thread-affinity bugs).
3. Skips the upstream image flip (mujoco.Renderer returns upright frames
   already; robosuite's OpenGL path returns upside-down and the wrapper flips
   to compensate).

Result: skill tools (`freespace_move`, `detect_object`, `vlm_query`) and the
policy rollout see the same `self._env.sim` state. No divergence between two
parallel sims.

### Obs-spec metadata on the backend class

`cap/policy/backends/grootpool_backend.py` attaches wire-contract metadata as
class attributes. This is data, not config — changing it makes inference wrong,
not different, so it lives on the checkpoint-adapter class rather than Hydra:

```python
class GrootpoolN15Backend(GrootpoolBackend):
    MODEL_VERSION     = "n15"
    LANGUAGE_KEY      = "annotation.human.task_description"
    CONTROLLER_TYPE   = "osc_pose"
    CAMERA_MAP        = {"video.robot0_agentview_left": "side_left", …}
```

`use_policy_output` reads `CONTROLLER_TYPE` to assert it matches the env's
pinned `osc_pose`. `LANGUAGE_KEY` and `CAMERA_MAP` are reserved for the N1.6
path / follow-ups that need explicit obs-key remap or separate video-recorder
camera lists.

### Hydra `PolicyConfig`

```yaml
# experiments/experiment/<task>.yaml
env:
  name: "robocasa:PickPlaceSinkToCounter"
  controller_type: osc_pose          # GR00T N1.5 trains on OSC
  camera_height: 256                 # GR00T trains at 256×256
  camera_width: 256
  seed: 42

policy:
  backend: grootpool
  model: n15
  endpoint: "tcp://127.0.0.1:7070"   # where the middleware listens

recording:
  enabled: true                      # ScriptRecorder owns video

execution:
  record: true                       # propagates recording to subprocesses
                                     # when execution.mode=parallel
```

`PolicyConfig` (`cap/agent/agent_config.py`) intentionally carries *only*
infrastructure — no `task_description` (env-owned), no `record_video`
(session-owned), no rollout budget (agent-owned).

### Running it

Direct (single episode, single env, single seed):

```bash
uv run python run_script.py \
  experiment=pick_place_sink_to_counter \
  script_file=cap/saved_scripts/gr00t_oracle.py
```

Via the agent pipeline with multiple seeds in parallel:

```bash
uv run python run_agent.py experiment=pick_place_sink_to_counter \
  oracle=cap/saved_scripts/gr00t_oracle.py \
  execution.n_seeds=10 execution.mode=parallel \
  execution.record=true
```

`SubprocessExecutorStep` spawns one `run_script.py` per seed, propagating
`env.{layout_id,style_id,controller_type,camera_height,camera_width}` and
`recording.enabled=true` (when `execution.record` is set). Per-seed logs land
under `logs/<run>/iterations/iter_000/seeds/seed_<NNN>/`.

## Architecture

```
 CAP script A ─┐
   use_policy_output() ─► GrootpoolN15Backend ─► GrootPoolClient ─► DEALER ─┐
 CAP script B ─┤                                                            │
   use_policy_output() ─► GrootpoolN15Backend ─► GrootPoolClient ─► DEALER ─┼──► ROUTER
 CAP script C ─┤                                                            │    asyncio
   use_policy_output() ─► GrootpoolN15Backend ─► GrootPoolClient ─► DEALER ─┘   (grootpool.server)
                                                                                 │
                                                                                 ├─ REQ ─ worker 0 (gpu 0)
                                                                                 ├─ REQ ─ worker 1 (gpu 1)
                                                                                 └─ REQ ─ worker 2 (gpu 2)
                                                                                 │
                                                                       FastAPI /status (7071)
```

- **Sticky**: session_id → worker_id is fixed until `close()`.
- **FIFO wait**: if all workers busy, `open()` queues up to `--open-timeout`.
- **External workers** (default for tmux launcher): middleware only connects to
  pre-running workers; worker lifecycle is owned by tmux (not the middleware).

## Runtime data flow

What happens during `scripts/run_grootpool_eval.py --n-episodes N` with M workers:

**Client state (one per script).** `GrootpoolBackend` opens **one ZMQ DEALER**
to the middleware (port 7070) and reuses it across every episode. The socket is
cheap; the per-episode unit is a **session**.

**Middleware state** (`SessionManager` in `session_manager.py`):

```
idle_workers : asyncio.Queue[int]              # e.g. [0, 1, 2] at startup
sessions     : dict[session_id → SessionInfo]  # worker_id, dealer_identity, last_activity
```

**Per-episode lifecycle:**

| Step | Call | Middleware action | Pool state |
|------|------|-------------------|-----------|
| 1. episode start | `policy.reset()` → `backend.open()` → `{op: open}` | `worker_id = idle_workers.get()` (pop front, FIFO). Record `sessions[uuid]=SessionInfo(worker_id, dealer_identity)`. Reply `{session_id, worker_id}`. | `idle=[1,2]`, `sessions=[{worker:0}]` |
| 2. env step loop | every `replan_horizon` steps: `{op: step, session_id, obs}` | Look up `sessions[session_id]` → forward obs to `worker.predict` (sync ZMQ REQ to :5555 in an executor thread). Reply with the action chunk. | unchanged |
| 3. episode end | `backend.reset()` → `{op: close, session_id}` | Pop `sessions[session_id]`. If worker is alive, push back onto `idle_workers`. | `idle=[1,2,0]`, `sessions=[]` |

At step 1 of **episode 2**, the FIFO pops the next idle worker (1), so episodes
naturally round-robin through the pool: ep1→w0, ep2→w1, ep3→w2, ep4→w0, …
Within any one episode all step calls hit the same worker (sticky).

**How the action gets back to the right client.** The middleware never tracks
client IPs. ZMQ `ROUTER` prepends the sender DEALER's identity frame on every
inbound message; the middleware stores that identity in `SessionInfo` and
prepends it on every reply. N concurrent scripts each have their own DEALER
identity, and replies route automatically.

**Single-script eval ≠ parallelism.** `run_grootpool_eval.py` is single-threaded,
one episode at a time — with 3 workers, 2 are idle at any moment. To exercise
parallelism, run the script N times from N shells (or wrap it with
`AsyncVectorEnv`); each CAP process opens its own session and the middleware
assigns them distinct workers until `N > M`, at which point `open()` FIFO-waits
in `idle_workers.get()` until someone closes.

**Failure handling inside `step`.**

| Symptom | Middleware action | Worker outcome |
|---------|-------------------|----------------|
| worker returned `{error: "gr00t server error: …"}` (app-level, alive) | pop session; push worker back to idle | serves next session |
| `asyncio.TimeoutError` on `step_timeout` (worker hung) | pop session; **don't** return worker | supervisor respawns managed workers; external workers must be restarted in their tmux window |
| CAP script dies without `close` | idle-timeout watchdog auto-closes after `--session-idle-timeout` | returned to idle |

### System design

Editable figure: [`docs/figures/grootpool_system.excalidraw`](figures/grootpool_system.excalidraw) — open at [excalidraw.com](https://excalidraw.com) (drag-drop) or via the Excalidraw VS Code extension.

```
   ┌─────────────── CAP SCRIPTS (1..N, same node or different) ───────────────┐
   │                                                                          │
   │  ┌────────────┐     ┌────────────┐     ┌────────────┐                    │
   │  │ script A   │     │ script B   │     │ script C   │                    │
   │  │ GrootPool  │     │ GrootPool  │     │ GrootPool  │  (one DEALER       │
   │  │ Client     │     │ Client     │     │ Client     │   per thread,      │
   │  │  DEALER    │     │  DEALER    │     │  DEALER    │   reused across    │
   │  │  identity  │     │  identity  │     │  identity  │   episodes)        │
   │  │   = X_A    │     │   = X_B    │     │   = X_C    │                    │
   │  └─────┬──────┘     └─────┬──────┘     └─────┬──────┘                    │
   │        │                  │                  │                           │
   │        └──────────┬───────┴──────────┬───────┘                           │
   │                   │  open / step /   │                                   │
   │                   │  close  (msgpack + ndarray)                          │
   └───────────────────┼──────────────────┼───────────────────────────────────┘
                       ▼                  ▼
   ┌───────────────────────────────── MIDDLEWARE  (cap.policy.grootpool.server) ───┐
   │                                                                               │
   │    ROUTER  tcp://0.0.0.0:7070   ── recv prepends sender DEALER identity ──    │
   │       ▲         │                                                             │
   │       │         ▼                                                             │
   │       │   asyncio _handle_request  ── per-frame op dispatch ──                │
   │       │         │                                                             │
   │       │         ▼                                                             │
   │       │   ┌────────────────────── SessionManager ──────────────────────┐      │
   │       │   │                                                            │      │
   │       │   │   sessions: { session_id → (worker_id, dealer_id,          │      │
   │       │   │                             opened_at, last_activity,      │      │
   │       │   │                             n_calls, latencies) }          │      │
   │       │   │                                                            │      │
   │       │   │   idle_workers: asyncio.Queue[int]   (FIFO, e.g. [1, 2])   │      │
   │       │   │                                                            │      │
   │       │   │   step_latencies: deque(maxlen=1024)   (for /status)       │      │
   │       │   │                                                            │      │
   │       │   │   watchdog: closes sessions idle > 60s                     │      │
   │       │   └──────────────────────────┬─────────────────────────────────┘      │
   │       │                              │  forward step to assigned worker       │
   │       │                              │  via loop.run_in_executor(…)           │
   │       │                              ▼                                        │
   │       │   ┌────────────────────── WorkerSupervisor ────────────────────┐      │
   │       │   │   WorkerHandle[0]    WorkerHandle[1]    WorkerHandle[2]    │      │
   │       │   │     sync REQ            sync REQ           sync REQ        │      │
   │       │   │     torch.save/         torch.save/        torch.save/     │      │
   │       │   │     torch.load          torch.load         torch.load      │      │
   │       │   │      :5555               :5556              :5557          │      │
   │       │   │   ┌──────┐             ┌──────┐           ┌──────┐         │      │
   │       │   └───┤ lock ├─────────────┤ lock ├───────────┤ lock ├─────────┘      │
   │       │       └──┬───┘             └──┬───┘           └──┬───┘                │
   │       │          │                    │                  │                    │
   │       │  reply to identity X_A (from SessionInfo)                             │
   │       └────── ROUTER send_multipart([X_A, payload]) ──────                    │
   │                                                                               │
   │     ──────────── admin ──── FastAPI  GET  /status  (port 7071) ──────────     │
   │     ──────────── admin ──── FastAPI  POST /kill-session/{id} ────────────     │
   └───────────────────────────────────────────────────────────────────────────────┘
                          │                   │                    │
                          ▼                   ▼                    ▼
    ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
    │  GR00T WORKER  (subproc) │  │  GR00T WORKER  (subproc) │  │  GR00T WORKER  (subproc) │
    │  _servers/n15.py         │  │  _servers/n15.py         │  │  _servers/n15.py         │
    │  REP   :5555             │  │  REP   :5556             │  │  REP   :5557             │
    │  CUDA_VISIBLE_DEVICES=0  │  │  CUDA_VISIBLE_DEVICES=1  │  │  CUDA_VISIBLE_DEVICES=2  │
    │                          │  │                          │  │                          │
    │  RobotInferenceServer    │  │  RobotInferenceServer    │  │  RobotInferenceServer    │
    │   ├─ ping                │  │   ├─ ping                │  │   ├─ ping                │
    │   └─ get_action ─► ...   │  │   └─ get_action ─► ...   │  │   └─ get_action ─► ...   │
    │                          │  │                          │  │                          │
    │  Gr00tPolicy (N1.5)      │  │  Gr00tPolicy (N1.5)      │  │  Gr00tPolicy (N1.5)      │
    │   EagleBackbone +        │  │   EagleBackbone +        │  │   EagleBackbone +        │
    │   Flow-matching          │  │   Flow-matching          │  │   Flow-matching          │
    │   action head            │  │   action head            │  │   action head            │
    │                          │  │                          │  │                          │
    │  owns: 12 GB ckpt on GPU │  │  owns: 12 GB ckpt on GPU │  │  owns: 12 GB ckpt on GPU │
    └──────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘
```

Legend:
- **DEALER → ROUTER** arrows: msgpack+ndarray (`cap/policy/grootpool/protocol.py`).
- **Middleware → Worker REQ** arrows: torch-serialized dict (gr00t's `BaseInferenceServer` wire format, unchanged).
- **FIFO** — idle workers enter the queue at startup and re-enter on `close()`; `open()` pops from the front.
- **Sticky** — once a session maps to a worker, every `step` for that session routes to the same REQ socket until `close()`.

## Protocol

msgpack with ndarray codec (same as `cap/policy/backends/zmq_backend.py`).

| op | request | response |
|----|---------|----------|
| `open` | `model`, `task_description` | `session_id`, `worker_id`, `queued_ms` |
| `step` | `session_id`, `seq`, `obs` | `seq`, `action` |
| `close` | `session_id` | `op: close_ok` |
| `ping` | — | `n_sessions`, `queue_depth`, `idle_workers` |

Errors: `{op: "error", code, message}` with `code ∈ {no_worker, unknown_session, worker_crash, worker_timeout, bad_request, internal}`.

## Files

| Path | Role |
|------|------|
| `cap/policy/grootpool/protocol.py` | Wire schema + msgpack/ndarray codec + error codes |
| `cap/policy/grootpool/supervisor.py` | Worker lifecycle — spawn/connect, respawn, mock, stub |
| `cap/policy/grootpool/session_manager.py` | Sticky map, idle queue, latency histogram, idle watchdog |
| `cap/policy/grootpool/server.py` | asyncio ROUTER loop + CLI entry |
| `cap/policy/grootpool/admin.py` | FastAPI `/status` + `/kill-session/{id}` |
| `cap/policy/grootpool/client.py` | `GrootPoolClient` — one DEALER per thread |
| `cap/policy/grootpool/backend.py` | `GrootpoolBackend` — wraps a session as `PolicyBackend` |
| `cap/policy/backends/grootpool_backend.py` | `GrootpoolN15Backend` / `GrootpoolN16Backend` + `GROOTPOOL_BACKENDS` registry — carry obs-spec metadata (`CONTROLLER_TYPE`, `LANGUAGE_KEY`, `CAMERA_MAP`) on the class |
| `cap/policy/grootpool/_servers/stub.py` | Torch-only fake GR00T server (for `--stub` mode) |
| `cap/agent/tools/grootpool.py` | Low-level agent tool (`session(...)` re-export) — superseded by `use_policy_output` for rollouts |
| `cap/env/robocasa/skills.py:use_policy_output` | CAP tool — resolves backend, asserts OSC pinning, delegates to `run_episode` |
| `cap/env/robocasa/env.py:as_gym_env` / `get_task_description` | RoboCasaEnv methods used by `use_policy_output` |
| `cap/saved_scripts/gr00t_oracle.py` | Reference oracle script — shape of agent-generated GR00T code |
| `cap/agent/agent_config.py:PolicyConfig` | Hydra dataclass (backend / model / endpoint) |
| `scripts/setup_grootpool.sh` | One-command env setup |
| `scripts/run_grootpool_eval.py` | Standalone RoboCasa365 × GR00T eval |
| `tmux/launch_grootpool.sh` | M workers + middleware in tmux |
| `scripts/launch_grootpool.sh` | Single-process launcher (mock / stub / real) |

## Modes

Mix and match for dev vs prod:

| Mode | Flag | Needs | Workers | Verifies |
|------|------|-------|---------|----------|
| `mock` | `--mock` | nothing | in-process stubs | middleware loop, session routing, `/status` |
| `stub` | `--stub` | torch + zmq | real subprocs speaking N1.5 wire protocol (no GR00T model) | above + subprocess lifecycle + torch-ZMQ |
| `external` | `--external-workers host:port,…` | pre-running workers (e.g. via tmux) | none (connect only) | operator-owned worker lifecycle |
| `real` | *(default)* | `Isaac-GR00T-benchmark/model_server_venv` + N1.5 ckpt + GPU | real N1.5 subprocs | full pipeline |

The **tmux launcher uses `external` mode** — workers run in their own windows so
a human can attach / restart them independently.

Local smoke on a laptop (no GPU, no gr00t):

```bash
bash scripts/launch_grootpool.sh --stub --n-workers 2 --gpu-ids 0,1 --startup-timeout 30

# in another shell
.venv/bin/python -c "
import numpy as np
from cap.policy.grootpool import GrootPoolClient
with GrootPoolClient().session(model='n15', task_description='test') as s:
    a = s.step({'video.x': np.zeros((256,256,3), np.uint8)})
    print(a['action.joint_position'].shape)   # (16, 7)
"
```

## Configuration

| Env var | Default | Notes |
|---------|---------|-------|
| `GROOTPOOL_ENDPOINT` | `tcp://127.0.0.1:7070` | Client-side — what CAP scripts dial |
| `GROOTPOOL_LISTEN_PORT` | `7070` | Server ROUTER port |
| `GROOTPOOL_ADMIN_PORT` | `7071` | FastAPI admin |
| `GROOTPOOL_N_WORKERS` | `2` | Workers to spawn (ignored in `external` mode) |
| `GROOTPOOL_GPU_IDS` | `0,1` | Cycled per worker |
| `GROOTPOOL_BASE_PORT` | `5555` | Worker ZMQ port; +1 per worker |
| `GROOTPOOL_MODEL_PATH` | — | Required for real mode (N1.5 checkpoint path) |
| `GROOT_BENCHMARK_ROOT` | — | Required for real mode (Isaac-GR00T-benchmark clone) |
| `GROOTPOOL_EXTERNAL_WORKERS` | — | Equivalent to `--external-workers`; comma-sep host:port |

CLI flags on `cap.policy.grootpool.server`: `--startup-timeout` (300 s),
`--step-timeout` (30 s), `--open-timeout` (60 s), `--session-idle-timeout` (60 s).

## Observability

`GET http://<host>:7071/status`:

```json
{
  "workers": [{"id": 0, "gpu": 0, "port": 5555, "pid": 12345, "alive": true, "respawning": false}],
  "n_sessions": 2,
  "queue_depth": 0,
  "idle_workers": 1,
  "step_p50_ms": 142.7,
  "step_p95_ms": 218.4,
  "sessions": [{"id": "…", "worker": 0, "model": "n15", "task": "…", "calls": 14, "last_activity_age_s": 0.4, "recent_mean_ms": 145.2}]
}
```

`POST /kill-session/{session_id}` force-closes a stuck session.

## Failure model

| Failure | Response |
|---------|----------|
| Worker crashes mid-step | Session → `worker_crash`; managed workers respawn; external workers reconnect on restart |
| Worker RCVTIMEO (hang) | Session → `worker_timeout`; worker SIGTERM'd (managed mode) |
| Script dies without `close` | Idle-timeout watchdog auto-closes after `--session-idle-timeout` |
| Middleware dies | DEALER sockets see timeouts on next `step` |
| `open` with no free worker | Blocks until `--open-timeout` → `no_worker` error |

## Teardown

```bash
tmux send-keys -t grootpool:middleware C-c     # graceful: expect "grootpool stopped"
tmux kill-session -t grootpool
nvidia-smi --query-compute-apps=pid,process_name --format=csv   # sanity
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `failed to start within 300s` | N1.5 load slow | `--startup-timeout 600`; check `logs/grootpool/worker_N_gpuM.log` |
| `Address already in use` on 7070/5555 | prior server still bound | `pkill -f grootpool.server; pkill -f "_servers/n15.py"` |
| `no_worker` from `open()` | all workers busy | raise `--n-workers` or `--open-timeout` |
| Worker `respawning=true` in `/status` | OOM / bad ckpt / CUDA driver | `logs/grootpool/worker_*.log`, change GPU |
| `ModuleNotFoundError: cap.policy.grootpool` | `PYTHONPATH` missing forge root | use `scripts/launch_grootpool.sh` or `tmux/launch_grootpool.sh`, not raw Python |
| `.env.grootpool not found` in tmux launcher | didn't run `setup_grootpool.sh` on this node | run setup first |

## Known limitations (v2)

- **N1.5 only at the obs-layer.** `GrootpoolN16Backend` is registered and has
  the right metadata, but N1.6 consumes different obs keys
  (`video.res256_image_side_0`, `annotation.human.action.task_description`,
  extra `state.*` keys). The current `_SharedRoboCasaGymView` produces the N1.5
  schema. To run N1.6, we need either an env subclass that remaps obs (like
  `_workers/robocasa365.py:_make_env_n16`) or an adapter layer between
  `gym_view` and the backend. Config-level obs-spec is already on the backend
  class to make this refactor small.
- Single-node.
- No cross-session batching (each `step` is 1-obs → 1-action-chunk).
- `/status` only — no Prometheus.

## Cross-references

- `docs/ROBOCASA_INTEGRATION_POLICY.md` — GR00T venvs, checkpoints, repo deps.
- `docs/remote_serving.md` — other OSMO-side services (cuRobo, SAM3, AnyGrasp).
- `cap/policy/backends/zmq_backend.py` — msgpack codec reused by `protocol.py`.
