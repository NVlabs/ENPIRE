# CAP System Dashboard

Standalone system inspector and control panel for all CAP/robot services.
Provides a unified view of service health, port allocation, communication
topology, tmux session management, and launch-profile control.

> **Related docs:**
> - `docs/CAP_DESIGN.md` -- CAP system architecture and layer stack
> - `docs/RL_PIPELINE_DESIGN.md` -- RL training pipeline (serve_rl_policy, learn_skill, diagnostics)
> - `docs/CAP_UI_DESIGN.md` -- CAP web UI layout, components, data flow
> - `docs/SAFETY_ZONE_DESIGN.md` -- task-aware EE safety zones for RL exploration
> - `docs/TABLE_BUSSING_SKILLS.md` -- table bussing skill tools
> - `docs/BUNDLESDF_OBJECT_DETECTION.md` -- BundleSDF multi-object 6-DOF pose tracking
> - `docs/grasp_orientation.md` -- Grasp orientation and AnyGrasp debug UI
> - `docs/VOICE_INPUT.md` -- voice input service
> - `docs/remote_serving.md` -- remote GPU serving (LeCAR-S1)

---

## What It Provides

- **Live snapshot** of services, ports, communication edges, and profile states
  (polled on a configurable interval, default 1 s).
- **Known port catalog** (23 ports) plus runtime-discovered **unmapped listeners**
  via `ss -H -ltnup`.
- **Health probes** per service: TCP connect, HTTP GET, Portal RPC call, or UDP
  listener check.
- **Profile tmux control** (`start` / `stop` / `restart`) with recreate
  semantics and automatic conflict resolution.
- **Tmux introspection** -- full session/window/pane tree, pane content capture,
  and pane input injection.
- **Kill-by-port** -- terminate the process holding a specific port.
- **WebSocket snapshot stream** (`/ws/v1/snapshot`) for real-time UI updates.

---

## Architecture Overview

```
 Browser (:5174)
   |
   |  Vite dev proxy  /api -> :8890,  /ws -> ws://:8890
   v
 System Dashboard Backend (FastAPI, :8890)
   |-- SystemRuntime        (bringup/system_runtime.py)
   |     |-- refresh_snapshot()   ss + tmux ls + probes
   |     |-- profile control      tmux new-session / kill-session
   |     |-- kill_port()          SIGTERM -> SIGKILL
   |     '-- tmux tree / pane content / pane send
   '-- SystemCatalog        (bringup/system_catalog.py)
         |-- KNOWN_PORTS    static port definitions
         |-- SERVICES       service definitions + primary probes
         |-- EDGES          communication topology
         '-- PROFILES       launch profiles + owned tmux sessions
```

### Key source files

| File | Purpose |
|------|---------|
| `bringup/system_dashboard.py` | FastAPI app factory, CLI entry point, static-file mount (`bringup/system_dashboard.py:34-145`) |
| `bringup/system_catalog.py` | Static catalog: `KNOWN_PORTS`, `SERVICES`, `EDGES`, `PROFILES`, `project_root()` (`bringup/system_catalog.py:1-473`) |
| `bringup/system_runtime.py` | `SystemRuntime` class: polling, probes, snapshot builder, profile launchers, tmux plumbing (`bringup/system_runtime.py:1-996`) |
| `bringup/ui/` | React + Vite frontend (TypeScript, no component library) |
| `bringup/ui/src/types.ts` | TypeScript snapshot / tmux types (`bringup/ui/src/types.ts:1-114`) |
| `bringup/ui/src/components/Dashboard.tsx` | Main dashboard component |
| `bringup/tests/test_system_runtime.py` | Unit tests for `SystemRuntime`, `parse_ss_output`, probe logic |
| `bringup/tests/test_system_api.py` | FastAPI endpoint tests with `FakeRuntime` |

---

## Backend

### Running

```bash
uv run -m bringup.system_dashboard --host 127.0.0.1 --port 8890
```

CLI flags (`bringup/system_dashboard.py:128-141`):

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8890` | Bind port |
| `--poll-interval-s` | `1.0` | Background snapshot refresh interval (seconds) |

On startup the app calls `rt.refresh_snapshot()` and `rt.start_poller()`.
If `bringup/ui/dist/` exists, the built frontend is served as static files at `/`.

### REST / WebSocket Endpoints

| Method | Path | Description | Ref |
|--------|------|-------------|-----|
| `GET` | `/api/v1/snapshot` | Full system snapshot (services, edges, profiles, ports, summary) | `system_dashboard.py:46` |
| `GET` | `/api/v1/profiles` | List all profiles with status | `system_dashboard.py:50` |
| `POST` | `/api/v1/profiles/{profile_id}/start` | Start profile (recreate semantics) | `system_dashboard.py:54` |
| `POST` | `/api/v1/profiles/{profile_id}/stop` | Stop profile (kill owned tmux sessions) | `system_dashboard.py:58` |
| `POST` | `/api/v1/profiles/{profile_id}/restart` | Restart profile | `system_dashboard.py:62` |
| `GET` | `/api/v1/events` | Event log (query param `limit`, default 100, ring buffer max 300) | `system_dashboard.py:66` |
| `POST` | `/api/v1/ports/{protocol}/{port}/kill` | Kill process on port (SIGTERM then SIGKILL), returns updated snapshot | `system_dashboard.py:70` |
| `GET` | `/api/v1/tmux/tree` | Full tmux session/window/pane tree | `system_dashboard.py:76` |
| `GET` | `/api/v1/tmux/pane/{pane_id}/content` | Capture pane content (query param `lines`, default 200, max 2000) | `system_dashboard.py:80` |
| `POST` | `/api/v1/tmux/pane/{pane_id}/send` | Send text to pane, body: `{"text": "...", "enter": true}` | `system_dashboard.py:84` |
| `WS` | `/ws/v1/snapshot` | Streaming snapshots at `poll_interval_s` rate | `system_dashboard.py:88` |

### Snapshot Shape

Returned by `GET /api/v1/snapshot` and the WebSocket stream
(`bringup/system_runtime.py:453-475`):

```jsonc
{
  "generated_at": 1712345678.9,
  "services":      [ /* ServiceState[] */ ],
  "edges":         [ /* EdgeState[]    */ ],
  "profiles":      [ /* ProfileState[] */ ],
  "ports":         [ /* PortState[]    */ ],
  "unmapped_ports": [ /* UnmappedPort[] */ ],
  "summary": {
    "services_up": 5,   "services_total": 22,
    "edges_up": 10,     "edges_total": 18,
    "profiles_running": 1, "profiles_total": 11,
    "mapped_ports_total": 23, "mapped_ports_up": 8,
    "unmapped_ports": 2
  }
}
```

TypeScript types for these shapes live in `bringup/ui/src/types.ts`.

---

## Frontend

### Running (dev mode)

```bash
cd bringup/ui
npm install
npm run dev          # Vite dev server on :5174
```

Default UI URL: `http://localhost:5174`

### Production build

```bash
cd bringup/ui
npm run build        # outputs to bringup/ui/dist/
```

When `bringup/ui/dist/` exists, the backend automatically serves it at `/`
(`bringup/system_dashboard.py:108-110`).

### Vite proxy configuration

Defined in `bringup/ui/vite.config.ts:8-17`:

| Path prefix | Target |
|-------------|--------|
| `/api` | `http://127.0.0.1:8890` |
| `/ws` | `ws://127.0.0.1:8890` |

### Tech stack

- React 19, TypeScript 5.6+, Vite 6
- No component library (plain CSS in `bringup/ui/src/styles.css`)
- Test runner: Vitest + @testing-library/react + jsdom

---

## Tmux Launch Script

A standalone tmux launcher is available at
`tmux/table_bussing/table_bussing_history/launch_system_dashboard.sh`.

```bash
./tmux/table_bussing/table_bussing_history/launch_system_dashboard.sh [--host HOST] [--backend-port PORT] [--ui-port PORT] [--no-attach]
```

Creates a `system-dashboard` tmux session with two panes:
- Pane 0: backend (`uv run -m bringup.system_dashboard`)
- Pane 1: frontend (`npm install && npm run dev`)

---

## Service Catalog

### Registered Services (22 total)

Defined in `bringup/system_catalog.py:89-243` as `SERVICES`:

| service_id | Port | Proto | Probe | Description |
|------------|------|-------|-------|-------------|
| `cap_ui_dev` | 5173 | tcp | HTTP `/` | Vite dev server for CAP UI |
| `system_ui_dev` | 5174 | tcp | HTTP `/` | Vite dev server for system dashboard |
| `serve_sam3` | 6767 | tcp | HTTP `/health` | SAM3 text-prompted segmentation (remote GPU / LeCAR-S1) |
| `viser` | 8080 | tcp | TCP connect | 3D viewer hosted by cap_agent |
| `serve_pose` | 8118 | tcp | HTTP `/health` | OWLv2 detection server |
| `serve_bundlesdf` | 8119 | tcp | HTTP `/health` | BundleSDF multi-object 6-DOF pose tracking |
| `serve_graspnet` | 8120 | tcp | HTTP `/health` | ContactGraspNet 6-DOF grasp planning (remote GPU) |
| `cap_agent` | 8200 | tcp | HTTP `/api/state` | CAP orchestrator REST/WS server |
| `claude_bridge` | 8201 | tcp | HTTP `/api/chat/status` | Agent bridge (Claude Code / OpenAI Codex) |
| `cap_server` | 8300 | tcp | Portal `get_state` | Control loop and Portal RPC owner |
| `smol_vlm` | 8401 | tcp | HTTP `/v1/models` | vLLM OpenAI-compatible API (SmolVLM) |
| `reward_server` | 8500 | tcp | Portal `get_reward({})` | Portal reward function server |
| `diag_dashboard` | 8888 | tcp | HTTP `/api/stats` | RL diagnostics dashboard |
| `system_dashboard` | 8890 | tcp | HTTP `/api/v1/snapshot` | This system dashboard backend |
| `pico_stream` | 8963 | tcp | TCP connect | Pico stream service |
| `policy_server` | 8964 | tcp | TCP connect | Flow-matching policy server |
| `rl_policy_server` | 8965 | tcp | TCP connect | RL policy server |
| `diag_udp` | 9999 | udp | UDP listener check | UDP diagnostics collector sink |
| `left_follower` | 11333 | tcp | Portal `get_observations` | Portal follower arm server (left) |
| `right_follower` | 11334 | tcp | Portal `get_observations` | Portal follower arm server (right) |
| `left_leader` | 11335 | tcp | Portal `get_info` | Portal leader/fello arm server (left) |
| `right_leader` | 11336 | tcp | Portal `get_info` | Portal leader/fello arm server (right) |

### Known Ports (23 total)

Defined in `bringup/system_catalog.py:62-86` as `KNOWN_PORTS`:

```
5173, 5174, 6767, 8080, 8118, 8119, 8120,
8200, 8201, 8300, 8401, 8402,
8500, 8888, 8890,
8963, 8964, 8965,
9999/udp,
11333, 11334, 11335, 11336
```

Port 8402 (Qwen3-VL/vLLM SSH tunnel) is in `KNOWN_PORTS` but has no
corresponding `ServiceDef`.

**Not yet in the catalog:**
- Port 8202 -- voice input server (`cap/voice/voice_server.py:21`, env
  `CAP_VOICE_PORT`). Used by `tmux/cap_agent_interface/start_cap_agent_interface.sh:23`.

Any listening port not in the catalog appears as **Unmapped** in the snapshot.

### Communication Edges (18 total)

Defined in `bringup/system_catalog.py:247-392` as `EDGES`:

```
browser -> cap_ui_dev                  (HTTP)
cap_ui_dev -> cap_agent                (REST+WS)
cap_ui_dev -> claude_bridge            (REST+WS)
cap_ui_dev -> viser                    (WS/TCP iframe)
cap_agent -> cap_server                (Portal RPC)
cap_agent -> serve_pose                (HTTP)
cap_agent -> serve_sam3                (HTTP)
cap_agent -> serve_bundlesdf           (HTTP)
cap_agent -> serve_graspnet            (HTTP)
cap_agent -> smol_vlm                  (HTTP)
cap_server -> left_follower            (Portal RPC)
cap_server -> right_follower           (Portal RPC)
cap_server -> left_leader              (Portal RPC)
cap_server -> right_leader             (Portal RPC)
cap_server -> policy_server            (Portal/TCP)
cap_server -> rl_policy_server         (Portal/TCP)
cap_server/reward_server -> diag_udp   (UDP)
reward_server -> cap_server            (logical dependency)
```

---

## Launch Profiles

Defined in `bringup/system_catalog.py:395-462` as `PROFILES` (11 total).
Profile start logic lives in `bringup/system_runtime.py:829-948`.

### CAP profiles

| profile_id | tmux session | What it starts | Ref |
|------------|-------------|----------------|-----|
| `cap_basics` | `cap` | cap_server, cap_agent, CAP UI dev, serve_pose (cuda :8118), diag dashboard | `system_runtime.py:841-870` |
| `rl_test` | `rl-test` | reward_server (constant-1 mode), diag dashboard | `system_runtime.py:872-887` |

### Robot profiles (via `launch.py`)

All robot profiles delegate to `uv run launch.py --mode <mode> [flags] --no-attach`
(`system_runtime.py:921-948`) and own the tmux sessions `robots`, `cameras`, `main`.

| profile_id | Mode | Fello | Voice |
|------------|------|-------|-------|
| `robot_dev_yam` | dev | no | -- |
| `robot_dev_fello` | dev | yes | -- |
| `robot_data_collection_yam` | data_collection | no | -- |
| `robot_data_collection_fello` | data_collection | yes | -- |
| `robot_data_collection_fello_voice` | data_collection | yes | yes |
| `robot_a5_data_collection` | a5_data_collection | -- | -- |
| `robot_evaluation_yam` | evaluation | no | -- |
| `robot_evaluation_fello` | evaluation | yes | -- |
| `robot_evaluation` | evaluation (legacy) | no | -- |

---

## Profile Control Semantics

Implemented in `bringup/system_runtime.py:376-430`:

1. **Start = recreate**: owned tmux sessions are killed first, then re-created.
2. **Conflict resolution**: before starting, any other profile that shares a tmux
   session is auto-stopped (`system_runtime.py:796-818`).
3. **Transient status**: while a profile action is in flight the profile reports
   `starting` / `stopping` for up to 120 s (`system_runtime.py:965-969`).
4. **No arbitrary command execution**: the only send-keys endpoint targets an
   already-existing tmux pane by ID.

---

## Probe System

Probe types and execution live in `bringup/system_runtime.py:693-755`:

| ProbeKind | Mechanism | Timeout |
|-----------|-----------|---------|
| `tcp` | `socket.create_connection` | `probe.timeout_s` (default 0.7 s) |
| `http` | `urllib.request.urlopen` GET, accepts 2xx-4xx | `probe.timeout_s` |
| `portal` | TCP pre-check then Portal RPC method call via `portal.Client`; retries once with fresh client | `probe.timeout_s + 0.3` |
| `udp` | Checks if `ss` reports a UDP listener on the port | instant |
| `logical` | Always returns True | instant |

Portal clients are cached per endpoint in `_portal_clients` dict
(`system_runtime.py:681-691`).

---

## UDP Diagnostics Emitter

The lightweight `emit()` function is defined in `cap/diag/emitter.py:31-43`.
It sends msgpack-encoded UDP datagrams to `DIAG_HOST:DIAG_PORT` (default
`127.0.0.1:9999`). Fire-and-forget with ~1 us overhead. Disable entirely with
`DIAG_ENABLED=0`.

### Emitter configuration (env vars)

| Var | Default | Description |
|-----|---------|-------------|
| `DIAG_ENABLED` | `1` | Set to `0` to disable all emit calls |
| `DIAG_HOST` | `127.0.0.1` | UDP target host |
| `DIAG_PORT` | `9999` | UDP target port |

### Sources that call `emit()`

| Source module | `src` field | Example events |
|---------------|-------------|----------------|
| `cap/server/cap_server.py` | `"cap_server"` | `step_start`, `fello_read`, `obs_built`, `rpc_call_start`, `rpc_call_end`, `reward_start`, `reward_end`, `action_applied`, `record_done`, `step_end` |
| `cap/server/cap_server.py` | `"control_loop"` | `tick` (every 20th tick, includes `dt_ms` meta) |
| `cap/server/cap_server.py` | `"fello_loop"` | tick events |
| `cap/server/cap_server.py` | `"camera"` | frame events (every 10th frame, includes `camera_name`, `dt_ms` meta) |
| `cap/reward/reward_server.py` | `"reward_server"` | `reward_recv`, `reward_compute`, `reward_send` |

---

## RL Diagnostics Dashboard (cap/diag/dashboard.py)

Separate from the system dashboard. This is the real-time RL pipeline
performance monitor. Inline HTML UI, no build step required.

Run: `uv run cap/diag/dashboard.py [--udp-port 9999] [--http-port 8888]`

| Feature | Detail | Ref |
|---------|--------|-----|
| UDP collector | Listens on `:9999`, ingests msgpack events into `Store` ring buffer | `cap/diag/dashboard.py:374-387` |
| Motor temp collector | Polls Portal RPC `get_motor_temperatures()` on all 4 arm servers (11333-11336) | `cap/diag/dashboard.py:390-428` |
| Step timeline | Stacked-bar visualization of loop segments (fello_read, build_obs, rpc_call, reward, apply_action, record, idle) | `cap/diag/dashboard.py:43-58` |
| RPC breakdown | base_action, sac_sample, buffer_insert, network_rtt | `cap/diag/dashboard.py:61-65` |
| Health panel | Episode, step, loop Hz, control Hz + jitter, fello Hz, camera FPS, action source, reward, motor temps | `cap/diag/dashboard.py:247-367` |
| HTTP endpoint | `GET /api/stats` returns `{ steps, stats, health }` | `cap/diag/dashboard.py:467-469` |
| WebSocket | `WS /ws` streams snapshots at 5 Hz (200 ms interval) | `cap/diag/dashboard.py:453-464` |

The system dashboard catalog registers this as the `diag_dashboard` service
on port 8888 and probes `GET /api/stats`.

---

## CAP Agent Interface Scripts

The `tmux/cap_agent_interface/` directory provides a non-tmux process-manager
alternative that launches the full CAP agent stack as background processes with
PID files and log files.

| Script | Purpose |
|--------|---------|
| `start_cap_agent_interface.sh` | Starts cap_server (:8300), cap_agent (:8200), bridge (:8201), voice (:8202), UI (:5173) as background processes. Waits for health checks. Logs to `.cap-agent-interface/`. |
| `stop_cap_agent_interface.sh` | Stops all processes using PID files. |
| `status_cap_agent_interface.sh` | Reports running/stopped status of each service. |

---

## Manual Validation

1. Start `cap_basics` from the dashboard and verify session `cap` is present in `tmux ls`.
2. Restart `cap_basics` and verify session recreation (fresh panes/commands).
3. Open an unmapped listener and verify it appears in Unmapped Ports:

```bash
python -m http.server 60000
```

4. Stop `cap_basics` and verify owned sessions disappear.
5. Use the tmux tree endpoint to browse panes and capture content:

```bash
curl http://localhost:8890/api/v1/tmux/tree | python -m json.tool
curl 'http://localhost:8890/api/v1/tmux/pane/%250/content?lines=50'
```

6. Kill a port and verify the process terminates:

```bash
curl -X POST http://localhost:8890/api/v1/ports/tcp/60000/kill
```

---

## Tests

Backend tests:

```bash
uv run pytest bringup/tests/test_system_runtime.py bringup/tests/test_system_api.py
```

- `test_system_runtime.py` -- `parse_ss_output`, probe logic, profile conflicts, snapshot builder
- `test_system_api.py` -- all REST endpoints and WebSocket via `FakeRuntime`

Frontend tests:

```bash
cd bringup/ui
npm test           # vitest run
```

- `bringup/ui/src/__tests__/dashboard.test.tsx` -- Dashboard component rendering

---

## Appendix: Full Port Map

Quick reference of every port the platform uses, sorted numerically.

| Port | Proto | Service | Probe | Notes |
|------|-------|---------|-------|-------|
| 5173 | tcp | CAP UI dev (Vite) | HTTP `/` | |
| 5174 | tcp | System dashboard UI dev (Vite) | HTTP `/` | |
| 6767 | tcp | serve_sam3 (SAM3 segmentation) | HTTP `/health` | Remote GPU / LeCAR-S1 |
| 8080 | tcp | Viser 3D viewer | TCP connect | Embedded in cap_agent |
| 8118 | tcp | serve_pose (OWLv2) | HTTP `/health` | |
| 8119 | tcp | serve_bundlesdf (6-DOF tracking) | HTTP `/health` | |
| 8120 | tcp | serve_graspnet (grasp planning) | HTTP `/health` | Remote GPU / LeCAR-S1 |
| 8200 | tcp | cap_agent | HTTP `/api/state` | FastAPI orchestrator |
| 8201 | tcp | agent_bridge | HTTP `/api/chat/status` | Claude Code / OpenAI Codex |
| 8202 | tcp | voice_server | -- | **Not in catalog yet** (`CAP_VOICE_PORT` env) |
| 8300 | tcp | cap_server | Portal `get_state` | Control loop, 30 Hz |
| 8401 | tcp | SmolVLM (vLLM) | HTTP `/v1/models` | |
| 8402 | tcp | Qwen3-VL (vLLM SSH tunnel) | -- | In KNOWN_PORTS, no ServiceDef |
| 8500 | tcp | reward_server | Portal `get_reward` | |
| 8888 | tcp | diag_dashboard (RL diagnostics) | HTTP `/api/stats` | Inline HTML UI |
| 8890 | tcp | system_dashboard | HTTP `/api/v1/snapshot` | This service |
| 8963 | tcp | pico_stream | TCP connect | |
| 8964 | tcp | policy_server | TCP connect | Flow-matching policy |
| 8965 | tcp | rl_policy_server | TCP connect | RL policy |
| 9999 | udp | diag UDP sink | UDP listener check | `cap/diag/emitter.py` target |
| 11333 | tcp | left_follower_arm | Portal `get_observations` | |
| 11334 | tcp | right_follower_arm | Portal `get_observations` | |
| 11335 | tcp | left_leader_fello | Portal `get_info` | |
| 11336 | tcp | right_leader_fello | Portal `get_info` | |
