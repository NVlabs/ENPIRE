# Remote Serving

This repo uses a lightweight client / remote GPU model server split. GPU-heavy
perception and motion-planning services run on a dedicated remote host
(LeCAR-S1, currently `LeCAR_4xRTX6000BlackWell_97GB` in SSH config). The local
client host (4070) runs the control loop, agent, UI, bridge, and voice. SSH
port-forwarding tunnels bridge the two so every local consumer sees
`localhost:<port>` transparently.

> **Cross-references**
>
> - `docs/CAP_DESIGN.md` -- CAP architecture and layer stack
> - `docs/BUNDLESDF_OBJECT_DETECTION.md` -- BundleSDF 6-DOF tracking system
> - `docs/CUROBO_UPDATE_SUMMARY.md` -- cuRobo integration details
> - `docs/lfs_setup.md` -- Git LFS strategy (lightweight clone, manual pull)
> - `docs/VLM_QUERY.md` -- VLM backends (Qwen3-VL tunnel, SmolVLM, Gemini)
> - `docs/TABLE_BUSSING_SKILLS.md` -- table-bussing skill tools
> - `docs/CAP_SYSTEM_DASHBOARD.md` -- system dashboard and service catalog
> - `bringup/system_catalog.py` -- canonical port/service/probe definitions

---

## Active remote model stack

Only these remote model services are part of the supported path:

| Service | Server script | Protocol | Port | Health probe |
|---------|--------------|----------|------|-------------|
| SAM3 | `tools/vision/serve_sam3.py` | HTTP (FastAPI) | 6767 | `GET /health` |
| BundleSDF (incl. SAM2 tracking) | `tools/vision/serve_bundlesdf.py` | HTTP (FastAPI) | 8119 | `GET /health` |
| AnyGrasp | `tools/vision/serve_anygrasp.py` via `tools/vision/launch_anygrasp_server.sh` | HTTP (FastAPI) | 8122 | `GET /health` |
| cuRobo motion planner | `experimental/serve_portal_motion_planner.py` | Portal RPC | 8611 | `portal.Client.health_check()` |

Deprecated backends removed from the active path:
- OWLvIT / OWLv2
- GraspNet / ContactGraspNet

---

## Port map

### Remote model host (LeCAR-S1)

| Port | Service | Protocol | Binding | Source |
|------|---------|----------|---------|--------|
| `6767` | SAM3 segmentation | HTTP | `0.0.0.0` | `tools/vision/serve_sam3.py --host 0.0.0.0 --port 6767` |
| `8119` | BundleSDF 6-DOF tracking | HTTP | `0.0.0.0` | `tools/vision/serve_bundlesdf.py --host 0.0.0.0 --port 8119` |
| `8122` | AnyGrasp grasp planning | HTTP | `0.0.0.0` | `tools/vision/launch_anygrasp_server.sh --host 0.0.0.0` |
| `8611` | cuRobo Portal planner | Portal RPC | `0.0.0.0` | `experimental/serve_portal_motion_planner.py --port 8611 --solver-speed fast` |
| `18300` | Reverse tunnel to client CAP server | Portal RPC | SSH `-R` | tunnelled to `client:8300` |

Port `8611` can be overridden with `CAP_CUROBO_PORT` or `CAP_CUROBO_REMOTE_PORT` env vars
(`tmux/remote_serving/launch_lecar_s1.sh:27`, `tmux/table_bussing/launch_table_bussing_remote.sh:60`).

### Local client host (4070)

| Port | Service | Protocol | Source |
|------|---------|----------|--------|
| `8300` | cap_server (control loop) | Portal RPC | `cap/server/cap_server.py` |
| `8200` | cap_agent (orchestrator) | REST + WebSocket | `cap/agent/cap_agent.py` |
| `8201` | agent bridge (Claude Code / Codex) | REST + WebSocket | `cap/bridge/agent_bridge.py` |
| `8202` | voice server | HTTP | `cap/voice/voice_server` module |
| `5173` | CAP UI dev server | HTTP | `cap/ui/` (Vite) |
| `8080` | Viser 3D viewer | WebSocket | embedded in cap_agent |
| `11333` | left follower arm | Portal RPC | `robot/` drivers |
| `11334` | right follower arm | Portal RPC | `robot/` drivers |
| `11335` | left leader (Fello) | Portal RPC | `robot/` drivers |
| `11336` | right leader (Fello) | Portal RPC | `robot/` drivers |

### VLM / LLM services (separate from remote model stack)

| Port | Service | Access method | Details |
|------|---------|---------------|---------|
| `8401` | SmolVLM (vLLM) | Direct LAN to `172.26.34.251:8401` | `cap/config.py:242` |
| `8402` | Qwen3-VL (vLLM) | SSH tunnel `localhost:8402 -> gpu-node:8000` | `cap/config.py:251`, see `docs/VLM_QUERY.md` |

VLM backend selection is controlled by `DEFAULT_VLM_BACKEND` env var (default: `"qwen"`).
See `third_party/vllm_serving/README.md` for vLLM server setup.

Port assignments are canonically defined in `cap/config.py:17-99` and
`bringup/system_catalog.py:62-86`.

---

## Architecture: local vs. remote split

`tmux/table_bussing/launch_table_bussing_remote.sh` preserves the original
client-side workflow from `tmux/table_bussing/launch_table_bussing_local.sh`
while moving only the GPU-heavy perception + planning services to S1.

### Stays on the local client (4070)

Started by `launch_table_bussing_remote.sh`:

| Process | How launched | Ref |
|---------|-------------|-----|
| `cap_server` | tmux pane | `launch_table_bussing_remote.sh:378` |
| `cap_agent` | tmux pane (waits for all remote services first) | `launch_table_bussing_remote.sh:380` |
| `cap_ui` | tmux pane (`npm install && npm run dev`) | `launch_table_bussing_remote.sh:385` |
| `agent_bridge` | `nohup` background process | `launch_table_bussing_remote.sh:344` |
| `voice_server` | `nohup` background process | `launch_table_bussing_remote.sh:350` |
| Evaluation (launch.py) | `nohup` background process | `launch_table_bussing_remote.sh:332` |
| SSH tunnel | tmux pane | `launch_table_bussing_remote.sh:382` |

### Moves to LeCAR-S1

Started by `tmux/remote_serving/launch_lecar_s1.sh`:

| Process | tmux session | Ref |
|---------|-------------|-----|
| `serve_sam3` | `lecar-s1-remote-serving` pane | `launch_lecar_s1.sh:214` |
| `serve_anygrasp` | `lecar-s1-remote-serving` pane | `launch_lecar_s1.sh:221` |
| `serve_bundlesdf` | `lecar-s1-remote-serving` pane | `launch_lecar_s1.sh:224` |
| `serve_portal_motion_planner` (cuRobo) | `lecar-s1-remote-serving` pane | `launch_lecar_s1.sh:228` |

The intended operator experience stays the same:
- Inspect the UI on `http://localhost:5173/`
- Talk to the bridge / switch agent backends locally on `:8201`
- Keep the robot-facing CAP server local on `:8300`
- Use S1 only for the heavy model-serving ports

---

## SSH forwarding logic

The lightweight client connects to the remote model host through SSH port forwarding.
The tunnel command is built in `launch_table_bussing_remote.sh:382`:

```
ssh -N -o ExitOnForwardFailure=yes \
  -L 6767:127.0.0.1:6767 \
  -L 8122:127.0.0.1:8122 \
  -L 8119:127.0.0.1:8119 \
  -L 8611:127.0.0.1:8611 \
  -R 18300:127.0.0.1:8300 \
  LeCAR_4xRTX6000BlackWell_97GB
```

### Forwarded local listeners (-L)

| Local endpoint | Remote endpoint | Service |
|---------------|-----------------|---------|
| `localhost:6767` | `S1:6767` | SAM3 |
| `localhost:8119` | `S1:8119` | BundleSDF |
| `localhost:8122` | `S1:8122` | AnyGrasp |
| `localhost:8611` | `S1:8611` | cuRobo Portal |

### Reverse listener (-R) back to the client

| Remote endpoint | Local endpoint | Purpose |
|----------------|----------------|---------|
| `S1:18300` | `client:8300` | Lets BundleSDF on S1 reach the local CAP server via Portal RPC for camera frames |

### Troubleshooting tunnel errors

If you see:
```
channel N: open failed: connect failed: Connection refused
```
one side of the tunnel was contacted before the target service on the other side
was listening yet. For example, S1 may try to call `127.0.0.1:18300` before the
local CAP server on `:8300` is ready, or the local client may probe a forwarded
model port before the remote model process has finished starting.
This is transient and resolves once the target service starts.

The SSH target defaults to `LeCAR_4xRTX6000BlackWell_97GB` and can be overridden:
- env var: `SSH_TARGET`
- flag: `--ssh-target <alias>`
- definition: `launch_table_bussing_remote.sh:57`

---

## Runtime environment configuration

Machine-specific runtime paths are configured outside the repo via
`tools/runtime_env.sh` (sourced by all launcher scripts).

### Environment variables

| Variable | Purpose | Set by |
|----------|---------|--------|
| `RUNTIME_TMP_ROOT` | Writable tmp/cache root for model caches, temp files, extracted runtimes | `configure_runtime_env.sh` |
| `RUNTIME_DEPS_ROOT` | Host-specific compatibility libraries (libssl1.1, OpenBLAS) kept outside the repo | `configure_runtime_env.sh` |
| `HF_HUB_OFFLINE` | Force Hugging Face model loading from local cache only | `configure_runtime_env.sh` |
| `TRANSFORMERS_OFFLINE` | Same offline behavior for Transformers | `configure_runtime_env.sh` |
| `HF_HOME` | Hugging Face cache directory | auto-detected by `tools/runtime_env.sh:60-85` |
| `CUDA_HOME` | CUDA toolkit root | auto-detected by `tools/runtime_env.sh:87-112` |
| `PROJECT_ROOT` | Repo root (set by `tools/runtime_env.sh:9`) | auto |
| `CAP_CUROBO_REMOTE_PORT` | cuRobo Portal port override (remote serving) | `tools/runtime_env.sh:136` (default `8611`) |
| `CAP_CUROBO_SSH_TARGET` | SSH alias for the remote GPU box | `tools/runtime_env.sh:137` |
| `CAP_CUROBO_HOST` | cuRobo host for `freespace_move` tool | default `127.0.0.1` (`cap/agent/tools/freespace_move.py:327`) |
| `CAP_CUROBO_START_SERVER` | If `0`, use external cuRobo server; if `1` or unset + no port, start in-process | `cap/agent/tools/freespace_move.py:330` |

### Source chain

1. `~/.config/lecar-tbd/runtime_env.sh` (user env file, written by `configure_runtime_env.sh`)
2. `~/.bashrc` (which sources the above if configured)
3. `tools/runtime_env.sh` (repo entry point, sourced by all launchers at startup)

Loading order in `tools/runtime_env.sh:162-165`:
```
source_user_runtime_env_file()   # reads ~/.config/lecar-tbd/runtime_env.sh
source_user_bashrc()             # reads ~/.bashrc
sanitize_python_launcher_env()   # unsets VIRTUAL_ENV/CONDA_PREFIX/etc.
export_repo_runtime_defaults()   # sets PROJECT_ROOT, CUDA_HOME, HF_HOME, etc.
```

### Current intended host setup
- **4070 (client)**: use a writable local tmp/cache path chosen by the user
- **S1 (remote GPU)**: use `/usr0/tonghez/tmp` because the home filesystem is space constrained

---

## Hugging Face model cache location

When `RUNTIME_TMP_ROOT` is set, `tools/runtime_env.sh:152-155` resolves the HF cache to:
```
$RUNTIME_TMP_ROOT/huggingface/hub
```

For S1 this means the live cache should be under:
- `/usr0/tonghez/tmp/huggingface/hub/models--facebook--sam3`
- `/usr0/tonghez/tmp/huggingface/hub/models--facebook--sam2.1-hiera-large`

These two cached models are required. Sync them using:
```bash
bash tmux/remote_serving/sync_remote_model_cache.sh [remote_host]
```
(`tmux/remote_serving/sync_remote_model_cache.sh` -- rsyncs `models--facebook--sam3` and
`models--facebook--sam2.1-hiera-large` from the local HF cache to the remote host.)

---

## Git LFS behavior

Large runtime artifacts (compiled `.so` files, model checkpoints) are tracked with
Git LFS. See `docs/lfs_setup.md` for full details.

After switching branches:
```bash
git lfs pull
```

The remote-serving launcher scripts also clear the repo-local `lfs.fetchexclude`
before pulling so fresh clones and branch switches restore required artifacts
(`launch_lecar_s1.sh:110-111`):
```bash
git -C "$PROJECT_DIR" config --local lfs.fetchexclude ""
git -C "$PROJECT_DIR" lfs pull
```

### Required LFS dependencies

Checked at S1 startup (`launch_lecar_s1.sh:113-119`):
- `checkpoint_detection.tar` (AnyGrasp checkpoint)
- `license_JalenLu.zip` (AnyGrasp license)
- `third_party/anygrasp_sdk/pointnet2/build/.../pointnet2/_ext.cpython-311-x86_64-linux-gnu.so`
- `third_party/anygrasp_sdk/dependencies/MinkowskiEngine/build/.../_C.cpython-311-x86_64-linux-gnu.so`
- `third_party/bundlesdf/libs/libBundleTrack.so`
- `third_party/bundlesdf/libs/libMY_CUDA_LIB.so`
- `third_party/bundlesdf/libs/my_cpp.cpython-311-x86_64-linux-gnu.so`

---

## Key scripts

### Remote serving (`tmux/remote_serving/`)

| Script | Purpose | Ref |
|--------|---------|-----|
| `launch_lecar_s1.sh` | Starts SAM3 + BundleSDF + AnyGrasp + cuRobo on S1 in a tmux session | `tmux/remote_serving/launch_lecar_s1.sh` |
| `stop_lecar_s1.sh` | Kills the remote tmux serving session | `tmux/remote_serving/stop_lecar_s1.sh` |
| `configure_runtime_env.sh` | Interactive/scripted writer for `~/.config/lecar-tbd/runtime_env.sh` | `tmux/remote_serving/configure_runtime_env.sh` |
| `sync_lecar_s1_assets.sh` | Runs `git lfs pull` on both local and remote clones, verifies dependencies | `tmux/remote_serving/sync_lecar_s1_assets.sh` |
| `sync_remote_model_cache.sh` | Rsyncs SAM3 + SAM2 HF model caches to S1 | `tmux/remote_serving/sync_remote_model_cache.sh` |
| `install_curobo_s1.sh` | Verifies or installs cuRobo into the S1 repo `.venv` | `tmux/remote_serving/install_curobo_s1.sh` |

### Table-bussing launchers (`tmux/table_bussing/`)

| Script | Purpose | Ref |
|--------|---------|-----|
| `launch_table_bussing_local.sh` | All-local stack (all services on 4070) | `tmux/table_bussing/launch_table_bussing_local.sh` |
| `launch_table_bussing_remote.sh` | Lightweight local client + SSH tunnel + remote GPU models | `tmux/table_bussing/launch_table_bussing_remote.sh` |

### Server entrypoints

| Script | Service | Ref |
|--------|---------|-----|
| `tools/vision/serve_sam3.py` | SAM3 text-prompted segmentation | `tools/vision/serve_sam3.py` |
| `tools/vision/serve_bundlesdf.py` | BundleSDF multi-object 6-DOF tracking (includes SAM2 internally) | `tools/vision/serve_bundlesdf.py` |
| `tools/vision/serve_anygrasp.py` | AnyGrasp 6-DOF grasp planning | `tools/vision/serve_anygrasp.py` |
| `tools/vision/launch_anygrasp_server.sh` | AnyGrasp launcher (LFS resolution, OpenSSL/OpenBLAS setup, port kill) | `tools/vision/launch_anygrasp_server.sh` |
| `experimental/serve_portal_motion_planner.py` | cuRobo Portal motion planner server | `experimental/serve_portal_motion_planner.py` |
| `tools/vision/warmup_anygrasp.py` | Sends a synthetic `/plan_viz` request to pre-warm AnyGrasp | `tools/vision/warmup_anygrasp.py` |

### Health / diagnostic tools

| Script | Purpose | Ref |
|--------|---------|-----|
| `tools/remote/check_motion_planner_portal.py` | cuRobo Portal health check + roundtrip probe | `tools/remote/check_motion_planner_portal.py` |
| `tools/runtime_env.sh` | Central env bootstrapper sourced by all launchers | `tools/runtime_env.sh` |

---

## cuRobo motion planner integration

The cuRobo motion planner is the default `freespace_move` backend. It runs as a
Portal RPC server (`experimental/serve_portal_motion_planner.py`), wrapping
`experimental/portal_motion_planner.PortalMotionPlannerServer`.

### How cap_agent connects to cuRobo

`cap/agent/tools/freespace_move.py:324-345` reads three env vars to decide how
to connect:

| Env var | Effect |
|---------|--------|
| `CAP_CUROBO_PORT` | Portal port to connect to (e.g. `8611`) |
| `CAP_CUROBO_HOST` | Host address (default `127.0.0.1`) |
| `CAP_CUROBO_START_SERVER` | `0` = use external server; `1` or unset with no port = start in-process |

The remote launcher sets these explicitly (`launch_table_bussing_remote.sh:380`):
```bash
CAP_CUROBO_HOST=127.0.0.1 CAP_CUROBO_PORT=8611 CAP_CUROBO_START_SERVER=0
```

### cuRobo S1 installation

`tmux/remote_serving/install_curobo_s1.sh` is run automatically before launching
(`launch_lecar_s1.sh:189`, `launch_table_bussing_remote.sh:272`). It:
1. Checks if cuRobo is already importable with CUDA support
2. If not, installs it via `uv pip install -e third_party/curobo --no-build-isolation`
3. Verifies `curobo.geom` is importable

### cuRobo reset between sessions

When a new client attaches, `launch_table_bussing_remote.sh:364` runs a cuRobo
reset that:
- Calls `health_check()`
- Disables finetune mode
- Clears the depth collision scene
- Clears debug collision balls
- Resets gripper qpos to `{left: 1.0, right: 1.0}`

---

## Roundtrip health checks

### Local client side (through SSH tunnel)
```bash
curl http://127.0.0.1:6767/health            # SAM3
curl http://127.0.0.1:8119/health            # BundleSDF
curl http://127.0.0.1:8122/health            # AnyGrasp
curl http://127.0.0.1:5173                   # CAP UI
curl http://127.0.0.1:8201/api/agent/options # agent bridge
```

### cuRobo Portal health check
```bash
# From S1 directly:
uv run python tools/remote/check_motion_planner_portal.py --port 8611

# From 4070 through SSH tunnel:
uv run python tools/remote/check_motion_planner_portal.py --host 127.0.0.1 --port 8611
```

The check script (`tools/remote/check_motion_planner_portal.py:22-47`) polls
`portal.Client.health_check()` and `set_gripper_qpos()` within a configurable
timeout (default 20s). Use `--quiet` for scripted checks.

### Reverse CAP tunnel check from S1
```python
import portal
client = portal.Client("127.0.0.1:18300")
client.get_state().result()
```

---

## Client-owned remote lifecycle

The client launcher (`launch_table_bussing_remote.sh`) owns the remote serving
session lifecycle:

### Startup sequence (`launch_table_bussing_remote.sh:258-273`)
1. Detect remote repo path via SSH (`detect_remote_repo()`, line 205)
2. Check if remote tmux session `lecar-s1-remote-serving` exists and is healthy
3. If healthy: reuse it. If exists but unhealthy: restart. If missing: start new.
4. Kill any stale local SSH tunnels and blocking ports
5. Start evaluation (launch.py), bridge, voice as background processes
6. Wait for follower arms to be ready (unless `--no-arms`)
7. Create local tmux session with cap_server, cap_agent, SSH tunnel, and UI panes
8. cap_agent pane waits for all remote health checks before starting

### Cleanup (`launch_table_bussing_remote.sh:282-294`)
On exit:
- If `--stop-remote-on-exit`: kills the remote serving tmux session
- Kills background bridge, voice, evaluation PIDs
- Kills local tmux session

---

## Launchers

### Original all-local stack
```bash
bash tmux/table_bussing/launch_table_bussing_local.sh
```

### Remote-model stack with the same local UI / agent flow
```bash
bash tmux/table_bussing/launch_table_bussing_remote.sh
```

## Common launch commands

### Local sim test
```bash
cd /home/lecar/Project/lecar-tbd
bash tmux/table_bussing/launch_table_bussing_local.sh --sim
```

### Remote real test
```bash
cd /home/lecar/Project/lecar-tbd
bash tmux/table_bussing/launch_table_bussing_remote.sh
```

### Remote sim smoke test
```bash
cd /home/lecar/Project/lecar-tbd
bash tmux/table_bussing/launch_table_bussing_remote.sh --sim --no-arms --no-browser
```

Both local and remote launchers proactively clear blocking client ports before
startup so repeated experiments can be relaunched cleanly without manual port
cleanup (`launch_table_bussing_remote.sh:134-155`,
`launch_table_bussing_local.sh:102-117`).

### Remote launcher CLI flags

| Flag | Effect | Ref |
|------|--------|-----|
| `--sim` | Run cap_server in MuJoCo sim mode | `launch_table_bussing_remote.sh:299` |
| `--viewer` | Add `--sim-viewer` to cap_server | line 300 |
| `--warp` | Add `--sim-warp` to cap_server | line 301 |
| `--no-arms` | Skip follower arm readiness wait | line 302 |
| `--no-ui` | Do not start cap/ui | line 303 |
| `--no-browser` | Do not auto-open the UI browser | line 304 |
| `--skip-bridge` | Do not start the local bridge process | line 305 |
| `--skip-voice` | Do not start the local voice process | line 306 |
| `--skip-evaluation` | Do not auto-start evaluation background process | line 307 |
| `--no-attach` | Do not attach tmux; keep the supervisor running | line 309 |
| `--ssh-target TARGET` | SSH target / alias for the S1 box | line 310 |
| `--remote-repo PATH` | Remote repo path on S1 (auto-detected when omitted) | line 311 |
| `--remote-session N` | Remote tmux session name on S1 | line 312 |
| `--curobo-port PORT` | Forward/use this remote cuRobo planner port | line 313 |
| `--skip-remote-start` | Do not auto-start the remote model-serving session | line 314 |
| `--stop-remote-on-exit` | Stop the remote model-serving session during cleanup | line 315 |
| `--restart-remote` | Stop and restart the remote serving session before attaching | line 317 |

### S1 launcher CLI flags (`launch_lecar_s1.sh`)

Services are positional args (default: `sam3 anygrasp bundlesdf curobo`):
```bash
bash tmux/remote_serving/launch_lecar_s1.sh sam3 bundlesdf  # only these two
```

| Flag | Effect |
|------|--------|
| `--sam3-port PORT` | Override SAM3 port |
| `--anygrasp-port PORT` | Override AnyGrasp port |
| `--bundlesdf-port PORT` | Override BundleSDF port |
| `--curobo-port PORT` | Override cuRobo port |
| `--cap-server-host HOST` | BundleSDF reverse-tunnel target host (default `127.0.0.1`) |
| `--cap-server-port PORT` | BundleSDF reverse-tunnel target port (default `18300`) |
| `--bundlesdf-camera NAME` | Camera name for BundleSDF (default `top`) |
| `--sam3-url URL` | SAM3 URL override for BundleSDF's internal SAM3 client |
| `--no-sam3-preload` | Do not preload SAM3 model at startup |
| `--no-wait-ready` | Do not block until all services are healthy |
| `--no-anygrasp-warmup` | Skip the synthetic AnyGrasp `/plan_viz` warmup |
| `--no-attach` | Do not attach to the tmux session |

---

## Persistent warm remote server behavior

The remote client launcher treats LeCAR-S1 as a persistent warm model host
(`launch_table_bussing_remote.sh:250-256`).

### Default behavior
- If the S1 serving session is already healthy, the client **reuses it**
- If it is missing or unhealthy, the client **starts it**
- When the local client exits, it **leaves the S1 serving session running**
- When a new client attaches, it **clears remote per-session state** before starting the agent

This keeps:
- SAM3 preloaded (weights stay in GPU memory)
- BundleSDF/SAM2 hot
- AnyGrasp warmed with one synthetic `/plan_viz` request on S1 startup

### Per-client reset behavior

When a new client session starts, `launch_table_bussing_remote.sh:366` sends
reset RPCs to each remote service:

| Service | Endpoint | Behavior | Source |
|---------|----------|----------|--------|
| SAM3 | `POST /reset_state` | GC + `torch.cuda.empty_cache()`, model stays loaded | `tools/vision/serve_sam3.py:311-322` |
| AnyGrasp | `POST /reset_state` | GC + `torch.cuda.empty_cache()`, model stays loaded | `tools/vision/serve_anygrasp.py:542-553` |
| BundleSDF | `POST /reset_state` | Clears all tracked detection sessions, removes registered objects from tracking loops, resets Portal connection, GC + empty_cache. Server process stays alive. | `tools/vision/serve_bundlesdf.py:1463-1487` |
| cuRobo | Portal RPC calls | `health_check()`, `set_finetune_enabled(False)`, `clear_depth_collision_scene()`, `clear_debug_collision_ball()`, `set_gripper_qpos({left: 1.0, right: 1.0})` | `launch_table_bussing_remote.sh:246-248` |

### Useful control flags

```bash
# Force-refresh the remote serving session before attaching
bash tmux/table_bussing/launch_table_bussing_remote.sh --restart-remote

# Stop the remote serving session when the local client exits
bash tmux/table_bussing/launch_table_bussing_remote.sh --stop-remote-on-exit

# Attach to an already-running remote serving session without starting it
bash tmux/table_bussing/launch_table_bussing_remote.sh --skip-remote-start
```

### Smoke test (lightweight)
```bash
bash tmux/table_bussing/launch_table_bussing_remote.sh --sim --no-arms --no-browser
```

---

## AnyGrasp server details

AnyGrasp requires host-specific runtime dependencies not in the repo:

| Dependency | Env var / location | Purpose |
|------------|-------------------|---------|
| OpenSSL 1.1 | `ANYGRASP_OPENSSL11_DIR` or `$RUNTIME_DEPS_ROOT/libssl11/...` | Legacy linked library |
| OpenBLAS | `OPENBLAS_HOME` or `$RUNTIME_DEPS_ROOT/openblas/lib` | Linear algebra |
| CUDA toolkit | `CUDA_HOME` or `/usr/local/cuda-12.8` | GPU kernels |

These are resolved in `tools/vision/launch_anygrasp_server.sh:82-131`.

The AnyGrasp launch script also:
- Resolves LFS pointer files for MinkowskiEngine and pointnet2 extensions (line 196-197)
- Resolves the checkpoint and license from repo-relative paths or env overrides (line 199-216)
- Kills any existing listener on the port before starting (line 248)

---

## Setup: first-time remote host configuration

### 1. Configure runtime env on S1

```bash
ssh LeCAR_4xRTX6000BlackWell_97GB
cd /path/to/lecar-tbd
bash tmux/remote_serving/configure_runtime_env.sh \
  --tmp-root /usr0/tonghez/tmp \
  --deps-root ~/runtime-deps \
  --hf-offline 1 \
  --yes
```

This writes `~/.config/lecar-tbd/runtime_env.sh` and adds a source line to `~/.bashrc`.

### 2. Sync Git LFS assets to both machines

```bash
# From the client (4070):
bash tmux/remote_serving/sync_lecar_s1_assets.sh
```

### 3. Sync HF model caches to S1

```bash
# From the client (4070):
bash tmux/remote_serving/sync_remote_model_cache.sh
```

### 4. Install cuRobo on S1

```bash
# Done automatically by launchers, or manually:
ssh LeCAR_4xRTX6000BlackWell_97GB
cd /path/to/lecar-tbd
bash tmux/remote_serving/install_curobo_s1.sh
```

### 5. Launch

```bash
# From the client (4070):
bash tmux/table_bussing/launch_table_bussing_remote.sh
```

---

## Tmux session layout

### Remote session on S1: `lecar-s1-remote-serving`

```
┌───────────┬───────────┬───────────┬───────────┐
│ sam3      │ anygrasp  │ bundlesdf │ curobo    │
│ :6767     │ :8122     │ :8119     │ :8611     │
└───────────┴───────────┴───────────┴───────────┘
```

### Local session: `table-bussing-remote`

```
┌────────────────────┬────────────────────┐
│ cap_server         │ cap_agent          │
│ :8300              │ :8200              │
├────────────────────┼────────────────────┤
│ ssh tunnel         │ cap_ui             │
│ -L/-R forwarding   │ :5173              │
└────────────────────┴────────────────────┘
```

Background processes (not in tmux): bridge (:8201), voice (:8202), evaluation.
Logs: `logs/remote_serving/bridge.log`, `logs/remote_serving/voice.log`,
`logs/remote_serving/evaluation.log`.

### All-local session: `table-bussing-anygrasp-bundlesdf`

```
┌────────────────────┬────────────────────┬────────────────────┐
│ cap_server         │ cap_agent          │ cap_ui             │
│ :8300              │ :8200              │ :5173              │
├────────────────────┼────────────────────┼────────────────────┤
│ serve_sam3         │ serve_anygrasp     │ serve_bundlesdf    │
│ :6767              │ :8122              │ :8119              │
└────────────────────┴────────────────────┴────────────────────┘
```
