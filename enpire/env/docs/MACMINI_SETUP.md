# Mac Mini Setup — CAP Sim Mode

Run the CAP simulation stack on an Apple Silicon Mac Mini (no real hardware needed).

> **Related docs:**
> - [`docs/CAP_DESIGN.md`](CAP_DESIGN.md) — Full CAP system architecture and layer stack
> - [`docs/remote_serving.md`](remote_serving.md) — Remote model server split (SAM3, BundleSDF, AnyGrasp, cuRobo on LeCAR-S1)
> - [`docs/RL_PIPELINE_DESIGN.md`](RL_PIPELINE_DESIGN.md) — RL training pipeline (`serve_rl_policy`, `learn_skill`, diagnostics)
> - [`docs/TABLE_BUSSING_SKILLS.md`](TABLE_BUSSING_SKILLS.md) — Table bussing skill tools (tracking, freespace move, nudge, gripper)
> - [`docs/launch_sim_diagram.svg`](launch_sim_diagram.svg) — Visual diagram of sim launch pane layout and Mac Mini legend

---

## Prerequisites

- macOS with Apple Silicon (M-series)
- Python 3.11 (`brew install python@3.11` or via pyenv)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- Node.js (for cap_ui dev server: `brew install node`)
- tmux (`brew install tmux`)

## Install

```bash
cd ~/Project/enpire
uv sync --only-group macmini
```

This installs only what CAP sim needs — no torch, transformers, pyrealsense2, gear, or CUDA packages.

The `macmini` dependency group is defined in `pyproject.toml:188-210` and includes:

| Package | Purpose |
|---|---|
| `i2rt` | Robot interface |
| `gymnasium==1.2.0` | Gym environment |
| `mink==0.0.12`, `pin-pink` | IK solver |
| `opencv-python==4.12.0.88` | Image processing |
| `viser==1.0.6` | 3D visualization |
| `pyzmq==27.1.0` | ZeroMQ messaging |
| `fastapi>=0.115`, `uvicorn>=0.34` | CAP agent HTTP server |
| `msgpack>=1.0` | Serialization |
| `pynput>=1.7` | Keyboard input |
| `numpy>2.0.0`, `Pillow`, `requests` | Core utilities |
| `pyroki`, `yourdfpy==0.0.59` | Robot kinematics / URDF parsing |
| `damiao-motor>=1.0.7b1` | Fello/YAM motor driver (with macOS echo-frame fix) |
| `tyro` | CLI argument parsing |

---

## Launch — Sim Mode

The primary sim launcher lives at `tmux/table_bussing/table_bussing_history/launch_sim.sh`.

```bash
./tmux/table_bussing/table_bussing_history/launch_sim.sh
```

The `--mac-mini` flag is **auto-detected** on macOS (`launch_sim.sh:78`: `if [[ "$(uname -s)" == "Darwin" ]]`). When active, it:

- **Skips `MUJOCO_GL=egl`** — macOS uses CGL natively (`launch_sim.sh:102-104`)
- **Omits BundleSDF host args** from `cap_agent.py` — oracle detection is used instead (`launch_sim.sh:111-113`)
- **Defaults `--rl-host` to `192.0.2.2`** (direct Ethernet to GPU box) (`launch_sim.sh:83-84`)
- **Launches only the right Fello server** when `--use-fello` is passed (`launch_sim.sh:146-150`)

### Flags

```bash
./tmux/table_bussing/table_bussing_history/launch_sim.sh --viewer       # with MuJoCo passive viewer
./tmux/table_bussing/table_bussing_history/launch_sim.sh --use-fello    # with Fello HIL takeover (launches right Fello server)
./tmux/table_bussing/table_bussing_history/launch_sim.sh --mac-mini     # explicit (auto-detected on Darwin)
./tmux/table_bussing/table_bussing_history/launch_sim.sh --rl-host IP   # override RL policy server host
```

### Tmux session layout (`cap-sim`)

```
┌────────────────┬────────────────┐
│  cap_server     │  cap_agent     │
│  (--sim)        │                │
├────────────────┼────────────────┤
│  ssh tunnel     │  ssh tunnel    │
│  SAM3 (:6767)   │  BundleSDF     │
│  → gpu-host     │  (:8119)       │
├────────────────┼────────────────┤
│  reward_server  │  cap_ui (dev)  │
│  (constant-0)   │  :5173         │
├────────────────┴────────────────┤
│  fello_server --side right      │
│  (only with --use-fello)        │
└─────────────────────────────────┘
```

Attach: `tmux attach -t cap-sim`

The cap_ui dev server auto-opens in the browser once ready. The SSH tunnels forward SAM3 and BundleSDF ports from `gpu-host` (`launch_sim.sh:124-129`).

---

## Platform-Specific Code

macOS (Darwin) platform detection drives several behaviors across the codebase:

### CAN Bus Configuration

**`robot/constants.py:6`** — Bus type selection:
```python
CAN_BUSTYPE = "gs_usb" if sys.platform == "darwin" else "socketcan"
```

**`robot/constants.py:15-24`** — macOS CAN interface names use USB serial numbers instead of Linux kernel interface names:
```python
if sys.platform == "darwin":
    RIGHT_LEADER_CAN_INTERFACE = "0046002E594E501820313332"
    LEFT_LEADER_CAN_INTERFACE = "ABCDE"   # TODO: set when left leader gs_usb connected
    LEFT_FOLLOWER_CAN_INTERFACE = "ABCDEF"  # TODO: set when left follower gs_usb connected
    RIGHT_FOLLOWER_CAN_INTERFACE = "ABCDEFG"  # TODO: set when right follower gs_usb connected
```

To discover USB serial numbers on macOS:
```python
python -c "from gs_usb.gs_usb import GsUsb
for i,d in enumerate(GsUsb.scan()): print(i, d.serial_number)"
```

### Station Profiles

**`robot/station_profiles.py`** — Multi-machine CAN profile system with per-platform support:
- `station_profiles.py:35` — Profile table maps `(linux StationCAN, darwin StationCAN)` tuples
- `station_profiles.py:95-97` — `_default_darwin()` returns macOS CAN serial numbers
- `station_profiles.py:177-178` — `active_station_can()` selects profile by `sys.platform`
- Resolution order (`station_profiles.py:7-11`):
  1. Env var `LECAR_STATION`
  2. Gitignored `robot/local_station.toml` with `station = "<name>"`
  3. `STATION_BY_HOSTNAME` table
  4. `"default"` profile

### Fello Arm — macOS-Specific Behavior

**`robot/fello/fello.py:33-66`** — `_reset_gs_usb_device()`: USB-CAN device reset for unclean shutdowns. Only runs on macOS (`fello.py:35`: `if sys.platform != "darwin": return`). Called at connect (`fello.py:186`) and reconnect (`fello.py:342`).

**`robot/fello/fello.py:579`** — Inter-motor command delay:
```python
_INTER_MOTOR_DELAY_S = 0.0005 if sys.platform == "darwin" else 0.0
```
On macOS gs_usb, each `bus.recv()` takes >=1ms, so without spacing the read pipeline falls behind for higher-numbered motors.

**`robot/fello/fello_server.py:414-427`** — CAN channel resolution: on macOS, resolves the CAN channel from `constants.py` serial numbers instead of requiring `--can-interface` CLI arg.

### DaMiao Motor Driver

**`third_party/damiao_motor_pkg/damiao_motor/core/controller.py:13`** — Default bus type:
```python
_DEFAULT_BUSTYPE = "gs_usb" if sys.platform == "darwin" else "socketcan"
```

**`controller.py:16-55`** — `_patch_gs_usb_for_macos()`: Patches `GsUsb.start` to handle macOS kernel driver detach gracefully. Auto-applied on import when `sys.platform == "darwin"` (`controller.py:54-55`).

The damiao-motor package (>=1.0.7b1) includes the **echo frame filtering fix** that eliminates macOS-only jitter.

### System ID Tools

**`robot/systemid/collect_torque_data.py:293`** and **`robot/systemid/torque_calibrator.py:129`** — Both default `--bustype` to `gs_usb` on Darwin, `socketcan` on Linux.

### Bringup Dashboard

**`bringup/system_runtime.py:890`** — Browser opener:
```python
opener = "open" if os.uname().sysname == "Darwin" else "xdg-open"
```

---

## Fello HIL (Human-in-the-Loop)

With `--use-fello`, the launch script starts the right Fello leader server (only right is physically connected on Mac Mini). The CAN channel is auto-resolved from `robot/constants.py` using the gs_usb USB serial number.

**Current state (`robot/constants.py:16-24`):**
- Right leader: `0046002E594E501820313332` (configured, working)
- Left leader: `ABCDE` (placeholder — not yet connected)
- Left follower: `ABCDEF` (placeholder)
- Right follower: `ABCDEFG` (placeholder)

On Linux, the fello server requires explicit `--can-interface` and `--port` args. On macOS, the `launch_sim.sh:150` path just calls `fello_server.py --side right` and lets it auto-resolve.

### Known Fello/macOS Issues

Key items:
- gs_usb replug sometimes needed after unclean shutdown (auto-reconnect tries USB reset first — `fello.py:33-66`)
- Higher CAN arbitration IDs (motor 7 / gripper) may have lower priority — bandwidth still under investigation
- Echo frame filtering (jitter fix) is in `damiao-motor>=1.0.7b1` (`pyproject.toml:208`)

---

## Network — RL Policy Server

The Mac Mini connects to the GPU machine running `rl_policy_server` via direct Ethernet:

| Connection | IP | Latency |
|---|---|---|
| Direct Ethernet (current) | `192.0.2.2` | ~2ms RTT |
| Lab LAN (old) | `192.0.2.216` | ~5-10ms RTT |

The RL host is set in `launch_sim.sh:83-84` as the Mac Mini default. Override with:
```bash
./tmux/table_bussing/table_bussing_history/launch_sim.sh --rl-host <IP>
```

See `docs/RL_PIPELINE_DESIGN.md` for the full RL server architecture.

---

## Network — Remote Model Serving (LeCAR-S1)

GPU-heavy perception services (SAM3, BundleSDF, AnyGrasp, cuRobo) run on a remote GPU server. The sim launch script creates SSH tunnels for:

| Service | Local Port | Remote (S1) Port |
|---|---|---|
| SAM3 | `localhost:6767` | `S1:6767` |
| BundleSDF | `localhost:8119` | `S1:8119` |

In oracle/sim mode, BundleSDF is not strictly needed (MuJoCo ground-truth positions are used). The SSH tunnels are set up for SAM3 VLM queries and optional real-detection testing.

For the full remote serving architecture and port map, see `docs/remote_serving.md`.

**Key remote scripts:**
- `tmux/remote_serving/launch_remote_gpu.sh` — Starts SAM3, AnyGrasp, BundleSDF, cuRobo on S1
- `tmux/table_bussing/launch_table_bussing_remote.sh` — Lightweight local client + SSH tunnel to S1
- `tmux/remote_serving/configure_runtime_env.sh` — Machine-specific runtime env config

---

## Running Saved Scripts

Once the tmux session is up (cap_server, cap_agent, reward_server, cap_ui all running),
open the CAP UI at `http://localhost:5173` and execute scripts via oracle mode. Example:

```
cap/saved_scripts/examples/pick_cube.py       # example pick-and-place
cap/saved_scripts/test_freespace_move.py      # YAM cuRobo free-space move
```

---

## Alternative Launch Scripts

The sim launch script above is the simplest Mac Mini entry point. For more advanced setups:

| Script | Purpose | macOS Support |
|---|---|---|
| `tmux/table_bussing/table_bussing_history/launch_sim.sh` | Sim mode with Mac Mini auto-detect | Yes (primary) |
| `tmux/table_bussing/launch_table_bussing_local.sh` | Full local table bussing (all perception local) | Linux only (needs CUDA) |
| `tmux/table_bussing/launch_table_bussing_remote.sh` | Local client + remote S1 perception servers | Linux client only |
| `tmux/cap_agent_interface/start_cap_agent_interface.sh` | Start/stop agent interface daemon | Linux (uses `.venv/bin/python`) |

---

## Log Sync Utility

**`scripts/sync_grip_logs.sh`** — Syncs HIL gripper debug logs from `<robot-host>` to the local Mac Mini. Runs `rsync` every 2 seconds:

```bash
./scripts/sync_grip_logs.sh    # Ctrl-C to stop
```

---

## Troubleshooting

### MuJoCo GL errors
macOS uses CGL natively. If you see `MUJOCO_GL` errors, ensure you are not manually setting `MUJOCO_GL=egl` — the launch script skips this on Darwin automatically.

### CAN bus "device busy" / replug needed
The `_reset_gs_usb_device()` function (`robot/fello/fello.py:33`) attempts a USB device reset before each connection. If that fails, physically replug the USB-CAN adapter.

### Fello jitter
Ensure `damiao-motor>=1.0.7b1` is installed (check with `uv pip show damiao-motor`). The echo frame filtering fix in this version eliminates macOS-specific jitter.

### Missing left Fello / follower CAN interfaces
The left leader, left follower, and right follower CAN interfaces are placeholder values in `robot/constants.py:18-24`. Set them to the actual USB serial numbers once the hardware is connected.

### Node.js / cap_ui build issues
The cap_ui dev server runs in the tmux session at `cap/ui/`. If `npm run dev` fails, try:
```bash
cd cap/ui && npm install && npx vite dev
```
