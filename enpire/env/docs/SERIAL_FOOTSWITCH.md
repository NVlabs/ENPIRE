# Serial Footswitch (Waveshare RP2040-Zero)

3-button USB serial device used as the primary button input for Fello teleop.
Supports three backends (serial, evdev, keyboard) behind a common `ButtonMonitor`
protocol, so all downstream consumers are backend-agnostic.

## Hardware

- **MCU**: Waveshare RP2040-Zero (CircuitPython)
- **USB**: Vendor `2e8a`, Product `101f`
- **Interface**: CDC ACM serial (`/dev/ttyACM*`)
- **Buttons**: 3 buttons on GPIO 27, 28, 29 (active-low, internal pull-up)
  (`hardware/serial_footswitch_firmware/code.py:5`)
- **Protocol**: Streams `[0, 0, 0]`-style JSON lines at **50 Hz** over USB CDC serial (115200 baud)
  (`hardware/serial_footswitch_firmware/code.py:15-16`)
- **Boot config**: Enables both CDC console and data channels
  (`hardware/serial_footswitch_firmware/boot.py:3`)

### Firmware

The firmware is pure CircuitPython. Source lives in `hardware/serial_footswitch_firmware/`:

| File | Purpose |
|------|---------|
| `code.py` | Main loop -- reads 3 GPIO pins, prints JSON array every 20 ms (50 Hz) |
| `boot.py` | Enables USB CDC console + data (`usb_cdc.enable(console=True, data=True)`) |

**Flashing**: See `hardware/serial_footswitch_firmware/README.md`. The board mounts as `CIRCUITPY`;
copy `boot.py` and `code.py` to the mount. CircuitPython auto-reloads on save.

## Architecture

### Data flow (serial backend)

```
 RP2040 firmware (50 Hz JSON lines)
        │
        ▼
 /dev/ttyACM* ──► SerialButtonHub._read_loop()     ← daemon thread, reads serial lines
                   │   parses JSON, updates per-button pressed state
                   │   min_hold_seconds latch (150 ms) keeps brief taps visible
                   │
                   ├──► SerialButtonMonitor(button_map[0])  →  buttons[0] = save
                   ├──► SerialButtonMonitor(button_map[1])  →  buttons[1] = takeover
                   └──► SerialButtonMonitor(button_map[2])  →  buttons[2] = start
                              │
                              ▼
                   FelloLeaderRobot.get_info()  →  (qpos, buttons[3])
                              │
                              ▼
                   FelloLeaderRobotServer._get_info()   ← Portal RPC
                              │
                              ▼
                   FelloTeleopPolicy.get_action()
                     buttons[0] → save_pressed    (edge-detected)
                     buttons[1] → takeover         (level: held = gravity mode)
                     buttons[2] → start_pressed   (edge-detected)
                              │
                              ▼
                   DualFelloTeleopPolicy  →  HILPolicyWrapper  →  control loop
```

### ButtonMonitor protocol

All three backends implement the same protocol
(`experimental/footswitch.py:20-28`):

```python
@runtime_checkable
class ButtonMonitor(Protocol):
    def is_pressed(self) -> bool: ...
    @property
    def device_path(self) -> str: ...
    def close(self) -> None: ...
```

### Backend implementations

| Class | File:Line | Input source | When used |
|-------|-----------|--------------|-----------|
| `SerialButtonMonitor` | `experimental/footswitch.py:424-444` | RP2040 serial JSON | `type: serial` in config |
| `FootSwitchMonitor` | `experimental/footswitch.py:31-208` | evdev `/dev/input/*` | `device_0`/`device_1`/`device_2` present, no `type` |
| `KeyboardFootSwitchMonitor` | `experimental/footswitch.py:210-291` | pynput global key listener | `keyboard_keys` present, no `type`, DISPLAY available |

`SerialButtonMonitor` is a thin per-button view into a shared `SerialButtonHub`
(`experimental/footswitch.py:293-422`). The hub opens the serial port once, runs a
background `_read_loop` thread, and exposes `get_button(index)`. A reference-counting
`acquire()`/`release()` pattern lets multiple monitors share one hub.

### Dispatch logic

`FelloLeaderRobot.__init__()` (`robot/fello/fello.py:710-780`) performs a 3-way dispatch:

1. **`type: serial`** (line 715) -- imports `SerialButtonHub`/`SerialButtonMonitor`, reads
   `serial_port`, `button_map`, `serial_baudrate` from config.
2. **evdev** (line 745) -- uses `FootSwitchMonitor` when `device_0`/`device_1`/`device_2`
   paths all exist on disk.
3. **keyboard** (line 753) -- uses `KeyboardFootSwitchMonitor` when `keyboard_keys` is set
   and a GUI session (`DISPLAY`/`WAYLAND_DISPLAY`) is available.

### get_info() output

`FelloLeaderRobot.get_info()` (`robot/fello/fello.py:782-792`) returns:

```python
(qpos: np.ndarray[7],  buttons: np.ndarray[3])
#  buttons = [save, takeover, start]  — each 0.0 or 1.0
```

The server wraps this via `FelloLeaderRobotServer._get_info()`
(`robot/fello/fello_server.py:133-146`), which maps joint positions
through arm/gripper sign conventions before publishing over Portal RPC.

### Downstream consumers

| Consumer | File:Line | What it reads |
|----------|-----------|---------------|
| `FelloTeleopPolicy.get_action()` | `robot/fello/fello_teleop_policy.py:316-352` | `buttons[0]` = save (edge), `buttons[1]` = takeover (level), `buttons[2]` = start (edge) |
| `FelloTeleopPolicy.poll_button_events()` | `robot/fello/fello_teleop_policy.py:354-371` | Same edge detection for save/start without commanding the arm |
| `DualFelloTeleopPolicy` | `robot/fello/fello_teleop_policy.py:385-466` | Merges left+right `save_pressed`/`start_pressed` events, ORs `footswitch_pressed` |
| `HILPolicyWrapper` | `experimental/hil_policy.py:16-240` | Reads `left_trigger`/`right_trigger` for per-arm takeover; forwards `save_pressed`/`start_pressed` to outer info |
| `yam_control_loop` | `experimental/yam_control_loop.py:1558-1569` | Reads `start_pressed` and `save_pressed` from policy info for manual episode recording |

### Button semantics

The 3-element button array uses a fixed logical ordering: `[save, takeover, start]`.
`button_map` remaps from hardware-specific raw serial indices to these logical slots.

| Logical slot | Index | FelloTeleopPolicy behavior | Edge/Level |
|-------------|-------|---------------------------|------------|
| save | 0 | `info["save_pressed"] = True` on rising edge | edge |
| takeover | 1 | Gravity mode while held; position mode on release | level |
| start | 2 | `info["start_pressed"] = True` on rising edge | edge |

## Configuration

### Config file loading

Fello code loads config through `robot/fello/fello_config.py::load_fello_config(side=...)`,
which reads the base **`robot/models/fello/fello_config.yaml`** and then deep-merges
per-station overrides from `robot/device/{station}.yaml` under `fello.{side}` (the
station key is resolved via `robot/station_profiles.py::resolve_station_key()`).

The legacy per-handle configs in `robot/models/fello_left/` and `robot/models/fello_right/`
are reference/legacy copies and are NOT the source of truth on stations with device
overrides. On the `tony` station, for example, `robot/device/tony.yaml` sets
`fello.left.button_map = [2, 0, 1]` and `fello.right.button_map = [2, 1, 0]  # [home, pause, start]`.

### Config schema

In `robot/models/fello_{left,right}/fello_config.yaml`:

```yaml
hardware:
  footswitch:
    type: serial                       # "serial" | "keyboard" | "evdev"
    serial_port: /dev/serial/by-id/usb-Waveshare_Electronics_RP2040-Zero_<SERIAL>-if00
    serial_baudrate: 115200            # default 115200, rarely needs change
    button_map: [0, 1, 2]             # raw serial CSV index → logical [save, takeover, start]
    # -- ui_control fields (config-only, NOT consumed by Python code yet) --
    # button_mode: ui_control          # aspirational: buttons = start / pause / home
    # ui_button_map: [0, 1, 2]        # aspirational: logical slots → [start, pause, home]
```

### Current per-handle configs

**Left handle** (`robot/models/fello_left/fello_config.yaml:11-16`):
```yaml
footswitch:
  type: serial
  button_map: [2, 0, 1]  # measure for your station
  serial_port: /dev/serial-left-buttons
```

**Right handle** (`robot/models/fello_right/fello_config.yaml:11-16`):
```yaml
footswitch:
  type: serial
  serial_port: /dev/serial-right-buttons
  button_map: [0, 2, 1]  # measure for your station
  button_mode: ui_control
  ui_button_map: [0, 2, 1]
```

**Legacy single-handle** (`robot/models/fello/fello_config.yaml:11-15`):
```yaml
footswitch:
  device_0: /dev/input/footswitch_0  # evdev save
  device_1: /dev/input/footswitch_1  # evdev takeover
  device_2: /dev/input/footswitch_2  # evdev start
  keyboard_keys: ["[", "]", ","]     # keyboard fallback
```

### Auto-detection fallback

When `type` is omitted, the dispatch in `FelloLeaderRobot.__init__()` preserves backward compatibility:
- `keyboard_keys` present + display available -> keyboard mode (pynput)
- `device_0`/`device_1`/`device_2` all exist on disk -> evdev mode
- Neither -> raises `ValueError`

### Serial port auto-detection

When `serial_port` is omitted, `SerialButtonHub._find_serial_device()`
(`experimental/footswitch.py:337-356`) tries:
1. Scan `/dev/serial/by-id` for entries containing `2e8a` or `rp2040`.
2. Fallback: scan `/sys/class/tty/ttyACM*` for matching vendor/product in `uevent`.

**NOTE**: When two RP2040 boards are connected, auto-detection returns whichever board
is found first. Always set `serial_port` explicitly in bimanual setups.

### Right-pedal UI control (implemented)

The right pedal's `[home, pause, start]` semantics are **hardcoded** in
`FelloTeleopPolicy.get_action()` / `poll_button_events()`
(`robot/fello/fello_teleop_policy.py:361-374, 395-407`). `DualFelloTeleopPolicy`
(`:466-475`) forwards `ui_home`/`ui_pause`/`ui_start` and populates
`right_button_states`; `HILPolicyWrapper._get_action`
(`experimental/hil_policy.py:201-206`) propagates them to outer info.

The YAML fields `button_mode` and `ui_button_map` are currently documentation-only —
no Python code reads them. Button-slot semantics are fixed per side:
left = `[save, takeover, start]`, right = `[home, pause, start]`.

## Setup

### Quick setup (single board)

1. Install udev rule for stable symlink and permissions:

   ```bash
   sudo cp hardware/99-serial-footswitch.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

   Or use the install script:

   ```bash
   bash scripts/setup/install_serial_footswitch.sh
   ```

   The udev rule (`hardware/99-serial-footswitch.rules:9-11`) matches `ttyACM*` devices
   with vendor `2e8a` and product `101f`, creates a `/dev/serial-footswitch` symlink,
   sets mode `0666`, and sets `ID_MM_DEVICE_IGNORE=1` to prevent ModemManager from
   grabbing the port.

2. Verify: `ls -la /dev/serial-footswitch`

3. Set `type: serial` in `robot/models/fello/fello_config.yaml` under `hardware.footswitch`.

### Bimanual setup (two boards)

Each RP2040 board has a unique USB serial number. Use `/dev/serial/by-id/` paths
(which encode the serial number) rather than `/dev/ttyACM*` (which can change
across reboots). Set each handle's `serial_port` to the full by-id path.

**NOTE**: The single `/dev/serial-footswitch` symlink from the udev rule only resolves
to one device. For bimanual setups, use the by-id paths directly.

### Firmware flashing

See `hardware/serial_footswitch_firmware/README.md` for full instructions:
1. Hold BOOT button, plug in USB -> mounts as `RPI-RP2`.
2. Flash CircuitPython UF2 -> reboots as `CIRCUITPY`.
3. Copy `hardware/serial_footswitch_firmware/{boot.py,code.py}` to `CIRCUITPY`.
4. Verify rate: `uv run scripts/setup/measure_serial_rate.py` (expect ~50 Hz).

## Debugging

### Debugging multiple boards

When two RP2040 button boards are connected, treat them as independent devices:
- The physical left/right handle assignment comes from `serial_port`.
- The raw button ordering comes from `button_map`.
- Do not assume both boards share the same `button_map`.

### Layer-by-layer debugging

Debug from raw hardware up through the stack:

**Layer 1 -- Raw serial** (is the RP2040 sending data?):

```bash
uv run scripts/setup/watch_serial_buttons.py
uv run scripts/setup/watch_serial_buttons.py --port /dev/ttyACM2 --port /dev/ttyACM4
```

The watcher (`scripts/setup/watch_serial_buttons.py`) auto-discovers RP2040 devices,
opens each as a serial port, and prints per-port edge events. Press only the left handle,
then only the right handle, to map physical handles to serial ports.

**Layer 2 -- Fello leader server** (is the server publishing buttons via Portal?):

```bash
uv run scripts/setup/watch_fello_leader_buttons.py
```

This script (`scripts/setup/watch_fello_leader_buttons.py`) connects to the left and
right Fello leader Portal servers and polls `get_info()`. If raw serial works but this
watcher shows no changes, the problem is in the `FelloLeaderRobot` or server-side path.

**Layer 3 -- Teleop policy** (is the teleop policy consuming buttons?):

If the leader server changes but the evaluation loop ignores the button, the problem is
in `FelloTeleopPolicy.get_action()` (`robot/fello/fello_teleop_policy.py:316-352`) or
downstream in `HILPolicyWrapper` / the control loop.

### Measuring button_map for a board

```bash
uv run scripts/setup/test_serial_footswitch.py --port /dev/ttyACM2
uv run scripts/setup/test_serial_footswitch.py --port /dev/ttyACM4
```

The tool (`scripts/setup/test_serial_footswitch.py`) interactively prompts you to press
each button, detects which raw serial index fires, and outputs the `button_map` in
canonical `[save, takeover, start]` order.

To probe in the human workflow order (start -> takeover -> save) while still getting
canonical output:

```bash
uv run scripts/setup/test_serial_footswitch.py --port /dev/ttyACM2 --prompt-order start,takeover,save
```

### Measuring serial rate

```bash
uv run scripts/setup/measure_serial_rate.py
uv run scripts/setup/measure_serial_rate.py --port /dev/ttyACM2 --duration 5
```

The firmware targets 50 Hz. If measured rate is below 15 Hz, brief taps may be missed.
The 150 ms min-hold latch in `SerialButtonHub` compensates for moderate jitter.

### Verifying raw serial output

```bash
sudo stty -F /dev/serial-footswitch 115200 raw -echo && cat /dev/serial-footswitch
```

Press each button -- you should see `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]`.

## Files

### Core implementation

| File | Key contents | Lines |
|------|-------------|-------|
| `experimental/footswitch.py` | `ButtonMonitor` protocol, `FootSwitchMonitor`, `KeyboardFootSwitchMonitor`, `SerialButtonHub`, `SerialButtonMonitor` | 1-445 |
| `robot/fello/fello.py` | `FelloLeaderRobot.__init__()` 3-way backend dispatch, `get_info()` | 699-792 |
| `robot/fello/fello_config.py` | `load_fello_config()`, `get_config_value()` -- loads `robot/models/fello/fello_config.yaml` | 1-26 |
| `robot/fello/fello_server.py` | `FelloLeaderRobotServer._get_info()` -- Portal RPC wrapper | 57-146 |
| `robot/fello/fello_teleop_policy.py` | `FelloTeleopPolicy` button consumption, `DualFelloTeleopPolicy` bimanual merge | 45-466 |
| `experimental/hil_policy.py` | `HILPolicyWrapper` -- forwards save/start events, per-arm takeover | 16-240 |

### Configuration

| File | Purpose |
|------|---------|
| `robot/models/fello/fello_config.yaml` | **Active config** loaded by Python runtime (legacy single-handle, evdev/keyboard) |
| `robot/models/fello_left/fello_config.yaml` | Left handle serial config (reference; not auto-loaded) |
| `robot/models/fello_right/fello_config.yaml` | Right handle serial config with `button_mode: ui_control` (reference; not auto-loaded) |

### Setup and debug scripts

| File | Purpose |
|------|---------|
| `scripts/setup/install_serial_footswitch.sh` | Copies udev rule, reloads udev, verifies device |
| `scripts/setup/watch_serial_buttons.py` | Watch raw serial edges from one or more RP2040 boards |
| `scripts/setup/watch_fello_leader_buttons.py` | Watch button states published by running Fello leader servers via Portal |
| `scripts/setup/test_serial_footswitch.py` | Interactive tool to measure `button_map` for one board |
| `scripts/setup/measure_serial_rate.py` | Measure serial update rate (expect ~50 Hz) |

### Hardware and firmware

| File | Purpose |
|------|---------|
| `hardware/99-serial-footswitch.rules` | udev rule: symlink + permissions + ModemManager ignore |
| `hardware/serial_footswitch_firmware/code.py` | CircuitPython main: reads GPIO 27/28/29, prints JSON at 50 Hz |
| `hardware/serial_footswitch_firmware/boot.py` | Enables USB CDC console + data channels |
| `hardware/serial_footswitch_firmware/README.md` | Flashing instructions for RP2040-Zero |
| `hardware/README.md` | General hardware setup (includes serial footswitch quick-setup section) |

## Cross-references

- **CAP design**: `docs/CAP_DESIGN.md` -- CAP agent framework that consumes teleop actions
- **RL pipeline**: `docs/RL_PIPELINE_DESIGN.md` -- RL training pipeline with HIL takeover
- **CAP UI**: `docs/CAP_UI_DESIGN.md` -- web UI that may consume `ui_start`/`ui_pause`/`ui_home` events
- **Table bussing skills**: `docs/TABLE_BUSSING_SKILLS.md` -- bimanual task skills using Fello teleop
- **Hardware setup**: `hardware/README.md` -- serial footswitch section under "Serial footswitch"
- **Firmware README**: `hardware/serial_footswitch_firmware/README.md` -- flashing and verification
