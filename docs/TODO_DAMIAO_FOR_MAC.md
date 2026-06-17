# Damiao Motor macOS (gs_usb) Support

DEBUG and fix the communication layer for Damiao motors on macOS. The Fello leader arm connects to a Mac Mini via a USB-CAN adapter using the gs_usb driver (no SocketCAN on macOS).

## Related docs

- `docs/MACMINI_SETUP.md` — Mac Mini CAP sim setup, `--use-fello` flag, gs_usb CAN channel auto-resolution
- `docs/CAP_DESIGN.md` — CAP system architecture (fello_server is a dependency of the control loop)
- `docs/CAP_SYSTEM_DASHBOARD.md` — System bringup dashboard (fello_server launch status)
- `docs/TABLE_BUSSING_SKILLS.md` — Gripper tool skills that depend on Fello arm feedback
- `docs/SERIAL_FOOTSWITCH.md` — Footswitch input wired through fello_server

## File index

| File | Role |
|------|------|
| `robot/constants.py` | Platform-aware CAN bus type and interface constants |
| `robot/fello/fello.py` | Core Fello robot class — gs_usb reset, connect, reconnect, gravity comp |
| `robot/fello/fello_server.py` | Portal RPC server wrapping FelloRobot, control loop with auto-reconnect |
| `robot/fello/fello_config.py` | YAML config loader (`robot/models/fello/fello_config.yaml`) |
| `robot/yam/yam_controller.py` | YAM follower arm — uses `DaMiaoController` directly |
| `robot/yam/arm_server.py` | YAM arm Portal RPC server (follower + leader modes) |
| `robot/monitor_motor_temps.py` | Live motor temp display (reads from arm servers via Portal RPC) |
| `third_party/damiao_motor_pkg/` | Local copy of damiao-motor (v1.0.6.post1) — **superseded by PyPI 1.0.7b1** |
| `third_party/i2rt/i2rt/motor_drivers/dm_driver.py` | i2rt's own DaMiao driver (used by YAM **leader** arm only) |
| `third_party/i2rt/i2rt/motor_drivers/can_interface.py` | i2rt's CAN interface wrapper |
| `hardware/systemd/canable2-slcan-*.service` | Linux systemd units for CANable2 slcan setup |
| `pyproject.toml` | Declares `damiao-motor>=1.0.7b1` from PyPI |

## TODO

- [x] Monitor in and out traffic at USB2CAN driver level
- [x] Cross check with fello_server logging to verify the result
- [x] Fix motor fault auto-recovery
- [x] Fix gs_usb echo frame contamination (jitter root cause)
- [x] Verify jittering is resolved after echo frame fix
- [ ] End-to-end test: fello tracks sim arm smoothly for 5+ minutes
- [ ] restart fello server to verify
- [ ] repeat if error happens

## Known issues

- Higher arbitration ID has lower CAN priority — motor 7 (gripper, ID `0x07`) could be throttled under heavy bus load. CAN stats (`DaMiaoController.get_can_stats()` at `controller.py:412`) provide TX/RX counts to verify.
- gs_usb may not cleanly recover after unclean shutdown. `_reset_gs_usb_device()` (`fello.py:33`) does a USB-level reset, but physical replug may still be needed in edge cases.
- ~~Control appears to be jittering on MAC (gs_usb), while on Linux (SocketCAN), we have verified it works fine.~~ Fixed — echo frame filtering in damiao-motor 1.0.7b1 (published to PyPI).

---

## Architecture overview

Two separate DaMiao driver stacks exist in this repo:

1. **`damiao-motor` PyPI package** (>=1.0.7b1) — used by `FelloRobot` and `YamRobot` (follower). This is the primary driver for all direct motor control.
2. **`i2rt` dm_driver** (`third_party/i2rt/i2rt/motor_drivers/dm_driver.py`) — used by YAM **leader** arm (`arm_server.py:368`, via `get_yam_robot()`). This is the i2rt MotorChainRobot driver for the teaching handle + encoder chain.

The local copy at `third_party/damiao_motor_pkg/` (v1.0.6.post1, see `third_party/damiao_motor_pkg/pyproject.toml:7`) is **no longer the active install**. The project pulls `damiao-motor>=1.0.7b1` from PyPI (`pyproject.toml:44`). The local copy remains as a development reference and for the GUI/CLI tools.

---

## Changes made

### 1. Echo frame filtering in `poll_feedback` (jitter fix)

**File:** Published in `damiao-motor` 1.0.7b1 on PyPI (not in local `third_party/damiao_motor_pkg`).

**Problem:** The gs_usb driver on macOS echoes back transmitted CAN frames. The background polling thread in `DaMiaoController.poll_feedback()` was processing these echo frames as motor feedback. Since echo frames contain TX command data (not actual motor feedback), the motor state would be overwritten with incorrectly-decoded values, causing jitter in position/velocity readings.

**Fix:** The published 1.0.7b1 package adds `is_rx` check: `if hasattr(msg, "is_rx") and not msg.is_rx: continue`. The python-can gs_usb interface sets `is_rx=False` for echoed TX frames and `is_rx=True` for genuine RX frames. This filters out all echo contamination.

**Current state:** The local `third_party/damiao_motor_pkg/damiao_motor/core/controller.py` does **not** have the `is_rx` filter — the `poll_feedback()` method at line 268 reads all frames without echo discrimination. The fix lives in the PyPI-published version only. The local copy's `poll_feedback` (`controller.py:268-311`) instead routes by logical motor ID from `msg.data[0] & 0x0F` without any `is_rx` guard.

**Impact:** Eliminates the jittering that only appeared on macOS (gs_usb). Linux (SocketCAN) is unaffected because it does not echo TX frames to userspace.

### 2. Motor fault auto-recovery

**File:** `third_party/damiao_motor_pkg/damiao_motor/core/motor.py:585-598`

**Problem:** The original `_check_motor_status()` only handled `DISABLED` and `LOST_COMM` states. Other fault states were not handled, causing the motor to stay in a permanent fault state until power-cycled.

Motor 7 (gripper) was observed entering `MOS_OVER_TEMP` fault immediately after startup (199 C reported — firmware error code, not real temperature).

**Fix:** `_check_motor_status()` (`motor.py:585`) now defines `_FAULT_CODES` (`motor.py:575-583`) as a frozenset of all fault status codes:
- `DM_MOTOR_OVER_VOLTAGE` (0x8) — `motor.py:235`
- `DM_MOTOR_UNDER_VOLTAGE` (0x9) — `motor.py:236`
- `DM_MOTOR_OVER_CURRENT` (0xA) — `motor.py:237`
- `DM_MOTOR_MOS_OVER_TEMP` (0xB) — `motor.py:238`
- `DM_MOTOR_ROTOR_OVER_TEMP` (0xC) — `motor.py:239`
- `DM_MOTOR_LOST_COMM` (0xD) — `motor.py:240`
- `DM_MOTOR_OVERLOAD` (0xE) — `motor.py:241`

When status matches any fault code, the method calls `clear_error()` then `enable()`, logging a warning. This is invoked on every command send (`send_cmd_mit` at `motor.py:624`, `send_cmd_pos_vel` at `motor.py:648`, `send_cmd_vel` at `motor.py:670`, `send_cmd_force_pos` at `motor.py:701`).

### 3. CAN traffic monitoring

**Files:** `third_party/damiao_motor_pkg/damiao_motor/core/controller.py`, `third_party/damiao_motor_pkg/damiao_motor/core/motor.py`

Per-motor TX/RX counters are tracked inside the controller:
- **TX counting:** `motor.py:447-448` — increments `ctrl._tx_count[motor_id]` on every `send_raw()` call.
- **RX counting:** `controller.py:304` — increments `self._rx_count[logical_id]` in `poll_feedback()`.
- **Drop counting:** `controller.py:301` — increments `self._rx_drop_count` for frames with no matching motor.
- **Stats API:** `controller.py:412-423` — `get_can_stats(reset=True)` returns `{"tx": {...}, "rx": {...}, "rx_drop": int}`.

This allows verifying:
- All 7 motors receive commands (TX balanced)
- All 7 motors return feedback (RX balanced)
- Motor 7 is not being throttled (RX[7] ~ TX[7])

**Note:** The fello_server control loop does not currently log CAN stats periodically. The `get_can_stats()` API is available but not called from any server code — it must be called manually or added to a monitoring loop.

### 4. Gripper value clamping

**File:** `robot/fello/fello_server.py:111`

`_map_real_to_sim()` (`fello_server.py:101-112`) now clamps the gripper value to [0, 1] with `np.clip(mapped[6], 0.0, 1.0)` at line 111. The gripper motor encoder can accumulate multi-revolution positions, producing values far outside the expected [0, 0.785] rad range. Without clamping, this caused bogus anchor values (e.g., 15.812) in the delta takeover computation.

### 5. gs_usb replug issue and auto-reconnect

**File:** `robot/fello/fello.py:33-66` (`_reset_gs_usb_device`), `robot/fello/fello.py:330-367` (`reconnect`)

**USB device reset:** `_reset_gs_usb_device()` at `fello.py:33` runs on macOS only (`sys.platform == "darwin"`). It:
1. Stops the gs_usb device via protocol (`GsUsb.scan()[index].stop()`)
2. Performs a USB-level reset via `usb.core` (`all_devs[target_index].reset()`)
3. Sleeps 0.5s for the device to re-enumerate

Called automatically during `FelloRobot.__init__()` at `fello.py:186` before creating the DaMiaoController.

**Reconnect flow:** `FelloRobot.reconnect()` at `fello.py:330`:
1. Marks `_connected = False`
2. Calls `controller.shutdown()`
3. Resets USB device via `_reset_gs_usb_device()`
4. Creates a fresh `DaMiaoController`
5. Re-adds and re-enables all 7 motors

**Auto-reconnect trigger:** The fello_server control loop (`fello_server.py:178-236`) tracks consecutive CAN failures. After `_CAN_FAIL_THRESHOLD = 3` consecutive ticks with failures (`fello_server.py:175`), it calls `robot.reconnect()` (`fello_server.py:219`). If reconnect fails, it sleeps 1s and retries next tick (`fello_server.py:224`).

### 6. Platform-aware CAN bus support

**Files:** `robot/constants.py:1-29`, `robot/fello/fello.py:69-192`, `robot/fello/fello_server.py:414-449`, `robot/yam/arm_server.py:326-364`, `robot/yam/yam_controller.py:22-59`

**Problem:** CAN bus type and interface names were hardcoded for Linux (socketcan). On macOS, gs_usb adapters use USB serial numbers instead of kernel interface names, and `DaMiaoController` needs `bustype="gs_usb"` and `bitrate=1000000` parameters.

**Fix:**
- `robot/constants.py:6` — detects platform (`sys.platform == "darwin"`) and sets `CAN_BUSTYPE` (`gs_usb` vs `socketcan`).
- `robot/constants.py:15-29` — per-side CAN interface names: USB serial numbers on macOS (e.g., `"0046002E594E501820313332"` for right leader at line 17), kernel names on Linux (e.g., `"can_leader_r"` at line 29). Left/follower macOS serials are still placeholder TODOs (`constants.py:18-24`).
- `FelloRobot.__init__()` accepts `bustype` parameter (`fello.py:91`), passes to `DaMiaoController(channel=..., bustype=..., bitrate=1000000)` at `fello.py:187-188`.
- `YamRobot.__init__()` accepts `bustype` parameter (`yam_controller.py:38`), passes to `DaMiaoController` at `yam_controller.py:58-59`.
- `fello_server.py:414-427` — resolves CAN channel from `constants.py` when `CAN_BUSTYPE == "gs_usb"`, instead of requiring `--can-interface` CLI arg.
- `arm_server.py:341-346` — validates CAN interface is configured before connecting, raises descriptive error if not.

**gs_usb kernel driver patch:** `DaMiaoController` (`controller.py:16-55`) applies `_patch_gs_usb_for_macos()` at module import time on macOS. This wraps `GsUsb.start()` to catch `USBError` from `detach_kernel_driver()` — a no-op on macOS that raises an error.

**gs_usb channel resolution:** `DaMiaoController.__init__()` (`controller.py:71-89`) detects `bustype == "gs_usb"` and converts the channel to an integer device index. For serial-number-based channels (no trailing digits), it falls back to device index 0.

**Impact:** Fello and YAM arm servers now work on macOS without manual CAN interface specification. The `launch_sim.sh --use-fello` Mac Mini path launches `fello_server.py --side right` and lets it resolve the CAN channel automatically.

### 7. damiao-motor upgraded to 1.0.7b1 from PyPI

**Files:** `pyproject.toml:44`, `pyproject.toml:208`

Switched from local editable install (`third_party/damiao_motor_pkg`) to published PyPI package `damiao-motor>=1.0.7b1`. This version includes the echo frame filtering fix (item 1) and motor fault auto-recovery (item 2) upstream.

The dependency appears in two places:
- Main project dependencies: `pyproject.toml:44`
- `macmini` dependency group (minimal macOS deps): `pyproject.toml:208`

The `uv.lock` resolves to `damiao_motor-1.0.7b1` from PyPI (`uv.lock:540-549`).

**Local copy status:** `third_party/damiao_motor_pkg/pyproject.toml:7` still declares version `1.0.6.post1`. The local package is not listed in `[tool.uv.sources]` and is not installed as an editable dependency. It remains in-tree as a development reference and for its CLI/GUI tools.

### 8. DaMiaoController gs_usb device index resolution

**File:** `third_party/damiao_motor_pkg/damiao_motor/core/controller.py:71-89`

When `bustype == "gs_usb"`, the controller constructor:
1. Extracts trailing digits from the channel string via regex (`controller.py:84-85`): e.g., `"can0"` -> device index `0`.
2. For serial-number-based channels with no trailing digits, falls back to device index `0`.
3. Passes `bitrate=1000000` to `can.interface.Bus()` (required for gs_usb, not needed for socketcan).

This means USB serial numbers from `robot/constants.py` are resolved to device index 0 by default. If multiple gs_usb adapters are connected, the ordering depends on USB enumeration order. The `GsUsb.scan()` discovery command (documented in `constants.py:12-13`) can be used to find serial numbers.

### 9. Gravity compensation and torque calibration

**Files:** `robot/fello/fello.py:233-275` (stiction compensation), `robot/fello/fello.py:206-231` (torque calibrators), `robot/yam/yam_controller.py:122-166` (YAM gravity comp)

Both Fello and YAM robots compute gravity compensation torques using MuJoCo inverse dynamics (`MuJoCoKDL` from `i2rt`). The Fello robot additionally applies:
- **Stiction compensation** (`fello.py:233-237`): adds a direction-dependent torque offset to overcome static friction, loaded from `fello_config.yaml`.
- **Per-motor torque calibrators** (`fello.py:206-231`): optional LUT-based calibration from CSV files (`MotorCalibrator` from `robot/systemid/motor_calibrator.py`).

The YAM robot uses a global scale factor `_GRAVITY_COMP_FACTOR = 1.3` (`yam_controller.py:19`) matching i2rt's default.

### 10. YAM follower gripper calibration

**File:** `robot/yam/yam_controller.py:101-120`

The YAM follower's `_calibrate_gripper()` method moves the gripper to both physical stops using `FORCE_POS` mode to determine close/open positions. It:
1. Commands to `-MOTOR_RANGE` and `+MOTOR_RANGE` with low velocity (5 rad/s) and torque limit (0.3 ratio)
2. Waits 2.5s at each stop for settling
3. Records `gripper_close_pos` (min) and `gripper_open_pos` (max) in env-space coordinates

This calibration runs automatically during `connect()` (`yam_controller.py:276`) for the gripper motor.

---

## CAN bus topology

```
macOS (Mac Mini)                           Linux (LECAR server)
─────────────────                          ─────────────────────
USB-CAN adapter (gs_usb)                   CANable2 (slcan → socketcan)
  │                                          │
  ├─ bustype: "gs_usb"                       ├─ bustype: "socketcan"
  ├─ channel: USB serial number              ├─ channel: "can_leader_r" / "can_follow_r"
  ├─ bitrate: 1000000                        ├─ bitrate: 1000000
  │                                          │
  └─ DaMiaoController                        └─ DaMiaoController (or i2rt CanInterface)
       └─ 7 motors (0x01-0x07)                    └─ 7 motors (0x01-0x07)
           Motors 1-6: arm (MIT mode)                  Motors 1-6: arm (MIT mode)
           Motor 7: gripper (FORCE_POS)                Motor 7: gripper (FORCE_POS)
```

**Linux CAN setup:** systemd services in `hardware/systemd/` bridge CANable2 slcan firmware to SocketCAN interfaces. Example: `canable2-slcan-left@.service` creates `can_follow_l` at 1 Mbit/s. udev rules in `hardware/99-canable2-*.rules` match USB serial numbers to specific tty devices.

**macOS CAN setup:** No kernel-level CAN subsystem. The `gs_usb` Python library talks to the USB-CAN adapter directly. The `_patch_gs_usb_for_macos()` function in `controller.py:16-55` patches `GsUsb.start()` to handle the unsupported `detach_kernel_driver` call on macOS.

---

## Motor configuration reference

From `robot/constants.py:43-59`:

| Constant | Value | Description |
|----------|-------|-------------|
| `YAM_ARM_MOTOR_IDS` | `[0x01..0x06]` | Arm joint CAN IDs |
| `YAM_ARM_MOTOR_TYPES` | `["4340","4340","4340","4310","4310","4310"]` | Motor type presets |
| `YAM_GRIPPER_MOTOR_ID` | `0x07` | Gripper CAN ID |
| `YAM_GRIPPER_MOTOR_TYPE` | `"4310"` | Linear 4310 gripper |
| `YAM_ARM_KP` | `[80,80,80,40,10,10]` | Default position gains |
| `YAM_ARM_KD` | `[5,5,5,1.5,1.5,1.5]` | Default velocity gains |
| `YAM_GRIPPER_KP` | `20.0` | Gripper position gain |
| `YAM_GRIPPER_KD` | `0.5` | Gripper velocity gain |
| `YAM_GRIPPER_VEL_LIMIT` | `30.0` rad/s | Gripper FORCE_POS velocity limit |
| `YAM_GRIPPER_TORQUE_LIMIT_NM` | `0.75` Nm | Gripper FORCE_POS torque limit |
| `YAM_GRIPPER_SIGN` | `-1` | Motor-to-env direction mapping |
| `MAX_JOINT_VELOCITY_RAD_S` | `6` rad/s | Safety: max joint speed for all layers |

Fello motor IDs and types are loaded from `robot/models/fello/fello_config.yaml` via `robot/fello/fello_config.py:6`.

---

## Server startup flow

### Fello leader server (`fello_server.py`)

```
main()
  ├─ load_fello_config()                          # fello_server.py:279
  ├─ Resolve CAN channel from constants.py        # fello_server.py:414-427
  ├─ FelloRobot(bustype=CAN_BUSTYPE, ...)         # fello_server.py:437-450
  │    ├─ _reset_gs_usb_device()                  # fello.py:186  (macOS only)
  │    └─ DaMiaoController(channel, bustype, 1M)  # fello.py:187-188
  ├─ robot.connect()                              # fello_server.py:452
  │    ├─ add_motor() x7                          # fello.py:286-289
  │    ├─ motor.enable() x7                       # fello.py:291
  │    └─ MuJoCoKDL(fello.xml)                    # fello.py:307
  ├─ start_control_loop(freq_hz)                  # fello_server.py:471-477
  │    └─ gravity comp + auto-reconnect loop      # fello_server.py:187-236
  └─ FelloLeaderRobotServer.serve()               # fello_server.py:515
```

### YAM follower server (`arm_server.py`)

```
main(mode="follower")
  ├─ CAN_INTERFACE_MAP[mode][side]                # arm_server.py:326-335
  ├─ Validate CAN interface configured            # arm_server.py:341-346
  ├─ YamRobot(bustype=CAN_BUSTYPE, ...)           # arm_server.py:353-364
  │    └─ DaMiaoController(channel, bustype, 1M)  # yam_controller.py:58-59
  ├─ robot.connect()                              # arm_server.py:365
  │    ├─ add_motor() x7                          # yam_controller.py:267-268
  │    ├─ motor.enable() x7                       # yam_controller.py:273
  │    ├─ _calibrate_gripper()                    # yam_controller.py:276
  │    └─ MuJoCoKDL(yam_4310_linear.xml)          # yam_controller.py:280-283
  └─ FollowerRobotServer.serve()                  # arm_server.py:366
```

---

## Debugging guide

### Verify gs_usb adapter is detected

```python
# Run on macOS to find connected USB-CAN adapters
python -c "from gs_usb.gs_usb import GsUsb
for i, d in enumerate(GsUsb.scan()): print(i, d.serial_number)"
```

Output should show the device serial matching `RIGHT_LEADER_CAN_INTERFACE` in `robot/constants.py:17`.

### Check CAN stats at runtime

```python
# From within a running FelloRobot or YamRobot instance:
stats = robot.controller.get_can_stats(reset=True)
print(f"TX: {stats['tx']}  RX: {stats['rx']}  Drops: {stats['rx_drop']}")
```

Expected: TX and RX counts should be roughly equal per motor. Significant RX deficit on motor 7 indicates CAN bus priority starvation.

### Monitor motor temperatures

```bash
uv run robot/monitor_motor_temps.py
```

This queries all four arm servers (follower L/R, leader L/R) via Portal RPC and displays a live table. Also launched automatically by `launch.py:149`.

### Check motor fault status

Motor fault codes are defined in `motor.py:233-252`. The `_check_motor_status()` method at `motor.py:585` auto-clears faults and re-enables the motor. Watch logs for:
```
Motor N: clearing fault MOS_OVER_TEMP
```

If a motor repeatedly enters fault state, it may indicate a real hardware issue (not just a firmware error code).
