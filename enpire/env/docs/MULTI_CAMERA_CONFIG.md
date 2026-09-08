# Design Doc: Multi-Source Camera Configuration

> Paths written as `cap/…`, `robot/…`, `experimental/…`, `tmux/…` or
> `experiments/…` are relative to `enpire/env/forge/`.

## Goal

Support mixed camera types (RealSense D405, ZED 2i, future others) in a single station through a centralized, per-station camera configuration. A dedicated **camera factory** module resolves backend type, serial numbers, resolution, and FPS from environment variables and station profiles, so that every consumer (data collection, CAP server, RL, VLM tools) gets cameras through a single creation path.

---

## Architecture

```
robot/station_profiles.py          (station identity + camera profiles)
  │
  │  CameraConfig(name, type, symlink?, device_id?)
  │  StationCameras(cameras=(...))
  │  resolve_station_key() → ENPIRE_STATION, else "default"
  │  active_station_cameras() → StationCameras
  │
  ▼
robot/camera_factory.py            (single entry point for camera creation)
  │
  │  get_camera_backend(name) → "realsense" | "zed"
  │  resolve_realsense_serial(name) → str
  │  resolve_zed_serial(name) → int
  │  create_camera(name, resolution, fps, enable_depth) → RealSenseCamera | ZedCamera
  │
  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  Camera drivers                                                                │
│                                                                                │
│  robot/realsense.py                                                            │
│    └─ RealSenseCamera  (pyrealsense2 SDK, D400-series)                         │
│                                                                                │
│  robot/zed.py                                                                  │
│    └─ ZedCamera  (pyzed SDK, ZED 2i, center-crop + resize to 640×480)          │
│                                                                                │
│  Both return CameraData(images={"rgb": ndarray}, timestamp, depth?, intrinsics?)│
└────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  Consumers                                                                     │
│                                                                                │
│  robot/yam/yam_real_env.py                                                     │
│    └─ NonBlockingCamera(camera_name)                                           │
│         wraps camera_factory.create_camera() in a background thread            │
│                                                                                │
│  robot/yam/_base_yam_env.py                                                    │
│    └─ camera_names param → obs space + MuJoCo cameras                          │
│                                                                                │
│  cap/config.py                                                                 │
│    └─ CAMERA_NAMES = active_station_cameras().names                            │
│                                                                                │
│  cap/server/cap_server.py                                                      │
│    └─ _CameraClient(camera_name)                                               │
│         uses camera_factory.create_camera() + get_camera_backend()             │
│         exposes get_rgb(), get_depth(), get_intrinsics()                        │
│    └─ SimCameraClient (sim/warp-sim mode)                                      │
│         renders from MuJoCo via SimBackend                                     │
│                                                                                │
│  cap/agent/tools/vlm_query.py                                                  │
│    └─ imports CAMERA_NAMES from cap/config.py                                  │
│                                                                                │
│  cap/agent/tools/segmentation.py                                               │
│    └─ imports CAMERA_NAMES from cap/config.py                                  │
│                                                                                │
│  scripts/smoke_test_camera_streams.py                                          │
│    └─ calls camera_factory.create_camera() directly for all three cameras      │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Station profile resolution

The active station comes from a single environment variable — there is no
hostname inference and no in-code profile table:

```python
# robot/station_profiles.py
def resolve_station_key() -> str:
    return os.environ.get("ENPIRE_STATION", "default").strip()
```

Profiles themselves live **outside the repository**, one YAML per station under
`~/.config/enpire/stations/<id>.yaml`, and are created and inspected with the
CLI rather than by editing Python:

```bash
uv run enpire station init     --station my-yam
uv run enpire station register --station my-yam   # detects CAN/USB serials
uv run enpire station show     --station my-yam
```

The `cameras:` block of that profile maps each role to a device and backend:

```yaml
cameras:
  top:
    device: '353322270678'     # librealsense serial
    backend: realsense
  left_wrist:
    device: '335122273181'
    backend: realsense
```

The default camera roles when no profile is loaded are `top`, `left_fixed`,
`left`, and `right`.

> Earlier revisions of this document described `CAMERA_PROFILES`,
> `STATION_BY_HOSTNAME`, `_THANOS_CAMERAS`, and a `local_station.toml`. None of
> those exist in the current code — station selection is `ENPIRE_STATION` plus
> the external profile above.

## Camera factory (`robot/camera_factory.py`)

The **camera factory** is the single entry point for creating camera objects. It resolves backend, resolution, FPS, and device serial through a layered environment variable system, then instantiates the correct driver.

### Backend resolution (`robot/camera_factory.py`)

```python
def get_camera_backend(camera_name: str, default: str | None = None) -> str:
```

Priority:
1. `CAP_{NAME}_CAMERA_BACKEND` env var (e.g. `CAP_TOP_CAMERA_BACKEND=zed`).
2. `CAP_CAMERA_BACKENDS` env var, comma-separated map (e.g. `top=zed,left=realsense`).
3. Built-in default: `"zed"` for `"top"`, `"realsense"` for everything else (`robot/camera_factory.py`).

Accepted backend aliases (`robot/camera_factory.py`):
| Alias | Normalized |
|-------|-----------|
| `realsense`, `rs`, `d405` | `"realsense"` |
| `zed`, `zed2i`, `stereolabs` | `"zed"` |

### Resolution and FPS overrides

| Env var | Scope | Default |
|---------|-------|---------|
| `CAP_{NAME}_CAMERA_RESOLUTION` | per-camera | caller-provided |
| `CAP_CAMERA_RESOLUTION` | global | caller-provided |
| `CAP_{NAME}_CAMERA_FPS` | per-camera | caller-provided |
| `CAP_CAMERA_FPS` | global | caller-provided |

Resolution format: `"640x480"` or `"640,480"` (`robot/camera_factory.py`).

### Serial resolution

**RealSense** (`robot/camera_factory.py`):
1. `CAP_{NAME}_REALSENSE_SERIAL` or `CAP_REALSENSE_SERIAL` env var.
2. Resolve from udev symlink `/dev/video_{name}` by matching the USB device physical port in `pyrealsense2`.

**ZED** (`robot/camera_factory.py`):
1. `CAP_{NAME}_ZED_SERIAL` or `CAP_ZED_SERIAL` env var.
2. If exactly one ZED is connected, uses that serial.
3. Otherwise, raises with list of connected serials.

### ZED-specific settings

| Env var | Default | Notes |
|---------|---------|-------|
| `CAP_{NAME}_ZED_NATIVE_RESOLUTION` | `"HD720"` | Capture resolution before crop+resize |
| `CAP_ZED_NATIVE_RESOLUTION` | `"HD720"` | Global fallback |
| `CAP_{NAME}_ZED_DEPTH_MODE` | `"NEURAL"` | `NEURAL`, `ULTRA`, `PERFORMANCE`, etc. |
| `CAP_ZED_DEPTH_MODE` | `"NEURAL"` | Global fallback |

### create_camera() (`robot/camera_factory.py`)

```python
def create_camera(
    camera_name: str,
    *,
    resolution: tuple[int, int] = (640, 480),
    fps: int = 60,
    enable_depth: bool = True,
) -> RealSenseCamera | ZedCamera:
```

Dispatches on resolved backend:
- `"realsense"` -> `robot.realsense.RealSenseCamera(device_id=..., resolution=..., fps=..., enable_depth=...)`
- `"zed"` -> `robot.zed.ZedCamera(device_id=..., resolution=..., fps=..., enable_depth=..., native_resolution=..., depth_mode=...)`

---

## Camera drivers

### RealSenseCamera (`robot/realsense.py`)

- Wraps `pyrealsense2` pipeline.
- Configured by serial number, resolution, FPS, auto-exposure, brightness.
- `read()` returns `CameraData(images={"rgb": ndarray}, timestamp, depth?, intrinsics?)`.
- `get_intrinsics()` returns `dict` with keys `fx, fy, cx, cy, width, height`.
- `stop()` closes the pipeline.

### ZedCamera (`robot/zed.py`)

- Wraps `pyzed.sl.Camera`.
- Performs center-crop and resize from native resolution to requested resolution (`robot/zed.py`).
- Intrinsics are transformed to match the cropped+resized output (`robot/zed.py`).
- `read()` returns `CameraData` matching the same interface as `RealSenseCamera`.
- Depth: `MEASURE.DEPTH` in metres, NaN replaced with 0.0, resized via nearest-neighbor.
- `get_intrinsics()` returns `dict` with keys `fx, fy, cx, cy, width, height, native_width, native_height`.
- `stop()` calls `sl.Camera.close()`.

### CameraData (shared return type)

Both `robot/realsense.py` and `robot/zed.py` define their own `CameraData` dataclass with the same fields:

```python
@dataclass
class CameraData:
    images: Dict[str, Optional[np.ndarray]]   # {"rgb": HxWx3 uint8}
    timestamp: float                            # milliseconds
    depth: Optional[np.ndarray] = None          # float32 metres, HxW
    intrinsics: Optional[Dict[str, float]] = None
```

---

## Output contract

All backends produce **RGB uint8 HxWx3** frames (default 480x640x3) via `read().images["rgb"]`. The observation key format remains `{name}_camera_image` (e.g. `top_camera_image`). Downstream consumers (data collection, RL, CAP tools) are unaffected.

For the CAP server `_CameraClient` (`cap/server/cap_server.py`), the public interface is:
- `get_rgb() -> np.ndarray` -- RGB uint8 copy
- `get_depth() -> np.ndarray | None` -- float32 metres copy
- `get_intrinsics() -> list[float] | None` -- `[fx, fy, cx, cy]` (converted from dict)

The CAP server exposes these over Portal RPC (`cap/server/cap_server.py`):
- `get_camera_image(camera) -> ndarray`
- `get_camera_depth(camera) -> ndarray`
- `get_camera_intrinsics(camera) -> list[float]`

---

## Consumer details

### YamRealEnv (`robot/yam/yam_real_env.py`)

`NonBlockingCamera` (`robot/yam/yam_real_env.py`) wraps `camera_factory.create_camera()` in a background thread. Each camera runs a worker that continuously calls `camera.read()` and caches the latest RGB frame.

The env creates cameras for `["top", "left", "right"]` when `enable_cameras=True` (`robot/yam/yam_real_env.py`). Camera names are currently hard-coded in this file (not yet driven by `station_profiles`).

Constructor also accepts:
- `enabled_camera_names: tuple[str, ...] | None` -- defaults to `("top", "left", "right")` (`robot/yam/yam_real_env.py,123-125`).
- `top_camera_source: "direct" | "rpc"` -- can route top camera through CAP server RPC instead of direct hardware access (`robot/yam/yam_real_env.py,127-131`).
- `enable_depth_cameras: bool` and `depth_camera_names: tuple[str, ...]` for selective depth.

### _BaseYamEnv (`robot/yam/_base_yam_env.py`)

- `camera_names` constructor parameter (`robot/yam/_base_yam_env.py`) defines which camera slots appear in the obs space and MuJoCo scene.
- Default is `("top", "left", "right")`.
- `CAMERA_HEIGHT = 480`, `CAMERA_WIDTH = 640` (`robot/yam/_base_yam_env.py`).
- Observation space includes `{name}_camera_image: Box(0, 255, (480, 640, 3), uint8)` for each name (`robot/yam/_base_yam_env.py`).
- MuJoCo camera resolution is set to match (`robot/yam/_base_yam_env.py`).

### YamSimEnv (`robot/yam/yam_sim_env.py`)

- Uses `self.camera_names` from base class (`robot/yam/yam_sim_env.py`).
- Camera IDs mapped by matching MuJoCo camera names containing `"top"`, `"left"`, `"right"`.

### CAP server (`cap/server/cap_server.py`)

- **Real mode** (`cap/server/cap_server.py`): creates `_CameraClient(name)` for each name in `CAMERA_NAMES`. Failures are caught and logged (camera becomes unavailable, not fatal).
- **Sim mode** (`cap/server/cap_server.py`): creates `SimCameraClient(backend, name)` for each name in `CAMERA_NAMES`.

`_CameraClient.__init__()` (`cap/server/cap_server.py`):
1. Calls `get_camera_backend(camera_name)` to determine type.
2. Calls `create_camera(camera_name, ...)` from `robot/camera_factory`.
3. Starts a background `_worker` thread.
4. ZED backend defaults to 30 FPS; RealSense defaults to 60 FPS.

The class also retains legacy `_init_realsense()` and `_init_zed()` methods (`cap/server/cap_server.py`) that are no longer called from `__init__` but exist for reference/fallback.

### RL observation building (`cap/server/cap_server.py`)

The `_build_rl_obs()` method resizes camera images to a square `_RL_IMAGE_SIZE` for each name in `CAMERA_NAMES`.

### VLM and segmentation tools

- `cap/agent/tools/vlm_query.py,46` -- imports `CAMERA_NAMES`, validates camera name against the set, supports `"camera:top"` media prefix.
- `cap/agent/tools/segmentation.py` -- imports `CAMERA_NAMES`, validates camera name.
- `cap/agent/executor.py` -- `_ask_vlm()` compatibility wrapper normalizes camera name strings.

---

## udev rules and hardware setup

`hardware/99-realsense.rules` and `hardware/99-zed.rules` are no longer shipped.
`uv run enpire station register --station <id>` inspects the attached devices
and writes owner-only registration artifacts, including candidate udev rules,
under the ENPIRE data home. Review them, have an administrator install them,
then reload udev and reconnect the devices.

Binding cameras to roles — including the alias-map and symlink priority order —
is documented in the "Binding cameras to roles" section of the top-level
`README.md`.

## Debugging and smoke testing

### smoke_test_camera_streams.py (`scripts/smoke_test_camera_streams.py`)

Quick validation that all three camera streams are working:

```bash
# Direct hardware capture (uses camera_factory)
uv run python scripts/smoke_test_camera_streams.py --source direct --scale-zed-to-640480

# Via CAP server RPC
uv run python scripts/smoke_test_camera_streams.py --source rpc --host 127.0.0.1 --port 8300
```

Captures a short video (default 5 seconds) per camera plus a combined side-by-side view. Converts to H.264 via ffmpeg.

### zed2i_depth.py (`tools/vision/zed2i_depth.py`)

Standalone ZED 2i depth reader for testing depth modes, point clouds, and RGB streaming:

```bash
uv run python tools/vision/zed2i_depth.py --show --frames 0
uv run python tools/vision/zed2i_depth.py --save-npy depth.npy --frames 1
```

### Camera calibration (`robot/camera_calibration_core.py`)

ChArUco board detection and camera calibration engine. Used for extrinsic calibration of camera-to-robot transforms.

---

## Environment variable reference

### Backend selection

| Variable | Scope | Example |
|----------|-------|---------|
| `CAP_{NAME}_CAMERA_BACKEND` | per-camera | `CAP_TOP_CAMERA_BACKEND=zed` |
| `CAP_CAMERA_BACKENDS` | global map | `top=zed,left=realsense,right=realsense` |

### Serial / device ID

| Variable | Scope | Example |
|----------|-------|---------|
| `CAP_{NAME}_REALSENSE_SERIAL` | per-camera | `CAP_LEFT_REALSENSE_SERIAL=000000000001` |
| `CAP_REALSENSE_SERIAL` | global | `CAP_REALSENSE_SERIAL=000000000001` |
| `CAP_{NAME}_ZED_SERIAL` | per-camera | `CAP_TOP_ZED_SERIAL=10000001` |
| `CAP_ZED_SERIAL` | global | (single-ZED fallback) |

### Resolution and FPS

| Variable | Scope | Format |
|----------|-------|--------|
| `CAP_{NAME}_CAMERA_RESOLUTION` | per-camera | `640x480` |
| `CAP_CAMERA_RESOLUTION` | global | `640x480` |
| `CAP_{NAME}_CAMERA_FPS` | per-camera | `30` |
| `CAP_CAMERA_FPS` | global | `60` |

### ZED-specific

| Variable | Scope | Default |
|----------|-------|---------|
| `CAP_{NAME}_ZED_NATIVE_RESOLUTION` | per-camera | `HD720` |
| `CAP_ZED_NATIVE_RESOLUTION` | global | `HD720` |
| `CAP_{NAME}_ZED_DEPTH_MODE` | per-camera | `NEURAL` |
| `CAP_ZED_DEPTH_MODE` | global | `NEURAL` |

### YamRealEnv-specific

| Variable | Default | Notes |
|----------|---------|-------|
| `YAM_TOP_CAMERA_SOURCE` | `"direct"` | Set to `"rpc"` to route top camera through CAP server |
| `CAP_SERVER_HOST` | `"127.0.0.1"` | Used when `top_camera_source="rpc"` |
| `CAP_SERVER_PORT` | `8300` | Used when `top_camera_source="rpc"` |

---

## Adding a new camera type

1. Add the type string to `CameraConfig.type` literal in `robot/station_profiles.py`.
2. Add aliases to `_BACKEND_ALIASES` in `robot/camera_factory.py`.
3. Create a driver module `robot/<backend>.py` with a class that has `read() -> CameraData`, `get_intrinsics()`, and `stop()`.
4. Add a branch in `camera_factory.create_camera()` (`robot/camera_factory.py`).
5. Add a serial resolver (e.g. `resolve_<backend>_serial()`) in `robot/camera_factory.py`.
6. Add udev rules in `hardware/` for the new device.

## Adding a new station

```bash
uv run enpire station init     --station <id>
uv run enpire station register --station <id>   # detects CAN/USB serials
```

Then fill in the `cameras:` block of `~/.config/enpire/stations/<id>.yaml` with
each role's librealsense serial and backend, bind the roles (see "Binding
cameras to roles" in the top-level `README.md`), and calibrate with
[`CALIBRATION_BOARD.md`](CALIBRATION_BOARD.md).

Nothing needs to be added to the repository for a new station — profiles,
calibration output, and device serials all live outside it.

## Tests

| Test file | What it covers |
|-----------|----------------|
| `tests/test_yam_real_env_zed.py` | `NonBlockingZed._resolve_device()` symlink resolution and fallback (NOTE: references legacy class that may be stale) |
| `tests/test_yam_sim_no_cameras.py` | `YamSimEnv` with `enable_cameras=False` returns blank frames, obs space correct |

---

## Cross-references

| Doc | Relationship |
|-----|-------------|
| [CAP_DESIGN.md](CAP_DESIGN.md) | CAP server control loop, Portal RPC API including camera image/depth/intrinsics endpoints |
| [RL_PIPELINE_DESIGN.md](RL_PIPELINE_DESIGN.md) | RL observation building resizes camera images to `_RL_IMAGE_SIZE` |
| [SKILL_LIBRARY.md](SKILL_LIBRARY.md) | Skill tools (VLM query, segmentation) use `CAMERA_NAMES` for camera selection |
