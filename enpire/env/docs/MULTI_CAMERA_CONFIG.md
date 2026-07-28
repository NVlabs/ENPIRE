# Design Doc: Multi-Source Camera Configuration

## Goal

Support mixed camera types (RealSense D405, ZED 2i, future others) in a single station through a centralized, per-station camera configuration. A dedicated **camera factory** module resolves backend type, serial numbers, resolution, and FPS from environment variables and station profiles, so that every consumer (data collection, CAP server, RL, VLM tools) gets cameras through a single creation path.

---

## Architecture

```
robot/station_profiles.py          (station identity + camera profiles)
  │
  │  CameraConfig(name, type, symlink?, device_id?)
  │  StationCameras(cameras=(...))
  │  CAMERA_PROFILES: dict[str, StationCameras]
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

`robot/station_profiles.py:142-165`

Resolution order (first hit wins, cached for process lifetime):

1. Environment variable `LECAR_STATION` (profile key).
2. Gitignored `robot/local_station.toml` with `station = "thor"`.
3. `STATION_BY_HOSTNAME`: `socket.gethostname().lower()` to profile key.
4. Falls back to `"default"`.

```python
# robot/station_profiles.py:116-119
STATION_BY_HOSTNAME: dict[str, str] = {
    "lecarlab-legion-t7-34irz8": "thanos",
    "lecar-legion-t5-26ara8": "thor",
}
```

---

## Configuration

### Data model (`robot/station_profiles.py:52-70`)

```python
@dataclass(frozen=True)
class CameraConfig:
    name: str                          # "top", "left", "right", ...
    type: Literal["realsense", "zed"]  # driver backend
    symlink: str | None = None         # udev device symlink
    device_id: str | None = None       # stable camera serial

@dataclass(frozen=True)
class StationCameras:
    cameras: tuple[CameraConfig, ...]

    @property
    def names(self) -> tuple[str, ...]: ...
```

### Profile table (`robot/station_profiles.py:108-113`)

```python
CAMERA_PROFILES: dict[str, StationCameras] = {
    "default": _DEFAULT_CAMERAS,   # 3x RealSense D405 via udev symlinks
    "thanos": _THANOS_CAMERAS,     # ZED 2i top + 2x D405 wrist via serial
    "thor":   _THANOS_CAMERAS,
    "tony":   _THANOS_CAMERAS,
}
```

Default cameras (`robot/station_profiles.py:77-81`):
```python
_DEFAULT_CAMERAS = StationCameras((
    CameraConfig("top",   "realsense", "/dev/video_top"),
    CameraConfig("left",  "realsense", "/dev/video_left"),
    CameraConfig("right", "realsense", "/dev/video_right"),
))
```

Thanos/Thor/Tony cameras (`robot/station_profiles.py:83-88`):
```python
_THANOS_CAMERAS = StationCameras((
    CameraConfig("top",   "zed",       "/dev/video_top_zed2i"),
    CameraConfig("left",  "realsense", device_id="000000000001"),
    CameraConfig("right", "realsense", device_id="000000000002"),
))
```

### CAMERA_NAMES in cap/config.py (`cap/config.py:213-222`)

```python
def _resolve_camera_names() -> tuple[str, ...]:
    try:
        from robot.station_profiles import active_station_cameras
        return active_station_cameras().names
    except Exception:
        return ("top", "left", "right")

CAMERA_NAMES: tuple[str, ...] = _resolve_camera_names()
```

All downstream CAP tools (vlm_query, segmentation, cap_server) import `CAMERA_NAMES` from `cap/config.py`.

---

## Camera factory (`robot/camera_factory.py`)

The **camera factory** is the single entry point for creating camera objects. It resolves backend, resolution, FPS, and device serial through a layered environment variable system, then instantiates the correct driver.

### Backend resolution (`robot/camera_factory.py:52-65`)

```python
def get_camera_backend(camera_name: str, default: str | None = None) -> str:
```

Priority:
1. `CAP_{NAME}_CAMERA_BACKEND` env var (e.g. `CAP_TOP_CAMERA_BACKEND=zed`).
2. `CAP_CAMERA_BACKENDS` env var, comma-separated map (e.g. `top=zed,left=realsense`).
3. Built-in default: `"zed"` for `"top"`, `"realsense"` for everything else (`robot/camera_factory.py:48-49`).

Accepted backend aliases (`robot/camera_factory.py:7-14`):
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

Resolution format: `"640x480"` or `"640,480"` (`robot/camera_factory.py:68-78`).

### Serial resolution

**RealSense** (`robot/camera_factory.py:99-121`):
1. `CAP_{NAME}_REALSENSE_SERIAL` or `CAP_REALSENSE_SERIAL` env var.
2. Resolve from udev symlink `/dev/video_{name}` by matching the USB device physical port in `pyrealsense2`.

**ZED** (`robot/camera_factory.py:124-142`):
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

### create_camera() (`robot/camera_factory.py:165-201`)

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

### RealSenseCamera (`robot/realsense.py:34-161`)

- Wraps `pyrealsense2` pipeline.
- Configured by serial number, resolution, FPS, auto-exposure, brightness.
- `read()` returns `CameraData(images={"rgb": ndarray}, timestamp, depth?, intrinsics?)`.
- `get_intrinsics()` returns `dict` with keys `fx, fy, cx, cy, width, height`.
- `stop()` closes the pipeline.

### ZedCamera (`robot/zed.py:42-189`)

- Wraps `pyzed.sl.Camera`.
- Performs center-crop and resize from native resolution to requested resolution (`robot/zed.py:134-148`).
- Intrinsics are transformed to match the cropped+resized output (`robot/zed.py:105-132`).
- `read()` returns `CameraData` matching the same interface as `RealSenseCamera`.
- Depth: `MEASURE.DEPTH` in metres, NaN replaced with 0.0, resized via nearest-neighbor.
- `get_intrinsics()` returns `dict` with keys `fx, fy, cx, cy, width, height, native_width, native_height`.
- `stop()` calls `sl.Camera.close()`.

### CameraData (shared return type)

Both `robot/realsense.py:24-30` and `robot/zed.py:34-38` define their own `CameraData` dataclass with the same fields:

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

For the CAP server `_CameraClient` (`cap/server/cap_server.py:395-406`), the public interface is:
- `get_rgb() -> np.ndarray` -- RGB uint8 copy
- `get_depth() -> np.ndarray | None` -- float32 metres copy
- `get_intrinsics() -> list[float] | None` -- `[fx, fy, cx, cy]` (converted from dict)

The CAP server exposes these over Portal RPC (`cap/server/cap_server.py:2256-2271`):
- `get_camera_image(camera) -> ndarray`
- `get_camera_depth(camera) -> ndarray`
- `get_camera_intrinsics(camera) -> list[float]`

---

## Consumer details

### YamRealEnv (`robot/yam/yam_real_env.py`)

`NonBlockingCamera` (`robot/yam/yam_real_env.py:40-98`) wraps `camera_factory.create_camera()` in a background thread. Each camera runs a worker that continuously calls `camera.read()` and caches the latest RGB frame.

The env creates cameras for `["top", "left", "right"]` when `enable_cameras=True` (`robot/yam/yam_real_env.py:155-162`). Camera names are currently hard-coded in this file (not yet driven by `station_profiles`).

Constructor also accepts:
- `enabled_camera_names: tuple[str, ...] | None` -- defaults to `("top", "left", "right")` (`robot/yam/yam_real_env.py:114,123-125`).
- `top_camera_source: "direct" | "rpc"` -- can route top camera through CAP server RPC instead of direct hardware access (`robot/yam/yam_real_env.py:115,127-131`).
- `enable_depth_cameras: bool` and `depth_camera_names: tuple[str, ...]` for selective depth.

### _BaseYamEnv (`robot/yam/_base_yam_env.py`)

- `camera_names` constructor parameter (`robot/yam/_base_yam_env.py:29`) defines which camera slots appear in the obs space and MuJoCo scene.
- Default is `("top", "left", "right")`.
- `CAMERA_HEIGHT = 480`, `CAMERA_WIDTH = 640` (`robot/yam/_base_yam_env.py:18`).
- Observation space includes `{name}_camera_image: Box(0, 255, (480, 640, 3), uint8)` for each name (`robot/yam/_base_yam_env.py:131-137`).
- MuJoCo camera resolution is set to match (`robot/yam/_base_yam_env.py:53-56`).

### YamSimEnv (`robot/yam/yam_sim_env.py`)

- Uses `self.camera_names` from base class (`robot/yam/yam_sim_env.py:57`).
- Camera IDs mapped by matching MuJoCo camera names containing `"top"`, `"left"`, `"right"`.

### CAP server (`cap/server/cap_server.py`)

- **Real mode** (`cap/server/cap_server.py:724-733`): creates `_CameraClient(name)` for each name in `CAMERA_NAMES`. Failures are caught and logged (camera becomes unavailable, not fatal).
- **Sim mode** (`cap/server/cap_server.py:697-707`): creates `SimCameraClient(backend, name)` for each name in `CAMERA_NAMES`.

`_CameraClient.__init__()` (`cap/server/cap_server.py:188-219`):
1. Calls `get_camera_backend(camera_name)` to determine type.
2. Calls `create_camera(camera_name, ...)` from `robot/camera_factory`.
3. Starts a background `_worker` thread.
4. ZED backend defaults to 30 FPS; RealSense defaults to 60 FPS.

The class also retains legacy `_init_realsense()` and `_init_zed()` methods (`cap/server/cap_server.py:223-391`) that are no longer called from `__init__` but exist for reference/fallback.

### RL observation building (`cap/server/cap_server.py:3615-3627`)

The `_build_rl_obs()` method resizes camera images to a square `_RL_IMAGE_SIZE` for each name in `CAMERA_NAMES`.

### VLM and segmentation tools

- `cap/agent/tools/vlm_query.py:32,46` -- imports `CAMERA_NAMES`, validates camera name against the set, supports `"camera:top"` media prefix.
- `cap/agent/tools/segmentation.py:21` -- imports `CAMERA_NAMES`, validates camera name.
- `cap/agent/executor.py:290-318` -- `_ask_vlm()` compatibility wrapper normalizes camera name strings.

---

## udev rules and hardware setup

### RealSense D405 (`hardware/99-realsense.rules`)

Maps serial numbers to stable symlinks per station:

| Station | top serial | left serial | right serial |
|---------|------------|-------------|--------------|
| 1 | `000000000003` | `000000000004` | `000000000005` |
| 2 | `000000000006` | `000000000007` | `000000000008` |
| 3 | `000000000009` | `000000000010` | `000000000011` |
| 4 | `000000000012` | `000000000013` | `000000000014` |
| 5 | `000000000015` | `000000000016` | `000000000017` |
| 6 | `000000000018` | `000000000019` | `000000000020` |
| 7 | (ZED top) | path-based: `usb-0:7.4.3.3:1.0` | path-based: `usb-0:6.1.3.2:1.0` |

Station 7 uses `ENV{ID_PATH}` matching instead of `ATTRS{serial}` because the D405 units are routed through a Dell dock hub (`hardware/99-realsense.rules:34-40`).

Symlinks created: `/dev/video_top`, `/dev/video_left`, `/dev/video_right`.

### ZED 2i (`hardware/99-zed.rules`)

```
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTR{index}=="0",
  ATTRS{idVendor}=="2b03", ATTRS{idProduct}=="f880", ATTRS{serial}=="OV0001",
  SYMLINK+="video_top_zed2i", MODE="0666"
```

Symlink created: `/dev/video_top_zed2i`.

### Helper scripts

| Script | Purpose |
|--------|---------|
| `hardware/print-realsense-video-udev.sh` | Print Intel V4L nodes and ATTRS for building `99-realsense.rules` |
| `hardware/realsense_alias_helper.py` | Runtime symlink creation for D405 cameras by serial match |
| `tools/debug/configure_realsense_video.sh` | Install udev rule, reload, and verify/create a single symlink |

---

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

1. Add the type string to `CameraConfig.type` literal in `robot/station_profiles.py:57`.
2. Add aliases to `_BACKEND_ALIASES` in `robot/camera_factory.py:7-14`.
3. Create a driver module `robot/<backend>.py` with a class that has `read() -> CameraData`, `get_intrinsics()`, and `stop()`.
4. Add a branch in `camera_factory.create_camera()` (`robot/camera_factory.py:177-200`).
5. Add a serial resolver (e.g. `resolve_<backend>_serial()`) in `robot/camera_factory.py`.
6. Add udev rules in `hardware/` for the new device.

## Adding a new station

1. Add an Avengers-slug entry to `PROFILES` (CAN) and `CAMERA_PROFILES` (cameras) in `robot/station_profiles.py:100-113`.
2. Add hostname-to-slug mapping in `STATION_BY_HOSTNAME` (`robot/station_profiles.py:116-119`), or set `LECAR_STATION` on the machine, or create `robot/local_station.toml`.
3. Add camera serial numbers to `hardware/99-realsense.rules` and/or `hardware/99-zed.rules`.
4. Install udev rules:
   ```bash
   sudo cp hardware/99-*.rules /etc/udev/rules.d/ && \
   sudo udevadm control --reload-rules && \
   sudo udevadm trigger
   ```
5. Verify with: `bash hardware/print-realsense-video-udev.sh` and `ls -l /dev/video_*`.

---

## Tests

| Test file | What it covers |
|-----------|----------------|
| `tests/test_yam_real_env_zed.py` | `NonBlockingZed._resolve_device()` symlink resolution and fallback (NOTE: references legacy class that may be stale) |
| `tests/test_yam_sim_no_cameras.py` | `YamSimEnv` with `enable_cameras=False` returns blank frames, obs space correct |

---

## Files changed (vs. original three-RealSense design)

| File | Role |
|------|------|
| `robot/station_profiles.py` | `CameraConfig`, `StationCameras`, `CAMERA_PROFILES`, `active_station_cameras()` |
| `robot/camera_factory.py` | Backend resolution, serial resolution, env-var overrides, `create_camera()` factory |
| `robot/realsense.py` | `RealSenseCamera` driver (D400-series via pyrealsense2) |
| `robot/zed.py` | `ZedCamera` driver (ZED 2i via pyzed, center-crop + resize) |
| `robot/yam/yam_real_env.py` | `NonBlockingCamera` wrapping `camera_factory.create_camera()` |
| `robot/yam/_base_yam_env.py` | `camera_names` constructor param, config-driven obs space |
| `robot/yam/yam_sim_env.py` | Uses `self.camera_names` from base class for MuJoCo rendering |
| `cap/config.py` | `CAMERA_NAMES` derived from `active_station_cameras()` |
| `cap/server/cap_server.py` | `_CameraClient` uses `camera_factory`; `SimCameraClient` for sim modes |
| `cap/agent/tools/vlm_query.py` | Imports `CAMERA_NAMES` from config |
| `cap/agent/tools/segmentation.py` | Imports `CAMERA_NAMES` from config |
| `cap/agent/executor.py` | `_ask_vlm()` normalizes camera names |
| `hardware/99-realsense.rules` | udev symlinks for D405 cameras (stations 1-7) |
| `hardware/99-zed.rules` | udev symlink for ZED 2i (`/dev/video_top_zed2i`) |
| `hardware/print-realsense-video-udev.sh` | Helper to discover RealSense USB devices |
| `hardware/realsense_alias_helper.py` | Runtime symlink creation by serial match |
| `tools/debug/configure_realsense_video.sh` | Install/verify single udev symlink |
| `scripts/smoke_test_camera_streams.py` | Multi-camera smoke test (direct + RPC) |
| `robot/camera_calibration_core.py` | ChArUco camera calibration engine |

---

## Cross-references

| Doc | Relationship |
|-----|-------------|
| [CAP_DESIGN.md](CAP_DESIGN.md) | CAP server control loop, Portal RPC API including camera image/depth/intrinsics endpoints |
| [RL_PIPELINE_DESIGN.md](RL_PIPELINE_DESIGN.md) | RL observation building resizes camera images to `_RL_IMAGE_SIZE` |
| [BUNDLESDF_OBJECT_DETECTION.md](BUNDLESDF_OBJECT_DETECTION.md) | BundleSDF tracking consumes camera RGB+depth from CAP server |
| [TABLE_BUSSING_SKILLS.md](TABLE_BUSSING_SKILLS.md) | Skill tools (VLM query, segmentation) use `CAMERA_NAMES` for camera selection |
| [DATA_STUDIO.md](DATA_STUDIO.md) | Data visualization uses camera image keys from recorded episodes |
| [SAFETY_ZONE_DESIGN.md](SAFETY_ZONE_DESIGN.md) | Safety zones may use camera extrinsics for workspace bounds |
| [CAP_UI_DESIGN.md](CAP_UI_DESIGN.md) | UI displays camera feeds via WebSocket from CAP agent |
| [VLM_QUERY.md](VLM_QUERY.md) | VLM query tool camera selection and media prefixes |
