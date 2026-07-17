from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_BACKEND_ALIASES = {
    "realsense": "realsense",
    "rs": "realsense",
    "d405": "realsense",
    "zed": "zed",
    "zed2i": "zed",
    "stereolabs": "zed",
}

_SYSTEM_REALSENSE_ALIAS_MAP = Path("/usr/local/lib/forge-rl-station/forge_rl_camera_aliases.json")


def _user_realsense_alias_map() -> Path:
    """Return the external alias map path; device identities never live in Git."""

    explicit = _env_first("ENPIRE_YAM_CAMERA_ALIASES")
    if explicit is not None:
        return Path(explicit).expanduser()
    data_home = _env_first("ENPIRE_DATA_HOME")
    if data_home is None:
        xdg_data_home = _env_first("XDG_DATA_HOME")
        root = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local/share"
        data_home = str(root / "enpire")
    return Path(data_home).expanduser() / "camera_aliases.json"


def _env_first(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _load_realsense_alias_map(path: Path) -> dict[str, str]:
    try:
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text())
        return {
            str(serial): str(alias)
            for serial, alias in raw.items()
            if not str(serial).startswith("_")
        }
    except Exception:
        return {}


def normalize_camera_backend(value: str) -> str:
    backend = _BACKEND_ALIASES.get(str(value).strip().lower())
    if backend is None:
        supported = ", ".join(sorted(set(_BACKEND_ALIASES.values())))
        raise ValueError(f"Unsupported camera backend {value!r}. Supported backends: {supported}")
    return backend


def _parse_backend_map(spec: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, backend = item.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid CAP_CAMERA_BACKENDS entry {item!r}; expected entries like 'top=zed,left=realsense'"
            )
        mapping[name.strip().lower()] = normalize_camera_backend(backend)
    return mapping


def _default_backend_for_camera(camera_name: str) -> str:
    if camera_name == "top":
        if (
            _env_first(
                "FORGE_STATION_XML",
                "YAM_STATION_CALIBRATED_XML",
                "YAM_STATION_CALIBRATED_XML_PATH",
            )
            is not None
        ):
            return "realsense"
    return "zed" if camera_name == "top" else "realsense"


def get_camera_backend(camera_name: str, default: str | None = None) -> str:
    camera_name = str(camera_name).strip().lower()

    explicit = _env_first(f"CAP_{camera_name.upper()}_CAMERA_BACKEND")
    if explicit is not None:
        return normalize_camera_backend(explicit)

    mapping_spec = _env_first("CAP_CAMERA_BACKENDS")
    if mapping_spec is not None:
        mapping = _parse_backend_map(mapping_spec)
        if camera_name in mapping:
            return mapping[camera_name]

    return normalize_camera_backend(default or _default_backend_for_camera(camera_name))


def parse_resolution(value: str) -> tuple[int, int]:
    raw = value.strip().lower().replace(" ", "")
    for sep in ("x", ","):
        if sep in raw:
            w_str, h_str = raw.split(sep, 1)
            width = int(w_str)
            height = int(h_str)
            if width <= 0 or height <= 0:
                break
            return width, height
    raise ValueError(f"Invalid resolution {value!r}; expected formats like '640x480' or '640,480'")


def center_square_crop(image: Any) -> Any:
    height, width = image.shape[:2]
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def crop_image_region(image: Any, region: Any) -> Any:
    x, y, w, h = map(int, region)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError(f"Invalid crop region {(x, y, w, h)} for image shape {image.shape}")
    return image[y : y + h, x : x + w]


def get_camera_resolution(camera_name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = _env_first(
        f"CAP_{camera_name.upper()}_CAMERA_RESOLUTION",
        "CAP_CAMERA_RESOLUTION",
    )
    if value is None:
        return default
    return parse_resolution(value)


def get_camera_fps(camera_name: str, default: int) -> int:
    value = _env_first(
        f"CAP_{camera_name.upper()}_CAMERA_FPS",
        "CAP_CAMERA_FPS",
    )
    return int(value) if value is not None else int(default)


def resolve_realsense_serial(camera_name: str) -> str:
    camera_name = str(camera_name).strip().lower()
    explicit = _env_first(
        f"CAP_{camera_name.upper()}_REALSENSE_SERIAL",
        "CAP_REALSENSE_SERIAL",
    )
    if explicit is not None:
        return explicit

    import pyrealsense2 as rs

    ctx = rs.context()

    connected_serials = {dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()}

    alias_fallbacks = {
        "left": ("video_left",),
        "left_fixed": ("video_left_third",),
        "left_third": ("video_left_third",),
        "left_wrist": ("video_left",),
    }
    desired_aliases = alias_fallbacks.get(camera_name, (f"video_{camera_name}",))

    def _resolve_from_alias_map(path: Path) -> str | None:
        alias_map = _load_realsense_alias_map(path)
        if not alias_map:
            return None
        candidate_serials = [
            serial for serial, alias in alias_map.items() if alias in desired_aliases
        ]
        if not candidate_serials:
            return None
        for serial in candidate_serials:
            if serial in connected_serials:
                return serial
        return None

    serial = _resolve_from_alias_map(_SYSTEM_REALSENSE_ALIAS_MAP)
    if serial is not None:
        return serial

    for desired_alias in desired_aliases:
        symlink = f"/dev/{desired_alias}"
        if not os.path.exists(symlink):
            continue
        video = os.path.basename(os.path.realpath(symlink))
        usb_id = os.path.basename(os.path.realpath(f"/sys/class/video4linux/{video}/device"))

        for dev in ctx.query_devices():
            if usb_id in dev.get_info(rs.camera_info.physical_port):
                return dev.get_info(rs.camera_info.serial_number)

    serial = _resolve_from_alias_map(_user_realsense_alias_map())
    if serial is not None:
        return serial

    missing = ", ".join(f"/dev/{alias}" for alias in desired_aliases)
    if not any(os.path.exists(f"/dev/{alias}") for alias in desired_aliases):
        raise FileNotFoundError(f"No symlink: {missing}")

    raise ValueError(f"No RealSense for symlink(s): {missing}")


def resolve_zed_serial(camera_name: str) -> int:
    explicit = _env_first(
        f"CAP_{camera_name.upper()}_ZED_SERIAL",
        "CAP_ZED_SERIAL",
    )
    if explicit is not None:
        return int(explicit)

    import pyzed.sl as sl

    devices = list(sl.Camera.get_device_list())
    if len(devices) == 1:
        return int(devices[0].serial_number)

    serials = [int(dev.serial_number) for dev in devices]
    raise RuntimeError(
        f"Could not resolve ZED serial for camera {camera_name!r}. "
        f"Set CAP_{camera_name.upper()}_ZED_SERIAL. Connected ZED serials: {serials}"
    )


def _get_zed_native_resolution(camera_name: str, default: str = "HD720") -> str:
    return str(
        _env_first(
            f"CAP_{camera_name.upper()}_ZED_NATIVE_RESOLUTION",
            "CAP_ZED_NATIVE_RESOLUTION",
        )
        or default
    ).upper()


def _get_zed_depth_mode(camera_name: str, default: str = "NEURAL") -> str:
    return str(
        _env_first(
            f"CAP_{camera_name.upper()}_ZED_DEPTH_MODE",
            "CAP_ZED_DEPTH_MODE",
        )
        or default
    ).upper()


def create_camera(
    camera_name: str,
    *,
    resolution: tuple[int, int] = (640, 480),
    fps: int = 60,
    enable_depth: bool = True,
) -> Any:
    camera_name = str(camera_name).strip().lower()
    resolution = get_camera_resolution(camera_name, resolution)
    fps = get_camera_fps(camera_name, fps)
    backend = get_camera_backend(camera_name)

    if backend == "realsense":
        from enpire.env.forge.robot.realsense import RealSenseCamera

        return RealSenseCamera(
            device_id=resolve_realsense_serial(camera_name),
            resolution=resolution,
            fps=fps,
            auto_exposure=True,
            brightness=10,
            enable_depth=enable_depth,
        )

    if backend == "zed":
        from enpire.env.forge.robot.zed import ZedCamera

        return ZedCamera(
            device_id=resolve_zed_serial(camera_name),
            resolution=resolution,
            fps=fps,
            enable_depth=enable_depth,
            native_resolution=_get_zed_native_resolution(camera_name),
            depth_mode=_get_zed_depth_mode(camera_name),
        )

    raise AssertionError(f"Unhandled camera backend: {backend}")


def get_camera_type_name(camera: Any) -> str:
    return str(getattr(camera, "camera_type", camera.__class__.__name__))
