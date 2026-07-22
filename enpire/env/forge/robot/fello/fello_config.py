# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_FELLO_CONFIG_PATH = Path(
    os.environ.get("ENPIRE_FELLO_CONFIG", Path(__file__).with_name("fello_config.yaml"))
).expanduser()
_MODELS_DIR = Path(
    os.environ.get("ENPIRE_FELLO_MODEL_ROOT", Path(__file__).parents[1] / "models")
).expanduser()
_LEGACY_FELLO_CONFIG_PATH = _MODELS_DIR / "fello" / "fello_config.yaml"
_FELLO_CONFIG_PATHS = {
    "left": _MODELS_DIR / "fello_left" / "fello_config.yaml",
    "right": _MODELS_DIR / "fello_right" / "fello_config.yaml",
}
_FELLO_XML_PATHS = {
    "left": _MODELS_DIR / "fello_left" / "fello.xml",
    "right": _MODELS_DIR / "fello_right" / "fello.xml",
}


def _normalize_side(side: str | None) -> str | None:
    if side is None:
        return None
    side = side.lower()
    if side not in _FELLO_CONFIG_PATHS:
        raise ValueError(f"Invalid Fello side: {side}")
    return side


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("fello_config.yaml must contain a mapping at the top level")
    return data


def get_fello_config_path(side: str | None = None) -> Path:
    if side is None:
        side = os.environ.get("FELLO_SIDE")
    side = _normalize_side(side)

    if _PACKAGE_FELLO_CONFIG_PATH.exists():
        return _PACKAGE_FELLO_CONFIG_PATH

    if side is not None and _FELLO_CONFIG_PATHS[side].exists():
        return _FELLO_CONFIG_PATHS[side]

    if _LEGACY_FELLO_CONFIG_PATH.exists():
        return _LEGACY_FELLO_CONFIG_PATH

    for default_side in ("left", "right"):
        path = _FELLO_CONFIG_PATHS[default_side]
        if path.exists():
            return path

    searched_paths = [
        _PACKAGE_FELLO_CONFIG_PATH,
        _LEGACY_FELLO_CONFIG_PATH,
        *_FELLO_CONFIG_PATHS.values(),
    ]
    searched = ", ".join(str(path) for path in searched_paths)
    raise FileNotFoundError(f"Missing Fello config. Looked in: {searched}")


FELLO_CONFIG_PATH = get_fello_config_path()


def load_fello_config(side: str | None = None) -> dict[str, Any]:
    if side is None:
        side = os.environ.get("FELLO_SIDE")
    side = _normalize_side(side)
    config_path = get_fello_config_path(side)
    data = _read_yaml_config(config_path)
    side_overrides = data.pop("sides", {}) or {}
    if side is not None:
        override = side_overrides.get(side, {})
        if not isinstance(override, dict):
            raise ValueError(
                f"fello_config.yaml side override for {side!r} must be a mapping"
            )
        data = _deep_merge(data, override)
    return data


def resolve_config_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def get_fello_xml_path(cfg: dict[str, Any], side: str | None = None) -> Path:
    xml_path = get_config_value(cfg, "model", "xml_path")
    if xml_path:
        return resolve_config_path(xml_path)
    side = _normalize_side(side)
    if side is None:
        side = _normalize_side(os.environ.get("FELLO_SIDE")) or "right"
    return _FELLO_XML_PATHS[side]


def get_config_value(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
