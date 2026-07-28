# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load per-environment tool API specs.

Embodiment specs live as markdown files in ``cap/prompt/embodiment/``.
The loader reads the ``## Tool API`` and ``## Environment Notes`` sections
and returns them as formatted prompt text.

Env name mapping::

    yam                                → embodiment/yam.md
    yam:PickPlace                      → embodiment/yam.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_SPEC_DIR = Path(__file__).resolve().parent / "env"

# Robot short-name mapping per platform
_ROBOT_KEY: dict[str, dict[str, str]] = {}

_ROBOT_DEFAULT: dict[str, str] = {}


def _resolve_spec_key(env_name: str) -> str:
    """Map an env name string to a YAML filename stem.

    >>> _resolve_spec_key("yam")
    'yam'
    """
    parts = env_name.split(":")
    platform = parts[0]
    robot = parts[2] if len(parts) >= 3 else None

    rmap = _ROBOT_KEY.get(platform, {})
    robot_key = rmap.get(robot, _ROBOT_DEFAULT.get(platform, "")) if robot else _ROBOT_DEFAULT.get(platform, "")
    return f"{platform}_{robot_key}" if robot_key else platform


def load_env_spec(env_name: str | None) -> dict[str, str] | None:
    """Load env spec and return formatted prompt sections.

    Tries markdown-based embodiment files first (``cap/prompt/embodiment/``),
    falls back to legacy YAML specs (``cap/prompt/env/``).

    Returns ``{"tool_docs": ..., "env_notes": ...}`` or *None* if no spec
    file matches *env_name*.
    """
    if not env_name:
        return None

    # --- Try markdown embodiment first ---
    try:
        from enpire.env.forge.cap.prompt.loader import PromptMemory

        pm = PromptMemory()
        result = pm.load_embodiment(env_name)
        if result is not None:
            logger.info("Loaded embodiment markdown for %s", env_name)
            return result
    except Exception as e:
        logger.debug("Markdown embodiment load failed for %s: %s", env_name, e)

    # --- Fallback: legacy YAML ---
    import yaml

    key = _resolve_spec_key(env_name)
    spec_path = ENV_SPEC_DIR / f"{key}.yaml"
    if not spec_path.exists():
        logger.debug("No env spec at %s", spec_path)
        return None

    with open(spec_path, encoding="utf-8") as f:
        spec: dict[str, Any] = yaml.safe_load(f)

    tool_docs = _format_tool_docs(spec)
    env_notes = _format_env_notes(spec)
    logger.info("Loaded env spec %s (%d tools)", spec_path.name, len(spec.get("tools", {})))
    return {"tool_docs": tool_docs, "env_notes": env_notes}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_tool_docs(spec: dict[str, Any]) -> str:
    """Convert YAML ``tools`` + ``return_types`` into Python-stub documentation."""
    lines: list[str] = []

    # --- function stubs ---
    for name, info in spec.get("tools", {}).items():
        params = info.get("params", {})
        sig_parts: list[str] = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "Any")
            if "default" in pinfo:
                sig_parts.append(f"{pname}: {ptype} = {pinfo['default']!r}")
            elif pinfo.get("optional"):
                sig_parts.append(f"{pname}: {ptype} = None")
            else:
                sig_parts.append(f"{pname}: {ptype}")
        sig = ", ".join(sig_parts)

        ret = info.get("returns", "")
        ret_ann = f" -> {ret}" if ret else ""
        lines.append(f"def {name}({sig}){ret_ann}:")
        lines.append(f'    """{info.get("description", "").strip()}')
        # per-param docs
        for pname, pinfo in params.items():
            doc = pinfo.get("doc", "")
            if doc:
                values = pinfo.get("values")
                suffix = f"  (one of {values})" if values else ""
                lines.append(f"    {pname}: {doc}{suffix}")
        lines.append('    """')
        lines.append("")

    # --- return type descriptions ---
    return_types = spec.get("return_types", {})
    if return_types:
        lines.append("# Return types")
        for tname, tinfo in return_types.items():
            lines.append(f"# {tname}: {tinfo.get('description', '')}")
            for fname, fdoc in tinfo.get("fields", {}).items():
                lines.append(f"#   .{fname} — {fdoc}")
        lines.append("")

    return "\n".join(lines)


def _format_env_notes(spec: dict[str, Any]) -> str:
    """Convert YAML metadata + constraints into a prompt-friendly notes block."""
    parts: list[str] = []

    robot = spec.get("robot", "")
    env = spec.get("env", "")
    if env or robot:
        parts.append(f"**Environment**: {env} — {robot}")

    arms = spec.get("arms", [])
    if arms:
        parts.append(f"**Arms**: {', '.join(arms)}")

    cameras = spec.get("cameras", [])
    if cameras:
        parts.append(f"**Cameras**: {', '.join(cameras)}")

    freq = spec.get("control_freq_hz")
    if freq:
        parts.append(f"**Control frequency**: {freq} Hz")

    grip = spec.get("gripper_range")
    if grip:
        parts.append(f"**Gripper range**: {grip[0]} (closed) to {grip[1]} (open)")

    constraints = spec.get("constraints", [])
    if constraints:
        parts.append("")
        for c in constraints:
            parts.append(f"- {c}")

    return "\n".join(parts)
