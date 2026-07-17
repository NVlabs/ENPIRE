"""Structured, credential-safe trial artifact storage."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from enpire.env.forge.security import redact


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ArtifactStore:
    """Write one trial into a self-contained directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Artifact path escapes run directory: {relative}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def write_json(self, relative: str | Path, value: Any) -> Path:
        target = self.path(relative)
        payload = json.dumps(redact(value), indent=2, sort_keys=True, default=_json_default)
        target.write_text(payload + "\n", encoding="utf-8")
        return target

    def append_event(self, event: dict[str, Any]) -> Path:
        target = self.path("events.jsonl")
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact(event), sort_keys=True, default=_json_default) + "\n")
        return target
