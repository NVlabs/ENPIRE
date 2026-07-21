"""Redaction helpers for artifacts and public diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "<redacted>"
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|credential)s?($|[_-])",
    re.IGNORECASE,
)


def is_sensitive_key(key: object) -> bool:
    return bool(_SENSITIVE_KEY.search(str(key)))


def redact(value: Any) -> Any:
    """Recursively redact values stored under credential-like keys."""

    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    return value
