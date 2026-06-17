from __future__ import annotations

import sys
from pathlib import Path


def import_pyroki():
    try:
        import pyroki as pk  # type: ignore
        return pk
    except ModuleNotFoundError as exc:
        if exc.name != "pyroki":
            raise

    root = Path(__file__).resolve().parents[1]
    candidate = root / "third_party" / "pyroki" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    import pyroki as pk  # type: ignore
    return pk
