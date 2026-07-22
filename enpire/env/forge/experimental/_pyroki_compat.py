# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from enpire.env.forge.paths import THIRD_PARTY_ROOT


def import_pyroki():
    try:
        import pyroki as pk  # type: ignore
        return pk
    except ModuleNotFoundError as exc:
        if exc.name != "pyroki":
            raise

    candidate = THIRD_PARTY_ROOT / "pyroki" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    import pyroki as pk  # type: ignore
    return pk
