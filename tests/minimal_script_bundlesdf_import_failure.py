# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBS_DIR = REPO_ROOT / "third_party" / "bundlesdf" / "libs"


def _sha256(path: Path) -> str:
    target = path.resolve() if path.is_symlink() else path
    h = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if os.environ.get("_BUNDLESDF_LD_READY") != "1":
    env = os.environ.copy()
    env["_BUNDLESDF_LD_READY"] = "1"
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{LIBS_DIR}:{current_ld}" if current_ld else str(LIBS_DIR)
    os.execve(sys.executable, [sys.executable, __file__], env)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"python={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")

for name in (
    "libBundleTrack.so",
    "libMY_CUDA_LIB.so",
    "my_cpp.cpython-311-x86_64-linux-gnu.so",
):
    path = LIBS_DIR / name
    kind = "symlink" if path.is_symlink() else "file"
    print(f"{path} [{kind}] sha256={_sha256(path)}")

print("importing bundlesdf ...")
import bundlesdf  # noqa: F401

print("bundlesdf import succeeded")
