# Installation Guide

This document covers the **end-to-end installation** of the CAP family of services
(cap_server, cap_agent, cap_ui, SAM3, AnyGrasp, BundleSDF, cuRobo) on a fresh
Linux workstation.

> **TL;DR**: `bash install/install_cap.sh`. It chains `install/compile_bundlesdf.sh` automatically.

The goal of this design is:
- One entry point: **`install/install_cap.sh`**.
- Everything project-local at runtime: only `.venv/` (uv) and `third_party/`.
- Build-time conda envs are **deletable** after install.
- **No system mutation**: no `apt install`, no `/usr/lib` writes, no driver/CUDA changes.
- Cross-machine portable: tarballs of `third_party/{bundlesdf_5090,anygrasp_libs,nodejs}/` work on any modern Linux with Python 3.11 + CUDA 12.x.

---

## 1. Prerequisites (host machine)

| Component | Why | Version |
|---|---|---|
| Linux x86_64 | OS | glibc ≥ 2.17 (any distro from 2014+) |
| NVIDIA driver | GPU access | matching CUDA 12.x (we use `/usr/local/cuda-12.8`) |
| `gcc`, `make`, `wget`, `git`, `git-lfs`, `tmux`, `curl` | tooling | any modern version |
| ~10 GB free in `$HOME` | conda envs + sources | first install only |

What you do **not** need installed system-wide:
- Python 3.11 (uv handles it)
- conda (the script bootstraps miniforge into `~/miniforge3/` if missing)
- Node.js / npm (downloaded into `third_party/nodejs/`)
- OpenSSL 1.1, OpenBLAS, PCL, OpenCV, Boost (all userspace — see below)

---

## 2. One-shot install

```bash
git clone <this repo>
cd forge
bash install/install_cap.sh
```

That's it. The script is **idempotent** — re-running it skips anything already done.

On a clean machine with good network and an RTX 5090 / Ada / Hopper GPU, the
first run takes **~60–80 minutes** dominated by the OpenCV-CUDA + BundleSDF
compile (Phase 5). Subsequent runs are seconds.

---

## 3. What `install/install_cap.sh` does, phase by phase

### Phase 1 — Submodules + Git LFS

- Initialises `third_party/robosuite` (uv resolves an editable path against it during sync).
- Sets `lfs.fetchexclude = third_party/bundlesdf/libs/**` locally so the LFS-tracked broken Ubuntu 24.04 prebuilt BundleSDF libs are **not** pulled. We rebuild them locally instead.
- Pulls remaining LFS objects (AnyGrasp checkpoint, license zip, MinkowskiEngine compiled `.so`, pointnet2 `.so`).

### Phase 2 — Python venv (uv)

- `uv sync --extra cap --extra cap_tools` — populates `.venv/` with all Python deps.
- Two extra installs for AnyGrasp (not in extras because they're only needed when the AnyGrasp server runs):
  - `graspnetAPI` (pinned by graspnet to `transforms3d==0.3.1`),
  - `transforms3d>=0.4` upgraded after the fact (the strict pin would break NumPy 2.x with `np.maximum_sctype` removed).

### Phase 3 — Hugging Face models

- Detects HF auth state: skip if already logged in, use `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN` env if set, otherwise prompt interactively (`hf auth login`).
- Downloads two models:
  - `facebook/sam3` (gated — accept the license at <https://huggingface.co/facebook/sam3> first),
  - `facebook/sam2.1-hiera-large` (public).

### Phase 4 — cuRobo

- Idempotently builds and installs cuRobo from `third_party/curobo` into `.venv/`. Verifies via `import curobo.geom` + `torch.cuda.is_available()`. Inlined from `tmux/remote_serving/install_curobo_s1.sh`.

### Phase 5 — BundleSDF compiled libs (delegates to `install/compile_bundlesdf.sh`)

Runs `install/compile_bundlesdf.sh` only if `third_party/bundlesdf_5090/libBundleTrack.so` is missing.

The sub-script does its own multi-phase build — see [§5 below](#5-installbundlesdfshs-internal-flow).

### Phase 6 — AnyGrasp runtime libs

Creates a tiny ephemeral conda env `forge-anygrasp-libs` with `openssl=1.1` + `openblas`, then **extracts** these libraries into `third_party/anygrasp_libs/` along with their transitive `.so` closure (e.g. `libgfortran` for OpenBLAS), with unversioned SONAME symlinks added.

After this, the conda env is no longer needed (it's optional cache).

Why a separate ephemeral env? Stock Ubuntu 22.04 ships only OpenSSL 3.0; AnyGrasp's `gsnet.so` was linked against 1.1. Conda-forge's PCL 1.15 (in the BundleSDF build env) requires OpenSSL 3, so 1.1 cannot coexist there.

### Phase 7 — Node.js

Downloads the official Linux x64 prebuild (`node-v22.11.0-linux-x64.tar.xz`, ~75 MB) into `third_party/nodejs/`. Self-contained; no conda dependency.

### Phase 8 — `.forge_env`

Generates a 5-line shell snippet at `<repo>/.forge_env`:

```bash
_FORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANYGRASP_OPENSSL11_DIR="$_FORGE_ROOT/third_party/anygrasp_libs"
export OPENBLAS_HOME="$_FORGE_ROOT/third_party/anygrasp_libs"
export PATH="$_FORGE_ROOT/third_party/nodejs/bin:${PATH}"
```

The launcher sources this so all panes pick up project-local OpenSSL 1.1, OpenBLAS, and node/npm regardless of the user's shell PATH.

---

## 4. End state on disk

```
forge/
├── .venv/                                # uv Python env (all Python deps)
├── .forge_env                            # auto-generated shell snippet
├── third_party/
│   ├── bundlesdf_5090/                   # built BundleSDF runtime + closure (~240 MB)
│   ├── anygrasp_libs/                    # libssl.so.1.1, libcrypto.so.1.1, libopenblas.so.0
│   ├── nodejs/                           # node, npm (Vite dev server)
│   ├── bundlesdf/libs/                   # untouched LFS-tracked dir (we don't use it)
│   └── ... (other third_party deps)
└── install/install_cap.sh, install/compile_bundlesdf.sh
```

Optional, deletable any time:
- `~/miniforge3/` — miniforge installer (~500 MB)
- `~/miniforge3/envs/bundlesdf-build/` — BundleSDF build deps (~3 GB)
- `~/miniforge3/envs/forge-anygrasp-libs/` — ephemeral, only used during install (~few hundred MB)

You can `rm -rf ~/miniforge3/envs/{bundlesdf-build,forge-anygrasp-libs}` after a successful install. Runtime depends only on `.venv/` and `third_party/`.

---

## 5. `install/compile_bundlesdf.sh`'s internal flow

Called automatically by `install/install_cap.sh` step 5. Can also be run manually.

| Phase | What | Why |
|---|---|---|
| 0 | Preflight | Checks `nvcc`, venv, pybind11, BundleTrack source, captures CUDA file hashes (verified unchanged at the end). |
| 1 | Miniforge | Installs `~/miniforge3/` if absent. Userspace only. |
| 2 | Conda env `bundlesdf-build` | Pulls **build-time** C++ deps: PCL 1.15, OpenCV (we replace this with a CUDA build in 2.5), Boost 1.86, yaml-cpp, freeglut, mesa, eigen 3.4, gcc 11, sysroot 2.17 (cos7), Python 3.11. |
| 2.5 | OpenCV-CUDA from source | Conda-forge's OpenCV is built `WITH_CUDA=OFF`. We clone OpenCV 4.11 + opencv_contrib, build with `-DWITH_CUDA=ON`, install into the conda env. ~25–35 min. |
| 3 | (Removed) | (was a backup step; no longer needed since we never write to `libs/`) |
| 4 | BundleTrack build | cmake + ninja against the conda env. Targets sm_75/80/86/89/90/120 by default (RTX 5090 = sm_120). System nvcc, conda gcc 11, project venv pybind11. Forces Python 3.11 paths so `my_cpp.cpython-*.so` matches the venv ABI. |
| 5 | Stage `third_party/bundlesdf_5090/` | Copies built `libBundleTrack.so`, `libMY_CUDA_LIB.so`, `my_cpp.cpython-311-*.so` plus the conda runtime closure (`libpcl_*.so.1.15`, `libopencv_*.so.4.11.0`, `libboost_*.so.1.88.0`, `libyaml-cpp.so.0.8`, `libstdc++.so.6`, `libgcc_s.so.1`, etc.). Adds unversioned SONAME symlinks. |
| 6 | Verify | `import bundlesdf; from bundlesdf import BundleSdf` with `BUNDLESDF_RUNTIME_LIB_DIR=third_party/bundlesdf_5090`. Tolerates the benign teardown segfault from PCL/CUDA destructor ordering. |
| 7 | CUDA hash check | Re-hashes `nvcc`, `libcuda.so.1`, `libcudart.so.12`, `nvidia-smi`. Fails loudly if any changed. |
| 8 | Tarball | Writes `bundlesdf-5090-cuda12-py311-x86_64.tar.gz` (~240 MB) for fleet distribution. |

### Distributing to other hosts

```bash
scp bundlesdf-5090-cuda12-py311-x86_64.tar.gz host:/path/to/forge/
ssh host 'cd /path/to/forge && tar -C third_party -xzf bundlesdf-5090-cuda12-py311-x86_64.tar.gz'
```

The target host needs Python 3.11 + CUDA 12.x driver + a covered GPU arch (default builds cover Turing through Blackwell). No conda needed on the target.

### Fast incremental rebuild

After source edits to `third_party/bundlesdf/BundleTrack/src/`, just re-run
`install/compile_bundlesdf.sh`. Phases 1, 2, and 2.5 all skip when the
existing conda env / OpenCV-CUDA build are detected, so only the
BundleTrack cmake + ninja step actually re-runs (~1–3 min for incremental
edits). Output stays in `third_party/bundlesdf_5090/`.

```bash
bash install/compile_bundlesdf.sh
```

(Note: `third_party/bundlesdf/build.sh` is the unrelated 4070/4090
variant's build script that targets `third_party/bundlesdf/libs/` — do
not run it for 5090 builds.)

---

## 6. Why BundleSDF lives in `bundlesdf_5090/`, not `bundlesdf/libs/`

`third_party/bundlesdf/libs/` is **Git-LFS-tracked**, with the binaries built on
Ubuntu 24.04 (require GLIBC_2.38, GLIBCXX_3.4.32). These do not load on stock
22.04 / RHEL 9 hosts, even with a fresh `git lfs pull`.

We rebuild from `third_party/bundlesdf/BundleTrack/src/` and write the result
to a **separate** dir — `third_party/bundlesdf_5090/` — so it never collides
with the LFS files. The bundlesdf Python module supports a
`BUNDLESDF_RUNTIME_LIB_DIR` environment variable (see
`third_party/bundlesdf/__init__.py:38-56`); the launcher exports it pointing at
`third_party/bundlesdf_5090/`.

`third_party/bundlesdf_5090/` is `.gitignore`-d — never committed.

---

## 7. Driver / CUDA safety

The install scripts deliberately do not touch:

| Path | Status |
|---|---|
| Kernel modules (`nvidia.ko`, etc.) | Untouched |
| `/lib/x86_64-linux-gnu/libcuda.so.1` | Untouched |
| `nvidia-smi`, `nvidia-settings` | Untouched |
| `/usr/local/cuda*` | Read-only |
| `/etc/ld.so.conf.d/*` | Untouched |
| `/etc/modprobe.d/*` | Untouched |

`install/compile_bundlesdf.sh` Phase 0 captures sha256 of `nvcc`, `libcuda.so.1`, `libcudart.so.12`, `nvidia-smi`. Phase 7 re-hashes and **aborts loudly** if any changed.

The conda env explicitly excludes `cuda-toolkit`, `cudatoolkit`, `cuda-runtime`, `cudnn`, `nccl`, `nvidia-*` packages. The compiler we invoke is the system `/usr/local/cuda-12.8/bin/nvcc`. The runtime SONAMEs we link against (`libcudart.so.12`, `libcuda.so.1`) come from the existing system install — same way every PyPI CUDA wheel does it.

---

## 8. Launcher integration

`tmux/table_bussing/launch_table_bussing_local_realsense.sh` brings up the full
CAP service stack. Layout (two tmux windows):

```
Window 0: "main" — agent layer (1×3)
┌─────────────┬─────────────┬─────────────┐
│ cap_server  │ cap_agent   │ cap_ui      │
│ :8300       │ :8200       │ :5173       │
└─────────────┴─────────────┴─────────────┘

Window 1: "services" — perception + control (2×2)
┌──────────────────┬──────────────────┐
│ serve_bundlesdf  │ serve_sam3       │
│ (uses SAM2 too)  │                  │
│ :8119            │ :6767            │
├──────────────────┼──────────────────┤
│ serve_anygrasp   │ serve_curobo     │
│ :8122            │ :8611            │
└──────────────────┴──────────────────┘
```

Switch windows with `Ctrl-b n` / `Ctrl-b p` (or `Ctrl-b 0` / `Ctrl-b 1`).
Session name: `cap_family_bucket`.

Key environment hookups inside the launcher:

- Sources `<repo>/.forge_env` so `ANYGRASP_OPENSSL11_DIR`, `OPENBLAS_HOME`, and node/npm `PATH` are set.
- The AnyGrasp pane re-sources `.forge_env` inside the tmux pane (in case the tmux server already had stale env).
- The BundleSDF pane exports `BUNDLESDF_RUNTIME_LIB_DIR=third_party/bundlesdf_5090` and prepends that dir to `LD_LIBRARY_PATH`.
- The cuRobo pane runs `experimental/serve_portal_motion_planner.py --port 8611 --robot-type yam`.
- cap_agent waits for SAM3 + AnyGrasp + BundleSDF + cuRobo before starting (each can be disabled via `--no-bundlesdf` / `--no-curobo`).

### Common flags

```bash
bash tmux/table_bussing/launch_table_bussing_local_realsense.sh             # full stack
bash tmux/table_bussing/launch_table_bussing_local_realsense.sh --no-bundlesdf
bash tmux/table_bussing/launch_table_bussing_local_realsense.sh --no-curobo
bash tmux/table_bussing/launch_table_bussing_local_realsense.sh --sim       # MuJoCo sim
```

---

## 9. Reversibility

| To undo | Command |
|---|---|
| All conda artifacts | `rm -rf ~/miniforge3` |
| Just BundleSDF build env | `rm -rf ~/miniforge3/envs/bundlesdf-build` |
| Local BundleSDF build | `rm -rf third_party/bundlesdf_5090 third_party/bundlesdf/BundleTrack/build` |
| Project Python env | `rm -rf .venv` |
| All extracted runtime libs | `rm -rf third_party/{anygrasp_libs,nodejs,bundlesdf_5090}` |

Nothing under `/usr`, `/lib`, `/etc`, or `/usr/local/cuda*` is ever modified, so
nothing system-wide to undo.

---

## 10. Cross-references

| Doc | Topic |
|---|---|
| [BUNDLESDF_OBJECT_DETECTION.md](BUNDLESDF_OBJECT_DETECTION.md) | BundleSDF tracking system |
| [lfs_setup.md](lfs_setup.md) | Git LFS configuration and which paths are excluded |
| [MACMINI_SETUP.md](MACMINI_SETUP.md) | Mac Mini setup (sim mode only — no CUDA) |
| [VLM_QUERY.md](VLM_QUERY.md) | VLM tool used by cap_agent |
| [TABLE_BUSSING_SKILLS.md](TABLE_BUSSING_SKILLS.md) | Skills the launched stack runs |
