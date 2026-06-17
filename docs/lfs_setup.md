# Git LFS Setup

This document describes how Git LFS is configured in lecar-tbd, which files
are tracked, how launcher scripts materialize binaries, and how the runtime
Python code handles unmaterialized pointer files at import time.

> **Cross-references**
>
> - `docs/remote_serving.md` — remote-model-serving architecture, SSH tunnels, port map
> - `docs/BUNDLESDF_OBJECT_DETECTION.md` — BundleSDF 6-DOF tracking system
> - `docs/TABLE_BUSSING_SKILLS.md` — table-bussing skill tools that depend on these binaries

---

## Why LFS Is Used

This repo tracks large binary artifacts with Git LFS so that the repository
stays lightweight for developers who only need the source code.  Tracked
objects include compiled shared libraries, model checkpoints, license archives,
and vendor packages.

---

## Tracked Files (`.gitattributes`)

**`.gitattributes`** (repo root) declares every LFS-tracked pattern:

| Line | Pattern | Description |
|------|---------|-------------|
| 1 | `third_party/bundlesdf/libs/*.so*` | All BundleSDF shared libraries (OpenCV, PCL, GTK, ZeroMQ, etc.) |
| 2 | `third_party/bundlesdf/libs/libBundleTrack.so` | BundleTrack compiled C++/CUDA tracking library |
| 3 | `checkpoint_detection.tar` | AnyGrasp detection checkpoint |
| 4 | `checkpoint_tracking.tar` | AnyGrasp tracking checkpoint |
| 5 | `license_JalenLu.zip` | AnyGrasp SDK license archive |
| 6 | `third_party/anygrasp_sdk/dependencies/MinkowskiEngine/build/lib.linux-x86_64-cpython-311/MinkowskiEngineBackend/_C.cpython-311-x86_64-linux-gnu.so` | MinkowskiEngine compiled sparse-conv backend |
| 7 | `third_party/anygrasp_sdk/pointnet2/build/lib.linux-x86_64-cpython-311/pointnet2/_ext.cpython-311-x86_64-linux-gnu.so` | PointNet2 compiled CUDA extension |
| 9 | `experimental/pico/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb` | XRobo vendor package |

---

## Default Behavior: Lightweight Clone

**`.lfsconfig`** (repo root, line 1-2) sets `fetchexclude = *` globally, which
tells `git lfs fetch` to skip all LFS objects by default:

```ini
[lfs]
    fetchexclude = *
```

This means:

- `git clone` or `git pull` downloads only **LFS pointer files** (a few
  bytes each) instead of the actual binaries.
- The working tree is fast to clone and cheap on disk.
- LFS pointer files are never the real binaries; trying to use them directly
  causes import errors (see "What breaks without materialization" below).

---

## Git Hooks and LFS

The repo ships custom git hooks under `.githooks/` (activated via
`git config core.hooksPath .githooks`).

### `.githooks/pre-push` (lines 27-29)

Runs `git lfs pre-push` so that any new LFS objects are uploaded before the
push completes.  Also enforces branch naming and blocks direct pushes to `main`.

### `.githooks/post-merge` (lines 1-4)

Runs `git lfs post-merge` after every merge/pull so that LFS pointer files
get an opportunity to resolve.  Note: with `fetchexclude = *` in `.lfsconfig`
this hook alone does **not** materialize objects -- you still need an explicit
`git lfs pull` (or a launcher script) to download the actual binaries.

---

## How to Get the Binaries

### Option 1: Use a launcher script (recommended)

The launcher scripts clear the repo-local fetchexclude override, pull all LFS
objects, and then verify that critical dependencies are real binaries (not
pointer files).

#### `tmux/remote_serving/launch_lecar_s1.sh`

The main S1 remote-serving launcher.  The LFS restoration happens inside the
`ensure_lfs_dependencies()` function (lines 108-120):

```bash
# launch_lecar_s1.sh:110-111
git -C "$PROJECT_DIR" config --local lfs.fetchexclude "" >/dev/null
git -C "$PROJECT_DIR" lfs pull >/dev/null
```

After pulling, it verifies seven critical files using `ensure_real_dependency()`
(lines 91-106), which checks each path for existence, rejects symlinks, and
detects LFS pointer content by reading the first line:

```bash
# launch_lecar_s1.sh:85-89  — is_lfs_pointer() helper
is_lfs_pointer() {
    local path="$1"
    [[ -f "$path" ]] || return 1
    head -n 1 "$path" | grep -q 'https://git-lfs.github.com/spec/v1'
}
```

**Verified dependencies** (launch_lecar_s1.sh:113-119):

| Variable / path | File |
|-----------------|------|
| `checkpoint_detection.tar` | AnyGrasp detection model |
| `license_JalenLu.zip` | AnyGrasp license archive |
| `$POINTNET2_EXT` | `third_party/anygrasp_sdk/pointnet2/build/.../pointnet2/_ext.cpython-311-x86_64-linux-gnu.so` |
| `$MINKOWSKI_EXT` | `third_party/anygrasp_sdk/dependencies/MinkowskiEngine/build/.../_C.cpython-311-x86_64-linux-gnu.so` |
| `third_party/bundlesdf/libs/libBundleTrack.so` | BundleTrack library |
| `third_party/bundlesdf/libs/libMY_CUDA_LIB.so` | Custom CUDA helper library |
| `third_party/bundlesdf/libs/my_cpp.cpython-311-x86_64-linux-gnu.so` | BundleSDF Python extension |

This verification runs conditionally: only when `anygrasp` or `bundlesdf` is
in the requested service list (line 185-186).

#### `tmux/remote_serving/sync_lecar_s1_assets.sh`

Restores LFS objects on **both** the local clone and the remote S1 clone in
one step.

Local restore (lines 76-78):

```bash
git -C "$PROJECT_DIR" config --local lfs.fetchexclude "" >/dev/null
git -C "$PROJECT_DIR" lfs pull >/dev/null
verify_local
```

Remote restore (lines 80-102): SSHes into the S1 host, runs the same
`git config --local lfs.fetchexclude ""` and `git lfs pull`, then runs an
inline Python script that verifies the same seven dependency paths exist, are
not symlinks, and do not start with the LFS pointer header.

Usage:

```bash
# Default (auto-detects remote host and repo path)
bash tmux/remote_serving/sync_lecar_s1_assets.sh

# Explicit host and path
bash tmux/remote_serving/sync_lecar_s1_assets.sh tonghez@lecar-s1.ri.cmu.edu /usr0/tonghez/lecar-tbd
```

#### `tmux/table_bussing/launch_table_bussing_remote.sh`

The remote table-bussing launcher (line 272) delegates LFS restoration to
`launch_lecar_s1.sh` when it starts the remote serving session over SSH:

```bash
ssh ... "bash tmux/remote_serving/launch_lecar_s1.sh sam3 anygrasp bundlesdf curobo --no-attach"
```

The local table-bussing launcher (`tmux/table_bussing/launch_table_bussing_local.sh`)
does **not** run `git lfs pull` itself -- it assumes the local clone already has
materialized binaries (either from a prior launcher run or manual pull).

### Option 2: Manual override

If you need to pull LFS objects without a launcher script -- for example, when
setting up a fresh clone or debugging on S1:

```bash
# Inside the repo root
git config --local lfs.fetchexclude ""
git lfs pull
```

The `--local` flag writes to `.git/config`, which overrides `.lfsconfig`
only for that clone.  `.lfsconfig` (repo-wide default) stays `fetchexclude = *`.

To restore lightweight behavior later:

```bash
git config --local --unset lfs.fetchexclude
# or explicitly set it back:
git config --local lfs.fetchexclude "*"
```

---

## Runtime LFS Pointer Resolution in Python

Two Python modules implement their own LFS pointer detection and materialization
at import time, so that services can start even when `git lfs pull` was not run
(as long as the LFS object store under `.git/lfs/objects/` has the data).

### `third_party/bundlesdf/__init__.py`

BundleSDF's `__init__.py` ensures that all `.so` files under
`third_party/bundlesdf/libs/` are usable at import time:

1. **`_resolve_git_lfs_pointer(path)`** (lines 109-137): reads a file's first
   line for the LFS pointer header (`version https://git-lfs.github.com/spec/v1`),
   parses the `oid sha256:` line, and returns the path to the local LFS object
   under `.git/lfs/objects/<oid[:2]>/<oid[2:4]>/<oid>`.

2. **`_materialize_git_lfs_pointer(path)`** (lines 140-148): replaces a pointer
   file or broken symlink with a symlink to the resolved local LFS object.

3. **Import-time loop** (lines 186-188): iterates over every `*.so*` file in
   the libs directory and calls `_materialize_git_lfs_pointer()` on each, so
   pointer files become usable symlinks before `ctypes.CDLL` loads them.

4. **`_resolve_symlink_lfs_target(path)`** (lines 85-106): handles the case
   where a file is already a symlink whose target looks like an LFS object hash
   (64-char hex basename) -- resolves it to the correct `.git/lfs/objects/` path.

### `cap/utils/anygrasp_runtime.py`

AnyGrasp's runtime module provides deeper LFS resolution with historical
fallback:

1. **`_materialize_lfs_pointer(path)`** (lines 78-100): detects LFS pointer
   content, parses the oid, finds the local object, and replaces the pointer
   file with a symlink to the object.

2. **`_historical_lfs_object_for_repo_path(path)`** (lines 103-151): when the
   current tree does not have a usable binary, walks `git log --follow` for
   the file, reads each historical blob, and checks whether a matching LFS
   object exists locally.  This handles cases where a file was re-added or
   the pointer was updated but the old object is still cached.

3. **`_resolve_binary_source(path)`** (lines 154-170): orchestrates resolution
   by trying symlink target, pointer materialization, and (optionally)
   historical fallback.

4. **`_prepare_minkowski_python_root()`** (lines 188-219) and
   **`_prebuilt_python_root("pointnet2")`** (lines 239-269): call
   `_materialize_lfs_pointer()` on the compiled `.so` files before constructing
   the Python import root for MinkowskiEngine and PointNet2.

### `tools/vision/launch_anygrasp_server.sh`

The AnyGrasp launcher shell script has its own `resolve_lfs_pointer()` function
(lines 26-66) and `materialize_lfs_symlink()` (lines 68-80) that perform the
same pointer-to-symlink resolution in bash.  It calls these on the two critical
AnyGrasp extensions (lines 196-197):

```bash
materialize_lfs_symlink "third_party/anygrasp_sdk/dependencies/MinkowskiEngine/build/.../MinkowskiEngineBackend/_C.cpython-311-x86_64-linux-gnu.so"
materialize_lfs_symlink "third_party/anygrasp_sdk/pointnet2/build/.../pointnet2/_ext.cpython-311-x86_64-linux-gnu.so"
```

It also resolves the AnyGrasp checkpoint and license zip through
`resolve_lfs_pointer()` (lines 215-216) so the server receives paths to real
files, not pointers.

---

## What Breaks Without Materialization

If LFS objects are not pulled and the runtime resolution also fails (e.g. the
LFS object store is empty because the clone never fetched them):

| Symptom | Root cause |
|---------|-----------|
| `ImportError: ... anygrasp` / `gsnet` | `gsnet.cpython-*.so` is an LFS pointer, not a real `.so` |
| `ModuleNotFoundError: MinkowskiEngine` | compiled backend `.so` not materialized |
| `FileNotFoundError` for `checkpoint_detection.tar` | checkpoint file is still a pointer |
| Segfault or garbage output from BundleTrack | `libBundleTrack.so` loaded from pointer bytes |
| `FileNotFoundError: BundleSDF runtime libs missing` | `__init__.py` found pointer files and no local LFS objects |

---

## SSH / Manual Deployment on S1

When working on LeCAR-S1 without the launcher (e.g. a manual `git pull` during
debugging), LFS objects will not be restored automatically.  Run the manual
override before starting any serving process:

```bash
cd /path/to/lecar-tbd
git config --local lfs.fetchexclude ""
git lfs pull
# then start services normally
```

Alternatively, call the sync script from the client machine:

```bash
bash tmux/remote_serving/sync_lecar_s1_assets.sh
```

which handles both the local clone and the remote S1 clone in one step.

---

## `.lfsconfig` vs `.git/config` Precedence

| Config location | Scope | Set by |
|-----------------|-------|--------|
| `.lfsconfig` (repo file, line 1-2) | All clones | Committed in repo -- default `fetchexclude = *` |
| `.git/config --local` | This clone only | Launcher scripts + manual override |

Git LFS reads `.git/config` first; if a local override exists it takes
precedence over `.lfsconfig`.  Deleting or unsetting the local value falls
back to the repo-wide default.

---

## File Reference Index

Quick lookup for coding agents and developers:

| File | Lines | Purpose |
|------|-------|---------|
| `.gitattributes` | 1-9 | Declares all LFS-tracked patterns |
| `.lfsconfig` | 1-2 | Sets `fetchexclude = *` repo-wide default |
| `.githooks/pre-push` | 27-29 | Runs `git lfs pre-push` on every push |
| `.githooks/post-merge` | 1-4 | Runs `git lfs post-merge` on every merge/pull |
| `tmux/remote_serving/launch_lecar_s1.sh` | 85-120 | `is_lfs_pointer()`, `ensure_real_dependency()`, `ensure_lfs_dependencies()` |
| `tmux/remote_serving/sync_lecar_s1_assets.sh` | 76-102 | LFS restore + verification on both local and remote clones |
| `tmux/table_bussing/launch_table_bussing_remote.sh` | 258-272 | Delegates remote LFS restore to `launch_lecar_s1.sh` over SSH |
| `tools/vision/launch_anygrasp_server.sh` | 26-80, 196-216 | Bash LFS pointer resolution for AnyGrasp `.so` files and checkpoints |
| `third_party/bundlesdf/__init__.py` | 81-148, 186-188 | Python LFS pointer resolution + `ctypes` preloading for BundleSDF |
| `cap/utils/anygrasp_runtime.py` | 49-170, 188-257 | Python LFS pointer resolution with git-history fallback for AnyGrasp SDK |
