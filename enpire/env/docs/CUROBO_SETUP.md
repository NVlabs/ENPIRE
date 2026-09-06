# cuRobo setup

cuRobo provides collision-aware motion planning. ENPIRE talks to it through a
Portal RPC server on port **8611**; `freespace_move` and every other planned
motion depends on it.

This is the one dependency that a coding agent will not set up on its own,
because the first failure looks like a `uv` problem rather than a cuRobo one.

---

## 1. The submodule — do this first

cuRobo is a **git submodule** at `third_party/curobo`, pinned to v0.8.0
(`4ea77366ca48ee453e7df139e39fa6532af49f3b`), and `pyproject.toml` declares it
as an editable path dependency:

```toml
[tool.uv.sources]
nvidia-curobo = { path = "third_party/curobo", editable = true }
```

uv resolves that path on **every** invocation. If the submodule is empty, every
`uv sync` and `uv run` in the repo fails — including ones that have nothing to
do with planning:

```
error: Failed to generate package metadata for `nvidia-curobo @ editable+third_party/curobo`
  Caused by: third_party/curobo does not appear to be a Python project, as
  neither `pyproject.toml` nor `setup.py` are present in the directory
```

```bash
git clone --recurse-submodules https://github.com/NVlabs/ENPIRE.git

# or, in an existing clone:
git submodule update --init --recursive
```

Verify before going further — this must list a commit with no leading `-`:

```bash
git submodule status
#  4ea77366ca48ee453e7df139e39fa6532af49f3b third_party/curobo (v0.8.0)
```

A leading `-` means uninitialized. Checking the submodule out only fetches
source; nothing is compiled yet.

## 2. Install the extras

```bash
uv sync --extra planning --extra planning-local
```

The two extras are deliberately separate:

| Extra | Contains | Needs a GPU? |
|---|---|---|
| `planning` | `mink`, `pyroki`, `mujoco`, Portal client | No — this is the client side, and contains **no cuRobo at all** |
| `planning-local` | `nvidia-curobo[cu12]`, `torch`, `warp-lang==1.12.0`, `ninja` | Yes — this is the CUDA server |

Keeping cuRobo out of `planning` is why a laptop can run the client tooling
without building anything CUDA.

`warp-lang` is pinned to `1.12.0`: it is the version validated against the
cuRobo v0.8 migration, and cuRobo calls Warp APIs that move between releases.
Do not float it.

## 3. What you actually need on the host

**A working NVIDIA driver** — that is the real requirement.

You do **not** need a system CUDA toolkit or `nvcc` on `PATH`. `nvidia-curobo[cu12]`
pulls `cuda-core` and `nvidia-cuda-nvcc-cu12` as wheels, and `torch` comes from
the `pytorch-cu128` index, so the CUDA 12.8 toolchain is installed into the
virtualenv. A machine with no `nvcc` and no `CUDA_HOME` builds and imports
cuRobo successfully.

> Some older notes in this repo say "a compiler, NVIDIA CUDA toolkit, and
> matching driver must already be present." Only the driver part is required
> for the standard `uv sync` path.

Check the driver:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Then confirm torch sees the GPU:

```bash
uv run --extra planning --extra planning-local python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.11.0+cu128 12.8 True
```

`torch.cuda.is_available()` printing `False` is a driver/GPU problem, not a
cuRobo one — fix it before continuing.

## 4. Verify cuRobo imports

```bash
uv run --extra planning --extra planning-local python -c \
  "import curobo, curobo.types, curobo.scene, curobo.motion_planner; print('ok')"
```

cuRobo v0.8 moved its package from `src/curobo` to `curobo/` at the repository
root, so imports are `curobo.motion_planner`, not `curobo.types.base` (a v0.6
path that no longer exists).

Then the ENPIRE planner wrapper:

```bash
uv run --extra planning --extra planning-local --extra control-yam python -c \
  "from enpire.env.forge.experimental.motion_planner_curobo import YamMotionPlannerCurobo; print('ok')"
```

If this raises `RuntimeError: Failed to import cuRobo. Install it into the same
Python env or keep third_party/curobo available in this repo...`, return to
step 1 — the submodule is almost certainly empty.

Imports succeed without touching the GPU. Constructing a planner is what first
requires a working driver.

## 5. Start the service

```bash
uv run enpire services start --services sam3,curobo,yam \
  --station my-yam-station-name --confirm-motion
```

Start every service you need in **one** call — `services start` refuses to reuse
an existing tmux session, so a second call aborts with `tmux session 'enpire'
already exists` having started nothing.

The server is `enpire.env.forge.experimental.serve_portal_motion_planner`,
launched with `--port 8611 --solver-speed fast --robot-type yam`.

## 6. First launch is slow — this is normal

cuRobo JIT-compiles its Warp kernels the first time the planner is constructed.
Expect **tens of seconds**, during which the process prints nothing, and
`~/.cache/warp/<version>` grows to a few hundred MB. Later starts reuse that
cache and are fast.

Until port 8611 is listening, `freespace_move` blocks retrying the connection
and **the arm never moves, with no error message**. Always confirm the port is
up before running a task:

```bash
ss -ltn | grep 8611
```

> Port 8611 is **Portal RPC, not HTTP**. `curl localhost:8611/health` will not
> work — unlike `sam3` on 6767, which is a real HTTP service. Use `ss` to check
> cuRobo.

Deleting `~/.cache/warp` forces a recompile; do that if kernels are stale after
changing the Warp or cuRobo version.

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `does not appear to be a Python project` on any `uv` command | Submodule empty — step 1. |
| `RuntimeError: Failed to import cuRobo` | Same, seen at runtime instead of install time. |
| `ModuleNotFoundError: No module named 'curobo.types.base'` | v0.6 import path; use `curobo.types` (step 4). |
| `torch.cuda.is_available()` is `False` | Driver/GPU issue, unrelated to cuRobo. |
| `freespace_move` hangs silently | Port 8611 not listening yet — still JIT-compiling, or the pane died. Check with `ss`, then read the tmux pane. |
| Warp API errors after an upgrade | `warp-lang` floated off `1.12.0`. Re-pin. |
| `tmux session 'enpire' already exists` | Start all services in one call, or `tmux kill-session -t enpire`. |

---

## Related

- [`INSTALL.md`](INSTALL.md) — full install matrix
- [`DEPENDENCIES.md`](DEPENDENCIES.md) — dependency and licensing overview
- [`ANYGRASP_SETUP.md`](ANYGRASP_SETUP.md) — the other externally-supplied dependency
- [`remote_serving.md`](remote_serving.md) — running the planner on a separate GPU host
