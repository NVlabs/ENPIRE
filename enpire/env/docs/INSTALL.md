# Installation

ENPIRE uses Python 3.11 and `uv`. The core package is deliberately small; robot,
camera, model, simulator, and learner dependencies are selected with extras.

## Clone with submodules first

cuRobo lives at `third_party/curobo` as a git submodule, and `pyproject.toml`
resolves `nvidia-curobo` from that path as an editable dependency. An empty
directory therefore breaks **every** `uv` command in the repo — including
`uv sync --extra dev`, which otherwise has nothing to do with cuRobo:

```
error: Failed to generate package metadata for `nvidia-curobo @ editable+third_party/curobo`
  Caused by: third_party/curobo does not appear to be a Python project, as
  neither `pyproject.toml` nor `setup.py` are present in the directory
```

```bash
git clone --recurse-submodules https://github.com/NVlabs/ENPIRE.git
cd ENPIRE

# or, in an existing clone:
git submodule update --init --recursive
```

Populating the submodule only checks out cuRobo's source. It does **not**
compile any CUDA code — that happens on first use, see
[`CUROBO_SETUP.md`](CUROBO_SETUP.md).

## Core development install

```bash
uv python install 3.11
uv sync --extra dev
uv run enpire doctor
uv run pytest -q tests/enpire
```

## Real YAM practitioner install

`planning-local` installs the vendored Apache-2.0 cuRobo v0.8.0 package with
its CUDA 12 `cuda.core` runtime. Install the full stack with:

```bash
uv sync \
  --extra dev \
  --extra cap \
  --extra vision \
  --extra vision-local \
  --extra grasping-local \
  --extra planning \
  --extra planning-local \
  --extra control-yam \
  --extra control-i2rt \
  --extra camera-realsense \
  --extra calibration \
  --extra real-rl
```

`control-i2rt` is a separate extra and is **not** pulled in by `control-yam`.
Omitting it leaves leader-arm and teaching-handle modes to fail at runtime with
`Leader/teaching-handle mode needs the optional i2rt package`, which is what the
`--use-spacemouse` data-collection workflows depend on.

A working NVIDIA driver must already be present. A system CUDA toolkit is *not*
required for the standard path: `nvidia-curobo[cu12]` brings `cuda-core` and
`nvidia-cuda-nvcc-cu12` in as wheels, and `torch` comes from the `pytorch-cu128`
index, so the toolchain is installed into the virtualenv. The install does not
modify drivers or system CUDA files. See [`CUROBO_SETUP.md`](CUROBO_SETUP.md)
for verification steps and the first-launch Warp compile.

The licensed AnyGrasp SDK, checkpoint, and license archive are not distributed
by this repository, and the license is issued per machine. Set
`ANYGRASP_SDK_ROOT`, `ANYGRASP_CHECKPOINT`, and `ANYGRASP_LICENSE_ZIP` to
externally obtained files before launching the local grasp service — see
[`ANYGRASP_SETUP.md`](ANYGRASP_SETUP.md) for how to apply for a license, build
the SDK, and the license-free 2D alternative.

## Real-world RL environment

ENPIRE's real-world RL environment uses a simplified version of the RL
infrastructure from [PLD (Probe, Learn, Distill)](https://wenlixiao.com/self-improve-VLA-PLD),
introduced in *Self-Improving Vision-Language-Action Models with Data Generation
via Residual RL*. ENPIRE reuses its actor/learner infrastructure for neural
policy improvement within the environment-owned reset, rollout, and verification
loop.

The actor executes the policy using robot observations, while the learner
consumes recorded experience, updates the policy, and sends updated parameters
to the actor. This runtime lives under `enpire/policy/pld/runtime`.

The JAX learner is isolated from the robot environment because its NumPy,
Gymnasium, JAX, and protobuf constraints differ:

```bash
uv sync --project enpire/policy/pld/runtime --extra dev
uv run enpire rl learner --task pin_insertion --dry-run
uv run enpire rl actor --task pin_insertion --dry-run
```

The `--dry-run` commands display the launch commands. For deployment and
actor/learner startup order, follow the
[real-world workflows](REAL_WORLD_WORKFLOWS.md#5-pin-insertion-pld-pipeline)
and [auto-research instructions](../../policy/autoresearch_instruction.md).

## Everything represented by the root lock

```bash
uv sync --all-extras
```

This installs all root capabilities, but not licensed model files, camera SDK
drivers, station calibration, datasets, checkpoints, or the isolated real-world
RL environment. See [DEPENDENCIES.md](DEPENDENCIES.md) for the complete inventory and
[REAL_WORLD_WORKFLOWS.md](REAL_WORLD_WORKFLOWS.md) for station setup.
