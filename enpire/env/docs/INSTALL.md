# Installation

ENPIRE uses Python 3.11 and `uv`. The core package is deliberately small; robot,
camera, model, simulator, and learner dependencies are selected with extras.

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
  --extra camera-realsense \
  --extra calibration \
  --extra real-rl
```

A compiler, NVIDIA CUDA toolkit, and matching driver must already be present.
The install does not modify drivers or system CUDA files.

The licensed AnyGrasp SDK, checkpoint, and license archive are not distributed
by this repository. Set `ANYGRASP_SDK_ROOT`, `ANYGRASP_CHECKPOINT`, and
`ANYGRASP_LICENSE_ZIP` to externally obtained files before launching the local
grasp service.

## PLD learner/actor runtime

The JAX learner is isolated from the robot environment because its NumPy,
Gymnasium, JAX, and protobuf constraints differ:

```bash
uv sync --project enpire/policy/pld/runtime --extra dev
uv run enpire rl learner --task pin_insertion --dry-run
uv run enpire rl actor --task pin_insertion --dry-run
```

## Everything represented by the root lock

```bash
uv sync --all-extras
```

This installs all root capabilities, but not licensed model files, camera SDK
drivers, station calibration, datasets, checkpoints, or the isolated PLD
runtime. See [DEPENDENCIES.md](DEPENDENCIES.md) for the complete inventory and
[REAL_WORLD_WORKFLOWS.md](REAL_WORLD_WORKFLOWS.md) for station setup.
