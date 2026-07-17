# ENPIRE

ENPIRE is a practitioner-oriented harness for repeatable robot policy
improvement. It connects environment-owned reset and verification with
code-as-policy scripts, reusable robot tools, learned policies, and automated
research loops.

```text
register/calibrate → reset → execute → verify → record → refine
```

This repository is being refactored from the original Forge implementations.
The public `enpire.*` package is a thin, tested facade; existing algorithms
remain in `cap/`, `robot/`, and pinned source branches while they are migrated.

## Quickstart

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
uv run enpire doctor
uv run enpire tools list
uv run enpire skills list
uv run enpire examples run 00_hello_environment
```

The example completes a hardware-free reset/execute/verify trial and writes a
structured result under `outputs/hello-environment/`.

## Optional capability installs

Install only what a workstation needs:

```bash
uv sync --extra vision              # clients for remote vision services
uv sync --extra vision-local        # local Torch/Transformers vision
uv sync --extra grasping-local      # local AnyGrasp service (SDK is external)
uv sync --extra planning            # kinematics and planning clients
uv sync --extra planning-local      # local CUDA cuRobo server
uv sync --extra control-yam         # YAM arm control and transport
uv sync --extra camera-realsense    # Intel RealSense runtime
uv sync --extra camera-zed          # Stereolabs Python SDK wheel
uv sync --extra calibration         # ChArUco and hand-eye calibration
uv sync --extra vlm                 # hosted/local VLM clients
uv sync --extra cap                 # source-faithful code-as-policy runner
uv sync --extra real-rl             # robot-side RL bridge and recording
uv sync --extra pld                 # lightweight PLD launcher
uv sync --extra robocasa            # RoboCasa simulation
```

The JAX PLD actor/learner uses its isolated project under
`enpire/policy/pld/runtime`. cuRobo is an explicit root extra and compiles only
when `planning-local` is selected.

For a real station, continue with [installation](docs/INSTALL.md), the
[complete dependency inventory](enpire/env/docs/DEPENDENCIES.md), and the
[real-world workflows](docs/REAL_WORLD_WORKFLOWS.md).

## Repository layout

```text
enpire/
├── env/
│   ├── docs/
│   ├── forge/
│   └── examples/
└── policy/
    ├── autoresearch_instruction.md
    └── interface.py
```

- `enpire/env/forge/` contains the public runtime, registries, artifact store,
  and YAM station support.
- `enpire/env/examples/` is an ordered learning path followed by real-world task
  capsules.
- `enpire/policy/` provides one interface for Python and learned policies.
- `cap/`, `experimental/`, and `robot/` retain source-faithful Forge code behind
  compatibility adapters.

## One-line real-world entries

```bash
uv run enpire services start --profile cap-real
uv run enpire services start --profile robot --confirm-motion
uv run enpire cap run cube-pick --station my-yam --confirm-motion
uv run enpire rl learner --task pin_insertion
uv run enpire rl actor --task pin_insertion
uv run enpire rl control health
uv run enpire rl score --data-dir /external/run --window 50
```

Station profiles, calibration, checkpoints, datasets, licensed model assets,
and credentials remain external to Git.

## Development

```bash
uv run pytest -q tests/enpire
uv run ruff check enpire tests/enpire
```

Hardware, network, integration, and slow tests use explicit pytest markers and
are not part of the default unit-test loop.

Before changing migrated behavior, consult
[`enpire/env/docs/source_provenance.yaml`](enpire/env/docs/source_provenance.yaml)
and add a characterization test. GPU insertion is sourced from
`haotian/gpu-insertion`; zip-tie scripts and reward logic are sourced from
`tonghe/ziptie-autorl`.

## Credentials and hardware safety

Never put API keys, private endpoints, station serials, calibration results,
datasets, or checkpoints in Git. See [`SECURITY.md`](SECURITY.md).

Real-robot execution must remain disabled until the station identity,
calibration, task requirements, and physical emergency stop have been checked.

## Agent usage

Coding agents should start with [`AGENTS.md`](AGENTS.md). Equivalent quickstart
instructions are provided under `.codex/` and `.claude/` so a new agent can
discover installation, tests, source branches, and safety constraints without
private workstation context.

## License status

The root release license and ownership review for migrated Forge and YAM
calibration code are still required before public redistribution. Third-party
components retain their own licenses. The provenance manifest records each
outstanding review explicitly.
