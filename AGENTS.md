# Working on ENPIRE

This file is the canonical quick reference for coding agents and contributors.

## Purpose

ENPIRE makes robot tasks repeatable through an environment-owned loop:

```text
register/calibrate → reset → execute → verify → record → refine
```

Python code is the primary policy representation. Existing Forge tools and CaP
scripts remain the implementation source; `enpire.*` provides stable facades,
packaging, orchestration, and task contracts.

## Safe first commands

```bash
uv sync --extra dev
uv run enpire --version
uv run enpire tools list
uv run enpire examples list
uv run pytest -q tests/enpire
uv run ruff check enpire tests/enpire
```

Optional dependencies are explicit:

```bash
uv sync --extra vision
uv sync --extra vision-local
uv sync --extra grasping-local
uv sync --extra planning
uv sync --extra planning-local
uv sync --extra control-yam
uv sync --extra camera-realsense
uv sync --extra calibration
uv sync --extra vlm
uv sync --extra cap
uv sync --extra real-rl
uv sync --extra pld
uv sync --extra robocasa
```

The JAX PLD learner has an isolated project under
`enpire/policy/pld/runtime`; do not merge its lock into the root environment.
Local cuRobo is the explicit `planning-local` root extra.

## Repository map

```text
enpire/
├── env/
│   ├── docs/       public documentation and provenance
│   ├── forge/      runtime, registries, station and tool adapters
│   └── examples/   learning path and real-world task capsules
└── policy/
    ├── interface.py
    ├── autoresearch_instruction.md
    └── pld/        optional actor/learner implementation
```

The original `cap/`, `experimental/`, and `robot/` packages are compatibility
implementations. Do not duplicate their algorithms in the public facade.

Practitioner commands and external-file requirements are documented in
`docs/REAL_WORLD_WORKFLOWS.md`; direct and native dependencies are documented
in `docs/DEPENDENCIES.md`.

## Implementation rules

1. Pin the source branch and commit in `enpire/env/docs/source_provenance.yaml`.
2. Add a characterization test for existing behavior before refactoring it.
3. Prefer moving code intact or using a thin adapter over rewriting it.
4. Keep optional dependencies lazy; `import enpire` must remain hardware-free.
5. Unit-test pure logic with fakes. Mark real hardware tests with `hardware`.
6. Never embed workstation paths, device serials, IP addresses, or credentials.
7. Real motion is opt-in and must run station, calibration, and safety preflight.
8. Policy code must not modify reset, verification, or safety implementation.

## Credentials

Credentials are supplied only through the process environment or an external
secret manager. Never echo them, serialize them, pass them as CLI arguments, or
write them to `.env`, YAML, test fixtures, agent instructions, or run artifacts.
Default tests use deterministic mocks. Credentialed network tests are opt-in.

## Source branches

- GPU insertion: Forge internal branch `@682f7937`
- Zip-tie scripts/reward: Forge internal branch `@1abbfeae`
- PushT: Forge internal branch `@3cc5e899`
- Pin/AutoRL: Forge internal branch `@4c37817d`
- Calibration: yam-calibration `main@37babca`
- PLD: minimal_policy sources listed in the provenance manifest

Do not assume the target checkout contains the latest task implementation.
