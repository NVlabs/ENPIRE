# Codex Quickstart

Read [`../AGENTS.md`](../AGENTS.md) before editing. The normal validation loop is:

```bash
uv sync --extra dev
uv run pytest -q tests/enpire
uv run ruff check enpire tests/enpire
uv run enpire tools list
```

Keep `enpire.*` as a thin public layer over characterized Forge code. Consult
`../enpire/env/docs/source_provenance.yaml` before migrating task or robot code.
Use `../docs/REAL_WORLD_WORKFLOWS.md` for supported commands and
`../enpire/env/docs/DEPENDENCIES.md` for install/runtime requirements.

Hardware and network tests are never part of the default loop. Run them only
with explicit user authorization and external credentials/configuration.
