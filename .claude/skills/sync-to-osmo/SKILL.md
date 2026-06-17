---
name: sync-to-osmo
description: "Sync forge codebase to OSMO Lustre remote and run experiments. Handles rsync with excludes, remote venv setup, and experiment launch via SSH."
allowed-tools: Bash(rsync *) Bash(ssh *) Bash(fswatch *) Bash(ps *) Bash(pkill *) Bash(cat *) Bash(ls *) Bash(du *) Read Glob Grep
---

# Sync to OSMO

Synchronize the local forge codebase to the OSMO cluster Lustre filesystem and optionally run experiments there.

## Remote target

| Field | Value |
|-------|-------|
| Host | `wenlix_train_1` (SSH config alias) |
| User | `root` |
| Dir | `/mnt/amlfs-02/shared/wenli_vla_ft/forge/` |

The remote mounts Lustre — code lives there, GPU nodes access it directly. No git on remote; all edits happen locally and sync via rsync.

## Key files

- `sync_to_remote.sh` — rsync + fswatch script
- `.rsyncignore` — exclusion list (`.venv/`, `.git/`, `node_modules/`, logs, caches, IDE state)

## Workflow

### Step 1 — Pre-flight checks

1. Verify SSH connectivity:
   ```
   ssh -o ConnectTimeout=5 root@wenlix_train_1 'echo ok'
   ```
2. Read `.rsyncignore` and confirm it exists.
3. Check for secrets that should NOT sync — warn the user if any of these exist outside `.gitignore`:
   - `.env`, `credentials.json`, `secrets.toml`, `*.pem`, `*.key`

### Step 2 — Sync

Run a one-shot rsync:
```bash
rsync -az --delete --stats --exclude-from=".rsyncignore" \
  /Users/wenli-crab-bot/Project/forge/ \
  root@wenlix_train_1:/mnt/amlfs-02/shared/wenli_vla_ft/forge/
```

Report the stats summary (files transferred, total size, speed).

Use `--progress` instead of `--stats` if the user wants verbose output.

### Step 3 — Watch mode (only if user asks)

Start the full script which includes fswatch:
```bash
bash sync_to_remote.sh
```

This runs in foreground — initial sync then watches for changes. Tell the user it's running and they can Ctrl-C to stop.

### Step 4 — Remote venv setup (only on first sync or if user asks)

The `.venv/` is excluded from sync. If the remote doesn't have one yet, guide the user through setup:

```bash
ssh root@wenlix_train_1 'cd /mnt/amlfs-02/shared/wenli_vla_ft/forge/ && \
  uv sync --extra robocasa'
```

For full experiment support (including CAP tools):
```bash
ssh root@wenlix_train_1 'cd /mnt/amlfs-02/shared/wenli_vla_ft/forge/ && \
  uv sync --extra robocasa --extra cap_tools'
```

RoboCasa kitchen assets (needed once):
```bash
ssh root@wenlix_train_1 'cd /mnt/amlfs-02/shared/wenli_vla_ft/forge/ && \
  uv run download-robocasa-assets'
```

### Step 5 — Run experiment on remote (only if user asks)

Available experiments:
- `dry_run` — open/close gripper, test pipeline
- `pick_place_sink_to_counter` — pick from sink, place on counter
- `pick_place_v1` — generic pick-and-place
- `microwave_v1` — microwave task

Run via SSH:
```bash
ssh root@wenlix_train_1 'cd /mnt/amlfs-02/shared/wenli_vla_ft/forge/ && \
  uv run python run_agent.py experiment=<NAME>'
```

With Hydra overrides:
```bash
ssh root@wenlix_train_1 'cd /mnt/amlfs-02/shared/wenli_vla_ft/forge/ && \
  uv run python run_agent.py experiment=pick_place_sink_to_counter env.seed=99 env.layout_id=5'
```

## What gets synced

Everything in the repo **except** (see `.rsyncignore`):
- `.venv/` — set up manually on remote
- `.git/`, `.githooks/` — remote doesn't use git
- `node_modules/`, `.vite/` — install on remote if needed
- `__pycache__/`, `*.pyc`, build artifacts — regenerated
- `logs/`, `outputs/`, `wandb/`, data dirs — runtime outputs
- `.claude/`, `.cursor/`, `.vscode/` — IDE/editor state
- `sync_to_*.sh`, `.rsyncignore` — local-only scripts

All `third_party/` code (robocasa, robosuite, curobo, anygrasp, etc.) IS synced — the remote runs everything independently.

## Important rules

- Always sync **before** running experiments on remote — stale code causes confusing failures.
- Never edit code on the remote — all changes happen locally, then sync.
- The `--delete` flag removes remote files that no longer exist locally. This is intentional for keeping the remote clean, but means local deletions propagate.
- If rsync is slow (>5 min), check if a large file snuck past `.rsyncignore`.
