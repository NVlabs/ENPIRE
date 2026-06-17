---
name: sync-docs
description: "Synchronize docs/ with codebase: update stale docs, discover undocumented features, rebuild CLAUDE.md index. Use after implementation work or periodically."
allowed-tools: Agent Read Glob Grep Edit Write Bash(ls *) Bash(git diff*) Bash(git log*) Bash(git status*) Bash(wc *)
---

# Sync Documentation with Codebase

You are a documentation synchronization orchestrator. Your job is to ensure every doc in `docs/` accurately reflects the current codebase, and that every significant feature is documented.

## Workflow

### Phase 1 — Inventory

1. List all `docs/*.md` files.
2. Read `CLAUDE.md` to capture the current Architecture index (the bulleted list under `## Architecture`).
3. Build two lists:
   - **docs_to_audit**: every `docs/*.md` file
   - **indexed_docs**: docs already linked in CLAUDE.md Architecture section

### Phase 2 — Parallel doc audit

For **each** doc in `docs_to_audit`, launch an Agent (subagent_type `general-purpose`) **in parallel** with this brief:

> You are auditing a single design doc for freshness. Your job:
>
> 1. Read the doc at `docs/<NAME>.md` end-to-end.
> 2. Identify every file path, class name, function name, CLI command, config key, or architectural claim the doc makes.
> 3. For each claim, verify it against the **current code** using Grep/Glob/Read. Check that:
>    - Referenced files still exist at the stated paths
>    - Referenced classes/functions still exist and behave as described
>    - CLI flags, env vars, config keys still work as documented
>    - Architectural descriptions (layer relationships, data flow) still match
> 4. Produce a structured report:
>    - **Doc**: `<filename>`
>    - **Status**: `up-to-date` | `needs-update` | `stale` (stale = doc describes something that no longer exists)
>    - **Issues** (list each): `[line ~N] <what the doc says> → <what the code actually does>`
>    - **Suggested fixes**: concrete edits (old text → new text) for each issue
>
> Be thorough but concise. Only flag genuine discrepancies, not stylistic preferences.
> Do NOT edit any files — only report findings.

Launch **all** doc-audit agents in parallel (batch them in a single message with multiple Agent tool calls). Maximize concurrency.

### Phase 3 — Discover undocumented features

Launch **one more Agent** (can run in parallel with Phase 2 agents, or after them if you prefer) with this brief:

> You are scanning the codebase for features that have **no corresponding design doc** in `docs/`.
>
> 1. Read the list of existing docs: $EXISTING_DOCS
> 2. Scan these key areas for significant functionality:
>    - `cap/env/` — env implementations (each env should have integration docs)
>    - `cap/agent/` — agent pipeline, tools, reflection
>    - `cap/server/` — server features, RPC endpoints
>    - `cap/prompt/` — prompt system, loader
>    - `cap/ui/` — UI components and features
>    - `cap/diag/` — diagnostics
>    - `robot/` — hardware drivers
>    - `bringup/` — launcher scripts
>    - `scripts/` — utility scripts
>    - `tools/` — standalone tools (vision, etc.)
>    - `tmux/` — launch configurations
> 3. For each significant feature/subsystem not covered by an existing doc, report:
>    - **Feature**: short name
>    - **Location**: key file paths
>    - **Description**: 1-2 sentence summary of what it does
>    - **Suggested doc name**: `docs/<SUGGESTED_NAME>.md`
>
> Only flag features substantial enough to warrant a design doc (not trivial helpers).
> Do NOT create any files — only report findings.

### Phase 4 — Apply updates

After all agents complete:

1. **Triage results**: Collect all agent reports. Separate into:
   - Docs that need updating (status `needs-update` or `stale`)
   - New docs that should be created
2. **Update stale docs**: For each doc with issues, apply the suggested fixes using Edit. Only fix factual inaccuracies — don't rewrite style or restructure unless the doc is misleading.
3. **Create new docs** (only for clearly missing coverage): Write a short design doc skeleton for each undocumented feature. Use the standard format:
   ```markdown
   # Feature Name

   Brief description.

   ## Overview
   ...

   ## Key Files
   - `path/to/file.py` — description
   ...
   ```
4. **Rebuild CLAUDE.md Architecture index**: Update the bulleted list under `## Architecture` in `CLAUDE.md` to include all docs (existing + new). Format: `- \`docs/NAME.md\` — Short description`. Keep alphabetical order by filename.
5. **Print summary** to the user:
   - How many docs audited
   - How many updated (and which)
   - How many new docs created (and which)
   - Any docs flagged as fully stale / candidates for removal

### Important rules

- **Never delete a doc** without explicit user confirmation — flag it as stale instead.
- **Preserve doc structure and voice** — make surgical fixes, not rewrites.
- **Run ruff** after editing any Python file (the PostToolUse hook handles this automatically).
- If a doc references code that was intentionally removed (not just moved), mark the relevant section with a note rather than deleting it, so the user can decide.
- The CLAUDE.md Architecture section is the canonical index. Every doc in `docs/` MUST have an entry there.
