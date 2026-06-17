# Claude Code Project Settings

Configuration for Claude Code hooks, attribution, and plugins in this repo.

**Cross-references**: [CLAUDE.md](../CLAUDE.md) (project rules)

---

## File layout

| File | Checked in | Purpose |
|------|-----------|---------|
| `.claude/settings.json` | Yes | Project-wide hooks, attribution, plugins |
| `.claude/settings.local.json` | No (gitignored) | Per-developer permission overrides |
| `.claude/hooks/pre-commit-docs-check.sh` | Yes | Pre-commit script enforcing Rule 6 |

---

## Hooks

### PreToolUse: docs check before git commit

Enforces **Rule 6** — every commit that touches code must also update `docs/`.

```
Event:    PreToolUse
Matcher:  Bash
If:       Bash(*git commit*)       ← glob, matches anywhere in the command
Script:   .claude/hooks/pre-commit-docs-check.sh
Timeout:  15s
```

**What the script checks:**

1. Scans `git diff --cached --name-only` for staged files
2. Classifies files as "code" or "docs/config" — `docs/*`, `CLAUDE.md`, `.claude/*`, and `*.md` are docs/config; everything else is code
3. If code files are staged but **no `docs/*.md`** files are staged → **blocks** the commit (exit 1)
4. If new `docs/*.md` files are staged but **`docs/index.html` is not** → **blocks** (exit 1, missing wiki index update)
5. Otherwise → allows (exit 0)

**Bypass scenarios** (commit proceeds without docs):
- Docs-only or config-only commits (no code files staged)
- Nothing staged

### PostToolUse: ruff lint + format on Python edits

Auto-runs after any `Edit` or `Write` to `**/*.py` files.

```
Event:    PostToolUse
Matcher:  Edit|Write
If:       Edit(**/*.py)|Write(**/*.py)
Command:  uv run ruff check --fix . && uv run ruff format .
Timeout:  120s
```

Enforces **Rule 3** — run ruff before committing Python changes. By running on every edit, the code is always lint-clean by the time a commit happens.

### SessionStart: docs index listing

Prints the list of all `docs/*.md` files with titles at session start, prompting Claude to acknowledge them per **Rule 7**.

```
Event:    SessionStart
Command:  (lists docs/*.md with titles, appends Rule 7 reminder)
Timeout:  15s
```

### SessionEnd: timestamp log

Logs session end timestamp.

```
Event:    SessionEnd
Command:  echo "Session ended at $(date)"
Timeout:  5s
```

---

## Attribution

```json
"attribution": {
  "commit": "",
  "pr": ""
}
```

Disables the `Co-Authored-By: Claude` trailer on git commits and pull requests. Commit authorship goes to the developer's git config only.

---

## Plugins

| Plugin | Source | Purpose |
|--------|--------|---------|
| `frontend-design` | claude-plugins-official | Production-grade frontend UI generation |
| `code-review` | claude-plugins-official | Code review against project guidelines |
| `pr-review-toolkit` | claude-plugins-official | Comprehensive PR review (type design, tests, silent failures, comments) |
| `babysit-pr` | claude-community | PR monitoring and follow-up |
| `prism` | claude-community | Code analysis |
| `eight-eyes` | claude-community | Multi-perspective code review |

---

## Adding a new hook

1. Write the script in `.claude/hooks/` (make it executable)
2. Add the hook entry in `.claude/settings.json` under the appropriate event
3. Test with a dry run before committing
4. Update this doc

### Hook event reference

| Event | When | Can block? |
|-------|------|-----------|
| `PreToolUse` | Before a tool executes | Yes (exit 1) |
| `PostToolUse` | After a tool succeeds | No |
| `SessionStart` | Session begins/resumes | No |
| `SessionEnd` | Session terminates | No |
| `UserPromptSubmit` | Before user prompt processes | Yes (exit 1) |
| `Stop` | Claude finishes responding | No |

### Matcher syntax

- `"Bash"` — match Bash tool only
- `"Edit|Write"` — match Edit or Write tools
- `"*"` or omitted — match all tools

### If-condition glob syntax

- `Bash(git commit*)` — command starts with `git commit`
- `Bash(*git commit*)` — command contains `git commit` anywhere
- `Edit(**/*.py)` — edit targets any `.py` file
- `Edit(**/*.py)|Write(**/*.py)` — edit or write to `.py` files
