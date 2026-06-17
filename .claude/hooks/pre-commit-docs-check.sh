#!/usr/bin/env bash
# Pre-commit hook: enforce Rule 6 — docs must be updated alongside code changes.
# Called by Claude Code PreToolUse hook before git commit.
# Exit 0 = allow commit, exit 1 = block with message.

set -euo pipefail

# Get staged files
staged=$(git diff --cached --name-only 2>/dev/null || true)
if [ -z "$staged" ]; then
  exit 0  # nothing staged, allow
fi

# Check if any code files (non-docs, non-config) are staged
code_changed=false
while IFS= read -r f; do
  case "$f" in
    docs/*|CLAUDE.md|.claude/*|*.md) ;;  # skip doc/config files
    *) code_changed=true; break ;;
  esac
done <<< "$staged"

if ! $code_changed; then
  exit 0  # docs-only commit, allow
fi

# Code files are staged — check if docs/ were also updated
docs_changed=false
docs_index_changed=false
while IFS= read -r f; do
  case "$f" in
    docs/*.md) docs_changed=true ;;
    docs/index.html) docs_index_changed=true ;;
  esac
done <<< "$staged"

if ! $docs_changed; then
  echo "BLOCKED: Code files are staged but no docs/*.md files were updated."
  echo ""
  echo "Per Rule 6: Always update docs/ after implementation or code design work."
  echo "  - Create or update the relevant doc in docs/"
  echo "  - Update docs/index.html if adding a new doc"
  echo "  - Update the Architecture section in CLAUDE.md if adding a new doc"
  echo ""
  echo "Staged code files:"
  echo "$staged" | grep -v '^docs/' | grep -v '^CLAUDE.md' | grep -v '^\.claude/' | sed 's/^/  /'
  exit 1
fi

# Docs changed — check if new docs were added without index update
new_docs=$(git diff --cached --name-only --diff-filter=A -- 'docs/*.md' 2>/dev/null || true)
if [ -n "$new_docs" ] && ! $docs_index_changed; then
  echo "WARNING: New doc(s) added but docs/index.html was not updated:"
  echo "$new_docs" | sed 's/^/  /'
  echo ""
  echo "Update the DOCS array in docs/index.html to include the new doc(s)."
  exit 1
fi

exit 0
