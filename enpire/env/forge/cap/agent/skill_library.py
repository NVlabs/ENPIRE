"""Episodic skill library for a single agent run.

Manages the per-run skill library stored under ``{run_dir}/skill_library/``.

Directory layout::

    {run_dir}/skill_library/
        namespace.py        # auto-generated at run start, re-exports tool names
        pick_object.py      # append-only, all versions of pick_object
        open_drawer.py      # append-only, all versions of open_drawer
        index.json          # per-skill metadata + stats

Typical lifecycle:

1. At run start, call :meth:`SkillLibrary.generate_namespace` with the tool
   namespace dict so skill files can ``from skill_library.namespace import *``.
2. After each code-generation step, call :meth:`SkillLibrary.append_skill` to
   persist the newly generated skill and update the index.
3. After each rollout, call :meth:`SkillLibrary.update_stats_from_logs` and
   :meth:`SkillLibrary.append_skill_logs` to record execution outcomes.
4. Before the next code-generation prompt, call
   :meth:`SkillLibrary.index_for_prompt` to inject available skills into the
   system prompt.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SkillRouter(Protocol):
    """Pluggable skill selection strategy. Default: return all versions."""

    def select(self, base_name: str, versions: list[dict]) -> list[dict]:
        """Return subset of version entries to surface to the LLM."""
        ...


class AllVersionsRouter:
    """Default router: surfaces all versions to the LLM."""

    def select(self, base_name: str, versions: list[dict]) -> list[dict]:
        return versions


class SkillLibrary:
    """Manages the skill library directory for one agent run.

    Directory layout::

        {run_dir}/skill_library/
            namespace.py        # auto-generated from tool namespace dict
            pick_object.py      # append-only, all versions of pick_object
            open_drawer.py      # append-only, all versions of open_drawer
            index.json          # per-skill metadata + stats
    """

    def __init__(self, run_dir: Path, router: SkillRouter | None = None) -> None:
        self.skill_dir = run_dir / "skill_library"
        self.skill_dir.mkdir(parents=True, exist_ok=True)
        self._router = router or AllVersionsRouter()
        self._index: dict[str, Any] = self._load_index()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def generate_namespace(self, namespace: dict[str, Any]) -> Path:
        """Generate ``namespace.py`` and inject a live counterpart into ``sys.modules``.

        Saved skill files start with ``from skill_library.namespace import *``.
        Python resolves that import by calling ``getattr(module, name)`` for
        every name in ``__all__`` — so the disk file alone is not enough
        (declaring a name in ``__all__`` without defining it raises
        ``AttributeError`` at import time). We do two things:

        1. Write ``namespace.py`` to disk so the package looks valid to
           static tools (IDE jump-to-definition, linters) and so
           ``from skill_library.namespace import *`` has a target even when
           the library is imported from a process that did *not* go through
           this code path (e.g. a fresh Python REPL).
        2. Register a live ``ModuleType`` in ``sys.modules`` with the real
           callables attached as attributes, so ``from ... import *`` in
           this process resolves to working functions. Subprocesses built
           by ``run_script.py`` replicate this injection themselves.

        Args:
            namespace: Mapping of name → callable (or other objects) from the
                current agent tool namespace.

        Returns:
            Path to the written ``namespace.py``.
        """
        import sys
        import types

        tools = [
            k for k, v in namespace.items() if callable(v) and not k.startswith("_")
        ]

        # Disk file — static record of what is available.
        lines = [
            "# namespace.py — auto-generated at run start, do not edit",
            "# Names listed in __all__ are also registered on the live",
            "# skill_library.namespace module in sys.modules for real imports.",
            "__all__ = [",
        ]
        for t in sorted(tools):
            lines.append(f"    {t!r},")
        lines.append("]")
        lines.append("")
        ns_path = self.skill_dir / "namespace.py"
        ns_path.write_text("\n".join(lines), encoding="utf-8")

        # Live module — actual resolution target for `from X import *`.
        mod = types.ModuleType("skill_library.namespace")
        for name in tools:
            setattr(mod, name, namespace[name])
        mod.__all__ = sorted(tools)  # type: ignore[attr-defined]
        sys.modules["skill_library.namespace"] = mod

        logger.info(
            "SkillLibrary: wrote namespace.py and registered %d tools on sys.modules",
            len(tools),
        )
        return ns_path

    def seed_from_dir(self, seed_dir: "str | Path") -> int:
        """Copy human-curated skill files from *seed_dir* into this library.

        Each *.py file (except namespace.py) is copied into skill_dir and its
        functions are indexed as status="verified" so the skill_author prompt
        shows them as trusted starting points.  Already-present files are
        skipped to avoid overwriting agent-authored refinements.

        Returns the number of skill files seeded.
        """
        import shutil as _shutil

        seed_dir = Path(seed_dir)
        if not seed_dir.exists():
            logger.warning("SkillLibrary.seed_from_dir: %s does not exist", seed_dir)
            return 0

        count = 0
        for src in sorted(seed_dir.glob("*.py")):
            if src.name == "namespace.py":
                continue
            dst = self.skill_dir / src.name
            if dst.exists():
                # Already present — just (re-)index without overwriting
                base = src.stem
                self._index_skill_from_file(base, dst, introduced_iteration=-1)
            else:
                _shutil.copy2(src, dst)
                base = src.stem
                self._index_skill_from_file(base, dst, introduced_iteration=-1)
                count += 1

            # Mark all entries from this file as verified (human-curated)
            for name, entry in self._index.items():
                if entry.get("file") == src.name and entry.get("status") == "pending":
                    entry["status"] = "verified"

        if count:
            self._save_index()
            logger.info("SkillLibrary.seed_from_dir: seeded %d file(s) from %s", count, seed_dir)
        return count

    # ------------------------------------------------------------------
    # Skill file management
    # ------------------------------------------------------------------

    def skill_path(self, base_name: str) -> Path:
        """Return the path to the skill file for *base_name*."""
        return self.skill_dir / f"{base_name}.py"

    def author_skill(
        self,
        base_name: str,
        source: str,
        classification: str,
        rationale: str,
        iteration: int,
        parent_name: str | None = None,
    ) -> dict:
        """Write a stage-1 authored skill directly to the library.

        This is the write path for the two-stage code generator (see
        ``docs/plan/two_stage_code_generator.md``). It validates the supplied
        source, classifies it, auto-versions, appends to
        ``skill_library/<base_name>.py``, and annotates the index entry with
        classification metadata for downstream analysis / curation.

        Args:
            base_name: Base skill name (no version suffix), e.g.
                ``"vertical_grasp"``. The versioned name is derived from the
                ``def`` inside *source* and cross-checked against
                *classification*.
            source: Full function source. Must contain exactly one top-level
                ``@skill``-decorated ``def`` whose name is ``<base_name>_vN``.
                May start with the ``@skill`` line — we accept both
                ``@skill\\ndef foo_v1(...)`` and a bare ``def`` (we'll prepend).
            classification: One of ``"new"`` / ``"refine"`` / ``"replace"``.
            rationale: One-sentence justification from the LLM; stored for
                future readers and debugging.
            iteration: Current agent iteration.
            parent_name: For ``refine``/``replace``, the full versioned name of
                the existing skill being refined or replaced.

        Returns:
            The new index entry dict.

        Raises:
            ValueError: On any validation failure (bad classification, name
                mismatch, missing parent, new-but-exists, etc.). Callers
                should feed the message back to the LLM for retry.
            SyntaxError: If *source* doesn't parse.
        """
        if classification not in ("new", "refine", "replace"):
            raise ValueError(
                f"classification must be one of new|refine|replace, got "
                f"{classification!r}"
            )

        # Parse + validate the source.
        tree = ast.parse(source)
        top_level_defs = [
            n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(top_level_defs) != 1:
            raise ValueError(
                f"author_skill expected exactly one top-level def in source, "
                f"found {len(top_level_defs)}"
            )
        fn = top_level_defs[0]
        fn_name = fn.name

        from enpire.env.forge.cap.agent.skill_registry import _parse_name

        parsed_base, parsed_version = _parse_name(fn_name)
        if parsed_base != base_name:
            raise ValueError(
                f"def name {fn_name!r} base {parsed_base!r} does not match "
                f"declared base_name {base_name!r}"
            )

        # Cross-check classification against library state.
        existing_versions = [
            e for e in self._index.values() if e["base_name"] == base_name
        ]
        if classification == "new":
            if existing_versions:
                raise ValueError(
                    f"classification=new but base_name {base_name!r} already "
                    f"has {len(existing_versions)} version(s) in library — "
                    f"use refine or replace instead"
                )
            if parent_name:
                raise ValueError(
                    f"classification=new does not take parent_name, got {parent_name!r}"
                )
        else:  # refine or replace
            if not parent_name:
                raise ValueError(
                    f"classification={classification} requires parent_name"
                )
            if parent_name not in self._index:
                raise ValueError(
                    f"parent {parent_name!r} not found in library — index keys: "
                    f"{sorted(self._index.keys())}"
                )
            if classification == "refine":
                parent_base, _ = _parse_name(parent_name)
                if parent_base != base_name:
                    raise ValueError(
                        f"refine parent {parent_name!r} must share base_name "
                        f"with new skill; got parent_base={parent_base!r} "
                        f"new base_name={base_name!r}"
                    )
                # Auto-correct version: next integer above max existing.
                max_v = max(e["version"] for e in existing_versions)
                if parsed_version <= max_v:
                    logger.info(
                        "author_skill: refine auto-bump %s → %s_v%d",
                        fn_name,
                        base_name,
                        max_v + 1,
                    )
                    new_name = f"{base_name}_v{max_v + 1}"
                    source = source.replace(
                        f"def {fn_name}(", f"def {new_name}(", 1
                    )
                    fn_name = new_name

        # Check for verbatim name collision.
        if fn_name in self._index:
            raise ValueError(
                f"skill name {fn_name!r} already exists in library"
            )

        # Ensure @skill is present as the sole decorator on the def.
        source_lines = source.splitlines()
        decor_names = {
            d.id if isinstance(d, ast.Name) else getattr(d, "attr", None)
            for d in fn.decorator_list
        }
        if "skill" not in decor_names:
            # Insert @skill before the def line.
            def_line_idx = next(
                i for i, ln in enumerate(source_lines) if ln.lstrip().startswith("def ")
            )
            indent = source_lines[def_line_idx][
                : len(source_lines[def_line_idx])
                - len(source_lines[def_line_idx].lstrip())
            ]
            source_lines.insert(def_line_idx, f"{indent}@skill")
        source = "\n".join(source_lines).rstrip() + "\n"

        # Commit to disk + index via the existing append path.
        self.append_skill(base_name, source, iteration)

        # Annotate the new entry with classification metadata.
        entry = self._index.get(fn_name)
        if entry is not None:
            entry["classification"] = classification
            entry["rationale"] = rationale
            if parent_name:
                entry["parent"] = parent_name
            self._save_index()
        logger.info(
            "SkillLibrary.author_skill: %s %s (parent=%s)",
            classification,
            fn_name,
            parent_name,
        )
        return entry or {}

    def append_skill(
        self, base_name: str, source: str, introduced_iteration: int
    ) -> None:
        """Append a new skill version to ``<base_name>.py`` and update the index.

        The file is created with a standard header on first write.  Subsequent
        calls append the new version source, preserving all prior versions.

        Args:
            base_name: Base skill name, e.g. ``"pick_object"``.
            source: Full function source including ``@skill`` decorator and any
                leading imports.
            introduced_iteration: Agent iteration number at which this version
                was introduced (stored in the index for diagnostics).
        """
        path = self.skill_path(base_name)

        # Write header if file is new
        if not path.exists():
            header = (
                f"# {base_name}.py — skill library, append-only\n"
                f"from skill_library.namespace import *  # noqa: F401, F403\n"
                f"from cap.agent.skill_registry import skill\n\n"
            )
            path.write_text(header, encoding="utf-8")

        # Append the new skill source
        existing = path.read_text(encoding="utf-8")
        new_content = existing.rstrip() + "\n\n" + source.strip() + "\n"
        path.write_text(new_content, encoding="utf-8")

        # Parse to get metadata
        self._index_skill_from_file(base_name, path, introduced_iteration)
        self._save_index()
        logger.info("SkillLibrary: appended skill to %s.py", base_name)

    def _index_skill_from_file(
        self, base_name: str, path: Path, introduced_iteration: int
    ) -> None:
        """Parse *path* and upsert index entries for all functions belonging to *base_name*."""
        from enpire.env.forge.cap.agent.skill_registry import _parse_name

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning("SkillLibrary: AST parse failed for %s: %s", path, e)
            return

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            name = node.name
            bn, version = _parse_name(name)
            if bn != base_name:
                continue
            docstring = ast.get_docstring(node) or ""
            start_line = node.lineno
            end_line = node.end_lineno or node.lineno
            line_count = end_line - start_line + 1

            if name not in self._index:
                self._index[name] = {
                    "name": name,
                    "base_name": bn,
                    "version": version,
                    "file": f"{base_name}.py",
                    "start_line": start_line,
                    "end_line": end_line,
                    "line_count": line_count,
                    "docstring": docstring,
                    "introduced_iteration": introduced_iteration,
                    "calls": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "status": "pending",  # pending → verified (≥1 success) → deprecated
                }

    # ------------------------------------------------------------------
    # Stats update (called after each iteration from skill logs)
    # ------------------------------------------------------------------

    # Lifecycle thresholds (tune as needed — kept as class constants so
    # external callers / tests can override without touching logic).
    DEPRECATE_MIN_CALLS = 10
    DEPRECATE_MAX_SUCCESS_RATE = 0.2

    def update_stats_from_logs(self, skill_logs: list[dict]) -> None:
        """Accumulate call/success stats and transition lifecycle state.

        Stats are accumulated across iterations (a single iter's numbers are
        noisy). Lifecycle transitions:

        * ``pending`` → ``verified`` on the first successful call
          (``successes >= 1``).
        * ``verified`` → ``deprecated`` when the skill has been exercised
          enough to judge (``calls >= DEPRECATE_MIN_CALLS``) but keeps
          failing (``success_rate <= DEPRECATE_MAX_SUCCESS_RATE``).

        ``deprecated`` entries stay on disk for provenance but are hidden
        from :meth:`index_for_prompt`, so the LLM never sees broken skills.

        Args:
            skill_logs: List of dicts shaped like::

                {"name": "pick_object_v1", "calls": 5, "success_rate": 0.6, "logs": [...]}
        """
        for entry in skill_logs:
            name = entry.get("name")
            if name not in self._index:
                continue

            iter_calls = int(entry.get("calls", 0) or 0)
            iter_sr = float(entry.get("success_rate", 0.0) or 0.0)
            iter_successes = int(round(iter_sr * iter_calls))

            prev_calls = int(self._index[name].get("calls", 0) or 0)
            prev_successes = int(self._index[name].get("successes", 0) or 0)

            total_calls = prev_calls + iter_calls
            total_successes = prev_successes + iter_successes

            self._index[name]["calls"] = total_calls
            self._index[name]["successes"] = total_successes
            self._index[name]["success_rate"] = (
                round(total_successes / total_calls, 3) if total_calls else 0.0
            )

            # Lifecycle transitions.
            cur_status = self._index[name].get("status", "pending")
            if cur_status == "pending" and total_successes >= 1:
                self._index[name]["status"] = "verified"
                logger.info("SkillLibrary: %s pending → verified", name)
            elif (
                cur_status == "verified"
                and total_calls >= self.DEPRECATE_MIN_CALLS
                and self._index[name]["success_rate"]
                <= self.DEPRECATE_MAX_SUCCESS_RATE
            ):
                self._index[name]["status"] = "deprecated"
                logger.info(
                    "SkillLibrary: %s verified → deprecated (%d calls, rate=%.2f)",
                    name,
                    total_calls,
                    self._index[name]["success_rate"],
                )
        self._save_index()

    def deprecate(self, name: str, reason: str = "") -> bool:
        """Manually retire a skill from the prompt index.

        The source file is kept on disk for provenance (so a later run can
        cite it or resurrect it by explicit call), but :meth:`index_for_prompt`
        will skip it.

        Args:
            name: Full versioned skill name, e.g. ``"vertical_grasp_v2"``.
            reason: Free-form note stored on the entry for future readers.

        Returns:
            True if the entry existed and was marked deprecated.
        """
        if name not in self._index:
            return False
        self._index[name]["status"] = "deprecated"
        if reason:
            self._index[name]["deprecated_reason"] = reason
        self._save_index()
        logger.info("SkillLibrary: %s manually deprecated (%s)", name, reason or "no reason")
        return True

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    # Prompt-index curation limits.
    MAX_INDEX_ENTRIES = 15
    # Per-family cap for the rendered index. Shows the N newest non-deprecated
    # versions; anything older is collapsed into a "… K older omitted" note.
    # 5 lets the assembler see the latest few refinements (which is what our
    # prompt asks it to prefer) without ballooning the prompt when the author
    # has spun out 30+ refinements in a single family.
    MAX_FAMILY_VERSIONS = 5

    def index_for_prompt(self) -> str:
        """Build a curated, ranked skill index for the code-generator prompt.

        Rules (see also the five-layer design in docs/plan/episodic_agent_skill_library.md):

        * **Lifecycle filter** — only ``deprecated`` entries are hidden.
          Pending skills (not yet called or not yet successful) are shown
          with a distinct ⚪ marker so the LLM imports them from iter 1
          onward instead of redefining inline. They graduate to 🟢/🟡
          after one successful call.
        * **Show every non-deprecated version with its stats and rationale.**
          Within a family, list versions newest-first (highest version
          number first) so the author's most recent refinement leads. The
          assembler's prompt tells it to default to the highest untried
          version — that only works if the highest version is actually in
          the index. Cap per-family at ``MAX_FAMILY_VERSIONS`` to keep the
          prompt bounded; anything beyond that is an extreme refinement
          chain that the author almost certainly wrote by mistake.
        * **Ranking** — families sorted by the best version's score
          (verified first, then ``success_rate × log(calls+1)`` descending,
          then recency).
        * **Length cap** — at most ``MAX_INDEX_ENTRIES`` families.

        Format::

            Available skills (import from skill_library.<base_name>):
              🟢 vertical_grasp_v1     [vertical_grasp.py:5-28]   success=0.97  calls=30
                "Descend to obj_pos, compliant-close, verify grasp."
              vertical_place family (3 versions):
                ⚪ vertical_place_v3   [vertical_place.py:60-98]   untested (pending)
                  refine: "hover-first descend for cabinet placement"
                ⚪ vertical_place_v2   [vertical_place.py:30-57]   untested (pending)
                  refine: "raise default z_offset 0.03 -> 0.05"
                🟡 vertical_place_v1   [vertical_place.py:5-27]   success=0.74  calls=750
                  "Lower EE to target_pos + [0,0,z_offset] and open gripper."
        """
        import math

        if not self._index:
            return ""

        # Group non-deprecated entries by base_name.
        by_base: dict[str, list[dict]] = defaultdict(list)
        for e in self._index.values():
            if e.get("status") == "deprecated":
                continue
            by_base[e["base_name"]].append(e)

        if not by_base:
            return ""

        def pick_row(versions: list[dict]) -> dict:
            # Prefer verified over pending; within same status, best rate wins.
            return max(
                versions,
                key=lambda v: (
                    1 if v.get("status") == "verified" else 0,
                    v.get("success_rate", 0.0),
                    v.get("calls", 0),
                ),
            )

        def family_score(versions: list[dict]) -> tuple:
            best = pick_row(versions)
            # Sort tuple: verified first, then by score, then by recency for pending.
            is_verified = 1 if best.get("status") == "verified" else 0
            score = best.get("success_rate", 0.0) * math.log(
                (best.get("calls") or 0) + 1
            )
            recency = best.get("introduced_iteration", 0)
            return (is_verified, score, recency)

        ranked_families = sorted(
            by_base.items(), key=lambda kv: family_score(kv[1]), reverse=True
        )
        if len(ranked_families) > self.MAX_INDEX_ENTRIES:
            ranked_families = ranked_families[: self.MAX_INDEX_ENTRIES]

        def marker(entry: dict) -> str:
            if entry.get("status") != "verified":
                return "⚪"
            rate = entry.get("success_rate", 0.0)
            if rate >= 0.8:
                return "🟢"
            if rate >= 0.5:
                return "🟡"
            return "⚪"

        def fmt_stats(entry: dict) -> str:
            if entry.get("status") == "verified":
                return (
                    f"success={entry['success_rate']:.2f}  calls={entry['calls']}"
                )
            if entry.get("calls", 0) > 0:
                return (
                    f"pending  calls={entry['calls']}  "
                    f"success_rate={entry['success_rate']:.2f}"
                )
            return "untested (pending)"

        def fmt_tag(entry: dict) -> str:
            """One-liner that follows a version row: rationale or docstring."""
            rationale = (entry.get("rationale") or "").strip()
            cls = entry.get("classification")
            if rationale and cls in ("refine", "replace"):
                return f'{cls}: "{rationale[:120]}"'
            if rationale and cls == "new":
                return f'new: "{rationale[:120]}"'
            doc = (entry.get("docstring") or "").strip()
            if doc:
                return f'"{doc.splitlines()[0][:120]}"'
            return ""

        lines = ["Available skills (import from skill_library.<base_name>):"]
        for base_name, versions in ranked_families:
            # Highest version number first so the author's newest refinement
            # leads. The assembler's prompt directive ("prefer highest
            # version, untried == not-failed") only works if the highest
            # version is visible here.
            versions_desc = sorted(
                versions, key=lambda v: v.get("version", 0), reverse=True
            )
            shown = versions_desc[: self.MAX_FAMILY_VERSIONS]
            hidden = len(versions_desc) - len(shown)

            if len(versions_desc) == 1:
                # Single version — render flat (the common case, preserves
                # the old compact format).
                row = shown[0]
                loc = f"{row['file']}:{row['start_line']}-{row['end_line']}"
                lines.append(
                    f"  {marker(row)} {row['name']:<24} [{loc}]   {fmt_stats(row)}"
                )
                tag = fmt_tag(row)
                if tag:
                    lines.append(f"    {tag}")
                continue

            # Multi-version family — header + one indented sub-bullet each.
            lines.append(f"  {base_name} family ({len(versions_desc)} versions):")
            for row in shown:
                loc = f"{row['file']}:{row['start_line']}-{row['end_line']}"
                lines.append(
                    f"    {marker(row)} {row['name']:<24} [{loc}]   {fmt_stats(row)}"
                )
                tag = fmt_tag(row)
                if tag:
                    lines.append(f"      {tag}")
            if hidden > 0:
                lines.append(
                    f"    (… {hidden} older version{'s' if hidden > 1 else ''} "
                    f"omitted — see skill_library/{shown[0]['file']})"
                )

        deprecated = sum(
            1 for e in self._index.values() if e.get("status") == "deprecated"
        )
        if deprecated:
            lines.append(
                f"({deprecated} deprecated entr{'ies' if deprecated > 1 else 'y'} "
                "hidden — known failures.)"
            )

        return "\n".join(lines)

    @property
    def total_skills(self) -> int:
        """Total number of versioned skill entries in the index."""
        return len(self._index)

    @property
    def total_families(self) -> int:
        """Number of distinct ``base_name`` families across all versions."""
        return len({e["base_name"] for e in self._index.values()})

    # ------------------------------------------------------------------
    # Log file management
    # ------------------------------------------------------------------

    def skill_log_path(self, base_name: str) -> Path:
        """Return the path to the JSON log file for *base_name*."""
        return self.skill_dir / f"{base_name}_log.json"

    def append_skill_logs(self, skill_logs: list[dict]) -> None:
        """Append per-skill execution logs to ``<skill_name>_log.json`` files.

        Args:
            skill_logs: List of dicts shaped like::

                {"name": "pick_object_v1", "logs": [{...}, ...]}
        """
        for entry in skill_logs:
            name = entry.get("name", "")
            from enpire.env.forge.cap.agent.skill_registry import _parse_name

            base_name, _ = _parse_name(name)
            log_path = self.skill_log_path(base_name)

            existing: dict[str, Any] = {"skill": base_name, "versions": {}}
            if log_path.exists():
                try:
                    existing = json.loads(log_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

            versions = existing.setdefault("versions", {})
            version_entry = versions.setdefault(name, {"calls": 0, "logs": []})
            version_entry["logs"].extend(entry.get("logs", []))
            version_entry["calls"] = len(version_entry["logs"])

            log_path.write_text(
                json.dumps(existing, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

    def load_skill_logs(self, base_name: str) -> dict[str, list[dict]]:
        """Load execution logs for a skill.

        Args:
            base_name: Base skill name, e.g. ``"pick_object"``.

        Returns:
            Mapping of versioned skill name → list of execution log dicts,
            e.g. ``{"pick_object_v1": [{...}, ...]}``.
        """
        log_path = self.skill_log_path(base_name)
        if not log_path.exists():
            return {}
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            return {
                v: info.get("logs", []) for v, info in data.get("versions", {}).items()
            }
        except (json.JSONDecodeError, OSError):
            return {}

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, Any]:
        index_path = self.skill_dir / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                return {e["name"]: e for e in data.get("skills", [])}
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_index(self) -> None:
        index_path = self.skill_dir / "index.json"
        payload = {
            "skills": sorted(
                self._index.values(), key=lambda e: (e["base_name"], e["version"])
            ),
            "total_skills": len(self._index),
        }
        index_path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def reload(self) -> None:
        """Reload the in-memory index from the on-disk skill library."""
        self._index = self._load_index()
