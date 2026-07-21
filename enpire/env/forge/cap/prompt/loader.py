"""Index-based prompt memory backed by markdown files on disk.

PromptMemory manages the ``cap/prompt/`` directory tree.  It scans subfolders
(system, tools, embodiment, task, heuristics) and builds a keyword index so
callers can resolve relevant files quickly without reading every file.

Dependency-free: only uses ``pathlib`` and ``re`` to avoid circular imports
with ``cap.agent``.

Usage::

    from enpire.env.forge.cap.prompt.loader import PromptMemory

    pm = PromptMemory()          # defaults to cap/prompt/
    pm.scan()                    # build keyword index

    # Direct load with variable substitution
    text = pm.load("system", "code_review", task="pick object", code="...")

    # Keyword search
    paths = pm.resolve(["robocasa", "pick"])
    prompt_fragment = pm.inject(paths, mode="full")

    # Embodiment spec (replaces YAML env_spec path)
    spec = pm.load_embodiment("robocasa")
    # -> {"tool_docs": "...", "env_notes": "..."}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


# Default root: cap/prompt/ relative to this file's location
_DEFAULT_ROOT = Path(__file__).resolve().parent

# Subfolder categories
CATEGORIES = ("system", "tools", "embodiment", "task", "heuristics")

# Env name -> embodiment file stem mapping
_EMBODIMENT_MAP: dict[str, str] = {
    "robocasa": "robocasa_panda",
    "robocasa_panda": "robocasa_panda",
    "robocasa_gr1": "robocasa_gr1",
}

# Robot short-name mapping (mirrors env_spec.py logic)
_ROBOT_KEY: dict[str, dict[str, str]] = {
    "robocasa": {
        "PandaOmron": "panda",
        "GR1ArmsOnly": "gr1",
    },
}
_ROBOT_DEFAULT: dict[str, str] = {
    "robocasa": "panda",
}


def _resolve_embodiment_stem(env_name: str) -> str:
    """Map an env name string to an embodiment markdown file stem.

    >>> _resolve_embodiment_stem("robocasa")
    'robocasa_panda'
    >>> _resolve_embodiment_stem("robocasa:PickPlace:GR1ArmsOnly")
    'robocasa_gr1'
    """
    # Check direct mapping first
    if env_name in _EMBODIMENT_MAP:
        return _EMBODIMENT_MAP[env_name]

    parts = env_name.split(":")
    platform = parts[0]
    robot = parts[2] if len(parts) >= 3 else None

    rmap = _ROBOT_KEY.get(platform, {})
    default = _ROBOT_DEFAULT.get(platform, "")
    robot_key = rmap.get(robot, default) if robot else default
    return f"{platform}_{robot_key}" if robot_key else platform


def _extract_title(path: Path) -> str:
    """Extract the first H1/H2 heading from a markdown file, or fall back to stem."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return line.lstrip("# ").strip()
        return path.stem.replace("_", " ").title()
    except OSError:
        return path.stem.replace("_", " ").title()


def _extract_keywords(path: Path) -> list[str]:
    """Extract keywords from filename and first two headings."""
    # Filename keywords: split on _ and .
    stem_words = re.split(r"[_.\-]", path.stem.lower())
    keywords = [w for w in stem_words if len(w) > 1]

    # Heading keywords: first 2 headings
    try:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    heading = re.sub(r"^#+\s*", "", line).strip().lower()
                    words = re.split(r"[\s_\-/,]+", heading)
                    keywords.extend(w for w in words if len(w) > 2)
                    count += 1
                    if count >= 2:
                        break
    except OSError:
        pass

    return list(dict.fromkeys(keywords))  # deduplicate, preserve order


def _extract_section(text: str, heading: str) -> str:
    """Extract content under a specific ## heading from markdown text."""
    pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


class PromptMemory:
    """Index-based prompt manager backed by markdown files on disk.

    Attributes:
        root: Absolute path to the prompt directory (``cap/prompt/``).
        index: Keyword -> list of absolute file paths, built by ``scan()``.
    """

    def __init__(self, prompt_root: Path | None = None) -> None:
        self.root = (prompt_root or _DEFAULT_ROOT).resolve()
        self.index: dict[str, list[Path]] = {}
        self._titles: dict[Path, str] = {}

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, folder: Path | None = None) -> dict[str, list[Path]]:
        """Build keyword index from markdown files in all category subfolders.

        If *folder* is given, only scan that single folder.  Otherwise scan
        all known category subfolders under ``self.root``.

        Returns the built index (also stored as ``self.index``).
        """
        index: dict[str, list[Path]] = {}
        titles: dict[Path, str] = {}

        folders = [folder] if folder else [self.root / c for c in CATEGORIES]

        for d in folders:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.md")):
                abs_path = f.resolve()
                keywords = _extract_keywords(f)
                title = _extract_title(f)
                titles[abs_path] = title

                # Also add the category name as a keyword
                category = d.name
                if category not in keywords:
                    keywords.append(category)

                for kw in keywords:
                    index.setdefault(kw, [])
                    if abs_path not in index[kw]:
                        index[kw].append(abs_path)

        self.index = index
        self._titles = titles
        return index

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, keywords: list[str]) -> list[Path]:
        """Find prompt files matching any of the given keywords.

        Returns a deduplicated, sorted list of absolute paths.
        """
        hits: dict[Path, int] = {}
        for kw in keywords:
            kw_lower = kw.lower()
            for idx_kw, paths in self.index.items():
                if kw_lower in idx_kw or idx_kw in kw_lower:
                    for p in paths:
                        hits[p] = hits.get(p, 0) + 1

        # Sort by match count (descending), then alphabetically
        return sorted(hits, key=lambda p: (-hits[p], str(p)))

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def inject(
        self,
        paths: list[Path],
        mode: Literal["full", "index"] = "full",
    ) -> str:
        """Render a prompt fragment from a list of file paths.

        Args:
            paths: List of absolute paths to markdown files.
            mode: ``"full"`` reads entire file content.
                  ``"index"`` returns only ``path — title`` lines.

        Returns:
            Concatenated prompt text.
        """
        parts: list[str] = []
        for p in paths:
            if mode == "index":
                title = self._titles.get(p, p.stem.replace("_", " ").title())
                parts.append(f"- [{title}]({p})")
            else:
                try:
                    content = p.read_text(encoding="utf-8").strip()
                    parts.append(content)
                except OSError:
                    parts.append(f"(file not found: {p})")
        separator = "\n" if mode == "index" else "\n\n"
        return separator.join(parts)

    # ------------------------------------------------------------------
    # Direct loading
    # ------------------------------------------------------------------

    def load(self, category: str, name: str, **variables: str) -> str:
        """Load a prompt file by category and name, with variable substitution.

        Args:
            category: Subfolder name (e.g. ``"system"``, ``"task"``).
            name: File stem without ``.md`` extension.
            **variables: Template variables to substitute (``{key}`` -> value).

        Returns:
            The rendered prompt text.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self.root / category / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if variables:
            # Use safe substitution: leave unknown {vars} untouched
            for key, value in variables.items():
                text = text.replace(f"{{{key}}}", str(value))
        return text

    def load_section(
        self, category: str, name: str, section: str, **variables: str
    ) -> str:
        """Load a specific ``## Section`` from a prompt file.

        Useful for files that contain multiple prompt variants
        (e.g. ``vision_reflection.md`` with ``## Scene Query`` and
        ``## Analysis Prompt`` sections).
        """
        path = self.root / category / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
        section_text = _extract_section(text, section)
        if not section_text:
            raise ValueError(f"Section '## {section}' not found in {path}")
        if variables:
            for key, value in variables.items():
                section_text = section_text.replace(f"{{{key}}}", str(value))
        return section_text

    # ------------------------------------------------------------------
    # Embodiment
    # ------------------------------------------------------------------

    def load_embodiment(self, env_name: str) -> dict[str, str] | None:
        """Load embodiment markdown and return tool_docs + env_notes.

        Returns ``{"tool_docs": ..., "env_notes": ...}`` or ``None``
        if no matching embodiment file exists.
        """
        stem = _resolve_embodiment_stem(env_name)
        path = self.root / "embodiment" / f"{stem}.md"
        if not path.exists():
            return None

        text = path.read_text(encoding="utf-8")
        tool_docs = _extract_section(text, "Tool API")
        env_notes = _extract_section(text, "Environment Notes")

        # Strip code fences from tool_docs if present
        if tool_docs.startswith("```"):
            lines = tool_docs.splitlines()
            # Remove first ```python and last ```
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            tool_docs = "\n".join(lines)

        return {"tool_docs": tool_docs, "env_notes": env_notes}

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_category(self, category: str) -> list[Path]:
        """List all markdown files in a category subfolder."""
        folder = self.root / category
        if not folder.is_dir():
            return []
        return sorted(f.resolve() for f in folder.glob("*.md"))

    def list_all(self) -> dict[str, list[Path]]:
        """List all markdown files grouped by category."""
        return {cat: self.list_category(cat) for cat in CATEGORIES}

    # ------------------------------------------------------------------
    # Filtering by config
    # ------------------------------------------------------------------

    def selected_files(
        self, category: str, stems: list[str] | None = None
    ) -> list[Path]:
        """Return files for a category, optionally filtered to specific stems.

        - ``None`` → return all files in the category (default).
        - ``[]`` → return nothing (explicitly empty).
        - ``["a", "b"]`` → return only those files (preserving order).
        """
        if stems is not None and len(stems) == 0:
            return []
        all_files = self.list_category(category)
        if stems is None:
            return all_files
        by_stem = {f.stem: f for f in all_files}
        return [by_stem[s] for s in stems if s in by_stem]

    def load_selected(self, category: str, stems: list[str] | None = None) -> str:
        """Load and concatenate files for a category, filtered by stems.

        Convenience method: ``selected_files()`` + ``inject(mode="full")``.
        """
        paths = self.selected_files(category, stems)
        return self.inject(paths, mode="full")

    def build_prompt(self, entries: list[str]) -> str:
        """Load and concatenate prompt files from a list of ``"category/name"`` paths.

        Each entry is a path relative to ``cap/prompt/`` without the ``.md``
        extension.  Example::

            pm.build_prompt([
                "system/robotics_engineer",
                "embodiment/robocasa_panda",
                "task/robocasa_pick_place",
            ])

        Returns the concatenated contents separated by double newlines.
        Missing files are reported as warnings and skipped.
        """
        import logging as _logging

        _logger = _logging.getLogger(__name__)
        parts: list[str] = []
        for entry in entries:
            path = self.root / f"{entry}.md"
            if not path.exists():
                _logger.warning("Prompt file not found: %s", path)
                continue
            parts.append(path.read_text(encoding="utf-8").strip())
        return "\n\n".join(parts)
