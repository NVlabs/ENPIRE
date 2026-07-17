from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "cap" / "saved_scripts"


def _task_scripts() -> list[Path]:
    return sorted((SCRIPTS / "gpu").glob("*.py")) + sorted(
        (SCRIPTS / "ziptie").rglob("*.py")
    )


def test_gpu_and_ziptie_scripts_compile() -> None:
    for path in _task_scripts():
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_static_load_module_targets_exist() -> None:
    missing: list[tuple[Path, str]] = []
    for path in _task_scripts():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "load_module" or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if not (SCRIPTS / argument.value).is_file():
                    missing.append((path.relative_to(ROOT), argument.value))
    assert not missing

