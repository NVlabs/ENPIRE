"""Regression test for the subprocess-mode reflection bug.

Reproduces the bug observed on iter_000 of logs/20260419T065553_agent_Pick_up_the_object_from_the_sink_and_pla:
the subprocess executor writes exec_NNN/exec.log and exec_NNN/result.json,
but nothing populates ctx.execution_result. Reflection used to read from
ctx.execution_result and get empty strings, leading the LLM to hallucinate
an "import error" even though the real stdout had 13 lines of successful
tool calls.

The fix: reflection reads from the on-disk artifacts via
session.load_run_artifacts(iteration). This test pins that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from cap.agent.agent_session import AgentRunSession


def _write_subprocess_artifacts(
    session: AgentRunSession,
    iteration: int,
    stdout: str,
    error: str = "",
    result: dict | None = None,
) -> None:
    """Mimic what run_script.py writes for one seed."""
    exec_dir = session.exec_dir(iteration, 0)
    parts = []
    if stdout:
        parts.append("--- stdout ---\n" + stdout)
    if error:
        parts.append("--- error ---\n" + error)
    (exec_dir / "exec.log").write_text("\n".join(parts) + "\n", encoding="utf-8")
    (exec_dir / "result.json").write_text(
        json.dumps(result or {"success": False, "score": 0.0}), encoding="utf-8"
    )


def test_load_run_artifacts_reads_subprocess_output(tmp_path: Path) -> None:
    session = AgentRunSession(
        run_dir=tmp_path / "run",
        task="pick up the apple",
        env_name="robocasa:PickPlaceSinkToCounter",
    )

    iter_stdout = (
        "Task: pick 'apple' from sink\n"
        "hover above obj: success=True\n"
        "grasp: success=True, has_object=True\n"
        "lift: success=False, status=Planning_Failed\n"
    )
    _write_subprocess_artifacts(session, iteration=0, stdout=iter_stdout)

    art = session.load_run_artifacts(0)

    assert art.exists
    assert "grasp: success=True" in art.stdout
    assert "Planning_Failed" in art.stdout
    assert art.error == ""
    assert art.result["success"] is False


def test_load_run_artifacts_captures_error_section(tmp_path: Path) -> None:
    """iter_002's real failure: TypeError in vertical_place_v3 — must reach reflection."""
    session = AgentRunSession(
        run_dir=tmp_path / "run",
        task="pick up the apple",
        env_name="robocasa:PickPlaceSinkToCounter",
    )
    stdout = "hover: success=True\ngrasp: success=True\n"
    error = (
        "Traceback (most recent call last):\n"
        "  File 'code.py', line 66, in <module>\n"
        "TypeError: vertical_place_v3() got an unexpected keyword argument 'target_quat'\n"
    )
    _write_subprocess_artifacts(session, iteration=2, stdout=stdout, error=error)

    art = session.load_run_artifacts(2)

    assert "grasp: success=True" in art.stdout
    assert "TypeError" in art.error
    assert "target_quat" in art.error


def test_load_run_artifacts_empty_when_executor_never_ran(tmp_path: Path) -> None:
    session = AgentRunSession(
        run_dir=tmp_path / "run", task="noop", env_name=None
    )
    art = session.load_run_artifacts(0)
    assert not art.exists
    assert art.stdout == ""
    assert art.error == ""
    assert art.result == {}


def test_save_execution_log_round_trips_through_load(tmp_path: Path) -> None:
    """In-process ExecutorStep calls session.save_execution_log(...). That
    path must produce the same exec.log shape that subprocess produces, so
    reflection sees identical artifacts in both modes."""
    from types import SimpleNamespace

    session = AgentRunSession(
        run_dir=tmp_path / "run", task="x", env_name=None
    )
    # Duck-type ExecutionResult: save_execution_log only reads stdout/stderr/error.
    result = SimpleNamespace(
        stdout="hover: success=True\ngrasp: success=True\n",
        stderr="",
        error="Traceback: boom",
    )
    session.save_execution_log(iteration=1, result=result)

    art = session.load_run_artifacts(1)
    assert "grasp: success=True" in art.stdout
    assert "boom" in art.error
