# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentRunSession — unified log folder for one agent run.

Layout v2 — organized subdirectories::

    {run_dir}/
      config.yaml                          # experiment config
      metadata.json                        # run summary
      snapshots/
        baseline/skill_library/            # last kept skill-library state
        trials/trial_NNN/skill_library/    # archived trial state per iteration

      iterations/
        iter_000/
          code.py                          # generated code
          thoughts.md                      # LLM reasoning
          reflection.md                    # improvement feedback
          review.md                        # code review (optional)
          vision_scene.md                  # VLM scene description (optional)
          visual_diff.md                   # before/after VLM diff (optional)
          eval_summary.json                # aggregated results across N seeds
          exec_000/                        # seed=0 execution (via run_script.py)
            result.json
            exec.log
            profiling.txt
            video/ frames_after/
          exec_001/                        # seed=1 execution
            ...

      results/                             # experiment-level aggregation
        summary.json
        reward_trace.json
        tool_calls.json

      conversations/                       # LLM I/O
      vis/                                 # detection/segmentation images
      vlm/                                 # VLM query artifacts
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.agent_context import AgentEvaluation, ExecutionMemory
    from enpire.env.forge.cap.agent.executor import ExecutionResult
    from enpire.env.forge.cap.server.cap_server import CapServer

LAYOUT_VERSION = 2


# ---------------------------------------------------------------------------
# Post-execution artifacts (single source of truth, read from disk)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunArtifacts:
    """Canonical post-execution artifacts for one seed of one iteration.

    Read from disk so consumers (reflection, retry, wandb) speak one shape
    regardless of whether the executor was in-process or subprocess.
    ``exists`` is False when no artifacts were written (e.g. executor never
    ran); callers should treat the other fields as empty but safe to render.
    """

    stdout: str
    error: str
    result: dict  # success/score/feedback/method/details (from result.json)
    exists: bool


@dataclass(frozen=True)
class SnapshotRecord:
    """One saved trial snapshot and its keep/discard decision."""

    iteration: int
    score: float
    success: bool
    decision: str
    timestamp: str
    trial_dir: str
    baseline_dir: str


def _split_exec_log(text: str) -> tuple[str, str]:
    """Split ``exec.log`` (as written by run_script.py) into stdout and error.

    Layout written by run_script.py::

        --- stdout ---
        <stdout text>
        --- error ---
        <traceback, may be absent>

    If the markers are not present (e.g. legacy log format), the whole text
    is returned as stdout.
    """
    stdout_tag = "--- stdout ---"
    error_tag = "--- error ---"
    si = text.find(stdout_tag)
    ei = text.find(error_tag)
    if si == -1 and ei == -1:
        return text.strip(), ""
    stdout_text = ""
    error_text = ""
    if si != -1:
        end = ei if ei != -1 and ei > si else len(text)
        stdout_text = text[si + len(stdout_tag) : end].strip()
    if ei != -1:
        error_text = text[ei + len(error_tag) :].strip()
    return stdout_text, error_text


def _format_exec_log(stdout: str, error: str) -> str:
    """Inverse of _split_exec_log — write the canonical exec.log format."""
    parts: list[str] = []
    if stdout:
        parts.append("--- stdout ---\n" + stdout.rstrip())
    if error:
        parts.append("--- error ---\n" + error.rstrip())
    return "\n".join(parts) + ("\n" if parts else "")


class AgentRunSession:
    """Manages a single agent run's output directory and all log files.

    Usage::

        session = AgentRunSession.create(
            task="Pick up the red object",
            env_name="yam",
            log_dir=Path("logs"),
        )
        # ... run pipeline ...
        session.close()
    """

    def __init__(
        self,
        run_dir: Path,
        task: str,
        env_name: str | None,
        agent_type: str = "agent",
    ) -> None:
        self.run_dir = run_dir
        self.task = task
        self.env_name = env_name
        self.agent_type = agent_type
        self._start_time = time.time()
        self._recorder = None
        self._iter_recorder = None
        self._reward_trace: list[dict[str, Any]] = []
        self._skill_library: Any = None  # SkillLibrary, set after namespace is ready
        self._snapshot_root = self.run_dir / "snapshots"
        self._baseline_score: float = 0.0
        self._baseline_iteration: int | None = None
        self._snapshot_records: list[SnapshotRecord] = []

    @classmethod
    def create(
        cls,
        task: str,
        env_name: str | None = None,
        log_dir: Path | str = Path("logs"),
        agent_type: str = "agent",
    ) -> AgentRunSession:
        """Create a new session with a timestamped directory."""
        log_dir = Path(log_dir)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        safe_task = re.sub(r"[^\w\-]", "_", task)[:40].strip("_")
        run_dir = log_dir / f"{stamp}_{agent_type}_{safe_task}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[session] Log dir: {run_dir}")
        return cls(run_dir=run_dir, task=task, env_name=env_name, agent_type=agent_type)

    # ------------------------------------------------------------------
    # Directory getters (lazy mkdir)
    # ------------------------------------------------------------------

    def iterations_dir(self, iteration: int) -> Path:
        """Return ``iterations/iter_NNN/``, creating it if needed."""
        d = self.run_dir / "iterations" / f"iter_{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exec_dir(self, iteration: int, exec_id: int) -> Path:
        """Return ``iterations/iter_NNN/exec_MMM/``, creating it if needed."""
        d = self.iterations_dir(iteration) / f"exec_{exec_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def results_dir(self) -> Path:
        """Return ``results/``, creating it if needed."""
        d = self.run_dir / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def snapshots_dir(self) -> Path:
        """Return ``snapshots/`` for baseline and trial state archives."""
        d = self._snapshot_root
        d.mkdir(parents=True, exist_ok=True)
        return d

    def baseline_snapshot_dir(self) -> Path:
        """Return ``snapshots/baseline/``."""
        d = self.snapshots_dir() / "baseline"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def baseline_skill_library_dir(self) -> Path:
        """Return the archived baseline skill-library directory."""
        d = self.baseline_snapshot_dir() / "skill_library"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def trial_snapshot_dir(self, iteration: int) -> Path:
        """Return ``snapshots/trials/trial_NNN/``."""
        d = self.snapshots_dir() / "trials" / f"trial_{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def baseline_code_path(self) -> Path:
        """Return the baseline assembly code.py path."""
        return self.baseline_snapshot_dir() / "code.py"

    def trial_code_path(self, iteration: int) -> Path:
        """Return the archived trial assembly code.py path."""
        return self.trial_snapshot_dir(iteration) / "code.py"

    def trial_skill_library_dir(self, iteration: int) -> Path:
        """Return the archived trial skill-library directory."""
        d = self.trial_snapshot_dir(iteration) / "skill_library"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def snapshot_state_path(self) -> Path:
        return self.snapshots_dir() / "state.json"

    def conversations_dir(self) -> Path:
        """Return ``conversations/``, creating it if needed."""
        d = self.run_dir / "conversations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # File path getters
    # ------------------------------------------------------------------

    def code_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "code.py"

    def thoughts_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "thoughts.md"

    def reflection_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "reflection.md"

    def review_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "review.md"

    def vision_scene_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "vision_scene.md"

    def visual_diff_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "visual_diff.md"

    def per_seed_reflections_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "per_seed_reflections.md"

    def eval_summary_path(self, iteration: int) -> Path:
        return self.iterations_dir(iteration) / "eval_summary.json"

    def tool_calls_path(self) -> Path:
        return self.results_dir() / "tool_calls.json"

    def reward_trace_path(self) -> Path:
        return self.results_dir() / "reward_trace.json"

    # ------------------------------------------------------------------
    # Code and iteration-level logs
    # ------------------------------------------------------------------

    def save_code(self, iteration: int, code: str) -> Path:
        path = self.code_path(iteration)
        path.write_text(code, encoding="utf-8")
        return path

    def save_review(self, iteration: int, feedback: str) -> Path:
        path = self.review_path(iteration)
        path.write_text(feedback, encoding="utf-8")
        return path

    def save_per_seed_reflections(
        self,
        iteration: int,
        task: str,
        per_seed: list[dict[str, Any]],
        reflections_by_seed: dict[int, str],
        backend: str = "",
    ) -> Path:
        """Write Phase A output (per-seed VLM reflections + result summaries) to disk.

        Survives the later ``SelfReflectionStep`` overwrite of
        ``ctx.evaluation.feedback`` — so the raw per-seed evidence is always
        inspectable under ``iter_NNN/per_seed_reflections.md`` even after
        Phase B's synthesis replaces the in-memory feedback.
        """
        n = len(per_seed)
        successes = sum(1 for r in per_seed if r.get("success"))
        avg_score = sum(r.get("score", 0.0) for r in per_seed) / n if n else 0.0

        lines: list[str] = [
            f"# Per-seed Reflections — Iteration {iteration}",
            "",
            f"**Task:** {task}",
            f"**Seeds:** {n}  |  **Successes:** {successes}/{n}  "
            f"|  **Avg score:** {avg_score:.3f}",
        ]
        if backend:
            lines.append(f"**VLM backend:** {backend}")
        lines.append("")

        for r in per_seed:
            seed = r.get("seed", -1)
            status = "SUCCESS" if r.get("success") else "FAIL"
            score = r.get("score", 0.0)
            fb = r.get("feedback", "") or ""
            refl = reflections_by_seed.get(int(seed), "") if seed != -1 else ""

            lines.append(f"## Seed {seed} — {status} (score={score:.3f})")
            lines.append("")
            if refl:
                lines.append("### VLM reflection")
                lines.append(refl.strip())
                lines.append("")
            if fb:
                lines.append("### Executor feedback (result.json + stdout tail)")
                lines.append("```")
                lines.append(fb.strip())
                lines.append("```")
                lines.append("")

        path = self.per_seed_reflections_path(iteration)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def save_eval_summary(
        self,
        iteration: int,
        per_seed: list[dict[str, Any]],
    ) -> Path:
        """Write aggregated eval across N seed executions.

        ``per_seed`` is a list of dicts, each with at least
        ``success`` (bool) and ``score`` (float).
        """
        n = len(per_seed)
        successes = sum(1 for r in per_seed if r.get("success"))
        scores = [r.get("score", 0.0) for r in per_seed]
        summary = {
            "n_seeds": n,
            "success_rate": successes / n if n else 0.0,
            "avg_score": sum(scores) / n if n else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "successes": successes,
            "per_seed": per_seed,
        }
        path = self.eval_summary_path(iteration)
        path.write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return path

    # ------------------------------------------------------------------
    # Snapshot workspace (baseline / trial rollback without git)
    # ------------------------------------------------------------------

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """Replace ``dst`` with a copy of ``src``."""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    def _write_snapshot_state(self) -> None:
        self.snapshots_dir()
        state = {
            "baseline_score": self._baseline_score,
            "baseline_iteration": self._baseline_iteration,
            "trials": [record.__dict__ for record in self._snapshot_records],
        }
        self.snapshot_state_path().write_text(
            json.dumps(state, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def initialize_skill_library_baseline(self) -> None:
        """Seed the baseline snapshot from the current active skill library."""
        sl = getattr(self, "_skill_library", None)
        if sl is None:
            return
        active_dir = Path(sl.skill_dir)
        if not active_dir.exists():
            return
        self.snapshots_dir()
        self._copy_dir(active_dir, self.baseline_skill_library_dir())
        self._baseline_score = 0.0
        self._baseline_iteration = None
        self._snapshot_records = []
        self._write_snapshot_state()

    def record_skill_library_trial(
        self,
        iteration: int,
        *,
        score: float,
        success: bool,
    ) -> str:
        """Archive the active skill library and keep/discard it by score.

        Returns:
            ``"keep"`` or ``"discard"``. The first kept trial becomes the new
            baseline. If the trial loses, the active skill library is restored
            from ``snapshots/baseline`` so the next iteration starts from the
            last accepted state.
        """
        sl = getattr(self, "_skill_library", None)
        if sl is None:
            return "discard"

        active_dir = Path(sl.skill_dir)
        if not active_dir.exists():
            return "discard"

        trial_dir = self.trial_skill_library_dir(iteration)
        self._copy_dir(active_dir, trial_dir)

        # Snapshot this iteration's assembly code.py alongside the skill library.
        active_code = self.iterations_dir(iteration) / "code.py"
        if active_code.exists():
            import shutil as _shutil
            self.trial_snapshot_dir(iteration).mkdir(parents=True, exist_ok=True)
            _shutil.copy2(active_code, self.trial_code_path(iteration))

        keep = score > self._baseline_score
        decision = "keep" if keep else "discard"
        if keep:
            self._copy_dir(active_dir, self.baseline_skill_library_dir())
            # Promote this iteration's code.py to baseline.
            if active_code.exists():
                import shutil as _shutil
                self.baseline_snapshot_dir().mkdir(parents=True, exist_ok=True)
                _shutil.copy2(active_code, self.baseline_code_path())
            self._baseline_score = float(score)
            self._baseline_iteration = iteration
        else:
            baseline_dir = self.baseline_skill_library_dir()
            if baseline_dir.exists():
                self._copy_dir(baseline_dir, active_dir)
                try:
                    sl.reload()
                except Exception:
                    # Reload is a convenience for the in-process prompt path.
                    # The on-disk restore still completed successfully.
                    pass
            # No revert needed for code.py — assembly is regenerated each iteration.
            # The baseline code.py is surfaced to the assembly_generator via the
            # prompt (=== CHAMPION ATTEMPT ===) and its snapshot path.

        record = SnapshotRecord(
            iteration=iteration,
            score=float(score),
            success=bool(success),
            decision=decision,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            trial_dir=str(trial_dir),
            baseline_dir=str(self.baseline_skill_library_dir()),
        )
        self._snapshot_records.append(record)
        self._write_snapshot_state()
        return decision

    # ------------------------------------------------------------------
    # Post-execution artifacts (canonical on-disk source of truth)
    # ------------------------------------------------------------------

    def load_run_artifacts(
        self, iteration: int, exec_id: int = 0
    ) -> RunArtifacts:
        """Read the canonical execution artifacts for one seed.

        Same shape whether the executor was in-process or subprocess. The
        subprocess executor writes ``exec_NNN/exec.log`` + ``result.json``
        directly; the in-process ``ExecutorStep`` writes the same layout
        via :meth:`save_execution_log`.
        """
        exec_dir = self.exec_dir(iteration, exec_id)

        stdout = ""
        error = ""
        exec_log = exec_dir / "exec.log"
        if exec_log.exists():
            stdout, error = _split_exec_log(exec_log.read_text(encoding="utf-8"))

        result: dict = {}
        result_json = exec_dir / "result.json"
        if result_json.exists():
            try:
                result = json.loads(result_json.read_text(encoding="utf-8"))
            except Exception:
                result = {}

        return RunArtifacts(
            stdout=stdout,
            error=error,
            result=result,
            exists=exec_log.exists() or result_json.exists(),
        )

    def save_execution_log(
        self, iteration: int, result: ExecutionResult, exec_id: int = 0
    ) -> Path:
        """Write ``exec_NNN/exec.log`` in the canonical format used by
        :meth:`load_run_artifacts`.

        Called by the in-process ``ExecutorStep``. The subprocess path
        writes the same file directly from ``run_script.py``.
        """
        stdout = (result.stdout or "") + (
            ("\n" + result.stderr) if result.stderr else ""
        )
        error = result.error or ""
        path = self.exec_dir(iteration, exec_id) / "exec.log"
        path.write_text(_format_exec_log(stdout, error), encoding="utf-8")
        return path

    def save_evaluation(self, iteration: int, evaluation: AgentEvaluation) -> Path:
        """Legacy: write single eval into iteration dir (inline executor)."""
        path = self.iterations_dir(iteration) / "eval.json"
        path.write_text(
            json.dumps(evaluation.to_dict(), indent=2),
            encoding="utf-8",
        )
        return path

    # ------------------------------------------------------------------
    # Profiling (tool call timing + reward trace)
    # ------------------------------------------------------------------

    def enable_profiling(
        self,
        state_fn: Callable | None = None,
        get_task_info_fn: Callable | None = None,
    ) -> Path:
        """Enable profiler file logging and reward tracing.

        Returns the profiler log path.
        """
        from enpire.env.forge.cap.agent.profiler import (
            enable_file_logging,
            set_state_fn,
            set_tool_event_hooks,
        )

        log_path = enable_file_logging(str(self.run_dir), "agent")
        if state_fn is not None:
            set_state_fn(state_fn)

        # Reward trace: sample get_task_info() after each motion tool
        _MOTION_TOOLS = {
            "_ik_servo",
            "set_gripper",
            "open_gripper",
            "close_gripper",
            "go_home",
        }
        reward_trace = self._reward_trace

        def _on_tool_end(
            name: str, call_id: int, result: Any, error: Any, elapsed_ms: float
        ) -> None:
            if name in _MOTION_TOOLS and get_task_info_fn is not None:
                try:
                    info = get_task_info_fn()
                    reward_trace.append(
                        {
                            "call_id": call_id,
                            "tool": name,
                            "elapsed_ms": elapsed_ms,
                            "reward": info.get("reward"),
                            "success": info.get("success"),
                            "obj_pos": info.get("obj_pos"),
                            "timestamp": time.time(),
                        }
                    )
                except Exception:
                    pass

        set_tool_event_hooks(on_end=_on_tool_end)
        return log_path

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(self, server: CapServer) -> None:
        """Start a whole-run recorder into video/ (used by run_script.py)."""
        from enpire.env.forge.cap.agent.recorder import ScriptRecorder

        video_dir = self.run_dir / "video"
        self._recorder = ScriptRecorder(server, video_dir)
        self._recorder.start()

    def stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None

    def start_iter_recording(self, server: CapServer, iteration: int) -> None:
        """Start a per-iteration recorder into video/iter_NNN/ (CapServer mode)."""
        from enpire.env.forge.cap.agent.recorder import ScriptRecorder

        self.stop_iter_recording()
        video_dir = self.run_dir / "video" / f"iter_{iteration:03d}"
        self._iter_recorder = ScriptRecorder(server, video_dir)
        self._iter_recorder.start()

    def start_iter_recording_env(self, env: object, iteration: int) -> None:
        """Start a per-iteration recorder from env directly (no CapServer)."""
        from enpire.env.forge.cap.agent.recorder import ScriptRecorder

        self.stop_iter_recording()
        video_dir = self.run_dir / "video" / f"iter_{iteration:03d}"
        cam_names = getattr(env, "_profile", None)
        cam_names = cam_names.camera_names if cam_names else ["top", "wrist"]
        self._iter_recorder = ScriptRecorder.from_env(
            camera_names=list(cam_names),
            output_dir=video_dir,
        )
        self._iter_env = env
        env.set_recorder(self._iter_recorder)
        self._iter_recorder.start()

    def stop_iter_recording(self) -> None:
        if self._iter_recorder is not None:
            if hasattr(self, "_iter_env") and self._iter_env is not None:
                self._iter_env.set_recorder(None)
                self._iter_env = None
            self._iter_recorder.stop()
            self._iter_recorder = None

    # ------------------------------------------------------------------
    # Memory and metadata
    # ------------------------------------------------------------------

    def save_tool_calls(self, memory: ExecutionMemory) -> None:
        path = self.tool_calls_path()
        path.write_text(
            json.dumps(memory.to_json(), indent=2),
            encoding="utf-8",
        )

    def save_metadata(self, data: dict[str, Any]) -> None:
        base = {
            "layout_version": LAYOUT_VERSION,
            "task": self.task,
            "env": self.env_name,
            "start_time": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self._start_time)
            ),
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": round(time.time() - self._start_time, 1),
            "snapshot_state": {
                "baseline_score": self._baseline_score,
                "baseline_iteration": self._baseline_iteration,
                "trial_count": len(self._snapshot_records),
            },
        }
        base.update(data)
        (self.run_dir / "metadata.json").write_text(
            json.dumps(base, indent=2, default=str),
            encoding="utf-8",
        )

    def close(
        self, memory: ExecutionMemory | None = None, metadata: dict | None = None
    ) -> None:
        """Stop recording, flush all logs, write final metadata."""
        self.stop_iter_recording()
        self.stop_recording()

        if self._reward_trace:
            self.reward_trace_path().write_text(
                json.dumps(self._reward_trace, indent=2),
                encoding="utf-8",
            )

        if memory is not None:
            self.save_tool_calls(memory)

        self.save_metadata(metadata or {})

        from enpire.env.forge.cap.agent.profiler import close_file_logging

        close_file_logging()

        print(f"[session] Run complete → {self.run_dir}")

    def init_skill_library(self, namespace: dict, seed_dir: "str | None" = None) -> Any:
        """Create and attach a SkillLibrary for this run. Returns it."""
        from enpire.env.forge.cap.agent.skill_library import SkillLibrary

        sl = SkillLibrary(self.run_dir)
        sl.generate_namespace(namespace)
        if seed_dir:
            sl.seed_from_dir(seed_dir)
        self._skill_library = sl
        self.initialize_skill_library_baseline()
        return sl
