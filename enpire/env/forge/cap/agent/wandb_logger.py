"""Weights & Biases logging for agent runs.

Thin wrapper that handles wandb not being installed (logs a warning and
no-ops).  All wandb interaction goes through this module so the rest of the
codebase stays decoupled.

Usage::

    from enpire.env.forge.cap.agent.wandb_logger import WandbLogger

    wb = WandbLogger(project="cap-agent", config={...})
    wb.log_iteration(iteration=0, metrics={...})
    wb.finish(exit_code=0)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wandb  # type: ignore[import-untyped]

    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False


class WandbLogger:
    """Manages a single wandb run for one agent execution."""

    @classmethod
    def from_config(cls, cfg, session) -> WandbLogger:
        """Build from Hydra config + session. Returns no-op logger if disabled."""
        from omegaconf import OmegaConf

        return cls(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.name}_{session.run_dir.name}",
            tags=list(cfg.wandb.tags) + [cfg.env.name or "unknown_env"],
            config=OmegaConf.to_container(cfg, resolve=True),
            log_dir=session.run_dir,
            enabled=cfg.wandb.enabled,
        )

    def __init__(
        self,
        project: str = "cap-agent",
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        log_dir: Path | str | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and _HAS_WANDB
        self._run = None
        self._log_dir = Path(log_dir) if log_dir else None

        if enabled and not _HAS_WANDB:
            logger.warning(
                "wandb not installed — install with: uv sync --extra robocasa  "
                "(or pip install wandb). Logging disabled."
            )
            return

        if not self._enabled:
            return

        # Collect RoboCasa env vars for config
        env_config = _collect_env_vars()
        full_config = {**(config or {}), **env_config}

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            config=full_config,
            dir=str(log_dir) if log_dir else None,
            reinit="finish_previous",
        )
        logger.info("wandb run initialized: %s", self._run.url)

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def run_url(self) -> str | None:
        return self._run.url if self._run else None

    def log_namespace_tools(self, namespace: dict) -> None:
        """Store the full list of callable tool names in the run config.

        Stored as ``namespace_tools`` in wandb config so it's queryable and
        available to plotting scripts that want to show zero-call tools.
        Skips non-string keys and internal helpers (leading underscore).
        """
        if not self.active:
            return
        tools = sorted(
            k for k in namespace
            if isinstance(k, str) and not k.startswith("_")
        )
        self._run.config.update({"namespace_tools": tools}, allow_val_change=True)

    def log_iteration(
        self,
        iteration: int,
        step_timings: dict[str, float],
        total_ms: float,
        score: float,
        success: bool,
        code_length: int = 0,
        tool_call_count: int = 0,
        execution_error: str | None = None,
        profiling_entries: list[tuple[int, Path | str]] | None = None,
        n_seeds: int = 1,
        successes: int | None = None,
        success_rate: float | None = None,
        avg_score: float | None = None,
        token_input: int = 0,
        token_output: int = 0,
    ) -> None:
        """Log metrics for one pipeline iteration.

        This is the **only** place that calls ``wandb.log`` per iteration,
        ensuring a single monotonic step sequence (step=iteration).

        Args:
            profiling_entries: Optional list of (seed, profiling.json path)
                tuples.  All seeds are merged into a single wandb Table.
            n_seeds, successes, success_rate, avg_score: per-seed aggregate
                stats for parallel runs (``iter/success`` stays binary —
                true only when all seeds succeeded).
        """
        if not self.active:
            return

        if successes is None:
            successes = int(success) * n_seeds
        if success_rate is None:
            success_rate = (successes / n_seeds) if n_seeds else 0.0
        if avg_score is None:
            avg_score = score

        metrics: dict[str, Any] = {
            "iteration": iteration,
            "iter/total_ms": total_ms,
            "iter/score": score,
            "iter/success": int(success),
            "iter/n_seeds": n_seeds,
            "iter/successes": successes,
            "iter/success_rate": success_rate,
            "iter/avg_score": avg_score,
            "iter/code_length": code_length,
            "iter/tool_calls": tool_call_count,
            "iter/has_error": int(execution_error is not None),
            "iter/token_input": token_input,
            "iter/token_output": token_output,
            "iter/token_total": token_input + token_output,
        }

        # Per-step timings: step/observer_ms, step/code_generator_ms, etc.
        for step_name, ms in step_timings.items():
            metrics[f"step/{step_name}_ms"] = ms

        # Merge tool profiling from all seeds into one Table
        table = self._build_profiling_table(iteration, profiling_entries or [])
        if table is not None:
            metrics["tool_profiling"] = table

        # Pull per-step policy timing from any use_policy_output calls across
        # seeds, average across seeds, emit as scalars. Missing silently →
        # wandb just skips these keys for runs without a policy rollout.
        policy_stats = self._aggregate_policy_profiling(profiling_entries or [])
        metrics.update(policy_stats)

        wandb.log(metrics, step=iteration)

    def log_summary(
        self,
        success: bool,
        iterations: int,
        duration_s: float,
        final_score: float,
        stop_reason: str,
    ) -> None:
        """Log final run-level summary metrics."""
        if not self.active:
            return

        wandb.summary["success"] = int(success)
        wandb.summary["iterations"] = iterations
        wandb.summary["duration_s"] = round(duration_s, 2)
        wandb.summary["final_score"] = final_score
        wandb.summary["stop_reason"] = stop_reason

    @staticmethod
    def _build_profiling_table(
        iteration: int,
        profiling_entries: list[tuple[int, Path | str]],
    ) -> wandb.Table | None:
        """Parse profiling.json files and merge into a single wandb Table.

        Args:
            profiling_entries: List of (seed, profiling_json_path) tuples.

        Returns ``None`` when no data is found so the caller can skip the key.
        """
        import json

        columns = [
            "iteration",
            "seed",
            "tool",
            "duration_ms",
            "idle_gap_ms",
            "has_error",
        ]
        rows: list[list] = []

        for seed, p in profiling_entries:
            path = Path(p)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for call in data.get("calls", []):
                rows.append(
                    [
                        iteration,
                        seed,
                        call.get("tool", "unknown"),
                        call.get("duration_ms", 0),
                        call.get("idle_gap_ms", 0),
                        1 if call.get("error") else 0,
                    ]
                )

        if not rows:
            return None

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*row)
        return table

    @staticmethod
    def _aggregate_policy_profiling(
        profiling_entries: list[tuple[int, Path | str]],
    ) -> dict[str, float]:
        """Collect ``use_policy_output`` per-step timing across seeds.

        Looks for ``result_data.profiling`` on each ``use_policy_output`` call
        (structure: ``{get_action_ms, env_step_ms, predict_ms, chunk_hit_ms}``
        each with ``{count, mean, p50, p95, max}``). Averages across seeds;
        emits as ``policy/<bucket>_<stat>`` scalar metrics.
        """
        import json

        buckets = ["get_action_ms", "env_step_ms", "predict_ms", "chunk_hit_ms"]
        stats = ["mean", "p50", "p95", "max"]
        # Per-(bucket, stat) lists of per-seed values to average.
        collected: dict[tuple[str, str], list[float]] = {
            (b, s): [] for b in buckets for s in stats
        }
        predict_counts: list[float] = []  # chunk refills per episode

        for _seed, p in profiling_entries:
            path = Path(p)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for call in data.get("calls", []):
                if call.get("tool") != "use_policy_output":
                    continue
                result_data = call.get("result_data")
                if not isinstance(result_data, dict):
                    continue
                prof = result_data.get("profiling")
                if not isinstance(prof, dict):
                    continue
                for bucket in buckets:
                    bdata = prof.get(bucket) or {}
                    for stat in stats:
                        val = bdata.get(stat)
                        if isinstance(val, (int, float)):
                            collected[(bucket, stat)].append(float(val))
                pm = prof.get("predict_ms") or {}
                cnt = pm.get("count")
                if isinstance(cnt, (int, float)):
                    predict_counts.append(float(cnt))

        metrics: dict[str, float] = {}
        for (bucket, stat), values in collected.items():
            if not values:
                continue
            metrics[f"policy/{bucket}_{stat}"] = sum(values) / len(values)
        if predict_counts:
            metrics["policy/chunk_refills_mean"] = sum(predict_counts) / len(
                predict_counts
            )
        return metrics

    def log_artifact(self, path: Path | str, name: str, type: str = "log") -> None:
        """Upload a file or directory as a wandb artifact."""
        if not self.active:
            return

        path = Path(path)
        if not path.exists():
            return

        artifact = wandb.Artifact(name=name, type=type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def log_tool_line_plots(self) -> None:
        """Load per-tool JSONs from ``tool_log/`` and log one line plot per tool.

        Produces:
        - ``tool_plot/{tool}`` — duration per call (x=call_index)
        - ``tool_table/{tool}`` — raw call table
        - ``iter_tool_usage/{tool}`` — call count per iteration (x=iteration)
        - ``tool_summary`` — aggregate stats table
        """
        if not self.active or self._log_dir is None:
            return

        from enpire.env.forge.cap.agent.tool_log import read_tool_logs

        tool_calls = read_tool_logs(self._log_dir)
        if not tool_calls:
            return

        summary_cols = [
            "tool",
            "count",
            "errors",
            "success_rate",
            "mean_duration_ms",
            "total_duration_ms",
        ]
        summary_table = wandb.Table(columns=summary_cols)
        payload: dict[str, Any] = {}

        for tool, calls in tool_calls.items():
            if not calls:
                continue
            calls_sorted = sorted(
                calls,
                key=lambda c: (
                    c.get("iteration", 0),
                    c.get("exec_id", 0),
                    c.get("call_id", 0),
                ),
            )
            table = wandb.Table(
                columns=[
                    "call_index",
                    "iteration",
                    "exec_id",
                    "duration_ms",
                    "idle_gap_ms",
                    "success",
                ]
            )
            total_ms = 0.0
            errors = 0
            counts_by_iter: dict[int, int] = {}
            for idx, c in enumerate(calls_sorted):
                dur = float(c.get("duration_ms", 0.0))
                total_ms += dur
                is_err = bool(c.get("error"))
                if is_err:
                    errors += 1
                it = int(c.get("iteration", 0))
                counts_by_iter[it] = counts_by_iter.get(it, 0) + 1
                table.add_data(
                    idx,
                    it,
                    int(c.get("exec_id", 0)),
                    dur,
                    float(c.get("idle_gap_ms", 0.0)),
                    0 if is_err else 1,
                )
            count = len(calls_sorted)
            summary_table.add_data(
                tool,
                count,
                errors,
                round((count - errors) / count, 3) if count else 0.0,
                round(total_ms / count, 1) if count else 0.0,
                round(total_ms, 1),
            )
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tool)
            payload[f"tool_plot/{safe}"] = wandb.plot.line(
                table,
                "call_index",
                "duration_ms",
                title=f"{tool} — duration per call",
            )
            payload[f"tool_table/{safe}"] = table

            # iter_tool_usage: call count per iteration
            usage_table = wandb.Table(columns=["iteration", "call_count"])
            for it in sorted(counts_by_iter):
                usage_table.add_data(it, counts_by_iter[it])
            payload[f"iter_tool_usage/{safe}"] = wandb.plot.line(
                usage_table,
                "iteration",
                "call_count",
                title=f"{tool} — calls per iteration",
            )

        payload["tool_summary"] = summary_table
        wandb.log(payload)

    def finish(self, exit_code: int = 0) -> None:
        """Finalize the wandb run."""
        if not self.active:
            return

        try:
            self.log_tool_line_plots()
        except Exception:
            logger.warning("Failed to log per-tool line plots", exc_info=True)

        wandb.finish(exit_code=exit_code)
        self._run = None


def _collect_env_vars() -> dict[str, Any]:
    """Collect relevant environment variables for wandb config."""
    env_keys = [
        "ROBOCASA_LAYOUT_ID",
        "ROBOCASA_STYLE_ID",
        "ROBOCASA_SEED",
        "PYTHONHASHSEED",
        "CAP_AGENT_NAME",
        "CAP_CUROBO_PORT",
    ]
    result: dict[str, Any] = {}
    for key in env_keys:
        val = os.environ.get(key)
        if val is not None:
            # Try to parse as int
            try:
                result[key] = int(val)
            except ValueError:
                result[key] = val
    return result
