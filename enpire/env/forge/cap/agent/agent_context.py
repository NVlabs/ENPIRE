"""Shared agent state and execution memory.

AgentContext flows through every step in an AgentPipeline, accumulating
results across iterations. ExecutionMemory records every tool call for
LLM context and persistent logging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.agent_session import AgentRunSession
    from enpire.env.forge.cap.prompt.loader import PromptMemory


# ---------------------------------------------------------------------------
# Tool call memory
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """Record of a single tool call captured via profiler hooks."""

    call_id: int
    tool_name: str
    args_repr: str = ""  # formatted args (truncated)
    result_repr: str = ""  # formatted result (truncated)
    error: str | None = None
    elapsed_ms: float = 0.0
    timestamp: str = ""
    iteration: int = 0


class ExecutionMemory:
    """Accumulates tool call records across all iterations.

    Used to build LLM context summaries and persist to tool_calls.json.
    """

    def __init__(self) -> None:
        self._calls: list[ToolCallRecord] = []
        self._by_iteration: dict[int, list[ToolCallRecord]] = {}

    def record(self, rec: ToolCallRecord) -> None:
        self._calls.append(rec)
        self._by_iteration.setdefault(rec.iteration, []).append(rec)

    def get_iteration(self, iteration: int) -> list[ToolCallRecord]:
        return list(self._by_iteration.get(iteration, []))

    def get_by_tool(self, tool_name: str) -> list[ToolCallRecord]:
        return [c for c in self._calls if c.tool_name == tool_name]

    def get_errors(self) -> list[ToolCallRecord]:
        return [c for c in self._calls if c.error is not None]

    def get_recent(self, n: int = 20) -> list[ToolCallRecord]:
        return self._calls[-n:]

    def summarize_for_llm(self, max_chars: int = 4000) -> str:
        """Compact text summary for LLM context, grouped by iteration."""
        if not self._calls:
            return "(no tool calls yet)"

        sections: list[str] = []
        for it, calls in sorted(self._by_iteration.items()):
            errors = [c for c in calls if c.error]
            status = f"FAILED ({errors[0].error})" if errors else "OK"
            header = f"## Iteration {it} ({status})"
            lines = [header]
            for c in calls:
                err = f"  ERROR: {c.error}" if c.error else ""
                lines.append(
                    f"  #{c.call_id} {c.tool_name}({c.args_repr}) "
                    f"→ {c.elapsed_ms:.0f}ms  {c.result_repr}{err}"
                )
            sections.append("\n".join(lines))

        summary = "\n\n".join(sections)
        if max_chars > 0 and len(summary) > max_chars:
            # Keep most recent content
            summary = "...(truncated)...\n\n" + summary[-max_chars:]
        return summary

    def to_json(self) -> list[dict[str, Any]]:
        return [
            {
                "call_id": c.call_id,
                "tool_name": c.tool_name,
                "args": c.args_repr,
                "result": c.result_repr,
                "error": c.error,
                "elapsed_ms": c.elapsed_ms,
                "timestamp": c.timestamp,
                "iteration": c.iteration,
            }
            for c in self._calls
        ]


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass
class AgentEvaluation:
    """Result of evaluating a code execution (reward, VLM, or self-reflection)."""

    success: bool
    score: float = 0.0  # 0.0–1.0
    feedback: str = ""  # human-readable, fed to next iteration
    method: str = "none"  # "reward" | "vlm" | "self_reflection" | "human"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "feedback": self.feedback,
            "method": self.method,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Iteration record
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """Snapshot of one complete pipeline iteration."""

    iteration: int
    code: str
    thoughts: str = ""  # LLM's reasoning/strategy before the code
    execution_result: Any = None  # ExecutionResult | None
    evaluation: AgentEvaluation | None = None
    robot_state_before: dict | None = None
    robot_state_after: dict | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


# ---------------------------------------------------------------------------
# Central context object
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Shared mutable state that flows through every AgentStep in a pipeline.

    Steps read from the context and write their outputs back to it.
    The pipeline iterates until ``should_stop`` is True or ``max_iterations``
    is reached.

    Example::

        ctx = AgentContext(
            task="Pick up the red object",
            namespace=tool_namespace,
            tool_schemas=registry.schemas(),
            session=session,
            max_iterations=5,
        )
        pipeline.run(ctx)
    """

    task: str
    max_iterations: int = 5
    iteration: int = 0

    # --- per-iteration pipeline state (reset each iteration by pipeline) ---
    robot_state: Any = None  # RobotState from ObserverStep
    task_info: dict | None = None  # from get_task_info()
    thoughts: str | None = (
        None  # LLM's reasoning before code (strategy, expected behavior)
    )
    code: str | None = None  # generated by CodeGeneratorStep
    code_candidates: list[dict[str, Any]] = field(default_factory=list)
    review_feedback: str | None = None  # from CodeReviewerStep
    execution_result: Any = None  # ExecutionResult from ExecutorStep
    evaluation: AgentEvaluation | None = None  # from EvaluatorStep
    frames_before: dict[str, Any] | None = (
        None  # camera_name → RGB array (pre-execution)
    )
    frames_after: dict[str, Any] | None = (
        None  # camera_name → RGB array (post-execution)
    )

    # --- accumulated across iterations ---
    history: list[IterationRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # conversation log
    memory: ExecutionMemory = field(default_factory=ExecutionMemory)
    llm_usage: list = field(default_factory=list)  # list[LLMUsageRecord], all calls this run

    # --- control flow ---
    should_stop: bool = False
    stop_reason: str = ""

    # --- injected dependencies (set by run_agent.py before pipeline.run) ---
    namespace: dict[str, Any] = field(default_factory=dict)
    tool_schemas: list[dict] = field(default_factory=list)
    env_spec: dict[str, str] | None = None  # from cap.prompt.env_spec.load_env_spec()
    config: Any = None  # AgentConfig from YAML
    session: AgentRunSession | None = field(default=None, repr=False)
    prompt_memory: PromptMemory | None = None  # index-based prompt manager

    # --- paused execution support (wait_for_agent) ---
    paused_namespace: dict[str, Any] | None = None

    # --- per-iteration history paragraph (written after score is known) ---
    history_paragraph: str | None = None

    def snapshot_iteration(self) -> IterationRecord:
        """Archive current iteration state into history."""
        robot_after: dict | None = None
        get_state = self.namespace.get("get_robot_state")
        if get_state is not None:
            try:
                st = get_state()
                robot_after = _serialize_robot_state(st)
            except Exception:
                pass

        rec = IterationRecord(
            iteration=self.iteration,
            code=self.code or "",
            thoughts=self.thoughts or "",
            execution_result=self.execution_result,
            evaluation=self.evaluation,
            robot_state_after=robot_after,
        )
        self.history.append(rec)
        return rec

    def reset_iteration_state(self) -> None:
        """Clear per-iteration fields before starting a new iteration."""
        self.thoughts = None
        self.code = None
        self.code_candidates = []
        self.review_feedback = None
        self.execution_result = None
        self.evaluation = None
        self.robot_state = None
        self.task_info = None
        self.frames_before = None
        self.frames_after = None


def _serialize_robot_state(state: Any) -> dict | None:
    """Convert RobotState to a plain dict for storage."""
    try:
        arms = {}
        for side, arm in state.arms.items():
            arms[side] = {
                "ee_pos": list(arm.ee_pos),
                "ee_rpy": list(arm.ee_rpy),
                "gripper_pos": float(arm.gripper_pos),
                "joint_pos": list(arm.joint_pos),
            }
        return {"arms": arms}
    except Exception:
        return None
