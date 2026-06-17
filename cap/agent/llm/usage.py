"""LLM token-usage tracking: per-call records, JSONL sink, and in-memory accumulation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cap.agent.llm.base import LLMBackend


@dataclass
class LLMUsageRecord:
    timestamp: str   # ISO-8601 UTC
    model: str
    step: str
    iteration: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def record_llm_usage(ctx: Any, llm: "LLMBackend", step_name: str) -> LLMUsageRecord | None:
    """Capture last_usage from an LLM backend, append to ctx and write a JSONL line.

    Safe to call even if the backend has no usage data — returns None in that case.
    """
    usage = getattr(llm, "last_usage", None)
    if not usage:
        return None

    rec = LLMUsageRecord(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model=str(usage.get("model", "unknown")),
        step=step_name,
        iteration=int(getattr(ctx, "iteration", -1)),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )

    llm_usage_list = getattr(ctx, "llm_usage", None)
    if isinstance(llm_usage_list, list):
        llm_usage_list.append(rec)

    session = getattr(ctx, "session", None)
    if session is not None:
        run_dir: Path = session.run_dir
        try:
            with open(run_dir / "llm_usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
        except OSError:
            pass

    return rec
