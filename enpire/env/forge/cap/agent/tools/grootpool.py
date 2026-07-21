"""Agent-facing grootpool tool.

Exposes a context-manager-based session API to agent-generated code.  CAP
scripts call this to get GR00T N1.5 actions routed through the grootpool
middleware (see ``docs/GROOTPOOL.md``).

Usage from generated code:

    from enpire.env.forge.cap.agent.tools import grootpool

    with grootpool.session(task_description="pick up the cup") as s:
        for _ in range(T):
            obs = collect_obs()
            action_chunk = s.step(obs)
            execute_chunk(action_chunk)
"""

from __future__ import annotations

from contextlib import contextmanager

from enpire.env.forge.cap.policy.grootpool.client import GrootPoolClient, GrootPoolError

_client: GrootPoolClient | None = None


def _get_client() -> GrootPoolClient:
    global _client
    if _client is None:
        _client = GrootPoolClient()
    return _client


@contextmanager
def session(model: str = "n15", task_description: str = "", **kwargs):
    """Open a sticky GR00T policy session for the duration of this block."""
    client = _get_client()
    with client.session(model=model, task_description=task_description, **kwargs) as s:
        yield s


__all__ = ["session", "GrootPoolError"]
