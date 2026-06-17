"""Abstract base class for LLM backends used by the CAP agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Interface that every LLM backend must implement."""

    @abstractmethod
    def generate_code(self, task: str, context: dict[str, Any]) -> str:
        """Generate Python code for the given task.

        Args:
            task: Natural-language task description from the user.
            context: Dictionary containing:
                - ``tools``: List of tool schemas (from ToolRegistry.schemas()).
                - ``robot_state``: Current RobotState snapshot (if available).
                - ``history``: List of previous action log entries.

        Returns:
            A string of executable Python code that uses the tool functions.
        """

    def generate_text(self, prompt: str) -> str:
        """Generate free-form text (analysis, reflection, review) from a prompt.

        Unlike generate_code(), this method does NOT append code-output
        instructions.  Override in subclasses for a cleaner implementation;
        the default falls back to generate_code() with a plain context.
        """
        return self.generate_code(prompt, {})
