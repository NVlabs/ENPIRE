"""Backward-compatible entrypoint for the generic CAP agent bridge."""

from cap.bridge.agent_bridge import main
from cap.bridge.agent_bridge import create_app

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
