from __future__ import annotations

from dataclasses import dataclass

from enpire.policy.rl.events import TERMINAL_EVENTS


@dataclass
class RLStateMachine:
    """Small state machine for core online RL collection."""

    state: str = "idle"

    def transition(self, event: str | None) -> None:
        prev = self.state
        if event == "home":
            self.state = "home"
        elif event == "parking":
            self.state = "parking"
        elif event == "author":
            self.state = "author"
        elif self.state == "author" and event == "discard_author":
            self.state = "home"
        elif event in ("next_pose", "prev_pose", "set_initial_position_index"):
            self.state = "parking" if self.state == "parking" else "change_pose"
        elif self.state == "parking" and event in (
            "init_boundary",
            "oor_boundary",
            "z_high",
            "z_low",
        ):
            self.state = "parking"
        elif self.state == "parking" and event == "start":
            self.state = "hover"
        elif self.state in ("idle", "home") and event == "start":
            self.state = "hover"
        elif self.state == "hover":
            self.state = "learn"
        elif self.state == "learn" and event in TERMINAL_EVENTS:
            self.state = "hover"

        if self.state != prev:
            print(f"[RL State Machine] {prev} -> {self.state}", flush=True)

