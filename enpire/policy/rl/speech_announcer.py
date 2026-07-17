from __future__ import annotations

import shutil
import subprocess


TERMINAL_EVENT_PHRASES = {
    "success": "success",
    "out-of-range": "out of range",
    "fail": "fail",
    "timeout": "timeout",
}

_COMMAND_CANDIDATES = ("spd-say", "say", "espeak-ng", "espeak")


class SpeechAnnouncer:
    """Best-effort speech announcements for operator-facing RL status."""

    def __init__(self, *, enabled: bool, command: str | None):
        self.enabled = enabled
        self.command = command

    @classmethod
    def for_mode(cls, mode: str) -> "SpeechAnnouncer":
        enabled = mode == "learn"
        command = _find_tts_command() if enabled else None
        if not enabled:
            print("[INFO] Speech announcer disabled outside learn mode", flush=True)
        elif command is None:
            print(
                "[WARN] Speech announcer enabled, but no TTS command found "
                f"from {_COMMAND_CANDIDATES}",
                flush=True,
            )
        else:
            print(f"[INFO] Speech announcer using: {command}", flush=True)
        return cls(enabled=enabled, command=command)

    def speak(self, phrase: str) -> None:
        if not self.enabled or self.command is None:
            return
        try:
            subprocess.Popen(
                [self.command, phrase],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            print(
                f"[WARN] Speech announcer failed to speak {phrase!r}: {exc}",
                flush=True,
            )
            return


def _find_tts_command() -> str | None:
    for candidate in _COMMAND_CANDIDATES:
        command = shutil.which(candidate)
        if command is not None:
            return command
    return None

