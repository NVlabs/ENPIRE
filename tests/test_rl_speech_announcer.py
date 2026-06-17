from rl.speech_announcer import SpeechAnnouncer


def test_speech_announcer_disabled_outside_learn_mode(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr("rl.speech_announcer.shutil.which", lambda _: "/usr/bin/spd-say")
    monkeypatch.setattr(
        "rl.speech_announcer.subprocess.Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    announcer = SpeechAnnouncer.for_mode("eval")
    announcer.speak("success")

    assert calls == []
    assert "disabled outside learn mode" in capsys.readouterr().out


def test_speech_announcer_noops_without_tts_command(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr("rl.speech_announcer.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "rl.speech_announcer.subprocess.Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    announcer = SpeechAnnouncer.for_mode("learn")
    announcer.speak("success")

    assert calls == []
    assert "no TTS command found" in capsys.readouterr().out


def test_speech_announcer_uses_first_available_command(monkeypatch, capsys):
    calls = []

    def which(command):
        if command == "say":
            return "/usr/bin/say"
        return None

    def popen(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("rl.speech_announcer.shutil.which", which)
    monkeypatch.setattr("rl.speech_announcer.subprocess.Popen", popen)

    announcer = SpeechAnnouncer.for_mode("learn")
    announcer.speak("out of range")

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == ["/usr/bin/say", "out of range"]
    assert "shell" not in kwargs
    assert kwargs["start_new_session"] is True
    assert "Speech announcer using: /usr/bin/say" in capsys.readouterr().out


def test_speech_announcer_logs_spawn_failure(monkeypatch, capsys):
    monkeypatch.setattr("rl.speech_announcer.shutil.which", lambda _: "/usr/bin/spd-say")

    def popen(*args, **kwargs):
        raise OSError("audio unavailable")

    monkeypatch.setattr("rl.speech_announcer.subprocess.Popen", popen)

    announcer = SpeechAnnouncer.for_mode("learn")
    capsys.readouterr()
    announcer.speak("success")

    output = capsys.readouterr().out
    assert "failed to speak 'success'" in output
    assert "audio unavailable" in output
