from __future__ import annotations

import pytest

from enpire.cli import main as cli_main
from enpire.policy.autoresearch import build_control_request
from enpire.policy.pld.launcher import build_pld_launch, build_score_launch


def test_actor_launch_preserves_gpu_task_mapping() -> None:
    launch = build_pld_launch("actor", "gpu_insertion")

    assert "task=remote_yam_left_gpu_pos3" in launch.command
    assert "+experiment=gpu_insertion_delta_eef" in launch.command
    assert "train.actor=true" in launch.command
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert launch.env["WANDB_MODE"] == "offline"


def test_learner_launch_preserves_pin_mapping_and_override(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    launch = build_pld_launch(
        "learner",
        "pin_insertion",
        device=3,
        overrides=("train.resume=true",),
    )

    assert "task=remote_yam_right_pos3" in launch.command
    assert "+experiment=pin_insertion_delta_eef" in launch.command
    assert launch.command[-1] == "train.resume=true"
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "3"
    assert launch.env["WANDB_MODE"] == "disabled"


def test_unknown_pld_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown PLD task"):
        build_pld_launch("actor", "unknown")


def test_rl_cli_dry_run_does_not_start_training(capsys) -> None:
    result = cli_main(["rl", "actor", "--task", "ziptie", "--dry-run"])

    assert result == 0
    output = capsys.readouterr().out
    assert "task=remote_yam_right_pos3" in output
    assert "+experiment=ziptie_delta_eef" in output


def test_score_launch_uses_isolated_runtime_and_external_data(tmp_path) -> None:
    launch = build_score_launch(tmp_path / "episodes", window=50)

    assert "enpire_pld.sr_rolling_window" in launch.command
    assert str((tmp_path / "episodes").resolve()) in launch.command
    assert launch.command[-1] == "--no-plot"


def test_control_dry_run_is_explicit_and_mutations_require_confirmation(capsys) -> None:
    request = build_control_request("pause", "http://localhost:8203/")
    assert request.method == "POST"
    assert request.url == "http://localhost:8203/pause"

    assert cli_main(["rl", "control", "resume", "--dry-run"]) == 0
    assert "POST http://127.0.0.1:8203/resume" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--confirm-control"):
        cli_main(["rl", "control", "restart"])
