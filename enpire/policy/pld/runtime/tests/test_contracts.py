from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pytest
from omegaconf import OmegaConf

from enpire_pld import disk_buffer_ingestor as ingestor
from enpire_pld import sr_rolling_window, train


def test_policy_action_dimension_and_wire_expansion() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"control_mode": "left", "action_repr": "delta_eef_rot6d"},
            "task": {"action_repr": "delta_eef_pos", "policy_action_dim": 3},
        }
    )

    assert train._configured_policy_action_dim(cfg) == 3
    expanded = train._expand_policy_action_to_env_action([0.1, -0.2, 0.3], cfg)

    np.testing.assert_allclose(expanded[:3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(expanded[3:9], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(expanded[9], 0.0)


def test_actor_action_transform_keeps_policy_and_wire_units_separate() -> None:
    cfg = OmegaConf.create({"action_transform": {"gamma": 0.5}})
    scale = lambda action: np.asarray(action) * np.array([1.0, 2.0, 3.0])

    policy, executed, sent = train._actor_action_transform([2.0, -0.5, 0.25], cfg, 3, scale)

    np.testing.assert_allclose(policy, [1.0, -0.5, 0.25])
    np.testing.assert_allclose(executed, [0.5, -0.25, 0.125])
    np.testing.assert_allclose(sent, [0.5, -0.5, 0.375])


def test_disk_ingestor_routes_source_labels_without_filesystem_mutation() -> None:
    assert ingestor._normalise_source("rl") == "rl"
    assert ingestor._normalise_source(b"manual") == "human"
    assert ingestor._normalise_source(np.asarray("policy")) == "policy"
    assert ingestor._normalise_source("unexpected") == "unknown"


def test_actor_and_learner_hydra_configs_compose() -> None:
    config_dir = Path(train.__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        actor = hydra.compose(
            config_name="config",
            overrides=[
                "system=gear-yam-24",
                "task=remote_yam_left_gpu_pos3",
                "+experiment=gpu_insertion_delta_eef",
                "train.actor=true",
            ],
        )
    assert actor.train.actor is True
    assert actor.env.control_mode == "left"
    assert actor.task.action_repr == "delta_eef_pos"

    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        learner = hydra.compose(
            config_name="config",
            overrides=[
                "system=gear-yam-24",
                "task=remote_yam_right_pos3",
                "+experiment=pin_insertion_delta_eef",
                "train.learner=true",
            ],
        )
    assert learner.train.learner is True
    assert learner.env.control_mode == "right"


def test_rolling_score_preserves_pure_rl_filter(tmp_path) -> None:
    for index, (terminal, source) in enumerate(
        (("success", "rl"), ("fail", "rl"), ("success", "human")), start=1
    ):
        episode = tmp_path / f"20260101T00000{index}000000"
        episode.mkdir()
        (episode / "metadata.json").write_text(
            f'{{"terminal_event": "{terminal}"}}', encoding="utf-8"
        )
        (episode / "action-source.json").write_text(f'["{source}"]', encoding="utf-8")

    rows = sr_rolling_window.compute_rows(tmp_path, window=2, success_event="success")

    assert len(rows) == 3
    assert rows[-1]["cumulative_success_rate"] == pytest.approx(2 / 3)
    pure_rl = [row for row in rows if row["is_pure_rl"]]
    assert pure_rl[-1]["pure_rl_rolling_success_rate"] == pytest.approx(0.5)
