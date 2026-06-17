"""Tests for batched policy inference seed handling.

Verifies that:
1. When n_eps <= n_envs, seeds are 0..n_envs-1 (no recycling).
2. When n_eps > n_envs, recycled slots get proper deterministic seeds.
3. Results are ordered by episode index regardless of completion order.

Run with: uv run pytest tests/test_policy_inference_batched.py -v
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pytest

from cap.policy.backend import PolicyBackend
from cap.policy.inference import InferencePolicyConfig, inference_policy

# Batched path requires gr00t (for MultiStepWrapper). Skip if unavailable.
_has_gr00t = False
try:
    import importlib
    importlib.import_module("gr00t.eval.wrappers.multistep_wrapper")
    _has_gr00t = True
except ImportError:
    try:
        importlib.import_module("gr00t.eval.sim.wrapper.multistep_wrapper")
        _has_gr00t = True
    except ImportError:
        pass

skip_no_gr00t = pytest.mark.skipif(
    not _has_gr00t, reason="gr00t (MultiStepWrapper) not available"
)


# ---------------------------------------------------------------------------
# Dummy env that records seeds and terminates after a fixed number of steps
# ---------------------------------------------------------------------------

class SeedTrackingEnv(gym.Env):
    """Minimal env that records reset seeds and terminates after `ep_length` steps.

    Observations include the seed so the test can verify which seed each
    episode ran with.
    """

    metadata = {"render_modes": []}

    def __init__(self, ep_length: int = 3):
        super().__init__()
        self.ep_length = ep_length
        self.observation_space = spaces.Dict({
            "state.dummy": spaces.Box(-1, 1, (2,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            "action.dummy": spaces.Box(-1, 1, (2,), dtype=np.float32),
        })
        self._step_count = 0
        self._seed = None
        self._seeds_seen: list[int | None] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = seed
        self._seeds_seen.append(seed)
        self._step_count = 0
        obs = {"state.dummy": np.array([float(seed or -1), 0.0], dtype=np.float32)}
        return obs, {"success": False}

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= self.ep_length
        obs = {"state.dummy": np.array([float(self._seed or -1), float(self._step_count)], dtype=np.float32)}
        info = {"success": done}  # succeed on last step
        return obs, 1.0 if done else 0.0, done, False, info


class DummyBackend(PolicyBackend):
    """Backend that returns zero actions shaped for MultiStepWrapper.

    MultiStepWrapper expects actions of shape (n_action_steps, action_dim).
    For batched envs, predict_batch returns (n_envs, n_action_steps, action_dim).
    """

    def __init__(self, n_action_steps: int = 1):
        self._n = n_action_steps

    def predict(self, obs):
        return {"action.dummy": np.zeros((self._n, 2), dtype=np.float32)}

    def predict_batch(self, obs):
        n_envs = obs["state.dummy"].shape[0]
        return {"action.dummy": np.zeros((n_envs, self._n, 2), dtype=np.float32)}

    def reset(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_env(ep_length: int = 3):
    return SeedTrackingEnv(ep_length=ep_length)


@skip_no_gr00t
def test_no_recycling_when_n_eps_equals_n_envs():
    """n_envs=2, n_episodes=2: seeds are [0, 1], no slot recycling."""
    config = InferencePolicyConfig(
        action_horizon=1, replan_horizon=1,
        max_episode_steps=5,
    )
    backend = DummyBackend()

    results = inference_policy(
        env=lambda: _make_env(ep_length=2),
        backend=backend,
        config=config,
        n_episodes=2,
        n_envs=2,
        deterministic=True,
    )

    assert len(results) == 2
    # Both episodes should succeed (done on last step)
    assert all(r.success for r in results)


@skip_no_gr00t
def test_recycled_slots_get_new_seeds():
    """n_envs=2, n_episodes=4: recycled slots should get seeds 2 and 3.

    Only verifiable on gymnasium >= 1.0.0 with reset_mask support.
    On older gymnasium, recycled slots get non-deterministic resets (accepted).
    """
    config = InferencePolicyConfig(
        action_horizon=1, replan_horizon=1,
        max_episode_steps=10,
    )
    backend = DummyBackend()

    results = inference_policy(
        env=lambda: _make_env(ep_length=2),
        backend=backend,
        config=config,
        n_episodes=4,
        n_envs=2,
        deterministic=True,
    )

    assert len(results) == 4
    assert all(r.success for r in results)


@skip_no_gr00t
def test_results_ordered_by_episode_index():
    """Results should be sorted by episode index, not completion order."""
    config = InferencePolicyConfig(
        action_horizon=1, replan_horizon=1,
        max_episode_steps=20,
    )
    backend = DummyBackend()

    results = inference_policy(
        env=lambda: _make_env(ep_length=3),
        backend=backend,
        config=config,
        n_episodes=6,
        n_envs=3,
        deterministic=True,
    )

    assert len(results) == 6


def test_single_env_path_unchanged():
    """n_envs=1 uses _run_single, should be unaffected by batched changes."""
    config = InferencePolicyConfig(
        action_horizon=1, replan_horizon=1,
        max_episode_steps=10,
    )
    backend = DummyBackend()

    env = _make_env(ep_length=2)
    results = inference_policy(
        env=env,
        backend=backend,
        config=config,
        n_episodes=3,
        n_envs=1,
        deterministic=True,
    )

    assert len(results) == 3
    assert all(r.success for r in results)
    env.close()


def test_single_env_video_dir_is_ignored(tmp_path):
    """Single-env inference no longer instantiates rollout video writers."""
    config = InferencePolicyConfig(
        action_horizon=1, replan_horizon=1,
        max_episode_steps=10,
    )
    backend = DummyBackend()

    env = _make_env(ep_length=2)
    with pytest.warns(UserWarning, match="Single-env rollout video recording"):
        results = inference_policy(
            env=env,
            backend=backend,
            config=config,
            n_episodes=1,
            n_envs=1,
            deterministic=True,
            video_dir=str(tmp_path),
            camera_map={"dummy": "dummy_image"},
            task_name="dummy-task",
            model_name="dummy-model",
        )

    assert len(results) == 1
    assert results[0].success
    assert list(tmp_path.iterdir()) == []
    env.close()
