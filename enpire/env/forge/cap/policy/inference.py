"""Top-level policy inference interface.

Usage (single env):
    results = inference_policy(env, backend, config, n_episodes=10)

Usage (batched parallel envs):
    results = inference_policy(env_fn, backend, config, n_episodes=50, n_envs=5)

The same interface covers:
    - Single env or batched parallel envs (n_envs)
    - Sync receding horizon (default)
    - Async background inference (use_async=True)
    - RTC with latency compensation (use_rtc=True, future)
    - Any PolicyBackend (ZMQ/HTTP/local)
    - Sim or real env (anything with reset/step)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Protocol

import numpy as np

from enpire.env.forge.cap.policy.backend import PolicyBackend
from enpire.env.forge.cap.policy.chunking import ChunkingConfig, ChunkingPolicy


def _summarize_ms(samples: list[float]) -> dict[str, float]:
    """Aggregate per-call ms samples into mean / p50 / p95 / max / count."""
    if not samples:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


class StepEnv(Protocol):
    """Minimal env protocol — gym.Env or any reset/step interface."""

    def reset(self, **kwargs: Any) -> tuple[dict, dict]: ...
    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]: ...
    def close(self) -> None: ...


@dataclass
class InferencePolicyConfig:
    """Full parameter set for inference_policy().

    Groups:
        chunking  — action horizon, replan horizon, async, RTC, smoothing
        episode   — max steps, terminate on success
    """

    # ---- chunking (delegated to ChunkingConfig) ----
    action_horizon: int = 16
    replan_horizon: int = 8
    use_async: bool = False
    use_rtc: bool = False
    use_chunk_smoothing: bool = False
    min_smooth_steps: int = 8
    max_get_action_seconds: float = 5.0
    control_hz: float = 20.0
    rtc_bootstrap_delay_steps: int = 4

    # ---- episode ----
    max_episode_steps: int = 720
    terminate_on_success: bool = True

    def to_chunking_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            action_horizon=self.action_horizon,
            replan_horizon=self.replan_horizon,
            use_async=self.use_async,
            use_rtc=self.use_rtc,
            use_chunk_smoothing=self.use_chunk_smoothing,
            min_smooth_steps=self.min_smooth_steps,
            max_get_action_seconds=self.max_get_action_seconds,
            control_hz=self.control_hz,
            rtc_bootstrap_delay_steps=self.rtc_bootstrap_delay_steps,
        )


@dataclass
class EpisodeResult:
    """Result of a single episode."""

    success: bool
    steps: int
    info: dict = field(default_factory=dict)


def set_seed_everywhere(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Single-env path (n_envs=1)
# ---------------------------------------------------------------------------


def run_episode(
    env: StepEnv,
    policy: ChunkingPolicy,
    config: InferencePolicyConfig,
    seed: int | None = None,
    initial_obs: dict | None = None,
) -> EpisodeResult:
    """Run one episode: reset -> step loop -> return result.

    Pass ``initial_obs`` to skip the env.reset() call and start from an
    already-current observation (e.g. when called from a tool like
    ``use_policy_output`` that operates on the live env state).

    Emits per-step timing stats into ``info["profiling"]``:

    - ``get_action_ms``  — time spent in ``policy.get_action`` (dominated by
      chunk refills when the queue is empty; near-zero on chunk hits).
    - ``env_step_ms``    — time spent in ``env.step`` (sim physics + obs pack).
    - ``predict_ms``     — backend.predict subset of ``get_action_ms`` — the
      actual inference + transport cost (ZMQ round-trip for remote backends).
    - ``chunk_hit_ms``   — ``get_action`` calls that hit the queue only.

    Each sub-dict has ``count / mean / p50 / p95 / max`` in ms.
    """
    if initial_obs is not None:
        obs, info = initial_obs, {}
    elif seed is not None:
        set_seed_everywhere(seed)
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    policy.reset()
    # Fresh stats per episode — ChunkingPolicy accumulates across calls.
    if hasattr(policy, "reset_stats"):
        policy.reset_stats()

    # Capture task_description from first obs (before info gets overwritten by step()).
    task_description = ""
    for _td_key in (
        "annotation.human.action.task_description",
        "annotation.human.task_description",
    ):
        if _td_key in obs:
            val = obs[_td_key]
            task_description = val if isinstance(val, str) else str(val)
            break

    get_action_ms: list[float] = []
    env_step_ms: list[float] = []

    success = False
    for step in range(config.max_episode_steps):
        t0 = time.monotonic()
        action = policy.get_action(obs)
        get_action_ms.append((time.monotonic() - t0) * 1000.0)

        # Guard against NaN/inf from learned policies (e.g. GR00T diffusion).
        # MuJoCo calls mju_error() -> abort() (SIGABRT, rc=134) if NaN/inf
        # reaches the physics solver. Replace with 0.0 and log the occurrence
        # so it surfaces in per-seed profiling logs for root-cause analysis.
        sanitized = {}
        for k, v in action.items():
            if (
                isinstance(v, np.ndarray)
                and np.issubdtype(v.dtype, np.floating)
                and not np.all(np.isfinite(v))
            ):
                n_bad = int(np.sum(~np.isfinite(v)))
                print(
                    f"[inference] WARNING: action[{k!r}] has {n_bad} non-finite "
                    f"value(s) at step {step} — replacing with 0.0 "
                    f"(nan={int(np.any(np.isnan(v)))}, "
                    f"inf={int(np.any(np.isinf(v)))})"
                )
                sanitized[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                sanitized[k] = v
        action = sanitized

        t1 = time.monotonic()
        obs, reward, done, truncated, info = env.step(action)
        env_step_ms.append((time.monotonic() - t1) * 1000.0)
        success |= bool(info.get("success", False))
        if done or truncated or (config.terminate_on_success and success):
            break
    info["task_description"] = task_description

    # Pull refill vs chunk-hit split from ChunkingPolicy; fall back gracefully
    # if the policy is not a ChunkingPolicy.
    predict_ms: list[float] = []
    chunk_hit_ms: list[float] = []
    if hasattr(policy, "get_stats"):
        stats = policy.get_stats()
        predict_ms = stats.get("predict_ms", [])
        chunk_hit_ms = stats.get("chunk_hit_ms", [])

    info["profiling"] = {
        "get_action_ms": _summarize_ms(get_action_ms),
        "env_step_ms": _summarize_ms(env_step_ms),
        "predict_ms": _summarize_ms(predict_ms),
        "chunk_hit_ms": _summarize_ms(chunk_hit_ms),
    }
    return EpisodeResult(success=success, steps=step + 1, info=info)


def _run_single(
    env: StepEnv,
    backend: PolicyBackend,
    config: InferencePolicyConfig,
    n_episodes: int,
    deterministic: bool = True,
) -> list[EpisodeResult]:
    from tqdm import tqdm

    policy = ChunkingPolicy(backend, config.to_chunking_config())
    results: list[EpisodeResult] = []
    pbar = tqdm(total=n_episodes, desc="Episodes", leave=True)
    for ep in range(n_episodes):
        r = run_episode(
            env,
            policy,
            config,
            seed=ep if deterministic else None,
        )
        results.append(r)
        rate = sum(r.success for r in results) / len(results) * 100
        pbar.update(1)
        pbar.set_postfix(
            rate=f"{rate:.0f}%", last=f"{'ok' if r.success else 'fail'}/{r.steps}s"
        )
    pbar.close()
    policy.close()
    return results


# ---------------------------------------------------------------------------
# Batched-env path (n_envs > 1)
# Mirrors eval_task_batchasync.py: MultiStepWrapper per env + AsyncVectorEnv
# ---------------------------------------------------------------------------


def _make_batched_env(env_fn, n_action_steps, max_episode_steps):
    """Wrap env_fn output with MultiStepWrapper (same as legacy code)."""
    # Prefer benchmark repo's wrapper — compatible with gymnasium 1.0.0.
    # Old Isaac-GR00T wrapper puts variable-length arrays in info that
    # AsyncVectorEnv can't stack.
    try:
        from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
    except ImportError:
        from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    import inspect

    env = env_fn()
    kwargs = dict(
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
    )
    if (
        "terminate_on_success"
        in inspect.signature(MultiStepWrapper.__init__).parameters
    ):
        kwargs["terminate_on_success"] = True
    return MultiStepWrapper(env, **kwargs)


def _run_batched(
    env_fn: Callable[[], StepEnv],
    backend: PolicyBackend,
    config: InferencePolicyConfig,
    n_episodes: int,
    n_envs: int,
    deterministic: bool = True,
    video_dir: str | None = None,
    camera_map: dict[str, str] | None = None,
    task_name: str = "",
    model_name: str = "",
    model_path: str = "",
) -> list[EpisodeResult]:
    """Batched eval — identical to legacy eval_task_batchasync.py.

    Each env is wrapped with MultiStepWrapper (executes replan_horizon
    inner steps per outer step). AsyncVectorEnv parallelizes outer steps.
    predict_batch is called once per outer step for all envs.
    """
    import gymnasium as gym
    from tqdm import tqdm

    env_fns = [
        partial(
            _make_batched_env,
            env_fn=env_fn,
            n_action_steps=config.replan_horizon,
            max_episode_steps=config.max_episode_steps,
        )
        for _ in range(n_envs)
    ]

    if n_envs == 1:
        vec_env = gym.vector.SyncVectorEnv(env_fns)
    else:
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    n_episodes = max(n_episodes, n_envs)

    if deterministic:
        set_seed_everywhere(0)
        obs, _ = vec_env.reset(seed=list(range(n_envs)))
    else:
        obs, _ = vec_env.reset()
    backend.reset()

    # Capture per-env task_description from first obs after reset.
    _task_descriptions: list[str] = [""] * n_envs
    for _td_key in (
        "annotation.human.action.task_description",
        "annotation.human.task_description",
    ):
        if _td_key in obs:
            for _i in range(n_envs):
                val = obs[_td_key][_i]
                _task_descriptions[_i] = val if isinstance(val, str) else str(val)
            break

    # Video tracking (optional, works with any n_envs)
    vt = None
    if video_dir and camera_map:
        from enpire.env.forge.cap.policy.video import BatchedVideoTracker

        meta = {
            "action_horizon": config.action_horizon,
            "replan_horizon": config.replan_horizon,
            "use_async": config.use_async,
            "use_rtc": config.use_rtc,
            "use_chunk_smoothing": config.use_chunk_smoothing,
            "max_episode_steps": config.max_episode_steps,
            "n_envs": n_envs,
            "model_name": model_name,
            "model_path": model_path,
            "task_description": _task_descriptions[0],
        }
        vt = BatchedVideoTracker(
            video_dir,
            n_envs,
            camera_map,
            task_name=task_name,
            model_name=model_name,
            config_extra=meta,
        )
        vt.start_all()
        vt.record(obs)

    completed = 0
    # indexed_results stores (ep_idx, result) so we can return results sorted
    # by ep_idx (== video ep_NNN directory number) rather than completion order.
    indexed_results: list[tuple[int, EpisodeResult]] = []
    cur_success = [False] * n_envs
    env_done = [False] * n_envs  # env slot retired after its episode is counted

    # Mirror BatchedVideoTracker ep numbering: env i starts on ep i, then
    # restarted envs get the next global ep number.
    ep_counter = n_envs
    env_ep = list(range(n_envs))

    pbar = tqdm(total=n_episodes, desc=f"Episodes ({n_envs} envs)", leave=True)

    while completed < n_episodes:
        actions = backend.predict_batch(obs)
        obs, rewards, terms, truncs, infos = vec_env.step(actions)

        for i in range(n_envs):
            if env_done[i]:
                continue

            if "success" in infos:
                s = infos["success"][i]
                if isinstance(s, (list, np.ndarray)):
                    s = bool(np.any(s))
                cur_success[i] |= bool(s)

            if "final_info" in infos and infos["final_info"][i] is not None:
                s = infos["final_info"][i].get("success", False)
                if isinstance(s, (list, np.ndarray)):
                    s = bool(np.any(s))
                cur_success[i] |= bool(s)

            if terms[i] or truncs[i]:
                if "final_info" in infos and infos["final_info"][i] is not None:
                    cur_success[i] |= bool(np.any(infos["final_info"][i]["success"]))
                success = cur_success[i]
                completed += 1
                ep_seed = env_ep[i] if deterministic else None
                if vt:
                    vt.finish_episode(
                        i, success, more_episodes=False, extra={"seed": ep_seed}
                    )
                indexed_results.append(
                    (
                        env_ep[i],
                        EpisodeResult(
                            success=success,
                            steps=0,
                            info={
                                "task_description": _task_descriptions[i],
                                "seed": ep_seed,
                            },
                        ),
                    )
                )
                rate = (
                    sum(r.success for _, r in indexed_results)
                    / len(indexed_results)
                    * 100
                )
                pbar.update(1)
                pbar.set_postfix(rate=f"{rate:.0f}%")
                cur_success[i] = False
                env_done[i] = True
                if completed >= n_episodes:
                    break

        # Restart only enough env slots to cover remaining episodes.
        # Don't restart if still-running envs already cover the gap.
        remaining = n_episodes - completed
        slots_to_reset: list[int] = []
        if remaining > 0:
            running = sum(1 for i in range(n_envs) if not env_done[i])
            need_restart = max(0, remaining - running)
            retired = [i for i in range(n_envs) if env_done[i]]
            for i in retired[:need_restart]:
                if vt:
                    vt._start(i)
                env_ep[i] = ep_counter
                ep_counter += 1
                env_done[i] = False
                slots_to_reset.append(i)

        # Explicitly reset recycled slots with deterministic seeds.
        # Gymnasium's autoreset already ran (NEXT_STEP mode) but without
        # a seed — this replaces it with a properly seeded reset.
        if slots_to_reset and deterministic:
            reset_mask = np.zeros(n_envs, dtype=bool)
            seed_list: list[int | None] = [None] * n_envs
            for i in slots_to_reset:
                reset_mask[i] = True
                seed_list[i] = env_ep[i]
            try:
                obs, _ = vec_env.reset(
                    seed=seed_list,
                    options={"reset_mask": reset_mask},
                )
                # Refresh task descriptions for reset slots.
                for _td_key in (
                    "annotation.human.action.task_description",
                    "annotation.human.task_description",
                ):
                    if _td_key in obs:
                        for i in slots_to_reset:
                            val = obs[_td_key][i]
                            _task_descriptions[i] = (
                                val if isinstance(val, str) else str(val)
                            )
                        break
            except (TypeError, NotImplementedError):
                # Old gymnasium (<1.0) without reset_mask support.
                import warnings

                warnings.warn(
                    "Gymnasium does not support reset_mask. "
                    "Recycled env slots will not be deterministically seeded.",
                    stacklevel=2,
                )

        if vt and completed < n_episodes:
            vt.record(obs)

    pbar.close()
    if vt:
        vt.close()
    vec_env.close()
    # Sort by ep_idx so successes[k] corresponds to ep_00k video directory.
    indexed_results.sort(key=lambda x: x[0])
    return [r for _, r in indexed_results[:n_episodes]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inference_policy(
    env: StepEnv | Callable[[], StepEnv],
    backend: PolicyBackend,
    config: InferencePolicyConfig | None = None,
    n_episodes: int = 1,
    n_envs: int = 1,
    deterministic: bool = True,
    video_dir: str | None = None,
    camera_map: dict[str, str] | None = None,
    task_name: str = "",
    model_name: str = "",
    model_path: str = "",
) -> list[EpisodeResult]:
    """Run a learned policy in an environment.

    Args:
        env:            For n_envs=1: an env instance (or callable).
                        For n_envs>1: a callable ``() -> env`` (required).
        backend:        Model server connection (ZMQ, HTTP, local, ...).
        config:         Chunking + episode parameters. Defaults to
                        sync receding horizon (predict 16, execute 8).
        n_episodes:     Total episodes to run.
        n_envs:         Parallel envs. 1 = sequential, >1 = AsyncVectorEnv
                        with MultiStepWrapper (same as legacy code).
        deterministic:  Seed each episode for reproducibility.
        video_dir:      If set and ``n_envs > 1``, save per-episode per-camera
                        videos under
                        ``{video_dir}/ep_000_s1/{side_left,side_right,wrist}.mp4``.
                        Single-env rollout video capture has been removed.
                        Episodes are numbered globally.
        task_name:      Overlay text on videos (e.g. "CloseDrawer").

    Returns:
        List of EpisodeResult (one per episode).
    """
    config = config or InferencePolicyConfig()

    if n_envs > n_episodes:
        import warnings

        warnings.warn(
            f"\033[33mn_envs ({n_envs}) > n_episodes ({n_episodes}). "
            f"Clamping n_envs to {n_episodes} to avoid wasting resources.\033[0m"
        )
        n_envs = n_episodes

    if n_envs <= 1:
        if video_dir:
            import warnings

            warnings.warn(
                "Single-env rollout video recording in cap.policy.inference "
                "has been removed. `video_dir` is ignored when n_envs <= 1.",
                stacklevel=2,
            )
        if callable(env) and not hasattr(env, "step"):
            env = env()
        return _run_single(
            env,
            backend,
            config,
            n_episodes,
            deterministic,
        )
    else:
        if not callable(env):
            raise ValueError("n_envs > 1 requires env to be a callable (env_fn)")
        return _run_batched(
            env,
            backend,
            config,
            n_episodes,
            n_envs,
            deterministic,
            video_dir,
            camera_map,
            task_name,
            model_name,
            model_path,
        )
