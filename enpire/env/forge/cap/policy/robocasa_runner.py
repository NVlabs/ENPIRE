"""Direct policy runner for RoboCasa envs.

Owns all GR00T ↔ robosuite format conversion — env.py stays a dumb data source.
Injected into skills.make_namespace via make_policy_runner(cfg).

env interface used here:
  env._obs                             raw robosuite obs dict
  env.render_rgb(alias)                render by CAP alias
  env.get_task_description()           cached task string
  env.step_dict(robosuite_dict)        step + success check → (raw_obs, reward, done, info)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# MuJoCo camera name → CAP alias used for ScriptRecorder
_CAM_MJ_TO_ALIAS = {
    "robot0_agentview_left": "top",
    "robot0_agentview_right": "right",
    "robot0_eye_in_hand": "wrist",
}

# Lazy-cached key-converter + camera config — loaded once per process
_KV = None
_CAM_POLICY_KEYS: list[str] | None = None   # GR00T obs keys  e.g. "video.robot0_agentview_left"
_CAM_MJ_NAMES: list[str] | None = None      # raw MuJoCo names e.g. "robot0_agentview_left"


def _get_kv():
    global _KV, _CAM_POLICY_KEYS, _CAM_MJ_NAMES
    if _KV is None:
        from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter

        _KV = PandaOmronKeyConverter
        _CAM_POLICY_KEYS, _CAM_MJ_NAMES, _, _ = PandaOmronKeyConverter.get_camera_config()
    return _KV, _CAM_POLICY_KEYS, _CAM_MJ_NAMES


def _build_obs(env: Any, raw_obs: dict) -> dict:
    """Build GR00T-format policy obs from raw robosuite state."""
    kv, cam_policy_keys, cam_mj_names = _get_kv()
    obs: dict = {}

    # Images — lazy property re-renders only when sim state changed since last read
    for policy_key, mj_name in zip(cam_policy_keys, cam_mj_names):
        obs[policy_key] = env.last_frames[_CAM_MJ_TO_ALIAS[mj_name]]

    # State — key_converter maps robosuite keys → hand.* / body.* → state.*
    for k, v in kv.map_obs(raw_obs).items():
        if k.startswith("hand.") or k.startswith("body."):
            obs["state." + k[5:]] = v

    obs["annotation.human.task_description"] = env.get_task_description()
    return obs


def _ms_stats(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(samples)
    n = len(s)
    return {
        "count": n,
        "mean": round(sum(s) / n, 2),
        "p50": round(s[n // 2], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "max": round(s[-1], 2),
    }


def make_policy_runner(cfg=None) -> Callable:
    """Return a runner callable for injection into skills.make_namespace.

    Signature: (env, task_description, **overrides) -> dict
    """

    def run_policy(env: Any, task_description: str, **overrides: Any) -> dict:
        from enpire.env.forge.cap.policy.backends import GROOTPOOL_BACKENDS

        pol = getattr(cfg, "policy", None) if cfg is not None else None

        backend_name = overrides.get("backend", getattr(pol, "backend", "grootpool"))
        version = overrides.get("model", getattr(pol, "model", "n15"))
        if isinstance(version, str) and "/" in version:
            backend_name, version = version.split("/", 1)
        model_str = f"{backend_name}/{version}"

        max_steps = int(overrides.get("max_episode_steps", 500))
        replan_horizon = int(overrides.get("replan_horizon", 16))
        endpoint = overrides.get("endpoint") or (
            getattr(pol, "endpoint", "") if pol is not None else ""
        )

        if backend_name != "grootpool":
            raise ValueError(f"Unknown backend {backend_name!r}. Supported: 'grootpool'.")
        if version not in GROOTPOOL_BACKENDS:
            raise ValueError(
                f"Unknown grootpool model {version!r}. Supported: {sorted(GROOTPOOL_BACKENDS)}"
            )
        backend_cls = GROOTPOOL_BACKENDS[version]

        required = getattr(backend_cls, "CONTROLLER_TYPE", None)
        if required and required.lower().strip() != "osc_pose":
            raise RuntimeError(
                f"Policy {model_str!r} declares CONTROLLER_TYPE={required!r}; "
                "RoboCasaEnv is pinned to osc_pose."
            )

        kv, _, _ = _get_kv()
        logger.info("run_policy: model=%s  task=%r", model_str, task_description)

        predict_ms: list[float] = []
        step_ms: list[float] = []
        obs = _build_obs(env, env._obs)
        action_chunk: dict | None = None
        chunk_idx = 0
        success = False
        n_steps = 0

        with backend_cls(
            task_description=task_description, endpoint=endpoint or None
        ) as backend:
            for n_steps in range(1, max_steps + 1):

                # Replan when chunk exhausted
                if action_chunk is None or chunk_idx >= replan_horizon:
                    t0 = time.monotonic()
                    action_chunk = backend.predict(obs)  # {k: (action_horizon, ...)}
                    predict_ms.append((time.monotonic() - t0) * 1000)
                    chunk_idx = 0

                # Slice one action from the chunk
                action = {k: v[chunk_idx] for k, v in action_chunk.items()}
                chunk_idx += 1

                # NaN/inf guard — GR00T diffusion can emit non-finite values
                for k, v in list(action.items()):
                    if (
                        isinstance(v, np.ndarray)
                        and np.issubdtype(v.dtype, np.floating)
                        and not np.all(np.isfinite(v))
                    ):
                        logger.warning(
                            "action[%r] has %d non-finite value(s) at step %d — zeroing",
                            k,
                            int(np.sum(~np.isfinite(v))),
                            n_steps,
                        )
                        action[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

                # Step: GR00T action dict → robosuite dict → numpy → physics
                t0 = time.monotonic()
                robosuite_dict = kv.unmap_action(dict(action))
                raw_obs, _, done, info = env.step_dict(robosuite_dict)
                obs = _build_obs(env, raw_obs)
                step_ms.append((time.monotonic() - t0) * 1000)

                success |= bool(info.get("success", False))
                if done or success:
                    break

        profiling = {
            "predict_ms": _ms_stats(predict_ms),
            "env_step_ms": _ms_stats(step_ms),
        }
        logger.info(
            "run_policy done: success=%s  steps=%d  predict_p50=%.1fms  step_p50=%.1fms",
            success,
            n_steps,
            profiling["predict_ms"]["p50"],
            profiling["env_step_ms"]["p50"],
        )
        return {
            "success": success,
            "steps": n_steps,
            "task_description": task_description,
            "model": model_str,
            "profiling": profiling,
        }

    return run_policy
