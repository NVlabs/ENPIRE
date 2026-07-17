import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .reporting import (
    ActorSnapshotProvider,
    ActorStepEvent,
    BufferWaitEvent,
    LearnerSnapshotProvider,
    LearnerUpdateEvent,
    NetworkSyncEvent,
    make_actor_reporter,
    make_learner_reporter,
)

config = None

_REMOTE_IMAGE_KEYS = (
    "top_camera_image",
    "left_camera_image",
    "right_camera_image",
    "left_wrist_camera_image",
    "wrist_camera_image",
)


def _expand_offline_learn_argv(argv: list[str] | None = None) -> list[str]:
    """Translate the convenience tag before Hydra parses argv."""
    args = list(sys.argv if argv is None else argv)
    if "--offline-learn" not in args:
        return args

    filtered = [args[0]]
    filtered.extend(arg for arg in args[1:] if arg != "--offline-learn")
    offline_overrides = [
        "train.learner_start_server=false",
        "task.online_data_buffer_path=null",
    ]
    insert_at = len(filtered)
    for idx, arg in enumerate(filtered[1:], start=1):
        if arg.startswith("--"):
            insert_at = idx
            break
    return filtered[:insert_at] + offline_overrides + filtered[insert_at:]


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def import_runtime_deps():
    global jax, jnp, tqdm, checkpoints, Box
    global TrainerClient, TrainerServer, QueuedDataStore
    global RecordEpisodeStatistics
    global MemoryEfficientReplayBufferDataStore
    global create_env_launcher, make_sac_mini_pixel_agent
    global make_trainer_config, make_wandb_logger, concat_batches
    global devices, sharding

    import jax
    import jax.numpy as jnp
    import tqdm
    from agentlace.data.data_store import QueuedDataStore
    from agentlace.trainer import TrainerClient, TrainerServer
    from flax.training import checkpoints
    from gymnasium.spaces import Box
    from gymnasium.wrappers import RecordEpisodeStatistics
    from serl_launcher.data_andy_fix.data_store import MemoryEfficientReplayBufferDataStore
    from serl_launcher.env.env_launcher import create_env_launcher
    from serl_launcher.utils.launcher import (
        make_sac_mini_pixel_agent,
        make_trainer_config,
        make_wandb_logger,
    )
    from serl_launcher.utils.train_utils import concat_batches

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)


def _configured_env_action_repr(cfg: DictConfig) -> str:
    action_repr = OmegaConf.select(cfg, "env.action_repr", default=None)
    if action_repr is None or str(action_repr).strip() == "":
        raise ValueError("train_pld.py requires explicit env.action_repr")
    return str(action_repr)


def _action_dim_for_repr(action_repr: str, control_mode: str, configured_dim=None) -> int:
    if control_mode in ("left", "right"):
        per_arm_dims = {
            "joint": 7,
            "delta_eef_quat": 8,
            "delta_eef_rot6d": 10,
            "delta_eef_pos": 3,
        }
        if action_repr in per_arm_dims:
            return per_arm_dims[action_repr]

    if configured_dim is not None:
        return int(configured_dim)

    bimanual_dims = {
        "joint": 14,
        "delta_eef_quat": 16,
        "delta_eef_rot6d": 20,
        "delta_eef_pos": 6,
    }
    if action_repr not in bimanual_dims:
        raise ValueError(f"Unsupported action_repr={action_repr!r}")
    return bimanual_dims[action_repr]


def _configured_env_action_dim(cfg: DictConfig) -> int:
    action_repr = _configured_env_action_repr(cfg)
    control_mode = str(OmegaConf.select(cfg, "env.control_mode", default="both"))
    configured_dim = OmegaConf.select(cfg, "env.remote_action_dim", default=None)
    return _action_dim_for_repr(action_repr, control_mode, configured_dim)


def _configured_policy_action_repr(cfg: DictConfig) -> str:
    action_repr = OmegaConf.select(
        cfg,
        "task.action_repr",
        default=OmegaConf.select(cfg, "env.action_repr", default=None),
    )
    if action_repr is None or str(action_repr).strip() == "":
        raise ValueError("train_pld.py requires explicit task.action_repr or env.action_repr")
    return str(action_repr)


def _configured_policy_action_dim(cfg: DictConfig) -> int:
    action_repr = _configured_policy_action_repr(cfg)
    control_mode = str(OmegaConf.select(cfg, "env.control_mode", default="both"))
    configured_dim = OmegaConf.select(cfg, "task.policy_action_dim", default=None)
    return _action_dim_for_repr(action_repr, control_mode, configured_dim)


def _policy_action(actions, action_dim: int):
    return np.clip(np.asarray(actions, dtype=np.float32).reshape(action_dim), -1.0, 1.0)


def _action_transform_gamma(cfg: DictConfig) -> float:
    return float(OmegaConf.select(cfg, "action_transform.gamma", default=1.0))


def _identity_action_transform(actions):
    return np.asarray(actions, dtype=np.float32).copy()


def _resolve_dotted_callable(path: str):
    module_name, fn_name = str(path).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def _make_action_scaling_fns(cfg: DictConfig):
    try:
        scaling_cfg = cfg.action_scaling
    except Exception:
        scaling_cfg = None
    if scaling_cfg is None:
        return _identity_action_transform, _identity_action_transform
    if not bool(OmegaConf.select(cfg, "action_scaling.enabled", default=True)):
        return _identity_action_transform, _identity_action_transform

    fwd_path = OmegaConf.select(cfg, "action_scaling.action_scaling_fn", default=None)
    inv_path = OmegaConf.select(cfg, "action_scaling.action_scaling_inv_fn", default=None)
    if not fwd_path or not inv_path:
        raise ValueError(
            "action_scaling is enabled but action_scaling_fn/action_scaling_inv_fn "
            "are not configured."
        )
    fwd = _resolve_dotted_callable(str(fwd_path))
    inv = _resolve_dotted_callable(str(inv_path))

    def action_scaling_fn(actions):
        return fwd(actions, action_scaling_config=scaling_cfg)

    def inv_action_scaling_fn(actions):
        return inv(actions, action_scaling_config=scaling_cfg)

    return action_scaling_fn, inv_action_scaling_fn


def _configured_image_size(cfg: DictConfig) -> tuple[int, int]:
    return (
        int(OmegaConf.select(cfg, "env.image_height", default=256)),
        int(OmegaConf.select(cfg, "env.image_width", default=256)),
    )


def _actor_action_transform(actions, cfg: DictConfig, action_dim: int, action_scaling_fn):
    policy_action = _policy_action(actions, action_dim)
    exec_norm_action = np.clip(policy_action * _action_transform_gamma(cfg), -1.0, 1.0).astype(
        np.float32
    )
    sent_action = np.asarray(action_scaling_fn(exec_norm_action), dtype=np.float32).reshape(
        action_dim
    )
    return policy_action, exec_norm_action, sent_action


def _expand_policy_action_to_env_action(policy_raw_action, cfg: DictConfig):
    policy_repr = _configured_policy_action_repr(cfg)
    env_repr = _configured_env_action_repr(cfg)
    control_mode = str(OmegaConf.select(cfg, "env.control_mode", default="both"))
    policy_action = np.asarray(policy_raw_action, dtype=np.float32).reshape(-1)

    if policy_repr == env_repr:
        return policy_action

    if policy_repr == "delta_eef_pos" and env_repr == "delta_eef_rot6d":
        identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        closed_grip = np.array([0.0], dtype=np.float32)
        if control_mode in ("left", "right"):
            if policy_action.shape[0] != 3:
                raise ValueError(
                    "delta_eef_pos -> delta_eef_rot6d single-arm expansion "
                    f"expects 3-D policy action, got {policy_action.shape}"
                )
            return np.concatenate([policy_action, identity_rot6d, closed_grip]).astype(np.float32)
        if policy_action.shape[0] != 6:
            raise ValueError(
                "delta_eef_pos -> delta_eef_rot6d bimanual expansion expects "
                f"6-D policy action, got {policy_action.shape}"
            )
        return np.concatenate(
            [
                policy_action[0:3],
                identity_rot6d,
                closed_grip,
                policy_action[3:6],
                identity_rot6d,
                closed_grip,
            ]
        ).astype(np.float32)

    raise ValueError(f"Unsupported policy-to-env action expansion: {policy_repr!r} -> {env_repr!r}")


def _network_payload(agent, step: int):
    return {
        "pld_mini_network_payload_version": 1,
        "step": int(step),
        "params": agent.state.params,
    }


def _decode_network_payload(payload):
    if isinstance(payload, dict) and "params" in payload:
        return payload["params"]
    return payload


def _resolve_optional_path(path):
    if path is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path))).strip()
    return expanded or None


def _make_action_space(cfg: DictConfig):
    return Box(-1.0, 1.0, (_configured_policy_action_dim(cfg),), np.float32)


def _configured_image_keys(cfg: DictConfig) -> tuple[str, ...]:
    image_keys = tuple(str(key) for key in cfg.env.image_keys)
    allowed_keys = image_keys
    if str(OmegaConf.select(cfg, "env.env_name", default="")) == "remote_deployment":
        allowed_keys = tuple(dict.fromkeys((*image_keys, *_REMOTE_IMAGE_KEYS)))
    image_filter = OmegaConf.select(cfg, "env.image_filter", default=None)
    if not image_filter:
        return image_keys
    if not hasattr(image_filter, "get"):
        raise TypeError("env.image_filter must be a mapping with mode/include fields")

    mode = str(image_filter.get("mode", "drop"))
    if mode != "drop":
        raise ValueError(f"env.image_filter.mode={mode!r}; expected 'drop'")

    include = image_filter.get("include", None)
    if isinstance(include, str):
        include = [include]
    active_keys: list[str] = []
    for key in include or []:
        key = str(key)
        if key not in allowed_keys:
            raise ValueError(
                f"Unknown env.image_filter include key {key!r}; expected one of "
                f"{sorted(allowed_keys)}"
            )
        if key not in active_keys:
            active_keys.append(key)
    if not active_keys:
        raise ValueError("env.image_filter.include must contain at least one image key")
    return tuple(active_keys)


def _make_env(cfg: DictConfig, action_scaling_fn=None):
    env_launcher = create_env_launcher(cfg.env)
    env = env_launcher.create_environment(
        fake_env=bool(cfg.train.learner),
        classifier=cfg.task.use_classifier,
        render_mode=cfg.render_mode,
        action_scaling_fn=action_scaling_fn or _identity_action_transform,
        task_cfg=cfg.task,
    )
    env = RecordEpisodeStatistics(env)
    expected_action_dim = _configured_env_action_dim(cfg)
    if int(env.action_space.shape[-1]) != expected_action_dim:
        raise ValueError(
            "train_pld.py action-space mismatch: env.action_space has "
            f"{int(env.action_space.shape[-1])} dims, but config resolves to "
            f"{expected_action_dim} dims for env.action_repr="
            f"{_configured_env_action_repr(cfg)!r}."
        )
    return env


def _make_agent(cfg: DictConfig, sample_obs, sample_action):
    dinov3_ckpt_path = _resolve_optional_path(
        OmegaConf.select(cfg, "train.dinov3_ckpt_path", default=None)
    )
    resnet10_ckpt_path = _resolve_optional_path(
        OmegaConf.select(cfg, "train.resnet10_ckpt_path", default=None)
    )
    resnet18_ckpt_path = _resolve_optional_path(
        OmegaConf.select(cfg, "train.resnet18_ckpt_path", default=None)
    )
    return make_sac_mini_pixel_agent(
        seed=cfg.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=_configured_image_keys(cfg),
        encoder_type=cfg.train.encoder_type,
        discount=cfg.train.discount,
        temperature_init=cfg.train.d0813_temperature_init,
        reward_bias=cfg.train.reward_bias,
        use_expectile=cfg.train.use_expectile,
        expectile=cfg.train.expectile,
        critic_warmup_steps=cfg.train.critic_warmup_steps,
        max_target_backup=cfg.train.max_target_backup,
        on_the_fly=cfg.train.on_the_fly,
        backup_entropy=bool(OmegaConf.select(cfg, "train.backup_entropy", default=False)),
        dinov3_ckpt_path=dinov3_ckpt_path,
        resnet10_ckpt_path=resnet10_ckpt_path,
        resnet18_ckpt_path=resnet18_ckpt_path,
        freeze_visual_encoder=bool(
            OmegaConf.select(cfg, "train.freeze_visual_encoder", default=True)
        ),
        init_final=None,  # 1e-6 #
    )


def _make_replay_buffer(cfg: DictConfig, obs_space, action_space, capacity: int):
    include_mc_returns = bool(OmegaConf.select(cfg, "train.include_mc_returns", default=False))
    return MemoryEfficientReplayBufferDataStore(
        obs_space,
        action_space,
        capacity=capacity,
        image_keys=_configured_image_keys(cfg),
        include_grasp_penalty=False,
        include_base_action=False,
        include_next_base_action=False,
        include_mc_returns=include_mc_returns,
        discount=cfg.train.discount,
        env_name=cfg.env.env_name,
    )


def _restore_actor_eval_checkpoint(agent, cfg: DictConfig):
    ckpt_path = OmegaConf.select(cfg, "train.resume_checkpoint_path", default=None)
    if ckpt_path is None or str(ckpt_path).strip() == "":
        raise ValueError("train.eval_mode=true requires train.resume_checkpoint_path to be set.")

    ckpt_dir = os.path.abspath(os.path.expanduser(str(ckpt_path)))
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"SACMini eval checkpoint path does not exist: {ckpt_dir}")

    # If ckpt_dir points directly at a step directory (Orbax format has _CHECKPOINT_METADATA),
    # restore from it directly; otherwise look for checkpoint_* subdirs.
    is_step_dir = os.path.exists(os.path.join(ckpt_dir, "_CHECKPOINT_METADATA"))
    if is_step_dir:
        restored_state = checkpoints.restore_checkpoint(ckpt_dir, agent.state)
        step_desc = os.path.basename(ckpt_dir)
    else:
        step = int(OmegaConf.select(cfg, "train.eval_checkpoint_step", default=0) or 0)
        if step <= 0 and checkpoints.latest_checkpoint(ckpt_dir) is None:
            raise FileNotFoundError(f"SACMini eval found no checkpoints in: {ckpt_dir}")
        restore_kwargs = {"step": step} if step > 0 else {}
        restored_state = checkpoints.restore_checkpoint(ckpt_dir, agent.state, **restore_kwargs)
        step_desc = f"step {step}" if step > 0 else "latest checkpoint"
    print_green(f"SACMini actor eval restored {step_desc} from {ckpt_dir}.")
    return agent.replace(state=restored_state)


def _sample_action(agent, obs, rng_key, action_dim: int, *, deterministic: bool = False):
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        seed=rng_key,
        argmax=deterministic,
    )
    return np.asarray(jax.device_get(actions), dtype=np.float32).reshape(action_dim)


def actor(agent, env, sampling_rng, cfg: DictConfig, action_scaling_fn):
    action_dim = _configured_policy_action_dim(cfg)
    data_store = QueuedDataStore(int(cfg.train.actor_buffer_capacity))
    send_online_data = bool(OmegaConf.select(cfg, "train.actor_send_online_data", default=False))
    connect_to_learner = bool(OmegaConf.select(cfg, "train.connect_to_learner", default=True))
    eval_mode = bool(OmegaConf.select(cfg, "train.eval_mode", default=False)) or bool(
        OmegaConf.select(cfg, "train.actor_eval_only", default=False)
    )
    eval_deterministic = bool(OmegaConf.select(cfg, "train.eval_deterministic", default=True))
    if eval_mode and connect_to_learner:
        raise ValueError(
            "train.eval_mode=true restores a local checkpoint and must not be "
            "overwritten by learner sync; set train.connect_to_learner=false."
        )
    if eval_mode:
        agent = _restore_actor_eval_checkpoint(agent, cfg)
    client = None

    if connect_to_learner:
        trainer_config = make_trainer_config(
            port_number=cfg.agentlace_port_number,
            broadcast_port=cfg.agentlace_broadcast_port,
        )
        if "get-online-data-buffer" not in trainer_config.request_types:
            trainer_config.request_types.append("get-online-data-buffer")
        client = TrainerClient(
            "actor_env",
            cfg.ip,
            trainer_config,
            data_stores={"actor_env": data_store},
            wait_for_server=True,
            timeout_ms=int(
                OmegaConf.select(cfg, "train.actor_learner_request_timeout_ms", default=30000)
            ),
        )

    pbar = tqdm.tqdm(range(int(cfg.train.max_steps)), dynamic_ncols=True)
    snapshot_provider = ActorSnapshotProvider(
        env=env,
        data_store=data_store,
        cfg=cfg,
        send_online_data=send_online_data,
    )
    reporter = make_actor_reporter(
        cfg,
        snapshot_fn=snapshot_provider.snapshot,
        client=client,
        pbar=pbar,
    )
    reporter.start()

    if client is None:
        reporter.network_sync(
            NetworkSyncEvent(role="actor", status="disabled", message="no learner client")
        )
    else:
        reporter.network_sync(NetworkSyncEvent(role="actor", status="connected"))

    try:

        def update_params(payload):
            nonlocal agent
            params = _decode_network_payload(payload)
            agent = agent.replace(state=agent.state.replace(params=params))
            reporter.network_sync(
                NetworkSyncEvent(
                    role="actor",
                    status="received",
                    step=int(payload.get("step", 0)) if isinstance(payload, dict) else None,
                )
            )
            print_green("Actor received learner network sync.")

        if client is not None:
            client.recv_network_callback(update_params)
            reporter.network_sync(NetworkSyncEvent(role="actor", status="request"))
            response = client.request(
                "get-network",
                {"reason": "train_pld_actor_initial_sync", "want_full": True},
            )
            if response:
                update_params(response)

        reporter.phase("actor", "waiting for robot env")
        obs, _ = env.reset()
        num_episodes = 0
        running_return = 0.0

        for step in pbar:
            warmup = (not eval_mode) and num_episodes < int(
                OmegaConf.select(cfg, "train.warmup_episodes", default=0)
            )
            phase = "eval" if eval_mode else "warmup" if warmup else "rl rollout"
            if bool(OmegaConf.select(cfg, "train.deactivate_rl_actor", default=False)):
                algo_action = np.zeros(action_dim, dtype=np.float32)
                source = "zero"
            elif warmup:
                if bool(OmegaConf.select(cfg, "train.d0812_warmup_actor", default=False)):
                    scale = float(
                        OmegaConf.select(cfg, "train.d0813_init_delta_scale", default=1.0)
                    )
                    algo_action = (
                        np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32) * scale
                    )
                    algo_action = np.clip(algo_action, -1.0, 1.0)
                    source = "warmup_random"
                else:
                    algo_action = np.zeros(action_dim, dtype=np.float32)
                    source = "warmup_zero"
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                algo_action = _sample_action(
                    agent,
                    obs,
                    key,
                    action_dim,
                    deterministic=eval_mode and eval_deterministic,
                )
                source = (
                    "eval"
                    if eval_mode and eval_deterministic
                    else "eval_stochastic"
                    if eval_mode
                    else "rl"
                )

            policy_action, exec_norm_action, sent_action = _actor_action_transform(
                algo_action, cfg, action_dim, action_scaling_fn
            )
            env_action = _expand_policy_action_to_env_action(sent_action, cfg)
            event = ActorStepEvent.from_arrays(
                step=step,
                episode=num_episodes,
                phase=phase,
                source=source,
                algo_action=policy_action,
                exec_norm_action=exec_norm_action,
                sent_action=sent_action,
            )
            reporter.actor_step(event)
            reporter.phase("actor", "waiting for robot step")

            next_obs, reward, done, truncated, info = env.step(env_action)
            episode_boundary = bool(done or truncated)
            transition = {
                "observations": obs,
                "actions": policy_action,
                "next_observations": next_obs,
                "rewards": reward,
                "masks": 1.0 - float(episode_boundary),
                "dones": episode_boundary,
            }
            if send_online_data:
                data_store.insert(transition)

            running_return += float(reward)
            reporter.actor_step(
                event.with_outcome(
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    running_return=running_return,
                )
            )
            obs = next_obs

            if episode_boundary:
                if client is not None:
                    client.update()
                running_return = 0.0
                num_episodes += 1
                reporter.phase("actor", "waiting for robot env")
                obs, _ = env.reset()
    finally:
        reporter.stop()
        pbar.close()


def _disk_inverse_action_fn(action_dim: int, inv_action_scaling_fn):
    def inv(raw_action):
        return np.asarray(inv_action_scaling_fn(raw_action), dtype=np.float32).reshape(action_dim)

    return inv


def _path_list_from_cfg(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [
            part.strip()
            for part in stripped.replace(",", os.pathsep).split(os.pathsep)
            if part.strip()
        ]
    try:
        return [str(part).strip() for part in value if str(part).strip()]
    except TypeError:
        text = str(value).strip()
        return [text] if text else []


def _configured_reuse_buffer_dirs(cfg: DictConfig) -> list[str]:
    dirs = _path_list_from_cfg(OmegaConf.select(cfg, "task.reuse_buffer_dirs", default=[]))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in dirs:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _configured_online_data_buffer_path(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "task.online_data_buffer_path", default="") or "")


def _fetch_online_data_buffer_path(
    handshake_url: str,
    timeout_ms: int = 30000,
    *,
    retry_s: float = 0.0,
    max_wait_s: float = 0.0,
) -> str:
    """Ask the forge runner for the current online-data-buffer directory."""
    import pickle

    import lz4.frame
    import zmq

    print_green(f"Requesting online-buffer path from forge: {handshake_url}")
    retry_s = max(0.0, float(retry_s))
    max_wait_s = float(max_wait_s)
    deadline = None if max_wait_s < 0.0 else time.monotonic() + max_wait_s
    attempt = 0
    last_exc: Exception | None = None
    while True:
        attempt += 1
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.connect(handshake_url)
        try:
            msg = {"type": "get-online-data-buffer", "payload": {}}
            sock.send(lz4.frame.compress(pickle.dumps(msg)))
            response = pickle.loads(lz4.frame.decompress(sock.recv()))
            break
        except zmq.Again as exc:
            last_exc = exc
        finally:
            sock.close()
            ctx.term()

        if retry_s <= 0.0 or (deadline is not None and time.monotonic() >= deadline):
            raise last_exc
        wait_s = (
            retry_s if deadline is None else min(retry_s, max(0.0, deadline - time.monotonic()))
        )
        print_green(
            "Forge online-buffer handshake not ready "
            f"(attempt {attempt}); retrying in {wait_s:.1f}s."
        )
        time.sleep(wait_s)
    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"online-buffer handshake failed: {response!r}")
    path = str(response.get("online_data_buffer_path") or "").strip()
    if not path:
        raise RuntimeError("online-buffer handshake returned an empty path")
    print_green(f"Received online-buffer path: {path}")
    return path


def _make_disk_ingestor(
    *,
    root: str,
    cfg: DictConfig,
    replay_buffer,
    demo_buffer,
    action_space,
    inv_action_scaling_fn,
    ingested_episode_paths: set[Path],
    force_human_to_replay: bool | None = None,
):
    from .disk_buffer_ingestor import DiskBufferIngestor

    human_to_replay = (
        bool(OmegaConf.select(cfg, "task.human_to_replay", default=False))
        if force_human_to_replay is None
        else bool(force_human_to_replay)
    )

    action_dim = int(action_space.shape[-1])
    prefer_delta_eef = bool(OmegaConf.select(cfg, "task.prefer_delta_eef", default=True))
    action_repr = OmegaConf.select(
        cfg,
        "task.action_repr",
        default=OmegaConf.select(cfg, "env.action_repr", default=None),
    )
    if not action_repr:
        raise ValueError(
            "disk buffer ingestion requires task.action_repr or env.action_repr to be explicit."
        )
    return DiskBufferIngestor(
        root=root,
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        buffer_update_freq=float(OmegaConf.select(cfg, "task.buffer_update_freq", default=30.0)),
        action_dim=action_dim,
        control_mode=str(OmegaConf.select(cfg, "env.control_mode", default="right")),
        unknown_to_replay=bool(OmegaConf.select(cfg, "task.unknown_to_replay", default=False)),
        human_to_replay=human_to_replay,
        prefer_delta_eef=prefer_delta_eef,
        action_repr=str(action_repr),
        strict_source_labels=bool(OmegaConf.select(cfg, "task.strict_source_labels", default=True)),
        inv_action_scaling_fn=_disk_inverse_action_fn(action_dim, inv_action_scaling_fn),
        action_gamma=_action_transform_gamma(cfg),
        use_base_actions=False,  # SACMini stores direct policy actions.
        image_size=_configured_image_size(cfg),
        min_episode_age_s=float(
            OmegaConf.select(cfg, "task.disk_buffer_min_episode_age_s", default=0.0)
        ),
        image_keys=_configured_image_keys(cfg),
        proprio_keys=tuple(cfg.env.proprio_keys),
        proprio_filter=OmegaConf.to_container(
            OmegaConf.select(cfg, "env.proprio_filter", default={}), resolve=True
        ),
        ingested_episode_paths=ingested_episode_paths,
    )


def _save_run_configs(cfg: DictConfig, ckpt_root: str) -> None:
    os.makedirs(ckpt_root, exist_ok=True)
    with open(os.path.join(ckpt_root, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    hydra_cfg = HydraConfig.get()
    choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
    configs_root = next(
        (str(s.path) for s in hydra_cfg.runtime.config_sources if s.schema == "file"),
        None,
    )
    if configs_root is None:
        return
    for group in ("system", "experiment"):
        name = choices.get(group)
        if not name:
            continue
        src = os.path.join(configs_root, group, f"{name}.yaml")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ckpt_root, f"{name}.yaml"))


def learner(agent, env, action_space, sampling_rng, cfg: DictConfig, inv_action_scaling_fn):
    replay_buffer = _make_replay_buffer(
        cfg,
        env.observation_space,
        action_space,
        int(cfg.train.replay_buffer_capacity),
    )
    demo_buffer = _make_replay_buffer(
        cfg,
        env.observation_space,
        action_space,
        int(cfg.train.demo_buffer_capacity),
    )
    wandb_logger = make_wandb_logger(
        project=cfg.wandb.project,
        project_dir=os.path.join(cfg.system.log_dir, "wandb"),
        description=f"{cfg.env.env_name}_{cfg.train.algo_name}",
        debug=cfg.debug,
        variant={
            **OmegaConf.to_container(cfg, resolve=True),
            "agent_config": dict(**agent.config),
        },
    )

    disk_ingestor = None
    ingested_episode_paths: set[Path] = set()

    def set_online_data_buffer_path(online_data_buffer_path: str):
        nonlocal disk_ingestor
        root_text = str(online_data_buffer_path or "").strip()
        if not root_text:
            raise ValueError("online data buffer path is empty")
        root = Path(root_text).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        root_text = str(root)
        OmegaConf.update(cfg, "task.online_data_buffer_path", root_text, merge=False)
        current_root = getattr(disk_ingestor, "root", None)
        if disk_ingestor is not None and str(current_root) == root_text:
            print_green(f"Handshake already using online data buffer: {root_text}")
            return {"ok": True, "online_data_buffer_path": root_text}
        if disk_ingestor is None:
            print_green(f"DiskBufferIngestor watching online data buffer: {root_text}")
            disk_ingestor = _make_disk_ingestor(
                root=root_text,
                cfg=cfg,
                replay_buffer=replay_buffer,
                demo_buffer=demo_buffer,
                action_space=action_space,
                inv_action_scaling_fn=inv_action_scaling_fn,
                ingested_episode_paths=ingested_episode_paths,
            )
        else:
            disk_ingestor.stop()
            disk_ingestor.root = root
            print_green(f"DiskBufferIngestor switched online data buffer: {root_text}")
        disk_ingestor.start()
        return {"ok": True, "online_data_buffer_path": root_text}

    profile_stats = {
        "updates": 0,
        "critic_updates": 0,
        "last_step_duration_s": 0.0,
        "last_sample_duration_s": 0.0,
        "last_critic_update_duration_s": 0.0,
        "last_train_update_duration_s": 0.0,
        "publish_count": 0,
        "last_publish_step": None,
        "last_publish_duration_s": 0.0,
        "last_publish_interval_s": None,
        "last_publish_epoch": None,
    }
    train_gate = {
        "started": False,
        "training_starts": int(cfg.train.training_starts),
        "replay_ready": False,
        "mc_required": bool(OmegaConf.select(cfg, "train.include_mc_returns", default=False)),
        "mc_ready": True,
    }
    reporter = None
    server = None

    if bool(OmegaConf.select(cfg, "train.learner_start_server", default=True)):
        handshake_url = OmegaConf.select(cfg, "forge_handshake_url", default=None)
        if not handshake_url:
            raise ValueError(
                "forge_handshake_url must be configured to pull the online-buffer "
                "path from the forge runner"
            )
        handshake_timeout_ms = int(
            OmegaConf.select(cfg, "forge_handshake_timeout_ms", default=30000)
        )
        handshake_retry_s = float(OmegaConf.select(cfg, "forge_handshake_retry_s", default=0.0))
        handshake_max_wait_s = float(
            OmegaConf.select(cfg, "forge_handshake_max_wait_s", default=0.0)
        )
        trainer_config = make_trainer_config(
            port_number=cfg.agentlace_port_number,
            broadcast_port=cfg.agentlace_broadcast_port,
        )
        if "get-online-data-buffer" not in trainer_config.request_types:
            trainer_config.request_types.append("get-online-data-buffer")

        def request_callback(request_type: str, payload: dict) -> dict:
            if request_type == "send-stats":
                actor_action = payload.get("actor_action") if isinstance(payload, dict) else None
                if reporter is not None and isinstance(actor_action, dict):
                    reporter.actor_step(ActorStepEvent.from_stats(actor_action))
                return {}
            if request_type == "get-online-data-buffer":
                root = getattr(disk_ingestor, "root", None)
                root_text = (
                    str(root) if root is not None else _configured_online_data_buffer_path(cfg)
                )
                return {"ok": True, "online_data_buffer_path": root_text}
            if request_type == "get-network":
                return _network_payload(agent, int(agent.state.step))
            raise ValueError(f"Invalid request type: {request_type}")

        server = TrainerServer(trainer_config, request_callback=request_callback)
        server.register_data_store("actor_env", replay_buffer)
        server.start(threaded=True)
        print_green(
            "Learner agentlace server listening: "
            f"request={cfg.agentlace_port_number} "
            f"broadcast={cfg.agentlace_broadcast_port}"
        )
        buffer_path = _fetch_online_data_buffer_path(
            str(handshake_url),
            timeout_ms=handshake_timeout_ms,
            retry_s=handshake_retry_s,
            max_wait_s=handshake_max_wait_s,
        )
        set_online_data_buffer_path(buffer_path)
    else:
        online_data_buffer_path = _configured_online_data_buffer_path(cfg)
        if online_data_buffer_path:
            set_online_data_buffer_path(online_data_buffer_path)

    for reuse_root in _configured_reuse_buffer_dirs(cfg):
        reuse_path = Path(reuse_root).expanduser()
        if not reuse_path.exists():
            raise FileNotFoundError(f"reuse buffer directory does not exist: {reuse_path}")
        if not reuse_path.is_dir():
            raise NotADirectoryError(f"reuse buffer path is not a directory: {reuse_path}")
        print_green(f"Preloading reuse buffer: {reuse_path}")
        preload_ingestor = _make_disk_ingestor(
            root=str(reuse_path),
            cfg=cfg,
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            action_space=action_space,
            inv_action_scaling_fn=inv_action_scaling_fn,
            ingested_episode_paths=ingested_episode_paths,
        )
        ingested = preload_ingestor.scan_once(show_progress=True)
        print()
        preload_stats = preload_ingestor.stats()
        print_green(
            "Preloaded reuse buffer "
            f"{reuse_path}: episodes={ingested} "
            f"transitions={preload_stats.get('transitions', 0)} "
            f"replay_size={len(replay_buffer)} demo_size={len(demo_buffer)}"
        )

    # Contrastive BC: failure demos loaded into demo_buffer as negatives.
    for demo_root in _path_list_from_cfg(
        OmegaConf.select(cfg, "task.reuse_buffer_demo_dirs", default=[])
    ):
        demo_path = Path(demo_root).expanduser()
        if not demo_path.is_dir():
            raise NotADirectoryError(f"reuse_buffer_demo_dir is not a directory: {demo_path}")
        print_green(f"Preloading NEGATIVE demo buffer: {demo_path}")
        neg_ingestor = _make_disk_ingestor(
            root=str(demo_path),
            cfg=cfg,
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            action_space=action_space,
            inv_action_scaling_fn=inv_action_scaling_fn,
            ingested_episode_paths=ingested_episode_paths,
            force_human_to_replay=False,  # route failures to demo_buffer
        )
        neg_ingestor.scan_once(show_progress=True)
        print()
        print_green(
            f"Preloaded negatives {demo_path}: "
            f"replay_size={len(replay_buffer)} demo_size={len(demo_buffer)}"
        )

    if cfg.train.resume and cfg.train.resume_checkpoint_path:
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(cfg.train.resume_checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        print("\033[1;92m" + "=" * 80, flush=True)
        print(f"  RESUMED FROM CHECKPOINT: {os.path.abspath(cfg.train.resume_checkpoint_path)}")
        print("=" * 80 + "\033[00m", flush=True)

    if bool(OmegaConf.select(cfg, "train.eval_only", default=False)):
        import numpy as np
        from serl_launcher.utils.train_utils import _unpack

        n_batches = int(OmegaConf.select(cfg, "train.eval_batches", default=60))
        ckpt_list = OmegaConf.select(cfg, "train.eval_checkpoints", default=None)
        if ckpt_list:
            ckpt_list = [str(p) for p in ckpt_list]
        else:
            ckpt_list = [os.path.abspath(cfg.train.resume_checkpoint_path)]

        # Pre-sample a fixed set of val batches so every checkpoint is scored on
        # identical data (apples-to-apples selection).
        eval_it = replay_buffer.get_iterator(
            sample_args={"batch_size": int(cfg.train.batch_size), "pack_obs_and_next_obs": True},
            device=sharding.replicate(),
        )
        fixed_batches = []
        for _ in range(n_batches):
            batch = next(eval_it)
            if agent.config["image_keys"][0] not in batch["next_observations"]:
                batch = _unpack(batch)
            fixed_batches.append(batch)

        rng = sampling_rng
        results = {}
        for ckpt_path in ckpt_list:
            restored = checkpoints.restore_checkpoint(os.path.abspath(ckpt_path), agent.state)
            eval_agent = agent.replace(state=restored)
            sq_err = []
            pred_all = []
            for batch in fixed_batches:
                rng, key = jax.random.split(rng)
                pred = np.asarray(
                    eval_agent.sample_actions(batch["observations"], seed=key, argmax=True)
                )
                tgt = np.asarray(batch["actions"])
                sq_err.append((pred - tgt) ** 2)
                pred_all.append(pred)
            sq = np.concatenate(sq_err, axis=0)
            mse = float(sq.mean())
            per_dim = [round(float(x), 5) for x in sq.mean(axis=0).tolist()]
            preds = np.concatenate(pred_all, axis=0)
            mean_pred = [round(float(x), 4) for x in preds.mean(axis=0).tolist()]
            frac_down = round(float((preds[:, 2] < 0).mean()), 3)  # z<0 = descending
            results[ckpt_path] = mse
            print_green(
                f"EVAL_ONLY {os.path.basename(ckpt_path)}: buffer={len(replay_buffer)} "
                f"MSE={mse:.5f} per_dim={per_dim} mean_pred={mean_pred} frac_down_z={frac_down}"
            )
        best = min(results, key=results.get)
        print_green(f"EVAL_ONLY BEST: {os.path.basename(best)} MSE={results[best]:.5f}")
        return

    reporter = make_learner_reporter(
        cfg,
        snapshot_fn=LearnerSnapshotProvider(
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            disk_ingestor=disk_ingestor,
            cfg=cfg,
            profile_stats=profile_stats,
            train_gate=train_gate,
        ).snapshot,
    )
    reporter.start()

    if server is not None:
        publish_t0 = time.perf_counter()
        server.publish_network(_network_payload(agent, 0))
        profile_stats["publish_count"] += 1
        profile_stats["last_publish_step"] = 0
        profile_stats["last_publish_duration_s"] = time.perf_counter() - publish_t0
        profile_stats["last_publish_epoch"] = time.time()
        reporter.network_sync(NetworkSyncEvent(role="learner", status="published", step=0))
        print_green("Published initial SACMini network to actors.")

    try:
        training_starts = int(cfg.train.training_starts)
        while len(replay_buffer) < training_starts:
            train_gate["replay_ready"] = len(replay_buffer) >= training_starts
            reporter.buffer_wait(
                BufferWaitEvent(
                    replay_size=len(replay_buffer),
                    demo_size=len(demo_buffer),
                    target_size=training_starts,
                )
            )
            print(
                f"\r  waiting for replay buffer: {len(replay_buffer)}/{training_starts}",
                end="",
                flush=True,
            )
            time.sleep(1)
        print()
        train_gate["replay_ready"] = True
        train_gate["started"] = True

        has_demos = len(demo_buffer) > 0
        replay_batch_size = (
            int(cfg.train.batch_size) // 2 if has_demos else int(cfg.train.batch_size)
        )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={"batch_size": replay_batch_size, "pack_obs_and_next_obs": True},
            device=sharding.replicate(),
        )
        demo_iterator = None
        if has_demos:
            demo_iterator = demo_buffer.get_iterator(
                sample_args={
                    "batch_size": int(cfg.train.batch_size) // 2,
                    "pack_obs_and_next_obs": True,
                },
                device=sharding.replicate(),
            )

        critic_update_set = frozenset({"critic"})
        full_update_set = frozenset({"critic", "actor", "temperature"})
        start_step = 0
        bc_mode = bool(OmegaConf.select(cfg, "train.bc_mode", default=False))
        neg_bc = bool(OmegaConf.select(cfg, "train.neg_bc", default=False)) and (
            demo_iterator is not None
        )
        if neg_bc:
            agent.config["neg_bc_weight"] = float(
                OmegaConf.select(cfg, "train.neg_bc_weight", default=0.3)
            )
            agent.config["neg_bc_margin"] = float(
                OmegaConf.select(cfg, "train.neg_bc_margin", default=0.0)
            )
        if bc_mode:
            msg = "CONTRASTIVE BC" if neg_bc else "BC MODE"
            print_green(
                f"{msg}: actor-only; success=replay neg=demo "
                f"(neg_bc={neg_bc}, w={agent.config.get('neg_bc_weight')})."
            )

        for step in range(start_step, int(cfg.train.max_steps)):
            step_t0 = time.perf_counter()
            sample_duration = 0.0
            critic_update_duration = 0.0
            train_update_duration = 0.0
            if bc_mode:
                sample_t0 = time.perf_counter()
                if neg_bc:
                    pos_batch = next(replay_iterator)
                    neg_batch = next(demo_iterator)
                    sample_duration += time.perf_counter() - sample_t0
                    update_t0 = time.perf_counter()
                    agent, update_info = agent.update_neg_bc(pos_batch, neg_batch)
                else:
                    batch = next(replay_iterator)
                    if demo_iterator is not None:
                        batch = concat_batches(batch, next(demo_iterator), axis=0)
                    sample_duration += time.perf_counter() - sample_t0
                    update_t0 = time.perf_counter()
                    agent, update_info = agent.update_bc(batch)
                jax.block_until_ready(update_info)
                train_update_duration += time.perf_counter() - update_t0
                profile_stats["updates"] += 1
                profile_stats["last_step_duration_s"] = time.perf_counter() - step_t0
                profile_stats["last_sample_duration_s"] = sample_duration
                profile_stats["last_critic_update_duration_s"] = critic_update_duration
                profile_stats["last_train_update_duration_s"] = train_update_duration
                reporter.learner_update(
                    LearnerUpdateEvent(
                        step=step,
                        metrics=update_info,
                        replay_size=len(replay_buffer),
                        demo_size=len(demo_buffer),
                    )
                )
                if step % int(cfg.train.log_period) == 0:
                    wandb_logger.log(update_info, step=step)
                if (
                    step > 0
                    and cfg.train.checkpoint_period
                    and step % int(cfg.train.checkpoint_period) == 0
                ):
                    ckpt_root = os.path.abspath(cfg.train.save_checkpoint_path)
                    checkpoints.save_checkpoint(
                        ckpt_root,
                        agent.state,
                        step=step,
                        keep=int(cfg.train.num_checkpoints_to_keep),
                        overwrite=True,
                    )
                    if not os.path.exists(os.path.join(ckpt_root, "config.yaml")):
                        _save_run_configs(cfg, ckpt_root)
                continue
            for _ in range(max(0, int(cfg.train.cta_ratio) - 1)):
                sample_t0 = time.perf_counter()
                batch = next(replay_iterator)
                if demo_iterator is not None:
                    batch = concat_batches(batch, next(demo_iterator), axis=0)
                sample_duration += time.perf_counter() - sample_t0
                update_t0 = time.perf_counter()
                agent, critic_info = agent.update(batch, networks_to_update=critic_update_set)
                jax.block_until_ready(critic_info)
                critic_update_duration += time.perf_counter() - update_t0
                profile_stats["critic_updates"] += 1

            sample_t0 = time.perf_counter()
            batch = next(replay_iterator)
            if demo_iterator is not None:
                batch = concat_batches(batch, next(demo_iterator), axis=0)
            sample_duration += time.perf_counter() - sample_t0
            update_t0 = time.perf_counter()
            agent, update_info = agent.update(batch, networks_to_update=full_update_set)
            jax.block_until_ready(update_info)
            train_update_duration += time.perf_counter() - update_t0
            profile_stats["updates"] += 1
            profile_stats["last_step_duration_s"] = time.perf_counter() - step_t0
            profile_stats["last_sample_duration_s"] = sample_duration
            profile_stats["last_critic_update_duration_s"] = critic_update_duration
            profile_stats["last_train_update_duration_s"] = train_update_duration
            reporter.learner_update(
                LearnerUpdateEvent(
                    step=step,
                    metrics=update_info,
                    replay_size=len(replay_buffer),
                    demo_size=len(demo_buffer),
                )
            )

            if server is not None and step > 0 and step % int(cfg.train.steps_per_update) == 0:
                publish_t0 = time.perf_counter()
                prev_publish_epoch = profile_stats["last_publish_epoch"]
                server.publish_network(_network_payload(agent, step))
                now_epoch = time.time()
                profile_stats["publish_count"] += 1
                profile_stats["last_publish_step"] = step
                profile_stats["last_publish_duration_s"] = time.perf_counter() - publish_t0
                if isinstance(prev_publish_epoch, (int, float)):
                    profile_stats["last_publish_interval_s"] = now_epoch - float(prev_publish_epoch)
                profile_stats["last_publish_epoch"] = now_epoch
                reporter.network_sync(
                    NetworkSyncEvent(role="learner", status="published", step=step)
                )

            if step % int(cfg.train.log_period) == 0:
                wandb_logger.log(update_info, step=step)

            if (
                step > 0
                and cfg.train.checkpoint_period
                and step % int(cfg.train.checkpoint_period) == 0
            ):
                ckpt_root = os.path.abspath(cfg.train.save_checkpoint_path)
                checkpoints.save_checkpoint(
                    ckpt_root,
                    agent.state,
                    step=step,
                    keep=int(cfg.train.num_checkpoints_to_keep),
                    overwrite=True,
                )
                if not os.path.exists(os.path.join(ckpt_root, "config.yaml")):
                    _save_run_configs(cfg, ckpt_root)
    finally:
        reporter.stop()
        if disk_ingestor is not None:
            disk_ingestor.stop()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    global config
    config = cfg

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if cfg.train.actor:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.system.actor_gpu_mem_fraction)
    else:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.system.learner_gpu_mem_fraction)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cfg.system.jax_cache_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.system.cuda_device_id)

    import_runtime_deps()

    if str(cfg.train.algo_name) != "sac_mini":
        raise ValueError("train_pld.py only supports train.algo_name=sac_mini")

    action_scaling_fn, inv_action_scaling_fn = _make_action_scaling_fns(cfg)
    action_space = _make_action_space(cfg)
    env = _make_env(cfg, action_scaling_fn=action_scaling_fn)
    sample_obs = env.observation_space.sample()
    sample_action = action_space.sample()
    agent = _make_agent(cfg, sample_obs, sample_action)
    agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())

    rng = jax.random.PRNGKey(cfg.seed)
    rng, sampling_rng = jax.random.split(rng)
    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

    if cfg.train.learner:
        learner(agent, env, action_space, sampling_rng, cfg, inv_action_scaling_fn)
    elif cfg.train.actor:
        actor(agent, env, sampling_rng, cfg, action_scaling_fn)
    else:
        raise ValueError("Set exactly one of train.learner=true or train.actor=true")


if __name__ == "__main__":
    sys.argv = _expand_offline_learn_argv()
    main()
