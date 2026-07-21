#!/usr/bin/env python3

from typing import Optional

import jax
import jax.numpy as jnp
from agentlace.trainer import TrainerConfig
from jax import nn
from ml_collections import ConfigDict

from serl_launcher.agents.continuous.sac_mini import SACMiniAgent
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.vision.data_augmentations import batched_random_crop


def make_sac_mini_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    temperature_init=1e-2,
    use_expectile=False,
    expectile=0.7,
    critic_warmup_steps=0,
    max_target_backup=False,
    on_the_fly=0,
    backup_entropy=False,
    dinov3_ckpt_path=None,
    resnet10_ckpt_path=None,
    resnet18_ckpt_path=None,
    freeze_visual_encoder=True,
    init_final=None,
):
    agent = SACMiniAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
            "activate_final": True,
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
            "activate_final": True,
        },
        temperature_init=temperature_init,
        discount=discount,
        backup_entropy=backup_entropy,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        use_expectile=use_expectile,
        expectile=expectile,
        critic_warmup_steps=critic_warmup_steps,
        max_target_backup=max_target_backup,
        on_the_fly=on_the_fly,
        freeze_visual_encoder=freeze_visual_encoder,
        resnet10_ckpt_path=resnet10_ckpt_path,
        resnet18_ckpt_path=resnet18_ckpt_path,
        init_final=init_final,
    )

    if encoder_type == "dinov3" and dinov3_ckpt_path is not None:
        from serl_launcher.utils.train_utils import load_dinov3_params

        agent = load_dinov3_params(agent, image_keys, dinov3_ckpt_path)

    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value


def make_batch_augmentation_func(image_keys) -> callable:
    # Training-only augmentation strength (no effect on the frozen inference path).
    # Larger crop padding => translation robustness to eval setup / object-pose offset
    # (the cycle-1/2 "batch A" visual-OOD failure mode). Configurable via env.
    import os as _os

    crop_pad = int(_os.environ.get("AUG_CROP_PAD", "4"))
    bright = float(_os.environ.get("AUG_BRIGHT", "0"))  # e.g. 0.2 => +-20% brightness

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            rng, crop_rng, b_rng = jax.random.split(rng, 3)
            img = batched_random_crop(
                observations[pixel_key], crop_rng, padding=crop_pad, num_batch_dims=2
            )
            if bright > 0:
                lead = img.shape[:-3]
                factor = jax.random.uniform(
                    b_rng, (*lead, 1, 1, 1), minval=1.0 - bright, maxval=1.0 + bright
                )
                img = jnp.clip(img.astype(jnp.float32) * factor, 0.0, 255.0).astype(img.dtype)
            observations = observations.copy(add_or_replace={pixel_key: img})
        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch

    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats", "get-network"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    project_dir=None,
    description: str = "serl_launcher",
    debug: bool = False,
    variant: Optional[ConfigDict] = {},
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=variant,
        debug=debug,
        wandb_output_dir=project_dir,
    )
    return wandb_logger

