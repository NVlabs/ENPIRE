from functools import partial
from typing import FrozenSet, Iterable, Optional, Tuple

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, ensemblize
from serl_launcher.networks.actor_critic_nets import PolicyBase as Policy
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.math import expectile_regression_loss
from serl_launcher.utils.train_utils import _unpack


def _policy_sigma_info(action_distributions: distrax.Distribution) -> dict:
    base_distribution = getattr(action_distributions, "distribution", action_distributions)
    sigmas = base_distribution.stddev()
    info = {
        "policy_sigma_mean": jnp.mean(sigmas),
        "policy_sigma_min": jnp.min(sigmas),
        "policy_sigma_max": jnp.max(sigmas),
    }
    if sigmas.shape[-1] >= 1:
        info["policy_sigma_dim0"] = jnp.mean(sigmas[..., 0])
    if sigmas.shape[-1] >= 2:
        info["policy_sigma_dim1"] = jnp.mean(sigmas[..., 1])
    if sigmas.shape[-1] >= 3:
        info["policy_sigma_dim2"] = jnp.mean(sigmas[..., 2])
    return info


class SACMiniAgent(flax.struct.PyTreeNode):
    """Minimal continuous SAC.

    This class intentionally models the algorithm action directly:
      - no base action input,
      - no residual/base composition,
      - no environment action scaling,
      - no action masking.

    Robot-specific execution transforms belong outside the algorithm, e.g. in
    ``get_exec_action()`` in the actor/data path.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        return self.forward_critic(
            observations,
            actions,
            rng=rng,
            grad_params=self.state.target_params,
        )

    @jax.jit
    def jitted_forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        return self.forward_target_critic(observations, actions, rng)

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_policy_and_sample(
        self,
        obs: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        repeat: Optional[int] = None,
    ):
        rng, sample_rng = jax.random.split(rng)
        action_dist = self.forward_policy(obs, rng, grad_params=grad_params)
        if repeat:
            new_actions, log_pi = action_dist.sample_and_log_prob(
                seed=sample_rng, sample_shape=repeat
            )
            new_actions = jnp.transpose(new_actions, (1, 0, 2))
            log_pi = jnp.transpose(log_pi, (1, 0))
        else:
            new_actions, log_pi = action_dist.sample_and_log_prob(seed=sample_rng)
        return new_actions, log_pi

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng, on_the_fly: bool = False):
        batch_size = batch["rewards"].shape[0]
        sample_n_actions = self.config["on_the_fly"] if on_the_fly else None

        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["next_observations"],
            rng,
            repeat=sample_n_actions,
        )

        if sample_n_actions:
            chex.assert_shape(next_actions_log_probs, (batch_size, sample_n_actions))
        else:
            chex.assert_shape(next_actions_log_probs, (batch_size,))
        return next_actions, next_actions_log_probs

    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_next_qs = target_next_qs - temperature * next_actions_log_probs

        if self.config["max_target_backup"]:
            max_target_indices = jnp.expand_dims(jnp.argmax(target_next_qs, axis=-1), -1)
            target_next_qs = jnp.take_along_axis(
                target_next_qs, max_target_indices, axis=-1
            ).squeeze(-1)

        return target_next_qs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch,
            next_action_sample_key,
            on_the_fly=self.config["max_target_backup"],
        )

        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_equal_shape([target_next_min_q, next_actions_log_probs])
        target_next_min_q = self._process_target_next_qs(
            target_next_min_q,
            next_actions_log_probs,
        )

        target_q = (
            batch["rewards"] + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        predicted_qs = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=rng,
            grad_params=params,
        )
        chex.assert_shape(predicted_qs, (self.config["critic_ensemble_size"], batch_size))

        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])

        critic_target = batch["mc_returns"] if "mc_returns" in batch else target_q
        critic_target_qs = critic_target[None].repeat(
            self.config["critic_ensemble_size"], axis=0
        )
        if self.config["use_expectile"]:
            warmup_critic_loss = expectile_regression_loss(
                predicted_qs,
                critic_target_qs,
                self.config["expectile"],
            )
        else:
            warmup_critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        td_critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
        critic_loss = jax.lax.cond(
            self.state.step < self.config["critic_warmup_steps"],
            lambda: warmup_critic_loss,
            lambda: td_critic_loss,
        )

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"],
            rng=policy_rng,
            grad_params=params,
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)
        actor_regression_loss = jnp.mean(actions**2)
        actor_loss = jax.lax.cond(
            self.state.step < self.config["critic_warmup_steps"],
            lambda: actor_regression_loss,
            lambda: actor_loss,
        )

        info = {
            "actor_loss": actor_loss,
            "alpha": temperature,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            **_policy_sigma_info(action_distributions),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        _, next_actions_log_probs = self._compute_next_actions(
            batch,
            next_action_sample_key,
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    def loss_fns(self, batch, **kwargs):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"}),
        **kwargs,
    ) -> Tuple["SACMiniAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if self.config.get("augmentation_function") is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        loss_fns = self.loss_fns(batch, **kwargs)
        assert networks_to_update.issubset(loss_fns.keys()), (
            f"Invalid gradient steps: {networks_to_update}"
        )
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns,
            pmap_axis=pmap_axis,
            has_aux=True,
        )

        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    def bc_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """Behavior-cloning loss: NLL of demo actions under the tanh-squashed policy.

        Used for offline imitation. Only the actor (and its trainable encoder
        projection) receive gradients; critic/temperature are untouched. The
        checkpoint structure is identical to a SAC run, so the frozen forge
        inference path (sample_actions -> forward_policy -> actor) restores it
        byte-compatibly.
        """
        rng, policy_rng = jax.random.split(rng)
        action_distributions = self.forward_policy(
            batch["observations"],
            rng=policy_rng,
            grad_params=params,
        )
        # Clip targets off the tanh saturation boundary so log_prob is finite.
        target_actions = jnp.clip(batch["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
        log_probs = action_distributions.log_prob(target_actions)
        bc_loss = -jnp.mean(log_probs)
        mode_actions = action_distributions.mode()
        bc_mse = jnp.mean((mode_actions - batch["actions"]) ** 2)
        info = {
            "bc_loss": bc_loss,
            "bc_log_prob": jnp.mean(log_probs),
            "bc_mse": bc_mse,
            **_policy_sigma_info(action_distributions),
        }
        return bc_loss, info

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_bc(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACMiniAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if self.config.get("augmentation_function") is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        loss_fns = {
            "actor": partial(self.bc_loss_fn, batch),
            "critic": lambda params, rng: (0.0, {}),
            "temperature": lambda params, rng: (0.0, {}),
        }
        new_state, info = self.state.apply_loss_fns(
            loss_fns,
            pmap_axis=pmap_axis,
            has_aux=True,
        )
        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    def neg_bc_loss_fn(self, pos_batch, neg_batch, params: Params, rng: PRNGKey):
        """Contrastive BC: imitate success (pos) actions, push the policy away from
        failure (neg) actions. Negative term is a hinge on log-prob so it only
        penalizes failure actions the policy currently finds *likely* (>margin),
        which keeps the success-imitation backbone intact (regression-safe)."""
        neg_weight = float(self.config.get("neg_bc_weight", 0.3))
        neg_margin = float(self.config.get("neg_bc_margin", 0.0))
        rng, r_pos, r_neg = jax.random.split(rng, 3)

        pos_dist = self.forward_policy(pos_batch["observations"], rng=r_pos, grad_params=params)
        pos_act = jnp.clip(pos_batch["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
        pos_logp = pos_dist.log_prob(pos_act)
        pos_loss = -jnp.mean(pos_logp)

        neg_dist = self.forward_policy(neg_batch["observations"], rng=r_neg, grad_params=params)
        neg_act = jnp.clip(neg_batch["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
        neg_logp = neg_dist.log_prob(neg_act)
        neg_pen = jnp.mean(jax.nn.relu(neg_logp - neg_margin))

        loss = pos_loss + neg_weight * neg_pen
        mode_actions = pos_dist.mode()
        info = {
            "bc_loss": pos_loss,
            "neg_pen": neg_pen,
            "neg_logp": jnp.mean(neg_logp),
            "pos_logp": jnp.mean(pos_logp),
            "bc_mse": jnp.mean((mode_actions - pos_batch["actions"]) ** 2),
            **_policy_sigma_info(pos_dist),
        }
        return loss, info

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_neg_bc(
        self,
        pos_batch: Batch,
        neg_batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACMiniAgent", dict]:
        if self.config["image_keys"][0] not in pos_batch["next_observations"]:
            pos_batch = _unpack(pos_batch)
        if self.config["image_keys"][0] not in neg_batch["next_observations"]:
            neg_batch = _unpack(neg_batch)
        rng, aug_rng, aug_rng2 = jax.random.split(self.state.rng, 3)
        if self.config.get("augmentation_function") is not None:
            pos_batch = self.config["augmentation_function"](pos_batch, aug_rng)
            neg_batch = self.config["augmentation_function"](neg_batch, aug_rng2)

        loss_fns = {
            "actor": partial(self.neg_bc_loss_fn, pos_batch, neg_batch),
            "critic": lambda params, rng: (0.0, {}),
            "temperature": lambda params, rng: (0.0, {}),
        }
        new_state, info = self.state.apply_loss_fns(
            loss_fns,
            pmap_axis=pmap_axis,
            has_aux=True,
        )
        new_state = new_state.replace(rng=rng)
        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            return dist.mode()
        return dist.sample(seed=seed)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        critic_warmup_steps: int = 0,
        use_expectile: bool = False,
        expectile: float = 0.7,
        max_target_backup: bool = False,
        on_the_fly: int = 0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        model_def = ModuleDict(networks)

        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                use_expectile=use_expectile,
                expectile=expectile,
                max_target_backup=max_target_backup,
                critic_warmup_steps=critic_warmup_steps,
                on_the_fly=on_the_fly,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        critic_warmup_steps: int = 0,
        use_expectile: bool = False,
        expectile: float = 0.7,
        reward_bias: float = 0.0,
        max_target_backup: bool = False,
        on_the_fly: int = 0,
        freeze_visual_encoder: bool = True,
        resnet10_ckpt_path: Optional[str] = None,
        resnet18_ckpt_path: Optional[str] = None,
        init_final: Optional[float] = None,
        **kwargs,
    ):
        image_keys = tuple(image_keys)
        policy_network_kwargs = dict(policy_network_kwargs)
        critic_network_kwargs = dict(critic_network_kwargs)
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type in ("resnet", "resnet-pretrained"):
            print(
                f"WARNING: train.encoder_type={encoder_type} is deprecated; "
                "use train.encoder_type=resnet10."
            )
            encoder_type = "resnet10"

        if encoder_type in ("resnet10", "resnet18"):
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            depth = 10 if encoder_type == "resnet10" else 18
            pretrained_encoder = resnetv1_configs[f"resnetv1-{depth}-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    freeze_backbone=freeze_visual_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "dinov3":
            from serl_launcher.vision.dinov3_flax import (
                PreTrainedDINOv3Encoder,
                dinov3_vits16,
            )

            backbone = dinov3_vits16(
                n_storage_tokens=4,
                layerscale_init=1e-5,
                norm_eps=1e-5,
                mask_k_bias=True,
            )
            encoders = {
                image_key: PreTrainedDINOv3Encoder(
                    backbone=backbone,
                    bottleneck_dim=256,
                    freeze_backbone=freeze_visual_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = Critic(
            encoder=encoder_def,
            network=critic_backbone,
            # Match HIL-SERL: freeze only the pretrained backbone; keep critic
            # visual/proprio projection trainable.
            freeze_encoder=False,
            name="critic",
            init_final=init_final
        )

        policy_def = Policy(
            encoder=encoder_def,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            # Actor-side encoder output is stopped, matching HIL-SERL Policy.
            freeze_encoder=freeze_visual_encoder,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            use_expectile=use_expectile,
            expectile=expectile,
            reward_bias=reward_bias,
            critic_warmup_steps=critic_warmup_steps,
            max_target_backup=max_target_backup,
            on_the_fly=on_the_fly,
            **kwargs,
        )

        if encoder_type == "resnet10":
            if resnet10_ckpt_path is None:
                print(
                    "WARNING: train.encoder_type=resnet10 but no ResNet-10 "
                    "checkpoint path was provided; using random initialization."
                )
            else:
                from serl_launcher.utils.train_utils import load_resnet10_params

                agent = load_resnet10_params(
                    agent, image_keys, ckpt_path=resnet10_ckpt_path
                )
        elif encoder_type == "resnet18":
            if resnet18_ckpt_path is None:
                print(
                    "WARNING: train.encoder_type=resnet18 but no ResNet-18 "
                    "checkpoint path was provided; using random initialization."
                )
            else:
                from serl_launcher.utils.train_utils import load_resnet18_params

                agent = load_resnet18_params(
                    agent, image_keys, ckpt_path=resnet18_ckpt_path
                )

        return agent

