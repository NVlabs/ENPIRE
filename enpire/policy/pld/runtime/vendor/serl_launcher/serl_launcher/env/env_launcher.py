# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment Launcher for SERL with Hydra Configuration Support

This module provides a centralized way to create environments using Hydra configurations,
eliminating the need for redundant config classes.
"""

import inspect
import os
from typing import Callable

from omegaconf import DictConfig, OmegaConf
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.utils import DoneWrapper

_REMOTE_IMAGE_KEYS = (
    "top_camera_image",
    "left_camera_image",
    "right_camera_image",
    "left_wrist_camera_image",
    "wrist_camera_image",
)


class EnvironmentLauncher:
    """
    Centralized environment launcher that works with Hydra configurations.

    This class eliminates the need for multiple TrainConfig classes by providing
    a single interface to create environments from Hydra DictConfig objects.
    """

    def __init__(self, env_config: DictConfig):
        """
        Initialize the environment launcher with configuration.

        Args:
            env_config: Hydra DictConfig containing environment parameters
        """
        self.config = env_config

        # Validate required fields (relaxed for remote_deployment)
        if "remote_deployment" not in str(env_config.get("teleop_wrapper_path", "")):
            if not hasattr(env_config, "env_name") or env_config.env_name == "":
                raise ValueError("Environment name is not set in config")

    @staticmethod
    def _resolve_image_keys(
        image_keys, image_filter, *, allow_remote_image_keys: bool = False
    ) -> tuple[str, ...]:
        configured_keys = tuple(str(key) for key in image_keys)
        allowed_keys = configured_keys
        if allow_remote_image_keys:
            allowed_keys = tuple(dict.fromkeys((*configured_keys, *_REMOTE_IMAGE_KEYS)))
        if not image_filter:
            return configured_keys
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

    def create_environment(
        self,
        fake_env: bool = False,
        classifier: bool = False,
        render_mode: str = "rgb_array",
        action_scaling_fn: Callable = None,
        task_cfg: DictConfig = None,
        wrap_chunking: bool = True,
    ):
        """
        Create and configure the environment based on the provided config.

        Args:
            fake_env: If True, skip spacemouse intervention (for learner)
            classifier: Whether to use reward classifier (not supported yet)
            render_mode: Rendering mode for the environment
            use_spacemouse: Whether to use spacemouse for intervention

        Returns:
            Configured environment ready for training/evaluation
        """

        task_suite = None
        image_size = (
            int(self.config.get("image_height", self.config.get("resolution", 256))),
            int(self.config.get("image_width", self.config.get("resolution", 256))),
        )
        active_image_keys = self._resolve_image_keys(
            self.config.get("image_keys", ["top_camera_image", "left_camera_image"]),
            self.config.get("image_filter", {}),
            allow_remote_image_keys=(
                "remote_deployment" in str(self.config.get("teleop_wrapper_path", ""))
            ),
        )
        # Create base environment
        if "remote_deployment" in self.config.teleop_wrapper_path:
            from serl_launcher.env.remote_deployment_env import RemoteDeploymentEnv

            host = self.config.get("remote_host", "0.0.0.0")
            port = self.config.get("remote_port", 8964)
            image_keys = list(active_image_keys)
            proprio_keys = list(
                self.config.get(
                    "proprio_keys",
                    [
                        "left_joint_pos",
                        "left_gripper_pos",
                        "right_joint_pos",
                        "right_gripper_pos",
                    ],
                )
            )
            instruction = self.config.get("instruction", "") or ""
            proprio_filter = OmegaConf.to_container(
                self.config.get("proprio_filter", {}), resolve=True
            )
            action_dim = self.config.get("remote_action_dim", 14)
            if "action_repr" not in self.config:
                raise ValueError(
                    "remote_deployment requires explicit env.action_repr "
                    "('joint', 'delta_eef_quat', 'delta_eef_rot6d', or 'delta_eef_pos'); "
                    "refusing to silently assume joint actions."
                )
            action_repr = str(self.config.action_repr)
            action_exec_horizon = self.config.get("remote_action_exec_horizon", 20)

            # Override action_dim and proprio_keys based on control_mode
            control_mode = self.config.get("control_mode", "both")
            if control_mode in ("left", "right"):
                if action_repr == "joint":
                    action_dim = 7
                    proprio_keys = [
                        f"{control_mode}_joint_pos",
                        f"{control_mode}_gripper_pos",
                    ]
                elif action_repr == "delta_eef_quat":
                    action_dim = 8
                elif action_repr == "delta_eef_rot6d":
                    action_dim = 10
                elif action_repr == "delta_eef_pos":
                    action_dim = 3
            image_h = self.config.get("remote_image_height", 480)
            image_w = self.config.get("remote_image_width", 640)

            if fake_env:
                # Learner doesn't need a Portal server — use a lightweight placeholder
                # that only exposes the correct observation/action spaces.
                env = RemoteDeploymentEnv._make_fake(
                    image_keys=image_keys,
                    proprio_keys=proprio_keys,
                    action_dim=action_dim,
                    action_exec_horizon=action_exec_horizon,
                    image_size=(image_h, image_w),
                    instruction=instruction,
                    control_mode=control_mode,
                    action_repr=action_repr,
                    host=host,
                    proprio_filter=proprio_filter,
                )
            else:
                env = RemoteDeploymentEnv(
                    host=host,
                    port=port,
                    image_keys=image_keys,
                    proprio_keys=proprio_keys,
                    action_dim=action_dim,
                    action_exec_horizon=action_exec_horizon,
                    image_size=(image_h, image_w),
                    instruction=instruction,
                    step_timeout=self.config.get("remote_step_timeout"),
                    control_mode=control_mode,
                    action_repr=action_repr,
                    proprio_filter=proprio_filter,
                )

        elif "simpler_env" in self.config.teleop_wrapper_path:
            import simpler_env
            from mani_skill2_real2sim.agents.controllers.controller_config_store import (
                controller_config_store,
            )

            controller_config_store.set_config(
                OmegaConf.to_container(self.config.controller_config, resolve=True)
            )
            print(
                f"[EnvironmentLauncher] Set controller config: {controller_config_store.get_config()}"
            )

            env = simpler_env.make(self.config.env_name, render_mode=render_mode)

        elif "yam_env" in self.config.teleop_wrapper_path:
            import yam_env

            max_steps = self.config.get("max_episode_length", 100)
            env = yam_env.make(
                self.config.env_name,
                control_mode="cartesian_position",
                max_episode_steps=max_steps,
            )
            from yam_env.utils.env.wrapper import ControlIntegrator

            fixed_arm = self.config.get("fixed_arm_mapping", None)
            env = ControlIntegrator(env, fixed_arm_mapping=fixed_arm)

            # Only add PhoneIntervention for actor (teleop); learner uses fake env
            if not fake_env:
                use_device = OmegaConf.select(
                    self.config, "teleop.abs_eef.use_device", default=False
                )
                from yam_env.utils.env.wrapper import PhoneIntervention

                env = PhoneIntervention(
                    env,
                    use_device=use_device,
                    env_name=self.config.env_name,
                    action_scaling_fn=action_scaling_fn,
                    fixed_arm_mapping=fixed_arm,
                )

        elif "libero" in self.config.teleop_wrapper_path:
            import libero_env

            env = libero_env.make(
                self.config.env_name,
                task_idx=task_cfg.task_idx,
                render_mode=render_mode,
                resolution=self.config.resolution,
                max_steps=task_cfg.max_episode_steps,
            )
        elif "mimicgen" in self.config.teleop_wrapper_path:
            import mimicgen_env

            env = mimicgen_env.make(
                self.config.env_name,
                task_idx=task_cfg.task_idx,
                robot_name=task_cfg.robot_name,
                render_mode=render_mode,
                resolution=self.config.resolution,
                max_steps=task_cfg.max_episode_steps,
            )
        elif "franka_env" in self.config.teleop_wrapper_path:
            import franka_env

            env = franka_env.make(
                self.config.env_name,
                env_cfg=self.config,
                task_cfg=task_cfg,
                fake_env=fake_env,
            )

        # Add spacemouse intervention if not fake environment (skip for remote_deployment)
        if not fake_env and "remote_deployment" not in self.config.teleop_wrapper_path:
            if (
                self.config.teleop_wrapper_path
                == "simpler_env.utils.env.wrapper.SpacemouseIntervention"
            ):
                from simpler_env.utils.env.wrapper import (
                    SpacemouseIntervention as TeleopWrapper,
                )
            elif (
                self.config.teleop_wrapper_path
                == "yam_env.utils.env.wrapper.GelloIntervention"
            ):
                from yam_env.utils.env.wrapper import GelloIntervention as TeleopWrapper
            elif (
                self.config.teleop_wrapper_path
                == "yam_env.utils.env.wrapper.SpacemouseIntervention"
            ):
                from yam_env.utils.env.wrapper import SpacemouseIntervention as TeleopWrapper
            elif (
                self.config.teleop_wrapper_path
                == "libero_env.utils.env.wrapper.SpacemouseIntervention"
            ):
                from libero_env.utils.env.wrapper import (
                    SpacemouseIntervention as TeleopWrapper,
                )
            elif (
                self.config.teleop_wrapper_path
                == "mimicgen_env.utils.env.wrapper.SpacemouseIntervention"
            ):
                from mimicgen_env.utils.env.wrapper import (
                    SpacemouseIntervention as TeleopWrapper,
                )
            elif (
                self.config.teleop_wrapper_path
                == "franka_env.utils.env.wrapper.SpacemouseIntervention"
            ):
                from franka_env.utils.env.wrapper import (
                    SpacemouseIntervention as TeleopWrapper,
                )
            elif (
                self.config.teleop_wrapper_path
                == "mimicgen_env.utils.env.wrapper.PhoneIntervention"
            ):
                from mimicgen_env.utils.env.wrapper import PhoneIntervention as TeleopWrapper
            elif (
                self.config.teleop_wrapper_path
                == "yam_env.utils.env.wrapper.SpacemouseIntervention"
            ):
                from yam_env.utils.env.wrapper import SpacemouseIntervention as TeleopWrapper
            else:
                raise ValueError(
                    f"Teleop wrapper {self.config.teleop_wrapper_path} not supported"
                )

            use_device = OmegaConf.select(
                self.config, "teleop.delta_eef.use_device", default=False
            )
            env = TeleopWrapper(
                env,
                use_device=use_device,
                env_name=self.config.env_name,
                action_scaling_fn=action_scaling_fn,
            )

        # Add observation wrapper
        if "remote_deployment" in self.config.teleop_wrapper_path:
            from yam_env.utils.env.observation_utils import SERLObsWrapper
        elif "simpler_env" in self.config.teleop_wrapper_path:
            from simpler_env.utils.env.observation_utils import SERLObsWrapper
        elif "yam_env" in self.config.teleop_wrapper_path:
            from yam_env.utils.env.observation_utils import SERLObsWrapper
        elif "libero_env" in self.config.teleop_wrapper_path:
            from libero_env.utils.env.observation_utils import SERLObsWrapper
        elif "franka_env" in self.config.teleop_wrapper_path:
            from franka_env.utils.env.observation_utils import SERLObsWrapper
        elif "mimicgen_env" in self.config.teleop_wrapper_path:
            from mimicgen_env.utils.env.observation_utils import SERLObsWrapper
        else:
            raise ValueError(
                f"Observation wrapper for {self.config.teleop_wrapper_path} not supported"
            )

        # For remote_deployment, use the resolved proprio_keys (respects control_mode);
        # for other envs, use the raw config keys.
        _proprio_keys = self.config["proprio_keys"]
        if "remote_deployment" in self.config.teleop_wrapper_path:
            _cm = self.config.get("control_mode", "both")
            _repr = str(self.config.get("action_repr", "joint"))
            if _repr == "joint" and _cm == "left":
                _proprio_keys = ["left_joint_pos", "left_gripper_pos"]
            elif _repr == "joint" and _cm == "right":
                _proprio_keys = ["right_joint_pos", "right_gripper_pos"]
        wrapper_kwargs = {
            "proprio_keys": _proprio_keys,
            "image_keys": list(active_image_keys),
        }
        if "image_size" in inspect.signature(SERLObsWrapper).parameters:
            wrapper_kwargs["image_size"] = image_size
        env = SERLObsWrapper(env, **wrapper_kwargs)

        if wrap_chunking:
            # Add chunking wrapper
            env = ChunkingWrapper(
                env, obs_horizon=self.config.get("obs_horizon", 1), act_exec_horizon=None
            )

        if classifier:
            import jax
            import jax.numpy as jnp
            from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper
            from serl_launcher.networks.reward_classifier import load_classifier_func

            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.config.classifier_keys,
                checkpoint_path=os.path.abspath(task_cfg.classifier_ckpt_path),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # print("classifier sigmoid output", sigmoid(classifier(obs)[0]))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)[0]) > 0.9)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        env = DoneWrapper(env)
        return env


def create_env_launcher(env_config: DictConfig) -> EnvironmentLauncher:
    """
    Factory function to create an EnvironmentLauncher from Hydra config.

    Args:
        env_config: Hydra DictConfig containing environment parameters

    Returns:
        Configured EnvironmentLauncher instance
    """
    return EnvironmentLauncher(env_config)

