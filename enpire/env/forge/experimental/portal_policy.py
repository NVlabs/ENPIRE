"""
PortalPolicy: Runs policy inference in a separate subprocess using Portal IPC.

This module provides:
- PortalPolicyConfig: Configuration dataclass for the policy subprocess
- PortalPolicyServer: Runs in subprocess, hosts GetActionPolicy with RobotInterface/OmniDiffusionPolicy
- PortalPolicy: Main process client that forwards get_action/reset calls to subprocess
"""

from dataclasses import dataclass
import os
import time
from typing import Any, Literal

import portal

from enpire.env.forge.experimental.key_remapping_utils import (
    _make_arrays_contiguous,
    map_action,
    map_observation,
)
import numpy as np


@dataclass
class PortalPolicyConfig:
    """Serializable configuration for policy subprocess."""

    ckpt_path: str
    embodiment_tag: str  # String value, convert to EmbodimentTag in subprocess
    device: str = "cuda:0"
    use_robot_interface: bool = False
    use_vllm: bool = True
    freq_bins: int = 20
    resolution: Literal[240, 480] = 240
    port: int = 8011
    # torch.compile settings
    use_torch_compile: bool = False
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    # Number of DiT inference steps to perform in forward pass (set to -1 to use default)
    num_inference_steps: int = 8


class PortalPolicyServer:
    """Policy server that runs in a subprocess.

    Hosts the actual model inference (RobotInterface or OmniDiffusionPolicy)
    wrapped with GetActionPolicy, and exposes get_action/reset via Portal RPC.
    """

    def __init__(self, config: PortalPolicyConfig):
        self.config = config
        self._load_policy()
        self._setup_server()

    def _load_policy(self):
        """Load RobotInterface or OmniDiffusionPolicy based on config."""
        # Import here to avoid loading CUDA in main process
        from groot.control.envs.yam.experimental.get_action_policy import (
            GetActionPolicy,
            PolicyAdapters,
        )
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.omni.inference.robot_interface import RobotInterface
        from groot.vla.omni.model.base.sim_policy import OmniDiffusionPolicy

        self.embodiment_tag = EmbodimentTag(self.config.embodiment_tag)
        resolution = self.config.resolution

        print(f"[PortalPolicyServer] Loading policy from {self.config.ckpt_path}...")
        print(f"[PortalPolicyServer] use_robot_interface={self.config.use_robot_interface}")

        if self.config.use_robot_interface:
            robot_interface = RobotInterface(
                checkpoint_dir=self.config.ckpt_path,
                embodiment_tag=self.embodiment_tag,
                device=self.config.device,
                freq_bins=self.config.freq_bins,
                use_vllm=self.config.use_vllm,
            )
        else:
            robot_interface = OmniDiffusionPolicy(
                model_path=self.config.ckpt_path,
                embodiment_tag=self.embodiment_tag,
                device=self.config.device,
            )

        if self.config.num_inference_steps is not None and self.config.num_inference_steps > 0:
            old_steps = robot_interface.model.config.num_inference_timesteps
            robot_interface.model.config.num_inference_timesteps = self.config.num_inference_steps
            print(
                "[PortalPolicyServer] Patching policy num_inference_steps to",
                f"{self.config.num_inference_steps} (was {old_steps})",
            )

        # Apply torch.compile to DiT action head if enabled
        if self.config.use_torch_compile:
            self._apply_torch_compile(robot_interface)

        # Create adapters with map_observation and map_action
        adapters = PolicyAdapters(
            map_observation=lambda obs: map_observation(obs, self.embodiment_tag, resolution),
            map_action=lambda action: map_action(action, self.embodiment_tag),
        )

        self.policy = GetActionPolicy(policy=robot_interface, adapters=adapters)
        print("[PortalPolicyServer] Policy loaded successfully")

    def _apply_torch_compile(self, robot_interface):
        """Apply torch.compile to the DiT action head."""
        import torch

        mode = self.config.torch_compile_mode
        print(f"[PortalPolicyServer] Applying torch.compile (mode={mode})...")

        # Find the model - could be robot_interface.model or nested deeper
        model = None
        if hasattr(robot_interface, "model"):
            model = robot_interface.model
        elif hasattr(robot_interface, "policy") and hasattr(robot_interface.policy, "model"):
            model = robot_interface.policy.model

        if model is not None and hasattr(model, "action_head"):
            try:
                print("[PortalPolicyServer] Compiling DiT action head...")
                model.action_head.get_action = torch.compile(
                    model.action_head.get_action,
                    mode=mode,
                    fullgraph=False,
                    dynamic=True,
                )
                print("[PortalPolicyServer] torch.compile applied to DiT action head.")
            except Exception as e:
                print(f"[PortalPolicyServer] WARNING: torch.compile failed: {e}")
                print("[PortalPolicyServer] Continuing without compilation...")
        else:
            print("[PortalPolicyServer] WARNING: Could not find model.action_head to compile.")

    def _setup_server(self):
        """Setup Portal server with bound methods."""
        self._server = portal.Server(self.config.port)
        self._server.bind("get_action", self._handle_get_action)
        self._server.bind("reset", self._handle_reset)
        self._server.bind("health_check", self._handle_health_check)

    def _handle_get_action(self, observation: dict) -> tuple[dict, dict]:
        """Portal RPC handler for get_action."""
        action, info = self.policy.get_action(observation)
        # Ensure arrays are contiguous for serialization
        return _make_arrays_contiguous(action), _make_arrays_contiguous(info)

    def _handle_reset(self) -> dict | None:
        """Portal RPC handler for reset."""
        result = self.policy.reset()
        return _make_arrays_contiguous(result) if result else None

    def _handle_health_check(self) -> bool:
        """Health check endpoint."""
        return True

    def serve(self):
        """Start the server (blocking)."""
        print(f"[PortalPolicyServer] Starting server on port {self.config.port}")
        self._server.start()


def _run_policy_server(config: PortalPolicyConfig):
    """Entry point for policy subprocess."""
    # Set environment variables for subprocess
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["HF_HUB_OFFLINE"] = "1"

    server = PortalPolicyServer(config)
    server.serve()


class PortalPolicy:
    """Policy wrapper that runs inference in a separate subprocess via Portal IPC.

    Implements the standard Policy interface, forwarding calls to the subprocess.
    """

    def __init__(
        self,
        config: PortalPolicyConfig,
        startup_timeout: float = 120.0,
    ):
        """
        Args:
            config: Configuration for the policy subprocess.
            startup_timeout: Maximum time to wait for subprocess to become ready.
                Model loading can take 60-120s, so default is generous.
        """
        self.config = config
        self._client: portal.Client | None = None
        self._process = None

        # Start subprocess
        self._start_subprocess()

        # Wait for server to be ready
        self._wait_for_ready(startup_timeout)

    def _start_subprocess(self):
        """Start the policy server in a subprocess."""
        config = self.config

        def _run_server():
            _run_policy_server(config)

        self._process = portal.Process(_run_server, start=True)
        print(f"[PortalPolicy] Started subprocess on port {self.config.port}")

    def _wait_for_ready(self, timeout: float):
        """Wait for subprocess to be ready."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                self._client = portal.Client(f"localhost:{self.config.port}")
                if self._client.health_check().result(timeout=5.0):
                    print(f"[PortalPolicy] Subprocess ready after {time.time() - start:.1f}s")
                    return
            except Exception:
                time.sleep(1.0)
                continue

        raise TimeoutError(
            f"[PortalPolicy] Subprocess not ready after {timeout}s. "
            "Model loading may have failed."
        )

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Forward get_action to subprocess."""
        # Ensure arrays are contiguous before sending
        obs_clean = _make_arrays_contiguous(observation)
        future = self._client.get_action(obs_clean)
        action, info = future.result()
        return action, info

    def reset(self) -> dict[str, Any] | None:
        """Forward reset to subprocess."""
        future = self._client.reset()
        return future.result()

    def shutdown(self):
        """Shutdown the subprocess."""
        # Portal.Process uses daemon processes, cleanup happens automatically
        # when the main process exits
        pass
