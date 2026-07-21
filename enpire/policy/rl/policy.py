import time

import numpy as np

from enpire.env.forge.experimental.get_action_policy import PolicyAdapters
from enpire.env.forge.experimental.key_remapping_utils import map_action, map_observation
from enpire.env.forge.experimental.rl_interface import RLInterface
from enpire.env.forge.robot.fello.fello_teleop_policy import _IDENTITY_ROT6D, DualFelloPolicy
from enpire.env.forge.robot.spacemouse.spacemouse_policy import SpaceMouseTeleopPolicy
from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.events import Z_UP_KEY, RIGHT_TAKEOVER_BUTTON


def _gripper_from_obs(obs: dict, side: str) -> np.ndarray:
    val = obs.get(f"{side}_gripper_pos")
    if val is None:
        return np.array([1.0], dtype=np.float32)
    return np.asarray(val, dtype=np.float32).reshape(-1)[:1].copy()


def _stub_fello_action(obs: dict) -> dict:
    return {
        "left_ee_pos": np.zeros(3, dtype=np.float32),
        "left_ee_rot6d": _IDENTITY_ROT6D.copy(),
        "left_gripper_pos": _gripper_from_obs(obs, "left"),
        "right_ee_pos": np.zeros(3, dtype=np.float32),
        "right_ee_rot6d": _IDENTITY_ROT6D.copy(),
        "right_gripper_pos": _gripper_from_obs(obs, "right"),
    }


def _stub_fello_info() -> dict:
    return {
        "source": "human",
        "left_buttons": np.zeros(3, dtype=np.float32),
        "right_buttons": np.zeros(3, dtype=np.float32),
    }


def build_policy_adapters(cfg: DataCollectionConfig) -> PolicyAdapters:
    return PolicyAdapters(
        map_observation=lambda obs: map_observation(
            obs, cfg.embodiment_tag, cfg.resolution
        ),
        map_action=lambda action: map_action(action, cfg.embodiment_tag),
    )


class PolicyRouter:
    def __init__(
        self,
        fello_policy: DualFelloPolicy | None,
        rl_policy: RLInterface,
        enabled_sides: str,
        z_up_step_m: float,
        demo_collection: bool = False,
        spacemouse_policy: SpaceMouseTeleopPolicy | None = None,
    ):
        self.fello_policy = fello_policy
        self.spacemouse_policy = spacemouse_policy
        self.rl_policy = rl_policy
        self.enabled_sides = (
            ("left", "right") if enabled_sides == "both" else (enabled_sides,)
        )
        self._z_up_step_m = float(z_up_step_m)
        self.demo_collection = demo_collection
        self.last_rl_info: dict = {}
        self._spacemouse_cached_action: dict | None = None
        self._spacemouse_cached_info: dict = {}

    def get_fello_action(self, obs) -> tuple[dict, dict, int]:
        t0 = time.perf_counter()
        if self.fello_policy is None:
            fello_action = _stub_fello_action(obs)
            fello_info = _stub_fello_info()
        else:
            fello_action, fello_info = self.fello_policy.get_action(obs)
        fello_action["__action_t"] = time.time()
        return fello_action, fello_info, int((time.perf_counter() - t0) * 1000)

    def refresh_spacemouse_action(self, obs) -> tuple[dict | None, dict]:
        if self.spacemouse_policy is None:
            self._spacemouse_cached_action = None
            self._spacemouse_cached_info = {}
            return None, {}
        action, info = self.spacemouse_policy.get_action(obs)
        self._spacemouse_cached_action = action
        self._spacemouse_cached_info = info
        return action, info

    def _get_spacemouse_action(self, obs) -> tuple[dict | None, dict]:
        if self.spacemouse_policy is None:
            return None, {}
        if self._spacemouse_cached_action is None:
            return self.refresh_spacemouse_action(obs)
        return self._spacemouse_cached_action, self._spacemouse_cached_info

    def route_action(
        self, obs, fello_action, fello_info, *, use_rl: bool, keyboard_info
    ) -> tuple[dict, int]:
        t0 = time.perf_counter()
        spacemouse_action, spacemouse_info = self._get_spacemouse_action(obs)
        spacemouse_active = bool(spacemouse_info.get("active", False))

        if keyboard_info[Z_UP_KEY]:
            self.last_rl_info = {}
            action_in_effect = dict(fello_action)
            for side in self.enabled_sides:
                ee_pos = np.asarray(
                    action_in_effect[f"{side}_ee_pos"], dtype=np.float32
                ).copy()
                ee_pos[:] = [0, 0, self._z_up_step_m]
                action_in_effect[f"{side}_ee_pos"] = ee_pos
            action_source = "human"
            print()
        elif spacemouse_action is not None and (
            spacemouse_active or self.demo_collection
        ):
            self.last_rl_info = {
                "spacemouse_active": spacemouse_active,
                "spacemouse_motion_norm": spacemouse_info.get("motion_norm", 0.0),
                "spacemouse_control_side": spacemouse_info.get("control_side"),
                "spacemouse_buttons": spacemouse_info.get("buttons"),
            }
            action_in_effect = spacemouse_action
            action_source = "human"
        elif use_rl and not fello_info["right_buttons"][RIGHT_TAKEOVER_BUTTON]:
            if self.demo_collection:
                self.last_rl_info = {}
                action_in_effect = _stub_fello_action(obs)
                action_source = "human"
            else:
                rl_action, rl_info = self.rl_policy.get_action(obs)
                self.last_rl_info = dict(rl_info)
                rl_action["__action_t"] = time.time()
                action_in_effect = rl_action
                action_source = rl_info.get("action_source", "rl")
        else:
            self.last_rl_info = {}
            action_in_effect = fello_action
            action_source = "human"

        action = dict(action_in_effect)
        action["source"] = action_source
        return action, int((time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        if self.spacemouse_policy is not None:
            self.spacemouse_policy.close()

