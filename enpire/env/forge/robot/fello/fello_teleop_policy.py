import argparse
import sys
import time
from typing import Literal, Sequence

import gymnasium as gym
import numpy as np
import portal
from scipy.spatial.transform import Rotation
import yaml

from enpire.env.forge.robot.constants import (
    LEFT_LEADER_PORT,
    RIGHT_LEADER_PORT,
)
from enpire.env.forge.robot.fello.fello_config import (
    FELLO_CONFIG_PATH,
    get_config_value,
    load_fello_config,
)
from enpire.env.forge.robot.fello.fello_kinematics import FelloKinematics
from enpire.policy.legacy import Action, Info, Observation, Options, Policy

_MAPPING_PATH = FELLO_CONFIG_PATH
_BUTTON_THRESHOLD = 0.5
_ACTION_TYPES = ("joint", "delta_eef")
_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_FELLO_RPY_DELTA_MAX = np.full(3, np.pi, dtype=np.float32)
_FELLO_DELTA_TO_YAM_DELTA_FRAME = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _quat_xyzw_to_rot6d(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return _IDENTITY_ROT6D.copy()
    mat = Rotation.from_quat(q / norm).as_matrix()
    return np.concatenate([mat[:, 0], mat[:, 1]], axis=0).astype(np.float32)


class FelloLeaderClient:
    """Portal client for the Fello leader arm."""

    def __init__(self, host: str, port: int | None = None):
        if port is None:
            cfg = load_fello_config()
            port = get_config_value(cfg, "server", "port")
        if port is None:
            raise ValueError("Missing server.port in fello_config.yaml")
        self._client = portal.Client(f"{host}:{port}")

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        return self._client.get_info().result()

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state).result()

    def command_joint_pos(self, joint_state: dict[str, np.ndarray]) -> None:
        self._client.command_joint_pos(joint_state).result()

    def set_mode(self, mode: str) -> None:
        self._client.set_mode(mode).result()


class FelloTeleopPolicy(Policy):
    """Teleoperation policy that maps a 7-DOF Fello arm to one YAM station arm."""

    def __init__(
        self,
        target_side: Literal["left", "right"],
        action_type: Literal["joint", "delta_eef"],
        scaled_control: bool | None,
        scaled_control_xyz_scale: Sequence[
            float
        ],  # [x, y, z] EE-position delta scale while scaled control is active
        delta_ee_translation_xyz_max: Sequence[
            float
        ],  # [x, y, z] per-step EE translation clip (m)
        decouple_translation: bool,  # true then only the lowest 3 joint contribute to translation delta
        use_footswitch: bool | None = True,
        takeover_button: int | None = 0,
    ):
        cfg = load_fello_config(side=target_side)
        teleop_cfg = get_config_value(cfg, "teleop", default={}) or {}
        control_cfg = get_config_value(cfg, "control", default={}) or {}
        scaled_control_cfg = get_config_value(cfg, "scaled_control", default={}) or {}
        host = teleop_cfg.get("host")
        if target_side not in ("left", "right"):
            raise ValueError(
                f"target_side must be 'left' or 'right', got {target_side}"
            )
        if action_type not in _ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {_ACTION_TYPES}, got {action_type!r}"
            )

        port = LEFT_LEADER_PORT if target_side == "left" else RIGHT_LEADER_PORT
        self._target_side = target_side
        self._action_type = action_type
        self._scaled_control = bool(scaled_control)
        self._decouple_translation_fk = decouple_translation
        self._takeover_button = takeover_button
        self._max_delta_per_timestep = self._parse_max_delta_per_timestep(
            delta_ee_translation_xyz_max
        )
        self._scaled_control_xyz_scale = self._parse_scaled_control_scale(
            scaled_control_xyz_scale,
            "scaled_control_xyz_scale",
        )
        self._scaled_control_rpy_scale = self._parse_scaled_control_scale(
            scaled_control_cfg.get("rpy_scale", [1.0, 1.0, 1.0]),
            "scaled_control.rpy_scale",
        )
        self._scaled_control_active_button = self._parse_button_index(
            scaled_control_cfg.get("active_button", 1),
            "scaled_control.active_button",
        )
        self._scaled_control_nominal_position = (
            self._parse_nominal_position(
                scaled_control_cfg.get("nominal_position"),
                "scaled_control.nominal_position",
            )
            if self._scaled_control
            else np.zeros(7, dtype=np.float32)
        )
        self._leader = FelloLeaderClient(host=host, port=port)
        self._kinematics = FelloKinematics(self._target_side)
        self._last_delta_eef_sample: dict[str, np.ndarray] | None = None
        self._last_gripper_print = 0.0
        self._last_footswitch_print = 0.0
        self._last_align_send = 0.0
        if teleop_cfg.get("print_interval_s") is None:
            raise ValueError("Missing teleop.print_interval_s in fello_config.yaml")
        if teleop_cfg.get("align_interval_s") is None:
            raise ValueError("Missing teleop.align_interval_s in fello_config.yaml")
        if teleop_cfg.get("align_lpf_alpha") is None:
            raise ValueError("Missing teleop.align_lpf_alpha in fello_config.yaml")
        self._print_interval_s = float(teleop_cfg.get("print_interval_s"))
        self._print_alignment = bool(teleop_cfg.get("print_alignment", True))
        self._align_interval_s = float(teleop_cfg.get("align_interval_s"))
        self._align_filtered_qpos = None
        self.clipped_qpos = None
        self._last_mode = None
        self._align_lpf_alpha = float(teleop_cfg.get("align_lpf_alpha"))
        self._default_kp = np.asarray(control_cfg.get("kp", []), dtype=np.float32)
        self._default_kd = np.asarray(control_cfg.get("kd", []), dtype=np.float32)
        if self._default_kp.size != 7 or self._default_kd.size != 7:
            raise ValueError(
                "control.kp and control.kd must each have 7 values in fello_config.yaml"
            )
        self._use_footswitch = bool(use_footswitch)
        footswitch_cfg = (
            get_config_value(cfg, "hardware", "footswitch", default={}) or {}
        )
        self._button_mode = footswitch_cfg.get("button_mode", "recording")
        ui_button_map = footswitch_cfg.get("ui_button_map", [0, 1, 2])
        if len(ui_button_map) != 3:
            raise ValueError("hardware.footswitch.ui_button_map must have 3 values")
        self._ui_button_map = tuple(int(i) for i in ui_button_map)
        if sorted(self._ui_button_map) != [0, 1, 2]:
            raise ValueError(
                "hardware.footswitch.ui_button_map must be a permutation of [0, 1, 2]"
            )
        self._external_takeover_pressed = False
        self._prev_save_pressed = False
        self._prev_start_pressed = False
        self._prev_home_pressed = False
        self._load_joint_mapping()

    def _load_joint_mapping(self) -> None:
        if not _MAPPING_PATH.exists():
            raise FileNotFoundError(f"Missing joint mapping file: {_MAPPING_PATH}")
        with _MAPPING_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        mapping = data.get("mapping", {})
        arm_index_map = mapping.get("arm_index_map")
        gripper_index_map = mapping.get("gripper_index_map")
        if arm_index_map is None or gripper_index_map is None:
            raise ValueError("fello_config.yaml missing required mapping keys")
        if len(arm_index_map) != 6:
            raise ValueError("arm_index_map must have length 6")

        self._arm_index_map = np.asarray(arm_index_map, dtype=int)

        self._gripper_index_map = int(gripper_index_map)

    def _map_fello_to_yam(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arm = qpos[self._arm_index_map]
        gripper_val = qpos[self._gripper_index_map]
        gripper = np.array([gripper_val], dtype=np.float32)
        return arm.astype(np.float32), gripper

    def _map_yam_to_fello(self, arm: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        qpos = np.zeros(7, dtype=np.float32)
        qpos[self._arm_index_map] = arm
        gripper_val = float(np.asarray(gripper).reshape(-1)[0])
        qpos[self._gripper_index_map] = gripper_val
        return qpos

    @staticmethod
    def _parse_max_delta_per_timestep(
        xyz_max: Sequence[float],
    ) -> np.ndarray:
        xyz = np.asarray(xyz_max, dtype=np.float32).reshape(-1)
        if xyz.size != 3:
            raise ValueError(
                "delta_ee_translation_xyz_max must have 3 values: [dx, dy, dz]"
            )
        if np.any(xyz < 0.0) or np.any(~np.isfinite(xyz)):
            raise ValueError(
                "delta_ee_translation_xyz_max values must be finite and non-negative"
            )
        return np.concatenate([xyz, _FELLO_RPY_DELTA_MAX]).astype(np.float32)

    @staticmethod
    def _parse_scaled_control_scale(value: Sequence[float], label: str) -> np.ndarray:
        scale = np.asarray(value, dtype=np.float32).reshape(-1)
        if scale.size != 3 or np.any(~np.isfinite(scale)):
            raise ValueError(f"{label} must have 3 finite values")
        return scale

    @staticmethod
    def _parse_button_index(value: int, label: str) -> int:
        button = int(value)
        if button not in (0, 1, 2):
            raise ValueError(f"{label} must be one of 0, 1, or 2")
        return button

    @staticmethod
    def _parse_nominal_position(
        value: Sequence[float] | None,
        label: str,
    ) -> np.ndarray:
        if value is None:
            raise ValueError(f"Missing {label} in fello_config.yaml")
        qpos = np.asarray(value, dtype=np.float32).reshape(-1)
        if qpos.size != 7 or np.any(~np.isfinite(qpos)):
            raise ValueError(f"{label} must have 7 finite values")
        return qpos

    def _joint_position_state(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "pos": np.asarray(qpos, dtype=np.float32).reshape(7),
            "vel": np.zeros(7, dtype=np.float32),
            "kp": self._default_kp.copy(),
            "kd": self._default_kd.copy(),
        }

    def _clip_delta_eef_action(
        self,
        delta_pos: np.ndarray,
        delta_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # delta_ee_translation_xyz_max is shared with YamEnv._translation_mode_mask,
        # which clips in the YAM action frame. Rotate the Fello-frame translation
        # delta into YAM, clip there, and rotate back so the post-_delta_eef_action
        # YAM-frame action lands inside the same box YamEnv enforces.
        max_delta = self._max_delta_per_timestep
        frame = _FELLO_DELTA_TO_YAM_DELTA_FRAME
        yam_delta_pos = frame @ np.asarray(delta_pos, dtype=np.float64).reshape(3)
        clipped_yam_delta_pos = np.clip(
            yam_delta_pos,
            -max_delta[:3].astype(np.float64),
            max_delta[:3].astype(np.float64),
        )
        clipped_delta_pos = (frame.T @ clipped_yam_delta_pos).astype(np.float32)
        delta_rpy = Rotation.from_quat(delta_quat).as_euler("xyz").astype(np.float32)
        clipped_delta_rpy = np.clip(
            delta_rpy,
            -max_delta[3:],
            max_delta[3:],
        ).astype(np.float32)
        clipped_delta_quat = Rotation.from_euler("xyz", clipped_delta_rpy).as_quat()
        return clipped_delta_pos, clipped_delta_quat.astype(np.float32)

    @staticmethod
    def _transform_delta_eef_frame(
        delta_pos: np.ndarray,
        delta_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame = _FELLO_DELTA_TO_YAM_DELTA_FRAME
        transformed_pos = frame @ np.asarray(delta_pos, dtype=np.float64).reshape(3)

        delta_rot = Rotation.from_quat(
            np.asarray(delta_quat, dtype=np.float64).reshape(4)
        )
        transformed_quat = Rotation.from_matrix(
            frame @ delta_rot.as_matrix() @ frame.T
        ).as_quat()
        if transformed_quat[3] < 0.0:
            transformed_quat = -transformed_quat

        return transformed_pos.astype(np.float32), transformed_quat.astype(np.float32)

    def _delta_eef_action(
        self,
        defaults: dict[str, np.ndarray],
        delta_pos: np.ndarray,
        delta_quat: np.ndarray,
        gripper: np.ndarray,
    ) -> Action:
        action = self._zero_delta_eef_action(defaults)

        # Fello FK axes differ from the YAM delta-action frame:
        # target x, y, z = -current y, current x, current z.
        delta_pos, delta_quat = self._transform_delta_eef_frame(delta_pos, delta_quat)

        action[f"{self._target_side}_ee_pos"] = delta_pos
        action[f"{self._target_side}_ee_rot6d"] = _quat_xyzw_to_rot6d(delta_quat)
        action[f"{self._target_side}_gripper_pos"] = gripper
        return action

    def _sample_fello_eef(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kin = self._kinematics
        pos, quat = kin.forward_kinematics(qpos)
        if self._decouple_translation_fk:
            # Translation depends only on joints 1-3; freeze the spherical
            # wrist (joints 4-6) so wrist re-orientation doesn't translate
            # the EEF target. Orientation still uses the full-FK quat above.
            qpos_pos = np.asarray(qpos, dtype=np.float32).copy()
            qpos_pos[3:6] = 0.0
            pos, _ = kin.forward_kinematics(qpos_pos)
        gripper = np.array(
            [float(np.asarray(qpos, dtype=np.float32)[self._gripper_index_map])],
            dtype=np.float32,
        )
        return pos.astype(np.float32), quat.astype(np.float32), gripper

    @staticmethod
    def _zero_delta_eef_action(defaults: dict[str, np.ndarray]) -> Action:
        return {
            "left_ee_pos": np.zeros(3, dtype=np.float32),
            "left_ee_rot6d": _IDENTITY_ROT6D.copy(),
            "left_gripper_pos": defaults["left_gripper_pos"],
            "right_ee_pos": np.zeros(3, dtype=np.float32),
            "right_ee_rot6d": _IDENTITY_ROT6D.copy(),
            "right_gripper_pos": defaults["right_gripper_pos"],
        }

    def _get_joint_action(
        self,
        qpos: np.ndarray,
        observation: Observation | None,
    ) -> Action:
        defaults = self._defaults_from_obs(observation)
        arm, gripper = self._map_fello_to_yam(qpos)
        action: Action = {
            "left_joint_pos": defaults["left_joint_pos"],
            "left_gripper_pos": defaults["left_gripper_pos"],
            "right_joint_pos": defaults["right_joint_pos"],
            "right_gripper_pos": defaults["right_gripper_pos"],
        }
        action[f"{self._target_side}_joint_pos"] = arm
        action[f"{self._target_side}_gripper_pos"] = gripper

        now = time.time()
        if (
            self._print_alignment
            and now - self._last_gripper_print >= self._print_interval_s
        ):
            if self._target_side == "left":
                obs_arm = (
                    observation.get("left_joint_pos", np.zeros(6))
                    if observation is not None
                    else np.zeros(6)
                )
                obs_grip = (
                    observation.get("left_gripper_pos", np.zeros(1))
                    if observation is not None
                    else np.zeros(1)
                )
            else:
                obs_arm = (
                    observation.get("right_joint_pos", np.zeros(6))
                    if observation is not None
                    else np.zeros(6)
                )
                obs_grip = (
                    observation.get("right_gripper_pos", np.zeros(1))
                    if observation is not None
                    else np.zeros(1)
                )
            arm_diff = arm - np.asarray(obs_arm, dtype=np.float32).reshape(6)
            grip_diff = gripper - np.asarray(obs_grip, dtype=np.float32).reshape(1)
            diff_str = ", ".join(f"{d:+.3f}" for d in arm_diff)
            print(
                f"[Fello] joint_diff [{self._target_side}]: [{diff_str}]  gripper_diff: {grip_diff[0]:+.3f}"
            )
            self._last_gripper_print = now

        return action

    def _get_delta_eef_action(
        self,
        qpos: np.ndarray,
        observation: Observation | None,
        *,
        xyz_scale: np.ndarray | None = None,
    ) -> Action:
        defaults = self._defaults_from_obs(observation)
        pos, quat, gripper = self._sample_fello_eef(qpos)
        scale = (
            np.ones(3, dtype=np.float32)
            if xyz_scale is None
            else np.asarray(xyz_scale, dtype=np.float32).reshape(3)
        )
        if self._last_delta_eef_sample is None:
            delta_pos = np.zeros(3, dtype=np.float32)
            delta_quat = _IDENTITY_QUAT_XYZW.copy()
            clipped_pos = pos.copy()
            clipped_quat = quat.copy()
        else:
            prev_pos = self._last_delta_eef_sample["pos"]
            prev_quat = self._last_delta_eef_sample["quat"]
            raw_delta_pos = (pos - prev_pos).astype(np.float32)
            unclipped_delta_pos = (raw_delta_pos * scale).astype(np.float32)
            delta_quat = (
                (Rotation.from_quat(quat) * Rotation.from_quat(prev_quat).inv())
                .as_quat()
                .astype(np.float32)
            )
            delta_pos, delta_quat = self._clip_delta_eef_action(
                unclipped_delta_pos,
                delta_quat,
            )
            if np.allclose(delta_pos, unclipped_delta_pos, rtol=1e-6, atol=1e-8):
                clipped_pos = pos.copy()
            else:
                clipped_raw_delta = raw_delta_pos.copy()
                nonzero_scale = np.abs(scale) > 1e-6
                clipped_raw_delta[nonzero_scale] = (
                    delta_pos[nonzero_scale] / scale[nonzero_scale]
                )
                clipped_pos = (prev_pos + clipped_raw_delta).astype(np.float32)
            clipped_quat = (
                (Rotation.from_quat(delta_quat) * Rotation.from_quat(prev_quat))
                .as_quat()
                .astype(np.float32)
            )

        self._last_delta_eef_sample = {
            "pos": clipped_pos.copy(),
            "quat": clipped_quat.copy(),
        }
        return self._delta_eef_action(defaults, delta_pos, delta_quat, gripper)

    def _get_scaled_delta_eef_action(
        self,
        qpos: np.ndarray,
        observation: Observation | None,
        active: bool,
        pressed: bool,
    ) -> Action:
        del pressed
        defaults = self._defaults_from_obs(observation)
        if not active:
            self._last_delta_eef_sample = None
            return self._zero_delta_eef_action(defaults)

        pos, quat, gripper = self._sample_fello_eef(qpos)
        if self._last_delta_eef_sample is None:
            self._last_delta_eef_sample = {
                "pos": pos.copy(),
                "quat": quat.copy(),
            }
            return self._delta_eef_action(
                defaults,
                np.zeros(3, dtype=np.float32),
                _IDENTITY_QUAT_XYZW.copy(),
                gripper,
            )

        prev_pos = self._last_delta_eef_sample["pos"]
        prev_quat = self._last_delta_eef_sample["quat"]
        delta_pos = ((pos - prev_pos) * self._scaled_control_xyz_scale).astype(
            np.float32
        )
        delta_rpy = (
            (Rotation.from_quat(quat) * Rotation.from_quat(prev_quat).inv())
            .as_euler("xyz")
            .astype(np.float32)
        )
        delta_quat = (
            Rotation.from_euler("xyz", delta_rpy * self._scaled_control_rpy_scale)
            .as_quat()
            .astype(np.float32)
        )
        delta_pos, delta_quat = self._clip_delta_eef_action(delta_pos, delta_quat)
        self._last_delta_eef_sample = {
            "pos": pos.copy(),
            "quat": quat.copy(),
        }
        return self._delta_eef_action(defaults, delta_pos, delta_quat, gripper)

    def _ui_control_button_states(self, buttons: np.ndarray) -> dict[str, bool]:
        max_idx = max(self._ui_button_map) if self._ui_button_map else 0
        if buttons.size <= max_idx:
            return {}  # invalid mapping, skip
        return {
            "start": bool(buttons[self._ui_button_map[0]] > _BUTTON_THRESHOLD),
            "pause": bool(buttons[self._ui_button_map[1]] > _BUTTON_THRESHOLD),
            "home": bool(buttons[self._ui_button_map[2]] > _BUTTON_THRESHOLD),
        }

    def set_external_takeover_pressed(self, pressed: bool) -> None:
        self._external_takeover_pressed = bool(pressed)

    def _defaults_from_obs(
        self, observation: Observation | None
    ) -> dict[str, np.ndarray]:
        if observation is None:
            return {
                "left_joint_pos": np.zeros(6, dtype=np.float32),
                "left_gripper_pos": np.array([1.0], dtype=np.float32),
                "right_joint_pos": np.zeros(6, dtype=np.float32),
                "right_gripper_pos": np.array([1.0], dtype=np.float32),
            }

        def _get(name: str, fallback: np.ndarray) -> np.ndarray:
            value = observation.get(name, fallback)
            return np.asarray(value, dtype=np.float32).reshape(fallback.shape)

        return {
            "left_joint_pos": _get("left_joint_pos", np.zeros(6, dtype=np.float32)),
            "left_gripper_pos": _get(
                "left_gripper_pos", np.array([1.0], dtype=np.float32)
            ),
            "right_joint_pos": _get("right_joint_pos", np.zeros(6, dtype=np.float32)),
            "right_gripper_pos": _get(
                "right_gripper_pos", np.array([1.0], dtype=np.float32)
            ),
        }

    def _rate_limit(
        self, target_qpos: np.ndarray, current_qpos: np.ndarray, max_step_rad: float
    ) -> np.ndarray:
        """limit the rate of change from current to target, return the clipped target"""
        delta = target_qpos - current_qpos
        max_step_rad = float(max_step_rad)
        if np.linalg.norm(delta) <= max_step_rad:
            return target_qpos
        else:
            clipped_qpos = current_qpos + (delta / np.linalg.norm(delta)) * max_step_rad
            return clipped_qpos.astype(np.float32)

    def _maybe_align_fello(self, observation: Observation, pressed: bool) -> None:
        if pressed:
            return
        now = time.time()
        if now - self._last_align_send < self._align_interval_s:
            return
        defaults = self._defaults_from_obs(observation)
        if self._target_side == "left":
            arm = defaults["left_joint_pos"]
            gripper = defaults["left_gripper_pos"]
        else:
            arm = defaults["right_joint_pos"]
            gripper = defaults["right_gripper_pos"]
        fello_qpos = self._map_yam_to_fello(arm, gripper)
        if self._align_filtered_qpos is None or self.clipped_qpos is None:
            self._align_filtered_qpos = fello_qpos.copy()
            self.clipped_qpos = fello_qpos.copy()
        else:
            alpha = self._align_lpf_alpha
            self._align_filtered_qpos = (
                alpha * fello_qpos + (1.0 - alpha) * self._align_filtered_qpos
            )
            self.clipped_qpos = self._rate_limit(
                self._align_filtered_qpos, self.clipped_qpos, 0.1
            )
        fello_qpos = self.clipped_qpos
        joint_state = {
            "pos": fello_qpos,
            "vel": np.zeros(7, dtype=np.float32),
            "kp": self._default_kp.copy(),
            "kd": self._default_kd.copy(),
        }
        self._leader.command_joint_pos(joint_state)
        self._last_align_send = now

    def slow_home(
        self,
        observation: Observation | None,
        duration_s: float = 2.0,
        steps: int = 40,
    ) -> None:
        defaults = self._defaults_from_obs(observation)
        if self._target_side == "left":
            arm = defaults["left_joint_pos"]
            gripper = defaults["left_gripper_pos"]
        else:
            arm = defaults["right_joint_pos"]
            gripper = defaults["right_gripper_pos"]
        target_qpos = self._map_yam_to_fello(arm, gripper)

        current_qpos, _ = self._leader.get_info()
        current_qpos = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
        if current_qpos.shape[0] != 7:
            raise ValueError(
                f"Expected 7-DOF Fello joint positions, got shape {current_qpos.shape}"
            )

        self._leader.set_mode("position")
        self._last_mode = "position"

        steps = max(1, int(steps))
        duration_s = max(0.0, float(duration_s))
        if steps == 1 or duration_s == 0.0:
            interp_qpos = target_qpos
            joint_state = {
                "pos": interp_qpos,
                "vel": np.zeros(7, dtype=np.float32),
                "kp": self._default_kp.copy(),
                "kd": self._default_kd.copy(),
            }
            self._leader.command_joint_pos(joint_state)
            # Reset alignment filter to target position after homing completes
            # This prevents the Fello from immediately returning to pre-homing position
            self._align_filtered_qpos = target_qpos.copy()
            return

        for alpha in np.linspace(0.0, 1.0, steps, dtype=np.float32):
            interp_qpos = (1.0 - alpha) * current_qpos + alpha * target_qpos
            joint_state = {
                "pos": interp_qpos,
                "vel": np.zeros(7, dtype=np.float32),
                "kp": self._default_kp.copy(),
                "kd": self._default_kd.copy(),
            }
            self._leader.command_joint_pos(joint_state)
            time.sleep(duration_s / steps)

        # Reset alignment filter to target position after homing completes
        # This prevents the Fello from immediately returning to pre-homing position
        self._align_filtered_qpos = target_qpos.copy()

    def hold_current_position(self) -> None:
        """Switch the leader to position mode while holding its current pose."""
        current_qpos, _ = self._leader.get_info()
        current_qpos = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
        if current_qpos.shape[0] != 7:
            raise ValueError(
                f"Expected 7-DOF Fello joint positions, got shape {current_qpos.shape}"
            )

        joint_state = {
            "pos": current_qpos,
            "vel": np.zeros(7, dtype=np.float32),
            "kp": self._default_kp.copy(),
            "kd": self._default_kd.copy(),
        }
        self._leader.command_joint_pos(joint_state)
        self._leader.set_mode("position")
        self._last_mode = "position"
        self._align_filtered_qpos = current_qpos.copy()

    def get_action(
        self, observation: Observation, options: Options | None = None
    ) -> tuple[Action, Info]:
        del options
        qpos, buttons = self._leader.get_info()
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        buttons = np.asarray(buttons, dtype=np.float32).reshape(-1)
        pressed = buttons[self._takeover_button] > _BUTTON_THRESHOLD

        scaled_control_active = False
        if self._scaled_control and self._use_footswitch:
            scaled_control_active = bool(
                buttons[self._scaled_control_active_button] > _BUTTON_THRESHOLD
            )
        if self._action_type == "joint":
            action = self._get_joint_action(qpos, observation)
        elif self._scaled_control:
            action = self._get_scaled_delta_eef_action(
                qpos,
                observation,
                scaled_control_active,
                pressed,
            )
            nominal_position_commanded = pressed
        else:
            action = self._get_delta_eef_action(qpos, observation)

        right_button_states: dict[str, bool] | None = None
        if (
            self._use_footswitch
            and self._target_side == "right"
            and self._button_mode == "ui_control"
        ):
            right_button_states = self._ui_control_button_states(buttons)
        mode = "gravity" if pressed else "position"
        # print(f"[Fello] mode: {self._target_side} {mode} (pressed={pressed})")

        if pressed:
            self._leader.set_mode("gravity")
        elif not pressed:
            if mode != self._last_mode and self._scaled_control:
                self._leader.set_mode("gravity")
            if self._scaled_control:
                self._leader.command_joint_pos(
                    self._joint_position_state(self._scaled_control_nominal_position)
                )
            else:
                self._maybe_align_fello(observation, pressed)
        self._last_mode = mode
        info: Info = {
            "buttons": buttons.copy(),
        }
        if right_button_states is not None:
            info["right_button_states"] = right_button_states
        return action, info

    def poll_button_events(self, observation: Observation | None = None) -> Info:
        """Read footswitch buttons and return edge-detected events without commanding the arm."""
        info: Info = {}
        if not self._use_footswitch:
            return info
        _, buttons = self._leader.get_info()
        buttons = np.asarray(buttons, dtype=np.float32).reshape(-1)
        if buttons.size < 3:
            if self._target_side == "right" and self._button_mode == "ui_control":
                info["right_button_states"] = {
                    "start": False,
                    "pause": False,
                    "home": False,
                }
            return info
        if self._target_side == "right" and self._button_mode == "ui_control":
            info["right_button_states"] = self._ui_control_button_states(buttons)
        if self._button_mode == "ui_control":
            ui_states = self._ui_control_button_states(buttons)
            start_cur = ui_states["start"]
            if start_cur and not self._prev_save_pressed:
                info["ui_start"] = True
            self._prev_save_pressed = start_cur
            pause_cur = ui_states["pause"]
            if pause_cur and not self._prev_start_pressed:
                info["ui_pause"] = True
            self._prev_start_pressed = pause_cur
            home_cur = ui_states["home"]
            if home_cur and not self._prev_home_pressed:
                info["ui_home"] = True
            self._prev_home_pressed = home_cur
        else:
            save_cur = bool(buttons[0] > _BUTTON_THRESHOLD)
            if save_cur and not self._prev_save_pressed:
                info["save_pressed"] = True
            self._prev_save_pressed = save_cur
            start_cur = bool(buttons[2] > _BUTTON_THRESHOLD)
            if start_cur and not self._prev_start_pressed:
                info["start_pressed"] = True
            self._prev_start_pressed = start_cur
        return info

    def reset(self) -> Info:
        self._last_mode = None
        self._align_filtered_qpos = None
        self._last_align_send = 0.0
        self._last_gripper_print = 0.0
        self._last_footswitch_print = 0.0
        self._last_delta_eef_sample = None
        self._external_takeover_pressed = False
        self._prev_save_pressed = False
        self._prev_start_pressed = False
        self._prev_home_pressed = False
        if self._action_type == "delta_eef":
            return {
                "initial_state": self._zero_delta_eef_action(
                    self._defaults_from_obs(None)
                )
            }
        action, _ = self.get_action(None)
        return {"initial_state": action}


class DualFelloPolicy(Policy):
    """Bimanual Fello teleoperation over the configured Fello side(s).

    Each enabled side connects to its own Fello leader server and manages its own
    footswitch / alignment independently.  The merged info reports per-side
    triggers so ``HILPolicyWrapper`` can override each arm separately.
    """

    def __init__(
        self,
        action_type: Literal["joint", "delta_eef"],
        scaled_control: bool,  # scaled control will scale down the eef deltas to make them easier to control
        scaled_control_xyz_scale: Sequence[
            float
        ],  # [x, y, z] EE-position delta scale, shared by both sides
        delta_ee_translation_xyz_max: Sequence[
            float
        ],  # [x, y, z] per-step EE translation clip (m), shared by both sides
        enabled_sides: Literal["left", "right", "both"],
        decouple_translation: bool,  # true then only the lowest 3 joint contribute to translation delta
        takeover_button: tuple[int, int],  # tuple of (left, right)
    ):
        self._enabled_sides = (
            ("left", "right") if enabled_sides == "both" else (enabled_sides,)
        )
        self._action_type = action_type
        self._left = (
            FelloTeleopPolicy(
                target_side="left",
                action_type=action_type,
                scaled_control=scaled_control,
                scaled_control_xyz_scale=scaled_control_xyz_scale,
                delta_ee_translation_xyz_max=delta_ee_translation_xyz_max,
                decouple_translation=decouple_translation,
                takeover_button=takeover_button[0],
            )
            if "left" in self._enabled_sides
            else None
        )
        self._right = (
            FelloTeleopPolicy(
                target_side="right",
                action_type=action_type,
                scaled_control=scaled_control,
                scaled_control_xyz_scale=scaled_control_xyz_scale,
                delta_ee_translation_xyz_max=delta_ee_translation_xyz_max,
                decouple_translation=decouple_translation,
                takeover_button=takeover_button[1],
            )
            if "right" in self._enabled_sides
            else None
        )

    @staticmethod
    def _zero_delta_eef_action(defaults: dict[str, np.ndarray]) -> Action:
        return {
            "left_ee_pos": np.zeros(3, dtype=np.float32),
            "left_ee_rot6d": _IDENTITY_ROT6D.copy(),
            "left_gripper_pos": defaults["left_gripper_pos"],
            "right_ee_pos": np.zeros(3, dtype=np.float32),
            "right_ee_rot6d": _IDENTITY_ROT6D.copy(),
            "right_gripper_pos": defaults["right_gripper_pos"],
        }

    def get_action(
        self, observation: Observation, options: Options | None = None
    ) -> tuple[Action, Info]:
        defaults = (self._left or self._right)._defaults_from_obs(observation)
        if self._action_type == "delta_eef":
            action: Action = self._zero_delta_eef_action(defaults)
            action_fields = ("ee_pos", "ee_rot6d", "gripper_pos")
        else:
            action = {
                "left_joint_pos": defaults["left_joint_pos"],
                "left_gripper_pos": defaults["left_gripper_pos"],
                "right_joint_pos": defaults["right_joint_pos"],
                "right_gripper_pos": defaults["right_gripper_pos"],
            }
            action_fields = ("joint_pos", "gripper_pos")
        info: Info = {
            "source": "human",
            "left_buttons": np.zeros(3, dtype=np.float32),
            "right_buttons": np.zeros(3, dtype=np.float32),
        }
        for side, side_policy in (("left", self._left), ("right", self._right)):
            if side not in self._enabled_sides or side_policy is None:
                continue
            side_action, side_info = side_policy.get_action(observation, options)
            for field in action_fields:
                action[f"{side}_{field}"] = side_action[f"{side}_{field}"]
            info[f"{side}_buttons"] = side_info["buttons"]
            if side == "right" and "right_button_states" in side_info:
                info["right_button_states"] = side_info["right_button_states"]
            for key in (
                "save_pressed",
                "start_pressed",
                "ui_start",
                "ui_pause",
                "ui_home",
            ):
                if side_info.get(key):
                    info[key] = True
        return action, info

    def get_action_for_sides(
        self,
        observation: Observation,
        enabled_sides: Sequence[Literal["left", "right"]],
        options: Options | None = None,
    ) -> tuple[Action, Info]:
        active = set(enabled_sides)
        invalid = active - {"left", "right"}
        if invalid:
            raise ValueError(f"Invalid Fello side(s): {sorted(invalid)!r}")
        active &= set(self._enabled_sides)

        defaults = (self._left or self._right)._defaults_from_obs(observation)
        if self._action_type == "delta_eef":
            action: Action = self._zero_delta_eef_action(defaults)
            action_fields = ("ee_pos", "ee_rot6d", "gripper_pos")
        else:
            action = {
                "left_joint_pos": defaults["left_joint_pos"],
                "left_gripper_pos": defaults["left_gripper_pos"],
                "right_joint_pos": defaults["right_joint_pos"],
                "right_gripper_pos": defaults["right_gripper_pos"],
            }
            action_fields = ("joint_pos", "gripper_pos")
        info: Info = {
            "source": "human",
            "is_taking_over": False,
            "left_trigger": False,
            "right_trigger": False,
        }

        for side in ("left", "right"):
            if side not in active:
                continue
            side_policy = self._left if side == "left" else self._right
            if side_policy is None:
                continue
            side_action, side_info = side_policy.get_action(observation, options)
            for field in action_fields:
                action[f"{side}_{field}"] = side_action[f"{side}_{field}"]

            trigger = bool(side_info.get(f"{side}_trigger"))
            info[f"{side}_trigger"] = trigger
            info["is_taking_over"] = bool(info["is_taking_over"]) or trigger
            if side == "right" and "right_button_states" in side_info:
                info["right_button_states"] = side_info["right_button_states"]
            for key in (
                "save_pressed",
                "start_pressed",
                "ui_start",
                "ui_pause",
                "ui_home",
            ):
                if side_info.get(key):
                    info[key] = True
        return action, info

    def slow_home(
        self,
        observation: Observation | None,
        duration_s: float = 2.0,
        steps: int = 40,
    ) -> None:
        import threading

        errors: list[tuple[str, BaseException]] = []
        errors_lock = threading.Lock()

        def _run(side: str, fn, *args, **kwargs) -> None:
            try:
                fn(*args, **kwargs)
            except BaseException as exc:
                with errors_lock:
                    errors.append((side, exc))

        threads = [
            threading.Thread(
                target=_run,
                args=(side, policy.slow_home, observation),
                kwargs={"duration_s": duration_s, "steps": steps},
            )
            for side, policy in (("left", self._left), ("right", self._right))
            if side in self._enabled_sides and policy is not None
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            detail = "; ".join(f"{side}: {exc!r}" for side, exc in errors)
            raise RuntimeError(f"Fello dual slow_home failed: {detail}") from errors[0][
                1
            ]

    def hold_current_position(self) -> None:
        import threading

        errors: list[tuple[str, BaseException]] = []
        errors_lock = threading.Lock()

        def _run(side: str, fn) -> None:
            try:
                fn()
            except BaseException as exc:
                with errors_lock:
                    errors.append((side, exc))

        threads = [
            threading.Thread(target=_run, args=(side, policy.hold_current_position))
            for side, policy in (("left", self._left), ("right", self._right))
            if side in self._enabled_sides and policy is not None
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            detail = "; ".join(f"{side}: {exc!r}" for side, exc in errors)
            raise RuntimeError(
                f"Fello dual hold_current_position failed: {detail}"
            ) from errors[0][1]

    def poll_button_events(self, observation: Observation | None = None) -> Info:
        merged: Info = {}
        for side, policy in (("left", self._left), ("right", self._right)):
            if side not in self._enabled_sides or policy is None:
                continue
            side_ev = policy.poll_button_events(observation)
            if side == "right" and "right_button_states" in side_ev:
                merged["right_button_states"] = side_ev["right_button_states"]
            for key in (
                "save_pressed",
                "start_pressed",
                "ui_start",
                "ui_pause",
                "ui_home",
            ):
                if side_ev.get(key):
                    merged[key] = True
        return merged

    def reset(self) -> Info:
        merged_state: dict[str, np.ndarray] = {}
        for side, policy in (("left", self._left), ("right", self._right)):
            if side not in self._enabled_sides or policy is None:
                continue
            policy.set_external_takeover_pressed(False)
            info = policy.reset()
            if isinstance(info, dict) and "initial_state" in info:
                merged_state.update(info["initial_state"])
        return {"initial_state": merged_state}


def _get_requested_side_from_argv(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--side" and i + 1 < len(argv):
            return argv[i + 1].lower()
        if arg.startswith("--side="):
            return arg.split("=", 1)[1].lower()
    return None


def main() -> None:
    fello_policy = DualFelloPolicy(
        action_type="delta_eef",
        scaled_control=True,
        scaled_control_xyz_scale=(0.25, 0.25, 1.0),
        delta_ee_translation_xyz_max=(0.0003, 0.0003, 0.0006),
        enabled_sides="right",
        decouple_translation=True,
        takeover_button=(1, 0),
    )

    obs = {}
    try:
        while True:
            obs[f"left_joint_pos"] = np.zeros(6, dtype=np.float32)
            obs[f"left_gripper_pos"] = np.zeros(1, dtype=np.float32)
            obs[f"right_joint_pos"] = np.zeros(6, dtype=np.float32)
            obs[f"right_gripper_pos"] = np.zeros(1, dtype=np.float32)
            action, info = fello_policy.get_action(obs)
            print(f"Action: {action['right_ee_pos']}")
            # obs, _, _, _, _ = env.step(action)
    finally:
        pass


if __name__ == "__main__":
    main()

