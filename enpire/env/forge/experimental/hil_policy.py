# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import numpy as np

from enpire.policy.legacy import Action, Info

# Module-level logger
logger = logging.getLogger(__name__)

# Delta-mode constants (matching CAP server behavior)
_GRIP_SETTLE_TICKS = 5
_GRIP_DEADBAND = 0.05


class HILPolicyWrapper:
    def __init__(self, base_policy: Any, hil_policy: Any, delta_mode: bool = False):
        self.base_policy = base_policy
        self.hil_policy = hil_policy
        self._delta_mode = delta_mode

        assert self.hil_policy._control_mode == "joint_position", (
            "HIL policy must be in joint position mode"
        )

        self._took_control_of_left = False
        self._last_left_joint_pos = None
        self._last_left_gripper_pos = None
        self._took_control_of_right = False
        self._last_right_joint_pos = None
        self._last_right_gripper_pos = None

        # Delta-mode anchors (set on trigger onset, cleared on release)
        self._left_anchor: dict | None = None
        self._right_anchor: dict | None = None

        # Only reset the base policy on takeover entry/release for RTC policies,
        # which need a clean state machine. For AsyncChunkingPolicy the reset
        # blocks waiting for the inference thread and adds unnecessary latency.
        self._reset_policy_during_human_takeover = bool(
            getattr(self.base_policy, "realtime_rtc_enabled", False)
        )

    def _compute_delta_cmd(
        self,
        side: str,
        hil_action: Action,
        observation: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute press-anchored delta command for one arm.

        On first call per takeover (anchor is None), records anchors from
        the current Fello and YAM joint positions. Subsequent calls compute:
            arm_cmd = yam_anchor + (fello_now - fello_anchor)
        Gripper uses range-aware scaling with settle period and deadband
        (matching CAP server behavior in cap_server.py:845-887).

        Returns (arm_cmd[6], grip_cmd[1]) as float32 arrays.
        """
        anchor_attr = f"_{side}_anchor"
        anchor = getattr(self, anchor_attr)

        fello_arm = np.asarray(hil_action[f"{side}_joint_pos"], dtype=np.float64)
        fello_grip = float(np.asarray(hil_action[f"{side}_gripper_pos"]).reshape(-1)[0])

        if anchor is None:
            # Takeover onset — record anchors
            yam_arm = np.asarray(
                observation.get(f"{side}_joint_pos", np.zeros(6)), dtype=np.float64
            )
            yam_grip = float(
                np.asarray(observation.get(f"{side}_gripper_pos", np.ones(1))).reshape(
                    -1
                )[0]
            )
            anchor = {
                "fello_arm": fello_arm.copy(),
                "fello_grip": fello_grip,
                "yam_arm": yam_arm.copy(),
                "yam_grip": yam_grip,
                "grip_settle": 0,
            }
            setattr(self, anchor_attr, anchor)

        # Arm: trivial delta
        arm_cmd = anchor["yam_arm"] + (fello_arm - anchor["fello_arm"])

        # Gripper: settle period then range-aware scaling
        if anchor["grip_settle"] < _GRIP_SETTLE_TICKS:
            grip_cmd = anchor["yam_grip"]
            anchor["grip_settle"] += 1
        else:
            fello_delta = fello_grip - anchor["fello_grip"]
            if abs(fello_delta) < _GRIP_DEADBAND:
                grip_cmd = anchor["yam_grip"]
            else:
                y_anc = anchor["yam_grip"]
                f_anc = anchor["fello_grip"]
                if fello_delta >= 0:
                    room_fello = max(1.0 - f_anc, 1e-4)
                    room_yam = 1.0 - y_anc
                else:
                    room_fello = max(f_anc, 1e-4)
                    room_yam = y_anc
                grip_cmd = y_anc + fello_delta * (room_yam / room_fello)

        grip_cmd = float(np.clip(grip_cmd, 0.0, 1.0))
        return arm_cmd.astype(np.float32), np.array([grip_cmd], dtype=np.float32)

    def get_action(self, observation) -> tuple[Action, Info]:
        hil_action, hil_info = self.hil_policy.get_action(observation)

        human_control_requested = bool(
            hil_info["left_trigger"] or hil_info["right_trigger"]
        )
        takeover_active = bool(
            self._took_control_of_left or self._took_control_of_right
        )
        entering_control = human_control_requested and not takeover_active

        # Detect human -> policy transition (both triggers released after takeover)
        releasing_control = (
            not hil_info["left_trigger"] and not hil_info["right_trigger"]
        ) and takeover_active

        if self._reset_policy_during_human_takeover and entering_control:
            logger.debug("Resetting base policy; HIL control acquired")
            self.base_policy.reset()

        if releasing_control:
            logger.debug("Resetting base policy; HIL control released")
            self.base_policy.reset()
        elif not human_control_requested:
            logger.debug(
                "HIL triggers - left=%s right=%s took_left=%s took_right=%s",
                hil_info["left_trigger"],
                hil_info["right_trigger"],
                self._took_control_of_left,
                self._took_control_of_right,
            )

        # If both triggers active → full human control, skip base policy entirely
        full_human = hil_info["left_trigger"] and hil_info["right_trigger"]
        if not full_human:
            base_action, policy_info = self.base_policy.get_action(observation)
            base_action = base_action.copy()
        else:
            base_action = dict(hil_action)
            policy_info = {}

        base_action["source"] = "policy"

        if hil_info["left_trigger"]:
            if self._delta_mode:
                arm, grip = self._compute_delta_cmd("left", hil_action, observation)
            else:
                arm = hil_action["left_joint_pos"]
                grip = hil_action["left_gripper_pos"]
            base_action["left_joint_pos"] = arm
            base_action["left_gripper_pos"] = grip
            self._took_control_of_left = True
            self._last_left_joint_pos = arm
            self._last_left_gripper_pos = grip
            base_action["source"] = "human"
        else:
            self._left_anchor = None
            if self._took_control_of_left:
                base_action["left_joint_pos"] = self._last_left_joint_pos
                base_action["left_gripper_pos"] = self._last_left_gripper_pos

        if hil_info["right_trigger"]:
            if self._delta_mode:
                arm, grip = self._compute_delta_cmd("right", hil_action, observation)
            else:
                arm = hil_action["right_joint_pos"]
                grip = hil_action["right_gripper_pos"]
            base_action["right_joint_pos"] = arm
            base_action["right_gripper_pos"] = grip
            self._took_control_of_right = True
            self._last_right_joint_pos = arm
            self._last_right_gripper_pos = grip
            base_action["source"] = "human"
        else:
            self._right_anchor = None
            if self._took_control_of_right:
                base_action["right_joint_pos"] = self._last_right_joint_pos
                base_action["right_gripper_pos"] = self._last_right_gripper_pos

        # Clear takeover flags AFTER override checks so the transition step
        # holds the last human position instead of jumping to the policy action
        if releasing_control:
            self._took_control_of_left = False
            self._took_control_of_right = False
            policy_info["human_takeover_reset"] = True

        # Forward footswitch button events from HIL policy to outer info
        for key in ("save_pressed", "start_pressed"):
            if hil_info.get(key):
                policy_info[key] = hil_info[key]

        policy_info["ui_start"] = bool(hil_info.get("ui_start", False))
        policy_info["ui_pause"] = bool(hil_info.get("ui_pause", False))
        policy_info["ui_home"] = bool(hil_info.get("ui_home", False))
        # right_button_states is a dict — validate it's a dict
        rbs = hil_info.get("right_button_states", {})
        policy_info["right_button_states"] = rbs if isinstance(rbs, dict) else {}

        return base_action, policy_info

    def poll_button_events(self, observation) -> Info:
        """Delegate button polling to the HIL policy."""
        if hasattr(self.hil_policy, "poll_button_events"):
            return self.hil_policy.poll_button_events(observation)
        return {}

    def reset(self) -> Info | None:
        """Reset both base and HIL policies and clear takeover state."""
        self._took_control_of_left = False
        self._last_left_joint_pos = None
        self._last_left_gripper_pos = None
        self._took_control_of_right = False
        self._last_right_joint_pos = None
        self._last_right_gripper_pos = None
        self._left_anchor = None
        self._right_anchor = None

        if hasattr(self.hil_policy, "reset"):
            self.hil_policy.reset()
        return self.base_policy.reset()

    def __getattr__(self, name):
        """
        Fallback method: If an attribute or method is not found
        in MyPolicyWrapper, it automatically looks for it in the
        wrapped instance (`self.policy`).
        """
        # 3. Delegate attribute access to the wrapped policy
        # if not defined by this class, delegate to the base policy
        return getattr(self.base_policy, name)
