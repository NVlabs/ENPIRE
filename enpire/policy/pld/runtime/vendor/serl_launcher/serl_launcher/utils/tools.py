# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def action_scaling(actions_, action_scaling_config: dict):
    """
    Scale actions based on configuration.

    Args:
        actions_: Input actions to scale
        action_scaling_config: Dictionary containing scaling parameters
            - scales: List of scaling factors for each action dimension
            - gripper_mode: How to handle gripper action (e.g., "binary_threshold")
            - gripper_threshold: Threshold for binary gripper conversion

    Returns:
        Scaled actions
    """
    actions = actions_.copy()
    actions[0] *= action_scaling_config.scales.eef_pose.x
    actions[1] *= action_scaling_config.scales.eef_pose.y
    actions[2] *= action_scaling_config.scales.eef_pose.z
    actions[3] *= action_scaling_config.scales.eef_pose.roll
    actions[4] *= action_scaling_config.scales.eef_pose.pitch
    actions[5] *= action_scaling_config.scales.eef_pose.yaw
    if actions[6] == action_scaling_config.scales.gripper.ths:
        actions[6] = 0.0
    else:
        actions[6] = (
            2.0 * (actions[6] > action_scaling_config.scales.gripper.ths) - 1.0
        ) * action_scaling_config.scales.gripper.sign

    return actions


def action_scaling_inv(actions_, action_scaling_config: dict):
    """
    Inverse scale actions based on configuration.

    Args:
        actions_: Input actions to inverse scale
        action_scaling_config: Dictionary containing scaling parameters
            - scales: List of scaling factors for each action dimension

    Returns:
        Inverse scaled actions
    """
    actions = actions_.copy()
    actions[0] /= action_scaling_config.scales.eef_pose.x
    actions[1] /= action_scaling_config.scales.eef_pose.y
    actions[2] /= action_scaling_config.scales.eef_pose.z
    actions[3] /= action_scaling_config.scales.eef_pose.roll
    actions[4] /= action_scaling_config.scales.eef_pose.pitch
    actions[5] /= action_scaling_config.scales.eef_pose.yaw
    actions[6] = (
        2.0 * (actions[6] > action_scaling_config.scales.gripper.ths) - 1.0
    ) * action_scaling_config.scales.gripper.sign
    return actions


def _yam_delta_eef_scale_and_offset(action_scaling_config):
    s = action_scaling_config.scales.bimanual_eef
    pos = float(s.pos)
    quat_xyz = float(s.quat_xyz)
    quat_w = float(s.quat_w)
    grip = float(s.gripper)
    per_arm_scale = np.array(
        [pos, pos, pos, quat_xyz, quat_xyz, quat_xyz, quat_w, grip],
        dtype=np.float64,
    )
    per_arm_offset = np.zeros(8, dtype=np.float64)
    return np.concatenate([per_arm_scale, per_arm_scale]), np.concatenate(
        [per_arm_offset, per_arm_offset]
    )


def action_scaling_yam_delta_eef(actions_, action_scaling_config):
    """SAC-normalised 8-D/16-D action -> raw delta-EE quaternion command."""
    actions = np.asarray(actions_, dtype=np.float64).copy()
    scale, offset = _yam_delta_eef_scale_and_offset(action_scaling_config)
    if actions.shape[-1] == 8:
        scale, offset = scale[:8], offset[:8]
    out = actions * scale + offset
    out[..., 7] = np.clip(0.5 * (actions[..., 7] + 1.0), 0.0, 1.0)
    if actions.shape[-1] == 16:
        out[..., 15] = np.clip(0.5 * (actions[..., 15] + 1.0), 0.0, 1.0)
    return out


def action_scaling_yam_delta_eef_inv(actions_, action_scaling_config):
    """Raw 8-D/16-D delta-EE quaternion action -> SAC-normalised action."""
    actions = np.asarray(actions_, dtype=np.float64).copy()
    scale, offset = _yam_delta_eef_scale_and_offset(action_scaling_config)
    if actions.shape[-1] == 8:
        scale, offset = scale[:8], offset[:8]
    out = (actions - offset) / scale
    out[..., 7] = 2.0 * actions[..., 7] - 1.0
    if actions.shape[-1] == 16:
        out[..., 15] = 2.0 * actions[..., 15] - 1.0
    return out


def _yam_delta_eef_rot6d_scale_and_offset(action_scaling_config):
    s = action_scaling_config.scales.bimanual_eef
    pos = float(s.pos)
    rot6d = float(s.rot6d)
    grip = float(s.gripper)
    per_arm_scale = np.array(
        [pos, pos, pos, rot6d, rot6d, rot6d, rot6d, rot6d, rot6d, grip],
        dtype=np.float64,
    )
    per_arm_offset = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return np.concatenate([per_arm_scale, per_arm_scale]), np.concatenate(
        [per_arm_offset, per_arm_offset]
    )


def action_scaling_yam_delta_eef_rot6d(actions_, action_scaling_config):
    """SAC-normalised 10-D/20-D action -> raw delta-EE rot6d command.

    Zero normalised rotation maps to identity 6D, so zero-base/zero-delta
    policies produce a true no-op orientation delta.
    """
    actions = np.asarray(actions_, dtype=np.float64).copy()
    scale, offset = _yam_delta_eef_rot6d_scale_and_offset(action_scaling_config)
    if actions.shape[-1] == 10:
        scale, offset = scale[:10], offset[:10]
    out = actions * scale + offset
    out[..., 9] = np.clip(0.5 * (actions[..., 9] + 1.0), 0.0, 1.0)
    if actions.shape[-1] == 20:
        out[..., 19] = np.clip(0.5 * (actions[..., 19] + 1.0), 0.0, 1.0)
    return out


def action_scaling_yam_delta_eef_rot6d_inv(actions_, action_scaling_config):
    """Raw 10-D/20-D delta-EE rot6d action -> SAC-normalised action."""
    actions = np.asarray(actions_, dtype=np.float64).copy()
    scale, offset = _yam_delta_eef_rot6d_scale_and_offset(action_scaling_config)
    if actions.shape[-1] == 10:
        scale, offset = scale[:10], offset[:10]
    out = (actions - offset) / scale
    out[..., 9] = 2.0 * actions[..., 9] - 1.0
    if actions.shape[-1] == 20:
        out[..., 19] = 2.0 * actions[..., 19] - 1.0
    return out


def _xyz_scale(value, *, name: str):
    raw = (
        list(value)
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
        else value
    )
    scale = np.asarray(raw, dtype=np.float64)
    if scale.ndim == 0:
        return np.repeat(float(scale), 3)
    scale = scale.reshape(-1)
    if scale.shape[0] != 3:
        raise ValueError(f"{name} must be a scalar or 3 values for [x, y, z]")
    return scale


def _yam_delta_eef_pos_scale(action_scaling_config, action_dim: int):
    scales = action_scaling_config.scales
    if hasattr(scales, "eef_pos"):
        scale = _xyz_scale(scales.eef_pos.pos, name="scales.eef_pos.pos")
    else:
        scale = _xyz_scale(scales.bimanual_eef.pos, name="scales.bimanual_eef.pos")
    if int(action_dim) == 6:
        return np.concatenate([scale, scale])
    return scale


def action_scaling_yam_delta_eef_pos3(actions_, action_scaling_config):
    """SAC-normalised 3-D/6-D delta-EEF position -> raw meter command."""
    actions = np.asarray(actions_, dtype=np.float64).copy()
    if actions.shape[-1] not in (3, 6):
        raise ValueError(
            "delta_eef_pos action scaling expects 3-D or 6-D action, "
            f"got shape {actions.shape}"
        )
    return actions * _yam_delta_eef_pos_scale(action_scaling_config, actions.shape[-1])


def action_scaling_yam_delta_eef_pos3_inv(actions_, action_scaling_config):
    """Raw 3-D/6-D delta-EEF position meters -> SAC-normalised action."""
    actions = np.asarray(actions_, dtype=np.float64).copy()
    if actions.shape[-1] not in (3, 6):
        raise ValueError(
            "delta_eef_pos inverse action scaling expects 3-D or 6-D action, "
            f"got shape {actions.shape}"
        )
    return actions / _yam_delta_eef_pos_scale(action_scaling_config, actions.shape[-1])


def mask_delta_rotation_action(actions_, action_repr: str):
    """Mask normalized delta-rotation action slots to no-op.

    For delta_eef_rot6d, normalized zeros map through
    action_scaling_yam_delta_eef_rot6d() to identity 6D rotation
    [1, 0, 0, 0, 1, 0]. This helper leaves position and gripper slots intact.
    """
    actions = np.asarray(actions_).copy()
    if str(action_repr) != "delta_eef_rot6d":
        return actions
    if actions.shape[-1] == 10:
        actions[..., 3:9] = 0.0
    elif actions.shape[-1] == 20:
        actions[..., 3:9] = 0.0
        actions[..., 13:19] = 0.0
    else:
        raise ValueError(
            "delta_eef_rot6d rotation mask expects 10-D or 20-D action, "
            f"got shape {actions.shape}"
        )
    return actions


def mask_gripper_action(actions_, action_repr: str, value: float = -1.0):
    """Mask normalized gripper action slots to a fixed value.

    YAM delta-EEF gripper actions are absolute and normalized in [-1, 1].
    value=-1 maps to raw gripper 0.0, i.e. closed.
    """
    actions = np.asarray(actions_).copy()
    repr_name = str(action_repr)
    if repr_name in ("delta_eef_rot6d", "delta_eef_quat"):
        if actions.shape[-1] in (8, 10):
            actions[..., -1] = value
        elif actions.shape[-1] == 16:
            actions[..., 7] = value
            actions[..., 15] = value
        elif actions.shape[-1] == 20:
            actions[..., 9] = value
            actions[..., 19] = value
        else:
            raise ValueError(
                f"{repr_name} gripper mask expects 8/10/16/20-D action, "
                f"got shape {actions.shape}"
            )
    elif repr_name == "joint":
        if actions.shape[-1] == 7:
            actions[..., 6] = value
        elif actions.shape[-1] == 14:
            actions[..., 6] = value
            actions[..., 13] = value
        else:
            raise ValueError(
                f"joint gripper mask expects 7-D or 14-D action, got shape {actions.shape}"
            )
    return actions


def zero_agent_launcher(
    config, instruction, sharding, seed, sample_obs, sample_action, inv_action_scaling_fn
):
    class AgentWrapper:
        def __init__(self):
            ...

        def policy_inference_fn(self, obs, rng):
            return np.zeros_like(sample_action)

    return AgentWrapper()

