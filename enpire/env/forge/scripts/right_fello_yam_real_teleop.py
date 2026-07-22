# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enpire.env.forge.robot.fello.fello_teleop_policy import FelloTeleopPolicy
from enpire.env.forge.robot.yam.yam_real_env import YamRealEnv


def _right_only_joint_action(
    right_action: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Pad a right-arm joint action with zero left-arm fields."""
    required = ("right_joint_pos", "right_gripper_pos")
    missing = [key for key in required if key not in right_action]
    if missing:
        raise ValueError(f"Right Fello policy action missing keys: {missing}")

    return {
        "left_joint_pos": np.zeros(6, dtype=np.float32),
        "left_gripper_pos": np.zeros(1, dtype=np.float32),
        "right_joint_pos": np.asarray(
            right_action["right_joint_pos"], dtype=np.float32
        ).reshape(6),
        "right_gripper_pos": np.asarray(
            right_action["right_gripper_pos"], dtype=np.float32
        ).reshape(1),
    }


def _summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        inner = ", ".join(
            f"{key}: {_summarize_value(val)}" for key, val in value.items()
        )
        return "{" + inner + "}"

    try:
        arr = np.asarray(value)
    except Exception:
        return repr(value)

    if arr.dtype == object:
        return repr(value)
    if arr.size <= 16:
        return np.array2string(arr, precision=4, suppress_small=True)
    if np.issubdtype(arr.dtype, np.number):
        return (
            f"array(shape={arr.shape}, dtype={arr.dtype}, "
            f"min={float(np.min(arr)):.4g}, max={float(np.max(arr)):.4g}, "
            f"mean={float(np.mean(arr)):.4g})"
        )
    return f"array(shape={arr.shape}, dtype={arr.dtype})"


def _print_mapping(label: str, mapping: dict[str, Any]) -> None:
    print(f"{label}:")
    for key in sorted(mapping):
        print(f"  {key}: {_summarize_value(mapping[key])}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Right-arm-only Fello teleop into YamRealEnv."
    )
    parser.add_argument("--policy-control-freq", type=float, default=30.0)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs forever.")
    parser.add_argument("--takeover-button", type=int, default=0)
    parser.add_argument(
        "--scaled-control",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--decouple-translation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = YamRealEnv(
        control_mode="joint_position",
        policy_control_freq=args.policy_control_freq,
        enable_cameras=not args.no_cameras,
    )
    fello_policy = FelloTeleopPolicy(
        target_side="right",
        action_type="joint",
        scaled_control=args.scaled_control,
        scaled_control_xyz_scale=(0.25, 0.25, 1.0),
        delta_ee_translation_xyz_max=(0.0003, 0.0003, 0.0006),
        decouple_translation=args.decouple_translation,
        takeover_button=args.takeover_button,
    )

    try:
        observation, reset_info = env.reset()
        policy_reset_info = fello_policy.reset()
        print("[right_fello_yam_real_teleop] reset_info:", reset_info)
        print("[right_fello_yam_real_teleop] policy_reset_info:", policy_reset_info)
        _print_mapping("initial_observation", observation)

        step_idx = 0
        while args.max_steps <= 0 or step_idx < args.max_steps:
            policy_action, policy_info = fello_policy.get_action(observation)
            action = _right_only_joint_action(policy_action)

            step_t0 = time.perf_counter()
            observation, reward, terminated, truncated, env_info = env.step(action)
            step_ms = (time.perf_counter() - step_t0) * 1000.0

            print(f"\nstep={step_idx} step_ms={step_ms:.2f} reward={reward}")
            _print_mapping("action", action)
            _print_mapping("observation", observation)
            print(f"policy_info: {_summarize_value(policy_info)}")
            print(f"env_info: {_summarize_value(env_info)}")

            step_idx += 1
            if terminated or truncated:
                observation, reset_info = env.reset()
                print("[right_fello_yam_real_teleop] reset_info:", reset_info)
    except KeyboardInterrupt:
        print("\n[right_fello_yam_real_teleop] stopped by KeyboardInterrupt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
