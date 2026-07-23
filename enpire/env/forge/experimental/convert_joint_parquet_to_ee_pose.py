# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the actions in a parquet file from joint space to ee_pose space.

This script reads a LeRobotDataset parquet file with 14D joint actions and
converts them to 16D ee_pose actions using forward kinematics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
_TBD_ROOT = Path(__file__).resolve().parents[1]
if str(_TBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TBD_ROOT))

from enpire.env.forge.robot.yam.kinematics import YamKinematics


def convert_joint_to_ee_pose(
    input_path: Path, output_path: Path, kinematics: YamKinematics
) -> None:
    """Convert 14D joint actions to 16D end-effector pose actions.

    Args:
        input_path: Path to input parquet file with 14D joint actions
        output_path: Path to output parquet file with 16D ee_pose actions
        kinematics: YamKinematics instance for forward kinematics
    """
    print(f"Loading parquet file from {input_path}")
    df = pd.read_parquet(input_path)

    # Extract actions - stored as list of arrays in "action" column
    actions = np.stack(df["action"].values)  # Shape: (T, 14)

    if actions.shape[1] != 14:
        raise ValueError(
            f"Expected 14D actions, but got {actions.shape[1]}D. "
            "This script only converts from joint space (14D) to ee_pose space (16D)."
        )

    print(f"Converting {len(actions)} actions from joint space to ee_pose space...")

    # Convert each action
    ee_pose_actions = []
    for i, action in enumerate(actions):
        # Extract joint positions and gripper positions from 14D action
        # Format: [left_joint_pos (6D), left_gripper_pos (1D), right_joint_pos (6D), right_gripper_pos (1D)]
        left_joint_pos = action[0:6]
        left_gripper_pos = action[6:7]
        right_joint_pos = action[7:13]
        right_gripper_pos = action[13:14]

        # Compute forward kinematics
        left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
            kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
        )

        # Build 16D ee_pose action:
        # [left_ee_pos (3D), left_ee_quat_xyzw (4D), left_gripper_pos (1D),
        #  right_ee_pos (3D), right_ee_quat_xyzw (4D), right_gripper_pos (1D)]
        ee_pose_action = np.concatenate(
            [
                left_ee_pos,  # 3D
                left_ee_quat_xyzw,  # 4D
                left_gripper_pos,  # 1D
                right_ee_pos,  # 3D
                right_ee_quat_xyzw,  # 4D
                right_gripper_pos,  # 1D
            ]
        )
        assert ee_pose_action.shape == (16,), f"Expected 16D, got {ee_pose_action.shape}"

        ee_pose_actions.append(ee_pose_action)

        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{len(actions)} actions...")

    # Replace action column with ee_pose actions
    df["action"] = ee_pose_actions

    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving converted parquet file to {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"Conversion complete! Converted {len(actions)} actions.")
    print(f"  Input:  {input_path} (14D joint actions)")
    print(f"  Output: {output_path} (16D ee_pose actions)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files from joint space to end-effector pose space"
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        help="Path to input parquet file with 14D joint actions",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to output parquet file with 16D ee_pose actions",
    )

    args = parser.parse_args()

    # Initialize kinematics
    print("Initializing kinematics...")
    kinematics = YamKinematics()

    # Convert
    convert_joint_to_ee_pose(args.input_path, args.output_path, kinematics)


if __name__ == "__main__":
    main()

