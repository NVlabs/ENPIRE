# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the actions in a parquet file from joint space to delta joint space. 

This script reads a LeRobotDataset parquet file with 14D joint actions and
converts them to delta joint actions (differences between consecutive frames).
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to path
_TBD_ROOT = Path(__file__).resolve().parents[1]
if str(_TBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TBD_ROOT))


def convert_joint_to_delta_joint(
    input_path: Path, output_path: Path
) -> None:
    """Convert 14D joint actions to delta joint actions.

    Computes the delta required to move from the current state to the target action:
    - delta[t] = target_action[t] - current_state[t]

    Args:
        input_path: Path to input parquet file with 14D joint actions
        output_path: Path to output parquet file with 14D delta joint actions
    """
    print(f"Loading parquet file from {input_path}")
    df = pd.read_parquet(input_path)

    # Extract actions and states
    # actions: The absolute joint position targets (T, 14)
    # state: The current joint positions from proprioception (T, 14)
    actions = np.stack(df["action"].values)
    state = np.stack(df["observation.state"].values)
 
    if actions.shape[1] != 14:
        raise ValueError(
            f"Expected 14D actions, but got {actions.shape[1]}D."
        )

    print(f"Converting {len(actions)} actions from joint space to delta joint space...")

    # --- VECTORIZED CALCULATION ---
    # No loop needed. NumPy handles element-wise subtraction for the whole array.
    # Logic: To get to 'action' (target) from 'state' (current), move by 'delta'
    delta_actions = actions - state
    
    # ------------------------------
    # Convert back to list of arrays for pandas storage
    # (Pandas Parquet often expects a list of arrays for array-columns)
    df["action"] = list(delta_actions)

    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving converted parquet file to {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"Conversion complete! Converted {len(actions)} actions.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files from joint space to delta joint space"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input parquet file with 14D joint actions",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output parquet file with 14D delta joint actions",
    )

    args = parser.parse_args()

    # Convert
    convert_joint_to_delta_joint(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

