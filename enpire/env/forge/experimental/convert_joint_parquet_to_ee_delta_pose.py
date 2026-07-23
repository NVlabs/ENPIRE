# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the actions in a parquet file from joint space to delta EE pose space.

This script reads a LeRobotDataset parquet file with 14D joint actions and
the corresponding 14D observation.state (current joint+gripper proprioception)
and produces 16D **delta** end-effector pose actions:

    delta_pos   = FK(target_joints).pos  - FK(current_joints).pos
    delta_quat  = FK(target_joints).quat * inv(FK(current_joints).quat)

Output layout per timestep (16D):
    [left_delta_pos (3), left_delta_quat_xyzw (4), left_gripper (1),
     right_delta_pos (3), right_delta_quat_xyzw (4), right_gripper (1)]

Gripper values are kept absolute (copied from the original action).

**Important**: ``observation.state`` is preserved as-is (14D joint format) in
the output parquet.  The replay pipeline uses the first frame's
``observation.state`` to calibrate the environment to the recording's initial
joint configuration before applying deltas.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Add project root to path
_TBD_ROOT = Path(__file__).resolve().parents[1]
if str(_TBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TBD_ROOT))

from enpire.env.forge.robot.yam.kinematics import YamKinematics

# Reasonable joint-position range for YAM arms (radians).
# Used as a sanity check, not a hard constraint.
_JOINT_RANGE_RAD = (-2 * np.pi, 2 * np.pi)
# Gripper range [0, 1] (closed to open) with small tolerance
_GRIPPER_RANGE = (-0.1, 1.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quat_delta(
    target_quat_xyzw: np.ndarray, current_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Compute the quaternion that rotates *current* into *target*.

    delta_quat such that: target = delta_quat * current
    => delta_quat = target * inv(current)

    Args:
        target_quat_xyzw:  (4,) target orientation in xyzw order.
        current_quat_xyzw: (4,) current orientation in xyzw order.

    Returns:
        (4,) delta quaternion in xyzw order.
    """
    current_rot = Rotation.from_quat(current_quat_xyzw)  # scipy uses xyzw
    target_rot = Rotation.from_quat(target_quat_xyzw)
    delta_rot = target_rot * current_rot.inv()
    return delta_rot.as_quat()  # xyzw


def _validate_joint_state(
    state: np.ndarray, label: str, idx: int | str = ""
) -> None:
    """Raise ValueError if a 14D state vector doesn't look like joint data."""
    if state.shape != (14,):
        raise ValueError(
            f"{label}[{idx}]: expected shape (14,), got {state.shape}"
        )

    left_jp = state[0:6]
    left_grip = state[6]
    right_jp = state[7:13]
    right_grip = state[13]

    lo, hi = _JOINT_RANGE_RAD
    for name, arr in [("left_joint_pos", left_jp), ("right_joint_pos", right_jp)]:
        if np.any(arr < lo) or np.any(arr > hi):
            raise ValueError(
                f"{label}[{idx}].{name} has values outside [{lo:.2f}, {hi:.2f}]: "
                f"{arr}"
            )

    glo, ghi = _GRIPPER_RANGE
    for name, val in [("left_gripper", left_grip), ("right_gripper", right_grip)]:
        if val < glo or val > ghi:
            raise ValueError(
                f"{label}[{idx}].{name} = {val:.4f} outside [{glo}, {ghi}]"
            )


def _roundtrip_check_frame0(
    state0: np.ndarray,
    action0: np.ndarray,
    delta0: np.ndarray,
    kinematics: YamKinematics,
    pos_tol: float = 1e-3,
    rot_tol_deg: float = 0.5,
) -> None:
    """Verify that applying delta0 to state0 recovers action0 (in EE space).

    This is the critical round-trip test: if the first frame fails, every
    subsequent frame will be offset.
    """
    # FK of state0 -> current EE
    cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = kinematics.forward_kinematics(
        state0[0:6], state0[7:13]
    )

    # Unpack delta0 (16D)
    dl_pos, dl_q = delta0[0:3], delta0[3:7]
    dr_pos, dr_q = delta0[8:11], delta0[11:15]

    # Apply delta -> reconstructed EE
    rec_l_pos = cur_l_pos + dl_pos
    rec_l_rot = Rotation.from_quat(dl_q) * Rotation.from_quat(cur_l_q)
    rec_r_pos = cur_r_pos + dr_pos
    rec_r_rot = Rotation.from_quat(dr_q) * Rotation.from_quat(cur_r_q)

    # FK of action0 -> expected EE
    exp_l_pos, exp_l_q, exp_r_pos, exp_r_q = kinematics.forward_kinematics(
        action0[0:6], action0[7:13]
    )
    exp_l_rot = Rotation.from_quat(exp_l_q)
    exp_r_rot = Rotation.from_quat(exp_r_q)

    # Position check
    l_pos_err = float(np.linalg.norm(rec_l_pos - exp_l_pos))
    r_pos_err = float(np.linalg.norm(rec_r_pos - exp_r_pos))
    if l_pos_err > pos_tol or r_pos_err > pos_tol:
        raise ValueError(
            f"Frame-0 round-trip FAILED: position error "
            f"left={l_pos_err:.6f}m, right={r_pos_err:.6f}m "
            f"(tolerance={pos_tol}m)"
        )

    # Rotation check (angle between reconstructed and expected)
    l_rot_err = float((rec_l_rot.inv() * exp_l_rot).magnitude()) * 180 / np.pi
    r_rot_err = float((rec_r_rot.inv() * exp_r_rot).magnitude()) * 180 / np.pi
    if l_rot_err > rot_tol_deg or r_rot_err > rot_tol_deg:
        raise ValueError(
            f"Frame-0 round-trip FAILED: rotation error "
            f"left={l_rot_err:.4f}°, right={r_rot_err:.4f}° "
            f"(tolerance={rot_tol_deg}°)"
        )

    print(
        f"  Frame-0 round-trip OK: pos_err left={l_pos_err:.6f}m, "
        f"right={r_pos_err:.6f}m; rot_err left={l_rot_err:.4f}°, "
        f"right={r_rot_err:.4f}°"
    )


# ---------------------------------------------------------------------------
# Gear_raw folder → joint parquet (single-episode)
# ---------------------------------------------------------------------------

_GEARRAW_REQUIRED_NPY = (
    "action-left-pos.npy",
    "action-right-pos.npy",
    "left-gripper_pos.npy",
    "right-gripper_pos.npy",
    "left-joint_pos.npy",
    "right-joint_pos.npy",
)


def _gearraw_folder_to_joint_dataframe(ep_dir: Path) -> pd.DataFrame:
    """Load a single gear_raw episode folder and build a joint-format DataFrame.

    Mirrors the state/action layout produced by convert_gearraw_to_lerobot.py
    (14D state, 14D action) but emits a single-episode parquet with no video
    dependency — length is min over the arm/gripper npy arrays only.
    """
    for fname in _GEARRAW_REQUIRED_NPY:
        if not (ep_dir / fname).exists():
            raise FileNotFoundError(
                f"Expected gear_raw file not found: {ep_dir / fname}"
            )

    left_pos = np.load(ep_dir / "action-left-pos.npy").astype(np.float32)
    right_pos = np.load(ep_dir / "action-right-pos.npy").astype(np.float32)
    left_gripper = np.load(ep_dir / "left-gripper_pos.npy").astype(np.float32)
    right_gripper = np.load(ep_dir / "right-gripper_pos.npy").astype(np.float32)
    left_joint = np.load(ep_dir / "left-joint_pos.npy").astype(np.float32)
    right_joint = np.load(ep_dir / "right-joint_pos.npy").astype(np.float32)

    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]

    arm_lens = {
        "action-left-pos": len(left_pos),
        "action-right-pos": len(right_pos),
        "left-gripper_pos": len(left_gripper),
        "right-gripper_pos": len(right_gripper),
        "left-joint_pos": len(left_joint),
        "right-joint_pos": len(right_joint),
    }
    min_len = min(arm_lens.values())
    if len(set(arm_lens.values())) > 1:
        print(
            f"  WARNING: arm logs have mismatched lengths {arm_lens}; "
            f"truncating to {min_len}."
        )
    else:
        print(f"  Arm-log length: {min_len} frames (all logs agree).")

    # 14D state: [left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]
    state = np.concatenate(
        [
            left_joint[:min_len],
            left_gripper[:min_len],
            right_joint[:min_len],
            right_gripper[:min_len],
        ],
        axis=-1,
    )

    # 14D action: [left_pos(6), left_gripper_cmd(1), right_pos(6), right_gripper_cmd(1)]
    # action-*-pos is 7D: cols 0-5 = joint target, col 6 = gripper command
    action = np.concatenate(
        [
            left_pos[:min_len, :6],
            left_pos[:min_len, 6:7],
            right_pos[:min_len, :6],
            right_pos[:min_len, 6:7],
        ],
        axis=-1,
    )

    ts_path = ep_dir / "timestamp.npy"
    if ts_path.exists():
        ts_raw = np.load(ts_path).astype(np.float32)[:min_len]
        ts = (ts_raw - ts_raw[0]).astype(np.float32)
    else:
        ts = (np.arange(min_len, dtype=np.float32) / 30.0)

    action_source: list | None = None
    src_npy = ep_dir / "action-source.npy"
    src_json = ep_dir / "action-source.json"
    if src_npy.exists():
        raw = np.load(src_npy, allow_pickle=True).tolist()
        action_source = [s if isinstance(s, str) else "unknown" for s in raw]
    elif src_json.exists():
        with open(src_json) as f:
            action_source = json.load(f)
    if action_source is not None:
        action_source = action_source[:min_len]
    else:
        action_source = ["unknown"] * min_len

    rows = {
        "index": list(range(min_len)),
        "episode_index": [0] * min_len,
        "frame_index": list(range(min_len)),
        "timestamp": ts.tolist(),
        "task_index": [0] * min_len,
        "next.done": [False] * (min_len - 1) + [True],
        "observation.state": [state[i].tolist() for i in range(min_len)],
        "action": [action[i].tolist() for i in range(min_len)],
        "action_source": action_source,
    }
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_joint_to_ee_delta_pose(
    input_path: Path, output_path: Path, kinematics: YamKinematics
) -> None:
    """Convert 14D joint actions + observation.state to 16D delta EE pose actions.

    Args:
        input_path:  Parquet file with 14D ``action`` and 14D ``observation.state``.
        output_path: Parquet file that will contain the 16D delta EE pose actions.
        kinematics:  :class:`YamKinematics` instance for FK.
    """
    print(f"Loading parquet file from {input_path}")
    df = pd.read_parquet(input_path)

    # ------------------------------------------------------------------
    # Pre-validation
    # ------------------------------------------------------------------
    if "action" not in df.columns:
        raise ValueError("Parquet is missing the 'action' column.")
    if "observation.state" not in df.columns:
        raise ValueError(
            "Parquet is missing the 'observation.state' column. "
            "This column is required for delta conversion (it provides the "
            "current joint state at each timestep)."
        )

    actions = np.stack(df["action"].values)             # (T, 14)
    states = np.stack(df["observation.state"].values)    # (T, 14)

    # Store as list of 1D arrays so pandas has one value per row (not a 2D column)
    df["joint_actions"] = [actions[i].copy() for i in range(len(actions))]

    if actions.shape[1] != 14:
        raise ValueError(
            f"Expected 14D joint actions, got {actions.shape[1]}D. "
            "This converter only works on joint-space parquet files."
        )
    if states.shape[1] != 14:
        raise ValueError(
            f"Expected 14D observation.state (joint format), got {states.shape[1]}D. "
            "The observation.state column must be in joint format "
            "[left_joint_pos(6), left_gripper(1), right_joint_pos(6), right_gripper(1)]."
        )

    # Validate first and last frame state vectors look like joint data
    _validate_joint_state(states[0], "observation.state", 0)
    _validate_joint_state(states[-1], "observation.state", len(states) - 1)
    _validate_joint_state(actions[0], "action", 0)
    print(
        "  Validated: observation.state and action are 14D joint format "
        "with reasonable value ranges."
    )

    # Print first-frame state for visual inspection
    print("  First-frame observation.state (14D joint):")
    print(f"    left_joint_pos  = {states[0, 0:6]}")
    print(f"    left_gripper    = {states[0, 6]:.4f}")
    print(f"    right_joint_pos = {states[0, 7:13]}")
    print(f"    right_gripper   = {states[0, 13]:.4f}")

    T = len(actions)
    print(f"Converting {T} timesteps from joint space to delta EE pose space ...")

    # ------------------------------------------------------------------
    # Conversion loop
    # ------------------------------------------------------------------
    delta_actions = []
    for i in range(T):
        # --- unpack current state ---
        cur_left_jp = states[i, 0:6]
        cur_right_jp = states[i, 7:13]

        # --- unpack target action ---
        tgt_left_jp = actions[i, 0:6]
        tgt_left_grip = actions[i, 6:7]
        tgt_right_jp = actions[i, 7:13]
        tgt_right_grip = actions[i, 13:14]

        # FK for current state
        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = kinematics.forward_kinematics(
            cur_left_jp, cur_right_jp
        )
        # FK for target action
        tgt_l_pos, tgt_l_q, tgt_r_pos, tgt_r_q = kinematics.forward_kinematics(
            tgt_left_jp, tgt_right_jp
        )

        # Delta position (simple subtraction)
        delta_l_pos = tgt_l_pos - cur_l_pos
        delta_r_pos = tgt_r_pos - cur_r_pos

        # Delta quaternion: delta * current = target => delta = target * inv(current)
        delta_l_q = _quat_delta(
            target_quat_xyzw=tgt_l_q, current_quat_xyzw=cur_l_q
        )
        delta_r_q = _quat_delta(
            target_quat_xyzw=tgt_r_q, current_quat_xyzw=cur_r_q
        )

        # 16D output
        delta_action = np.concatenate([
            delta_l_pos,       # 3
            delta_l_q,         # 4
            tgt_left_grip,     # 1  (absolute)
            delta_r_pos,       # 3
            delta_r_q,         # 4
            tgt_right_grip,    # 1  (absolute)
        ])
        assert delta_action.shape == (16,), f"Expected 16D, got {delta_action.shape}"
        delta_actions.append(delta_action)

        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{T} timesteps ...")

    # ------------------------------------------------------------------
    # Post-validation: round-trip check on frame 0
    # ------------------------------------------------------------------
    print("Running frame-0 round-trip sanity check ...")
    _roundtrip_check_frame0(states[0], actions[0], delta_actions[0], kinematics)

    # ------------------------------------------------------------------
    # Save — observation.state is preserved as-is (14D joint format)
    # ------------------------------------------------------------------
    df["action"] = delta_actions

    # Verify observation.state is unchanged in output
    out_states = np.stack(df["observation.state"].values)
    if not np.array_equal(out_states, states):
        raise RuntimeError(
            "BUG: observation.state was accidentally modified during conversion!"
        )
    print(
        f"  Verified: observation.state preserved in output "
        f"({out_states.shape[1]}D joint format, {len(out_states)} frames)."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving converted parquet to {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"Done!  {T} timesteps converted.")
    print(f"  Input:  {input_path}  (14D joint actions)")
    print(f"  Output: {output_path} (16D delta EE pose actions)")
    print(
        "  observation.state: preserved as 14D joint format "
        "(first-frame calibration for replay)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet from joint space to delta EE pose space. "
        "--input_path may be either an existing joint-format parquet file, or a "
        "gear_raw episode directory (npy layout); in the latter case the folder "
        "is first packed into a single-episode joint parquet, saved at "
        "--joint_parquet_path, then converted to delta EE pose."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Input joint parquet file OR gear_raw episode directory.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output parquet with 16D delta EE pose actions",
    )
    parser.add_argument(
        "--joint_parquet_path",
        type=Path,
        default=None,
        help="When --input_path is a directory, the intermediate joint parquet "
        "is saved here. Defaults to <output_path stem>_joint.parquet next to "
        "--output_path.",
    )
    args = parser.parse_args()

    print("Initializing kinematics ...")
    kinematics = YamKinematics()

    if args.input_path.is_dir():
        joint_parquet_path = args.joint_parquet_path
        if joint_parquet_path is None:
            joint_parquet_path = args.output_path.with_name(
                args.output_path.stem + "_joint.parquet"
            )
        print(
            f"Input is a directory — packing gear_raw folder into joint parquet:"
            f"\n  src: {args.input_path}\n  dst: {joint_parquet_path}"
        )
        df = _gearraw_folder_to_joint_dataframe(args.input_path)
        joint_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(joint_parquet_path, index=False)
        print(
            f"  Saved joint parquet: {joint_parquet_path} "
            f"({len(df)} frames, 14D state + 14D action)."
        )
        parquet_input = joint_parquet_path
    else:
        parquet_input = args.input_path

    convert_joint_to_ee_delta_pose(parquet_input, args.output_path, kinematics)


if __name__ == "__main__":
    main()
