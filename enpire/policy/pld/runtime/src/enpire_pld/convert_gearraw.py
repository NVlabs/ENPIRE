"""Convert forge gearraw episodes to PLD-Lite demo .pkl files.

Forge's data-collection pipeline (`launch.py --mode=data_collection --use-fello` →
`run_data_collection.py` + `record_episode_wrapper.py`) writes episodes as a
directory of NPY + MP4 files (the "gearraw" layout). This script converts those
episodes into the flat `List[Dict]` pickle format consumed by PLD-Lite's demo
loader at ``scripts/pld_lite/train_delta_rlpd.py`` (around line 834).

Output format (one .pkl per episode when ``--split-per-episode``, else one
combined .pkl)::

    List[Dict]  where each Dict is a per-step transition:
        {
            "observations":      {"state": (1, D) float32,
                                  "top_camera_image":   (1, 256, 256, 3) uint8,
                                  "left_camera_image":  (1, 256, 256, 3) uint8,
                                  "right_camera_image": (1, 256, 256, 3) uint8},
            "next_observations": {...same keys...},
            "actions":           (14,) float32 in RAW env action space,
            "rewards":           float  (1.0 on terminal frame, else 0.0),
            "masks":             float  (0.0 on terminal, else 1.0),
            "dones":             float  (1.0 on terminal, else 0.0),
            "infos":             {"grasp_penalty": 0.0},
        }

The leading singleton axis on every observation value matches what PLD's
online replay buffer stores: the actor runs inside ``ChunkingWrapper`` with
``obs_horizon=1`` (see ``configs/env/remote_deployment.yaml:51`` and
``serl_launcher/wrappers/chunking.py:stack_obs``), which stacks each obs key
along a new leading dim. Demos must live in the same shape so the 50/50
demo/online batch mix (``train_delta_rlpd.py:479-497``) stacks cleanly.

State and action ordering
-------------------------
The default remote-YAM PLD path uses 20-D end-effector rot6d proprio:
``[left_xyz(3), left_rot6d(6), left_gripper(1), right_xyz(3),
right_rot6d(6), right_gripper(1)]`` from ``state-eef-rot6d.npy``. Its action is
the matching 20-D delta-EEF rot6d command from ``action-delta-eef-rot6d.npy``.

Legacy joint episodes use the 14-D layout
``[left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]`` --
matching ``RemoteDeploymentEnv`` joint defaults.

Actions are emitted in forge's RAW env space (joint radians + gripper in
``[0, 1]``). The learner's ``inv_action_scaling_fn`` handles the mapping into
the SAC normalized ``[-1, 1]`` delta space at load time
(``train_delta_rlpd.py:850``).

Image size
----------
Images default to 256×256 to match what the live replay buffer stores.
At runtime, ``yam_env.utils.env.observation_utils.SERLObsWrapper`` resizes
every camera frame to ``env.image_height`` × ``env.image_width`` before the
actor inserts the transition. Demos must live in the same shape so the 50/50
demo/online batch mix stacks cleanly. Override ``--image-size`` only if it
matches the runtime env image config.

Usage
-----
    uv run python scripts/data/convert_gearraw_to_pld_demo.py \\
        --input-root $YAM_RAW_PATH/<operator>_<task>_<timestamp>-YAM-01 \\
        --output-dir /data/pld_demos/<task> \\
        --split-per-episode

Then pass ``task.demo_path=/data/pld_demos/<task>`` to the learner -- PLD's
``pkl_browser`` globs the directory for ``*.pkl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _load_video_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _truncate_to_min_length(
    items: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    lengths = [len(item) for item in items if item is not None]
    if not lengths:
        return items, 0
    min_len = min(lengths)
    trimmed = [item[:min_len] if item is not None else None for item in items]
    return trimmed, min_len


# Match the runtime resize applied by yam_env.SERLObsWrapper (see
# flywheel/yam_env/yam_env/utils/env/observation_utils.py:92). Keeping the
# demo buffer's image shape aligned with the online replay buffer is a hard
# requirement -- PLD samples both in the same batch (train_delta_rlpd.py:479).
_RUNTIME_IMAGE_SIZE: tuple[int, int] = (256, 256)
_CAMERA_FILES: dict[str, str] = {
    "top_camera_image": "top_camera-images-rgb.mp4",
    "left_camera_image": "left_camera-images-rgb.mp4",
    "right_camera_image": "right_camera-images-rgb.mp4",
    "left_wrist_camera_image": "left_wrist_camera-images-rgb.mp4",
    "wrist_camera_image": "wrist_camera-images-rgb.mp4",
}
_DEFAULT_IMAGE_KEYS: tuple[str, ...] = (
    "top_camera_image",
    "left_camera_image",
    "right_camera_image",
)
_PROPRIO_FILES: dict[str, str] = {
    "state_eef_rot6d": "state-eef-rot6d.npy",
}
_DEFAULT_DELTA_EEF_ROT6D_PROPRIO_KEYS: tuple[str, ...] = ("state_eef_rot6d",)
_COMMON_REQUIRED_FILES: tuple[str, ...] = (
    "left-gripper_pos.npy",
    "right-gripper_pos.npy",
    "left-joint_pos.npy",
    "right-joint_pos.npy",
)
_JOINT_ACTION_FILES: tuple[str, ...] = ("action-left-pos.npy", "action-right-pos.npy")
_DELTA_EEF_ACTION_FILE = "action-delta-eef.npy"
_DELTA_EEF_ROT6D_ACTION_FILE = "action-delta-eef-rot6d.npy"
_EEF_ROT6D_STATE_FILE = "state-eef-rot6d.npy"
_PLD_DELTA_EEF_STATUS_FILE = "pld_delta_eef_status.json"
_MISSING_ROT6D_STATUS_WARNED_ROOTS: set[str] = set()


def _normalise_image_keys(image_keys: Sequence[str] | None) -> tuple[str, ...]:
    if image_keys is None:
        return _DEFAULT_IMAGE_KEYS
    normalised: list[str] = []
    for key in image_keys:
        key = str(key)
        if key not in _CAMERA_FILES:
            raise ValueError(
                f"Unsupported image key {key!r}; expected one of {sorted(_CAMERA_FILES)}"
            )
        if key not in normalised:
            normalised.append(key)
    if not normalised:
        raise ValueError("At least one image key is required for PLD conversion")
    return tuple(normalised)


def _normalise_proprio_keys(
    proprio_keys: Sequence[str] | None, *, action_repr: str
) -> tuple[str, ...] | None:
    if proprio_keys is None:
        return (
            _DEFAULT_DELTA_EEF_ROT6D_PROPRIO_KEYS
            if action_repr in ("delta_eef_rot6d", "delta_eef_pos")
            else None
        )
    normalised: list[str] = []
    for key in proprio_keys:
        key = str(key)
        if key not in _PROPRIO_FILES:
            raise ValueError(
                f"Unsupported disk proprio key {key!r}; expected one of {sorted(_PROPRIO_FILES)}"
            )
        if key not in normalised:
            normalised.append(key)
    if not normalised:
        raise ValueError("At least one proprio key is required when configured")
    return tuple(normalised)


def _require_rot6d_status_metadata(dirpath: str, action_shape: tuple[int, ...]) -> None:
    """Validate the Forge-side status file for strict rot6d PLD ingestion.

    Teleop captures do not write this file; a missing file is treated as a
    warning rather than an error so live ingestion works for both teleop and
    RL-rollout episodes.
    """
    status_path = os.path.join(dirpath, _PLD_DELTA_EEF_STATUS_FILE)
    if not os.path.exists(status_path):
        episode_root = os.path.dirname(os.path.abspath(dirpath))
        if episode_root not in _MISSING_ROT6D_STATUS_WARNED_ROOTS:
            _MISSING_ROT6D_STATUS_WARNED_ROOTS.add(episode_root)
            logger.warning(
                "%s not found under %s; suppressing further missing-status "
                "warnings for this root (normal for teleop episodes). "
                "First missing episode: %s",
                _PLD_DELTA_EEF_STATUS_FILE,
                episode_root,
                dirpath,
            )
        return
    try:
        with open(status_path, "r") as f:
            status = json.load(f)
    except Exception as exc:
        raise ValueError(f"Could not parse {status_path}: {exc}") from exc
    if not isinstance(status, dict):
        raise ValueError(f"{status_path} must contain a JSON object")

    expected = {
        "pld_ready": True,
        "action_repr": "delta_eef_rot6d",
        "required_file": _DELTA_EEF_ROT6D_ACTION_FILE,
        "rotation_repr": "zhou_6d_first_two_columns",
        "rot6d_order": "[R[:,0], R[:,1]]",
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise ValueError(f"{status_path} has {key}={status.get(key)!r}; expected {value!r}")
    recorded_shape = status.get("action_delta_eef_shape")
    if recorded_shape is None:
        raise ValueError(f"{status_path} must declare action_delta_eef_shape")
    if list(recorded_shape) != list(action_shape):
        raise ValueError(
            f"{status_path} action_delta_eef_shape={recorded_shape} does not match "
            f"{_DELTA_EEF_ROT6D_ACTION_FILE} shape={list(action_shape)}"
        )


def _load_first_npz_array(
    archive: np.lib.npyio.NpzFile, names: tuple[str, ...]
) -> np.ndarray | None:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    return None


def _load_reward_done_labels(
    dirpath: str, length: int, *, required: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load optional timestep-aligned reward/done labels.

    The saved labels are indexed by recorded action timestep, so transition t
    receives exactly reward[t] and dones[t]. The loader does not synthesize or
    shift terminal labels when done labels are present.
    """
    reward_npz_path = os.path.join(dirpath, "reward.npz")
    if os.path.exists(reward_npz_path):
        with np.load(reward_npz_path) as payload:
            rewards_raw = _load_first_npz_array(payload, ("rewards", "reward", "r"))
            dones_raw = _load_first_npz_array(
                payload,
                ("dones", "done", "terminals", "terminal", "is_terminal"),
            )
        if rewards_raw is None:
            raise KeyError(f"{reward_npz_path} must contain one of: rewards, reward, r")
        rewards = np.asarray(rewards_raw, dtype=np.float32).reshape(-1)
        if dones_raw is None:
            dones = np.zeros(length, dtype=np.bool_)
            if length >= 2:
                # reward.npz is the source of reward truth. If this newer label
                # file omits terminal flags, keep the episode boundary explicit
                # on the final emitted transition without changing rewards.
                dones[length - 2] = True
        else:
            dones = np.asarray(dones_raw, dtype=np.bool_).reshape(-1)
        if rewards.shape[0] != length or dones.shape[0] != length:
            raise ValueError(
                f"reward.npz labels must match recorded length {length}; got "
                f"reward={rewards.shape}, dones={dones.shape} in {dirpath}"
            )
        return rewards, dones

    reward_path = os.path.join(dirpath, "reward.npy")
    dones_path = os.path.join(dirpath, "dones.npy")
    reward_exists = os.path.exists(reward_path)
    dones_exists = os.path.exists(dones_path)
    if reward_exists != dones_exists:
        missing = "dones.npy" if reward_exists else "reward.npy"
        raise FileNotFoundError(f"{dirpath} has only one reward/done label file; missing {missing}")
    if not reward_exists:
        if required:
            raise FileNotFoundError(
                f"Expected reward.npz or reward.npy and dones.npy for strict delta_eef_rot6d "
                f"PLD ingestion; refusing to synthesize post-hoc labels: {dirpath}"
            )
        return None, None

    rewards = np.asarray(np.load(reward_path), dtype=np.float32).reshape(-1)
    dones = np.asarray(np.load(dones_path), dtype=np.bool_).reshape(-1)
    if rewards.shape[0] != length or dones.shape[0] != length:
        raise ValueError(
            f"reward.npy/dones.npy must match recorded length {length}; got "
            f"reward={rewards.shape}, dones={dones.shape} in {dirpath}"
        )
    return rewards, dones


def _require_length(
    name: str, value: Sequence[Any] | np.ndarray, expected_length: int, dirpath: str
) -> None:
    if len(value) != expected_length:
        raise ValueError(
            f"Strict delta_eef_rot6d PLD requires {name} length "
            f"{expected_length}; got {len(value)} in {dirpath}"
        )


def _find_episode_dirs(root: str, image_keys: Sequence[str] | None = None) -> list[str]:
    """Find PLD-compatible gearraw episodes.

    Legacy joint episodes carry ``action-left-pos.npy`` and
    ``action-right-pos.npy``. Current PLD delta-EE episodes may instead carry
    only ``action-delta-eef-rot6d.npy``. Legacy quaternion delta-EE episodes
    are accepted only when explicitly requested.
    """
    image_keys = _normalise_image_keys(image_keys)
    camera_files = {_CAMERA_FILES[key] for key in image_keys}
    episode_dirs: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        if not camera_files.issubset(files):
            continue
        has_joint = all(name in files for name in _JOINT_ACTION_FILES)
        has_delta_eef_rot6d = _DELTA_EEF_ROT6D_ACTION_FILE in files
        has_delta_eef = _DELTA_EEF_ACTION_FILE in files
        if has_delta_eef_rot6d or has_delta_eef:
            # EEF-repr episodes: require EEF state instead of joint/gripper files.
            if _EEF_ROT6D_STATE_FILE in files:
                episode_dirs.append(dirpath)
        elif has_joint and all(name in files for name in _COMMON_REQUIRED_FILES):
            episode_dirs.append(dirpath)
    episode_dirs.sort()
    return episode_dirs


def _resize_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    """Bilinear resize each frame to (H, W). No-op if already at target size."""
    out: list[np.ndarray] = []
    for f in frames:
        if f.shape[:2] == size:
            out.append(f)
        else:
            out.append(cv2.resize(f, (size[1], size[0])))  # cv2 takes (W, H)
    return out


def _build_state(
    left_joint: np.ndarray,
    left_gripper: np.ndarray,
    right_joint: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    """Concat proprio in PLD runtime order: [jL(6), gL(1), jR(6), gR(1)]."""
    return np.concatenate([left_joint, left_gripper, right_joint, right_gripper], axis=-1).astype(
        np.float32
    )


def _build_action(
    left_joint_act: np.ndarray,
    left_gripper_act: np.ndarray,
    right_joint_act: np.ndarray,
    right_gripper_act: np.ndarray,
) -> np.ndarray:
    """Concat action in PLD runtime order: [jL(6), gL(1), jR(6), gR(1)]."""
    return np.concatenate(
        [left_joint_act, left_gripper_act, right_joint_act, right_gripper_act], axis=-1
    ).astype(np.float32)


def _load_episode_as_transitions(
    dirpath: str,
    filter_noop: bool = False,
    noop_eps: float = 1e-6,
    image_size: tuple[int, int] = _RUNTIME_IMAGE_SIZE,
    prefer_delta_eef: bool = False,
    action_repr: str = "joint",
    image_keys: Sequence[str] | None = None,
    proprio_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read a single gearraw episode dir; return flat list of PLD transitions.

    If ``prefer_delta_eef`` is True and ``action_repr`` is ``delta_eef_rot6d``,
    this strictly requires ``action-delta-eef-rot6d.npy`` ([T, 20]) and
    ``state-eef-rot6d.npy`` ([T, 20]) and never falls back to joint
    actions/proprio. If ``action_repr`` is ``delta_eef_quat``, it reads the
    legacy ``action-delta-eef.npy`` ([T, 16]).
    """
    if action_repr is None or str(action_repr).strip() == "":
        raise ValueError(
            "action_repr must be explicit: 'joint', 'delta_eef_quat', "
            "'delta_eef_rot6d', or 'delta_eef_pos'."
        )
    action_repr = str(action_repr)
    if action_repr not in ("joint", "delta_eef_quat", "delta_eef_rot6d", "delta_eef_pos"):
        raise ValueError(f"Unsupported action_repr={action_repr!r}")
    if prefer_delta_eef and action_repr == "joint":
        raise ValueError(
            "prefer_delta_eef=True requires explicit action_repr="
            "'delta_eef_quat', 'delta_eef_rot6d', or 'delta_eef_pos'; "
            "refusing to infer from files."
        )
    image_keys = _normalise_image_keys(image_keys)
    proprio_keys = _normalise_proprio_keys(proprio_keys, action_repr=action_repr)
    # Actions. Strictly load the explicitly configured delta-EE representation
    # when ``prefer_delta_eef`` is set; the current PLD path is 20-D rot6d.
    delta_eef_path = os.path.join(dirpath, _DELTA_EEF_ACTION_FILE)
    delta_eef_rot6d_path = os.path.join(dirpath, _DELTA_EEF_ROT6D_ACTION_FILE)
    action_left_path = os.path.join(dirpath, "action-left-pos.npy")
    action_right_path = os.path.join(dirpath, "action-right-pos.npy")
    use_delta_eef = bool(prefer_delta_eef)
    use_eef_rot6d_state = use_delta_eef and action_repr in (
        "delta_eef_rot6d",
        "delta_eef_pos",
    )
    strict_rot6d = use_eef_rot6d_state
    if use_delta_eef:
        if action_repr in ("delta_eef_rot6d", "delta_eef_pos"):
            expected_path = delta_eef_rot6d_path
            expected_dim = 20
        else:
            expected_path = delta_eef_path
            expected_dim = 16
        if not os.path.exists(expected_path):
            raise FileNotFoundError(
                f"Expected {os.path.basename(expected_path)} for "
                f"prefer_delta_eef=True action_repr={action_repr}; no fallback "
                f"to joint actions is allowed in strict PLD mode: {dirpath}"
            )
        action_delta_eef = _ensure_2d(np.load(expected_path))
        if action_delta_eef.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected {os.path.basename(expected_path)} to be [T, {expected_dim}], got "
                f"{action_delta_eef.shape} in {dirpath}"
            )
        if action_repr in ("delta_eef_rot6d", "delta_eef_pos"):
            _require_rot6d_status_metadata(dirpath, tuple(action_delta_eef.shape))
        action_left = None
        action_right = None
    else:
        # Forge stores action-{left,right}-pos.npy as [T, 7] = [joint(6), grip(1)]
        if not (os.path.exists(action_left_path) and os.path.exists(action_right_path)):
            raise FileNotFoundError(
                f"{dirpath} has no legacy joint action files. Re-run with "
                "prefer_delta_eef=True if this is a delta-EE PLD episode."
            )
        action_left = _ensure_2d(np.load(action_left_path))
        action_right = _ensure_2d(np.load(action_right_path))
        if action_left.shape[-1] != 7 or action_right.shape[-1] != 7:
            raise ValueError(
                f"Expected action-{{left,right}}-pos.npy to be [T, 7] (6 joint + 1 grip), "
                f"got left={action_left.shape} right={action_right.shape} in {dirpath}"
            )
        action_delta_eef = None

    # Proprio (obs). The current strict PLD path must consume the same 20-D
    # end-effector rot6d state used by the live remote env; no joint fallback.
    if use_eef_rot6d_state:
        if proprio_keys != _DEFAULT_DELTA_EEF_ROT6D_PROPRIO_KEYS:
            raise ValueError(
                "delta_eef_rot6d disk ingestion currently supports "
                f"proprio_keys={list(_DEFAULT_DELTA_EEF_ROT6D_PROPRIO_KEYS)}; "
                f"got {list(proprio_keys or [])}"
            )
        state_eef_rot6d_path = os.path.join(dirpath, _PROPRIO_FILES["state_eef_rot6d"])
        if not os.path.exists(state_eef_rot6d_path):
            raise FileNotFoundError(
                f"Expected {_EEF_ROT6D_STATE_FILE} for prefer_delta_eef=True "
                f"action_repr={action_repr}; no fallback to 14-D joint proprio "
                f"is allowed in strict PLD mode: {dirpath}"
            )
        state_eef_rot6d = _ensure_2d(np.load(state_eef_rot6d_path))
        if state_eef_rot6d.shape[-1] != 20:
            raise ValueError(
                f"Expected {_EEF_ROT6D_STATE_FILE} to be [T, 20], got "
                f"{state_eef_rot6d.shape} in {dirpath}"
            )
        if use_delta_eef and action_repr in ("delta_eef_rot6d", "delta_eef_pos"):
            if state_eef_rot6d.shape[0] != action_delta_eef.shape[0]:
                raise ValueError(
                    f"Strict delta_eef_rot6d PLD requires matching T for "
                    f"{_EEF_ROT6D_STATE_FILE} and {_DELTA_EEF_ROT6D_ACTION_FILE}; "
                    f"got state={state_eef_rot6d.shape}, action={action_delta_eef.shape} "
                    f"in {dirpath}"
                )
        left_joint = right_joint = left_gripper = right_gripper = None
    else:
        state_eef_rot6d = None
        left_joint = _ensure_2d(np.load(os.path.join(dirpath, "left-joint_pos.npy")))
        right_joint = _ensure_2d(np.load(os.path.join(dirpath, "right-joint_pos.npy")))
        left_gripper = _ensure_2d(np.load(os.path.join(dirpath, "left-gripper_pos.npy")))
        right_gripper = _ensure_2d(np.load(os.path.join(dirpath, "right-gripper_pos.npy")))

    # Videos — resize to the runtime image size so demos match online buffer shape.
    frames_by_key = {
        key: _resize_frames(
            _load_video_frames(os.path.join(dirpath, _CAMERA_FILES[key])),
            image_size,
        )
        for key in image_keys
    }

    if strict_rot6d:
        expected_length = int(action_delta_eef.shape[0])
        _require_length(
            _DELTA_EEF_ROT6D_ACTION_FILE,
            action_delta_eef,
            expected_length,
            dirpath,
        )
        _require_length(_EEF_ROT6D_STATE_FILE, state_eef_rot6d, expected_length, dirpath)
        for key in image_keys:
            _require_length(
                _CAMERA_FILES[key],
                frames_by_key[key],
                expected_length,
                dirpath,
            )
        rewards, dones = _load_reward_done_labels(
            dirpath,
            expected_length,
            required=True,
        )

    truncate_inputs = [frames_by_key[key] for key in image_keys]
    if use_eef_rot6d_state:
        truncate_inputs.append(state_eef_rot6d)
    else:
        truncate_inputs.extend([left_joint, right_joint, left_gripper, right_gripper])
    if use_delta_eef:
        truncate_inputs.append(action_delta_eef)
    else:
        truncate_inputs.extend([action_left, action_right])
    if strict_rot6d:
        length = expected_length
    else:
        arrays, length = _truncate_to_min_length(truncate_inputs)
        frame_arrays = arrays[: len(image_keys)]
        frames_by_key = dict(zip(image_keys, frame_arrays))
        rest = arrays[len(image_keys) :]
        if use_eef_rot6d_state:
            state_eef_rot6d = rest.pop(0)
        else:
            left_joint, right_joint, left_gripper, right_gripper = rest[:4]
            rest = rest[4:]
        if use_delta_eef:
            action_delta_eef = rest[0]
        else:
            action_left, action_right = rest

    if length < 2:
        # Need at least two frames to form one transition (obs → next_obs).
        return []
    if not strict_rot6d:
        rewards, dones = _load_reward_done_labels(dirpath, length, required=False)

    # Build per-frame obs dicts once so we can index both obs and next_obs.
    # Every value gets a leading singleton axis so the shape matches what the
    # online replay buffer sees (ChunkingWrapper with obs_horizon=1).
    obs_list: list[dict[str, np.ndarray]] = []
    for t in range(length):
        if use_eef_rot6d_state:
            state_flat = np.asarray(state_eef_rot6d[t], dtype=np.float32)
        else:
            state_flat = _build_state(
                left_joint[t], left_gripper[t], right_joint[t], right_gripper[t]
            )
        obs = {"state": state_flat[None, ...]}
        for key in image_keys:
            obs[key] = frames_by_key[key][t][None, ...]  # (1, H, W, 3)
        obs_list.append(obs)

    transitions: list[dict[str, Any]] = []
    last_t = length - 2  # index of the final transition (T-1 transitions total)
    for t in range(length - 1):
        if use_delta_eef:
            action = action_delta_eef[t].astype(np.float32)
        else:
            action = _build_action(
                action_left[t, :6],
                action_left[t, 6:7],
                action_right[t, :6],
                action_right[t, 6:7],
            )
        if filter_noop and float(np.linalg.norm(action)) <= noop_eps:
            continue
        is_terminal = t == last_t
        if rewards is not None and dones is not None:
            label_idx = t
            reward = float(rewards[label_idx])
            done = bool(dones[label_idx])
            if is_terminal and length >= 2:
                # Forge records labels per raw action timestep. The final raw
                # action has no recorded next_observation, so the converter
                # emits only T-1 transitions. Preserve terminal labels written
                # on that final raw frame by folding them into the last emitted
                # transition instead of silently dropping them.
                final_label_idx = length - 1
                if final_label_idx != label_idx and bool(dones[final_label_idx]):
                    reward = max(reward, float(rewards[final_label_idx]))
                    done = True
        else:
            reward = 1.0 if is_terminal else 0.0
            done = bool(is_terminal)
        transitions.append(
            {
                "observations": obs_list[t],
                "next_observations": obs_list[t + 1],
                "actions": action,
                "rewards": reward,
                "masks": 0.0 if done else 1.0,
                "dones": float(done),
                "infos": {"grasp_penalty": 0.0},
            }
        )
    return transitions


def convert_gearraw_to_pld_demo(
    input_root: str,
    output_dir: str,
    split_per_episode: bool = False,
    filter_noop: bool = False,
    noop_eps: float = 1e-6,
    image_size: tuple[int, int] = _RUNTIME_IMAGE_SIZE,
    prefer_delta_eef: bool = False,
    action_repr: str = "joint",
    image_keys: Sequence[str] | None = None,
    proprio_keys: Sequence[str] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image_keys = _normalise_image_keys(image_keys)
    proprio_keys = _normalise_proprio_keys(proprio_keys, action_repr=action_repr)
    episode_dirs = _find_episode_dirs(input_root, image_keys=image_keys)
    if not episode_dirs:
        raise FileNotFoundError(f"No valid gearraw episodes found under: {input_root}")

    total_transitions = 0
    combined: list[dict[str, Any]] = []
    for ep_idx, ep_dir in enumerate(episode_dirs):
        transitions = _load_episode_as_transitions(
            ep_dir,
            filter_noop=filter_noop,
            noop_eps=noop_eps,
            image_size=image_size,
            prefer_delta_eef=prefer_delta_eef,
            action_repr=action_repr,
            image_keys=image_keys,
            proprio_keys=proprio_keys,
        )
        if not transitions:
            print(f"[skip] {ep_dir}: no transitions (episode too short?)")
            continue
        total_transitions += len(transitions)

        if split_per_episode:
            # Name after the episode directory's trailing path components for traceability.
            ep_name = Path(ep_dir).name
            out_path = os.path.join(output_dir, f"ep_{ep_idx:06d}_{ep_name}.pkl")
            with open(out_path, "wb") as f:
                pkl.dump(transitions, f)
            print(f"[wrote] {out_path}  ({len(transitions)} transitions)")
        else:
            combined.extend(transitions)

    if not split_per_episode:
        out_path = os.path.join(output_dir, "pld_demo.pkl")
        with open(out_path, "wb") as f:
            pkl.dump(combined, f)
        print(f"[wrote] {out_path}  ({len(combined)} transitions)")

    print(
        f"[done] {len(episode_dirs)} episode dirs scanned, "
        f"{total_transitions} total transitions emitted."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert forge gearraw episodes to PLD-Lite demo .pkl format."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory holding forge gearraw episode subdirs "
        "(e.g. $YAM_RAW_PATH/<operator>_<task>_<ts>-YAM-01/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for .pkl files (created if missing).",
    )
    parser.add_argument(
        "--split-per-episode",
        action="store_true",
        help="Emit one .pkl per episode (preferred; PLD's pkl_browser globs the dir). "
        "Without this flag, a single combined pld_demo.pkl is written.",
    )
    parser.add_argument(
        "--filter-noop",
        action="store_true",
        help="Drop transitions whose action L2-norm is <= --noop-eps.",
    )
    parser.add_argument(
        "--noop-eps",
        type=float,
        default=1e-6,
        help="Threshold for --filter-noop (L2 norm of the raw 14-D action).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=list(_RUNTIME_IMAGE_SIZE),
        help=(
            "Target camera resolution (H W). Default 256 256 matches the default "
            "env.image_height/env.image_width used by yam_env.SERLObsWrapper."
        ),
    )
    parser.add_argument("--prefer-delta-eef", action="store_true")
    parser.add_argument(
        "--action-repr",
        choices=("joint", "delta_eef_quat", "delta_eef_rot6d", "delta_eef_pos"),
        default="joint",
    )
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=list(_DEFAULT_IMAGE_KEYS),
        choices=tuple(_CAMERA_FILES),
        help=(
            "Observation image keys to emit. Defaults to all legacy cameras; "
            "PLD live ingestion passes cfg.env.image_keys."
        ),
    )
    parser.add_argument(
        "--proprio-keys",
        nargs="+",
        default=None,
        choices=tuple(_PROPRIO_FILES),
        help=(
            "Disk proprio keys to use. PLD live ingestion passes cfg.env.proprio_keys; "
            "delta_eef_rot6d defaults to state_eef_rot6d."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_gearraw_to_pld_demo(
        input_root=args.input_root,
        output_dir=args.output_dir,
        split_per_episode=args.split_per_episode,
        filter_noop=args.filter_noop,
        noop_eps=args.noop_eps,
        image_size=tuple(args.image_size),  # type: ignore[arg-type]
        prefer_delta_eef=args.prefer_delta_eef,
        action_repr=args.action_repr,
        image_keys=args.image_keys,
        proprio_keys=args.proprio_keys,
    )
