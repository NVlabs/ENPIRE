# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiskBufferIngestor — load gearraw episodes into PLD's live replay/demo buffers.

This is the **read-side** half of the PLD ↔ forge integration. Forge's existing
`RecordEpisodeWrapper` writes one gearraw episode dir per saved rollout (NPY +
MP4) into ``$YAM_RAW_PATH/<task>/`` (or any directory passed to it). The
ingestor below polls that directory periodically and converts each newly
finalised episode into PLD transition dicts via the existing converter
(``scripts/data/convert_gearraw_to_pld_demo.py:_load_episode_as_transitions``),
then inserts them into the live replay/demo buffers using ``action-source.npy[t]``:

    "rl"      → replay_buffer  (online RL rollout step)
    "human"   → demo_buffer    (teleop / takeover step; preserves PLD's
                                  actor_env_intvn weighting)
    "manual"  → demo_buffer    (Forge manual state; normalised to "human")
    "policy"  → demo_buffer    (legacy pre-trained base policy step)
    "unknown" → rejected by default; only allowed when strict_source_labels=False

Strict source mode requires every raw action-source entry in an episode folder
to have a valid source label. Mixed rl/human folders are supported: the
ingestor logically splits the saved rollout by per-step source and routes each
transition to the matching live buffer.

Invariants:
    - At-most-once ingestion per episode dir is enforced by process memory.
      ``.pld_ingested`` files from older workflows are ignored; they are not
      created and are never used as the source of truth for whether an episode
      has already been loaded.
    - Insert order is FIFO by directory mtime.
    - Read-side is non-destructive: never moves / deletes source files.
    - Reads are safe against in-flight writes because RecordEpisodeWrapper
      finalises an episode by atomically renaming a temp dir to its
      timestamped name; a dir present at the final path with all
      ``REQUIRED_FILES`` is by construction complete.

Phase 1 design choice (per the approved plan §Phase 1 Validation):
    The ingestor zero-fills ``base_actions`` / ``next_base_actions`` whenever
    the on-disk gearraw lacks ``action-base-*.npy`` (always true for forge's
    teleop captures today). Live RL rollouts that do record those files will
    populate the keys with the recorded values once the converter is extended
    in M3. Either way the in-memory dict shape is identical.

Usage:
    from scripts.pld_lite.disk_buffer_ingestor import DiskBufferIngestor

    ingestor = DiskBufferIngestor(
        root="/path/to/gearraw/root",
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        buffer_update_freq=30.0,
    )
    ingestor.start()    # daemon thread; non-blocking
    # ... later ...
    ingestor.stop()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Legacy marker name. Older workflows may have these files on disk; this
# ingestor deliberately ignores them and tracks ingestion in memory instead.
INGEST_MARKER = ".pld_ingested"
RSYNC_LOCK_MARKER = ".pld_rsync_in_progress"
ACTION_RANGE_EPS = 1e-5

CAMERA_FILES: dict[str, str] = {
    "top_camera_image": "top_camera-images-rgb.mp4",
    "left_camera_image": "left_camera-images-rgb.mp4",
    "right_camera_image": "right_camera-images-rgb.mp4",
    "left_wrist_camera_image": "left_wrist_camera-images-rgb.mp4",
    "wrist_camera_image": "wrist_camera-images-rgb.mp4",
}
DEFAULT_IMAGE_KEYS: tuple[str, ...] = (
    "top_camera_image",
    "left_camera_image",
    "right_camera_image",
)
JOINT_STATE_FILES: tuple[str, ...] = (
    "left-gripper_pos.npy",
    "right-gripper_pos.npy",
    "left-joint_pos.npy",
    "right-joint_pos.npy",
)
EEF_STATE_FILE = "state-eef-rot6d.npy"
STATE_EEF_ROT6D_COMPONENTS: dict[str, tuple[int, ...]] = {
    "L_x": (0,),
    "L_y": (1,),
    "L_z": (2,),
    "L_rot6d": (3, 4, 5, 6, 7, 8),
    "L_grip": (9,),
    "R_x": (10,),
    "R_y": (11,),
    "R_z": (12,),
    "R_rot6d": (13, 14, 15, 16, 17, 18),
    "R_grip": (19,),
}
PROPRIO_COMPONENT_MAPS: dict[str, dict[str, tuple[int, ...]]] = {
    "state_eef_rot6d": STATE_EEF_ROT6D_COMPONENTS,
}
PROPRIO_KEY_DIMS: dict[str, int] = {
    "state_eef_rot6d": 20,
}
# Reward label files: Forge writes one of these formats per episode.
# Either format is accepted; absence means the episode has no reward signal
# and will be skipped (see _episode_complete).
REWARD_LABEL_FILES: tuple[str, ...] = ("reward.npz", "reward.npy")

COMMON_REQUIRED_FILES: tuple[str, ...] = ("action-source.npy",)
JOINT_ACTION_FILES: tuple[str, ...] = ("action-left-pos.npy", "action-right-pos.npy")
DELTA_EEF_ACTION_FILE = "action-delta-eef.npy"
DELTA_EEF_ROT6D_ACTION_FILE = "action-delta-eef-rot6d.npy"
BASE_DELTA_EEF_ACTION_FILE = "action-base-delta-eef.npy"
BASE_DELTA_EEF_ROT6D_ACTION_FILE = "action-base-delta-eef-rot6d.npy"

# Required on-disk files per action_repr.  Add a new repr here when Forge
# introduces a new recording format.
ACTION_REPR_STATE_FILES: dict[str, tuple[str, ...]] = {
    "joint": JOINT_STATE_FILES,
    "delta_eef_quat": JOINT_STATE_FILES,  # legacy quat uses joint proprio
    "delta_eef_rot6d": (EEF_STATE_FILE,),
    "delta_eef_pos": (EEF_STATE_FILE,),
}
ACTION_REPR_ACTION_FILES: dict[str, tuple[str, ...]] = {
    "joint": JOINT_ACTION_FILES,
    "delta_eef_quat": (DELTA_EEF_ACTION_FILE,),
    "delta_eef_rot6d": (DELTA_EEF_ROT6D_ACTION_FILE,),
    "delta_eef_pos": (DELTA_EEF_ROT6D_ACTION_FILE,),
}


def _normalise_image_keys(image_keys: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if image_keys is None:
        return DEFAULT_IMAGE_KEYS
    normalised: list[str] = []
    for key in image_keys:
        key = str(key)
        if key not in CAMERA_FILES:
            raise ValueError(
                f"Unsupported image key {key!r}; expected one of {sorted(CAMERA_FILES)}"
            )
        if key not in normalised:
            normalised.append(key)
    if not normalised:
        raise ValueError("At least one image key is required for disk ingestion")
    return tuple(normalised)


def _import_converter():
    """Lazy import of the gearraw → PLD converter helper.

    ``scripts/data/`` is not a package (no __init__.py), so we add it to
    sys.path and import by module name. Mirrors the pattern the existing CLI
    relies on.
    """
    from .convert_gearraw import _load_episode_as_transitions

    return _load_episode_as_transitions


def _normalise_source(value: Any) -> str:
    """Map a single per-step disk label to the routing key.

    Returns one of ``"rl" | "human" | "policy" | "unknown"``. Handles numpy
    scalar wrappers, bytes, and missing values defensively. Forge's explicit
    manual state writes ``"manual"``, which is human/demo data for PLD.
    """
    if value is None:
        return "unknown"
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return "unknown"
    s = str(value)
    if s == "manual":
        return "human"
    return s if s in ("rl", "human", "policy") else "unknown"


def _resolve_proprio_filter(proprio_filter: Any) -> dict[str, tuple[str, tuple[int, ...]]]:
    if not proprio_filter:
        return {}
    if not hasattr(proprio_filter, "items"):
        raise TypeError("proprio_filter must be a mapping from proprio key to config")

    resolved: dict[str, tuple[str, tuple[int, ...]]] = {}
    for key, cfg in proprio_filter.items():
        if key not in PROPRIO_COMPONENT_MAPS:
            raise ValueError(
                f"Unsupported proprio_filter key {key!r}; expected one of "
                f"{sorted(PROPRIO_COMPONENT_MAPS)}"
            )
        if cfg is None:
            continue
        if not hasattr(cfg, "get"):
            raise TypeError(
                f"proprio_filter[{key!r}] must be a mapping with mode/include or "
                "mode/exclude fields"
            )
        mode = str(cfg.get("mode", "zero"))
        if mode not in {"zero", "drop"}:
            raise ValueError(f"proprio_filter[{key!r}].mode={mode!r}; expected 'zero' or 'drop'")

        component_map = PROPRIO_COMPONENT_MAPS[key]
        has_include = "include" in cfg and cfg.get("include") is not None
        has_exclude = "exclude" in cfg and cfg.get("exclude") is not None
        if has_include and has_exclude:
            raise ValueError(f"proprio_filter[{key!r}] must specify only one of include or exclude")

        def component_indices(components: Any, *, field: str) -> list[int]:
            if isinstance(components, str):
                components = [components]
            indices: list[int] = []
            for component in components or []:
                component_name = str(component)
                if component_name not in component_map:
                    raise ValueError(
                        f"Unknown proprio_filter component {component_name!r} for "
                        f"{key!r}.{field}; expected one of {sorted(component_map)}"
                    )
                indices.extend(component_map[component_name])
            return indices

        if has_include:
            included_indices = set(component_indices(cfg.get("include"), field="include"))
            all_indices = {idx for indices in component_map.values() for idx in indices}
            excluded_indices = sorted(all_indices - included_indices)
        else:
            excluded_indices = component_indices(cfg.get("exclude", []), field="exclude")
        resolved[key] = (mode, tuple(sorted(set(excluded_indices))))
    return resolved


def _apply_proprio_filter_to_state(
    state: np.ndarray,
    *,
    key: str,
    filter_spec: tuple[str, tuple[int, ...]],
) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32)
    raw_dim = PROPRIO_KEY_DIMS[key]
    mode, excluded_indices = filter_spec
    effective_dim = raw_dim - len(excluded_indices) if mode == "drop" else raw_dim
    if mode == "drop" and values.shape[-1] == effective_dim:
        return values
    if values.shape[-1] != raw_dim:
        raise ValueError(
            f"Cannot apply proprio_filter for {key!r}: expected final dim {raw_dim}, "
            f"got shape {values.shape}"
        )

    if not excluded_indices:
        return values
    if mode == "zero":
        filtered = values.copy()
        filtered[..., list(excluded_indices)] = 0.0
        return filtered
    return np.delete(values, list(excluded_indices), axis=-1).astype(np.float32, copy=False)


_ACTION_ABS_SAMPLES_MAX = 500_000


def _empty_action_range_stats(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "epsilon": ACTION_RANGE_EPS,
        "checked_total": 0,
        "checked_by_buffer": {"replay": 0, "demo": 0},
        "out_of_range_total": 0,
        "out_of_range_by_buffer": {"replay": 0, "demo": 0},
        "out_of_range_by_label": {"rl": 0, "human": 0, "policy": 0, "unknown": 0},
        "max_abs": 0.0,
        "last_max_abs": 0.0,
        "last": None,
        "demo_q01": None,
        "demo_q99": None,
        "last_demo_q01": None,
        "last_demo_q99": None,
    }


def _episode_complete(
    ep_dir: Path,
    *,
    action_repr: str = "joint",
    image_keys: tuple[str, ...] | list[str] | None = None,
    # prefer_delta_eef kept for call-site compatibility but unused: action_repr
    # is now the sole source of truth for which files are required.
    prefer_delta_eef: bool = False,
) -> bool:
    """True iff the dir contains all files required for the configured action_repr.

    Reward labels (reward.npz or reward.npy) are mandatory: Forge is expected
    to write one of these formats for every episode. Episodes without reward
    labels are treated as incomplete so the ingestor skips them rather than
    silently inserting transitions with synthesized rewards that don't reflect
    actual task outcomes.
    """
    image_keys = _normalise_image_keys(image_keys)
    if not all((ep_dir / name).exists() for name in COMMON_REQUIRED_FILES):
        return False
    if not all((ep_dir / CAMERA_FILES[key]).exists() for key in image_keys):
        return False
    state_files = ACTION_REPR_STATE_FILES.get(action_repr, JOINT_STATE_FILES)
    if not all((ep_dir / name).exists() for name in state_files):
        return False
    action_files = ACTION_REPR_ACTION_FILES.get(action_repr, JOINT_ACTION_FILES)
    if not all((ep_dir / name).exists() for name in action_files):
        return False
    return any((ep_dir / name).exists() for name in REWARD_LABEL_FILES)


def _root_sync_locked(root: Path) -> bool:
    """True while an external sync process is updating the watched root."""
    return (root / RSYNC_LOCK_MARKER).exists()


def _episode_has_base(
    ep_dir: Path, *, prefer_delta_eef: bool = False, action_repr: str = "joint"
) -> bool:
    """True iff the gearraw dir holds RL-rollout pre-residual base actions."""
    if prefer_delta_eef and action_repr in ("delta_eef_rot6d", "delta_eef_pos"):
        return (ep_dir / BASE_DELTA_EEF_ROT6D_ACTION_FILE).exists()
    if prefer_delta_eef and (ep_dir / BASE_DELTA_EEF_ACTION_FILE).exists():
        return True
    return (ep_dir / "action-base-left-pos.npy").exists() and (
        ep_dir / "action-base-right-pos.npy"
    ).exists()


def _attach_base_actions_zeros(transitions: list[dict[str, Any]], action_dim: int = 14) -> None:
    """Ensure every transition has base_actions / next_base_actions keys.

    Phase 1 default for teleop episodes (no action-base-*.npy on disk): fill
    with zeros of the appropriate shape. The PLD demo loader at
    train_delta_rlpd.py:843-848 already recomputes base actions via the base
    agent at insert time when zeros are present, so this is a no-op for the
    actual training math — we only need it so the buffer's array-of-structures
    layout sees consistent dict keys on every insert.
    """
    z = np.zeros(action_dim, dtype=np.float32)
    for t in transitions:
        t.setdefault("base_actions", z.copy())
        t.setdefault("next_base_actions", z.copy())


def _as_2d_actions(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _per_arm_action_dim(action_repr: str) -> int | None:
    return {
        "joint": 7,
        "delta_eef_quat": 8,
        "delta_eef_rot6d": 10,
        "delta_eef_pos": 3,
    }.get(str(action_repr))


def _select_action_for_mode(
    action: np.ndarray,
    *,
    action_dim: int,
    action_repr: str,
    control_mode: str,
    label: str,
) -> np.ndarray:
    """Return an action matching the configured replay-buffer action space.

    Forge records bimanual YAM delta-EEF episodes as 20-D rot6d. A single-arm
    PLD learner still uses the full 20-D state, but its action space is one arm
    only: left [0:10] or right [10:20].
    """
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat.shape[0] == int(action_dim):
        return flat

    if str(action_repr) == "delta_eef_pos":
        if flat.shape[0] == 20 and int(action_dim) == 3:
            if control_mode == "left":
                return flat[0:3].copy()
            if control_mode == "right":
                return flat[10:13].copy()
            raise ValueError(
                f"{label} has bimanual rot6d shape 20 but action_dim=3; "
                "set control_mode to 'left' or 'right' to select one arm."
            )
        if flat.shape[0] == 20 and int(action_dim) == 6:
            return np.concatenate([flat[0:3], flat[10:13]]).astype(np.float32)
        if flat.shape[0] == 10 and int(action_dim) == 3:
            return flat[0:3].copy()

    per_arm_dim = _per_arm_action_dim(action_repr)
    if per_arm_dim is None:
        raise ValueError(f"Unsupported action_repr={action_repr!r}")
    if flat.shape[0] == 2 * per_arm_dim and int(action_dim) == per_arm_dim:
        if control_mode == "left":
            return flat[:per_arm_dim].copy()
        if control_mode == "right":
            return flat[per_arm_dim : 2 * per_arm_dim].copy()
        raise ValueError(
            f"{label} has bimanual shape {flat.shape[0]} but action_dim={action_dim}; "
            "set control_mode to 'left' or 'right' to select one arm."
        )

    raise ValueError(
        f"{label} has shape {flat.shape}; expected action_dim={action_dim} for "
        f"action_repr={action_repr!r} control_mode={control_mode!r}"
    )


class DiskBufferIngestor:
    """Background daemon that ingests gearraw episodes into PLD live buffers.

    Args:
        root: Directory containing gearraw episode subdirs. Each subdir must
            include the gearraw ``REQUIRED_FILES``. Subdirs missing any file
            are skipped (and re-checked on the next scan).
        replay_buffer: Object with ``.insert(transition_dict)`` for online RL
            transitions (action-source ``"rl"``).
        demo_buffer: Object with ``.insert(transition_dict)`` for human /
            policy / unknown-source transitions.
        base_agent_fn: Optional ``obs_dict -> np.ndarray[14]`` callable. If
            provided AND the episode lacks recorded base actions, replaces
            zero-filled ``base_actions`` / ``next_base_actions`` with
            on-the-fly recomputed values. Phase 1 validation does NOT use this;
            zero-filling is fine because the PLD demo loader recomputes anyway.
        buffer_update_freq: Period in **seconds** between buffer-update
            scans (despite the name, this is a period, not a Hz rate; we
            keep the name for API consistency with the watch script CLI).
            Default 30s. Tune lower for more on-policy-ish behaviour, higher
            to amortise scan cost over more learner steps.
        image_size: Camera resize target passed through to the converter.
        action_dim: Flat action dimension. For YAM delta-EEF rot6d this is 20
            for bimanual or 10 for a single arm.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        replay_buffer,
        demo_buffer,
        *,
        base_agent_fn: Optional[Callable[[dict], np.ndarray]] = None,
        buffer_update_freq: float = 30.0,
        image_size: tuple[int, int] = (256, 256),
        action_dim: int = 14,
        control_mode: str = "both",
        unknown_to_replay: bool = False,
        human_to_replay: bool = False,
        prefer_delta_eef: bool = False,
        action_repr: str | None = None,
        strict_source_labels: bool = True,
        inv_action_scaling_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        action_gamma: float = 1.0,
        use_base_actions: bool = True,
        min_episode_age_s: float = 0.0,
        reuse_ingested_marked: bool = False,
        ingested_episode_paths: set[Path] | None = None,
        image_keys: tuple[str, ...] | list[str] | None = None,
        proprio_keys: tuple[str, ...] | list[str] | None = None,
        proprio_filter: Any = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.replay = replay_buffer
        self.demo = demo_buffer
        self.base_agent_fn = base_agent_fn
        self.buffer_update_freq = float(buffer_update_freq)
        self.image_size = tuple(image_size)
        self.image_keys = _normalise_image_keys(image_keys)
        self.proprio_keys = tuple(proprio_keys) if proprio_keys is not None else None
        self.proprio_filter = _resolve_proprio_filter(proprio_filter)
        self.action_dim = int(action_dim)
        self.control_mode = str(control_mode)
        if self.control_mode not in ("left", "right", "both"):
            raise ValueError(
                f"Unsupported control_mode={self.control_mode!r}; expected "
                "'left', 'right', or 'both'"
            )
        # Flag for smokes / bootstrap runs: route "unknown" labelled
        # transitions (e.g. raw teleop captures with no source label) to the
        # replay buffer instead of the demo buffer. The learner blocks on
        # replay_buffer >= training_starts before training begins, so this
        # is what unblocks pure-teleop bring-up before any "rl" data exists.
        self.unknown_to_replay = bool(unknown_to_replay)
        self.human_to_replay = bool(human_to_replay)
        # When True, load the explicitly configured delta-EE representation
        # instead of legacy joint target actions. Forge records bimanual
        # action-delta-eef-rot6d.npy ([T, 20]); single-arm PLD learners slice
        # that file down to their configured 10-D action space.
        self.prefer_delta_eef = bool(prefer_delta_eef)
        self.strict_source_labels = bool(strict_source_labels)
        if action_repr is None or str(action_repr).strip() == "":
            raise ValueError(
                "DiskBufferIngestor requires explicit action_repr "
                "('joint', 'delta_eef_quat', 'delta_eef_rot6d', or 'delta_eef_pos'); "
                "refusing to silently assume joint actions."
            )
        self.action_repr = str(action_repr)
        if self.action_repr not in (
            "joint",
            "delta_eef_quat",
            "delta_eef_rot6d",
            "delta_eef_pos",
        ):
            raise ValueError(f"Unsupported action_repr={self.action_repr!r}")
        if self.prefer_delta_eef and self.action_repr == "joint":
            raise ValueError(
                "prefer_delta_eef=True requires explicit action_repr="
                "'delta_eef_quat', 'delta_eef_rot6d', or 'delta_eef_pos'; "
                "refusing to infer from files."
            )
        per_arm_dim = _per_arm_action_dim(self.action_repr)
        valid_dims = (
            {per_arm_dim, 2 * per_arm_dim} if per_arm_dim is not None else {self.action_dim}
        )
        if self.action_dim not in valid_dims:
            raise ValueError(
                f"action_repr={self.action_repr!r} requires action_dim in "
                f"{sorted(valid_dims)}; got action_dim={self.action_dim}"
            )
        # Per-step transformation applied to raw transition["actions"] before
        # buffer insertion. Residual SAC uses (inv(raw) - base) / gamma; clean
        # SACMini uses inv(raw) / gamma and does not store base action fields.
        # When None, transitions are inserted as-is (pre-W4 behaviour).
        self.inv_action_scaling_fn = inv_action_scaling_fn
        self.action_gamma = float(action_gamma)
        self.use_base_actions = bool(use_base_actions)
        self.min_episode_age_s = max(0.0, float(min_episode_age_s))
        # Kept as a no-op compatibility option for older configs. Markers are
        # no longer used for ingestion decisions.
        self.reuse_ingested_marked = bool(reuse_ingested_marked)
        self._session_ingested: set[Path] = (
            ingested_episode_paths if ingested_episode_paths is not None else set()
        )
        self._action_range_check_enabled = (
            self.inv_action_scaling_fn is not None and self.action_gamma > 0.0
        )
        self._action_abs_samples: list[float] = []

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = dict(
            scans=0,
            episodes=0,
            transitions=0,
            nonzero_reward_transitions=0,
            errors=0,
            last_error=None,
            last_error_path=None,
            last_error_epoch=None,
            routed_per_label={"rl": 0, "human": 0, "policy": 0, "unknown": 0},
            last_scan_duration_s=0.0,
            last_discover_duration_s=0.0,
            last_episode_duration_s=0.0,
            last_load_duration_s=0.0,
            last_insert_duration_s=0.0,
            last_episode_transitions=0,
            last_episode_path=None,
            last_scan_start_epoch=None,
            last_scan_end_epoch=None,
            total_scan_duration_s=0.0,
            total_load_duration_s=0.0,
            total_insert_duration_s=0.0,
            reuse_ingested_marked=self.reuse_ingested_marked,
            reusable_marked_episodes=0,
            session_ingested_episodes=len(self._session_ingested),
            tracking_mode="memory",
            action_range=_empty_action_range_stats(enabled=self._action_range_check_enabled),
        )

        self._load_episode = _import_converter()

    def _apply_proprio_filter(self, transitions: list[dict[str, Any]]) -> None:
        """Match disk-loaded proprio shape to the live env observation space."""
        if not self.proprio_filter:
            return
        if self.proprio_keys != ("state_eef_rot6d",):
            raise ValueError(
                "Disk proprio_filter is only supported for proprio_keys="
                "('state_eef_rot6d',); got "
                f"{self.proprio_keys!r}"
            )

        filter_spec = self.proprio_filter.get("state_eef_rot6d")
        if filter_spec is None:
            return
        for transition in transitions:
            for obs_key in ("observations", "next_observations"):
                obs = transition.get(obs_key)
                if not isinstance(obs, dict) or "state" not in obs:
                    raise KeyError(
                        f"Transition {obs_key} is missing flat 'state' for proprio_filter"
                    )
                obs["state"] = _apply_proprio_filter_to_state(
                    obs["state"],
                    key="state_eef_rot6d",
                    filter_spec=filter_spec,
                )

    # ----------------------------------------------------------------------
    # Daemon lifecycle
    # ----------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pld-disk-ingestor")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to exit and wait for it."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of ingestion counters."""
        with self._lock:
            # Shallow copy keeps callers' code simple; routed_per_label is a
            # nested dict, so copy nested stats too.
            out = dict(self._stats)
            out["routed_per_label"] = dict(self._stats["routed_per_label"])
            action_range = dict(self._stats["action_range"])
            action_range["checked_by_buffer"] = dict(action_range["checked_by_buffer"])
            action_range["out_of_range_by_buffer"] = dict(action_range["out_of_range_by_buffer"])
            action_range["out_of_range_by_label"] = dict(action_range["out_of_range_by_label"])
            if isinstance(action_range.get("last"), dict):
                action_range["last"] = dict(action_range["last"])
            out["action_range"] = action_range
            return out

    def disk_inventory(self) -> dict[str, Any]:
        """Return a lightweight inventory of complete episode dirs on disk.

        This is intended for live status displays. It only checks the same file
        completeness predicate used by the ingestor and, when possible, reads
        the action array header with ``mmap_mode='r'`` to estimate transition
        count without loading the full episode.
        """
        inventory = {
            "root": str(self.root),
            "root_exists": self.root.exists(),
            "sync_in_progress": _root_sync_locked(self.root),
            "complete_episodes": 0,
            "pending_episodes": 0,
            "ingested_episodes": 0,
            "marker_ingested_episodes": 0,
            "reusable_marked_episodes": 0,
            "session_ingested_episodes": 0,
            "reuse_ingested_marked": self.reuse_ingested_marked,
            "tracking_mode": "memory",
            "estimated_transitions": 0,
            "pending_estimated_transitions": 0,
            "latest_complete_mtime": None,
            "latest_pending_mtime": None,
            "seconds_since_latest_complete": None,
            "seconds_since_latest_pending": None,
            "errors": 0,
            "last_error": None,
        }
        if not self.root.exists():
            return inventory
        if inventory["sync_in_progress"]:
            return inventory

        candidates: list[Path] = []
        for depth1 in self._iter_subdirs(self.root):
            candidates.append(depth1)
            candidates.extend(self._iter_subdirs(depth1))

        seen: set[Path] = set()
        for ep in candidates:
            if ep in seen:
                continue
            seen.add(ep)
            try:
                if not _episode_complete(
                    ep,
                    prefer_delta_eef=self.prefer_delta_eef,
                    action_repr=self.action_repr,
                    image_keys=self.image_keys,
                ):
                    continue
                transition_estimate = self._estimate_episode_transitions(ep)
                inventory["complete_episodes"] += 1
                inventory["estimated_transitions"] += transition_estimate
                mtime = ep.stat().st_mtime
                latest_complete = inventory["latest_complete_mtime"]
                if latest_complete is None or mtime > float(latest_complete):
                    inventory["latest_complete_mtime"] = mtime
                session_ingested = self._is_session_ingested(ep)
                if session_ingested:
                    inventory["ingested_episodes"] += 1
                    inventory["session_ingested_episodes"] += 1
                else:
                    inventory["pending_episodes"] += 1
                    inventory["pending_estimated_transitions"] += transition_estimate
                    latest_pending = inventory["latest_pending_mtime"]
                    if latest_pending is None or mtime > float(latest_pending):
                        inventory["latest_pending_mtime"] = mtime
            except Exception as e:
                inventory["errors"] += 1
                inventory["last_error"] = f"{ep.name}: {e!r}"
        now = time.time()
        if inventory["latest_complete_mtime"] is not None:
            inventory["seconds_since_latest_complete"] = max(
                0.0, now - float(inventory["latest_complete_mtime"])
            )
        if inventory["latest_pending_mtime"] is not None:
            inventory["seconds_since_latest_pending"] = max(
                0.0, now - float(inventory["latest_pending_mtime"])
            )
        return inventory

    def _estimate_episode_transitions(self, ep_dir: Path) -> int:
        if self.prefer_delta_eef and self.action_repr in ("delta_eef_rot6d", "delta_eef_pos"):
            action_path = ep_dir / DELTA_EEF_ROT6D_ACTION_FILE
        elif self.prefer_delta_eef:
            action_path = ep_dir / DELTA_EEF_ACTION_FILE
        else:
            action_path = ep_dir / "action-left-pos.npy"
        arr = np.load(action_path, mmap_mode="r")
        if arr.ndim == 0:
            return 0
        return max(0, int(arr.shape[0]) - 1)

    # ----------------------------------------------------------------------
    # Internal: scan / ingest
    # ----------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._stats["scans"] += 1
            try:
                self.scan_once()
            except Exception as e:
                logger.exception("[pld-ingestor] scan failed: %s", e)
                with self._lock:
                    self._stats["errors"] += 1
                    self._stats["last_error"] = repr(e)
                    self._stats["last_error_path"] = None
                    self._stats["last_error_epoch"] = time.time()
            self._stop.wait(self.buffer_update_freq)

    def scan_once(self, show_progress: bool = False) -> int:
        """One scan-and-ingest pass. Returns count of episodes ingested."""
        scan_t0 = time.perf_counter()
        scan_start_epoch = time.time()
        if not self.root.exists():
            self._record_scan_timing(scan_t0, scan_start_epoch, 0.0)
            return 0
        if _root_sync_locked(self.root):
            logger.debug(
                "[pld-ingestor] sync lock present at %s; skipping scan",
                self.root / RSYNC_LOCK_MARKER,
            )
            self._record_scan_timing(scan_t0, scan_start_epoch, 0.0)
            return 0
        ingested = 0
        discover_t0 = time.perf_counter()
        episodes = self._discover()
        discover_duration = time.perf_counter() - discover_t0
        total = len(episodes)
        for ep in episodes:
            try:
                ep_t0 = time.perf_counter()
                n, timing = self._ingest_episode(ep)
                episode_duration = time.perf_counter() - ep_t0
                self._mark_session_ingested(ep)
                with self._lock:
                    self._stats["episodes"] += 1
                    self._stats["transitions"] += n
                    self._stats["session_ingested_episodes"] = len(self._session_ingested)
                    self._stats["last_episode_duration_s"] = episode_duration
                    self._stats["last_load_duration_s"] = timing["load_duration_s"]
                    self._stats["last_insert_duration_s"] = timing["insert_duration_s"]
                    self._stats["last_episode_transitions"] = n
                    self._stats["last_episode_path"] = str(ep)
                    self._stats["total_load_duration_s"] += timing["load_duration_s"]
                    self._stats["total_insert_duration_s"] += timing["insert_duration_s"]
                    if self._stats.get("last_error_path") == str(ep):
                        self._stats["last_error"] = None
                        self._stats["last_error_path"] = None
                        self._stats["last_error_epoch"] = None
                ingested += 1
                if show_progress:
                    print(
                        f"\r  preloading buffer: {ingested}/{total} episodes",
                        end="",
                        flush=True,
                    )
                logger.debug("[pld-ingestor] ingested %d transitions from %s", n, ep.name)
            except Exception as e:
                logger.exception("[pld-ingestor] failed to ingest %s: %s", ep, e)
                with self._lock:
                    self._stats["errors"] += 1
                    self._stats["last_error"] = f"{ep.name}: {e!r}"
                    self._stats["last_error_path"] = str(ep)
                    self._stats["last_error_epoch"] = time.time()
        self._record_scan_timing(scan_t0, scan_start_epoch, discover_duration)
        return ingested

    def _record_scan_timing(
        self, scan_t0: float, scan_start_epoch: float, discover_duration: float
    ) -> None:
        duration = time.perf_counter() - scan_t0
        with self._lock:
            self._stats["last_scan_duration_s"] = duration
            self._stats["last_discover_duration_s"] = discover_duration
            self._stats["last_scan_start_epoch"] = scan_start_epoch
            self._stats["last_scan_end_epoch"] = time.time()
            self._stats["total_scan_duration_s"] += duration

    def _discover(self) -> list[Path]:
        """Return episode dirs ready to ingest, FIFO by mtime.

        Walks ``self.root`` recursively (one level deep is the common case;
        forge writes ``$YAM_RAW_PATH/<operator>_<task>_<ts>-YAM-<station>/<ep_ts>/``,
        so we look at depth 1 and 2). Skips dirs already recorded in memory
        and dirs missing any required file.
        """
        out: list[tuple[float, Path]] = []
        # Search up to 2 levels deep so both
        # $YAM_RAW_PATH/episode_dir/ and
        # $YAM_RAW_PATH/<run_dir>/<episode_dir>/ shapes work.
        for depth1 in self._iter_subdirs(self.root):
            if self._is_ingestable(depth1):
                out.append((depth1.stat().st_mtime, depth1))
                continue
            for depth2 in self._iter_subdirs(depth1):
                if self._is_ingestable(depth2):
                    out.append((depth2.stat().st_mtime, depth2))
        return [p for _, p in sorted(out, key=lambda x: x[0])]

    @staticmethod
    def _iter_subdirs(p: Path):
        try:
            return [c for c in p.iterdir() if c.is_dir()]
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return []

    def _is_ingestable(self, ep: Path) -> bool:
        if self._is_session_ingested(ep):
            return False
        if self.min_episode_age_s > 0.0:
            age_s = self._seconds_since_latest_write(ep)
            if age_s is None or age_s < self.min_episode_age_s:
                return False
        return _episode_complete(
            ep,
            prefer_delta_eef=self.prefer_delta_eef,
            action_repr=self.action_repr,
            image_keys=self.image_keys,
        )

    @staticmethod
    def _episode_key(ep: Path) -> Path:
        try:
            return ep.expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            return ep.expanduser().absolute()

    def _is_session_ingested(self, ep: Path) -> bool:
        return self._episode_key(ep) in self._session_ingested

    def _mark_session_ingested(self, ep: Path) -> None:
        self._session_ingested.add(self._episode_key(ep))

    @staticmethod
    def _seconds_since_latest_write(ep: Path) -> float | None:
        try:
            mtimes = [ep.stat().st_mtime]
            mtimes.extend(child.stat().st_mtime for child in ep.iterdir() if child.is_file())
            return max(0.0, time.time() - max(mtimes))
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return None

    def _attach_recorded_delta_eef_base_actions(
        self, ep_dir: Path, transitions: list[dict[str, Any]]
    ) -> None:
        """Attach recorded raw base delta-EE actions when forge saved them.

        The replay buffer stores base actions in SAC-normalised space, matching
        the base-agent output used online. Disk files are raw env commands, so
        the same inverse scaling used for executed actions is applied here.
        """
        base_path = (
            ep_dir / BASE_DELTA_EEF_ROT6D_ACTION_FILE
            if self.action_repr in ("delta_eef_rot6d", "delta_eef_pos")
            else ep_dir / BASE_DELTA_EEF_ACTION_FILE
        )
        if not base_path.exists():
            return
        base_raw = _as_2d_actions(np.load(base_path))
        if len(base_raw) == 0:
            return

        for idx, transition in enumerate(transitions):
            cur_idx = min(idx, len(base_raw) - 1)
            next_idx = min(idx + 1, len(base_raw) - 1)
            base = _select_action_for_mode(
                base_raw[cur_idx],
                action_dim=self.action_dim,
                action_repr=self.action_repr,
                control_mode=self.control_mode,
                label=base_path.name,
            )
            next_base = _select_action_for_mode(
                base_raw[next_idx],
                action_dim=self.action_dim,
                action_repr=self.action_repr,
                control_mode=self.control_mode,
                label=base_path.name,
            )
            if self.inv_action_scaling_fn is not None:
                base = np.asarray(self.inv_action_scaling_fn(base), dtype=np.float32)
                next_base = np.asarray(self.inv_action_scaling_fn(next_base), dtype=np.float32)
            transition["base_actions"] = base.reshape(-1).astype(np.float32)
            transition["next_base_actions"] = next_base.reshape(-1).astype(np.float32)

    @staticmethod
    def _action_range_violation(
        action: Any,
        *,
        episode: str,
        transition_index: int,
        label: str,
        buffer_name: str,
    ) -> tuple[float, dict[str, Any] | None]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0, None
        finite = np.isfinite(arr)
        if not bool(np.all(finite)):
            bad_indices = np.flatnonzero(~finite)
            max_index = int(bad_indices[0]) if bad_indices.size else 0
            return float("inf"), {
                "episode": episode,
                "transition_index": int(transition_index),
                "label": str(label),
                "buffer": str(buffer_name),
                "max_abs": float("inf"),
                "max_index": max_index,
                "value": float(arr[max_index]),
                "nonfinite": True,
            }

        abs_action = np.abs(arr)
        max_index = int(np.argmax(abs_action))
        max_abs = float(abs_action[max_index])
        if max_abs <= 1.0 + ACTION_RANGE_EPS:
            return max_abs, None
        return max_abs, {
            "episode": episode,
            "transition_index": int(transition_index),
            "label": str(label),
            "buffer": str(buffer_name),
            "max_abs": max_abs,
            "max_index": max_index,
            "value": float(arr[max_index]),
            "nonfinite": False,
        }

    def _ingest_episode(self, ep_dir: Path) -> tuple[int, dict[str, float]]:
        load_t0 = time.perf_counter()
        transitions = self._load_episode(
            str(ep_dir),
            image_size=self.image_size,
            prefer_delta_eef=self.prefer_delta_eef,
            action_repr=self.action_repr,
            image_keys=self.image_keys,
            proprio_keys=self.proprio_keys,
        )
        load_duration = time.perf_counter() - load_t0
        if not transitions:
            return 0, {"load_duration_s": load_duration, "insert_duration_s": 0.0}
        self._apply_proprio_filter(transitions)

        for t in transitions:
            t["actions"] = _select_action_for_mode(
                t["actions"],
                action_dim=self.action_dim,
                action_repr=self.action_repr,
                control_mode=self.control_mode,
                label="transition action",
            )

        if self.use_base_actions:
            _attach_base_actions_zeros(transitions, action_dim=self.action_dim)
        if self.prefer_delta_eef and self.use_base_actions:
            self._attach_recorded_delta_eef_base_actions(ep_dir, transitions)

        # Optional: recompute base actions when no recorded base + agent provided.
        if (
            self.use_base_actions
            and self.base_agent_fn is not None
            and not _episode_has_base(
                ep_dir, prefer_delta_eef=self.prefer_delta_eef, action_repr=self.action_repr
            )
        ):
            for t in transitions:
                try:
                    t["base_actions"] = _select_action_for_mode(
                        self.base_agent_fn(t["observations"]),
                        action_dim=self.action_dim,
                        action_repr=self.action_repr,
                        control_mode=self.control_mode,
                        label="base_agent_fn output",
                    )
                    t["next_base_actions"] = _select_action_for_mode(
                        self.base_agent_fn(t["next_observations"]),
                        action_dim=self.action_dim,
                        action_repr=self.action_repr,
                        control_mode=self.control_mode,
                        label="base_agent_fn output",
                    )
                except Exception as e:  # one bad call shouldn't kill the episode
                    logger.warning("[pld-ingestor] base_agent_fn raised %r — leaving zeros", e)

        # Convert raw env actions into algorithm actions. Residual SAC stores
        # residuals; SACMini stores the direct normalized 3D action.
        if self.prefer_delta_eef and self.inv_action_scaling_fn is not None:
            for t in transitions:
                raw_a = np.asarray(t["actions"], dtype=np.float32)
                inv_a = np.asarray(self.inv_action_scaling_fn(raw_a), dtype=np.float32).reshape(-1)
                if self.action_gamma > 0.0:
                    if self.use_base_actions:
                        base_a = np.asarray(t["base_actions"], dtype=np.float32).reshape(-1)
                        t["actions"] = ((inv_a - base_a) / self.action_gamma).astype(np.float32)
                    else:
                        t["actions"] = (inv_a / self.action_gamma).astype(np.float32)
                else:
                    t["actions"] = np.zeros_like(inv_a, dtype=np.float32)

        # Per-step routing.
        sources = np.load(ep_dir / "action-source.npy", allow_pickle=True)
        if sources.size == 0:
            if self.strict_source_labels:
                logger.error(
                    "[pld-ingestor] %s: action-source.npy is empty; strict "
                    "source validation is enabled, so this episode will be "
                    "rejected before any buffer insert.",
                    ep_dir.name,
                )
            else:
                logger.warning(
                    "[pld-ingestor] %s: action-source.npy is empty; routing all "
                    "transitions to demo_buffer with label 'unknown'",
                    ep_dir.name,
                )

        raw_normalized_labels: list[str] = []
        if self.strict_source_labels:
            if sources.size == 0:
                raise ValueError(
                    f"{ep_dir.name}: action-source.npy is empty. Expected every "
                    "raw action-source entry to be exactly 'rl' or 'human'."
                )
            for raw_idx in range(len(sources)):
                label_raw = sources[raw_idx]
                label = _normalise_source(label_raw)
                if label not in ("rl", "human"):
                    logger.error(
                        "[pld-ingestor] %s: rejecting episode before buffer insert "
                        "because raw action-source entry %d has invalid label "
                        "%r; expected exactly 'rl' or 'human'.",
                        ep_dir.name,
                        raw_idx,
                        label_raw,
                    )
                    raise ValueError(
                        f"{ep_dir.name}: invalid raw action-source label at index "
                        f"{raw_idx}: {label_raw!r}. Expected every raw source "
                        "label to be exactly 'rl' or 'human'."
                    )
                raw_normalized_labels.append(label)
            raw_label_counts = Counter(raw_normalized_labels)

        normalized_labels: list[str] = []
        raw_labels: list[Any] = []
        for idx in range(len(transitions)):
            label_raw = sources[idx] if idx < len(sources) else None
            label = _normalise_source(label_raw)
            if self.strict_source_labels and label not in ("rl", "human"):
                logger.error(
                    "[pld-ingestor] %s: rejecting episode before buffer insert "
                    "because transition %d has invalid action-source label "
                    "%r; expected exactly 'rl' or 'human'.",
                    ep_dir.name,
                    idx,
                    label_raw,
                )
                raise ValueError(
                    f"{ep_dir.name}: invalid action-source label at transition {idx}: "
                    f"{label_raw!r}. Expected every source label to be exactly "
                    "'rl' or 'human'."
                )
            normalized_labels.append(label)
            raw_labels.append(label_raw)

        if self.strict_source_labels:
            logger.debug(
                "[pld-ingestor] %s: strict source validation passed for %d raw "
                "labels and %d transitions; mixed-source episodes are routed "
                "per transition: %s",
                ep_dir.name,
                len(raw_normalized_labels),
                len(transitions),
                dict(Counter(raw_normalized_labels)),
            )

        unknown_seen = False
        per_label = Counter()
        nonzero_rewards = 0
        episode_demo_abs: list[float] = []
        range_checked = 0
        range_checked_by_buffer = Counter()
        range_out_of_range = 0
        range_out_of_range_by_buffer = Counter()
        range_out_of_range_by_label = Counter()
        range_episode_max_abs = 0.0
        range_last_violation: dict[str, Any] | None = None
        insert_t0 = time.perf_counter()
        for idx, transition in enumerate(transitions):
            label_raw = raw_labels[idx]
            label = normalized_labels[idx]
            if label == "unknown" and not unknown_seen:
                dest = "replay_buffer" if self.unknown_to_replay else "demo_buffer"
                logger.warning(
                    "[pld-ingestor] %s: encountered 'unknown' source label "
                    "(raw=%r); routing to %s (warned once per episode)",
                    ep_dir.name,
                    label_raw,
                    dest,
                )
                unknown_seen = True
            if label == "rl" or (label == "human" and self.human_to_replay):
                target = self.replay
                buffer_name = "replay"
            elif label == "unknown" and self.unknown_to_replay:
                target = self.replay
                buffer_name = "replay"
            else:
                target = self.demo
                buffer_name = "demo"
            if self._action_range_check_enabled:
                max_abs, violation = self._action_range_violation(
                    transition.get("actions"),
                    episode=ep_dir.name,
                    transition_index=idx,
                    label=label,
                    buffer_name=buffer_name,
                )
                range_checked += 1
                range_checked_by_buffer[buffer_name] += 1
                range_episode_max_abs = max(range_episode_max_abs, max_abs)
                if violation is not None:
                    range_out_of_range += 1
                    range_out_of_range_by_buffer[buffer_name] += 1
                    range_out_of_range_by_label[label] += 1
                    range_last_violation = violation
            target.insert(transition)
            per_label[label] += 1
            if buffer_name == "demo":
                try:
                    abs_vals = np.abs(
                        np.asarray(transition.get("actions"), dtype=np.float32).reshape(-1)
                    )
                    if abs_vals.size > 0:
                        episode_demo_abs.extend(abs_vals.tolist())
                except Exception:
                    pass
            reward = transition.get("rewards", 0)
            try:
                if float(reward) != 0.0:
                    nonzero_rewards += 1
            except (TypeError, ValueError):
                pass
        insert_duration = time.perf_counter() - insert_t0

        if range_out_of_range:
            logger.warning(
                "[pld-ingestor] %s: normalized buffer actions exceeded [-1, 1] "
                "after inverse scaling; out_of_range=%d demo=%d replay=%d "
                "max_abs=%s. This can indicate teleop/action-scale mismatch.",
                ep_dir.name,
                range_out_of_range,
                range_out_of_range_by_buffer.get("demo", 0),
                range_out_of_range_by_buffer.get("replay", 0),
                f"{range_episode_max_abs:.6g}",
            )

        last_demo_q01 = last_demo_q99 = None
        demo_q01 = demo_q99 = None
        if episode_demo_abs:
            ep_arr = np.array(episode_demo_abs, dtype=np.float32)
            last_demo_q01 = float(np.percentile(ep_arr, 1))
            last_demo_q99 = float(np.percentile(ep_arr, 99))
            self._action_abs_samples.extend(episode_demo_abs)
            if len(self._action_abs_samples) > _ACTION_ABS_SAMPLES_MAX:
                del self._action_abs_samples[
                    : len(self._action_abs_samples) - _ACTION_ABS_SAMPLES_MAX
                ]
            global_arr = np.array(self._action_abs_samples, dtype=np.float32)
            demo_q01 = float(np.percentile(global_arr, 1))
            demo_q99 = float(np.percentile(global_arr, 99))

        with self._lock:
            self._stats["nonzero_reward_transitions"] += nonzero_rewards
            for k, v in per_label.items():
                self._stats["routed_per_label"][k] = self._stats["routed_per_label"].get(k, 0) + v
            range_stats = self._stats["action_range"]
            range_stats["enabled"] = bool(self._action_range_check_enabled)
            range_stats["checked_total"] += range_checked
            for k, v in range_checked_by_buffer.items():
                range_stats["checked_by_buffer"][k] = range_stats["checked_by_buffer"].get(k, 0) + v
            range_stats["out_of_range_total"] += range_out_of_range
            for k, v in range_out_of_range_by_buffer.items():
                range_stats["out_of_range_by_buffer"][k] = (
                    range_stats["out_of_range_by_buffer"].get(k, 0) + v
                )
            for k, v in range_out_of_range_by_label.items():
                range_stats["out_of_range_by_label"][k] = (
                    range_stats["out_of_range_by_label"].get(k, 0) + v
                )
            if range_checked:
                range_stats["last_max_abs"] = range_episode_max_abs
                range_stats["max_abs"] = max(
                    float(range_stats.get("max_abs", 0.0) or 0.0),
                    range_episode_max_abs,
                )
            if range_last_violation is not None:
                range_stats["last"] = range_last_violation
            if last_demo_q01 is not None:
                range_stats["last_demo_q01"] = last_demo_q01
                range_stats["last_demo_q99"] = last_demo_q99
                range_stats["demo_q01"] = demo_q01
                range_stats["demo_q99"] = demo_q99

        return len(transitions), {
            "load_duration_s": load_duration,
            "insert_duration_s": insert_duration,
        }
