# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuRobo motion planner for single-arm Franka Panda (7-DOF).

Simplified version of ``motion_planner_curobo.YamMotionPlannerCurobo`` for
single-arm robots.  Uses cuRobo's built-in ``franka.yml`` config (URDF +
collision spheres), so no custom robot model is needed.

For MVP the world config is empty (no kitchen obstacles).
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from enpire.env.forge.paths import THIRD_PARTY_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CUROBO_CONTENT = THIRD_PARTY_ROOT / "curobo" / "src" / "curobo" / "content"
_FRANKA_CFG = _CUROBO_CONTENT / "configs" / "robot" / "franka.yml"

# ---------------------------------------------------------------------------
# Constants (shared with YAM planner)
# ---------------------------------------------------------------------------

# Offset from panda_hand (cuRobo EE) to grip_site (robosuite EE) along
# the local Z-axis.  From robosuite panda_gripper.xml: body "eef" pos="0 0 0.097".
_GRIP_TO_HAND_OFFSET = 0.097

_INTERPOLATION_DT = 1.0 / 30.0
_DEFAULT_POSITION_THRESHOLD_M = 0.005
_DEFAULT_ROTATION_THRESHOLD_RAD = 0.05
_DEFAULT_CSPACE_THRESHOLD_RAD = 0.05
_USE_CUDA_GRAPH_BY_DEFAULT = True
_ENABLE_GRAPH_SEARCH_BY_DEFAULT = True
_CUROBO_TORCH_COMPILE_DISABLE_DEFAULT = "1"
_DEFAULT_BATCH_PLANNER_CAPACITY = 16
_DEFAULT_SOLVER_SPEED = "fast"
_DEFAULT_COLLISION_CACHE_OBB = 4096
_DEFAULT_COLLISION_CACHE_MESH = 1024

_SOLVER_PRESET_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "slow": {
        "motion_gen": {
            "num_ik_seeds": 32,
            "num_graph_seeds": 12,
            "num_trajopt_seeds": 12,
            "trajopt_tsteps": 48,
            "ik_opt_iters": 256,
            "grad_trajopt_iters": 256,
        },
        "plan": {
            "enable_graph_attempt": 4,
            "max_attempts": 10,
            "timeout": 10.0,
            "time_dilation_factor_single": 0.5,
            "time_dilation_factor_batch": None,
        },
    },
    "fast": {
        "motion_gen": {
            "num_ik_seeds": 8,
            "num_graph_seeds": 1,
            "num_trajopt_seeds": 2,
            "trajopt_tsteps": 32,
            "ik_opt_iters": 96,
            "grad_trajopt_iters": 96,
        },
        "plan": {
            "enable_graph_attempt": 1,
            "max_attempts": 2,
            "timeout": 2.5,
            "time_dilation_factor_single": 0.5,
            "time_dilation_factor_batch": None,
        },
    },
}

warnings.filterwarnings(
    "ignore",
    message=r"The symbol `warp\.torch\.device_from_torch` will soon be removed from the public API\..*",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_local_curobo_on_syspath() -> None:
    candidate = THIRD_PARTY_ROOT / "curobo" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _silence_curobo_import_logs() -> None:
    logging.getLogger("curobo").setLevel(logging.ERROR)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1).astype(
        np.float64
    )


def _normalize_solver_speed(value: str | None) -> str:
    solver_speed = str(value or _DEFAULT_SOLVER_SPEED).strip().lower()
    if solver_speed in _SOLVER_PRESET_CONFIGS:
        return solver_speed
    warnings.warn(
        f"Unknown cuRobo solver_speed {value!r}; falling back to {_DEFAULT_SOLVER_SPEED!r}",
        RuntimeWarning,
        stacklevel=2,
    )
    return _DEFAULT_SOLVER_SPEED


def _curobo_timing_info(result: Any) -> dict[str, Any]:
    return {
        "curobo_solve_time_ms": 1000.0 * float(getattr(result, "solve_time", 0.0)),
        "curobo_total_time_ms": 1000.0 * float(getattr(result, "total_time", 0.0)),
        "curobo_ik_time_ms": 1000.0 * float(getattr(result, "ik_time", 0.0)),
        "curobo_graph_time_ms": 1000.0 * float(getattr(result, "graph_time", 0.0)),
        "curobo_trajopt_time_ms": 1000.0 * float(getattr(result, "trajopt_time", 0.0)),
        "curobo_finetune_time_ms": 1000.0
        * float(getattr(result, "finetune_time", 0.0)),
        "curobo_attempts": int(getattr(result, "attempts", 0)),
        "curobo_trajopt_attempts": int(getattr(result, "trajopt_attempts", 0)),
        "curobo_used_graph": bool(getattr(result, "used_graph", False)),
    }


def _as_numpy_metric_array(metric: Any, batch_size: int) -> np.ndarray:
    if metric is None:
        return np.full((batch_size,), np.nan, dtype=np.float64)
    if hasattr(metric, "detach"):
        metric = metric.detach().cpu().numpy()
    metric = np.asarray(metric, dtype=np.float64).reshape(-1)
    if metric.shape[0] == batch_size:
        return metric
    if metric.shape[0] == 1:
        return np.full((batch_size,), float(metric[0]), dtype=np.float64)
    raise ValueError(
        f"Unexpected cuRobo metric shape {metric.shape}; expected batch={batch_size}"
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class PandaMotionPlannerCurobo:
    """cuRobo-backed motion planner for a single Franka Panda arm (7-DOF).

    Return format is compatible with the YAM planner but uses ``positions``
    (N, 7) instead of separate left/right arrays.
    """

    DOF = 7
    ROBOT_TYPE = "panda"

    def __init__(
        self,
        robot_cfg_path: str | Path | None = None,
        device: str = "cuda:0",
        enable_finetune_trajopt: bool = False,
        solver_speed: str = _DEFAULT_SOLVER_SPEED,
        position_threshold: float = _DEFAULT_POSITION_THRESHOLD_M,
        rotation_threshold: float = _DEFAULT_ROTATION_THRESHOLD_RAD,
        cspace_threshold: float = _DEFAULT_CSPACE_THRESHOLD_RAD,
        collision_cache_obb: int = _DEFAULT_COLLISION_CACHE_OBB,
        collision_cache_mesh: int = _DEFAULT_COLLISION_CACHE_MESH,
    ) -> None:
        self._robot_cfg_path = Path(robot_cfg_path or _FRANKA_CFG).resolve()
        self._device = device
        self._gripper: float = 1.0  # 0=closed, 1=open
        self._enable_finetune_trajopt = bool(enable_finetune_trajopt)
        self._solver_speed = _normalize_solver_speed(solver_speed)
        self._solver_preset = deepcopy(_SOLVER_PRESET_CONFIGS[self._solver_speed])
        self._position_threshold = float(position_threshold)
        self._rotation_threshold = float(rotation_threshold)
        self._cspace_threshold = float(cspace_threshold)
        self._collision_cache_obb = int(collision_cache_obb)
        self._collision_cache_mesh = int(collision_cache_mesh)

        self._imports = self._import_curobo_dependencies()
        self._torch = self._imports["torch"]
        if not self._torch.cuda.is_available():
            raise RuntimeError(
                "cuRobo planner requires CUDA, but torch.cuda.is_available() is False"
            )

        self._TensorDeviceType = self._imports["TensorDeviceType"]
        self._Pose = self._imports["Pose"]
        self._JointState = self._imports["JointState"]
        self._WorldConfig = self._imports["WorldConfig"]
        self._CollisionCheckerType = self._imports["CollisionCheckerType"]
        self._MotionGen = self._imports["MotionGen"]
        self._MotionGenConfig = self._imports["MotionGenConfig"]
        self._MotionGenPlanConfig = self._imports["MotionGenPlanConfig"]
        self._MotionGenStatus = self._imports["MotionGenStatus"]

        self._tensor_args = self._TensorDeviceType(
            device=self._torch.device(self._device)
        )
        self._joint_names: list[str] | None = None
        self._joint_limit_lower: np.ndarray | None = None
        self._joint_limit_upper: np.ndarray | None = None
        self._batch_planner_capacity = int(_DEFAULT_BATCH_PLANNER_CAPACITY)
        self._motion_gen = None
        self._plan_config_single = None
        self._plan_config_batch = None
        self._setup_motion_gen()
        print(
            "[PandaCurobo] initialized "
            f"device={self._device} "
            f"solver_speed={self._solver_speed} "
            f"batch_capacity={self._batch_planner_capacity} "
            f"finetune={'on' if self._enable_finetune_trajopt else 'off'} "
            f"collision_cache={{'obb': {self._collision_cache_obb}, 'mesh': {self._collision_cache_mesh}}}"
        )

    # ------------------------------------------------------------------
    # cuRobo dependency loading
    # ------------------------------------------------------------------

    @staticmethod
    def _import_curobo_dependencies() -> dict[str, Any]:
        os.environ.setdefault(
            "CUROBO_TORCH_COMPILE_DISABLE",
            _CUROBO_TORCH_COMPILE_DISABLE_DEFAULT,
        )
        _silence_curobo_import_logs()
        _ensure_local_curobo_on_syspath()
        import torch
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.util.logger import setup_curobo_logger
        from curobo.wrap.reacher.motion_gen import (
            MotionGen,
            MotionGenConfig,
            MotionGenPlanConfig,
            MotionGenStatus,
        )

        setup_curobo_logger("error")
        return {
            "torch": torch,
            "CollisionCheckerType": CollisionCheckerType,
            "WorldConfig": WorldConfig,
            "TensorDeviceType": TensorDeviceType,
            "Pose": Pose,
            "JointState": JointState,
            "MotionGen": MotionGen,
            "MotionGenConfig": MotionGenConfig,
            "MotionGenPlanConfig": MotionGenPlanConfig,
            "MotionGenStatus": MotionGenStatus,
        }

    # ------------------------------------------------------------------
    # Robot & world config
    # ------------------------------------------------------------------

    def _load_robot_cfg(self) -> dict[str, Any]:
        robot_cfg = deepcopy(
            yaml.safe_load(self._robot_cfg_path.read_text())["robot_cfg"]
        )
        kin = robot_cfg["kinematics"]
        kin["use_usd_kinematics"] = False
        kin["usd_path"] = ""
        kin["isaac_usd_path"] = ""
        # Resolve URDF / asset root relative to the curobo content/assets directory
        assets_root = _CUROBO_CONTENT / "assets"
        urdf_rel = kin.get("urdf_path", "")
        asset_root_rel = kin.get("asset_root_path", "")
        kin["urdf_path"] = str(assets_root / urdf_rel)
        kin["asset_root_path"] = str(assets_root / asset_root_rel)
        # franka.yml uses ee_link="panda_hand" (wrist frame). Robosuite's
        # eef_pos reports the grip_site (fingertip), ~10cm below panda_hand.
        # We keep panda_hand as EE and apply a constant offset in the script
        # when transforming targets. This avoids frame convention mismatches
        # between cuRobo's panda_hand and robosuite's grip_site orientations.

        all_joint_names = list(kin["cspace"]["joint_names"])
        locked = set(kin.get("lock_joints", {}).keys())
        # cuRobo handles locked joints internally — only track active joints
        self._joint_names = [n for n in all_joint_names if n not in locked]
        return robot_cfg

    def _build_world_cfg(self):
        """Minimal world with a ground plane — cuRobo MESH checker requires >= 1 obstacle."""
        from curobo.geom.types import Cuboid

        ground = Cuboid(
            name="ground_plane",
            pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],  # 5cm below origin
            dims=[10.0, 10.0, 0.01],
        )
        return self._WorldConfig(cuboid=[ground])

    def update_world_from_sim(
        self,
        sim,
        base_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        exclude_body_prefixes: tuple[str, ...] = ("robot0", "gripper", "mobilebase"),
        max_dist: float = 1.2,
        min_size: float = 0.03,
    ) -> int:
        """Extract MuJoCo collision geoms and update cuRobo's world config.

        Filters environment collision geoms (group 0) by distance from arm base,
        converts boxes and cylinders to cuRobo Cuboids, and transforms to the
        arm base frame.

        Args:
            sim: robosuite sim object (has .model._model and .data._data)
            base_pos: arm base position in world frame (3,)
            base_quat_xyzw: arm base quaternion in world frame (4,) xyzw
            exclude_body_prefixes: body name prefixes to skip (robot parts)
            max_dist: only include geoms within this distance from base (meters)
            min_size: skip geoms smaller than this (meters)

        Returns:
            Number of obstacles added.
        """
        import mujoco
        from curobo.geom.types import Cuboid
        from scipy.spatial.transform import Rotation as R

        if self._motion_gen is None:
            raise RuntimeError("Motion generator is not initialized")

        model = sim.model._model
        data = sim.data._data
        base_pos = np.asarray(base_pos, dtype=np.float64)
        R_base = R.from_quat(np.asarray(base_quat_xyzw, dtype=np.float64))

        cuboids = []
        # Always include ground plane
        cuboids.append(
            Cuboid(
                name="ground_plane",
                pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
                dims=[10.0, 10.0, 0.01],
            )
        )

        for i in range(model.ngeom):
            # Only collision geoms (group 0)
            if model.geom_group[i] != 0:
                continue

            gtype = int(model.geom_type[i])
            # Only handle box (6) and cylinder (5) — skip mesh/sphere/etc for now
            if gtype not in (5, 6):
                continue

            # Skip robot geoms
            body_id = model.geom_bodyid[i]
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            if any(body_name.startswith(p) for p in exclude_body_prefixes):
                continue

            # Get world-frame pose
            pos_world = data.geom_xpos[i].copy()
            rot_world = data.geom_xmat[i].reshape(3, 3).copy()

            # Distance filter
            dist = float(np.linalg.norm(pos_world[:2] - base_pos[:2]))
            if dist > max_dist:
                continue

            # Size filter
            size = model.geom_size[i].copy()
            if gtype == 6:  # box: size = half-extents
                dims = size * 2.0
            elif gtype == 5:  # cylinder: size = [radius, half-height, 0]
                dims = np.array([size[0] * 2, size[0] * 2, size[1] * 2])
            else:
                continue

            if np.max(dims) < min_size:
                continue

            # Skip all thin slabs (counter tops, shelves, doors, walls).
            # These are thin in one dimension and wide in the other two.
            sorted_dims = sorted(dims)
            if sorted_dims[0] < 0.05 and sorted_dims[1] > 0.15:
                continue

            # Transform to arm base frame
            pos_base = R_base.inv().apply(pos_world - base_pos)
            R_geom_world = R.from_matrix(rot_world)
            R_geom_base = R_base.inv() * R_geom_world
            quat_base_wxyz = R_geom_base.as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz

            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            pose = [
                float(pos_base[0]),
                float(pos_base[1]),
                float(pos_base[2]),
                float(quat_base_wxyz[0]),
                float(quat_base_wxyz[1]),
                float(quat_base_wxyz[2]),
                float(quat_base_wxyz[3]),
            ]
            cuboids.append(Cuboid(name=name, pose=pose, dims=dims.tolist()))

        world_cfg = self._WorldConfig(cuboid=cuboids)
        self._motion_gen.update_world(world_cfg)
        n = len(cuboids)
        print(f"[PandaCurobo] Updated world: {n} obstacles ({n - 1} from sim + ground)")
        return n

    def update_world_from_geoms(
        self,
        payload: dict,
        base_pos: np.ndarray | list | None = None,
        base_quat_xyzw: np.ndarray | list | None = None,
    ) -> int:
        """Update cuRobo world from numpy-array payload (from CapServer RPC).

        Args:
            payload: dict with keys: names (list[str]), positions (N,3),
                     rot_mats (N,3,3), dims_array (N,3), base_pos (3,),
                     base_quat_xyzw (4,)
            base_pos: override base position (optional)
            base_quat_xyzw: override base quaternion (optional)

        Returns:
            Number of obstacles added.
        """
        from curobo.geom.types import Cuboid
        from scipy.spatial.transform import Rotation as R

        if self._motion_gen is None:
            raise RuntimeError("Motion generator is not initialized")

        bp = np.asarray(
            base_pos if base_pos is not None else payload["base_pos"], dtype=np.float64
        )
        bq = np.asarray(
            base_quat_xyzw if base_quat_xyzw is not None else payload["base_quat_xyzw"],
            dtype=np.float64,
        )
        R_base = R.from_quat(bq)

        names = payload.get("names", [])
        positions = np.asarray(
            payload.get("positions", np.empty((0, 3))), dtype=np.float64
        )
        rot_mats = np.asarray(
            payload.get("rot_mats", np.empty((0, 3, 3))), dtype=np.float64
        )
        dims_arr = np.asarray(
            payload.get("dims_array", np.empty((0, 3))), dtype=np.float64
        )

        cuboids = [
            Cuboid(
                name="ground_plane",
                pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
                dims=[10.0, 10.0, 0.01],
            )
        ]

        for i in range(len(names)):
            dims = dims_arr[i]
            pos_world = positions[i]
            rot_world = rot_mats[i]

            # Transform to arm base frame
            pos_base = R_base.inv().apply(pos_world - bp)
            R_geom_base = R_base.inv() * R.from_matrix(rot_world)
            quat_base_wxyz = R_geom_base.as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz

            pose = [
                float(pos_base[0]),
                float(pos_base[1]),
                float(pos_base[2]),
                float(quat_base_wxyz[0]),
                float(quat_base_wxyz[1]),
                float(quat_base_wxyz[2]),
                float(quat_base_wxyz[3]),
            ]
            cuboids.append(Cuboid(name=names[i], pose=pose, dims=dims.tolist()))

        world_cfg = self._WorldConfig(cuboid=cuboids)
        self._motion_gen.update_world(world_cfg)
        n = len(cuboids)
        print(
            f"[PandaCurobo] Updated world: {n} obstacles ({n - 1} from geoms + ground)"
        )
        return n

    # ------------------------------------------------------------------
    # MotionGen setup
    # ------------------------------------------------------------------

    def _setup_motion_gen(self) -> None:
        robot_cfg = self._load_robot_cfg()
        world_cfg = self._build_world_cfg()
        motion_gen_preset = self._solver_preset["motion_gen"]
        motion_gen_cfg = self._MotionGenConfig.load_from_robot_config(
            deepcopy(robot_cfg),
            world_cfg,
            self._tensor_args,
            collision_checker_type=self._CollisionCheckerType.MESH,
            self_collision_check=True,
            self_collision_opt=True,
            # Pre-allocate a larger cuboid cache for full-scene RoboCasa updates.
            collision_cache={
                "obb": self._collision_cache_obb,
                "mesh": self._collision_cache_mesh,
            },
            use_cuda_graph=_USE_CUDA_GRAPH_BY_DEFAULT,
            interpolation_dt=_INTERPOLATION_DT,
            collision_activation_distance=0.01,
            num_ik_seeds=int(motion_gen_preset["num_ik_seeds"]),
            num_graph_seeds=int(motion_gen_preset["num_graph_seeds"]),
            num_trajopt_seeds=int(motion_gen_preset["num_trajopt_seeds"]),
            position_threshold=self._position_threshold,
            rotation_threshold=self._rotation_threshold,
            cspace_threshold=self._cspace_threshold,
            trajopt_tsteps=int(motion_gen_preset["trajopt_tsteps"]),
            maximum_trajectory_dt=0.5,
            fixed_iters_trajopt=True,
            ik_opt_iters=int(motion_gen_preset["ik_opt_iters"]),
            grad_trajopt_iters=int(motion_gen_preset["grad_trajopt_iters"]),
        )
        self._motion_gen = self._MotionGen(motion_gen_cfg)
        warmup_kwargs = {
            "enable_graph": _ENABLE_GRAPH_SEARCH_BY_DEFAULT,
            "warmup_js_trajopt": False,
            "batch": int(self._batch_planner_capacity),
        }
        self._motion_gen.warmup(**warmup_kwargs)
        joint_limits = (
            self._motion_gen.kinematics.kinematics_config.joint_limits.position.detach()
            .cpu()
            .numpy()
        )
        self._joint_limit_lower = np.asarray(joint_limits[0], dtype=np.float64)
        self._joint_limit_upper = np.asarray(joint_limits[1], dtype=np.float64)
        self._plan_config_single = self._build_plan_config(batch_mode=False)
        self._plan_config_batch = self._build_plan_config(batch_mode=True)

    def _build_plan_config(self, *, batch_mode: bool = False):
        plan_preset = self._solver_preset["plan"]
        return self._MotionGenPlanConfig(
            enable_graph=_ENABLE_GRAPH_SEARCH_BY_DEFAULT,
            enable_graph_attempt=int(plan_preset["enable_graph_attempt"]),
            enable_finetune_trajopt=self._enable_finetune_trajopt,
            max_attempts=int(plan_preset["max_attempts"]),
            timeout=float(plan_preset["timeout"]),
            time_dilation_factor=(
                plan_preset["time_dilation_factor_batch"]
                if batch_mode
                else plan_preset["time_dilation_factor_single"]
            ),
            use_start_state_as_retract=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_joint_state(self, jp: np.ndarray):
        """Wrap joint positions as cuRobo JointState.

        Args:
            jp: shape (batch, 7) or (7,) — active arm DOF only.
                cuRobo handles locked joints (fingers) internally.
        """
        assert self._joint_names is not None
        q = np.asarray(jp, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return self._JointState.from_position(
            self._tensor_args.to_device(q),
            joint_names=self._joint_names,
        )

    def _make_pose(self, pos: np.ndarray, quat_xyzw: np.ndarray):
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        quat_xyzw = quat_xyzw / np.maximum(
            np.linalg.norm(quat_xyzw, axis=-1, keepdims=True), 1e-12
        )
        return self._Pose(
            position=self._tensor_args.to_device(np.asarray(pos, dtype=np.float32)),
            quaternion=self._tensor_args.to_device(
                _xyzw_to_wxyz(quat_xyzw).astype(np.float32)
            ),
        )

    def _joint_state_positions(self, plan) -> np.ndarray:
        """Extract (N, 7) joint positions from a cuRobo plan."""
        if self._motion_gen is None:
            raise RuntimeError("Motion generator is not initialized")
        plan = self._motion_gen.get_full_js(plan)
        if plan.joint_names is not None and self._joint_names is not None:
            plan = plan.get_ordered_joint_state(self._joint_names)
        positions = plan.position.detach().cpu().numpy()
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions[0]
        return np.asarray(positions, dtype=np.float64)

    def _pad_batch_array(
        self, arr: np.ndarray, *, target_batch_size: int, pad_value: np.ndarray
    ) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        actual = arr.shape[0]
        if actual >= target_batch_size:
            return arr[:target_batch_size]
        pad_rows = np.tile(pad_value.reshape(1, -1), (target_batch_size - actual, 1))
        return np.concatenate([arr, pad_rows], axis=0)

    def _clamp_joint_positions(self, q: np.ndarray) -> np.ndarray:
        if self._joint_limit_lower is None or self._joint_limit_upper is None:
            return q
        dof = self.DOF
        q_arm = q[..., :dof]
        q_clipped = np.clip(
            q_arm,
            self._joint_limit_lower[:dof],
            self._joint_limit_upper[:dof],
        )
        max_delta = float(np.max(np.abs(q_clipped - q_arm)))
        if max_delta > 1e-6:
            print(f"[PandaCurobo] Clipped joints to limits (max Δ={max_delta:.4f} rad)")
        return q_clipped

    # ------------------------------------------------------------------
    # Public: forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
        """Compute FK for a single configuration.

        Returns:
            {"position": ndarray(3,), "quaternion_xyzw": ndarray(4,)}
        """
        if self._motion_gen is None:
            raise RuntimeError("Motion generator is not initialized")
        js = self._make_joint_state(joint_pos)
        kin_state = self._motion_gen.kinematics.get_state(js.position)
        ee_pos = kin_state.ee_position.detach().cpu().numpy().reshape(-1)[:3]
        ee_quat_wxyz = kin_state.ee_quaternion.detach().cpu().numpy().reshape(-1)[:4]
        # Convert wxyz → xyzw
        ee_quat_xyzw = np.array(
            [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]],
            dtype=np.float64,
        )
        return {"position": ee_pos, "quaternion_xyzw": ee_quat_xyzw}

    # ------------------------------------------------------------------
    # Public: set_finetune_enabled / set_gripper_qpos
    # ------------------------------------------------------------------

    def set_finetune_enabled(self, enabled: bool) -> None:
        self._enable_finetune_trajopt = bool(enabled)
        if self._plan_config_single is not None:
            self._plan_config_single.enable_finetune_trajopt = (
                self._enable_finetune_trajopt
            )
        if self._plan_config_batch is not None:
            self._plan_config_batch.enable_finetune_trajopt = (
                self._enable_finetune_trajopt
            )

    def set_gripper_qpos(
        self,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> None:
        # Single arm — use right_gripper (or left as fallback)
        if right_gripper is not None:
            self._gripper = float(right_gripper)
        elif left_gripper is not None:
            self._gripper = float(left_gripper)

    # ------------------------------------------------------------------
    # Public: plan_to_pose  (single target)
    # ------------------------------------------------------------------

    def plan_to_pose(
        self,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        target_left_pos: np.ndarray | None = None,
        target_left_quat_xyzw: np.ndarray | None = None,
        target_right_pos: np.ndarray | None = None,
        target_right_quat_xyzw: np.ndarray | None = None,
        side: str = "right",
        validate_trajectory: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan single-target via the batch backend with batch size 1.

        Uses the same CUDA graph as batch planning — no graph reset needed.
        """
        batch_result = self.plan_batch_to_pose(
            current_left_jp=current_left_jp,
            current_right_jp=current_right_jp,
            target_left_pos=(
                None
                if target_left_pos is None
                else np.asarray(target_left_pos, dtype=np.float64).reshape(1, 3)
            ),
            target_left_quat_xyzw=(
                None
                if target_left_quat_xyzw is None
                else np.asarray(target_left_quat_xyzw, dtype=np.float64).reshape(1, 4)
            ),
            target_right_pos=(
                None
                if target_right_pos is None
                else np.asarray(target_right_pos, dtype=np.float64).reshape(1, 3)
            ),
            target_right_quat_xyzw=(
                None
                if target_right_quat_xyzw is None
                else np.asarray(target_right_quat_xyzw, dtype=np.float64).reshape(1, 4)
            ),
            side=side,
            validate_trajectory=validate_trajectory,
            **kwargs,
        )

        status_list = batch_result.get("status_by_index", [])
        status = status_list[0] if status_list else "Planning_Failed"
        detail_list = batch_result.get("status_detail_by_index", [])
        detail = detail_list[0] if detail_list else None

        pos_err_arr = np.asarray(
            batch_result.get("position_error_m", [float("nan")]), dtype=np.float64
        ).ravel()
        rot_err_arr = np.asarray(
            batch_result.get("rotation_error_deg", [float("nan")]), dtype=np.float64
        ).ravel()

        positions_list = batch_result.get("right_positions_by_index", [None])
        left_positions_list = batch_result.get("left_positions_by_index", [None])

        timing_keys = {k: v for k, v in batch_result.items() if k.startswith("curobo_")}

        if status != "Success":
            return {
                "status": status,
                "status_detail": detail,
                "position": np.empty((0, self.DOF)),
                "left_positions": np.empty((0, self.DOF)),
                "right_positions": np.empty((0, self.DOF)),
                "position_error_m": float(pos_err_arr[0]),
                "rotation_error_deg": float(rot_err_arr[0]),
                **timing_keys,
            }

        right_positions = positions_list[0]
        if right_positions is None:
            right_positions = np.empty((0, self.DOF))
        else:
            right_positions = np.asarray(right_positions, dtype=np.float64)
            if right_positions.ndim < 2:
                right_positions = right_positions.reshape(-1, self.DOF)

        left_positions = left_positions_list[0] if left_positions_list else None
        if left_positions is None:
            left_positions = np.zeros_like(right_positions)
        else:
            left_positions = np.asarray(left_positions, dtype=np.float64)
            if left_positions.ndim < 2:
                left_positions = left_positions.reshape(-1, self.DOF)

        return {
            "status": "Success",
            "status_detail": None,
            "position": right_positions,
            "left_positions": left_positions,
            "right_positions": right_positions,
            "position_error_m": float(pos_err_arr[0]),
            "rotation_error_deg": float(rot_err_arr[0]),
            **timing_keys,
        }

    # ------------------------------------------------------------------
    # Public: plan_batch_to_pose  (batch targets — for grasp ranking)
    # ------------------------------------------------------------------

    def plan_batch_to_pose(
        self,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        target_left_pos: np.ndarray | None = None,
        target_left_quat_xyzw: np.ndarray | None = None,
        target_right_pos: np.ndarray | None = None,
        target_right_quat_xyzw: np.ndarray | None = None,
        side: str = "right",
        validate_trajectory: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """Batch planning — same bimanual signature for portal compatibility."""
        target_pos = (
            target_right_pos if target_right_pos is not None else target_left_pos
        )
        target_quat = (
            target_right_quat_xyzw
            if target_right_quat_xyzw is not None
            else target_left_quat_xyzw
        )
        if target_pos is None or target_quat is None:
            return self._error_result("IK_Failed", "No target pose provided")

        current_jp = current_right_jp
        if current_jp is None or len(current_jp) == 0:
            current_jp = current_left_jp

        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1, 3)
        target_quat = np.asarray(target_quat, dtype=np.float64).reshape(-1, 4)
        batch_size = target_pos.shape[0]
        current_jp = np.asarray(current_jp, dtype=np.float64).ravel()[: self.DOF]

        from scipy.spatial.transform import Rotation as _R

        ee_z = _R.from_quat(target_quat).apply([0, 0, 1])
        target_pos = target_pos - _GRIP_TO_HAND_OFFSET * ee_z

        # Split into chunks of _batch_planner_capacity
        results = []
        for start in range(0, batch_size, self._batch_planner_capacity):
            end = min(start + self._batch_planner_capacity, batch_size)
            chunk_result = self._plan_batch_chunk(
                current_jp=current_jp,
                target_pos=target_pos[start:end],
                target_quat_xyzw=target_quat[start:end],
            )
            results.append(chunk_result)

        return self._merge_batch_results(results)

    # ------------------------------------------------------------------
    # Internal: batch planning chunk
    # ------------------------------------------------------------------

    def _plan_batch_chunk(
        self,
        *,
        current_jp: np.ndarray,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
    ) -> dict[str, Any]:
        if self._motion_gen is None or self._plan_config_batch is None:
            raise RuntimeError("Motion generator is not initialized")

        actual_batch_size = target_pos.shape[0]
        fixed_batch_size = int(self._batch_planner_capacity)

        # Pad to fixed batch size
        pad_pos = target_pos[-1]
        pad_quat = target_quat_xyzw[-1]
        padded_pos = self._pad_batch_array(
            target_pos, target_batch_size=fixed_batch_size, pad_value=pad_pos
        )
        padded_quat = self._pad_batch_array(
            target_quat_xyzw, target_batch_size=fixed_batch_size, pad_value=pad_quat
        )

        start_state = self._make_joint_state(
            np.tile(current_jp.reshape(1, self.DOF), (fixed_batch_size, 1))
        )
        goal_pose = self._make_pose(padded_pos, padded_quat)

        plan_cfg = self._plan_config_batch.clone()
        print(
            f"[PandaCurobo] plan_batch actual={actual_batch_size} "
            f"fixed={fixed_batch_size} solver={self._solver_speed}"
        )

        try:
            result = self._motion_gen.plan_batch(start_state, goal_pose, plan_cfg)
        except Exception as exc:
            raise RuntimeError(
                f"cuRobo batch plan failed (batch={actual_batch_size}): {exc}"
            ) from exc

        # Parse results
        success_all = (
            np.asarray(result.success.detach().cpu().numpy(), dtype=bool).reshape(-1)
            if result.success is not None
            else np.zeros((fixed_batch_size,), dtype=bool)
        )
        success = success_all[:actual_batch_size].copy()

        status_detail = getattr(result.status, "value", str(result.status))
        status_detail = None if status_detail in {"None", "null"} else status_detail
        failed_status = (
            "IK_Failed"
            if result.status == self._MotionGenStatus.IK_FAIL
            else "Planning_Failed"
        )
        status_by_index = np.where(success, "Success", failed_status).tolist()
        positions_by_index: list[np.ndarray | None] = [None] * actual_batch_size

        if np.any(success):
            interpolated_plan = getattr(result, "interpolated_plan", None)
            if interpolated_plan is not None:
                plan_batch = (
                    result.get_paths()
                    if getattr(result, "path_buffer_last_tstep", None) is not None
                    else [interpolated_plan[idx] for idx in range(fixed_batch_size)]
                )
                for idx, ok in enumerate(success.tolist()):
                    if not ok:
                        continue
                    positions = self._joint_state_positions(plan_batch[idx])
                    # Take only the arm DOF columns (exclude locked finger joints)
                    positions_by_index[idx] = positions[:, : self.DOF]

        timing_info = _curobo_timing_info(result)
        batch_position_error = _as_numpy_metric_array(
            getattr(result, "position_error", None), fixed_batch_size
        )[:actual_batch_size]
        batch_rotation_error_rad = _as_numpy_metric_array(
            getattr(result, "rotation_error", None), fixed_batch_size
        )[:actual_batch_size]

        status_detail_by_index: list[str | None] = [
            None if s == "Success" else (status_detail or s) for s in status_by_index
        ]
        left_positions_by_index: list[np.ndarray | None] = [
            np.zeros_like(p) if p is not None else None for p in positions_by_index
        ]

        return {
            "status": (
                "Success"
                if np.all(success)
                else ("Partial_Success" if np.any(success) else "Planning_Failed")
            ),
            "status_detail": status_detail,
            "success_mask": success,
            "status_by_index": status_by_index,
            "status_detail_by_index": status_detail_by_index,
            "position_error_m": batch_position_error,
            "rotation_error_deg": np.rad2deg(batch_rotation_error_rad),
            "left_positions_by_index": left_positions_by_index,
            "right_positions_by_index": positions_by_index,
            **timing_info,
        }

    # ------------------------------------------------------------------
    # Merge batch chunks
    # ------------------------------------------------------------------

    def _merge_batch_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return self._error_result("Planning_Failed", "No results")
        if len(results) == 1:
            return results[0]
        merged_success = np.concatenate([r["success_mask"] for r in results])
        merged_status_by_index = sum((r["status_by_index"] for r in results), [])
        merged_status_detail_by_index = sum(
            (
                r.get("status_detail_by_index", [None] * len(r["status_by_index"]))
                for r in results
            ),
            [],
        )
        merged_left_positions = sum(
            (
                r.get("left_positions_by_index", [None] * len(r["status_by_index"]))
                for r in results
            ),
            [],
        )
        merged_right_positions = sum(
            (
                r.get("right_positions_by_index", [None] * len(r["status_by_index"]))
                for r in results
            ),
            [],
        )
        merged_pos_err = np.concatenate([r["position_error_m"] for r in results])
        merged_rot_err = np.concatenate([r["rotation_error_deg"] for r in results])
        timing = {k: v for k, v in results[0].items() if k.startswith("curobo_")}
        return {
            "status": (
                "Success"
                if np.all(merged_success)
                else (
                    "Partial_Success" if np.any(merged_success) else "Planning_Failed"
                )
            ),
            "status_detail": results[0].get("status_detail"),
            "success_mask": merged_success,
            "status_by_index": merged_status_by_index,
            "status_detail_by_index": merged_status_detail_by_index,
            "position_error_m": merged_pos_err,
            "rotation_error_deg": merged_rot_err,
            "left_positions_by_index": merged_left_positions,
            "right_positions_by_index": merged_right_positions,
            **timing,
        }

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _error_result(self, status: str, detail: str) -> dict[str, Any]:
        return {
            "status": status,
            "status_detail": detail,
            "position": np.empty((0, self.DOF)),
            "left_positions": np.empty((0, self.DOF)),
            "right_positions": np.empty((0, self.DOF)),
            "position_error_m": float("nan"),
            "rotation_error_deg": float("nan"),
            "success_mask": np.array([False]),
            "status_by_index": [status],
            "status_detail_by_index": [detail],
            "left_positions_by_index": [None],
            "right_positions_by_index": [None],
            "curobo_solve_time_ms": 0.0,
            "curobo_total_time_ms": 0.0,
            "curobo_ik_time_ms": 0.0,
            "curobo_graph_time_ms": 0.0,
            "curobo_trajopt_time_ms": 0.0,
            "curobo_finetune_time_ms": 0.0,
            "curobo_attempts": 0,
            "curobo_trajopt_attempts": 0,
            "curobo_used_graph": False,
        }
