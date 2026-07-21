"""
Base class for YAM bimanual robot station. Contains code shared between simulation and real environments.

Mostly adapted from `yam_env.py` in the starter code.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from enpire.env.forge.robot.camera_factory import center_square_crop, crop_image_region
from enpire.env.forge.robot.constants import (
    DEFAULT_COMPRESSED_VIDEO_SHAPE,
    DEFAULT_RESET_JOINT_STATE,
    HOVER_EE_POSE,
    RESET_TARGET_EE_POSE_DEFAULT_MAX_IK_ITERS,
    RESET_TARGET_EE_POSE_DEFAULT_POSITION_TOLERANCE_M,
    RESET_TARGET_EE_POSE_DEFAULT_QUAT_TOLERANCE,
)
from enpire.env.forge.robot.yam.kinematics import (
    YamKinematics,
    _quat_xyzw_to_rot6d,
    _rot6d_to_quat_xyzw,
    _rot6d_to_rot_matrix,
    _rpy_display_to_quat_xyzw,
)

DELTA_EE_CONTROL_MODES = ("delta_ee_pose", "delta_ee_pose_translation")


class _BaseYamEnv(gym.Env):
    CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640

    def __init__(
        self,
        control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        ] = "joint_position",
        enable_cameras: bool = True,
        enabled_camera_names: Tuple[str, ...] = ("top", "left", "right"),
        enabled_sides: (Literal["left", "right", "both"] | Sequence[str] | None) = None,
        crop_camera_names: Tuple[str, ...] = (),
        crop_region: Tuple[str, ...] = ("center",),
        delta_ee_translation_xyz_max: Tuple[float, float, float] = (
            0.0003,
            0.0003,
            0.0006,
        ),
    ):
        self.control_mode = control_mode
        self._delta_ee_translation_xyz_max = np.asarray(
            delta_ee_translation_xyz_max, dtype=np.float32
        ).reshape(3)
        self._is_delta_ee_mode = self.control_mode in (
            "delta_ee_pose",
            "delta_ee_pose_translation",
        )
        self._is_delta_mode = self.control_mode in (
            "delta_joint_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        )
        self._use_cartesian_proprioception = self.control_mode in (
            "cartesian_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        )
        self.enable_cameras = bool(enable_cameras)
        self.camera_names = enabled_camera_names
        self.crop_camera_names, self.crop_region = (
            tuple(str(x) for x in crop_camera_names),
            tuple(str(x) for x in crop_region),
        )
        self._delta_ee_translation_active_sides = self._normalize_delta_ee_translation_sides(
            enabled_sides
        )
        # Track current joint positions for delta control modes
        self._current_joint_pos = {
            "left_joint_pos": np.zeros(6, dtype=np.float32),
            "right_joint_pos": np.zeros(6, dtype=np.float32),
        }
        # In delta_ee_pose mode, accumulate deltas on the last commanded EE
        # target (_delta_ee_mode_last_commanded_ee_pose) instead of the measured EE pose.
        # This avoids feeding tracking
        # error back into the next command and reduces drift during policy
        # rollout. The measured joints are still used as IK seeds.
        self.last_commanded_ee_pose: dict[str, dict[str, np.ndarray]] | None = None
        self.last_commanded_gripper: dict[str, np.ndarray] | None = None
        self._last_observed_ee_pos: dict[str, np.ndarray] | None = None
        self._last_delta_ee_pos = {
            "left": np.zeros(3, dtype=np.float32),
            "right": np.zeros(3, dtype=np.float32),
        }
        self._delta_ee_orientation_filter_sides: set[str] = set()
        self._delta_ee_orientation_filter_quat_xyzw: np.ndarray | None = None
        # Grasp-site offset along TCP local +Z (metres).
        # When non-zero, delta_ee_pose commands are applied at the *grasp site*
        # (fingertips) instead of the raw TCP (gripper base).  This gives more
        # intuitive control when teleoperating with grippers.
        # Set via ``set_grasp_site_offset(m)`` — default 0.0 for backward compat.
        self._grasp_site_offset_m: float = 0.0
        from enpire.env.forge.robot.models.station.paths import get_station_xml

        model_path = get_station_xml()
        self._spec = mujoco.MjSpec.from_file(str(model_path))
        self._spec.copy_during_attach = True
        self._spec = self._build_task_spec(self._spec)

        # Change camera resolution to 640x480
        for camera_name in self.camera_names:
            model_camera_name = self._resolve_model_camera_name(camera_name)
            camera = next(x for x in self._spec.cameras if x.name == model_camera_name)
            camera.resolution = (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
            camera.sensor_size = (0.003148, 0.002364)  # FOV from rs.rs2_fov()

        # Compile model
        self._model = self._spec.compile()

        # Actuator mappings
        left_actuator_names = [x.name for x in self._spec.actuators if x.name.startswith("left_")]
        right_actuator_names = [x.name for x in self._spec.actuators if x.name.startswith("right_")]
        self.left_actuator_ids = [self._model.actuator(name).id for name in left_actuator_names]
        self.right_actuator_ids = [self._model.actuator(name).id for name in right_actuator_names]

        # Kinematics
        self._kinematics = YamKinematics()
        self.kin = self._kinematics
        self._kin_lock = threading.RLock()

        # Observation and action spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _resolve_model_camera_name(self, camera_name: str) -> str:
        """Resolve a logical observation camera to the MuJoCo camera name.

        ``left_third`` is the fixed left/top third-view camera.  Legacy
        ``left`` remains the physical left wrist camera, while ``left_wrist``
        is an explicit alias to the same wrist model frame.
        """
        logical = str(camera_name).strip().lower()
        available = [str(x.name) for x in self._spec.cameras]

        def _env(name: str) -> str | None:
            value = os.environ.get(name, "").strip()
            return value or None

        candidate_map = {
            "top": (
                _env("CAP_TOP_CAMERA_FRAME"),
                "top_camera_d405",
                "top_camera_d435",
                "top_camera_zed2i",
            ),
            "left": (
                _env("CAP_LEFT_CAMERA_FRAME"),
                "left_camera_d405",
            ),
            "left_third": (
                _env("CAP_LEFT_THIRD_CAMERA_FRAME"),
                "top_camera_left_d435",
                "top_camera_left_d405",
            ),
            "left_fixed": (
                _env("CAP_LEFT_THIRD_CAMERA_FRAME"),
                "top_camera_left_d435",
                "top_camera_left_d405",
            ),
            "left_wrist": ("left_camera_d405",),
            "right": ("right_camera_d405",),
            "right_wrist": ("right_camera_d405",),
        }
        for candidate in candidate_map.get(logical, (camera_name,)):
            if candidate and candidate in available:
                return candidate

        if logical not in candidate_map:
            for name in available:
                if logical in name:
                    return name

        raise ValueError(
            f"No MuJoCo camera for logical camera {camera_name!r}. Available cameras: {available}"
        )

    def get_camera_extrinsics(self, camera_name: str) -> dict:
        from enpire.env.forge.robot.models.station.paths import (
            get_camera_extrinsics_override,
            needs_optical_flip,
        )

        override = get_camera_extrinsics_override(camera_name)
        if override is not None:
            return override

        frame_name = self._resolve_model_camera_name(camera_name)
        state = self._read_current_joint_state()
        qpos = self._kinematics.configuration.data.qpos.copy()
        qpos[:6] = np.asarray(state["left_joint_pos"], dtype=np.float64).reshape(6)
        qpos[8:14] = np.asarray(state["right_joint_pos"], dtype=np.float64).reshape(6)
        self._kinematics.configuration.update(qpos)
        transform = self._kinematics.configuration.get_transform_frame_to_world(
            frame_name,
            "body",
        )
        rot = transform.rotation()
        if hasattr(rot, "as_matrix"):
            rot_mat = rot.as_matrix()
        else:
            maybe_matrix = getattr(rot, "matrix", rot)
            rot_mat = maybe_matrix() if callable(maybe_matrix) else maybe_matrix
        return {
            "position": transform.translation().tolist(),
            "rotation": np.asarray(rot_mat, dtype=np.float64).reshape(3, 3).tolist(),
            "needs_optical_flip": needs_optical_flip(camera_name),
        }

    def _blank_camera_image(self) -> np.ndarray:
        return np.zeros((self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 3), dtype=np.uint8)

    def _crop_camera_image(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        if camera_name not in self.crop_camera_names:
            return image
        regions = dict(item.split(":", 1) for item in self.crop_region if ":" in item)
        region = tuple(
            (regions[camera_name] if regions else " ".join(self.crop_region))
            .replace(",", " ")
            .split()
        )
        return (
            center_square_crop(image) if region == ("center",) else crop_image_region(image, region)
        )

    def _build_task_spec(self, station_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Override this method to add task-specific objects to the scene."""
        return station_spec

    def _build_observation_space(self) -> gym.spaces.Dict:
        space = {}

        # Arm and gripper states
        for side in ["left", "right"]:
            if self.control_mode in (
                "joint_position",
                "delta_joint_position",
            ):
                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                )
            elif self.control_mode in ("cartesian_position", *DELTA_EE_CONTROL_MODES):
                space.update(
                    {
                        f"{side}_ee_pos": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        f"{side}_ee_rot6d": gym.spaces.Box(
                            low=-1,
                            high=1,
                            shape=(6,),
                            dtype=np.float32,
                        ),
                    }
                )

            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            )

        # Camera images
        for camera in self.camera_names:
            space[f"{camera}_camera_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    DEFAULT_COMPRESSED_VIDEO_SHAPE[1],
                    DEFAULT_COMPRESSED_VIDEO_SHAPE[0],
                    3,
                ),
                dtype=np.uint8,
            )

        return gym.spaces.Dict(space)

    def set_grasp_site_offset(self, offset_m: float) -> None:
        """Set the grasp-site offset for delta_ee_pose control.

        When ``offset_m > 0`` the delta commands are applied at a virtual
        *grasp site* located ``offset_m`` metres along the TCP's local +Z
        axis (i.e. further out along the finger/tool direction).  This makes
        teleoperation more intuitive because the user controls the fingertips
        rather than the gripper base.

        The offset only affects delta end-effector modes; other modes are unchanged.
        """
        self._grasp_site_offset_m = float(offset_m)

    @staticmethod
    def _normalize_delta_ee_translation_sides(
        sides: Literal["left", "right", "both"] | Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        if sides is None:
            return None
        if isinstance(sides, str):
            values: Sequence[str] = ("left", "right") if sides == "both" else (sides,)
        else:
            values = sides
        normalized = tuple(dict.fromkeys(str(side) for side in values))
        invalid = tuple(side for side in normalized if side not in ("left", "right"))
        if invalid:
            raise ValueError(
                f"enabled_sides must contain only 'left'/'right'/'both', got {invalid}"
            )
        return normalized

    def set_delta_ee_translation_mask(
        self,
        *,
        enabled_sides: Literal["left", "right", "both"] | Sequence[str] | None = None,
    ) -> None:
        self._delta_ee_translation_active_sides = self._normalize_delta_ee_translation_sides(
            enabled_sides
        )

    @property
    def enabled_sides(self) -> tuple[str, ...] | None:
        return self._delta_ee_translation_active_sides

    def set_control_mode(
        self,
        control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        ],
        observation: dict[str, np.ndarray] | None = None,
    ) -> None:
        if control_mode == self.control_mode:
            return
        self.control_mode = control_mode
        self._is_delta_ee_mode = self.control_mode in (
            "delta_ee_pose",
            "delta_ee_pose_translation",
        )
        self._is_delta_mode = self.control_mode in (
            "delta_joint_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        )
        self._use_cartesian_proprioception = self.control_mode in (
            "cartesian_position",
            "delta_ee_pose",
            "delta_ee_pose_translation",
        )
        # Rebuild spaces to match new mode
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()
        # Seed delta control state if joint positions are available
        if observation and "left_joint_pos" in observation and "right_joint_pos" in observation:
            self._current_joint_pos["left_joint_pos"] = observation["left_joint_pos"].copy()
            self._current_joint_pos["right_joint_pos"] = observation["right_joint_pos"].copy()
            self._record_ee_pose_cmd(observation)
            self._record_gripper_cmd(observation, overwrite=False)

    @staticmethod
    def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _array(
        value: Any,
        *,
        shape: tuple[int, ...],
        label: str,
        dtype: type[np.floating] = np.float32,
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=dtype).reshape(shape)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{label} must be finite, got {value!r}")
        return arr.copy()

    @classmethod
    def _normalize_joint_state(
        cls,
        joint_state: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, np.ndarray]:
        required = (
            "left_joint_pos",
            "left_gripper_pos",
            "right_joint_pos",
            "right_gripper_pos",
        )
        missing = [key for key in required if key not in joint_state]
        if missing:
            raise ValueError(f"{label} is missing keys: {missing}")
        return {
            "left_joint_pos": cls._array(
                joint_state["left_joint_pos"],
                shape=(6,),
                label=f"{label}.left_joint_pos",
            ),
            "left_gripper_pos": cls._array(
                joint_state["left_gripper_pos"],
                shape=(1,),
                label=f"{label}.left_gripper_pos",
            ),
            "right_joint_pos": cls._array(
                joint_state["right_joint_pos"],
                shape=(6,),
                label=f"{label}.right_joint_pos",
            ),
            "right_gripper_pos": cls._array(
                joint_state["right_gripper_pos"],
                shape=(1,),
                label=f"{label}.right_gripper_pos",
            ),
        }

    @staticmethod
    def _load_reset_target_ee_pose(path: str | Path) -> Mapping[str, Any]:
        target_path = Path(path).expanduser()
        if not target_path.exists():
            raise FileNotFoundError(f"reset target EE pose file not found: {target_path}")
        if target_path.suffix.lower() != ".json":
            raise ValueError(f"reset target EE pose files must be JSON for now; got {target_path}")
        data = json.loads(target_path.read_text())
        if not isinstance(data, Mapping):
            raise ValueError(f"reset target EE pose file must contain a JSON object: {target_path}")
        return data

    @classmethod
    def _side_ee_pose_from_mapping(
        cls,
        target: Mapping[str, Any],
        side: Literal["left", "right"],
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Get one arm's pos+rot6d+gripper from target mapping
        """
        side_payload = target.get(side, {})
        if side_payload is None:
            side_payload = {}
        if not isinstance(side_payload, Mapping):
            raise ValueError(f"target_ee_pose[{side!r}] must be an object")

        pos_value = cls._first_present(
            target,
            f"{side}_ee_pos",
            f"{side}_position",
            f"{side}_xyz",
        )
        quat_value = cls._first_present(
            target,
            f"{side}_ee_quat_xyzw",
            f"{side}_quat_xyzw",
            f"{side}_quaternion",
        )
        rot6d_value = cls._first_present(
            target,
            f"{side}_ee_rot6d",
            f"{side}_rot6d",
        )
        rpy_value = cls._first_present(
            target,
            f"{side}_ee_rpy_deg",
            f"{side}_rpy_deg",
            f"{side}_rpy",
        )
        grip_value = cls._first_present(
            target,
            f"{side}_gripper_pos",
            f"{side}_gripper",
        )

        if pos_value is None:
            pos_value = cls._first_present(side_payload, "ee_pos", "position", "pos", "xyz")
        if quat_value is None:
            quat_value = cls._first_present(
                side_payload,
                "ee_quat_xyzw",
                "quat_xyzw",
                "quaternion",
                "quat",
            )
        if rot6d_value is None:
            rot6d_value = cls._first_present(
                side_payload,
                "ee_rot6d",
                "rot6d",
            )
        if rpy_value is None:
            rpy_value = cls._first_present(
                side_payload,
                "ee_rpy_deg",
                "rpy_deg",
                "rpy",
                "global_rpy_deg",
            )
        if grip_value is None:
            grip_value = cls._first_present(side_payload, "gripper_pos", "gripper")

        orientation_count = sum(value is not None for value in (quat_value, rot6d_value, rpy_value))
        if orientation_count > 1:
            raise ValueError(
                f"{side} target EE pose must specify only one of quat_xyzw, rot6d, or rpy_deg"
            )
        orientation_value = (
            quat_value
            if quat_value is not None
            else rot6d_value
            if rot6d_value is not None
            else rpy_value
        )
        if (pos_value is None) != (orientation_value is None):
            missing = "orientation" if pos_value is not None else "position"
            raise ValueError(f"incomplete {side} target EE pose; missing {missing}")

        pos = (
            cls._array(pos_value, shape=(3,), label=f"{side}.target_ee_pos")
            if pos_value is not None
            else None
        )
        quat = None
        rot6d = None
        if quat_value is not None:
            quat = cls._array(quat_value, shape=(4,), label=f"{side}.target_ee_quat_xyzw")
        elif rot6d_value is not None:
            rot6d = cls._array(rot6d_value, shape=(6,), label=f"{side}.target_ee_rot6d")
            quat = _rot6d_to_quat_xyzw(rot6d)
        elif rpy_value is not None:
            quat = _rpy_display_to_quat_xyzw(
                cls._array(rpy_value, shape=(3,), label=f"{side}.target_ee_rpy_deg")
            )
        if quat is not None:
            norm = float(np.linalg.norm(quat))
            if norm < 1e-6:
                raise ValueError(f"{side}.target_ee_quat_xyzw has near-zero norm")
            quat = (quat / norm).astype(np.float32)
        gripper = (
            cls._array(grip_value, shape=(1,), label=f"{side}.target_gripper_pos")
            if grip_value is not None
            else None
        )
        if rot6d is None:
            rot6d = _quat_xyzw_to_rot6d(quat) if quat is not None else None
        return pos, rot6d, gripper

    def _get_joint_position_target_from_ee_pose_target(
        self,
        target_ee_pose: Mapping[str, Any],
        *,
        seed_joint_state: Mapping[str, Any],
        position_tolerance_m: float = RESET_TARGET_EE_POSE_DEFAULT_POSITION_TOLERANCE_M,
        quat_tolerance: float = RESET_TARGET_EE_POSE_DEFAULT_QUAT_TOLERANCE,
        max_iters: int = RESET_TARGET_EE_POSE_DEFAULT_MAX_IK_ITERS,
    ) -> dict[str, np.ndarray]:
        seed = self._normalize_joint_state(seed_joint_state, label="seed_joint_state")

        left_pos, left_rot6d, left_gripper = self._side_ee_pose_from_mapping(target_ee_pose, "left")
        left_quat = _rot6d_to_quat_xyzw(left_rot6d) if left_rot6d is not None else None

        right_pos, right_rot6d, right_gripper = self._side_ee_pose_from_mapping(
            target_ee_pose, "right"
        )
        right_quat = _rot6d_to_quat_xyzw(right_rot6d) if right_rot6d is not None else None

        left_active = left_pos is not None and left_quat is not None
        right_active = right_pos is not None and right_quat is not None
        if not (left_active or right_active):
            raise ValueError(
                "target_ee_pose must contain at least one complete side target "
                "(position + quat_xyzw)"
            )

        left_joint_pos, right_joint_pos, _, _ = self._kinematics.inverse_kinematics_full(
            left_pos if left_active else None,
            left_quat if left_active else None,
            right_pos if right_active else None,
            right_quat if right_active else None,
            left_seed=seed["left_joint_pos"],
            right_seed=seed["right_joint_pos"],
            max_iters=int(max_iters),
        )

        target_state = {
            "left_joint_pos": (
                np.asarray(left_joint_pos, dtype=np.float32).reshape(6)
                if left_active
                else seed["left_joint_pos"].copy()
            ),
            "left_gripper_pos": (
                left_gripper.copy() if left_gripper is not None else seed["left_gripper_pos"].copy()
            ),
            "right_joint_pos": (
                np.asarray(right_joint_pos, dtype=np.float32).reshape(6)
                if right_active
                else seed["right_joint_pos"].copy()
            ),
            "right_gripper_pos": (
                right_gripper.copy()
                if right_gripper is not None
                else seed["right_gripper_pos"].copy()
            ),
        }
        self._validate_target_joint_state(target_state, label="target_ee_pose IK result")
        self._validate_target_ee_pose_solution(
            target_state,
            left_pos=left_pos if left_active else None,
            left_quat=left_quat if left_active else None,
            right_pos=right_pos if right_active else None,
            right_quat=right_quat if right_active else None,
            position_tolerance_m=float(position_tolerance_m),
            quat_tolerance=float(quat_tolerance),
        )
        return target_state

    def _validate_target_joint_state(
        self,
        joint_state: Mapping[str, Any],
        *,
        label: str,
        tolerance: float = 1e-4,
    ) -> None:
        for side, actuator_ids in (
            ("left", self.left_actuator_ids),
            ("right", self.right_actuator_ids),
        ):
            joint_pos = np.asarray(joint_state[f"{side}_joint_pos"], dtype=np.float64).reshape(6)
            ctrl_ranges = np.asarray(
                self._model.actuator_ctrlrange[actuator_ids[:6]], dtype=np.float64
            )
            low = ctrl_ranges[:, 0] - float(tolerance)
            high = ctrl_ranges[:, 1] + float(tolerance)
            if np.any(joint_pos < low) or np.any(joint_pos > high):
                violations = [
                    f"j{i + 1}={value:.3f} not in [{lo:.3f}, {hi:.3f}]"
                    for i, (value, lo, hi) in enumerate(zip(joint_pos, low, high))
                    if value < lo or value > hi
                ]
                raise ValueError(f"{label} violates {side} joint limits: " + "; ".join(violations))

    def _validate_target_ee_pose_solution(
        self,
        joint_state: Mapping[str, Any],
        *,
        left_pos: np.ndarray | None,
        left_quat: np.ndarray | None,
        right_pos: np.ndarray | None,
        right_quat: np.ndarray | None,
        position_tolerance_m: float,
        quat_tolerance: float,
    ) -> None:
        fk_left_pos, fk_left_quat, fk_right_pos, fk_right_quat = (
            self._kinematics.forward_kinematics(
                np.asarray(joint_state["left_joint_pos"], dtype=np.float32).reshape(6),
                np.asarray(joint_state["right_joint_pos"], dtype=np.float32).reshape(6),
            )
        )
        checks = (
            ("left", left_pos, left_quat, fk_left_pos, fk_left_quat),
            ("right", right_pos, right_quat, fk_right_pos, fk_right_quat),
        )
        errors: list[str] = []
        for side, target_pos, target_quat, actual_pos, actual_quat in checks:
            if target_pos is None or target_quat is None:
                continue
            pos_err = float(
                np.linalg.norm(
                    np.asarray(actual_pos, dtype=np.float64)
                    - np.asarray(target_pos, dtype=np.float64)
                )
            )
            actual_q = np.asarray(actual_quat, dtype=np.float64).reshape(4)
            target_q = np.asarray(target_quat, dtype=np.float64).reshape(4)
            quat_err = min(
                float(np.linalg.norm(actual_q - target_q)),
                float(np.linalg.norm(actual_q + target_q)),
            )
            if pos_err > position_tolerance_m or quat_err > quat_tolerance:
                errors.append(f"{side}: pos_err={pos_err:.4f}m, quat_err={quat_err:.4f}")
        if errors:
            raise RuntimeError(
                "Seeded IK did not reach target EE pose within tolerance "
                f"(position_tolerance={position_tolerance_m:.3f}m, "
                f"quat_tolerance={quat_tolerance:.3f}): " + "; ".join(errors)
            )

    def _read_current_joint_state(self) -> dict[str, np.ndarray]:
        # Inherited by yam_real_env (read from follower arms) and yam_sim_env (read from mujoco.MjData)
        raise NotImplementedError

    def _resolve_reset_target_state(
        self,
        options: Mapping[str, Any] | None,
    ) -> dict[str, np.ndarray]:
        """
        Parse reset options to a concrete joint/gripper target.

        Accepted options:
          {"alias":"home"}: DEFAULT_RESET_JOINT_STATE
          {"alias":"hover"}: HOVER_EE_POSE solved through IK
          {"alias":"current"}: current measured joint/gripper state
          {"target_joint_position": {...}}: explicit joint/gripper state
          {"target_ee_pose": {...}}: explicit EE pose solved through IK
        """
        if options is None:
            reset_target_type = "target_joint_position"
            reset_target_value = DEFAULT_RESET_JOINT_STATE
        elif isinstance(options, Mapping):
            if options.get("alias", None) is not None:
                reset_alias = options["alias"].lower()
                if reset_alias == "home":
                    reset_target_type = "target_joint_position"
                    reset_target_value = DEFAULT_RESET_JOINT_STATE
                elif reset_alias == "hover":
                    reset_target_type = "target_ee_pose"
                    reset_target_value = HOVER_EE_POSE
                    # reset_target_type = "target_joint_position"
                    # reset_target_value = HOVER_JOINT_POSE
                elif reset_alias == "current":
                    reset_target_type = "target_joint_position"
                    reset_target_value = self._read_current_joint_state()
                else:
                    raise ValueError(
                        f"Unknown reset alias {options!r}; expected 'home', 'hover', or 'current'"
                    )
            else:
                target_keys = [
                    key for key in ("target_joint_position", "target_ee_pose") if key in options
                ]
                if len(target_keys) != 1:
                    raise ValueError(
                        "reset options must contain exactly one of "
                        "'target_joint_position' or 'target_ee_pose'"
                    )
                reset_target_type = target_keys[0]
                reset_target_value = options[reset_target_type]
        else:
            raise TypeError(
                f"reset options must be None, a string alias, or a mapping; got {type(options).__name__}"
            )

        if reset_target_type == "target_joint_position":
            return self._normalize_joint_state(
                reset_target_value,
                label="reset.target_joint_position",
            )

        if reset_target_type == "target_ee_pose":
            if not isinstance(reset_target_value, Mapping):
                raise ValueError("target_ee_pose must be a mapping")
            return self._get_joint_position_target_from_ee_pose_target(
                reset_target_value,
                seed_joint_state=self._read_current_joint_state(),
                position_tolerance_m=RESET_TARGET_EE_POSE_DEFAULT_POSITION_TOLERANCE_M,
                quat_tolerance=RESET_TARGET_EE_POSE_DEFAULT_QUAT_TOLERANCE,
                max_iters=RESET_TARGET_EE_POSE_DEFAULT_MAX_IK_ITERS,
            )

        raise RuntimeError(f"Unhandled reset target type: {reset_target_type}")

    def _update_current_joint_pos_from_observation(self, obs: dict[str, np.ndarray]) -> None:
        """Refresh IK/delta seeds from measured joint observations when present."""
        if "left_joint_pos" not in obs or "right_joint_pos" not in obs:
            return
        self._current_joint_pos["left_joint_pos"] = np.asarray(
            obs["left_joint_pos"], dtype=np.float32
        ).copy()
        self._current_joint_pos["right_joint_pos"] = np.asarray(
            obs["right_joint_pos"], dtype=np.float32
        ).copy()

    def _record_ee_pose_cmd(self, source: Mapping[str, Any]) -> None:
        # Use FK to get commanded end-effector pose from initial joint command.
        # Later we will add the delta ee command on the last end-effector pose command in delta ee pose mode.
        left_pos, left_quat, right_pos, right_quat = self._kinematics.forward_kinematics(
            np.asarray(source["left_joint_pos"], dtype=np.float32).reshape(6),
            np.asarray(source["right_joint_pos"], dtype=np.float32).reshape(6),
        )
        left_rot6d = _quat_xyzw_to_rot6d(left_quat)
        right_rot6d = _quat_xyzw_to_rot6d(right_quat)

        self.last_commanded_ee_pose = {
            "left": {
                "pos": np.asarray(left_pos, dtype=np.float32).reshape(3).copy(),
                "rot6d": np.asarray(left_rot6d, dtype=np.float32).reshape(6).copy(),
            },
            "right": {
                "pos": np.asarray(right_pos, dtype=np.float32).reshape(3).copy(),
                "rot6d": np.asarray(right_rot6d, dtype=np.float32).reshape(6).copy(),
            },
        }

    def _seed_delta_ee_command(self, observation: dict[str, np.ndarray]) -> None:
        """Seed the delta-EE command accumulator from an observed joint state."""
        if "left_joint_pos" in observation and "right_joint_pos" in observation:
            joint_source = observation
        else:
            joint_source = self._current_joint_pos
        self._record_ee_pose_cmd(joint_source)
        self._record_gripper_cmd(observation, overwrite=False)
        self._last_delta_ee_pos = {
            "left": np.zeros(3, dtype=np.float32),
            "right": np.zeros(3, dtype=np.float32),
        }

    def _record_gripper_cmd(self, source: Mapping[str, Any], *, overwrite: bool = True) -> None:
        recorded = dict(getattr(self, "last_commanded_gripper", None) or {})
        for side in ("left", "right"):
            grip_key = f"{side}_gripper_pos"
            if grip_key in source and (overwrite or side not in recorded):
                recorded[side] = np.asarray(source[grip_key], dtype=np.float32).reshape(1).copy()
        if recorded:
            self.last_commanded_gripper = recorded

    def set_delta_ee_orientation_filter(
        self,
        enabled_sides: tuple[str, ...] | list[str] | set[str],
        quat_xyzw: np.ndarray,
    ) -> None:
        sides = set(enabled_sides)
        invalid = sides.difference({"left", "right"})
        if invalid:
            raise ValueError(f"Invalid delta-EE orientation filter sides: {sorted(invalid)}")
        q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm < 1e-6:
            raise ValueError("delta-EE orientation filter quaternion has near-zero norm")
        self._delta_ee_orientation_filter_sides = sides
        self._delta_ee_orientation_filter_quat_xyzw = (q / norm).astype(np.float32)

    def clear_delta_ee_orientation_filter(self) -> None:
        self._delta_ee_orientation_filter_sides = set()
        self._delta_ee_orientation_filter_quat_xyzw = None

    def _build_action_space(self) -> gym.spaces.Dict:
        space = {}

        for side, actuator_ids in [
            ("left", self.left_actuator_ids),
            ("right", self.right_actuator_ids),
        ]:
            if self.control_mode == "joint_position":
                # Actuator control ranges
                ctrl_ranges = self._model.actuator_ctrlrange[actuator_ids]
                low = ctrl_ranges[:, 0].astype(np.float32)
                high = ctrl_ranges[:, 1].astype(np.float32)

                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=low[:6],
                    high=high[:6],
                    dtype=np.float32,
                )
            elif self.control_mode == "delta_joint_position":
                # Delta actions: relative changes to current joint positions
                # Use reasonable limits for delta (e.g., ±0.5 rad per step)
                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                )
            elif self.control_mode in ("cartesian_position", *DELTA_EE_CONTROL_MODES):
                space.update(
                    {
                        f"{side}_ee_pos": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        f"{side}_ee_rot6d": gym.spaces.Box(
                            low=-1,
                            high=1,
                            shape=(6,),
                            dtype=np.float32,
                        ),
                    }
                )

            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )

        return gym.spaces.Dict(space)

    def _convert_from_joint_to_cartesian(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # Extract joint positions
        left_joint_pos = obs["left_joint_pos"]
        right_joint_pos = obs["right_joint_pos"]
        del obs["left_joint_pos"]
        del obs["right_joint_pos"]

        # Run forward kinematics
        left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
            self._kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
        )

        # Construct end-effector pos and rotation matrix's 6d representation
        obs["left_ee_pos"] = left_ee_pos
        obs["left_ee_rot6d"] = _quat_xyzw_to_rot6d(left_ee_quat_xyzw)
        obs["right_ee_pos"] = right_ee_pos
        obs["right_ee_rot6d"] = _quat_xyzw_to_rot6d(right_ee_quat_xyzw)
        self._last_observed_ee_pos = {
            "left": np.asarray(left_ee_pos, dtype=np.float32).reshape(3).copy(),
            "right": np.asarray(right_ee_pos, dtype=np.float32).reshape(3).copy(),
        }

        return obs

    def _convert_from_cartesian_to_joint(
        self, action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        # Check if action is already in joint format (shouldn't happen in cartesian mode, but handle gracefully)
        if "left_joint_pos" in action and "right_joint_pos" in action:
            # Action is already in joint format, return as-is
            return action

        enabled_sides: dict[str, bool] = {}
        for side in ("left", "right"):
            pos_key = f"{side}_ee_pos"
            rot6d_key = f"{side}_ee_rot6d"
            has_pos = pos_key in action
            has_rot6d = rot6d_key in action
            if has_pos != has_rot6d:
                missing = rot6d_key if has_pos else pos_key
                raise ValueError(
                    f"Incomplete {side} EE pose action: expected both {pos_key} "
                    f"and {rot6d_key}; missing {missing}. Got keys: {list(action.keys())}"
                )
            enabled_sides[side] = has_pos

        left_active = enabled_sides["left"]
        right_active = enabled_sides["right"]
        if not (left_active or right_active):
            raise ValueError(
                f"Expected cartesian action with EE pose keys for at least one arm, but got keys: {list(action.keys())}"
            )

        converted = action.copy()
        left_ee_pos = converted.pop("left_ee_pos", None)
        left_ee_rot6d = converted.pop("left_ee_rot6d", None)
        left_ee_quat_xyzw = (
            _rot6d_to_quat_xyzw(left_ee_rot6d) if left_ee_rot6d is not None else None
        )
        right_ee_pos = converted.pop("right_ee_pos", None)
        right_ee_rot6d = converted.pop("right_ee_rot6d", None)
        right_ee_quat_xyzw = (
            _rot6d_to_quat_xyzw(right_ee_rot6d) if right_ee_rot6d is not None else None
        )

        # Run inverse kinematics. For single-arm actions, inverse_kinematics_full
        # holds the inactive arm at the current joint seed.
        left_joint_pos, right_joint_pos, _, _ = self._kinematics.inverse_kinematics_full(
            left_ee_pos if left_active else None,
            left_ee_quat_xyzw if left_active else None,
            right_ee_pos if right_active else None,
            right_ee_quat_xyzw if right_active else None,
            left_seed=self._current_joint_pos["left_joint_pos"],
            right_seed=self._current_joint_pos["right_joint_pos"],
        )

        # Construct joint positions
        converted["left_joint_pos"] = left_joint_pos
        converted["right_joint_pos"] = right_joint_pos

        return converted

    def _convert_from_delta_joint_to_absolute(
        self, action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert delta joint actions to absolute by adding the last joint position command.

        Args:
            action: Action dict with delta joint position command (keys: left_joint_pos, right_joint_pos)

        Returns:
            Action dict with absolute joint positions
        """
        raise NotImplementedError

    def _convert_from_delta_ee_to_cartesian(
        self, action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert delta EE pose actions to absolute EE poses.

        Args:
            action: commanded delta end-effector change (with absolute gripper width commands)


        This is the inverse operation of the conversion file at
        ``experimental/convert_joint_parquet_to_ee_delta_pose.py``.

        Uses the previously commanded end-effector pose as the reference, then
        applies the delta:
            new_pos  = last_command_pos  + delta_pos
            new_quat = delta_quat * last_command_quat   (quaternion pre-multiply)

        In ``delta_ee_pose_translation`` mode, XYZ deltas are clipped before
        this conversion; rotations are still interpreted as relative deltas.

        The command accumulator must be seeded explicitly by reset, a control
        mode transition, or a fixed-start commanded joint target. Refusing an
        unseeded conversion prevents silently switching back to measured FK,
        which would reintroduce drift in policy rollout.

        When ``_grasp_site_offset_m > 0`` the deltas are applied at a virtual
        *grasp site* (offset along TCP local +Z) and then projected back to
        the TCP so that IK targets the correct link.  Concretely::

            grasp_pos  = tcp_pos + tcp_rot @ [0, 0, offset]
            new_grasp  = grasp_pos + delta_pos
            new_rot    = delta_rot * tcp_rot
            new_tcp    = new_grasp - new_rot @ [0, 0, offset]

        The result is an absolute cartesian action that can be fed into the
        existing ``_convert_from_cartesian_to_joint`` pipeline.

        Grippers are assumed to be absolute and are passed through unchanged.

        Args:
            action: Dict with delta EE keys:
                ``{side}_ee_pos`` (3,), ``{side}_ee_rot6d`` (6,),
                ``{side}_gripper_pos`` (1,).

        Returns:
            Dict with the same keys but values replaced by **absolute** EE poses.
        """
        action = action.copy()
        translation_mode = self.control_mode == "delta_ee_pose_translation"
        if translation_mode:
            action = self._translation_mode_mask(action)

        if self.last_commanded_ee_pose is None:
            raise RuntimeError(
                "delta_ee_pose command accumulator is not seeded. Refusing to "
                "silently use measured FK as the command reference. Seed with "
                "seed_delta_ee_command_pose_from_joint_state(...) after fixed "
                "start, or seed_delta_ee_command_pose_from_observation(...) "
                "during reset/control-mode setup."
            )

        new_commanded_ee_pose = action.copy()  # Keep gripper commands absolute
        offset = self._grasp_site_offset_m
        offset_vec = np.array([0.0, 0.0, offset], dtype=np.float64)  # TCP local +Z

        for side in ("left", "right"):
            pos_key = f"{side}_ee_pos"
            rot6d_key = f"{side}_ee_rot6d"

            if pos_key in action and rot6d_key in action:
                last_commanded_ee_pose = self.last_commanded_ee_pose[side]
                last_commanded_pos = np.asarray(
                    last_commanded_ee_pose["pos"], dtype=np.float64
                ).copy()
                last_commanded_rot6d = np.asarray(last_commanded_ee_pose["rot6d"], dtype=np.float64)
                last_commanded_rot = Rotation.from_matrix(
                    _rot6d_to_rot_matrix(last_commanded_rot6d)
                )

                delta_ee_pos = np.asarray(action[pos_key], dtype=np.float64)
                action_rot6d = np.asarray(action[rot6d_key], dtype=np.float64)
                # The right arm may contact the fixture while descending. On
                # the first upward reversal, re-anchor Z to measured state so
                # an accumulated command error cannot drag through contact.
                last_delta_z = float(
                    np.asarray(self._last_delta_ee_pos[side], dtype=np.float64).reshape(3)[2]
                )
                if side == "right" and last_delta_z < 0.0 and delta_ee_pos[2] > 0.0:
                    right_observed_pos = np.asarray(
                        self._last_observed_ee_pos["right"], dtype=np.float64
                    )
                    last_commanded_pos[2] = right_observed_pos.reshape(3)[2]
                self._last_delta_ee_pos[side] = delta_ee_pos.astype(np.float32).copy()
                delta_ee_rot = Rotation.from_matrix(_rot6d_to_rot_matrix(action_rot6d))
                new_rot = delta_ee_rot * last_commanded_rot
                new_q = new_rot.as_quat()  # xyzw
                new_rot6d = _quat_xyzw_to_rot6d(new_q)

                if (
                    side in getattr(self, "_delta_ee_orientation_filter_sides", set())
                    and getattr(self, "_delta_ee_orientation_filter_quat_xyzw", None) is not None
                ):
                    new_q = getattr(self, "_delta_ee_orientation_filter_quat_xyzw").astype(
                        np.float64
                    )
                    new_rot = Rotation.from_quat(new_q)
                    new_rot6d = _quat_xyzw_to_rot6d(new_q)

                # Get newly commanded ee_pose position
                if offset > 0.0:
                    # Apply delta at the grasp site, then project back to TCP.
                    grasp_pos = last_commanded_pos + last_commanded_rot.apply(offset_vec)
                    new_grasp = grasp_pos + delta_ee_pos
                    new_pos = new_grasp - new_rot.apply(offset_vec)
                else:
                    # No offset — apply delta directly at TCP (original path).
                    new_pos = last_commanded_pos + delta_ee_pos

                new_commanded_ee_pose[pos_key] = new_pos.astype(np.float32)
                new_commanded_ee_pose[rot6d_key] = new_rot6d.astype(np.float32)
                last_commanded_ee_pose["pos"] = new_commanded_ee_pose[pos_key].copy()
                last_commanded_ee_pose["rot6d"] = new_commanded_ee_pose[rot6d_key].copy()

        return new_commanded_ee_pose

    def _translation_mode_mask(
        self,
        action: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Mask a delta-EE action for translation-limited control.

        ``delta_ee_pose_translation`` still consumes the same action schema as
        ``delta_ee_pose``. XYZ deltas are clipped, rotation deltas are forced
        to identity, and gripper commands are locked to the last commanded
        gripper width.

        When the env is configured with translation active sides, only those
        side(s) receive clipped translation deltas. Inactive sides are forced
        to zero translation and identity rotation deltas so they hold the last
        commanded EE pose.
        """
        masked = action.copy()
        enabled_sides = getattr(self, "_delta_ee_translation_active_sides", None)
        active_set: set[str] | None = set(enabled_sides) if enabled_sides is not None else None
        xyz_max_delta = self._delta_ee_translation_xyz_max
        last_commanded_gripper = getattr(self, "last_commanded_gripper", None) or {}
        identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        for side in ("left", "right"):
            pos_key = f"{side}_ee_pos"
            rot6d_key = f"{side}_ee_rot6d"
            grip_key = f"{side}_gripper_pos"
            if active_set is not None and side not in active_set:
                # Inactive arms hold the last commanded EE pose.
                masked[pos_key] = np.zeros(3, dtype=np.float32)
                masked[rot6d_key] = identity_rot6d.copy()
                if grip_key in action:
                    gripper = last_commanded_gripper.get(side, action[grip_key])
                    masked[grip_key] = np.asarray(gripper, dtype=np.float32).reshape(1).copy()
                continue
            if pos_key in action and rot6d_key in action:
                delta_pos = np.asarray(action[pos_key], dtype=np.float32).reshape(3)
                clipped_delta_pos = np.clip(delta_pos, -xyz_max_delta, xyz_max_delta).astype(
                    np.float32
                )
                if not np.allclose(clipped_delta_pos, delta_pos):
                    print(
                        f"\033[31m[YamEnv] clipped translation delta for {side}: "
                        f"requested={delta_pos} clipped={clipped_delta_pos}\033[0m",
                        flush=True,
                    )
                masked[pos_key] = clipped_delta_pos
                masked[rot6d_key] = identity_rot6d.copy()
            if grip_key in action:
                gripper = last_commanded_gripper.get(side, action[grip_key])
                masked[grip_key] = np.asarray(gripper, dtype=np.float32).reshape(1).copy()
        return masked
