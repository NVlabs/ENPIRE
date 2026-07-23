# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa365 environment for CAP.

Wraps a RoboCasa/robosuite environment and implements EnvProtocol,
EefControlProtocol, and TaskProtocol.

Supports two controller modes:
- ``"osc_pose"`` (default): Uses robosuite's OSC_POSE for EE delta control.
- ``"joint_position"``: Uses robosuite's JOINT_POSITION with absolute input
  for joint-level trajectory playback (e.g. from cuRobo motion planning).

Usage::

    uv run cap/server/cap_server.py --env robocasa:PickPlaceCounterToCabinet
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import threading
from typing import TYPE_CHECKING

# Must be set before `import mujoco` — gl_context.py reads MUJOCO_GL at import
# time to select the GL backend. On headless NVIDIA machines, EGL is required.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

if TYPE_CHECKING:
    from enpire.env.forge.cap.env.profile import RobotProfile


class RoboCasaEnv:
    """RoboCasa environment — EnvProtocol + EefControlProtocol + TaskProtocol.

    Pinned to robosuite's ``OSC_POSE`` controller — actions are EE deltas in
    the robot's base frame. Policy rollouts (GR00T N1.5, etc.) that emit OSC
    actions run natively. Skills that plan joint trajectories (cuRobo) are
    executed by FK'ing each waypoint to an EE target and driving OSC to it
    tick-by-tick — see ``_execute_osc_trajectory`` in ``skills.py``.
    """

    CAMERA_HEIGHT = 480
    CAMERA_WIDTH = 640

    def __init__(
        self,
        env_name: str,
        robot: str = "PandaOmron",
        profile: RobotProfile | None = None,
        camera_height: int = 480,
        camera_width: int = 640,
        has_renderer: bool = False,
        control_freq: int = 20,
        layout_ids: int | list[int] = -3,
        style_ids: int | list[int] = -3,
        **robocasa_kwargs,
    ):
        import robocasa  # noqa: F401 — registers envs
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        self.CAMERA_HEIGHT = camera_height
        self.CAMERA_WIDTH = camera_width
        self._env_name = env_name
        self._robot_name = robot
        self._profile = profile
        self._has_renderer = has_renderer
        self._layout_ids = layout_ids
        self._style_ids = style_ids
        self._seed = robocasa_kwargs.get("seed", None)

        # Camera name mapping
        if profile and profile.camera_obs_key_map:
            self._camera_map = dict(profile.camera_obs_key_map)
        else:
            self._camera_map = {
                "top": "robot0_agentview_left_image",
                "right": "robot0_agentview_right_image",
                "wrist": "robot0_eye_in_hand_image",
            }

        # Always OSC_POSE. No controller switching ever.
        controller_configs = load_composite_controller_config(robot=robot)

        # Disable robosuite's built-in camera rendering — we use a separate
        # mujoco.Renderer on a dedicated thread (same pattern as SimBackend)
        # to avoid OpenGL context thread-affinity issues.
        self._env = robosuite.make(
            env_name,
            robots=robot,
            controller_configs=controller_configs,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_heights=camera_height,
            camera_widths=camera_width,
            control_freq=control_freq,
            ignore_done=True,
            layout_ids=layout_ids,
            style_ids=style_ids,
            **robocasa_kwargs,
        )

        self._obs = self._env.reset()
        self._reward: float = 0.0
        self._done: bool = False
        self._info: dict = {}

        # Capture sim state AFTER reset (objects placed, settled) for
        # deterministic restore. robosuite's sim_state_initial is captured
        # too early (before _reset_internal places objects).
        self._post_reset_state = self._env.sim.get_state()

        # Cache the task description once per real reset. robocasa assembles
        # the string lazily in ``get_ep_meta()``; with use_novel_instructions
        # it re-samples a paraphrase on every call. Caching pins the phrasing
        # for the episode so reset_to_initial() keeps the original wording.
        self._task_description: str = self._env.get_ep_meta().get("lang", "") or ""
        self._obs_generation: int = 0
        self._frames_generation: int = -1
        self._frames_cache: dict[str, np.ndarray] = {}

        # Action space info
        self._robot = self._env.robots[0]
        self._action_dim: int = self._robot.action_dim
        cc = self._robot.composite_controller
        self._action_splits: dict[str, tuple[int, int]] = dict(cc._action_split_indexes)

        # OSC output scaling (robosuite OSC_POSE default).
        self._osc_pos_max = 0.05  # m per action unit
        self._osc_ori_max = 0.5  # rad per action unit

        # Pending OSC action from compute_eef_action. Consumed by step().
        self._pending_eef_action: np.ndarray | None = None

        # Optional callback invoked after every env.step() — used by Viser
        # for per-step visualization updates without RPC overhead.
        self._step_callback: callable | None = None

        self._lock = threading.Lock()

        # Capture initial EE pose + joint positions per arm (for go_home)
        self._initial_ee: dict[str, dict[str, np.ndarray]] = {}
        for side in profile.arm_names if profile else ("right",):
            obs = self.get_arm_observation(side)
            if "ee_pos" in obs and "ee_quat" in obs:
                self._initial_ee[side] = {
                    "pos": obs["ee_pos"].copy(),
                    "quat": obs["ee_quat"].copy(),
                    "joint_pos": obs["joint_pos"].copy(),
                }

        # Recorder attached via set_recorder() — frames pushed after every step.
        self._recorder = None
        self._debug_markers: list[dict] = []

        # Separate renderers on a dedicated thread (EGL context affinity)
        self._rgb_renderer: mujoco.Renderer | None = None
        self._depth_renderer: mujoco.Renderer | None = None
        self._render_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="robocasa-render"
        )
        # Robosuite XML: visual geoms are group 1, collision geoms are group 0.
        # Hide group 0 so only visual meshes appear in rendered images.
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[0] = 0

    # ------------------------------------------------------------------
    # EnvProtocol: Physics
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Step physics. Uses pending EEF action if set, else zero OSC delta (hold)."""
        if self._pending_eef_action is not None:
            action = self._pending_eef_action
            self._pending_eef_action = None
        else:
            action = np.zeros(self._action_dim)
        self._step_env_action(action)

    def _step_env_action(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, dict]:
        with self._lock:
            self._obs, self._reward, self._done, self._info = self._env.step(action)
            raw_obs = self._obs
            reward = self._reward
            done = self._done
            info = self._info
        self._obs_generation += 1
        if self._step_callback is not None:
            self._step_callback()
        if self._has_renderer:
            self._env.render()
        if self._recorder is not None:
            for alias, frame in self.last_frames.items():
                self._recorder.push_frame(alias, frame)
        return raw_obs, reward, done, info

    # ------------------------------------------------------------------
    # EnvProtocol: State
    # ------------------------------------------------------------------

    def get_arm_observation(self, side: str) -> dict[str, np.ndarray]:
        prefix = self._obs_prefix(side)
        with self._lock:
            obs = self._obs
        result: dict[str, np.ndarray] = {
            "joint_pos": obs[f"{prefix}joint_pos"].copy(),
            "gripper_pos": self._normalize_gripper(obs[f"{prefix}gripper_qpos"]),
        }
        if f"{prefix}eef_pos" in obs:
            result["ee_pos"] = obs[f"{prefix}eef_pos"].copy()
        if f"{prefix}eef_quat" in obs:
            result["ee_quat"] = obs[f"{prefix}eef_quat"].copy()
        return result

    def command_arm(self, side: str, cmd: dict) -> None:
        """Accept a command dict; in OSC-only mode only the gripper is used.

        Kept for protocol compatibility (set_gripper / open_gripper /
        close_gripper route through here). The ``pos`` array is interpreted
        as ``[... , gripper_value]`` and only the final element reaches the
        sim as a zero-arm gripper-only OSC action. Arm motion must go through
        ``compute_eef_action`` (or ``_execute_osc_trajectory`` in skills.py).
        """
        pos = np.asarray(cmd["pos"], dtype=np.float64)
        if self._pending_eef_action is None and len(pos) > 0:
            gripper_val = float(np.clip(pos[-1], 0.0, 1.0))
            self._apply_gripper_action(side, gripper_val)

    def _apply_gripper_action(self, side: str, gripper_val: float) -> None:
        """Issue a zero-arm OSC action with the given gripper value.

        gripper_val: 0.0 = fully closed, 1.0 = fully open (CAP convention).
        robosuite convention: +1 = close, -1 = open, so the mapping is
        action = 1.0 - 2.0 * gripper_val.
        """
        action = np.zeros(self._action_dim)
        # Find the gripper key in action_splits
        gripper_key = None
        for k in self._action_splits:
            if "gripper" in k.lower():
                gripper_key = k
                break
        if gripper_key is None:
            print(
                f"[RoboCasaEnv] WARNING: no gripper key in action_splits={list(self._action_splits.keys())}"
            )
            return
        grip_start, grip_end = self._action_splits[gripper_key]
        robosuite_val = 1.0 - 2.0 * gripper_val  # 0->+1(close), 1->-1(open)
        action[grip_start] = robosuite_val
        with self._lock:
            self._pending_eef_action = action

    # ------------------------------------------------------------------
    # EnvProtocol: Rendering
    # ------------------------------------------------------------------

    def set_recorder(self, recorder) -> None:
        """Attach a ScriptRecorder (push mode). Pass None to detach."""
        self._recorder = recorder

    def render_rgb(self, camera_name: str) -> np.ndarray:
        if camera_name not in self._camera_map:
            raise KeyError(f"Unknown camera: {camera_name}")
        return self._render_executor.submit(self._render_rgb_impl, camera_name).result()

    def set_debug_markers(self, markers: list[dict]) -> None:
        normalized = []
        for marker in markers:
            if "position" not in marker:
                continue
            normalized.append(
                {
                    "name": str(marker.get("name", "debug_marker")),
                    "label": str(marker.get("label", marker.get("name", "debug_marker"))),
                    "position": np.asarray(marker["position"], dtype=np.float64).copy(),
                    "color": tuple(int(x) for x in marker.get("color", [64, 255, 255])),
                    "radius_m": float(marker.get("radius_m", 0.03)),
                    "alpha": float(marker.get("alpha", 0.75)),
                }
            )
        with self._lock:
            self._debug_markers = normalized

    def clear_debug_markers(self) -> None:
        with self._lock:
            self._debug_markers = []

    def _project_world_to_pixel(
        self,
        pos_3d: np.ndarray,
        cam_pos: np.ndarray,
        cam_rot: np.ndarray,
        intrinsics: list[float],
    ) -> tuple[int, int, float] | None:
        fx, fy, cx, cy = intrinsics
        p_world = np.asarray(pos_3d, dtype=np.float64)
        p_cam = np.linalg.inv(cam_rot) @ (p_world - cam_pos)
        p_cam[0] = -p_cam[0]
        p_cam[1] = -p_cam[1]
        if p_cam[2] <= 0:
            return None
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        return int(round(float(u))), int(round(float(v))), float(p_cam[2])

    def _draw_debug_markers(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        with self._lock:
            markers = list(self._debug_markers)
        if not markers:
            return image

        extr = self.get_camera_extrinsics(camera_name)
        cam_pos = np.asarray(extr["position"], dtype=np.float64)
        cam_rot = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        intrinsics = self.get_camera_intrinsics(camera_name)

        overlay = image.copy()
        h, w = image.shape[:2]
        fx = float(intrinsics[0])

        for marker in markers:
            projected = self._project_world_to_pixel(
                marker["position"], cam_pos, cam_rot, intrinsics
            )
            if projected is None:
                continue
            px, py, depth = projected
            if px < 0 or px >= w or py < 0 or py >= h:
                continue
            radius_px = max(4, int(round(fx * marker["radius_m"] / max(depth, 1e-6))))
            color_bgr = tuple(int(c) for c in reversed(marker["color"]))
            cv2.circle(overlay, (px, py), radius_px, color_bgr, thickness=-1)
            cv2.circle(overlay, (px, py), radius_px + 1, (255, 255, 255), thickness=1)
            label = marker.get("label", "")
            if label:
                cv2.putText(
                    overlay,
                    str(label),
                    (px + radius_px + 4, max(14, py - radius_px - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color_bgr,
                    1,
                    cv2.LINE_AA,
                )

        alpha = max(0.0, min(1.0, float(markers[0].get("alpha", 0.75))))
        return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)

    def _render_rgb_impl(self, camera_name: str) -> np.ndarray:
        mj_cam_name = self._resolve_mujoco_camera_name(camera_name)
        with self._lock:
            if self._rgb_renderer is None:
                self._rgb_renderer = mujoco.Renderer(
                    self._env.sim.model._model,
                    self.CAMERA_HEIGHT,
                    self.CAMERA_WIDTH,
                )
                self._rgb_renderer.disable_depth_rendering()
            self._rgb_renderer.update_scene(
                self._env.sim.data._data,
                camera=mj_cam_name,
                scene_option=self._scene_option,
            )
        # render() only reads the internal scene graph — no physics lock needed
        image = self._rgb_renderer.render().copy()
        return self._draw_debug_markers(image, camera_name)

    def render_depth(self, camera_name: str) -> np.ndarray:
        if camera_name not in self._camera_map:
            return np.zeros((self.CAMERA_HEIGHT, self.CAMERA_WIDTH), dtype=np.float32)
        return self._render_executor.submit(
            self._render_depth_impl, camera_name
        ).result()

    def _render_depth_impl(self, camera_name: str) -> np.ndarray:
        mj_cam_name = self._resolve_mujoco_camera_name(camera_name)
        with self._lock:
            if self._depth_renderer is None:
                self._depth_renderer = mujoco.Renderer(
                    self._env.sim.model._model,
                    self.CAMERA_HEIGHT,
                    self.CAMERA_WIDTH,
                )
                self._depth_renderer.enable_depth_rendering()
            self._depth_renderer.update_scene(
                self._env.sim.data._data,
                camera=mj_cam_name,
                scene_option=self._scene_option,
            )
        return self._depth_renderer.render().copy().astype(np.float32)

    def get_camera_intrinsics(self, camera_name: str) -> list[float]:
        mj_cam_name = self._resolve_mujoco_camera_name(camera_name)
        model = self._env.sim.model
        cam_id = model.camera_name2id(mj_cam_name)
        if hasattr(model, "cam_focal"):
            focal = model.cam_focal[cam_id]
            sensor = model.cam_sensorsize[cam_id]
            fx = focal[0] / sensor[0] * self.CAMERA_WIDTH
            fy = focal[1] / sensor[1] * self.CAMERA_HEIGHT
        else:
            fovy = model.cam_fovy[cam_id]
            fy = self.CAMERA_HEIGHT / (2.0 * math.tan(math.radians(fovy) / 2.0))
            fx = fy
        return [
            float(fx),
            float(fy),
            float(self.CAMERA_WIDTH / 2.0),
            float(self.CAMERA_HEIGHT / 2.0),
        ]

    def get_camera_extrinsics(self, camera_name: str) -> dict:
        mj_cam_name = self._resolve_mujoco_camera_name(camera_name)
        model = self._env.sim.model
        data = self._env.sim.data
        cam_id = model.camera_name2id(mj_cam_name)
        pos = data.cam_xpos[cam_id].copy()
        rot = data.cam_xmat[cam_id].copy().reshape(3, 3)
        rot = rot @ np.diag([-1.0, 1.0, -1.0])
        return {"position": pos.tolist(), "rotation": rot.tolist()}

    def _dict_action_to_env_action(self, action_dict: dict) -> np.ndarray:
        env_action = []
        for robot in self._env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act
            if cc.__class__.__name__ == "HybridMobileBase":
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        assert len(action_dict) == 0, f"Unprocessed actions: {action_dict}"
        return np.concatenate(env_action)

    def step_dict(self, robosuite_dict: dict) -> tuple[dict, float, bool, dict]:
        """Step physics with a robosuite-format action dict.

        Converts the dict to a numpy action vector, steps physics, checks task
        success, and returns (raw_obs, reward, done, info). Obs is in raw
        robosuite format — callers (e.g. robocasa_runner) build policy obs.
        """
        env_action = self._dict_action_to_env_action(robosuite_dict)
        raw_obs, _, done, info = self._step_env_action(env_action)
        success = bool(self._env._check_success())
        reward = 1.0 if success else 0.0
        step_info = dict(info)
        step_info["success"] = success
        with self._lock:
            self._reward = reward
            self._info = step_info
        return raw_obs, reward, done, step_info

    @property
    def last_frames(self) -> dict[str, np.ndarray]:
        """Rendered camera frames for the current sim state — lazy, cached by generation."""
        if self._frames_generation != self._obs_generation:
            self._frames_cache = {alias: self.render_rgb(alias) for alias in self._camera_map}
            self._frames_generation = self._obs_generation
        return self._frames_cache

    # ------------------------------------------------------------------
    # EefControlProtocol: per-tick EE control (does NOT step physics)
    # ------------------------------------------------------------------

    def compute_eef_action(
        self,
        side: str,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        gripper: float | None = None,
    ) -> dict:
        """Compute ONE tick's OSC_POSE action to move toward an EE target.

        Called by cap_server's control loop every tick. Does NOT step physics.
        Returns a command dict for ``command_arm``.
        """
        from scipy.spatial.transform import Rotation as R

        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = np.asarray(target_quat_xyzw, dtype=np.float64)
        prefix = self._obs_prefix(side)
        arm_key = side
        gripper_key = f"{side}_gripper"

        with self._lock:
            cur_pos = self._obs[f"{prefix}eef_pos"].copy()
            cur_quat = self._obs[f"{prefix}eef_quat"].copy()
            base_quat = self._obs.get("robot0_base_quat")

        # Position delta in world frame
        delta_pos_world = target_pos - cur_pos

        # Orientation delta (axis-angle) in world frame
        R_cur = R.from_quat(cur_quat)
        R_target = R.from_quat(target_quat)
        delta_ori_world = (R_target * R_cur.inv()).as_rotvec()

        # OSC input_ref_frame="base" -> transform to base frame
        if base_quat is not None:
            R_base = R.from_quat(base_quat)
            delta_pos = R_base.inv().apply(delta_pos_world)
            delta_ori = R_base.inv().apply(delta_ori_world)
        else:
            delta_pos = delta_pos_world
            delta_ori = delta_ori_world

        # Build OSC_POSE action (full env action array)
        action = np.zeros(self._action_dim)

        if arm_key in self._action_splits:
            arm_start, arm_end = self._action_splits[arm_key]
            arm_action = np.zeros(arm_end - arm_start)
            arm_action[:3] = np.clip(delta_pos / self._osc_pos_max, -1.0, 1.0)
            if len(arm_action) >= 6:
                arm_action[3:6] = np.clip(delta_ori / self._osc_ori_max, -1.0, 1.0)
            action[arm_start:arm_end] = arm_action

        if gripper is not None and gripper_key in self._action_splits:
            grip_start, _ = self._action_splits[gripper_key]
            action[grip_start] = 1.0 - 2.0 * gripper

        # Store as pending action so step() uses it
        self._pending_eef_action = action
        return {"pos": action}  # command_arm format

    # ------------------------------------------------------------------
    # TaskProtocol
    # ------------------------------------------------------------------

    def reset_env(self) -> dict:
        with self._lock:
            self._obs = self._env.reset()
            self._reward = 0.0
            self._done = False
            self._info = {}
            # New episode → new task description allowed. Refresh the cache.
            self._task_description = self._env.get_ep_meta().get("lang", "") or ""
            self._pending_eef_action = None
        self._obs_generation += 1
        return {"ok": True, "task": self._env_name}

    def reset_to_initial(self) -> dict:
        """Restore the MuJoCo state captured after the first reset.

        Same layout, same objects, same positions — no re-randomization.
        """
        sim = self._env.sim
        sim.set_state(self._post_reset_state)
        sim.forward()
        # Must mirror robosuite's reset path: clear obs cache, reset
        # observables, then force-update observations so object positions
        # are recomputed from the restored sim state.
        self._env._obs_cache = {}
        if hasattr(self._env, "_reset_observables"):
            self._env._reset_observables()
        self._obs = self._env._get_observations(force_update=True)
        self._reward = 0.0
        self._done = False
        self._info = {}
        self._env.timestep = 0
        self._env.done = False
        self._pending_eef_action = None
        self._obs_generation += 1
        return {"ok": True, "task": self._env_name}

    def load_task(self, task_name: str) -> dict:
        """Load a new RoboCasa task (tears down and recreates the env)."""
        import robocasa  # noqa: F401
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        # Close renderers on the render thread (EGL affinity)
        self._close_renderers()

        self._env.close()
        self._env_name = task_name

        controller_configs = load_composite_controller_config(robot=self._robot_name)

        self._env = robosuite.make(
            task_name,
            robots=self._robot_name,
            controller_configs=controller_configs,
            has_renderer=self._has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_heights=self.CAMERA_HEIGHT,
            camera_widths=self.CAMERA_WIDTH,
            control_freq=int(self._profile.control_freq_hz) if self._profile else 20,
            ignore_done=True,
            layout_ids=self._layout_ids,
            style_ids=self._style_ids,
        )

        # Re-read action space info
        self._robot = self._env.robots[0]
        self._action_dim = self._robot.action_dim
        cc = self._robot.composite_controller
        self._action_splits = dict(cc._action_split_indexes)

        with self._lock:
            self._obs = self._env.reset()
            self._reward = 0.0
            self._done = False
            self._info = {}
            self._task_description = self._env.get_ep_meta().get("lang", "") or ""
            self._pending_eef_action = None

        return {"ok": True, "task": task_name}

    def get_last_reward(self) -> float:
        return self._reward

    def get_task_description(self) -> str:
        """Return the canonical natural-language task description.

        Cached at reset time, not queried per call — with
        ``use_novel_instructions=True`` robocasa re-samples a paraphrase on
        every ``get_ep_meta()`` call, which would let two successive readers
        see different strings. The cache pins the phrasing for the episode.
        """
        return self._task_description

    def get_task_info(self) -> dict:
        # Call _check_success() directly on the robosuite env — the canonical
        # way RoboCasa evaluates task completion (see robocasa/utils/eval_utils.py).
        # The info dict from env.step() is always empty, and self._reward is
        # the last-step cache which can be stale by the time get_task_info() is
        # called from another thread. In RoboCasa, reward() is just
        # float(_check_success()), so derive the reward from the fresh check
        # to keep `success` and `reward` mutually consistent — otherwise
        # downstream consumers see success=True with score=0.0.
        with self._lock:
            try:
                success = bool(self._env._check_success())
            except Exception:
                success = self._reward > 0

        info = {
            "done": self._done,
            "reward": float(success),
            "success": success,
            "env_name": self._env_name,
        }
        # Include object positions from obs (for scripting without vision)
        with self._lock:
            for k, v in self._obs.items():
                if (
                    k.endswith("_pos")
                    and not k.startswith("robot0")
                    and hasattr(v, "tolist")
                ):
                    info[k] = v.tolist()
        # Extract object names from MJCFObject model paths (e.g. .../tomato/tomato_0/model.xml -> "tomato")
        objects = getattr(self._env, "objects", {})
        if isinstance(objects, dict):
            for key, obj in objects.items():
                path = getattr(obj, "mjcf_path", "") or ""
                parts = path.replace("\\", "/").split("/")
                # Pattern: .../category/instance/model.xml
                if len(parts) >= 3 and parts[-1].endswith(".xml"):
                    info[f"{key}_name"] = parts[-3]
        return info

    def get_close_blender_lid_debug_info(self) -> dict:
        from robocasa.utils import object_utils as OU

        with self._lock:
            env = self._env
            blender = getattr(env, "blender", None)
            if blender is None:
                raise RuntimeError("CloseBlenderLid debug info unavailable: blender fixture missing")

            lid_fixture = getattr(blender, "blender_lid", None)
            if lid_fixture is None:
                raise RuntimeError("CloseBlenderLid debug info unavailable: blender lid fixture missing")

            lid_body = f"{lid_fixture.name}_main"
            curr_lid_pos = blender.get_curr_lid_pos(env)
            closed_lid_pos = blender.get_lid_closed_pos(env)
            fixture_state = blender.get_state()
            lid_on_blender = bool(fixture_state.get("lid_on_blender", False))
            pos_err = (
                None
                if curr_lid_pos is None or closed_lid_pos is None
                else float(np.linalg.norm(np.asarray(curr_lid_pos) - np.asarray(closed_lid_pos)))
            )
            upright_ok = bool(OU.check_fxtr_upright(env, lid_body, th=7)) if curr_lid_pos is not None else False
            gripper_far = bool(OU.gripper_fxtr_far(env, lid_body, th=0.15))

            ee_pos = None
            gripper_pos = None
            if "robot0_eef_pos" in self._obs:
                ee_pos = np.asarray(self._obs["robot0_eef_pos"], dtype=np.float64).tolist()
            if "robot0_gripper_qpos" in self._obs:
                grip = np.asarray(self._obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
                if grip.size:
                    gripper_pos = float(grip[0])

            return {
                "success": bool(lid_on_blender and gripper_far),
                "lid_on_blender": lid_on_blender,
                "gripper_far": gripper_far,
                "lid_pos": None if curr_lid_pos is None else np.asarray(curr_lid_pos, dtype=np.float64).tolist(),
                "closed_lid_pos": None if closed_lid_pos is None else np.asarray(closed_lid_pos, dtype=np.float64).tolist(),
                "lid_pos_err_m": pos_err,
                "lid_pos_thresh_m": 0.04,
                "upright_ok": upright_ok,
                "upright_thresh_deg": 7.0,
                "gripper_far_thresh_m": 0.15,
                "ee_pos": ee_pos,
                "gripper_pos": gripper_pos,
            }

    def get_close_fridge_debug_info(self) -> dict:
        with self._lock:
            env = self._env
            fridge = getattr(env, "fxtr", None)
            if fridge is None:
                fridge = getattr(env, "fridge", None)
            if fridge is None:
                raise RuntimeError("CloseFridge debug info unavailable: fridge fixture missing")

            joint_names = list(getattr(fridge, "_fridge_door_joint_names", []) or [])
            if not joint_names:
                joint_names = list(getattr(fridge, "door_joint_names", []) or [])
            if not joint_names:
                raise RuntimeError("CloseFridge debug info unavailable: fridge door joints missing")

            joint_state = fridge.get_joint_state(env, joint_names)
            closed_thresh = 0.005
            joint_qpos = {name: float(joint_state.get(name, 1.0)) for name in joint_names}
            max_joint_qpos = max(joint_qpos.values()) if joint_qpos else None
            most_open_joint = None
            if joint_qpos:
                most_open_joint = max(joint_qpos, key=joint_qpos.get)

            ee_pos = None
            if "robot0_eef_pos" in self._obs:
                ee_pos = np.asarray(self._obs["robot0_eef_pos"], dtype=np.float64).tolist()

            return {
                "success": bool(fridge.is_closed(env)),
                "is_closed": bool(fridge.is_closed(env)),
                "closed_thresh": closed_thresh,
                "joint_qpos": joint_qpos,
                "max_joint_qpos": None if max_joint_qpos is None else float(max_joint_qpos),
                "remaining_to_close": (
                    None
                    if max_joint_qpos is None
                    else float(max(max_joint_qpos - closed_thresh, 0.0))
                ),
                "most_open_joint": most_open_joint,
                "ee_pos": ee_pos,
            }

    # ------------------------------------------------------------------
    # Base pose (for get_state)
    # ------------------------------------------------------------------

    def get_base_pose(self) -> dict[str, np.ndarray]:
        with self._lock:
            result = {}
            if "robot0_base_pos" in self._obs:
                result["base_pos"] = self._obs["robot0_base_pos"].copy()
            if "robot0_base_quat" in self._obs:
                result["base_quat"] = self._obs["robot0_base_quat"].copy()
            return result

    # ------------------------------------------------------------------
    # Collision geometry (for cuRobo world updates)
    # ------------------------------------------------------------------

    def get_collision_geoms(
        self,
        exclude_body_prefixes: list[str] | None = None,
        max_dist: float = 1.2,
        min_size: float = 0.03,
    ) -> dict:
        """Extract collision geometry from MuJoCo sim as numpy arrays.

        Returns dict with base_pos, base_quat_xyzw, names, positions,
        rot_mats, dims_array, n_geoms — same format as CapServer.
        """
        import mujoco as mj

        if exclude_body_prefixes is None:
            exclude_body_prefixes = ["robot0", "gripper", "mobilebase"]

        bp = self.get_base_pose()
        base_pos = np.asarray(bp.get("base_pos", np.zeros(3)), dtype=np.float64)
        base_quat = np.asarray(bp.get("base_quat", [0, 0, 0, 1]), dtype=np.float64)

        sim = self._env.sim
        model = sim.model._model
        data = sim.data._data

        names: list[str] = []
        positions_list: list[np.ndarray] = []
        rot_mats_list: list[np.ndarray] = []
        dims_list: list[np.ndarray] = []

        for i in range(model.ngeom):
            if model.geom_group[i] != 0:
                continue
            gtype = int(model.geom_type[i])
            if gtype not in (5, 6):  # cylinder, box
                continue
            body_id = model.geom_bodyid[i]
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
            if any(body_name.startswith(p) for p in exclude_body_prefixes):
                continue
            pos_world = data.geom_xpos[i].copy()
            if float(np.linalg.norm(pos_world[:2] - base_pos[:2])) > max_dist:
                continue
            size = model.geom_size[i].copy()
            if gtype == 6:
                dims = size * 2.0
            elif gtype == 5:
                dims = np.array([size[0] * 2, size[0] * 2, size[1] * 2])
            else:
                continue
            if np.max(dims) < min_size:
                continue
            names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}")
            positions_list.append(pos_world)
            rot_mats_list.append(data.geom_xmat[i].reshape(3, 3).copy())
            dims_list.append(dims)

        n = len(names)
        return {
            "base_pos": base_pos,
            "base_quat_xyzw": base_quat,
            "names": names,
            "positions": np.array(positions_list) if n else np.empty((0, 3)),
            "rot_mats": np.array(rot_mats_list) if n else np.empty((0, 3, 3)),
            "dims_array": np.array(dims_list) if n else np.empty((0, 3)),
            "n_geoms": n,
        }

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def get_floor_bounds(self, padding: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        """Estimate navigable floor bounds from collision geom positions.

        Returns (min_xy, max_xy) as (2,) numpy arrays in world frame.
        """
        coll = self.get_collision_geoms(max_dist=10.0)
        positions = coll["positions"]
        dims = coll["dims_array"]
        n = coll["n_geoms"]

        if n == 0:
            bp = self.get_base_pose()
            base_xy = bp.get("base_pos", np.zeros(3))[:2]
            return base_xy - 2.0, base_xy + 2.0

        half_dims = dims[:, :2] / 2.0
        mins = positions[:, :2] - half_dims
        maxs = positions[:, :2] + half_dims

        floor_min = np.min(mins, axis=0) - padding
        floor_max = np.max(maxs, axis=0) + padding
        return floor_min, floor_max

    def command_base_action(
        self,
        v_fwd: float,
        v_side: float,
        omega: float,
        n_steps: int = 1,
    ) -> None:
        """Drive the mobile base using the standard RoboCasa action pipeline.

        Builds a full action vector with base velocity commands at the
        ``base`` action-split indices, sets ``base_mode = +1`` (last element),
        holds arm joints, and calls ``env.step()`` — identical to how a
        learned policy or the RoboCasa eval script drives the base.

        Args:
            v_fwd: Forward velocity command in [-1, 1].
            v_side: Sideways velocity command in [-1, 1].
            omega: Yaw rate command in [-1, 1].
            n_steps: Number of env.step() calls (default 1).
        """
        action = np.zeros(self._action_dim)

        # Base velocity at the "base" action split
        if "base" in self._action_splits:
            base_start, base_end = self._action_splits["base"]
            base_vel = np.array(
                [
                    np.clip(v_fwd, -1.0, 1.0),
                    np.clip(v_side, -1.0, 1.0),
                    np.clip(omega, -1.0, 1.0),
                ]
            )
            action[base_start:base_end] = base_vel[: base_end - base_start]

        # base_mode = +1 (last element) activates HybridMobileBase
        action[-1] = 1.0

        # OSC_POSE: arm slot stays zero → arm holds position while base moves.

        # Step physics through the wrapper so callbacks and recording stay active.
        for _ in range(n_steps):
            self._step_env_action(action)

    # ------------------------------------------------------------------
    # Oracle target API
    # ------------------------------------------------------------------

    # --- Pose / geometry helpers ---

    def _body_pose_dict(self, body_name: str) -> dict:
        from scipy.spatial.transform import Rotation as R

        body_id = self._env.sim.model.body_name2id(body_name)
        pos = self._env.sim.data.body_xpos[body_id].copy()
        rot = self._env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
        quat_xyzw = R.from_matrix(rot).as_quat()
        return {
            "body_name": body_name,
            "pos": pos.tolist(),
            "quat_xyzw": quat_xyzw.tolist(),
            "rot_mat": rot.tolist(),
        }

    def _geom_pose_dict(self, geom_name: str, *, reference_pos: np.ndarray | None = None) -> dict:
        from scipy.spatial.transform import Rotation as R

        geom_id = self._env.sim.model.geom_name2id(geom_name)
        pos = self._env.sim.data.geom_xpos[geom_id].copy()
        rot = self._env.sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
        quat_xyzw = R.from_matrix(rot).as_quat()
        size = self._env.sim.model.geom_size[geom_id].copy()
        geom_type = int(self._env.sim.model.geom_type[geom_id])
        geom_type_name = {
            2: "sphere", 3: "capsule", 4: "ellipsoid",
            5: "cylinder", 6: "box", 7: "mesh",
        }.get(geom_type, f"geom_type_{geom_type}")

        if geom_type == 5:  # cylinder: normal along local Z
            normal = rot[:, 2].copy()
        else:
            axis_idx = int(np.argmin(size))
            normal = rot[:, axis_idx].copy()
        if reference_pos is not None and np.dot(normal, pos - reference_pos) < 0:
            normal *= -1.0

        return {
            "geom_name": geom_name,
            "geom_type_id": geom_type,
            "geom_type_name": geom_type_name,
            "pos": pos.tolist(),
            "quat_xyzw": quat_xyzw.tolist(),
            "rot_mat": rot.tolist(),
            "size": size.tolist(),
            "surface_normal": normal.tolist(),
        }

    def _joint_pose_dict(self, joint_name: str) -> dict:
        joint_id = self._env.sim.model.joint_name2id(joint_name)
        qpos_addr = self._env.sim.model.jnt_qposadr[joint_id]
        qpos = float(self._env.sim.data.qpos[qpos_addr])
        joint_range = self._env.sim.model.jnt_range[joint_id].copy()
        joint_type_id = int(self._env.sim.model.jnt_type[joint_id])
        axis_local = self._env.sim.model.jnt_axis[joint_id].copy()
        body_id = self._env.sim.model.jnt_bodyid[joint_id]
        body_pos = self._env.sim.data.body_xpos[body_id].copy()
        body_rot = self._env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
        anchor_local = self._env.sim.model.jnt_pos[joint_id].copy()
        anchor_world = body_pos + body_rot @ anchor_local
        axis_world = body_rot @ axis_local
        joint_type_name = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(
            joint_type_id, f"unknown:{joint_type_id}"
        )
        lo, hi = float(joint_range[0]), float(joint_range[1])
        normalized_qpos = float(np.clip((qpos - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.0
        return {
            "joint_name": joint_name,
            "joint_type": joint_type_name,
            "qpos": qpos,
            "range": joint_range.tolist(),
            "normalized_qpos": normalized_qpos,
            "anchor_world": anchor_world.tolist(),
            "axis_world": axis_world.tolist(),
        }

    def _site_pose_dict(self, site_name: str) -> dict:
        from scipy.spatial.transform import Rotation as R

        site_id = self._env.sim.model.site_name2id(site_name)
        pos = self._env.sim.data.site_xpos[site_id].copy()
        rot = self._env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
        return {
            "site_name": site_name,
            "pos": pos.tolist(),
            "quat_xyzw": R.from_matrix(rot).as_quat().tolist(),
            "rot_mat": rot.tolist(),
        }

    @staticmethod
    def _object_category(obj) -> str:
        path = getattr(obj, "mjcf_path", "") or ""
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[-1].endswith(".xml"):
            return parts[-3]
        return getattr(obj, "name", "object")

    def _object_payload(self, obj_name: str) -> dict:
        obj = self._env.objects[obj_name]
        payload = self._body_pose_dict(obj.root_body)
        payload.update(
            name=obj_name,
            category=self._object_category(obj),
            root_body=obj.root_body,
            horizontal_radius=float(getattr(obj, "horizontal_radius", 0.0)),
            mjcf_path=getattr(obj, "mjcf_path", ""),
        )
        return payload

    def _estimate_object_handle_pose(self, obj_name: str) -> dict:
        obj = self._env.objects[obj_name]
        body_id = self._env.sim.model.body_name2id(obj.root_body)
        root_pos = self._env.sim.data.body_xpos[body_id].copy()
        root_rot = self._env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
        prefix = getattr(obj, "naming_prefix", "")
        reference_geom = None
        reference_pos = None
        method = "fallback_local_neg_y"
        if prefix:
            reference_geom = self._find_prefixed_geom_name(
                prefix, ("reg", "int"), reference_pos=root_pos
            )
            if reference_geom is None:
                reference_geom = self._find_prefixed_geom_name(
                    prefix, ("liquid",), reference_pos=root_pos
                )
            if reference_geom is not None:
                geom_id = self._env.sim.model.geom_name2id(reference_geom)
                reference_pos = self._env.sim.data.geom_xpos[geom_id].copy()
        if reference_pos is not None:
            handle_dir = reference_pos - root_pos
            handle_dir[2] = 0.0
            handle_norm = float(np.linalg.norm(handle_dir))
            if handle_norm > 1e-5:
                handle_dir = -handle_dir / handle_norm
                method = f"opposite_{reference_geom}"
            else:
                handle_dir = -root_rot[:, 1].copy(); handle_dir[2] = 0.0
        else:
            handle_dir = -root_rot[:, 1].copy(); handle_dir[2] = 0.0
        handle_norm = float(np.linalg.norm(handle_dir))
        handle_dir = handle_dir / handle_norm if handle_norm > 1e-5 else np.array([0., -1., 0.])
        handle_offset = max(float(getattr(obj, "horizontal_radius", 0.0)) * 0.92, 0.04)
        handle_pos = root_pos + handle_dir * handle_offset
        if reference_pos is not None:
            handle_pos[2] = reference_pos[2]
        return {
            "kind": "object_handle",
            "object_name": obj_name,
            "pos": handle_pos.tolist(),
            "dir_world": handle_dir.tolist(),
            "reference_geom": reference_geom,
            "reference_pos": reference_pos.tolist() if reference_pos is not None else None,
            "method": method,
        }

    def _find_prefixed_geom_names(
        self,
        prefix: str,
        include_tokens: tuple,
        *,
        exclude_tokens: tuple = (),
        reference_pos: np.ndarray | None = None,
        prefer: str = "nearest",
    ) -> list:
        candidates: list = []
        for geom_id in range(self._env.sim.model.ngeom):
            geom_name = self._env.sim.model.geom_id2name(geom_id)
            if not geom_name or not geom_name.startswith(prefix):
                continue
            suffix = geom_name[len(prefix):].lower()
            if not all(tok.lower() in suffix for tok in include_tokens):
                continue
            if any(tok.lower() in suffix for tok in exclude_tokens):
                continue
            pos = self._env.sim.data.geom_xpos[geom_id].copy()
            dist = float(np.linalg.norm(pos - reference_pos)) if reference_pos is not None else 0.0
            score = dist if prefer == "nearest" else -dist
            candidates.append((score, len(suffix), geom_name))
        candidates.sort()
        return [name for _, _, name in candidates]

    def _find_prefixed_geom_name(
        self,
        prefix: str,
        include_tokens: tuple,
        *,
        exclude_tokens: tuple = (),
        reference_pos: np.ndarray | None = None,
        prefer: str = "nearest",
    ) -> str | None:
        names = self._find_prefixed_geom_names(
            prefix, include_tokens,
            exclude_tokens=exclude_tokens,
            reference_pos=reference_pos,
            prefer=prefer,
        )
        return names[0] if names else None

    def _find_prefixed_body_name(
        self,
        prefix: str,
        include_tokens: tuple,
        *,
        exclude_tokens: tuple = (),
        reference_pos: np.ndarray | None = None,
        prefer: str = "nearest",
    ) -> str | None:
        candidates: list = []
        for body_id in range(self._env.sim.model.nbody):
            body_name = self._env.sim.model.body_id2name(body_id)
            if not body_name or not body_name.startswith(prefix):
                continue
            suffix = body_name[len(prefix):].lower()
            if not all(tok.lower() in suffix for tok in include_tokens):
                continue
            if any(tok.lower() in suffix for tok in exclude_tokens):
                continue
            pos = self._env.sim.data.body_xpos[body_id].copy()
            dist = float(np.linalg.norm(pos - reference_pos)) if reference_pos is not None else 0.0
            score = dist if prefer == "nearest" else -dist
            candidates.append((score, len(suffix), body_name))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    @staticmethod
    def _match_joint_name(joint_names: list, include_tokens: tuple) -> str | None:
        lowered = [tok.lower() for tok in include_tokens]
        for joint_name in joint_names:
            if all(tok in joint_name.lower() for tok in lowered):
                return joint_name
        return None

    def _aggregate_handle_pose_dict(
        self,
        *,
        explicit_name: str | None,
        prefix: str,
        include_tokens: tuple,
        reference_pos: np.ndarray | None = None,
    ) -> dict | None:

        round_types = {"cylinder", "capsule", "sphere", "ellipsoid"}
        geom_ids: list = []
        source_body_name: str | None = None

        if explicit_name:
            try:
                body_id = self._env.sim.model.body_name2id(explicit_name)
                source_body_name = explicit_name
                geom_ids = [
                    gid for gid in range(self._env.sim.model.ngeom)
                    if int(self._env.sim.model.geom_bodyid[gid]) == body_id
                    and int(self._env.sim.model.geom_group[gid]) == 0
                ]
            except Exception:
                pass
        if not geom_ids and explicit_name:
            try:
                geom_ids = [self._env.sim.model.geom_name2id(explicit_name)]
            except Exception:
                pass
        if not geom_ids:
            geom_names = self._find_prefixed_geom_names(
                prefix, include_tokens, reference_pos=reference_pos
            )
            collision = [
                self._env.sim.model.geom_name2id(n)
                for n in geom_names
                if int(self._env.sim.model.geom_group[self._env.sim.model.geom_name2id(n)]) == 0
            ]
            geom_ids = collision or [self._env.sim.model.geom_name2id(n) for n in geom_names]
        if not geom_ids:
            return None

        geom_infos = []
        for gid in geom_ids:
            gname = self._env.sim.model.geom_id2name(gid)
            if not gname:
                continue
            info = self._geom_pose_dict(gname, reference_pos=reference_pos)
            info["geom_group"] = int(self._env.sim.model.geom_group[gid])
            geom_infos.append(info)
        if not geom_infos:
            return None

        positions = np.array([i["pos"] for i in geom_infos], dtype=np.float64)
        centroid = positions.mean(axis=0)
        normals = np.array([i["surface_normal"] for i in geom_infos], dtype=np.float64)
        surface_normal = normals.mean(axis=0)
        n = float(np.linalg.norm(surface_normal))
        surface_normal = surface_normal / n if n > 1e-6 else np.array(geom_infos[0]["surface_normal"])

        rep = min(geom_infos, key=lambda i: float(np.linalg.norm(np.array(i["pos"]) - centroid)))
        geom_type_name = rep["geom_type_name"]
        geom_type_id = rep["geom_type_id"]
        for i in geom_infos:
            if i["geom_type_name"] in round_types:
                geom_type_name = i["geom_type_name"]
                geom_type_id = i["geom_type_id"]
                break

        geom_sizes = np.array([i["size"] for i in geom_infos], dtype=np.float64)
        approx_half_extents = np.max(np.abs(positions - centroid) + geom_sizes, axis=0)

        if source_body_name is not None:
            bp = self._body_pose_dict(source_body_name)
            quat_xyzw = bp["quat_xyzw"]
            rot_mat = bp["rot_mat"]
        else:
            quat_xyzw = rep["quat_xyzw"]
            rot_mat = rep["rot_mat"]

        return {
            "geom_name": rep["geom_name"],
            "geom_type_id": geom_type_id,
            "geom_type_name": geom_type_name,
            "pos": centroid.tolist(),
            "quat_xyzw": quat_xyzw,
            "rot_mat": rot_mat,
            "size": approx_half_extents.tolist(),
            "surface_normal": surface_normal.tolist(),
            "member_geom_names": [i["geom_name"] for i in geom_infos],
            "member_geom_type_names": [i["geom_type_name"] for i in geom_infos],
            "source_body_name": source_body_name,
            "target_method": "handle_body_center" if source_body_name else "handle_geom_centroid",
        }

    def get_oracle_targets(self) -> dict:
        """Return structured oracle targets for the current task.

        Covers all major RoboCasa365 task families: press (microwave, kettle),
        turn (faucet, stove), handle (drawer, cabinet, dishwasher), pick-place,
        and navigate. Returns ``supported=False`` with a ``reason`` key for
        unrecognised tasks.
        """
        task_name = self._env_name.split(":")[-1]
        result: dict = {"env_name": self._env_name, "task_name": task_name, "supported": False}
        task_info = self.get_task_info()
        result["task_info"] = task_info

        with self._lock:
            env = self._env

            def _fixture_payload(fixture) -> dict:
                payload = self._body_pose_dict(fixture.root_body)
                payload.update(
                    name=fixture.name,
                    type=type(fixture).__name__,
                    nat_lang=getattr(fixture, "nat_lang", fixture.name),
                    root_body=fixture.root_body,
                )
                return payload

            def _find_fridge_handle_geom_name(
                *,
                ctrl_name: str,
                prefix: str,
                reference_pos: np.ndarray,
            ) -> str | None:
                if "left" in ctrl_name:
                    token_sets = [
                        ("fridge", "left", "door", "handle"),
                        ("left", "door", "handle"),
                        ("left", "handle"),
                    ]
                    exclude_tokens = ("freezer", "right")
                elif "right" in ctrl_name:
                    token_sets = [
                        ("fridge", "right", "door", "handle"),
                        ("right", "door", "handle"),
                        ("right", "handle"),
                    ]
                    exclude_tokens = ("freezer", "left")
                else:
                    token_sets = [
                        ("fridge", "door", "handle"),
                        ("door", "handle"),
                        ("fridge", "handle"),
                        ("handle",),
                    ]
                    exclude_tokens = ("freezer",)

                for tokens in token_sets:
                    geom_name = self._find_prefixed_geom_name(
                        prefix,
                        tokens,
                        exclude_tokens=exclude_tokens,
                        reference_pos=reference_pos,
                    )
                    if geom_name:
                        return geom_name
                for tokens in token_sets:
                    geom_name = self._find_prefixed_geom_name(
                        "",
                        tokens,
                        exclude_tokens=exclude_tokens,
                        reference_pos=reference_pos,
                    )
                    if geom_name:
                        return geom_name
                return None

            try:
                if task_name in {"TurnOnMicrowave", "TurnOffMicrowave"}:
                    fixture = env.microwave
                    fixture_info = _fixture_payload(fixture)
                    ctrl = "start_button" if task_name == "TurnOnMicrowave" else "stop_button"
                    target = self._geom_pose_dict(
                        f"{fixture.naming_prefix}{ctrl}",
                        reference_pos=np.array(fixture_info["pos"]),
                    )
                    target.update(kind="press", recommended_standoff=0.06,
                                  recommended_press_distance=0.025, recommended_retreat_distance=0.10)
                    result.update(supported=True, fixture=fixture_info,
                                  behavior="turn_on" if task_name == "TurnOnMicrowave" else "turn_off",
                                  active_control=ctrl, target=target,
                                  controls={ctrl: target}, fixture_state=fixture.get_state())

                elif task_name == "TurnOnElectricKettle":
                    fixture = env.electric_kettle
                    fixture_info = _fixture_payload(fixture)
                    target = self._geom_pose_dict(
                        f"{fixture.naming_prefix}switch_main",
                        reference_pos=np.array(fixture_info["pos"]),
                    )
                    target.update(self._joint_pose_dict(fixture._joint_names["switch"]))
                    target.update(kind="press", recommended_standoff=0.04,
                                  recommended_press_distance=0.02, recommended_retreat_distance=0.08)
                    result.update(supported=True, fixture=fixture_info, behavior="turn_on",
                                  active_control="switch", target=target,
                                  controls={"switch": target}, fixture_state=fixture.get_state(env))

                elif task_name in {"TurnOnSinkFaucet", "TurnOffSinkFaucet"}:
                    fixture = env.sink
                    fixture_info = _fixture_payload(fixture)
                    target = self._geom_pose_dict(
                        f"{fixture.naming_prefix}handle_main",
                        reference_pos=np.array(fixture_info["pos"]),
                    )
                    target.update(self._joint_pose_dict(f"{fixture.naming_prefix}handle_joint"))
                    target.update(kind="turn", recommended_standoff=0.05,
                                  recommended_turn_amount=0.30, recommended_retreat_distance=0.08)
                    result.update(supported=True, fixture=fixture_info,
                                  behavior="turn_on" if task_name == "TurnOnSinkFaucet" else "turn_off",
                                  active_control="handle", target=target,
                                  controls={"handle": target},
                                  fixture_state=fixture.get_handle_state(env))

                elif task_name in {"TurnOnStove", "TurnOffStove", "LowerHeat"}:
                    fixture = env.stove
                    fixture_info = _fixture_payload(fixture)
                    knob_name = getattr(env, "knob", None)
                    if knob_name is None:
                        raise KeyError("stove knob not exposed on task env")
                    target = self._geom_pose_dict(
                        f"{fixture.naming_prefix}knob_{knob_name}_main",
                        reference_pos=np.array(fixture_info["pos"]),
                    )
                    target.update(self._joint_pose_dict(
                        f"{fixture.naming_prefix}knob_{knob_name}_joint"
                    ))
                    target.update(kind="turn", recommended_standoff=0.05,
                                  recommended_turn_amount=0.60, recommended_retreat_distance=0.08)
                    result.update(supported=True, fixture=fixture_info,
                                  behavior=getattr(env, "behavior", "turn_on"),
                                  active_control=knob_name, target=target,
                                  controls={knob_name: target},
                                  fixture_state=fixture.get_knobs_state(env))

                elif task_name == "OpenDrawer":
                    fixture = env.drawer
                    fixture_info = _fixture_payload(fixture)
                    joint_name = fixture.door_joint_names[0]
                    state = fixture.get_door_state(env)
                    sem_frac = float(next(iter(state.values())))
                    handle_name = getattr(fixture, "handle_name", None)
                    if handle_name:
                        try:
                            self._env.sim.model.geom_name2id(handle_name)
                        except Exception:
                            handle_name = None
                    if not handle_name:
                        handle_name = self._find_prefixed_geom_name(
                            fixture.naming_prefix, ("handle",),
                            reference_pos=np.array(fixture_info["pos"]),
                        )
                    if not handle_name:
                        raise KeyError("drawer handle geom unavailable")
                    target = self._geom_pose_dict(
                        handle_name, reference_pos=np.array(fixture_info["pos"])
                    )
                    target.update(self._joint_pose_dict(joint_name))
                    target.update(kind="slider_handle", desired_fraction=1.0,
                                  current_fraction=sem_frac, normalized_qpos=sem_frac,
                                  recommended_standoff=0.05, recommended_contact_offset=0.015,
                                  recommended_travel_distance=0.18, recommended_retreat_distance=0.10)
                    result.update(supported=True, fixture=fixture_info, behavior="open",
                                  active_control="handle", target=target,
                                  controls={"handle": target}, fixture_state=state)

                elif task_name in {"OpenCabinet", "CloseFridge", "CloseToasterOvenDoor"}:
                    if task_name == "CloseToasterOvenDoor":
                        fixture = env.toaster_oven
                    else:
                        fixture = getattr(env, "fxtr", None)
                    if fixture is None:
                        raise KeyError(f"{task_name} fixture unavailable")
                    fixture_info = _fixture_payload(fixture)
                    ref_pos = np.asarray(fixture_info["pos"], dtype=np.float64)
                    controls: dict = {}

                    if task_name == "CloseToasterOvenDoor":
                        joint_name = fixture._joint_names["door"]
                        handle_name = (
                            self._find_prefixed_geom_name(
                                fixture.naming_prefix,
                                ("door", "handle"),
                                reference_pos=ref_pos,
                            )
                            or self._find_prefixed_geom_name(
                                fixture.naming_prefix,
                                ("handle",),
                                reference_pos=ref_pos,
                            )
                            or self._find_prefixed_geom_name(
                                fixture.naming_prefix,
                                ("door",),
                                reference_pos=ref_pos,
                                prefer="farthest",
                            )
                        )
                        if not handle_name:
                            raise KeyError("toaster oven door handle unavailable")
                        target = self._geom_pose_dict(handle_name, reference_pos=ref_pos)
                        target.update(self._joint_pose_dict(joint_name))
                        target.update(
                            kind="hinge_handle",
                            desired_fraction=0.0,
                            recommended_standoff=0.06,
                            recommended_contact_offset=0.015,
                            recommended_travel_distance=0.80,
                            recommended_retreat_distance=0.10,
                        )
                        controls["door_handle"] = target
                        result.update(
                            supported=True,
                            fixture=fixture_info,
                            behavior="close",
                            active_control="door_handle",
                            target=target,
                            controls=controls,
                            fixture_state=fixture.get_state(env),
                        )
                        return result

                    if task_name == "CloseFridge":
                        door_joint_names = list(
                            getattr(fixture, "_fridge_door_joint_names", [])
                        )
                    else:
                        door_joint_names = list(getattr(fixture, "door_joint_names", []))
                    explicit_handles: list = []
                    if task_name == "CloseFridge":
                        if len(door_joint_names) > 1:
                            explicit_handles = [
                                (
                                    "left_handle",
                                    _find_fridge_handle_geom_name(
                                        ctrl_name="left_handle",
                                        prefix=fixture.naming_prefix,
                                        reference_pos=ref_pos,
                                    ),
                                    self._match_joint_name(door_joint_names, ("left",)),
                                ),
                                (
                                    "right_handle",
                                    _find_fridge_handle_geom_name(
                                        ctrl_name="right_handle",
                                        prefix=fixture.naming_prefix,
                                        reference_pos=ref_pos,
                                    ),
                                    self._match_joint_name(door_joint_names, ("right",)),
                                ),
                            ]
                        else:
                            explicit_handles = [
                                (
                                    "handle",
                                    _find_fridge_handle_geom_name(
                                        ctrl_name="handle",
                                        prefix=fixture.naming_prefix,
                                        reference_pos=ref_pos,
                                    ),
                                    door_joint_names[0] if door_joint_names else None,
                                ),
                            ]

                    elif hasattr(fixture, "left_handle_name") and hasattr(fixture, "right_handle_name"):
                        explicit_handles = [
                            (
                                "left_handle",
                                getattr(fixture, "left_handle_name", None),
                                self._match_joint_name(door_joint_names, ("left",)),
                            ),
                            (
                                "right_handle",
                                getattr(fixture, "right_handle_name", None),
                                self._match_joint_name(door_joint_names, ("right",)),
                            ),
                        ]
                    elif hasattr(fixture, "handle_name"):
                        explicit_handles = [
                            (
                                "handle",
                                getattr(fixture, "handle_name", None),
                                door_joint_names[0] if door_joint_names else None,
                            ),
                        ]

                    if not explicit_handles:
                        if len(door_joint_names) > 1:
                            explicit_handles = [
                                (
                                    "left_handle",
                                    self._find_prefixed_geom_name(
                                        fixture.naming_prefix,
                                        ("handle", "left"),
                                        reference_pos=ref_pos,
                                    ),
                                    self._match_joint_name(door_joint_names, ("left",)),
                                ),
                                (
                                    "right_handle",
                                    self._find_prefixed_geom_name(
                                        fixture.naming_prefix,
                                        ("handle", "right"),
                                        reference_pos=ref_pos,
                                    ),
                                    self._match_joint_name(door_joint_names, ("right",)),
                                ),
                            ]
                        else:
                            explicit_handles = [
                                (
                                    "handle",
                                    self._find_prefixed_geom_name(
                                        fixture.naming_prefix,
                                        ("handle",),
                                        reference_pos=ref_pos,
                                    ),
                                    door_joint_names[0] if door_joint_names else None,
                                ),
                            ]

                    for ctrl_name, geom_name, jname in explicit_handles:
                        if task_name == "CloseFridge":
                            tokens = (
                                ("fridge", "left", "door", "handle")
                                if "left" in ctrl_name
                                else ("fridge", "right", "door", "handle")
                                if "right" in ctrl_name
                                else ("fridge", "door", "handle")
                            )
                        else:
                            tokens = (
                                ("handle", "left")
                                if "left" in ctrl_name
                                else ("handle", "right")
                                if "right" in ctrl_name
                                else ("handle",)
                            )
                        tgt = self._aggregate_handle_pose_dict(
                            explicit_name=geom_name,
                            prefix=fixture.naming_prefix,
                            include_tokens=tokens,
                            reference_pos=ref_pos,
                        )
                        if tgt is None or not jname:
                            continue
                        tgt.update(self._joint_pose_dict(jname))
                        joint_range = tgt.get("range", [0.0, 0.0])
                        joint_min = float(joint_range[0]) if len(joint_range) >= 2 else 0.0
                        joint_max = float(joint_range[1]) if len(joint_range) >= 2 else 0.0
                        semantic_fraction = float(tgt.get("normalized_qpos", 0.0))
                        if joint_max > joint_min and joint_min < 0.0:
                            semantic_fraction = 1.0 - semantic_fraction
                        tgt.update(
                            kind="hinge_handle",
                            desired_fraction=1.0 if task_name == "OpenCabinet" else 0.0,
                            current_fraction=semantic_fraction,
                            normalized_qpos=semantic_fraction,
                            recommended_standoff=0.06,
                            recommended_contact_offset=0.015,
                            recommended_travel_distance=0.80,
                            recommended_retreat_distance=0.10,
                        )
                        controls[ctrl_name] = tgt

                    active_control = None
                    active_target = None
                    if controls:
                        def _gap(item):
                            t = item[1]
                            return abs(
                                float(t.get("desired_fraction", 0.0))
                                - float(t.get("normalized_qpos", 0.0))
                            )

                        active_control, active_target = max(controls.items(), key=_gap)

                    fixture_state_getter = getattr(fixture, "get_door_state", None)
                    if callable(fixture_state_getter):
                        fixture_state = fixture_state_getter(env)
                    elif len(controls) == 1:
                        only_target = next(iter(controls.values()))
                        fixture_state = {
                            "door": float(only_target.get("current_fraction", 0.0))
                        }
                    else:
                        fixture_state = {}

                    result.update(
                        supported=bool(controls),
                        fixture=fixture_info,
                        behavior="open" if task_name == "OpenCabinet" else "close",
                        active_control=active_control,
                        target=active_target,
                        controls=controls,
                        fixture_state=fixture_state,
                    )
                    if not controls:
                        result["reason"] = (
                            f"no hinge-handle controls found for fixture "
                            f"(prefix={fixture.naming_prefix}, joints={door_joint_names})"
                        )

                elif task_name == "SlideDishwasherRack":
                    fixture = env.dishwasher
                    fixture_info = _fixture_payload(fixture)
                    ref_pos = np.array(fixture_info["pos"])
                    handle_name = (
                        self._find_prefixed_geom_name(fixture.naming_prefix, ("rack", "handle"), reference_pos=ref_pos)
                        or self._find_prefixed_geom_name(fixture.naming_prefix, ("handle",), exclude_tokens=("door",), reference_pos=ref_pos)
                        or self._find_prefixed_geom_name(fixture.naming_prefix, ("reg", "rack1"), reference_pos=ref_pos)
                    )
                    if not handle_name:
                        raise KeyError("dishwasher rack target geom unavailable")
                    joint_name = fixture._joint_names["rack"]
                    state = fixture.get_state(env)
                    target = self._geom_pose_dict(handle_name, reference_pos=ref_pos)
                    target.update(self._joint_pose_dict(joint_name))
                    should_pull = getattr(env, "should_pull", True)
                    target.update(kind="slider_handle", desired_fraction=1.0 if should_pull else 0.0,
                                  recommended_standoff=0.06, recommended_contact_offset=0.01,
                                  recommended_travel_distance=0.16, recommended_retreat_distance=0.08)
                    result.update(supported=True, fixture=fixture_info,
                                  behavior="pull" if should_pull else "push",
                                  active_control="rack_handle", target=target,
                                  controls={"rack_handle": target}, fixture_state=state)

                elif task_name == "OpenStandMixerHead":
                    fixture = env.stand_mixer
                    fixture_info = _fixture_payload(fixture)
                    ref_pos = np.array(fixture_info["pos"])
                    button_geom = self._find_prefixed_geom_name(
                        fixture.naming_prefix, ("button", "head", "lock"), reference_pos=ref_pos
                    )
                    head_joint_name = fixture._joint_names["head"]
                    head_joint = self._joint_pose_dict(head_joint_name)
                    head_ref = np.array(head_joint["anchor_world"])
                    head_geom = self._find_prefixed_geom_name(
                        fixture.naming_prefix, ("head",), exclude_tokens=("button", "lock"),
                        reference_pos=head_ref, prefer="farthest"
                    )
                    head_target = (
                        self._geom_pose_dict(head_geom, reference_pos=ref_pos)
                        if head_geom else
                        self._body_pose_dict(
                            self._find_prefixed_body_name(fixture.naming_prefix, ("head",),
                                                          reference_pos=head_ref, prefer="farthest")
                        ) if self._find_prefixed_body_name(
                            fixture.naming_prefix, ("head",), reference_pos=head_ref, prefer="farthest"
                        ) else None
                    )
                    controls = {}
                    if button_geom:
                        bt = self._geom_pose_dict(button_geom, reference_pos=ref_pos)
                        bt.update(self._joint_pose_dict(fixture._joint_names["button_head_lock"]))
                        bt.update(kind="press", recommended_standoff=0.04,
                                  recommended_press_distance=0.015, recommended_retreat_distance=0.06)
                        controls["button_head_lock"] = bt
                    if head_target is not None:
                        head_target.update(head_joint)
                        head_target.update(kind="hinge_handle", desired_fraction=1.0,
                                           recommended_standoff=0.06, recommended_contact_offset=0.01,
                                           recommended_travel_distance=0.14, recommended_retreat_distance=0.10)
                        controls["head"] = head_target
                    result.update(supported=bool(controls), fixture=fixture_info, behavior="open",
                                  active_control="head" if "head" in controls else next(iter(controls), None),
                                  target=controls.get("head") or next(iter(controls.values()), None),
                                  controls=controls, fixture_state=fixture.get_state(env))

                elif task_name == "CloseBlenderLid":
                    fixture = env.blender
                    fixture_info = _fixture_payload(fixture)
                    lid_fixture = fixture.blender_lid
                    if lid_fixture is None:
                        raise KeyError("blender lid fixture unavailable")
                    lid_body = f"{lid_fixture.name}_main"
                    lid_info = self._body_pose_dict(lid_body)
                    target_pos = fixture.get_lid_closed_pos(env)
                    result.update(supported=True, fixture=fixture_info, behavior="close",
                                  lid_fixture={**lid_info, "name": lid_fixture.name,
                                               "type": type(lid_fixture).__name__,
                                               "nat_lang": getattr(lid_fixture, "nat_lang", lid_fixture.name),
                                               "root_body": lid_body},
                                  target={"kind": "pick_place",
                                          "pos": np.array(target_pos, dtype=np.float64).tolist(),
                                          "quat_xyzw": fixture_info["quat_xyzw"],
                                          "recommended_pick_standoff": 0.12,
                                          "recommended_place_standoff": 0.14},
                                  fixture_state=fixture.get_state())

                elif task_name == "CoffeeSetupMug":
                    fixture = env.coffee_machine
                    fixture_info = _fixture_payload(fixture)
                    mug_info = self._object_payload("obj")
                    mug_handle = self._estimate_object_handle_pose("obj")
                    mug_info["handle"] = mug_handle
                    site_name = fixture._receptacle_pouring_site.get("name")
                    target = self._site_pose_dict(site_name)
                    target.update(kind="pick_place", recommended_pick_standoff=0.12,
                                  recommended_place_standoff=0.10)
                    result.update(supported=True, fixture=fixture_info,
                                  behavior="counter_to_machine",
                                  active_control="receptacle_place_site", target=target,
                                  controls={"receptacle_place_site": target},
                                  object=mug_info, object_handle=mug_handle,
                                  fixture_state=fixture.get_state())

                elif task_name == "NavigateKitchen":
                    target_pos = np.array(getattr(env, "target_pos"), dtype=np.float64)
                    target_ori = np.array(getattr(env, "target_ori"), dtype=np.float64)
                    target_fixture = getattr(env, "target_fixture", None)
                    result.update(supported=True, behavior="navigate",
                                  fixture=_fixture_payload(target_fixture) if target_fixture else None,
                                  target={"kind": "base_pose",
                                          "pos": target_pos.tolist(),
                                          "yaw": float(target_ori[2])})

            except Exception as exc:
                result["reason"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._close_renderers()
        self._render_executor.shutdown(wait=True)
        self._env.close()

    def _close_renderers(self) -> None:
        def _close():
            if self._rgb_renderer is not None:
                self._rgb_renderer.close()
                self._rgb_renderer = None
            if self._depth_renderer is not None:
                self._depth_renderer.close()
                self._depth_renderer = None

        self._render_executor.submit(_close).result()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _obs_prefix(self, side: str) -> str:
        if self._profile and self._profile.is_bimanual:
            return f"robot0_{side}_"
        return "robot0_"

    def _normalize_gripper(self, gripper_qpos: np.ndarray) -> np.ndarray:
        raw = np.abs(gripper_qpos).mean()
        normalized = np.clip(raw / 0.04, 0.0, 1.0)
        return np.array([normalized], dtype=np.float64)

    def _resolve_mujoco_camera_name(self, camera_name: str) -> str:
        obs_key = self._camera_map.get(camera_name, "")
        return obs_key.replace("_image", "").replace("_depth", "")
