# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo Warp simulation backend for the YAM station (--env yam-warp).

GPU-accelerated physics (MuJoCo Warp) with ray-traced rendering via NVIDIA
Warp's BVH API.  Same public interface as SimBackend so SimArmClient and
SimCameraClient work unchanged.
"""

from __future__ import annotations

import threading

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from enpire.env.forge.cap.config import CONTROL_PERIOD_S
from enpire.env.forge.cap.server.scene_manager import add_objects_to_spec, list_scenes, load_scene
from enpire.env.forge.robot.models.station.paths import get_station_xml

_MODEL_PATH = get_station_xml()

# Pre-allocated on first use to avoid per-call overhead.
_NJMAX = 256


class WarpSimBackend:
    """Shared MuJoCo Warp simulation state for both arms and all cameras.

    Drop-in replacement for SimBackend — uses GPU physics (mujoco-warp) and
    ray-traced rendering instead of CPU MuJoCo + OpenGL rasterization.
    """

    GRIPPER_CTRL_SCALE = 0.041
    GRIPPER_QPOS_SCALE = 0.0376
    CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640

    def __init__(self):
        # Initialize Warp runtime
        wp.init()

        spec = self._load_base_spec()

        # Store stable name lists from spec (survive recompilation)
        self._left_joint_names = [
            x.name for x in spec.joints if x.name.startswith("left_")
        ]
        self._right_joint_names = [
            x.name for x in spec.joints if x.name.startswith("right_")
        ]
        self._left_act_names = [
            x.name for x in spec.actuators if x.name.startswith("left_")
        ]
        self._right_act_names = [
            x.name for x in spec.actuators if x.name.startswith("right_")
        ]
        self._mj_camera_names: dict[str, str] = {}
        for side in ["top", "left", "right"]:
            cam = next(x for x in spec.cameras if side in x.name)
            self._mj_camera_names[side] = cam.name

        # Compile CPU model (needed for intrinsics, name lookups, recompilation)
        self._mj_model, self._mj_data = self._compile_from_spec(spec)

        # Set initial state on CPU and run forward pass so all derived
        # quantities (xpos, xmat, cam_xpos, cam_xmat) are computed before
        # uploading to GPU.
        self._init_grippers_open()
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # Physics stepping
        self._n_substeps = round(CONTROL_PERIOD_S / self._mj_model.opt.timestep)

        # Thread safety
        self._physics_lock = threading.Lock()

        # Create GPU model/data from the forward-computed CPU state
        self._warp_model = mjw.put_model(self._mj_model)
        self._warp_data = mjw.put_data(self._mj_model, self._mj_data, njmax=_NJMAX)

        # Render context — cam_res=(width, height), enable both RGB and depth
        self._render_ctx = mjw.create_render_context(
            self._mj_model,
            cam_res=(self.CAMERA_WIDTH, self.CAMERA_HEIGHT),
            render_rgb=True,
            render_depth=True,
        )

        # Pre-allocate output buffers for rendering (nworld=1, H, W)
        self._rgb_buf = wp.zeros(
            (1, self.CAMERA_HEIGHT, self.CAMERA_WIDTH), dtype=wp.vec3
        )
        self._depth_buf = wp.zeros(
            (1, self.CAMERA_HEIGHT, self.CAMERA_WIDTH), dtype=float
        )

        # Scene tracking
        self._active_scene: str | None = None
        self._scene_body_names: list[str] = []

    # ------------------------------------------------------------------
    # Spec / compile helpers
    # ------------------------------------------------------------------

    def _load_base_spec(self) -> mujoco.MjSpec:
        """Load station.xml into a fresh MjSpec with camera overrides."""
        spec = mujoco.MjSpec.from_file(str(_MODEL_PATH))
        spec.copy_during_attach = True
        for side in ["top", "left", "right"]:
            camera = next(x for x in spec.cameras if side in x.name)
            camera.resolution = (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
            camera.sensor_size = (0.003148, 0.002364)
        return spec

    def _compile_from_spec(
        self, spec: mujoco.MjSpec
    ) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Compile spec and resolve joint/actuator IDs by name."""
        model = spec.compile()
        data = mujoco.MjData(model)
        self._joint_ids = {
            "left": np.array([model.joint(n).id for n in self._left_joint_names]),
            "right": np.array([model.joint(n).id for n in self._right_joint_names]),
        }
        self._actuator_ids = {
            "left": np.array([model.actuator(n).id for n in self._left_act_names]),
            "right": np.array([model.actuator(n).id for n in self._right_act_names]),
        }
        self._camera_names: dict[str, str] = dict(self._mj_camera_names)
        return model, data

    def _init_grippers_open(self) -> None:
        """Set gripper joints and actuators to the open position (CPU side)."""
        for side in ("left", "right"):
            joint_ids = self._joint_ids[side]
            act_ids = self._actuator_ids[side]
            self._mj_data.qpos[joint_ids[6:8]] = [
                self.GRIPPER_QPOS_SCALE,
                -self.GRIPPER_QPOS_SCALE,
            ]
            self._mj_data.ctrl[act_ids[6:7]] = self.GRIPPER_CTRL_SCALE

    # ------------------------------------------------------------------
    # CPU ↔ GPU sync helpers
    # ------------------------------------------------------------------

    def _sync_gpu_to_cpu(self) -> None:
        """Pull GPU state back to CPU MjData."""
        mjw.get_data_into(self._mj_data, self._mj_model, self._warp_data)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def _recompile_with_scene(self, spec: mujoco.MjSpec) -> None:
        """Hot-swap model/data/render-ctx thread-safely from a modified spec."""
        # Save current robot ctrl from GPU
        self._sync_gpu_to_cpu()
        old_ctrl = {}
        for side in ("left", "right"):
            act_ids = self._actuator_ids[side]
            old_ctrl[side] = self._mj_data.ctrl[act_ids].copy()

        with self._physics_lock:
            # Compile new CPU model
            new_model, new_data = self._compile_from_spec(spec)

            # Restore robot ctrl
            for side in ("left", "right"):
                act_ids = self._actuator_ids[side]
                new_data.ctrl[act_ids] = old_ctrl[side]

            # Swap CPU model/data
            self._mj_model = new_model
            self._mj_data = new_data

            self._init_grippers_open()
            mujoco.mj_forward(self._mj_model, self._mj_data)

            # Re-create GPU objects (forward already computed, so BVH is correct)
            self._warp_model = mjw.put_model(self._mj_model)
            self._warp_data = mjw.put_data(self._mj_model, self._mj_data, njmax=_NJMAX)
            self._render_ctx = mjw.create_render_context(
                self._mj_model,
                cam_res=(self.CAMERA_WIDTH, self.CAMERA_HEIGHT),
                render_rgb=True,
                render_depth=True,
            )
            # Re-allocate render output buffers
            self._rgb_buf = wp.zeros(
                (1, self.CAMERA_HEIGHT, self.CAMERA_WIDTH), dtype=wp.vec3
            )
            self._depth_buf = wp.zeros(
                (1, self.CAMERA_HEIGHT, self.CAMERA_WIDTH), dtype=float
            )

    def setup_scene(self, name: str) -> dict:
        """Load a scene YAML and inject objects into the simulation."""
        scene = load_scene(name)
        spec = self._load_base_spec()
        body_names = add_objects_to_spec(spec, scene)
        self._recompile_with_scene(spec)
        self._active_scene = name
        self._scene_body_names = body_names
        return {"ok": True, "scene": name, "objects": body_names}

    def clear_table(self) -> dict:
        """Remove all scene objects, restoring the bare station."""
        removed = len(self._scene_body_names)
        if removed == 0:
            return {"ok": True, "removed": 0}
        spec = self._load_base_spec()
        self._recompile_with_scene(spec)
        self._active_scene = None
        self._scene_body_names = []
        return {"ok": True, "removed": removed}

    def set_body_pose(
        self,
        name: str,
        pos: list[float],
        quat_wxyz: list[float],
        gravity_comp: bool = True,
    ) -> dict:
        """Set a scene body's pose (warp backend stub — not yet implemented)."""
        import logging

        logging.getLogger(__name__).warning(
            "set_body_pose not implemented for warp backend"
        )
        return {"ok": False, "error": "set_body_pose not implemented for warp backend"}

    def get_object_positions(self) -> dict:
        """Return positions and sizes of all scene objects."""
        result = {}
        with self._physics_lock:
            self._sync_gpu_to_cpu()
            for name in self._scene_body_names:
                body_id = self._mj_model.body(name).id
                pos = self._mj_data.xpos[body_id].copy().tolist()
                quat = self._mj_data.xquat[body_id].copy().tolist()
                geom_name = f"{name}_geom"
                try:
                    size = self._mj_model.geom(geom_name).size.copy().tolist()
                except Exception:
                    size = []
                result[name] = {"pos": pos, "quat": quat, "size": size}
        return {"ok": True, "objects": result}

    def get_scenes(self) -> dict:
        """List available scenes and the currently active one."""
        return {"ok": True, "scenes": list_scenes(), "active": self._active_scene}

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance physics by n_substeps on GPU. Call once per control tick."""
        with self._physics_lock:
            for _ in range(self._n_substeps):
                mjw.step(self._warp_model, self._warp_data)

    def get_arm_observation(self, side: str) -> dict[str, np.ndarray]:
        """Read joint positions for one arm. Returns {'joint_pos': 6D, 'gripper_pos': 1D}."""
        with self._physics_lock:
            self._sync_gpu_to_cpu()
            joint_ids = self._joint_ids[side]
            qpos = self._mj_data.qpos[joint_ids].copy()
        qpos_arm = qpos[:6]
        qpos_gripper = np.abs(qpos[6:]).mean(keepdims=True) / self.GRIPPER_QPOS_SCALE
        return {
            "joint_pos": qpos_arm,
            "gripper_pos": qpos_gripper,
        }

    def command_arm(self, side: str, cmd: dict) -> None:
        """Write commanded positions to ctrl on GPU.

        Only updates the ctrl array — does NOT recreate the entire warp Data,
        which would reset qpos/qvel and undo physics progress.
        """
        pos = np.asarray(cmd["pos"], dtype=np.float64)
        act_ids = self._actuator_ids[side]
        ctrl = np.empty(len(act_ids))
        ctrl[:6] = pos[:6]
        ctrl[6] = pos[6] * self.GRIPPER_CTRL_SCALE  # 0-1 → MuJoCo scale
        with self._physics_lock:
            # Read current ctrl from GPU, update only our actuators, write back
            ctrl_gpu = self._warp_data.ctrl.numpy()  # (nworld, nu)
            ctrl_gpu[0, act_ids] = ctrl
            self._warp_data.ctrl.assign(ctrl_gpu)

    # ------------------------------------------------------------------
    # Rendering (ray-traced)
    # ------------------------------------------------------------------

    def render_rgb(self, camera_name: str) -> np.ndarray:
        """Render ray-traced RGB image (H×W×3 uint8) for a named camera."""
        mj_cam_name = self._camera_names[camera_name]
        cam_id = self._mj_model.camera(mj_cam_name).id
        with self._physics_lock:
            mjw.refit_bvh(self._warp_model, self._warp_data, self._render_ctx)
            mjw.render(self._warp_model, self._warp_data, self._render_ctx)
            mjw.get_rgb(self._render_ctx, cam_id, self._rgb_buf)
        # rgb_buf is (1, H, W) of vec3 floats [0,1] — convert to (H, W, 3) uint8
        rgb_np = self._rgb_buf.numpy()[0]  # (H, W, 3)
        return (rgb_np * 255).clip(0, 255).astype(np.uint8)

    def render_depth(self, camera_name: str) -> np.ndarray:
        """Render ray-traced depth image (H×W float32) for a named camera."""
        mj_cam_name = self._camera_names[camera_name]
        cam_id = self._mj_model.camera(mj_cam_name).id
        with self._physics_lock:
            mjw.refit_bvh(self._warp_model, self._warp_data, self._render_ctx)
            mjw.render(self._warp_model, self._warp_data, self._render_ctx)
            mjw.get_depth(self._render_ctx, cam_id, 1.0, self._depth_buf)
        return self._depth_buf.numpy()[0].astype(np.float32)  # (H, W)

    def get_camera_intrinsics(self, camera_name: str) -> list[float]:
        """Compute [fx, fy, cx, cy] from MuJoCo camera model parameters."""
        import math

        mj_cam_name = self._camera_names[camera_name]
        cam_id = self._mj_model.camera(mj_cam_name).id

        if hasattr(self._mj_model, "cam_focal"):
            focal = self._mj_model.cam_focal[cam_id]
            sensor = self._mj_model.cam_sensorsize[cam_id]
            fx = focal[0] / sensor[0] * self.CAMERA_WIDTH
            fy = focal[1] / sensor[1] * self.CAMERA_HEIGHT
        else:
            fovy = self._mj_model.cam_fovy[cam_id]
            fy = self.CAMERA_HEIGHT / (2.0 * math.tan(math.radians(fovy) / 2.0))
            fx = fy

        cx = self.CAMERA_WIDTH / 2.0
        cy = self.CAMERA_HEIGHT / 2.0
        return [float(fx), float(fy), float(cx), float(cy)]

    def get_camera_extrinsics(self, camera_name: str) -> dict:
        """Return camera position and rotation from MuJoCo (matches rendered images).

        MuJoCo camera convention: x-right, y-up, -z-forward.
        Pinocchio/URDF camera convention: x-left, y-up, z-forward.
        We convert by negating the x and z columns of the rotation matrix.
        """
        mj_cam_name = self._camera_names[camera_name]
        cam_id = self._mj_model.camera(mj_cam_name).id
        with self._physics_lock:
            self._sync_gpu_to_cpu()
            pos = self._mj_data.cam_xpos[cam_id].copy()
            rot = self._mj_data.cam_xmat[cam_id].copy().reshape(3, 3)
        # Convert MuJoCo camera frame to Pinocchio convention (x-left, y-up, z-forward)
        rot = rot @ np.diag([-1.0, 1.0, -1.0])
        return {"position": pos.tolist(), "rotation": rot.tolist()}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._warp_data = None
        self._warp_model = None
        self._render_ctx = None
