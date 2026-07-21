"""MuJoCo simulation backend for the YAM station (--env yam).

Provides SimBackend (shared MuJoCo state), SimArmClient, and SimCameraClient
as drop-in replacements for _ArmClient and _CameraClient.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
from pathlib import Path

# Must be set before `import mujoco` — gl_context.py reads MUJOCO_GL at import
# time to select the GL backend. On headless NVIDIA machines, EGL is required.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np

from enpire.env.forge.cap.config import CONTROL_PERIOD_S
from enpire.env.forge.cap.server.scene_manager import add_objects_to_spec, list_scenes, load_scene

from enpire.env.forge.robot.models.station.paths import get_station_xml

_MODEL_PATH = get_station_xml()


class SimBackend:
    """Shared MuJoCo simulation state for both arms and all cameras."""

    GRIPPER_CTRL_SCALE = 0.041
    GRIPPER_QPOS_SCALE = 0.0376
    CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640

    def __init__(self, viewer: bool = False):
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

        # Compile and resolve IDs
        self._model, self._data = self._compile_from_spec(spec)

        # Physics stepping
        self._n_substeps = round(CONTROL_PERIOD_S / self._model.opt.timestep)

        # Thread safety
        self._physics_lock = threading.Lock()

        # Renderers (EGL thread affinity)
        self._rgb_renderer: mujoco.Renderer | None = None
        self._depth_renderer: mujoco.Renderer | None = None
        self._render_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mj-render"
        )

        # Scene tracking
        self._active_scene: str | None = None
        self._scene_body_names: list[str] = []
        # body_id -> list of geom ids (built once at scene setup, avoids full scan)
        self._body_geom_ids: dict[int, list[int]] = {}

        self._init_grippers_open()
        mujoco.mj_forward(self._model, self._data)

        # Optional passive viewer
        self._viewer = None
        if viewer:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

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
        # Reduce contact slip in soft-constraint model
        spec.option.impratio = 50.0  # friction impedance >> normal impedance
        spec.option.noslip_iterations = 10  # post-solve friction slip suppression
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
        """Set gripper joints and actuators to the open position."""
        for side in ("left", "right"):
            joint_ids = self._joint_ids[side]
            act_ids = self._actuator_ids[side]
            self._data.qpos[joint_ids[6:8]] = [
                self.GRIPPER_QPOS_SCALE,
                -self.GRIPPER_QPOS_SCALE,
            ]
            self._data.ctrl[act_ids[6:7]] = self.GRIPPER_CTRL_SCALE

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def _close_renderers(self) -> None:
        """Close and clear renderers on the render thread (EGL affinity).

        Both close and None-assignment happen on the render thread to avoid
        races with _render_rgb_impl / _render_depth_impl.
        """

        def _close():
            if self._rgb_renderer is not None:
                self._rgb_renderer.close()
                self._rgb_renderer = None
            if self._depth_renderer is not None:
                self._depth_renderer.close()
                self._depth_renderer = None

        self._render_executor.submit(_close).result()

    def _recompile_with_scene(self, spec: mujoco.MjSpec) -> None:
        """Hot-swap model/data/renderers thread-safely from a modified spec."""
        # Save current robot ctrl state
        old_ctrl = {}
        for side in ("left", "right"):
            act_ids = self._actuator_ids[side]
            old_ctrl[side] = self._data.ctrl[act_ids].copy()

        # Close renderers on render thread before swapping model
        self._close_renderers()

        # Compile new model
        new_model, new_data = self._compile_from_spec(spec)

        # Restore robot ctrl
        for side in ("left", "right"):
            act_ids = self._actuator_ids[side]
            new_data.ctrl[act_ids] = old_ctrl[side]

        # Swap under physics lock (include init + forward to avoid data races)
        with self._physics_lock:
            self._model = new_model
            self._data = new_data
            self._init_grippers_open()
            mujoco.mj_forward(self._model, self._data)

        # Invalidate any renderer that was lazily created between the first
        # _close_renderers() and the model swap — it would be bound to the
        # old model and unable to render the new scene objects.
        self._close_renderers()

        # Update viewer if active
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def setup_scene(self, name: str) -> dict:
        """Load a scene YAML and inject objects into the simulation.

        Returns {"ok": True, "scene": name, "objects": [...]}.
        """
        scene = load_scene(name)

        # Re-parse base spec (MjSpec has no copy)
        spec = self._load_base_spec()
        body_names = add_objects_to_spec(spec, scene)

        self._recompile_with_scene(spec)
        self._active_scene = name
        self._scene_body_names = body_names
        self._build_body_geom_index()

        return {"ok": True, "scene": name, "objects": body_names}

    def clear_table(self) -> dict:
        """Remove all scene objects, restoring the bare station.

        Returns {"ok": True, "removed": count}.
        """
        removed = len(self._scene_body_names)
        if removed == 0:
            return {"ok": True, "removed": 0}

        # Recompile from clean base spec
        spec = self._load_base_spec()
        self._recompile_with_scene(spec)
        self._active_scene = None
        self._scene_body_names = []
        self._body_geom_ids = {}

        return {"ok": True, "removed": removed}

    def _build_body_geom_index(self) -> None:
        """Build body_id → [geom_id, ...] lookup for scene bodies only."""
        self._body_geom_ids = {}
        for name in self._scene_body_names:
            body_id = self._model.body(name).id
            gids = []
            for gid in range(self._model.ngeom):
                if self._model.geom_bodyid[gid] == body_id:
                    gids.append(gid)
            self._body_geom_ids[body_id] = gids

    def set_body_pose(self, name: str, pos: list[float], quat_wxyz: list[float], gravity_comp: bool = True) -> dict:
        """Set a scene body's pose and optionally enable gravity compensation.

        Args:
            name: Body name (must be in active scene).
            pos: [x, y, z] world position.
            quat_wxyz: [w, x, y, z] quaternion (MuJoCo convention).
            gravity_comp: If True, apply gravity compensation so the body floats.

        Returns {"ok": True} on success.
        """
        if name not in self._scene_body_names:
            return {"ok": False, "error": f"Body {name!r} not in active scene"}

        with self._physics_lock:
            body_id = self._model.body(name).id
            jnt_id = self._model.body_jntadr[body_id]
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            qvel_adr = self._model.jnt_dofadr[jnt_id]

            # freejoint qpos: [x, y, z, qw, qx, qy, qz]
            self._data.qpos[qpos_adr: qpos_adr + 3] = pos
            self._data.qpos[qpos_adr + 3: qpos_adr + 7] = quat_wxyz
            # zero velocity
            self._data.qvel[qvel_adr: qvel_adr + 6] = 0.0

            if gravity_comp:
                self._model.body_gravcomp[body_id] = 1.0

            mujoco.mj_forward(self._model, self._data)

        return {"ok": True}

    def get_object_positions(self) -> dict:
        """Return positions and sizes of all scene objects.

        Returns {"ok": True, "objects": {name: {"pos", "quat", "size", "geoms"}, ...}}.
        Geom static data (type/size/local_pos/rgba) is read from the model once;
        only body pos/quat is read from data each call.
        """
        result = {}
        with self._physics_lock:
            for name in self._scene_body_names:
                body_id = self._model.body(name).id
                pos = self._data.xpos[body_id].copy().tolist()
                quat = self._data.xquat[body_id].copy().tolist()

                geoms = []
                for gid in self._body_geom_ids.get(body_id, ()):
                    geoms.append(
                        {
                            "type": int(self._model.geom_type[gid]),
                            "size": self._model.geom_size[gid].copy().tolist(),
                            "pos": self._model.geom_pos[gid].copy().tolist(),
                            "rgba": self._model.geom_rgba[gid].copy().tolist(),
                        }
                    )

                # Single-geom shortcut for backward compat
                geom_name = f"{name}_geom"
                try:
                    size = self._model.geom(geom_name).size.copy().tolist()
                except Exception:
                    size = []

                result[name] = {"pos": pos, "quat": quat, "size": size, "geoms": geoms}
        return {"ok": True, "objects": result}

    def get_scenes(self) -> dict:
        """List available scenes and the currently active one.

        Returns {"ok": True, "scenes": [...], "active": name|None}.
        """
        return {"ok": True, "scenes": list_scenes(), "active": self._active_scene}

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance physics by n_substeps. Call once per control tick."""
        with self._physics_lock:
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def get_arm_observation(self, side: str) -> dict[str, np.ndarray]:
        """Read joint positions for one arm. Returns {'joint_pos': 6D, 'gripper_pos': 1D}."""
        with self._physics_lock:
            joint_ids = self._joint_ids[side]
            qpos = self._data.qpos[joint_ids].copy()
        qpos_arm = qpos[:6]
        qpos_gripper = np.abs(qpos[6:]).mean(keepdims=True) / self.GRIPPER_QPOS_SCALE
        return {
            "joint_pos": qpos_arm,
            "gripper_pos": qpos_gripper,
        }

    def command_arm(self, side: str, cmd: dict) -> None:
        """Write commanded positions to MuJoCo ctrl. Expects cmd['pos'] = 7D (6 joint + 1 gripper)."""
        pos = np.asarray(cmd["pos"], dtype=np.float64)
        act_ids = self._actuator_ids[side]
        ctrl = np.empty(len(act_ids))
        ctrl[:6] = pos[:6]
        ctrl[6] = pos[6] * self.GRIPPER_CTRL_SCALE  # 0-1 → MuJoCo scale
        with self._physics_lock:
            self._data.ctrl[act_ids] = ctrl

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_rgb(self, camera_name: str) -> np.ndarray:
        """Render RGB image (H×W×3 uint8) for a named camera."""
        return self._render_executor.submit(self._render_rgb_impl, camera_name).result()

    def _render_rgb_impl(self, camera_name: str) -> np.ndarray:
        mj_cam_name = self._camera_names[camera_name]
        with self._physics_lock:
            if self._rgb_renderer is None:
                self._rgb_renderer = mujoco.Renderer(
                    self._model, self.CAMERA_HEIGHT, self.CAMERA_WIDTH
                )
                self._rgb_renderer.disable_depth_rendering()
                self._rgb_renderer.disable_segmentation_rendering()
            self._rgb_renderer.update_scene(self._data, camera=mj_cam_name)
        # render() only reads from the internal scene graph — no physics access needed
        return self._rgb_renderer.render().copy()

    def render_depth(self, camera_name: str) -> np.ndarray:
        """Render depth image (H×W float32) for a named camera."""
        return self._render_executor.submit(
            self._render_depth_impl, camera_name
        ).result()

    def _render_depth_impl(self, camera_name: str) -> np.ndarray:
        mj_cam_name = self._camera_names[camera_name]
        with self._physics_lock:
            if self._depth_renderer is None:
                self._depth_renderer = mujoco.Renderer(
                    self._model, self.CAMERA_HEIGHT, self.CAMERA_WIDTH
                )
                self._depth_renderer.enable_depth_rendering()
                self._depth_renderer.disable_segmentation_rendering()
            self._depth_renderer.update_scene(self._data, camera=mj_cam_name)
        # render() only reads from the internal scene graph — no physics access needed
        return self._depth_renderer.render().copy().astype(np.float32)

    def get_camera_intrinsics(self, camera_name: str) -> list[float]:
        """Compute [fx, fy, cx, cy] from MuJoCo camera model parameters."""
        import math

        mj_cam_name = self._camera_names[camera_name]
        cam_id = self._model.camera(mj_cam_name).id

        # cam_focal/cam_sensorsize were added in MuJoCo ≥3.2;
        # fall back to cam_fovy for older versions.
        if hasattr(self._model, "cam_focal"):
            focal = self._model.cam_focal[cam_id]
            sensor = self._model.cam_sensorsize[cam_id]
            fx = focal[0] / sensor[0] * self.CAMERA_WIDTH
            fy = focal[1] / sensor[1] * self.CAMERA_HEIGHT
        else:
            fovy = self._model.cam_fovy[cam_id]
            fy = self.CAMERA_HEIGHT / (2.0 * math.tan(math.radians(fovy) / 2.0))
            fx = fy  # square pixels assumed

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
        cam_id = self._model.camera(mj_cam_name).id
        with self._physics_lock:
            pos = self._data.cam_xpos[cam_id].copy()
            rot = self._data.cam_xmat[cam_id].copy().reshape(3, 3)
        # Convert MuJoCo camera frame to Pinocchio convention (x-left, y-up, z-forward)
        rot = rot @ np.diag([-1.0, 1.0, -1.0])
        return {"position": pos.tolist(), "rotation": rot.tolist()}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        if self._rgb_renderer is not None:
            self._rgb_renderer.close()
        if self._depth_renderer is not None:
            self._depth_renderer.close()
        self._render_executor.shutdown(wait=False)


class SimArmClient:
    """Drop-in replacement for _ArmClient backed by SimBackend."""

    def __init__(self, backend: SimBackend, side: str):
        self._backend = backend
        self._side = side

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._backend.get_arm_observation(self._side)

    def command_joint_state(self, cmd: dict) -> None:
        self._backend.command_arm(self._side, cmd)

    def get_joint_pos(self) -> np.ndarray:
        return self._backend.get_arm_observation(self._side)["joint_pos"]


class SimCameraClient:
    """Drop-in replacement for _CameraClient backed by SimBackend.

    Renders frames in a background thread and caches them, so callers
    (including the recorder) never block the physics/control loop.
    """

    _RENDER_FPS = 15  # background render rate (matches recorder)

    def __init__(self, backend: SimBackend, name: str):
        self._backend = backend
        self._name = name
        self._rgb: np.ndarray = np.zeros(
            (backend.CAMERA_HEIGHT, backend.CAMERA_WIDTH, 3),
            dtype=np.uint8,
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop,
            daemon=True,
            name=f"sim-cam-{name}",
        )
        self._thread.start()

    def _render_loop(self) -> None:
        import time as _time

        interval = 1.0 / self._RENDER_FPS
        while self._running:
            t0 = _time.monotonic()
            try:
                self._rgb = self._backend.render_rgb(self._name)
            except Exception:
                pass  # renderer not ready yet
            elapsed = _time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                _time.sleep(remaining)

    def get_rgb(self) -> np.ndarray:
        return self._rgb.copy()

    def get_depth(self) -> np.ndarray:
        return self._backend.render_depth(self._name)

    def get_intrinsics(self) -> list[float]:
        return self._backend.get_camera_intrinsics(self._name)

    def close(self) -> None:
        self._running = False
