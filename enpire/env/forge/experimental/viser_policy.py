# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple
import atexit
import queue
import threading
import time
from datetime import datetime

from enpire.env.forge.experimental._solve_ik_with_multiple_targets import solve_ik_with_multiple_targets
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R
import trimesh
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from enpire.env.forge.experimental._pyroki_compat import import_pyroki

pk = import_pyroki()

from enpire.env.forge.experimental._types import VLAStepData
from enpire.env.forge.experimental.robot_interface import RobotInterface
from enpire.env.forge.experimental.portal_policy import PortalPolicy


@dataclass
class PolicyAdapters:
    """Adapters for mapping between env <-> policy and URDF joint orders.

    Attributes:
        map_observation: Optional function to map environment observation to
            (images, proprio) used by the policy.
        map_action: Optional function to map policy-step action to environment action.
    """

    map_observation: Optional[Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]] = (
        None
    )
    map_action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


# convert to delta
def convert_to_relative_to_proprio(
    action: Dict[str, Any], proprio: Dict[str, Any]
) -> Dict[str, Any]:
    # """Convert action to delta action."""
    # delta_action = {}
    # for key, value in action.items():
    #     delta_a = action[key][:, 1:] - action[key][:, 0]
    #     delta_action[key] = delta_a
    # # apply delta to proprio
    # new_action = {}
    # for key, value in delta_action.items():
    #     proprio_key = key.replace("_action_", "_obs_")
    #     new_action[key] = value + proprio[proprio_key]
    #     new_action[key] = new_action[key][:, :10]
    # import ipdb; ipdb.set_trace()
    # return new_action

    # import ipdb; ipdb.set_trace()
    ## START CONVERT FROM DELTA ACTION TO ABSOLUTE ACTION (gripper is absolute)
    for key, value in action.items():
        if "gripper" not in key:
            action[key][:, 0] = value[:, 0] + proprio[key.replace("_action_", "_obs_")]
            for i in range(1, value.shape[1]):
                action[key][:, i] = action[key][:, i - 1] + action[key][:, i]
        else:
            # completely close the gripper if the action is less than 0.1
            action[key][action[key] < 0.1] = 0
    ## END CONVERT FROM DELTA ACTION TO ABSOLUTE ACTION

    # import ipdb; ipdb.set_trace()
    # adjust action horizon to 20 (pred 40)
    for key, value in action.items():
        # action[key] = value[:, :20] # good
        action[key] = value[:, :]  # good
        # action[key] = value[:, :25]
        # action[key] = value[:, :16]
        # action[key] = value[:, :30] # too open loop?
    return action


class ActionType(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class ViserPolicy:
    """Viser-driven policy wrapper usable for both sim and real environments.

    This class owns a Viser server, shows camera images, exposes a policy/IK toggle,
    and returns an action via a `step(image, proprio, task_description)` API compatible
    with existing policies.
    """

    grasp_site_offset_m = 0.1347

    def __init__(
        self,
        policy: RobotInterface | PortalPolicy,
        adapters: PolicyAdapters,
        urdf_path: Path,
        *,
        action_exec_horizon: int = 40,
        action_type: ActionType = ActionType.ABSOLUTE,
        ik_tcp_link_names: Tuple[str, str] = ("right_tcp", "left_tcp"),
        record_episode: bool = False,
        video_enabled: bool = True,
        video_fps: int = 30,
        video_realtime: bool = True,
        video_queue_size: int = 512,
    ) -> None:
        self.policy: RobotInterface | PortalPolicy = policy
        self.use_proprio = not isinstance(self.policy, RobotInterface)
        self.adapters = adapters or PolicyAdapters()
        self.server = viser.ViserServer()
        self.task_command: str = ""
        self.run_eval: bool = True
        self.step_once: bool = False
        self.goto_home: bool = False
        self.controller_mode: str = "viser-ik"
        self.execution_mode: str = "receding-horizon"
        self.action_type = action_type
        self.task: Task = Task.OTHER
        self.overlay: Optional[FrameOverlay] = None
        self.rs_mngr: RecordingStateManager | None = (
            RecordingStateManager() if record_episode else None
        )
        self.overlay_image_popup: Optional[Any] = None
        self._discard_recording: bool = False
        self._video_enabled = bool(video_enabled)
        self._video_fps = int(video_fps)
        self._video_realtime = bool(video_realtime)
        self._video_queue_size = int(video_queue_size)
        self._video_dir: Optional[Path] = None
        self._video_writer: Any | None = None
        self._video_path: Optional[Path] = None
        self._video_lock = threading.Lock()
        self._video_queue: queue.Queue[Tuple[np.ndarray, int]] | None = None
        self._video_thread: threading.Thread | None = None
        self._video_stop = threading.Event()
        self._video_drop_count = 0
        self._imageio = None
        self._last_video_ts: Optional[float] = None
        self._font = ImageFont.load_default()
        self._title_font = self._load_title_font()
        self._init_video_output()
        atexit.register(self.close)

        with self.server.gui.add_folder("Policy control"):
            start_btn = self.server.gui.add_button("Start")
            pause_btn = self.server.gui.add_button("Pause")
            home_btn = self.server.gui.add_button("Home")
            step_btn = self.server.gui.add_button("Step once")
            home_and_discard_bth = self.server.gui.add_button("Home and Discard")
            self.controller_select = self.server.gui.add_dropdown(
                label="Controller", options=["policy", "viser-ik"], initial_value="viser-ik"
            )
            self.exec_select = self.server.gui.add_dropdown(
                label="Execution",
                options=["temporal-ensemble", "receding-horizon"],
                initial_value=self.execution_mode,
            )
            self.cmd_input = self.server.gui.add_text(
                label="Task command", initial_value=self.task_command
            )
            self.task_select = self.server.gui.add_dropdown(
                label="Task", options=[task.value for task in Task], initial_value="OTHER"
            )

        @start_btn.on_click
        def _(_e: Any) -> None:
            self.run_eval = True
            if not self.rs_mngr:
                return
            self.rs_mngr.change_state("Start Recording")

        @home_btn.on_click
        def _(_e: Any) -> None:
            self.goto_home = True
            if not self.rs_mngr:
                return
            self.rs_mngr.change_state("Stop Recording")

        @pause_btn.on_click
        def _(_e: Any) -> None:
            self.run_eval = False
            if not self.rs_mngr:
                return
            self.rs_mngr.change_state("Stop Recording")

        @home_and_discard_bth.on_click
        def _(_e: Any) -> None:
            self.goto_home = True
            self._discard_recording = True
            if not self.rs_mngr:
                return
            self.rs_mngr.change_state("Discard Recording")

        @step_btn.on_click
        def _(_e: Any) -> None:
            self.step_once = True

        @self.task_select.on_update  # type: ignore[attr-defined]
        def _(_e: Any) -> None:
            self.task = Task(self.task_select.value)
            self._overlay_status.content = f"*Scanning episodes for {self.task.value}...*"
            self._overlay_progress.value = 0
            self._overlay_progress.visible = True

            def _on_progress(done: int, total: int) -> None:
                pct = int(done / total * 100) if total else 100
                self._overlay_progress.value = pct
                self._overlay_status.content = f"*Scanning: {done}/{total} folders...*"

            def _on_ready() -> None:
                self._overlay_progress.visible = False
                n = self.overlay.total_frames() if self.overlay else 0
                self._overlay_status.content = f"**Overlay ready** ({n} episodes)"

            self.overlay = FrameOverlay(self.task, on_progress=_on_progress, on_ready=_on_ready)
            if self.rs_mngr:
                self.rs_mngr.register_callback_on_episode_end(self.overlay.increment_frame_counter)

        @self.controller_select.on_update  # type: ignore[attr-defined]
        def _(_e: Any) -> None:
            new_mode = self.controller_select.value  # type: ignore[attr-defined]
            self.controller_mode = new_mode
            # Auto-pause only when switching into policy mode
            if new_mode == "policy":
                self.run_eval = False
                # Stage a preview inference on the next step() call
                self._needs_preview_inference = True
            # Snap IK targets to current EE pose when entering IK mode
            if new_mode == "viser-ik":
                # Ensure IK goes live immediately when switching back
                self.run_eval = True
                self._snap_ik_to_current()
                self._ik_snapped_once = True
            else:
                self._ik_snapped_once = False

        @self.exec_select.on_update  # type: ignore[attr-defined]
        def _(_e: Any) -> None:
            self.execution_mode = self.exec_select.value  # type: ignore[attr-defined]

        @self.cmd_input.on_update  # type: ignore[attr-defined]
        def _(_e: Any) -> None:
            self.task_command = self.cmd_input.value  # type: ignore[attr-defined]

        self._overlay_folder = self.server.gui.add_folder("Overlay Controls", visible=True)
        with self._overlay_folder:
            self._overlay_status = self.server.gui.add_markdown("*Overlay: select a task to load*")
            self._overlay_progress = self.server.gui.add_progress_bar(0, animated=True, visible=False)
            overlay_popup_btn = self.server.gui.add_button("Overlay Popup")

        @overlay_popup_btn.on_click
        def _(_e: Any) -> None:
            with self.server.gui.add_modal(title="Top View Overlay") as modal:
                self.overlay_image_popup = self.server.gui.add_image(
                    label="Overlay Image", image=np.zeros((240 * 2, 320 * 2, 3), dtype=np.uint8)
                )
                next_frame_btn = self.server.gui.add_button("Next Frame")
                prev_frame_btn = self.server.gui.add_button("Previous Frame")
                close_overlay_btn = self.server.gui.add_button("Close")
                frame_numb_txt = self.server.gui.add_text(label="Frame number", initial_value="0/0")
                frame_count_txt = self.server.gui.add_text(
                    label="Episodes Complete on this frame", initial_value="0"
                )

                def _update_overlay_info():
                    if self.overlay is None:
                        return
                    frame_numb_txt.value = (
                        f"{self.overlay.get_frame_idx() + 1}/{self.overlay.total_frames()}"
                    )
                    frame_count_txt.value = str(self.overlay.get_current_frame_counter())

                @close_overlay_btn.on_click
                def _(_e: Any) -> None:
                    modal.close()

                @next_frame_btn.on_click
                def _(_e: Any) -> None:
                    if not self.overlay or not self.overlay.ready:
                        return
                    _ = self.overlay.get_next_frame()
                    _update_overlay_info()

                @prev_frame_btn.on_click
                def _(_e: Any) -> None:
                    if self.overlay is None or not self.overlay.ready:
                        return
                    _ = self.overlay.get_prev_frame()
                    _update_overlay_info()

        # Image panes (optional, to avoid duplication if a host already provides them)
        self.gui_images: Dict[str, Any] = {}
        with self.server.gui.add_folder("Images"):
            self.gui_images["top"] = self.server.gui.add_image(
                label="top", image=np.zeros((240, 320, 3), dtype=np.uint8)
            )
            self.gui_images["left"] = self.server.gui.add_image(
                label="left", image=np.zeros((240, 320, 3), dtype=np.uint8)
            )
            self.gui_images["right"] = self.server.gui.add_image(
                label="right", image=np.zeros((240, 320, 3), dtype=np.uint8)
            )

        # Visualization toggles
        with self.server.gui.add_folder("Visualization"):
            self.ee_as_poses_cb = self.server.gui.add_checkbox("EE as poses (frames)", False)
            self.viz_temporal_ensemble_cb = self.server.gui.add_checkbox(
                "Visualize temporal ensemble (instead of model pred)", False
            )

        # URDF visualization and IK/FK robot
        self.ik_tcp_link_names = ik_tcp_link_names
        self.urdf_vis = ViserUrdf(self.server, urdf_or_path=urdf_path, load_meshes=True)
        self._urdf_joint_limits = dict(self.urdf_vis.get_actuated_joint_limits())  # type: ignore[arg-type]
        self.urdf_joint_names = list(self._urdf_joint_limits.keys())

        # Build IK robot regardless of server usage if possible
        urdf_model = URDF.load(str(urdf_path))
        self.ik_robot = pk.Robot.from_urdf(urdf_model)
        self._link_names: List[str] = list(self.ik_robot.links.names)  # type: ignore[attr-defined]
        self._ik_actuated_names: List[str] = list(self.ik_robot.joints.actuated_names)  # type: ignore[attr-defined]
        self._ik_idx_right: List[int] = [
            i for i, n in enumerate(self._ik_actuated_names) if n.startswith("right_joint")
        ][:6]
        self._ik_idx_left: List[int] = [
            i for i, n in enumerate(self._ik_actuated_names) if n.startswith("left_joint")
        ][:6]

        self.ik_left = self.server.scene.add_transform_controls(
            "/ik_target_left",
            scale=0.15,
            position=(0.3, 0.3, 0.6),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=(self.controller_mode == "viser-ik"),
        )
        self.ik_right = self.server.scene.add_transform_controls(
            "/ik_target_right",
            scale=0.15,
            position=(0.3, -0.3, 0.6),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=(self.controller_mode == "viser-ik"),
        )

        # Smoothing buffers for policy chunked outputs
        self.action_exec_horizon = int(action_exec_horizon)
        self.action_queue: Dict[str, Deque[Deque[np.ndarray]]] = {}
        self.action_keys: List[str] = []
        self.last_action_chunk: Optional[Dict[str, Any]] = None
        # Receding horizon state
        self._receding_chunk: Optional[Dict[str, Any]] = None
        self._receding_index: int = 0

        # Preview state: when switching into policy while paused, run one inference to visualize
        self._needs_preview_inference: bool = False

        # EE visualization handles
        self.ee_points_left: List[Any] = []
        self.ee_points_right: List[Any] = []
        self.ee_frames_left: List[Any] = []
        self.ee_frames_right: List[Any] = []

        # Cache tcp link indices if available
        self._tcp_link_idx: Dict[str, int] = {}
        for name in self.ik_tcp_link_names:
            if hasattr(self, "_link_names") and name in getattr(self, "_link_names", []):
                self._tcp_link_idx[name] = self._link_names.index(name)
        # Track last proprio and IK snap state
        self._last_proprio: Optional[Dict[str, Any]] = None
        self._ik_snapped_once: bool = False

    # ---------- Prediction overlays (URDF FK-based) ----------
    def _point_color(self, t_norm: float) -> Tuple[float, float, float, float]:
        r = float(t_norm)
        g = 0.2
        b = float(1.0 - t_norm)
        return (r, g, b, 1.0)

    def _ensure_ee_points(self, count: int) -> None:
        radius = 0.008
        while len(self.ee_points_left) < count:
            i = len(self.ee_points_left)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            h = self.server.scene.add_mesh_trimesh(
                f"/pred/left_pt_{i}", sphere, position=(0.0, 0.0, 0.0)
            )
            self.ee_points_left.append(h)
        while len(self.ee_points_right) < count:
            i = len(self.ee_points_right)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            h = self.server.scene.add_mesh_trimesh(
                f"/pred/right_pt_{i}", sphere, position=(0.0, 0.0, 0.0)
            )
            self.ee_points_right.append(h)

    def _ensure_ee_frames(self, count: int) -> None:
        while len(self.ee_frames_left) < count:
            i = len(self.ee_frames_left)
            t_norm = i / max(count - 1, 1)
            rgb = self._point_color(t_norm)[:3]
            rgb255 = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            h = self.server.scene.add_frame(
                f"/pred/left_pose_{i}",
                show_axes=True,
                axes_length=0.02,
                axes_radius=0.002,
                origin_radius=0.005,
                origin_color=rgb255,
                visible=False,
            )
            self.ee_frames_left.append(h)
        while len(self.ee_frames_right) < count:
            i = len(self.ee_frames_right)
            t_norm = i / max(count - 1, 1)
            rgb = self._point_color(t_norm)[:3]
            rgb255 = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            h = self.server.scene.add_frame(
                f"/pred/right_pose_{i}",
                show_axes=True,
                axes_length=0.02,
                axes_radius=0.002,
                origin_radius=0.005,
                origin_color=rgb255,
                visible=False,
            )
            self.ee_frames_right.append(h)

    def update_prediction(self, action_chunk: Dict[str, Any]) -> None:
        if self.ik_robot is None:
            return

        # Extract sequences
        def _get_seq(key_candidates: List[str]) -> Optional[np.ndarray]:
            for k in key_candidates:
                if k in action_chunk:
                    seq = np.asarray(action_chunk[k], dtype=float)
                    if seq.ndim == 3:
                        seq = seq[0]
                    return seq
            return None

        left_seq = _get_seq(["left_joint_pos", "joint_pos_action_left", "left_arm_joints"])  # (H,6)
        right_seq = _get_seq(
            ["right_joint_pos", "joint_pos_action_right", "right_arm_joints"]
        )  # (H,6)
        steps = max(
            (left_seq.shape[0] if left_seq is not None else 0),
            (right_seq.shape[0] if right_seq is not None else 0),
        )
        if steps <= 0:
            return
        idxs = np.linspace(0, steps - 1, num=int(steps), dtype=int)

        # Prepare outputs
        left_positions: List[np.ndarray] = []
        right_positions: List[np.ndarray] = []
        left_wxyzs: List[np.ndarray] = []
        right_wxyzs: List[np.ndarray] = []

        for s in idxs:
            # Build actuated cfg
            cfg = np.zeros((len(self._ik_actuated_names),), dtype=float)
            if right_seq is not None and len(self._ik_idx_right) >= 6:
                cfg[np.array(self._ik_idx_right[: right_seq.shape[-1]])] = right_seq[s]
            if left_seq is not None and len(self._ik_idx_left) >= 6:
                cfg[np.array(self._ik_idx_left[: left_seq.shape[-1]])] = left_seq[s]
            Ts_world_links = np.asarray(self.ik_robot.forward_kinematics(cfg[None, ...]))  # type: ignore[arg-type]  # (1, L, 7)
            for side, acc in (
                ("right", (right_positions, right_wxyzs)),
                ("left", (left_positions, left_wxyzs)),
            ):
                name = self.ik_tcp_link_names[0] if side == "right" else self.ik_tcp_link_names[1]
                link_idx = self._tcp_link_idx.get(name, None)
                if link_idx is None:
                    continue
                wxyz_xyz = Ts_world_links[0, link_idx]
                p_tcp = np.asarray(wxyz_xyz[4:7], dtype=float)
                q_tcp = np.asarray(wxyz_xyz[0:4], dtype=float)
                # Apply grasp-site offset along local +Z of TCP to align with MJCF grasp site
                R_tcp = R.from_quat(q_tcp, scalar_first=True).as_matrix()
                p_grasp = p_tcp + R_tcp[:, 2] * float(self.grasp_site_offset_m)
                acc[0].append(p_grasp)
                acc[1].append(q_tcp)

        npts = len(idxs)
        if self.ee_as_poses_cb is not None and bool(self.ee_as_poses_cb.value):  # type: ignore[attr-defined]
            # Frames mode
            self._ensure_ee_frames(npts)
            # Hide points
            for h in self.ee_points_left:
                h.visible = False
            for h in self.ee_points_right:
                h.visible = False
            # Update frames
            for i in range(npts):
                if i < len(left_positions):
                    p = left_positions[i]
                    q = (
                        left_wxyzs[i]
                        if i < len(left_wxyzs)
                        else np.array([1, 0, 0, 0], dtype=float)
                    )
                    hl = self.ee_frames_left[i]
                    hl.position = (float(p[0]), float(p[1]), float(p[2]))
                    hl.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    hl.visible = True
                if i < len(right_positions):
                    p = right_positions[i]
                    q = (
                        right_wxyzs[i]
                        if i < len(right_wxyzs)
                        else np.array([1, 0, 0, 0], dtype=float)
                    )
                    hr = self.ee_frames_right[i]
                    hr.position = (float(p[0]), float(p[1]), float(p[2]))
                    hr.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    hr.visible = True
            # Hide any extra frames beyond npts
            for i in range(npts, len(self.ee_frames_left)):
                self.ee_frames_left[i].visible = False
            for i in range(npts, len(self.ee_frames_right)):
                self.ee_frames_right[i].visible = False
        else:
            # Points mode
            self._ensure_ee_points(npts)
            # Hide frames
            for h in self.ee_frames_left:
                h.visible = False
            for h in self.ee_frames_right:
                h.visible = False
            # Update points
            for i in range(npts):
                if i < len(left_positions):
                    p = left_positions[i]
                    hl = self.ee_points_left[i]
                    hl.position = (float(p[0]), float(p[1]), float(p[2]))
                    hl.visible = True
                if i < len(right_positions):
                    p = right_positions[i]
                    hr = self.ee_points_right[i]
                    hr.position = (float(p[0]), float(p[1]), float(p[2]))
                    hr.visible = True
            # Hide extras
            for i in range(npts, len(self.ee_points_left)):
                self.ee_points_left[i].visible = False
            for i in range(npts, len(self.ee_points_right)):
                self.ee_points_right[i].visible = False

    def _prepare_vla_step_data(
        self, image: Dict[str, Image.Image], proprio: Dict[str, Any], task_description: str
    ) -> VLAStepData:
        reformatted_images = {k.replace("observation.images.", ""): v for k, v in image.items()}
        if not self.use_proprio:
            proprio = {}
        return VLAStepData(
            images=reformatted_images,
            states=proprio,
            actions={},
            text=task_description,
            embodiment=self.policy.embodiment_tag,
        )

    def _truncate_action_chunk_to_exec_horizon(
        self, action_chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Truncate action chunk to respect action_exec_horizon."""
        truncated_chunk = {}
        for key, value in action_chunk.items():
            seq = np.asarray(value)
            # Handle both (H, D) and (B, H, D) shapes
            if seq.ndim == 3:
                truncated_chunk[key] = seq[:, : self.action_exec_horizon, :]
            elif seq.ndim == 2:
                truncated_chunk[key] = seq[: self.action_exec_horizon, :]
            else:
                truncated_chunk[key] = value
        return truncated_chunk

    # --------------- Public API ---------------
    def step(
        self,
        *,
        image: Image.Image | List[Image.Image] | Dict[str, Image.Image],
        proprio: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Run one policy step or IK depending on UI toggle and return an action.

        Args:
            image: A single image, list of images, or dict of camera name -> PIL.Image.
            proprio: Policy-specific proprio dict.
            task_description: Optional language command.

        Returns:
            Action dict with keys: "left_joint_pos", "left_gripper_pos",
            "right_joint_pos", "right_gripper_pos". Shapes are (6,) and (1,).
        """
        if self.goto_home:
            self.goto_home = False
            # resetting all the variables
            self._last_proprio = None
            self._receding_chunk = None
            self._receding_index = 0
            self.last_action_chunk = None
            self.action_queue = {}
            self._needs_preview_inference = True
            self.run_eval = False
            self.step_once = False
            self._ik_snapped_once = False
            if not self.rs_mngr:
                return "Homing"
            _ = self.rs_mngr.get_current_state_and_change_next_state()
            if self._discard_recording:
                self._discard_recording = False
                return "Homing and Discard Recording"
            return "Homing"

        # Check for recording state events
        # Returns event string ("Start Recording", "Stop Recording", "Discard Recording")
        # and automatically transitions to next state (Recording → Idle, etc.)
        # Returns None if in a stable state (Idle or Recording)
        if self.rs_mngr:
            recording_event = self.rs_mngr.get_current_state_and_change_next_state()
            if recording_event is not None:
                return recording_event

        # Update images and URDF viz first
        self.update_images(image)
        self._update_urdf_cfg_from_proprio(proprio)
        # Keep IK controls visible to allow dragging; use controller mode only to choose action path
        self.ik_left.visible = True
        self.ik_right.visible = True

        if (not self.run_eval) and (not self.step_once):
            self._snap_ik_to_current()
            # If we owe a preview, run one inference to update overlays and stage first action, then keep paused
            if self.controller_mode == "policy" and self._needs_preview_inference:
                td = self.task_command
                print("task command: ", td)
                vla_step_data = self._prepare_vla_step_data(image, proprio, td)
                action_chunk = self.policy.step(vla_step_data=vla_step_data)[0]

                if self.action_type == ActionType.RELATIVE:
                    action_chunk = convert_to_relative_to_proprio(action_chunk, proprio)
                action_chunk = self._truncate_action_chunk_to_exec_horizon(action_chunk)
                self.last_action_chunk = action_chunk
                self._update_prediction_with_toggle(action_chunk)
                # Stage the chunk for receding-horizon to start from t=0 on first Step
                if self.execution_mode == "receding-horizon":
                    self._receding_chunk = action_chunk
                    self._receding_index = 0
                else:
                    # For temporal ensemble: warm buffers so the next Step executes visualized action
                    self._append_chunk_without_advance(action_chunk)
                # Clear preview request and stay paused
                self._needs_preview_inference = False
                # While paused, continue holding current pose (do not actuate preview)
                if self._last_proprio is None:
                    self._last_proprio = proprio
                    self.last_control_action = self._hold_action_from_proprio(self._last_proprio)
                return self.last_control_action
            # Normal paused behavior: hold last action or construct from proprio on first pause
            if self._last_proprio is None:
                self._last_proprio = proprio
                self.last_control_action = self._hold_action_from_proprio(self._last_proprio)
                return self.last_control_action
            else:
                return self.last_control_action

        self._last_proprio = proprio
        # Consume step_once if it was a single-step request
        if self.step_once:
            self.step_once = False

        if self.controller_mode == "viser-ik" and self.ik_robot is not None:
            if not self._ik_snapped_once:
                self._snap_ik_to_current()
                self._ik_snapped_once = True
            self.last_control_action = self._ik_action_from_controls(proprio)
            return self.last_control_action

        # Delegate to provided policy with two execution modes
        if self.execution_mode == "receding-horizon":
            # If there is a staged chunk with actions remaining, execute next visualized action
            if self._receding_chunk is not None and self._has_more_in_chunk(
                self._receding_chunk, self._receding_index
            ):
                action = self._chunk_index_to_action(self._receding_chunk, self._receding_index)
                self._receding_index += 1
                if not self._has_more_in_chunk(self._receding_chunk, self._receding_index):
                    self._receding_chunk = None
                    self._receding_index = 0
                self.last_control_action = action
                return action
            # No staged actions: run prediction to update visualization only; stage for next step
            td = self.task_command
            vla_step_data = self._prepare_vla_step_data(image, proprio, td)
            action_chunk = self.policy.step(vla_step_data=vla_step_data)[0]
            if self.action_type == ActionType.RELATIVE:
                action_chunk = convert_to_relative_to_proprio(action_chunk, proprio)
            action_chunk = self._truncate_action_chunk_to_exec_horizon(action_chunk)
            self.last_action_chunk = action_chunk
            self._update_prediction_with_toggle(action_chunk)
            self._receding_chunk = action_chunk
            self._receding_index = 0
            self.last_control_action = self._hold_action_from_proprio(self._last_proprio)
            return self.last_control_action
        else:
            # Temporal ensemble with safety
            has_history = (
                any(len(q) > 0 for q in self.action_queue.values()) if self.action_queue else False
            )
            if not has_history:
                # No history: predict to update viz/buffers, do not execute
                td = self.task_command
                vla_step_data = self._prepare_vla_step_data(image, proprio, td)
                action_chunk = self.policy.step(vla_step_data=vla_step_data)[0]
                if self.action_type == ActionType.RELATIVE:
                    action_chunk = convert_to_relative_to_proprio(action_chunk, proprio)
                action_chunk = self._truncate_action_chunk_to_exec_horizon(action_chunk)
                self.last_action_chunk = action_chunk
                self._update_prediction_with_toggle(action_chunk)
                self._append_chunk_without_advance(action_chunk)
                self.last_control_action = self._hold_action_from_proprio(self._last_proprio)
                return self.last_control_action
            # History present: execute ensembled action from history, then predict for next time
            action_from_history = self._consume_temporal_ensemble()
            td = self.task_command
            vla_step_data = self._prepare_vla_step_data(image, proprio, td)
            action_chunk = self.policy.step(vla_step_data=vla_step_data)[0]
            if self.action_type == ActionType.RELATIVE:
                action_chunk = convert_to_relative_to_proprio(action_chunk, proprio)
            action_chunk = self._truncate_action_chunk_to_exec_horizon(action_chunk)
            self.last_action_chunk = action_chunk
            self._update_prediction_with_toggle(action_chunk)
            self._append_chunk_without_advance(action_chunk)
            self.last_control_action = action_from_history
            return self.last_control_action

    def control_step_from_observation(
        self,
        *,
        observation: Dict[str, Any],
        default_task: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Convenience: map observation -> (images, proprio) and produce an action.

        If `map_observation` is not provided, this will raise.
        """
        if self.adapters.map_observation is None:
            raise RuntimeError(
                "PolicyAdapters.map_observation must be provided for control_step_from_observation()."
            )
        images, proprio = self.adapters.map_observation(observation)
        if default_task is not None and not self.task_command:
            self.task_command = default_task
        return self.step(image=images, proprio=proprio)

    def get_overlay_frame(self):
        if self.overlay and self.overlay.ready:
            return self.overlay.get_frame()
        return None

    def update_images(
        self, images: Image.Image | List[Image.Image] | Dict[str, Image.Image]
    ) -> None:
        """Update GUI image panes from various input formats."""
        if not self.gui_images:
            return

        overlay_frame = self.get_overlay_frame()

        def to_uint8(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                mx = float(np.max(a)) if a.size > 0 else 0.0
                mi = float(np.min(a)) if a.size > 0 else 0.0
                if mx <= 1.0 and mi >= 0.0:
                    a = (a * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    a = np.clip(a, 0, 255).astype(np.uint8)
                return a
            return arr

        if isinstance(images, dict):
            img_map = images
        elif isinstance(images, list):
            keys = ["top", "left", "right"]
            img_map = {keys[i]: images[i] for i in range(min(len(images), 3))}
        else:
            img_map = {"top": images}
        normalized_images: Dict[str, np.ndarray] = {}
        normalized_labels: Dict[str, str] = {}
        for k, v in img_map.items():
            # Prefer explicit camera substrings if present
            if isinstance(k, str):
                if "top" in k:
                    key = "top"
                elif "left" in k:
                    key = "left"
                elif "right" in k:
                    key = "right"
                else:
                    key = None
            else:
                key = None
            if key is None:
                continue
            if isinstance(v, Image.Image):
                arr = np.array(v)
            else:
                arr = np.asarray(v)
            if overlay_frame is not None and self.overlay_image_popup is not None and key == "top":
                resized_arr = cv2.resize(arr, (320 * 2, 240 * 2))
                overlay_frame = cv2.resize(overlay_frame, (320 * 2, 240 * 2))
                self.overlay_image_popup.image = to_uint8(
                    cv2.addWeighted(resized_arr, 0.5, overlay_frame, 0.5, 0)
                )

            arr_uint8 = to_uint8(arr)
            self.gui_images[key].image = arr_uint8
            normalized_images[key] = arr_uint8
            normalized_labels[key] = str(k) if isinstance(k, str) else key
        self._maybe_write_video_frame(normalized_images, normalized_labels)

    def _init_video_output(self) -> None:
        if not self._video_enabled:
            return
        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError:
            print("[ViserPolicy] imageio missing; video recording disabled.", flush=True)
            self._video_enabled = False
            return
        self._imageio = imageio
        repo_root = Path(__file__).resolve().parents[1]
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._video_dir = repo_root / "log" / "deploy" / timestamp
        self._video_dir.mkdir(parents=True, exist_ok=True)
        print(f"\033[34m[ViserPolicy] Video output dir: {self._video_dir}\033[0m", flush=True)
        self._video_queue = queue.Queue(maxsize=self._video_queue_size)
        self._video_thread = threading.Thread(
            target=self._video_worker, name="viser-video-writer", daemon=True
        )
        self._video_thread.start()

    def _maybe_write_video_frame(
        self, images: Dict[str, np.ndarray], labels: Dict[str, str]
    ) -> None:
        if (
            not self._video_enabled
            or self._video_dir is None
            or self._imageio is None
            or self._video_queue is None
        ):
            return
        frame = self._compose_video_frame(images, labels)
        if frame is None:
            return
        repeats = 1
        if self._video_realtime:
            now = time.time()
            if self._last_video_ts is not None:
                dt = max(0.0, now - self._last_video_ts)
                repeats = max(1, int(round(dt * self._video_fps)))
            self._last_video_ts = now
        try:
            self._video_queue.put_nowait((frame, repeats))
        except queue.Full:
            self._video_drop_count += 1
            if self._video_drop_count % 100 == 1:
                print(
                    f"[ViserPolicy] Video queue full; dropping frames (dropped={self._video_drop_count}).",
                    flush=True,
                )

    def _compose_video_frame(
        self, images: Dict[str, np.ndarray], labels: Dict[str, str]
    ) -> Optional[np.ndarray]:
        order = ["top", "left", "right"]
        ordered: List[Tuple[str, np.ndarray]] = []
        for key in order:
            if key in images:
                ordered.append((key, images[key]))
        if not ordered:
            return None
        target_h = ordered[0][1].shape[0]
        resized: List[Tuple[str, np.ndarray]] = []
        for key, image_np in ordered:
            if image_np.shape[0] != target_h:
                scale = target_h / float(image_np.shape[0])
                target_w = int(round(image_np.shape[1] * scale))
                image_np = cv2.resize(image_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append((key, image_np))
        total_w = sum(img.shape[1] for _, img in resized)
        font_size = getattr(self._title_font, "size", 36)
        title_h = max(36, int(font_size) + 12)
        canvas = Image.new("RGB", (total_w, target_h + title_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        title = self.task_command or "Task command"
        left, top, right, bottom = draw.textbbox((0, 0), title, font=self._title_font)
        title_w, title_h_text = right - left, bottom - top
        draw.text(
            ((canvas.width - title_w) // 2, (title_h - title_h_text) // 2),
            title,
            fill=(255, 255, 255),
            font=self._title_font,
        )
        x0 = 0
        for key, image_np in resized:
            canvas.paste(Image.fromarray(image_np), (x0, title_h))
            label = labels.get(key, key)
            left, top, right, bottom = draw.textbbox((0, 0), label, font=self._font)
            label_w, label_h = right - left, bottom - top
            label_x = x0 + 6
            label_y = title_h + 6
            draw.rectangle(
                (label_x - 2, label_y - 2, label_x + label_w + 2, label_y + label_h + 2),
                fill=(0, 0, 0),
            )
            draw.text((label_x, label_y), label, fill=(255, 255, 255), font=self._font)
            x0 += image_np.shape[1]
        return np.asarray(canvas)

    def _load_title_font(self) -> ImageFont.ImageFont:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(path, size=36)
            except Exception:
                continue
        return ImageFont.load_default()

    def _video_worker(self) -> None:
        if self._video_dir is None or self._imageio is None or self._video_queue is None:
            return
        while not self._video_stop.is_set() or not self._video_queue.empty():
            try:
                frame, repeats = self._video_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._video_lock:
                if self._video_writer is None:
                    self._video_path = self._video_dir / "viser_views.mp4"
                    self._video_writer = self._imageio.get_writer(
                        str(self._video_path), fps=self._video_fps
                    )
                    print(f"\033[34m[ViserPolicy] Saving video stream to {self._video_path}\033[0m", flush=True)
                for _ in range(repeats):
                    self._video_writer.append_data(frame)
            self._video_queue.task_done()

    def close(self) -> None:
        self._shutdown_viser_server()
        if self._video_thread is not None:
            self._video_stop.set()
            self._video_thread.join(timeout=2.0)
        if self._video_writer is None:
            return
        with self._video_lock:
            try:
                self._video_writer.close()
            except Exception:
                print("[ViserPolicy] Failed to close video writer", flush=True)
            self._video_writer = None

    def _shutdown_viser_server(self) -> None:
        server = getattr(self, "server", None)
        if server is None:
            return
        for attr in ("shutdown", "close", "stop"):
            fn = getattr(server, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    print(f"[ViserPolicy] Failed to {attr} server: {exc}", flush=True)
                break
        nested = getattr(server, "server", None) or getattr(server, "_server", None)
        if nested is not None:
            for attr in ("shutdown", "close", "stop"):
                fn = getattr(nested, attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        print(f"[ViserPolicy] Failed to {attr} server: {exc}", flush=True)
                    break
        time.sleep(0.1)

    # --------------- Internals ---------------
    def _hold_action_from_proprio(self, proprio: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        # Fallback: attempt common keys in provided proprio
        try_keys = [
            ("left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"),
            (
                "joint_pos_obs_left",
                "gripper_pos_obs_left",
                "joint_pos_obs_right",
                "gripper_pos_obs_right",
            ),
        ]
        for lq, lg, rq, rg in try_keys:
            if lq in proprio and rq in proprio:
                return {
                    "left_joint_pos": np.asarray(proprio[lq], dtype=np.float32).reshape(-1)[:6],
                    "left_gripper_pos": np.asarray(
                        proprio.get(lg, np.zeros(1, np.float32)), dtype=np.float32
                    ).reshape(-1)[:1],
                    "right_joint_pos": np.asarray(proprio[rq], dtype=np.float32).reshape(-1)[:6],
                    "right_gripper_pos": np.asarray(
                        proprio.get(rg, np.zeros(1, np.float32)), dtype=np.float32
                    ).reshape(-1)[:1],
                }
        return None

    def _snap_ik_to_current(self) -> None:
        # Use URDF FK with last proprio to put IK handles at current EE pose
        if self._last_proprio is None or self.ik_robot is None:
            return
        # Build actuated cfg in PyRoki actuated joint order
        if not hasattr(self, "_ik_actuated_names") or not self._ik_actuated_names:
            return
        cfg = np.zeros((len(self._ik_actuated_names),), dtype=float)
        hold = self._hold_action_from_proprio(self._last_proprio)
        if hold is None:
            return
        name_to_val = {f"right_joint{i+1}": float(hold["right_joint_pos"][i]) for i in range(6)}
        name_to_val.update({f"left_joint{i+1}": float(hold["left_joint_pos"][i]) for i in range(6)})
        for i, n in enumerate(self._ik_actuated_names):
            cfg[i] = name_to_val.get(n)
        Ts = np.asarray(self.ik_robot.forward_kinematics(cfg[None, ...]))  # type: ignore[arg-type]  # (1, L, 7)
        # Right then left
        for name, handle in zip(self.ik_tcp_link_names, (self.ik_right, self.ik_left)):
            idx = self._tcp_link_idx.get(name, None)
            if idx is None:
                continue
            wxyz_xyz = Ts[0, idx]
            p_tcp = np.asarray(wxyz_xyz[4:7], dtype=float)
            q_tcp = np.asarray(wxyz_xyz[0:4], dtype=float)
            # Place IK handles at grasp site pose for alignment with MJCF
            R_tcp = R.from_quat(q_tcp, scalar_first=True).as_matrix()
            p = p_tcp + R_tcp[:, 2] * float(self.grasp_site_offset_m)
            q = q_tcp
            handle.position = (float(p[0]), float(p[1]), float(p[2]))
            handle.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _update_urdf_cfg_from_proprio(self, proprio: Dict[str, Any]) -> None:
        if self.urdf_vis is None:
            return
        # Try to assemble full actuated joint vector in the URDF's name order
        q_map = self._hold_action_from_proprio(proprio)
        if q_map is None:
            return
        joint_names = self.urdf_joint_names or [
            *[f"right_joint{i}" for i in range(1, 7)],
            *[f"left_joint{i}" for i in range(1, 7)],
        ]
        name_to_val: Dict[str, float] = {}
        for i in range(6):
            name_to_val[f"left_joint{i+1}"] = float(q_map["left_joint_pos"][i])
            name_to_val[f"right_joint{i+1}"] = float(q_map["right_joint_pos"][i])
        for side in ("left", "right"):
            gripper_open = float(
                np.asarray(q_map.get(f"{side}_gripper_pos", np.zeros(1)), dtype=float).reshape(-1)[0]
            )
            for finger_side in ("left", "right"):
                joint_name = f"{side}_{finger_side}_finger_joint"
                limits = self._urdf_joint_limits.get(joint_name)
                if limits is None:
                    continue
                limit_arr = np.asarray(limits, dtype=float).reshape(-1)
                if limit_arr.size < 2:
                    continue
                open_extent = float(limit_arr[:2][np.argmax(np.abs(limit_arr[:2]))])
                name_to_val[joint_name] = float(np.clip(gripper_open, 0.0, 1.0) * open_extent)
        cfg = np.array([name_to_val.get(n, 0.0) for n in joint_names], dtype=float)
        try:
            self.urdf_vis.update_cfg(cfg)
        except Exception:
            pass

    def _ik_action_from_controls(self, proprio: Dict[str, Any]) -> Dict[str, np.ndarray]:
        assert self.ik_robot is not None
        left_p = np.asarray(self.ik_left.position, dtype=float)
        left_q = np.asarray(self.ik_left.wxyz, dtype=float)
        right_p = np.asarray(self.ik_right.position, dtype=float)
        right_q = np.asarray(self.ik_right.wxyz, dtype=float)
        Rl = R.from_quat(left_q, scalar_first=True).as_matrix()
        Rr = R.from_quat(right_q, scalar_first=True).as_matrix()
        left_tcp_p = left_p - Rl[:, 2] * self.grasp_site_offset_m
        right_tcp_p = right_p - Rr[:, 2] * self.grasp_site_offset_m
        # Solve bimanual IK in order (right, left)

        q_sol = solve_ik_with_multiple_targets(
            robot=self.ik_robot,
            target_link_names=list(self.ik_tcp_link_names),
            target_positions=np.array([right_tcp_p, left_tcp_p], dtype=float),
            target_wxyzs=np.array([right_q, left_q], dtype=float),
        )
        q_sol = np.asarray(q_sol, dtype=np.float32)
        # Map by name ordering if available; otherwise assume [right1..6, left1..6]
        joint_names = (
            self._ik_actuated_names
            if hasattr(self, "_ik_actuated_names") and self._ik_actuated_names
            else self.urdf_joint_names
        ) or [
            *[f"right_joint{i}" for i in range(1, 7)],
            *[f"left_joint{i}" for i in range(1, 7)],
        ]
        name_to_idx = {n: i for i, n in enumerate(joint_names)}

        def pick(names: Iterable[str]) -> np.ndarray:
            idxs = [name_to_idx[n] for n in names if n in name_to_idx]
            return q_sol[idxs] if len(idxs) > 0 else np.zeros(6, dtype=np.float32)

        q_right = pick([f"right_joint{i}" for i in range(1, 7)])
        q_left = pick([f"left_joint{i}" for i in range(1, 7)])
        hold = self._hold_action_from_proprio(proprio) or {}
        return {
            "left_joint_pos": q_left.astype(np.float32),
            "left_gripper_pos": np.asarray(
                hold.get("left_gripper_pos", np.zeros(1, np.float32)), dtype=np.float32
            ),
            "right_joint_pos": q_right.astype(np.float32),
            "right_gripper_pos": np.asarray(
                hold.get("right_gripper_pos", np.zeros(1, np.float32)), dtype=np.float32
            ),
        }

    def _ensure_action_buffers(self, action_chunk: Dict[str, Any]) -> None:
        if not self.action_keys:
            # Prefer policy-provided grouping if available
            if hasattr(self.policy, "action_joint_groups"):
                self.action_keys = list(getattr(self.policy, "action_joint_groups"))
            else:
                # Use detected keys from first chunk
                self.action_keys = [k for k in action_chunk.keys() if k.endswith("joint_pos")]
        if not self.action_queue:
            # Action chunks are already truncated to action_exec_horizon before reaching here
            self.action_queue = {
                k: deque([], maxlen=self.action_exec_horizon) for k in self.action_keys
            }

    def _chunk_to_step_action(self, action_chunk: Dict[str, Any]) -> Dict[str, np.ndarray]:
        self._ensure_action_buffers(action_chunk)
        step_action: Dict[str, np.ndarray] = {}
        # Smooth each joint group separately
        for key in self.action_keys:
            seq = np.asarray(action_chunk[key], dtype=np.float32)
            if seq.ndim == 3:
                seq = seq[0]
            assert seq.ndim == 2, f"Expected (H, D) for {key}, got {seq.shape}"
            action_dim = seq.shape[-1]
            new_actions = deque(seq[: self.action_queue[key].maxlen])
            self.action_queue[key].append(new_actions)
            actions_current_timestep = np.empty(
                (len(self.action_queue[key]), action_dim), dtype=np.float32
            )
            for i, q in enumerate(self.action_queue[key]):
                actions_current_timestep[i] = q.popleft()
            k = 0.05
            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0], dtype=np.float32))
            exp_weights = exp_weights / exp_weights.sum()
            step_action[key] = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
        # Map to env action keys
        assert self.adapters.map_action is not None, "PolicyAdapters.map_action must be provided."
        return self.adapters.map_action(step_action)

    def _chunk_index_to_action(self, action_chunk: Dict[str, Any], t: int) -> Dict[str, np.ndarray]:
        # Extract action at index t from a chunk (H, D) or (1, H, D)
        def pick_row(arr: Any) -> np.ndarray:
            seq = np.asarray(arr)
            if seq.ndim == 3:
                seq = seq[0]
            idx = min(t, max(seq.shape[0] - 1, 0))
            val = seq[idx]
            return np.asarray(val, dtype=np.float32).reshape(-1)

        # Build a step dict using whatever keys the policy produced
        step_raw: Dict[str, np.ndarray] = {}
        for k, v in action_chunk.items():
            step_raw[k] = pick_row(v)
        assert self.adapters.map_action is not None, "PolicyAdapters.map_action must be provided."
        return self.adapters.map_action(step_raw)

    def _has_more_in_chunk(self, action_chunk: Dict[str, Any], t: int) -> bool:
        # Determine horizon as the maximum time dimension across any array-like entries
        horizon = 0
        for v in action_chunk.values():
            seq = np.asarray(v)
            if seq.ndim == 3:
                seq = seq[0]
            if seq.ndim >= 1:
                horizon = max(horizon, int(seq.shape[0]))
        return t < horizon

    # ---------- Helpers for safe step-once temporal ensemble ----------
    def _update_prediction_with_toggle(self, action_chunk: Dict[str, Any]) -> None:
        """Update overlays using either the raw model chunk or the temporal-ensembled sequence.

        Controlled via GUI checkbox; temporal ensemble visualization only applies when
        execution mode is "temporal-ensemble".
        """
        use_ensemble = (
            self.execution_mode == "temporal-ensemble"
            and getattr(self, "viz_temporal_ensemble_cb", None) is not None
            and bool(self.viz_temporal_ensemble_cb.value)  # type: ignore[attr-defined]
        )
        if use_ensemble:
            ensembled_chunk = self._compute_temporal_ensembled_chunk_like(action_chunk)
            if ensembled_chunk is not None:
                self.update_prediction(ensembled_chunk)
                return
        self.update_prediction(action_chunk)

    def _compute_temporal_ensembled_chunk_like(
        self, fallback_chunk: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Compute a chunk-shaped dict with (H,D) sequences from temporal buffers, without mutation.

        Returns None if no usable history exists; caller should fall back to `fallback_chunk`.
        """
        if not self.action_queue:
            return None
        keys_to_make: List[str] = []
        # Prefer explicit env keys used by update_prediction()
        for k in ("left_joint_pos", "right_joint_pos"):
            if k in self.action_queue or k in fallback_chunk:
                keys_to_make.append(k)
        if not keys_to_make:
            # Try detect any joint_pos keys
            keys_to_make = [k for k in self.action_queue.keys() if k.endswith("joint_pos")]
            if not keys_to_make:
                return None
        result: Dict[str, Any] = {}
        k_weight = 0.05
        for key in keys_to_make:
            if key not in self.action_queue:
                # No history for this key; try to reuse fallback chunk sequence
                if key in fallback_chunk:
                    result[key] = np.asarray(fallback_chunk[key])
                continue
            window_deques = list(self.action_queue[key])
            if len(window_deques) == 0:
                continue
            max_len = max(len(w) for w in window_deques)
            # Determine action dimension from first non-empty window
            first_non_empty = next((w for w in window_deques if len(w) > 0), None)
            if first_non_empty is None:
                continue
            action_dim = int(np.asarray(first_non_empty[0]).shape[-1])
            seq = np.empty((max_len, action_dim), dtype=np.float32)
            exp_weights = np.exp(k_weight * np.arange(len(window_deques), dtype=np.float32))
            exp_weights = exp_weights / exp_weights.sum()
            for t in range(max_len):
                stacked = np.empty((len(window_deques), action_dim), dtype=np.float32)
                for i, w in enumerate(window_deques):
                    if t < len(w):
                        stacked[i] = np.asarray(w[t], dtype=np.float32)
                    else:
                        stacked[i] = np.zeros(action_dim, dtype=np.float32)
                seq[t] = (stacked * exp_weights[:, None]).sum(axis=0)
            result[key] = seq
        return result if result else None

    def _append_chunk_without_advance(self, action_chunk: Dict[str, Any]) -> None:
        """Append a new chunk into temporal buffers without consuming from them.

        This warms the buffer so that the next call can ensemble over history.
        """
        self._ensure_action_buffers(action_chunk)
        for key in self.action_keys:
            seq = np.asarray(action_chunk[key], dtype=np.float32)
            if seq.ndim == 3:
                seq = seq[0]
            assert seq.ndim == 2, f"Expected (H, D) for {key}, got {seq.shape}"
            new_actions = deque(seq[: self.action_queue[key].maxlen])
            self.action_queue[key].append(new_actions)

    def _consume_temporal_ensemble(self) -> Dict[str, np.ndarray]:
        """Consume one step from existing temporal buffers and return ensembled action.

        This pops from the left of each inner deque (one time step), applies exponential
        weights across the buffer-of-buffers, and returns the mapped action.
        """
        assert self.action_queue, "No temporal history available."
        step_action: Dict[str, np.ndarray] = {}
        k = 0.05
        for key, qdeque in self.action_queue.items():
            if len(qdeque) == 0:
                continue
            # Build matrix [num_windows, action_dim] by consuming one step from each window
            first_q = next((qq for qq in qdeque if len(qq) > 0), None)
            if first_q is None:
                continue
            action_dim = int(np.asarray(first_q[0]).shape[-1])
            actions_current_timestep = np.empty((len(qdeque), action_dim), dtype=np.float32)
            for i, q in enumerate(qdeque):
                if len(q) > 0:
                    actions_current_timestep[i] = np.asarray(q.popleft(), dtype=np.float32)
                else:
                    actions_current_timestep[i] = np.zeros(action_dim, dtype=np.float32)
            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0], dtype=np.float32))
            exp_weights = exp_weights / exp_weights.sum()
            step_action[key] = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
        assert self.adapters.map_action is not None, "PolicyAdapters.map_action must be provided."
        return self.adapters.map_action(step_action)
