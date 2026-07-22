# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viser-based 3D visualizer for the CAP agent.

Shows the YAM robot station with live joint positions and detection markers.
Runs a Viser server on a separate port, embeddable as an iframe in the UI.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import viser
from viser.extras import ViserUrdf

from enpire.env.forge.cap.config import VISER_PORT

# Path to the station URDF
from enpire.env.forge.robot.models.station.paths import get_station_urdf

_URDF_PATH = get_station_urdf()


class CapVisualizer:
    """3D visualization of the robot and detected objects."""

    def __init__(self, port: int = VISER_PORT):
        self._server = viser.ViserServer(port=port)
        self._server.gui.configure_theme(dark_mode=True)
        self._port = port

        # Load robot URDF
        self._urdf = None
        self._joint_names: list[str] = []
        urdf_path = _URDF_PATH
        if not urdf_path.exists():
            print(f"[Visualizer] WARNING: URDF not found at {urdf_path}")
        else:
            try:
                import yourdfpy

                urdf_model = yourdfpy.URDF.load(
                    str(urdf_path),
                    build_collision_scene_graph=False,
                    load_collision_meshes=False,
                )
                self._urdf = ViserUrdf(
                    self._server, urdf_or_path=urdf_model, load_meshes=True
                )
                self._joint_names = list(self._urdf.get_actuated_joint_limits().keys())
                print(f"[Visualizer] URDF loaded: {len(self._joint_names)} joints")
            except Exception as e:
                print(f"[Visualizer] WARNING: Failed to load URDF: {e}")
                print("[Visualizer] 3D robot model will not be shown")

        # Detection marker handles (cleared and re-created on each update)
        self._det_markers: list[Any] = []
        self._det_labels: list[Any] = []

        # Prediction sphere handles for trajectory visualization
        self._pred_left: list[Any] = []
        self._pred_right: list[Any] = []

        # Kinematics for FK computation on prediction chunks
        from enpire.env.forge.robot.yam.kinematics import YamKinematics

        self._kin = YamKinematics()

        # Scene object handles (keyed by body name)
        self._scene_handles: dict[str, list[Any]] = {}

        # Focus mode: hide all debug overlays, show only robot + scene objects
        self._focus_mode = self._server.gui.add_checkbox("Focus", initial_value=True)
        self._debug_gui_folders: list[Any] = []

        @self._focus_mode.on_update
        def _on_toggle_focus(_) -> None:
            self._apply_focus_mode()

        # Safety zone visualization handles
        self._safety_handles: list[Any] = []
        self._safety_zone_config: dict | None = None  # cached to avoid re-rendering
        self._show_safety_zone = self._server.gui.add_checkbox(
            "Show Safety Zone",
            initial_value=False,
        )

        @self._show_safety_zone.on_update
        def _on_toggle_safety(_) -> None:
            if self._show_safety_zone.value:
                if self._safety_zone_config is not None:
                    cfg = self._safety_zone_config
                    self._safety_zone_config = None  # force re-render
                    self.update_safety_zone(cfg)
            else:
                self._clear_safety_handles(reset_config=False)

        # Pose gizmo for debugging grasp targets
        self._pose_gizmo = None
        self._gizmo_texts: dict[str, Any] = {}
        self._setup_pose_gizmo()

        # EE coordinate frames
        self._left_ee_frame = self._server.scene.add_frame(
            "/ee/left",
            axes_length=0.08,
            axes_radius=0.004,
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            visible=False,
        )
        self._right_ee_frame = self._server.scene.add_frame(
            "/ee/right",
            axes_length=0.08,
            axes_radius=0.004,
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            visible=False,
        )

        # Right EE gizmo
        self._ee_gizmo = None
        self._ee_gizmo_texts: dict[str, Any] = {}
        self._setup_ee_gizmo()

        # Grasp pose visualization
        self._grasp_handles: list[Any] = []
        self._sample_grasp_fn: Any = None
        self._setup_grasp_pose_ui()

        # Grasp rotation tester
        self._ik_servo_fn: Any = None
        self._get_state_fn: Any = None
        self._setup_grasp_rotation_ui()

        # Apply initial focus mode (hides debug GUI)
        self._apply_focus_mode()

        # Latest state
        self._latest_joints: np.ndarray | None = None
        self._latest_detections: list[dict] | None = None
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        return self._port

    @property
    def focused(self) -> bool:
        return self._focus_mode.value

    def _apply_focus_mode(self) -> None:
        """Show/hide all debug overlays based on Focus toggle."""
        show = not self._focus_mode.value

        # GUI sidebar elements
        self._show_safety_zone.visible = show
        for folder in self._debug_gui_folders:
            folder.visible = show

        # EE coordinate frames
        self._left_ee_frame.visible = show
        self._right_ee_frame.visible = show

        # Gizmos
        if self._pose_gizmo is not None:
            self._pose_gizmo.visible = show
        if self._ee_gizmo is not None:
            self._ee_gizmo.visible = show

        # Detection markers + labels
        for h in self._det_markers:
            h.visible = show
        for h in self._det_labels:
            h.visible = show

        # Grasp pose frames
        for h in self._grasp_handles:
            h.visible = show

        # Prediction spheres
        for h in self._pred_left:
            h.visible = show
        for h in self._pred_right:
            h.visible = show

        # Safety zone
        for h in self._safety_handles:
            h.visible = show

    def update_robot_state(
        self,
        left_joint_pos: list[float] | np.ndarray,
        right_joint_pos: list[float] | np.ndarray,
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
    ) -> None:
        """Update the robot visualization with current joint positions."""
        if self._urdf is None:
            return

        # Build full joint config matching URDF actuated joint order
        # URDF joint order: left_joint_1..6, left_gripper_joint, left_inner_finger_joint,
        #                   right_joint_1..6, right_gripper_joint, right_inner_finger_joint
        left_jp = np.asarray(left_joint_pos, dtype=np.float64)
        right_jp = np.asarray(right_joint_pos, dtype=np.float64)

        # Gripper: convert from [-1, 1] to joint angle
        # The URDF gripper joint and inner_finger_joint are mimic joints
        left_g = float(left_gripper) * 0.0376
        right_g = float(right_gripper) * 0.0376

        cfg = np.concatenate(
            [
                left_jp,
                [left_g, -left_g],  # gripper_joint, inner_finger_joint (mimic)
                right_jp,
                [right_g, -right_g],
            ]
        )

        # Pad or trim to match URDF joint count
        n_joints = len(self._joint_names)
        if len(cfg) < n_joints:
            cfg = np.pad(cfg, (0, n_joints - len(cfg)))
        elif len(cfg) > n_joints:
            cfg = cfg[:n_joints]

        with self._lock:
            self._latest_joints = cfg

        self._urdf.update_cfg(cfg)

    def update_ee_poses(
        self,
        left_pos: list[float],
        left_quat_xyzw: list[float],
        right_pos: list[float],
        right_quat_xyzw: list[float],
    ) -> None:
        """Update EE coordinate frame visualizations."""
        self._left_ee_frame.position = tuple(float(v) for v in left_pos)
        self._left_ee_frame.wxyz = (
            float(left_quat_xyzw[3]),
            float(left_quat_xyzw[0]),
            float(left_quat_xyzw[1]),
            float(left_quat_xyzw[2]),
        )
        self._left_ee_frame.visible = not self.focused

        self._right_ee_frame.position = tuple(float(v) for v in right_pos)
        self._right_ee_frame.wxyz = (
            float(right_quat_xyzw[3]),
            float(right_quat_xyzw[0]),
            float(right_quat_xyzw[1]),
            float(right_quat_xyzw[2]),
        )
        self._right_ee_frame.visible = not self.focused

    # ------------------------------------------------------------------
    # Right EE gizmo (draggable)
    # ------------------------------------------------------------------

    def _setup_ee_gizmo(self) -> None:
        """Draggable gizmo for right EE with sync button and pose readout."""

        self._ee_gizmo = self._server.scene.add_transform_controls(
            "/debug/ee_gizmo",
            scale=0.10,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )

        _folder = self._server.gui.add_folder("Right EE Gizmo")
        self._debug_gui_folders.append(_folder)
        with _folder:
            self._ee_gizmo_texts["pos"] = self._server.gui.add_text(
                label="Position XYZ",
                initial_value="—",
                disabled=True,
            )
            self._ee_gizmo_texts["quat"] = self._server.gui.add_text(
                label="Quat XYZW",
                initial_value="—",
                disabled=True,
            )
            self._ee_gizmo_texts["rpy"] = self._server.gui.add_text(
                label="RPY (deg)",
                initial_value="—",
                disabled=True,
            )
            sync_btn = self._server.gui.add_button("Sync Gizmo to Right EE")
            move_btn = self._server.gui.add_button("Move Right EE to Gizmo")

            @sync_btn.on_click
            def _on_sync(_) -> None:
                if self._get_state_fn is None:
                    return
                state = self._get_state_fn()
                pos = state.right_ee_pos
                quat_xyzw = state.right_ee_quat
                self._ee_gizmo.position = tuple(float(v) for v in pos)
                self._ee_gizmo.wxyz = (
                    float(quat_xyzw[3]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                )
                self._ee_gizmo.visible = not self.focused
                self._update_ee_gizmo_readout()

            @move_btn.on_click
            def _on_move_to_gizmo(_) -> None:
                import threading as _th

                if self._ik_servo_fn is None or self._ee_gizmo is None:
                    return
                pos = list(self._ee_gizmo.position)
                wxyz = self._ee_gizmo.wxyz
                quat_xyzw = [
                    float(wxyz[1]),
                    float(wxyz[2]),
                    float(wxyz[3]),
                    float(wxyz[0]),
                ]

                def _move():
                    try:
                        self._ik_servo_fn("right", pos, quat_xyzw)
                    except Exception as e:
                        print(f"[Visualizer] Move right EE to gizmo failed: {e}")

                _th.Thread(target=_move, daemon=True).start()

        @self._ee_gizmo.on_update
        def _on_ee_gizmo_move(_) -> None:
            self._update_ee_gizmo_readout()

    def _update_ee_gizmo_readout(self) -> None:
        from scipy.spatial.transform import Rotation

        pos = self._ee_gizmo.position
        wxyz = self._ee_gizmo.wxyz
        quat_xyzw = [float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])]
        rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        self._ee_gizmo_texts["pos"].value = f"{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
        self._ee_gizmo_texts[
            "quat"
        ].value = f"{quat_xyzw[0]:.3f}, {quat_xyzw[1]:.3f}, {quat_xyzw[2]:.3f}, {quat_xyzw[3]:.3f}"
        self._ee_gizmo_texts["rpy"].value = f"{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}"

    # ------------------------------------------------------------------
    # Pose gizmo (draggable debug tool)
    # ------------------------------------------------------------------

    def _setup_pose_gizmo(self) -> None:
        """Create a draggable gizmo and a GUI panel showing its live pose."""
        from scipy.spatial.transform import Rotation

        self._pose_gizmo = self._server.scene.add_transform_controls(
            "/debug/pose_gizmo",
            scale=0.12,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )

        _folder = self._server.gui.add_folder("Pose Gizmo")
        self._debug_gui_folders.append(_folder)
        with _folder:
            self._gizmo_texts["pos"] = self._server.gui.add_text(
                label="Position XYZ",
                initial_value="—",
                disabled=True,
            )
            self._gizmo_texts["quat"] = self._server.gui.add_text(
                label="Quat XYZW",
                initial_value="—",
                disabled=True,
            )
            self._gizmo_texts["rpy"] = self._server.gui.add_text(
                label="RPY (deg)",
                initial_value="—",
                disabled=True,
            )
            self._gizmo_sync_btn = self._server.gui.add_button("Sync from Left EEF")
            self._gizmo_move_btn = self._server.gui.add_button("Move Left EEF to Gizmo")

        @self._pose_gizmo.on_update
        def _on_gizmo_move(_) -> None:
            pos = self._pose_gizmo.position
            wxyz = self._pose_gizmo.wxyz
            quat_xyzw = [float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])]
            rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
            self._gizmo_texts["pos"].value = f"{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
            self._gizmo_texts[
                "quat"
            ].value = f"{quat_xyzw[0]:.3f}, {quat_xyzw[1]:.3f}, {quat_xyzw[2]:.3f}, {quat_xyzw[3]:.3f}"
            self._gizmo_texts["rpy"].value = f"{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}"

        @self._gizmo_sync_btn.on_click
        def _on_sync_from_eef(_) -> None:
            """Copy current left EEF pose to the gizmo."""
            if self._get_state_fn is None:
                return
            try:
                state = self._get_state_fn()
                self.set_pose_gizmo(state.left_ee_pos, state.left_ee_quat)
            except Exception as e:
                print(f"[Visualizer] Sync gizmo from EEF failed: {e}")

        @self._gizmo_move_btn.on_click
        def _on_ik_servo_to_gizmo(_) -> None:
            """Move left EEF to the gizmo's current pose."""
            import threading as _th

            if self._ik_servo_fn is None or self._pose_gizmo is None:
                return
            pos = self._pose_gizmo.position
            wxyz = self._pose_gizmo.wxyz
            quat_xyzw = [float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])]

            def _move():
                try:
                    self._ik_servo_fn("left", list(pos), quat_xyzw)
                except Exception as e:
                    print(f"[Visualizer] Move EEF to gizmo failed: {e}")

            _th.Thread(target=_move, daemon=True).start()

    def set_pose_gizmo(
        self, position: list | tuple, quaternion_xyzw: list | tuple
    ) -> None:
        """Move the debug gizmo to a specific pose and make it visible.

        Called automatically when a 6-DOF detection arrives, or manually from scripts.
        """
        from scipy.spatial.transform import Rotation

        if self._pose_gizmo is None:
            return
        pos = (float(position[0]), float(position[1]), float(position[2]))
        wxyz = (
            float(quaternion_xyzw[3]),
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
        )
        self._pose_gizmo.position = pos
        self._pose_gizmo.wxyz = wxyz
        self._pose_gizmo.visible = not self.focused

        # Update readout
        quat_xyzw = [float(v) for v in quaternion_xyzw]
        rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        self._gizmo_texts["pos"].value = f"{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
        self._gizmo_texts[
            "quat"
        ].value = f"{quat_xyzw[0]:.3f}, {quat_xyzw[1]:.3f}, {quat_xyzw[2]:.3f}, {quat_xyzw[3]:.3f}"
        self._gizmo_texts["rpy"].value = f"{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}"

    # ------------------------------------------------------------------
    # Grasp pose visualization
    # ------------------------------------------------------------------

    def _setup_grasp_pose_ui(self) -> None:
        """Add a "Show Grasps" button that calls the active grasp planner and renders results."""
        import threading as _th

        folder = self._server.gui.add_folder("Grasp Planning")
        self._debug_gui_folders.append(folder)

        with folder:
            self._grasp_query = self._server.gui.add_text(
                label="Object",
                initial_value="grapes",
            )
            self._grasp_camera = self._server.gui.add_dropdown(
                label="Camera",
                options=["top", "left", "right"],
                initial_value="top",
            )
            self._grasp_max = self._server.gui.add_slider(
                label="Max grasps",
                min=1,
                max=20,
                step=1,
                initial_value=5,
            )
            self._grasp_btn = self._server.gui.add_button("Show Grasps")
            self._grasp_clear_btn = self._server.gui.add_button("Clear Grasps")
            self._grasp_status = self._server.gui.add_text(
                label="Status",
                initial_value="—",
                disabled=True,
            )

        @self._grasp_btn.on_click
        def _on_show_grasps(_) -> None:
            if self._sample_grasp_fn is None:
                self._grasp_status.value = "sample_grasp_pose_anygrasp not available"
                return
            self._grasp_status.value = "Planning..."

            def _run():
                try:
                    result = self._sample_grasp_fn(
                        self._grasp_query.value,
                        camera=self._grasp_camera.value,
                        max_grasps=int(self._grasp_max.value),
                    )
                    if result and len(result) > 0:
                        self.show_grasp_poses(result)
                        self._grasp_status.value = f"{len(result)} grasps"
                    else:
                        self._grasp_status.value = "No grasps found"
                except Exception as e:
                    self._grasp_status.value = f"Error: {e}"

            _th.Thread(target=_run, daemon=True).start()

        @self._grasp_clear_btn.on_click
        def _on_clear_grasps(_) -> None:
            self.clear_grasp_poses()
            self._grasp_status.value = "—"

    def show_grasp_poses(self, grasps: list) -> None:
        """Render grasp candidates as coordinate frames in the 3D scene."""
        from scipy.spatial.transform import Rotation

        self.clear_grasp_poses()

        for i, g in enumerate(grasps):
            pos = g.position if hasattr(g, "position") else g["position"]
            if hasattr(g, "rpy"):
                rpy = g.rpy
            else:
                rpy = g["rpy"]
            quat = Rotation.from_euler("xyz", rpy, degrees=True).as_quat().tolist()
            score = g.score if hasattr(g, "score") else g.get("score", 0)

            wxyz = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]

            # Scale frame size by rank — best grasp is largest
            scale = 0.08 if i == 0 else 0.04
            opacity = 1.0 if i == 0 else 0.5

            frame = self._server.scene.add_frame(
                f"/grasps/grasp_{i}",
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                wxyz=tuple(wxyz),
                axes_length=scale,
                axes_radius=scale * 0.08,
            )
            self._grasp_handles.append(frame)

            label = self._server.scene.add_label(
                f"/grasps/grasp_{i}/label",
                text=f"#{i + 1} ({score:.0%})",
                wxyz=(1, 0, 0, 0),
                position=(0, 0, scale * 1.5),
            )
            self._grasp_handles.append(label)

    def clear_grasp_poses(self) -> None:
        """Remove all grasp pose visualizations."""
        for h in self._grasp_handles:
            h.remove()
        self._grasp_handles.clear()

    # ------------------------------------------------------------------
    # Grasp rotation tester
    # ------------------------------------------------------------------

    # 8 common grasp rotations: identity + 90° increments about X, Y, Z + 180° combos
    _GRASP_ROTATIONS: dict[str, list[float]] = {}  # populated in _setup

    def _setup_grasp_rotation_ui(self) -> None:
        """Dropdown + button to test grasp rotations on the left arm."""
        import threading as _th
        from scipy.spatial.transform import Rotation

        # Build 8 candidate rotations (as quaternion xyzw)
        candidates = {
            "Identity": Rotation.identity(),
            "X +90": Rotation.from_euler("x", 90, degrees=True),
            "X -90": Rotation.from_euler("x", -90, degrees=True),
            "X 180": Rotation.from_euler("x", 180, degrees=True),
            "Y +90": Rotation.from_euler("y", 90, degrees=True),
            "Y -90": Rotation.from_euler("y", -90, degrees=True),
            "Z +90": Rotation.from_euler("z", 90, degrees=True),
            "Z -90": Rotation.from_euler("z", -90, degrees=True),
        }
        self._GRASP_ROTATIONS = {
            name: rot.as_quat().tolist() for name, rot in candidates.items()
        }

        options = list(self._GRASP_ROTATIONS.keys())

        _folder = self._server.gui.add_folder("Grasp Rotation Tester")
        self._debug_gui_folders.append(_folder)
        with _folder:
            dropdown = self._server.gui.add_dropdown(
                "Grasp rotation",
                options=options,
                initial_value=options[0],
            )
            self._grasp_rot_result = self._server.gui.add_text(
                label="EE Quat XYZW",
                initial_value="—",
                disabled=True,
            )
            btn = self._server.gui.add_button("Rotate Right Arm")

            @btn.on_click
            def _on_rotate(_) -> None:
                if self._ik_servo_fn is None or self._get_state_fn is None:
                    return
                rot_name = dropdown.value
                grasp_rot_xyzw = self._GRASP_ROTATIONS[rot_name]

                # Get current gizmo pose (plate pose)
                if self._pose_gizmo is None:
                    return
                wxyz = self._pose_gizmo.wxyz
                plate_quat_xyzw = [
                    float(wxyz[1]),
                    float(wxyz[2]),
                    float(wxyz[3]),
                    float(wxyz[0]),
                ]

                # Compose: ee_quat = plate_quat * grasp_rotation
                plate_rot = Rotation.from_quat(plate_quat_xyzw)
                grasp_rot = Rotation.from_quat(grasp_rot_xyzw)
                ee_rot = plate_rot * grasp_rot
                ee_quat_xyzw = ee_rot.as_quat().tolist()

                # Get current right arm EE position (rotate in-place)
                state = self._get_state_fn()
                ee_pos = list(state.right_ee_pos)

                self._grasp_rot_result.value = (
                    f"{ee_quat_xyzw[0]:.3f}, {ee_quat_xyzw[1]:.3f}, "
                    f"{ee_quat_xyzw[2]:.3f}, {ee_quat_xyzw[3]:.3f}"
                )

                # Move in background thread to avoid blocking viser UI
                def _move():
                    try:
                        self._ik_servo_fn("right", ee_pos, ee_quat_xyzw)
                    except Exception as e:
                        print(f"[Visualizer] Grasp rotation move failed: {e}")

                _th.Thread(target=_move, daemon=True).start()

    def set_robot_callbacks(self, ik_servo_fn, get_state_fn, sample_grasp_fn=None) -> None:
        """Wire up robot control callbacks (called from cap_agent after callables are built)."""
        self._ik_servo_fn = ik_servo_fn
        self._get_state_fn = get_state_fn
        self._sample_grasp_fn = sample_grasp_fn

    def update_detections(self, detections: list[dict]) -> None:
        """Update detection markers in the 3D scene.

        Each detection should have:
            - label: str
            - score: float
            - position_3d: [x, y, z] in world frame
            - quaternion_xyzw: [x, y, z, w] (optional — shows coordinate frame)
        """

        with self._lock:
            self._latest_detections = detections

        # Remove old markers
        for h in self._det_markers:
            h.remove()
        for h in self._det_labels:
            h.remove()
        self._det_markers.clear()
        self._det_labels.clear()

        if not detections:
            return

        for i, det in enumerate(detections):
            pos = det.get("position_3d", [])
            if not pos or len(pos) < 3:
                continue

            score = det.get("score", 0.0)
            label = det.get("label", "?")
            quat = det.get("quaternion_xyzw", [])

            # Color by confidence: green > 0.5, yellow > 0.2, red otherwise
            if score > 0.5:
                color = (0, 200, 0)
            elif score > 0.2:
                color = (200, 200, 0)
            else:
                color = (200, 0, 0)

            position = (float(pos[0]), float(pos[1]), float(pos[2]))

            _vis = not self.focused

            if quat and len(quat) == 4:
                # 6-DOF detection (bundlesdf) — show coordinate frame
                # viser uses wxyz quaternion convention
                wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
                frame = self._server.scene.add_frame(
                    f"/detections/frame_{i}",
                    wxyz=wxyz,
                    position=position,
                    axes_length=0.06,
                    axes_radius=0.003,
                    visible=_vis,
                )
                self._det_markers.append(frame)
            else:
                # Position-only detection — show sphere
                marker = self._server.scene.add_icosphere(
                    f"/detections/sphere_{i}",
                    radius=0.015,
                    color=color,
                    position=position,
                    visible=_vis,
                )
                self._det_markers.append(marker)

            # Label above the marker
            label_pos = (position[0], position[1], position[2] + 0.04)
            lbl = self._server.scene.add_label(
                f"/detections/label_{i}",
                text=f"{label} ({score:.2f})",
                position=label_pos,
                visible=_vis,
            )
            self._det_labels.append(lbl)

        # Auto-snap gizmo to the first 6-DOF detection
        for det in detections:
            quat = det.get("quaternion_xyzw", [])
            pos = det.get("position_3d", [])
            if quat and len(quat) == 4 and pos and len(pos) >= 3:
                self.set_pose_gizmo(pos, quat)
                break

    # ------------------------------------------------------------------
    # Trajectory prediction visualization
    # ------------------------------------------------------------------

    @staticmethod
    def _point_color(t_norm: float) -> tuple[int, int, int]:
        """Blue (early) -> red (late) color gradient for prediction spheres."""
        r = int(t_norm * 255)
        g = int(0.2 * 255)
        b = int((1.0 - t_norm) * 255)
        return (r, g, b)

    def _ensure_pred_spheres(self, count: int) -> None:
        """Lazily create prediction icospheres up to *count*."""
        while len(self._pred_left) < count:
            i = len(self._pred_left)
            t_norm = i / max(count - 1, 1)
            color = self._point_color(t_norm)
            h = self._server.scene.add_icosphere(
                f"/pred/left_{i}",
                radius=0.008,
                color=color,
                position=(0.0, 0.0, 0.0),
            )
            h.visible = False
            self._pred_left.append(h)
        while len(self._pred_right) < count:
            i = len(self._pred_right)
            t_norm = i / max(count - 1, 1)
            color = self._point_color(t_norm)
            h = self._server.scene.add_icosphere(
                f"/pred/right_{i}",
                radius=0.008,
                color=color,
                position=(0.0, 0.0, 0.0),
            )
            h.visible = False
            self._pred_right.append(h)

    def update_prediction(self, action_chunk: dict) -> None:
        """Render predicted EE trajectory as colored spheres.

        Accepts an action chunk dict with keys like left_joint_pos (H,6),
        right_joint_pos (H,6) — or ee_pose keys which are converted via IK.
        """

        def _get_seq(keys: list[str]) -> np.ndarray | None:
            for k in keys:
                if k in action_chunk:
                    seq = np.asarray(action_chunk[k], dtype=np.float64)
                    if seq.ndim == 3:
                        seq = seq[0]  # strip batch dim
                    return seq
            return None

        # Determine joint sequences — convert from ee_pose if needed
        if "left_ee_pos" in action_chunk and "right_ee_pos" in action_chunk:
            left_ee_pos = _get_seq(["left_ee_pos"])
            left_ee_quat = _get_seq(["left_ee_quat_xyzw"])
            right_ee_pos = _get_seq(["right_ee_pos"])
            right_ee_quat = _get_seq(["right_ee_quat_xyzw"])
            if (
                left_ee_pos is None
                or left_ee_quat is None
                or right_ee_pos is None
                or right_ee_quat is None
            ):
                return
            horizon = left_ee_pos.shape[0]
            left_seq = np.zeros((horizon, 6))
            right_seq = np.zeros((horizon, 6))
            for s in range(horizon):
                left_seq[s], right_seq[s] = self._kin.inverse_kinematics(
                    left_ee_pos[s],
                    left_ee_quat[s],
                    right_ee_pos[s],
                    right_ee_quat[s],
                    seeded=(s > 0),
                )
        else:
            left_seq = _get_seq(["left_joint_pos", "joint_pos_action_left"])
            right_seq = _get_seq(["right_joint_pos", "joint_pos_action_right"])

        n_left = left_seq.shape[0] if left_seq is not None else 0
        n_right = right_seq.shape[0] if right_seq is not None else 0
        steps = max(n_left, n_right)
        if steps <= 0:
            return

        self._ensure_pred_spheres(steps)

        # Compute FK for each timestep and position spheres
        for i in range(steps):
            lj = left_seq[i] if left_seq is not None and i < n_left else np.zeros(6)
            rj = right_seq[i] if right_seq is not None and i < n_right else np.zeros(6)
            try:
                l_pos, _, r_pos, _ = self._kin.forward_kinematics(lj[:6], rj[:6])
            except Exception:
                continue

            if i < len(self._pred_left):
                self._pred_left[i].position = (
                    float(l_pos[0]),
                    float(l_pos[1]),
                    float(l_pos[2]),
                )
                self._pred_left[i].visible = not self.focused
            if i < len(self._pred_right):
                self._pred_right[i].position = (
                    float(r_pos[0]),
                    float(r_pos[1]),
                    float(r_pos[2]),
                )
                self._pred_right[i].visible = not self.focused

        # Hide leftover spheres from previous (longer) chunk
        for i in range(steps, len(self._pred_left)):
            self._pred_left[i].visible = False
        for i in range(steps, len(self._pred_right)):
            self._pred_right[i].visible = False

    def clear_prediction(self) -> None:
        """Hide all prediction spheres."""
        for h in self._pred_left:
            h.visible = False
        for h in self._pred_right:
            h.visible = False

    # ------------------------------------------------------------------
    # Scene object visualization
    # ------------------------------------------------------------------

    # MuJoCo geom type enum values
    _GEOM_BOX = 6

    def update_scene_objects(self, objects: dict[str, dict]) -> None:
        """Render or update scene objects (boxes) from sim get_object_positions().

        *objects* maps body name -> {"pos", "quat" (wxyz), "geoms": [{"type", "size", "pos", "rgba"}, ...]}.
        Only box geoms are rendered. Handles are created once, then only position/
        orientation is updated on subsequent calls.
        """
        for name, data in objects.items():
            body_pos = np.asarray(data["pos"])
            bq = data["quat"]  # wxyz from MuJoCo
            wxyz = (float(bq[0]), float(bq[1]), float(bq[2]), float(bq[3]))

            # Filter to box geoms only
            box_geoms = [
                g for g in data.get("geoms", []) if g["type"] == self._GEOM_BOX
            ]
            if not box_geoms:
                continue

            existing = self._scene_handles.get(name)
            if existing is not None and len(existing) != len(box_geoms):
                # Geom count changed (scene reloaded) — recreate
                for h in existing:
                    h.remove()
                existing = None
                del self._scene_handles[name]

            if existing is not None:
                # Fast path: only update positions (body moved)
                # Recompute world positions using simple quaternion rotation
                bq_xyzw = np.array([bq[1], bq[2], bq[3], bq[0]])
                for i, geom in enumerate(box_geoms):
                    g_local = np.asarray(geom["pos"])
                    world_pos = body_pos + self._quat_rotate(bq_xyzw, g_local)
                    existing[i].position = (
                        float(world_pos[0]),
                        float(world_pos[1]),
                        float(world_pos[2]),
                    )
                    existing[i].wxyz = wxyz
            else:
                # First time: create box handles
                bq_xyzw = np.array([bq[1], bq[2], bq[3], bq[0]])
                handles = []
                for i, geom in enumerate(box_geoms):
                    g_size = np.asarray(geom["size"])
                    g_local = np.asarray(geom["pos"])
                    g_rgba = geom["rgba"]

                    world_pos = body_pos + self._quat_rotate(bq_xyzw, g_local)
                    h = self._server.scene.add_box(
                        f"/scene/{name}/geom_{i}",
                        dimensions=(2 * g_size).tolist(),
                        color=(
                            int(g_rgba[0] * 255),
                            int(g_rgba[1] * 255),
                            int(g_rgba[2] * 255),
                        ),
                        opacity=float(g_rgba[3]) if len(g_rgba) > 3 else 1.0,
                        position=(
                            float(world_pos[0]),
                            float(world_pos[1]),
                            float(world_pos[2]),
                        ),
                        wxyz=wxyz,
                    )
                    handles.append(h)
                self._scene_handles[name] = handles

    @staticmethod
    def _quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector *v* by quaternion *q_xyzw* (no scipy overhead)."""
        q = q_xyzw
        t = 2.0 * np.cross(q[:3], v)
        return v + q[3] * t + np.cross(q[:3], t)

    def clear_scene_objects(self) -> None:
        """Remove all scene object visuals."""
        for handles in self._scene_handles.values():
            for h in handles:
                h.remove()
        self._scene_handles.clear()

    # ------------------------------------------------------------------
    # Safety zone visualization
    # ------------------------------------------------------------------

    _SAFE_COLOR = (0, 180, 255)  # cyan-ish for the safe volume
    _HARD_COLOR = (255, 80, 80)  # red for hard boundary
    _KEYPOSE_COLOR = (255, 220, 0)  # yellow for keypose markers

    def update_safety_zone(self, zone_config: dict) -> None:
        """Render the task safety zone using boxes (avoids Viser icosphere crash).

        Per arm renders:
          - Yellow box (small) at each keypose position
          - One cyan transparent box = AABB of all keyposes + pos_margin (safe zone)
          - One red transparent box = AABB of all keyposes + pos_margin + elastic_band (hard cutoff)

        The AABB is computed per-axis from the keypose positions, so the Z extent
        is tight to the actual hover→insertion range rather than a uniform cube.
        """
        # Skip if config hasn't changed
        if zone_config == self._safety_zone_config:
            return

        logger.info(
            "[safety_viz] zone config changed, active=%s",
            zone_config.get("active") if zone_config else None,
        )

        # Clear old handles (this resets _safety_zone_config to None)
        self._clear_safety_handles()

        # Set cache AFTER clearing so it doesn't get nulled
        self._safety_zone_config = zone_config

        if not zone_config or not zone_config.get("active"):
            logger.info("[safety_viz] zone inactive, cleared visuals")
            return

        if not self._show_safety_zone.value:
            return

        for side in ("left", "right"):
            arm_cfg = zone_config.get(side)
            if arm_cfg is None:
                continue

            keyposes = arm_cfg.get("keyposes", [])
            pos_margin = arm_cfg.get("pos_margin", 0.08)
            elastic_band = arm_cfg.get("elastic_band", 0.04)
            logger.info(
                "[safety_viz] rendering %s: %d keyposes, margin=%.3f, band=%.3f",
                side,
                len(keyposes),
                pos_margin,
                elastic_band,
            )

            if not keyposes:
                continue

            # Compute AABB of all keypose positions
            positions = np.array(
                [[float(kp[0]), float(kp[1]), float(kp[2])] for kp in keyposes]
            )
            aabb_min = positions.min(axis=0)
            aabb_max = positions.max(axis=0)

            # Keypose markers — small solid yellow boxes
            for i, kp in enumerate(keyposes):
                pos = (float(kp[0]), float(kp[1]), float(kp[2]))
                try:
                    d = 0.02
                    h = self._server.scene.add_box(
                        f"/safety/{side}/keypose_{i}",
                        dimensions=(d, d, d),
                        color=self._KEYPOSE_COLOR,
                        position=pos,
                    )
                    self._safety_handles.append(h)
                except Exception:
                    logger.exception("[safety_viz] FAILED to add keypose_%d", i)

            # Safe zone — one cyan box = AABB + pos_margin per axis
            try:
                safe_min = aabb_min - pos_margin
                safe_max = aabb_max + pos_margin
                safe_dims = safe_max - safe_min
                safe_center = (safe_min + safe_max) / 2.0
                h = self._server.scene.add_box(
                    f"/safety/{side}/safe",
                    dimensions=(
                        float(safe_dims[0]),
                        float(safe_dims[1]),
                        float(safe_dims[2]),
                    ),
                    color=self._SAFE_COLOR,
                    opacity=0.12,
                    side="back",
                    position=(
                        float(safe_center[0]),
                        float(safe_center[1]),
                        float(safe_center[2]),
                    ),
                )
                self._safety_handles.append(h)
            except Exception:
                logger.exception("[safety_viz] FAILED to add safe box for %s", side)

            # Hard boundary — one red box = AABB + pos_margin + elastic_band
            try:
                hard_margin = pos_margin + elastic_band
                hard_min = aabb_min - hard_margin
                hard_max = aabb_max + hard_margin
                hard_dims = hard_max - hard_min
                hard_center = (hard_min + hard_max) / 2.0
                h = self._server.scene.add_box(
                    f"/safety/{side}/hard",
                    dimensions=(
                        float(hard_dims[0]),
                        float(hard_dims[1]),
                        float(hard_dims[2]),
                    ),
                    color=self._HARD_COLOR,
                    opacity=0.06,
                    side="back",
                    position=(
                        float(hard_center[0]),
                        float(hard_center[1]),
                        float(hard_center[2]),
                    ),
                )
                self._safety_handles.append(h)
            except Exception:
                logger.exception("[safety_viz] FAILED to add hard box for %s", side)

        logger.info(
            "[safety_viz] rendering complete: %d handles total",
            len(self._safety_handles),
        )

    def _clear_safety_handles(self, reset_config: bool = True) -> None:
        for i, h in enumerate(self._safety_handles):
            try:
                h.remove()
            except Exception:
                logger.exception("[safety_viz] FAILED to remove handle %d", i)
        self._safety_handles.clear()
        if reset_config:
            self._safety_zone_config = None

    def clear_safety_zone(self) -> None:
        """Remove all safety zone visuals."""
        logger.info("[safety_viz] clear_safety_zone called")
        self._clear_safety_handles()

    def close(self) -> None:
        """Shut down the Viser server."""
        pass  # ViserServer doesn't have a close method; it stops when GC'd
