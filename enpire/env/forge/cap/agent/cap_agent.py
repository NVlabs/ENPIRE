# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAP Agent — orchestrator with REST + WebSocket API.

Provides:
- POST /api/task      — submit a natural-language task (agent mode)
- POST /api/execute   — run arbitrary code (oracle mode)
- POST /api/approve   — approve LLM-generated code
- POST /api/reject    — reject / request revision
- POST /api/mode      — switch between agent / oracle mode
- POST /api/debug_mode — toggle simulation debug mode (no real robot)
- POST /api/estop     — emergency stop
- POST /api/pause     — pause execution
- POST /api/resume    — resume execution
- POST /api/stop      — stop current task
- POST /api/home      — send robot home
- WS   /ws            — real-time state updates, code proposals, execution logs
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Ensure project root is on sys.path so `cap.*` imports work when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Make skill_library importable from executor sandbox (cap/saved_scripts/skill_library/)
_SAVED_SCRIPTS_ROOT = str(Path(_PROJECT_ROOT) / "cap" / "saved_scripts")
if _SAVED_SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SAVED_SCRIPTS_ROOT)

import threading

import numpy as np
import portal
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from enpire.env.forge.cap.agent.executor import ExecutionLog, ExecutionResult, Executor
from enpire.env.forge.cap.agent.tools import ToolRegistry, create_default_registry
from enpire.env.forge.cap.agent.visualizer import CapVisualizer
from enpire.env.forge.cap.config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    CAP_AGENT_PORT,
    CAP_SERVER_PORT,
    DEFAULT_LLM_MODEL,
    DEFAULT_POLICY_SERVER,
    DEFAULT_VLM_BACKEND,
    DETECTION_SERVER_PORT,
    VISER_PORT,
)

logger = logging.getLogger(__name__)

CONTACT_DETECT_THRESHOLD_LOW = 0.005
CONTACT_DETECT_THRESHOLD_HIGH = 0.99
POLICY_PREDICTION_RPC_TIMEOUT_S = 2.0
VISER_AXIS_LEN_M = 0.04
SAFETY_VIZ_ENABLED = os.environ.get("CAP_SAFETY_VIZ_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    description: str


class TaskResponse(BaseModel):
    task_id: str


class ExecuteRequest(BaseModel):
    code: str


class RejectRequest(BaseModel):
    feedback: str | None = None


class ModeRequest(BaseModel):
    mode: Literal["agent", "oracle"]


class DebugModeRequest(BaseModel):
    enabled: bool


class CameraStreamRequest(BaseModel):
    camera: str
    enabled: bool


class OkResponse(BaseModel):
    ok: bool


class SaveScriptRequest(BaseModel):
    name: str
    code: str


class RenameScriptRequest(BaseModel):
    new_name: str


class ScriptInfo(BaseModel):
    name: str
    size: int
    modified: str


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    mode: Literal["agent", "oracle"] = "oracle"
    status: Literal[
        "idle",
        "generating",
        "awaiting_approval",
        "executing",
        "paused",
        "stopped",
        "waiting_for_agent",
    ] = "idle"
    current_task: str = ""
    proposed_code: str = ""
    task_counter: int = 0
    action_log: list[dict[str, Any]] = field(default_factory=list)
    # Preserved namespace from wait_for_agent pause
    paused_namespace: dict[str, Any] | None = None
    # Debug mode: simulate motions in Viser, no real robot commands
    debug_mode: bool = False


MAX_AGENT_ACTION_LOG_ENTRIES = 200


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.remove(ws)

    async def broadcast(self, msg_type: str, data: Any) -> None:
        payload = json.dumps(
            {"type": msg_type, "data": data, "timestamp": time.strftime("%H:%M:%S")},
            default=str,
        )
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._connections.remove(ws)


# ---------------------------------------------------------------------------
# Simulated robot for debug mode
# ---------------------------------------------------------------------------


class _SimulatedRobot:
    """In-memory simulated robot for debug mode.

    Delegates to a YamEnv (from cap.env.yam) which handles FK/IK via
    pinocchio. No hardcoded joint sizes or arm names — works for any
    robot that has an env with EefControlProtocol.
    """

    def __init__(
        self,
        visualizer: CapVisualizer,
        cancel_event: threading.Event | None = None,
    ) -> None:
        from enpire.env.forge.cap.env.yam import YamEnv

        self._env = YamEnv(viewer=False)
        self._vis = visualizer
        self._cancel = cancel_event
        self._lock = threading.Lock()

    # -- public API (matches tool callable signatures) -----------------------

    def get_state(self):
        from scipy.spatial.transform import Rotation as R

        from enpire.env.forge.cap.agent.tools.base import ArmState, RobotState

        def _rpy(q):
            e = R.from_quat(q).as_euler("xyz", degrees=True)
            return [float(e[1]), float(-e[0]), float(-(e[2] + 90))]

        arms = {}
        for side in ("left", "right"):
            obs = self._env.get_arm_observation(side)
            ee_pos = obs.get("ee_pos", [0, 0, 0])
            ee_quat = obs.get("ee_quat", [0, 0, 0, 1])
            arms[side] = ArmState(
                joint_pos=[float(v) for v in obs["joint_pos"]],
                gripper_pos=float(obs["gripper_pos"][0]),
                ee_pos=[float(v) for v in ee_pos],
                ee_quat=[float(v) for v in ee_quat],
                ee_rpy=_rpy(ee_quat),
            )
        return RobotState(arms=arms)

    def _ik_servo(
        self,
        side: str,
        pos,
        quat,
        gripper=None,
        ignore_safety=False,
        check_feasibility: bool = False,
        tol: float = 0.02,
        dry_run_only: bool = False,
    ):
        from enpire.env.forge.cap.agent.tools.base import MoveResult

        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)

        if dry_run_only and check_feasibility:
            return MoveResult(
                reached=False, feasible=True, cmd_pos_err=0.0,
                final_pos=list(pos), final_quat=list(quat),
            )

        result = self._env._ik_servo(side, pos, quat, gripper=gripper, tol=tol)

        return MoveResult(
            reached=result.get("success", False),
            feasible=True,
            cmd_pos_err=result.get("pos_err", 0.0),
            final_pos=list(pos),
            final_quat=list(quat),
        )

    def open_gripper(self, side: str, vel_limit=None) -> bool:
        self._env.set_gripper(side, 1.0)
        return True

    def close_gripper(self, side: str, vel_limit=None) -> bool:
        self._env.set_gripper(side, 0.0)
        return True

    def go_home(self) -> bool:
        self._env.go_home()
        return True

    def set_gripper(self, side: str, value: float, vel_limit=None, torque_limit=None) -> bool:
        self._env.set_gripper(side, value)
        return True


# ---------------------------------------------------------------------------
# Build the FastAPI app
# ---------------------------------------------------------------------------


def create_app(
    cap_server_host: str = "localhost",
    cap_server_port: int = CAP_SERVER_PORT,
    detection_host: str = "localhost",
    detection_port: int = DETECTION_SERVER_PORT,
    policy_server: str = DEFAULT_POLICY_SERVER,
    llm_backend: str = "claude",
    llm_model: str = DEFAULT_LLM_MODEL,
    viser_port: int = VISER_PORT,
    bundlesdf_host: str | None = None,
    bundlesdf_port: int | None = None,
    sam3_host: str | None = None,
    sam3_port: int | None = None,
) -> FastAPI:
    app = FastAPI(title="CAP Agent")
    manager = ConnectionManager()
    state = AgentState()
    _streaming_cameras: set[str] = set()  # cameras actively streaming to UI

    # 3D visualizer (Viser)
    visualizer = CapVisualizer(port=viser_port)
    logger.info(f"Viser 3D visualizer started on port {viser_port}")

    # Simulated robot for debug mode (lazy-initialized)
    _sim_robot: _SimulatedRobot | None = None

    def _get_sim_robot() -> _SimulatedRobot:
        nonlocal _sim_robot
        if _sim_robot is None:
            _sim_robot = _SimulatedRobot(visualizer, cancel_event=_exec_cancel)
            logger.info("Simulated robot initialized for debug mode")
        return _sim_robot

    # Tool registry
    registry: ToolRegistry = create_default_registry(
        cap_server_host=cap_server_host,
        cap_server_port=cap_server_port,
        detection_host=detection_host,
        detection_port=detection_port,
        policy_server=policy_server,
        bundlesdf_host=bundlesdf_host,
        bundlesdf_port=bundlesdf_port,
        sam3_host=sam3_host,
        sam3_port=sam3_port,
    )

    # Realtime detection: stop event shared between tool and Home/Stop/E-Stop
    _realtime_stop_event = threading.Event()

    # Executor cancel/pause events
    _exec_cancel = threading.Event()  # when set, executor aborts at next statement
    _exec_go = threading.Event()  # when clear, executor blocks (pause)
    _exec_go.set()  # initially: execution allowed

    # Track the currently running executor so E-Stop can force-cancel it
    _current_executor: Executor | None = None

    # LLM backend (lazy-loaded on first use)
    _llm = None

    def _get_llm():
        nonlocal _llm
        if _llm is None:
            if llm_backend == "claude":
                from enpire.env.forge.cap.agent.llm.cloud import CloudLLM

                _llm = CloudLLM(model=llm_model)
            else:
                raise ValueError(f"Unknown LLM backend: {llm_backend}")
        return _llm

    # Helpers ---

    _log_counter = 0

    def _make_log_id() -> str:
        nonlocal _log_counter
        _log_counter += 1
        return f"log-{_log_counter}"

    def _source_to_tool_name(source: str) -> str:
        """Extract a clean function/tool name from a Python source snippet."""
        if not source:
            return "exec"
        first_line = source.split("\n")[0].strip()
        # Find the last identifier before the first '(' — that's the function name
        before_paren = first_line.split("(")[0]
        words = before_paren.split()
        if words:
            return words[-1].rstrip(".")
        return first_line[:40] or "exec"

    def _should_log_action_entry(log: ExecutionLog) -> bool:
        if log.node_type in {"for", "function_def"}:
            return True
        src = (log.source or "").strip()
        if not src:
            return False
        first = src.split("\n", 1)[0].strip()
        if first.startswith(("import ", "from ", "#")):
            return False
        if first in {"pass", "break", "continue"}:
            return False
        # Log call-like statements (including assignment from a call).
        return "(" in first and ")" in first
    def _log_entry(
        tool: str,
        args: dict,
        result: Any = None,
        status: str = "success",
        error: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        entry_id: str | None = None,
        parent_id: str | None = None,
        node_type: str | None = None,
    ) -> dict[str, Any]:
        eid = entry_id or _make_log_id()
        entry: dict[str, Any] = {
            "id": eid,
            "tool": tool,
            "args": args,
            "result": str(error) if error else result,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if stdout is not None:
            entry["stdout"] = stdout
        if stderr is not None:
            entry["stderr"] = stderr
        if parent_id is not None:
            entry["parent_id"] = parent_id
        if node_type is not None:
            entry["node_type"] = node_type
        # Update existing entry if ID was reused, otherwise append
        for i, e in enumerate(state.action_log):
            if e["id"] == eid:
                state.action_log[i] = entry
                return entry
        state.action_log.append(entry)
        if len(state.action_log) > MAX_AGENT_ACTION_LOG_ENTRIES:
            del state.action_log[:-MAX_AGENT_ACTION_LOG_ENTRIES]
        return entry

    async def _broadcast_status() -> None:
        await manager.broadcast(
            "status_change",
            {
                "mode": state.mode,
                "status": state.status,
                "task": state.current_task,
                "debug_mode": state.debug_mode,
            },
        )

    async def _broadcast_error(
        message: str,
        *,
        source: str | None = None,
        detail: str | None = None,
    ) -> None:
        summary = message
        if detail:
            lines = [line.strip() for line in detail.splitlines() if line.strip()]
            if lines:
                summary = lines[-1]
        await manager.broadcast(
            "error",
            {
                "message": summary,
                "source": source,
                "detail": detail or message,
            },
        )

    def _reset_runtime_visual_state() -> None:
        state.action_log.clear()
        visualizer.clear_prediction()
        visualizer.clear_grasp_poses()
        visualizer.clear_scene_objects()
        visualizer.update_detections([])

    def _make_executor(loop: asyncio.AbstractEventLoop) -> Executor:
        callables = registry.callable_dict()
        tool_log_ids: dict[int, str] = {}

        def _project_world_to_pixel(
            pos_3d: list[float],
            cam_pos: np.ndarray,
            cam_rot: np.ndarray,
            intrinsics: list[float],
        ) -> list[float] | None:
            """Project a world-frame 3D point to pixel coordinates."""
            if not pos_3d or len(pos_3d) < 3:
                return None
            fx, fy, cx, cy = intrinsics
            p_world = np.asarray(pos_3d, dtype=np.float64)
            p_cam = np.linalg.inv(cam_rot) @ (p_world - cam_pos)
            p_cam[0] = -p_cam[0]
            p_cam[1] = -p_cam[1]
            if p_cam[2] <= 0:
                return None
            u = fx * p_cam[0] / p_cam[2] + cx
            v = fy * p_cam[1] / p_cam[2] + cy
            return [float(u), float(v)]

        def _project_oriented_bbox(
            pos_3d: list[float],
            quat_xyzw: list[float],
            cam_pos: np.ndarray,
            cam_rot: np.ndarray,
            intrinsics: list[float],
            half_extents: tuple[float, float, float] = (0.03, 0.03, 0.03),
        ) -> list[list[float]] | None:
            """Project an oriented 3D bounding box (8 corners) to 2D pixels."""
            from scipy.spatial.transform import Rotation

            center = np.asarray(pos_3d, dtype=np.float64)
            R = Rotation.from_quat(quat_xyzw).as_matrix()
            hx, hy, hz = half_extents
            # 8 corners of the box in object frame
            corners_local = np.array(
                [
                    [-hx, -hy, -hz],
                    [+hx, -hy, -hz],
                    [+hx, +hy, -hz],
                    [-hx, +hy, -hz],
                    [-hx, -hy, +hz],
                    [+hx, -hy, +hz],
                    [+hx, +hy, +hz],
                    [-hx, +hy, +hz],
                ]
            )
            corners_world = (R @ corners_local.T).T + center
            projected = []
            for c in corners_world:
                px = _project_world_to_pixel(c.tolist(), cam_pos, cam_rot, intrinsics)
                if px is None:
                    return None
                projected.append(px)
            return projected

        def _broadcast_detections(detections, camera: str) -> None:
            """Update Viser markers + broadcast debug image to UI."""
            det_dicts = [
                {
                    "label": d.label,
                    "score": d.score,
                    "position_3d": d.position_3d,
                    "quaternion_xyzw": d.quaternion_xyzw,
                }
                for d in detections
            ]
            visualizer.update_detections(det_dicts)

            try:
                # Use BundleSDF's rendered visualization if available (has mask + pose axes)
                _bsdf_vis = None
                for d in detections:
                    if getattr(d, "vis_b64", None):
                        _bsdf_vis = d.vis_b64
                        break

                cam_result = registry.call("get_camera_image", camera=camera)
                if cam_result.success and cam_result.data is not None:
                    img = np.asarray(cam_result.data)
                    if img.size > 100:
                        h, w = img.shape[:2]

                        # Get camera params for 3D→2D projection
                        detect_tool = registry.get("detect_object")
                        cam_pos = cam_rot = intrinsics = None
                        if detect_tool is not None:
                            try:
                                client = detect_tool._get_portal_client()
                                extr = client.get_camera_extrinsics(camera).result()
                                cam_pos = np.asarray(extr["position"], dtype=np.float64)
                                cam_rot = np.asarray(
                                    extr["rotation"], dtype=np.float64
                                ).reshape(3, 3)
                                intr_raw = client.get_camera_intrinsics(camera).result()
                                intrinsics = [float(x) for x in intr_raw]
                            except Exception:
                                pass

                        ui_dets = []
                        for d in detections:
                            entry: dict = {
                                "label": d.label,
                                "confidence": d.score,
                                "bbox": d.box_2d,
                                "position_3d": d.position_3d,
                                "quaternion_xyzw": d.quaternion_xyzw,
                            }
                            # Project 3D pose axes + oriented bbox for BundleSDF-style viz
                            if (
                                d.position_3d
                                and d.quaternion_xyzw
                                and cam_pos is not None
                            ):
                                from scipy.spatial.transform import Rotation as _Rot

                                _axis_len = 0.06  # 6cm axes (same as BundleSDF debug UI)
                                _center = np.asarray(d.position_3d, dtype=np.float64)
                                _R = _Rot.from_quat(d.quaternion_xyzw).as_matrix()
                                _origin_px = _project_world_to_pixel(
                                    _center.tolist(), cam_pos, cam_rot, intrinsics
                                )
                                if _origin_px is not None:
                                    _axes = {}
                                    _all_ok = True
                                    for _ai, _aname in enumerate(("x", "y", "z")):
                                        _tip = _center + _axis_len * _R[:, _ai]
                                        _tip_px = _project_world_to_pixel(
                                            _tip.tolist(), cam_pos, cam_rot, intrinsics
                                        )
                                        if _tip_px is None:
                                            _all_ok = False
                                            break
                                        _axes[_aname] = _tip_px
                                    if _all_ok:
                                        entry["pose_axes"] = {
                                            "origin": _origin_px,
                                            **_axes,
                                        }

                                # Also project oriented bbox
                                if not d.box_2d:
                                    he = (
                                        tuple(d.half_extents)
                                        if d.half_extents and len(d.half_extents) == 3
                                        else (0.03, 0.03, 0.03)
                                    )
                                    corners = _project_oriented_bbox(
                                        d.position_3d,
                                        d.quaternion_xyzw,
                                        cam_pos,
                                        cam_rot,
                                        intrinsics,
                                        half_extents=he,
                                    )
                                    if corners is not None:
                                        entry["bbox_3d_projected"] = corners
                            ui_dets.append(entry)

                        # Render BundleSDF-style visualization locally:
                        #   green SAM3 mask overlay + contours + RGB pose axes + label
                        vis_img = img.copy()
                        try:
                            import cv2 as _cv2
                            from scipy.spatial.transform import Rotation as _Rot

                            fx, fy, cx, cy = (intrinsics or [0, 0, 0, 0])
                            _K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                            _label_y = 24

                            for d in detections:
                                # 1) SAM3 mask overlay (call SAM3 server to get segmentation)
                                try:
                                    _seg_result = registry.call("segment_object", query=d.label, camera=camera)
                                    if _seg_result.success and _seg_result.data is not None:
                                        _mask = np.asarray(_seg_result.data.mask, dtype=np.uint8)
                                        if _mask.shape[:2] == vis_img.shape[:2]:
                                            # Green mask overlay (alpha blend)
                                            _green = np.zeros_like(vis_img)
                                            _green[_mask > 0] = (0, 255, 0)
                                            _cv2.addWeighted(_green, 0.35, vis_img, 1.0, 0, vis_img)
                                            # Green contours
                                            _contours, _ = _cv2.findContours(_mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                                            _cv2.drawContours(vis_img, _contours, -1, (0, 255, 0), 2)
                                except Exception:
                                    pass  # mask is best-effort

                                # 2) Pose axes (X=red, Y=green, Z=blue)
                                if (
                                    d.position_3d and d.quaternion_xyzw
                                    and len(d.quaternion_xyzw) == 4
                                    and cam_pos is not None and fx > 0
                                ):
                                    _R = _Rot.from_quat(d.quaternion_xyzw).as_matrix()
                                    _t = np.asarray(d.position_3d, dtype=np.float64)
                                    # World → camera frame → OpenCV frame
                                    _R_cam = np.linalg.inv(cam_rot) @ _R
                                    _t_cam = np.linalg.inv(cam_rot) @ (_t - cam_pos)
                                    _F = np.diag([-1.0, -1.0, 1.0])
                                    _ob_in_cam = np.eye(4, dtype=np.float64)
                                    _ob_in_cam[:3, :3] = _F @ _R_cam
                                    _ob_in_cam[:3, 3] = _F @ _t_cam

                                    def _proj(pt_h):
                                        p = (_ob_in_cam @ pt_h)[:3]
                                        px = _K @ p
                                        if px[2] > 0:
                                            return int(px[0] / px[2]), int(px[1] / px[2])
                                        return None

                                    _o = _proj(np.array([0, 0, 0, 1.0]))
                                    if _o is not None:
                                        _axis_len = 0.05
                                        for _ai, (_aname, _color) in enumerate([
                                            ("X", (0, 0, 255)), ("Y", (0, 255, 0)), ("Z", (255, 0, 0))
                                        ]):
                                            _tip_h = np.zeros(4)
                                            _tip_h[_ai] = _axis_len
                                            _tip_h[3] = 1.0
                                            _tip = _proj(_tip_h)
                                            if _tip is not None:
                                                _cv2.arrowedLine(vis_img, _o, _tip, _color, 3, tipLength=0.2)
                                        _cv2.circle(vis_img, _o, 5, (255, 255, 255), -1)
                                        _cv2.circle(vis_img, _o, 5, (0, 0, 0), 1)

                                # 3) Label + position
                                _score_pct = f"{d.score * 100:.0f}%" if d.score else ""
                                _pos_str = ""
                                if d.position_3d:
                                    _p = d.position_3d
                                    _pos_str = f"  [{_p[0]:+.3f} {_p[1]:+.3f} {_p[2]:+.3f}]m"
                                _lbl = f"{d.label} {_score_pct}{_pos_str}"
                                _cv2.putText(vis_img, _lbl, (8, _label_y), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                                _label_y += 28
                        except Exception as _ve:
                            logger.debug(f"BundleSDF-style rendering failed: {_ve}")

                        debug_data = {
                            "image": _encode_rgb_jpeg(vis_img),
                            "camera": camera,
                            "imageWidth": w,
                            "imageHeight": h,
                            "detections": ui_dets,
                        }
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast("detection_debug", debug_data), loop
                        )
            except Exception as e:
                logger.debug(f"Detection debug broadcast failed: {e}")

        def _broadcast_segmentation(
            seg_result,
            camera: str,
            query: str,
            hu_dist: float | None = None,
            passed: bool | None = None,
            is_reference: bool = False,
            media: str = "",
        ) -> None:
            """Broadcast segmentation mask overlay to UI."""
            try:
                cam_result = registry.call("get_camera_image", camera=camera)
                if cam_result.success and cam_result.data is not None:
                    img = np.asarray(cam_result.data)
                    if img.size > 100:
                        h, w = img.shape[:2]
                        from PIL import Image

                        mask = seg_result.mask
                        # Color: green=pass, red=fail, cyan=reference/unknown
                        if is_reference:
                            color = [0, 150, 255, 100]  # blue for reference
                        elif passed is True:
                            color = [0, 220, 0, 100]  # green for pass
                        elif passed is False:
                            color = [220, 0, 0, 100]  # red for fail
                        else:
                            color = [0, 220, 220, 100]  # cyan for unknown
                        overlay = np.zeros(
                            (mask.shape[0], mask.shape[1], 4), dtype=np.uint8
                        )
                        overlay[mask > 0] = color
                        overlay_img = Image.fromarray(overlay, "RGBA")
                        buf = io.BytesIO()
                        overlay_img.save(buf, format="PNG")
                        mask_png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                        debug_data = {
                            "image": _encode_rgb_jpeg(img),
                            "camera": camera,
                            "imageWidth": w,
                            "imageHeight": h,
                            "mask_overlay": mask_png_b64,
                            "query": query,
                            "score": seg_result.score,
                            "mask_area": seg_result.mask_area,
                            "bbox_xywh": seg_result.bbox_xywh,
                            "hu_dist": hu_dist,
                            "passed": passed,
                            "is_reference": is_reference,
                            "media": media,
                        }
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast("segmentation_debug", debug_data), loop
                        )
            except Exception as e:
                logger.debug(f"Segmentation debug broadcast failed: {e}")

        # Wrap segment_object to broadcast mask overlay to UI
        _orig_segment = callables.get("segment_object")
        if _orig_segment is not None:

            def _segment_and_visualize(*args, **kwargs):
                camera = "top"
                if len(args) > 1:
                    camera = args[1]
                elif "camera" in kwargs:
                    camera = kwargs["camera"]
                query = args[0] if args else kwargs.get("query", "")

                result = _orig_segment(*args, **kwargs)
                if result is not None:
                    _broadcast_segmentation(result, camera, query)
                return result

            _segment_and_visualize.__name__ = "segment_object"
            _segment_and_visualize.__doc__ = _orig_segment.__doc__
            callables["segment_object"] = _segment_and_visualize

        def _mark_segmentation(
            seg_result,
            camera="top",
            query="",
            hu_dist=None,
            passed=None,
            is_reference=False,
            media="",
        ):
            """Update segmentation visualization with shape distance and pass/fail status."""
            _broadcast_segmentation(
                seg_result,
                camera,
                query,
                hu_dist=hu_dist,
                passed=passed,
                is_reference=is_reference,
                media=media,
            )

        _mark_segmentation.__name__ = "mark_segmentation"
        _mark_segmentation.__doc__ = (
            "Update segmentation visualization with IoU and pass/fail status."
        )
        callables["mark_segmentation"] = _mark_segmentation

        def _mark_contact_detect(
            side: str = "left",
            success: bool | None = None,
            gripper_width: float | None = None,
            object_name: str = "",
            message: str = "",
            threshold_low: float = CONTACT_DETECT_THRESHOLD_LOW,
            threshold_high: float = CONTACT_DETECT_THRESHOLD_HIGH,
            status: str | None = None,
        ) -> None:
            """Broadcast grasp contact detection status to Skill Vis."""
            try:
                width = gripper_width
                if width is None:
                    get_state = callables.get("get_robot_state")
                    if get_state is not None:
                        st = get_state()
                        if side == "right":
                            width = float(getattr(st, "right_gripper_pos", 0.0))
                        else:
                            width = float(getattr(st, "left_gripper_pos", 0.0))
                payload = {
                    "side": side,
                    "success": success,
                    "status": status or ("success" if success else "failed"),
                    "gripper_width": width,
                    "object_name": object_name,
                    "message": message,
                    "threshold_low": threshold_low,
                    "threshold_high": threshold_high,
                }
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast("contact_detect", payload), loop
                )
            except Exception as e:
                logger.debug(f"Contact detect broadcast failed: {e}")

        _mark_contact_detect.__name__ = "mark_contact_detect"
        _mark_contact_detect.__doc__ = (
            "Update Skill Vis with grasp contact success/failure and gripper width."
        )
        callables["mark_contact_detect"] = _mark_contact_detect

        from enpire.env.forge.cap.agent.tools.segmentation import match_mask_shape

        callables["match_mask_shape"] = match_mask_shape

        # Wrap detect_object to update 3D visualizer + broadcast debug image
        _orig_detect = callables.get("detect_object")
        if _orig_detect is not None:

            def _detect_and_visualize(*args, **kwargs):
                camera = "top"
                if len(args) > 1:
                    camera = args[1]
                elif "camera" in kwargs:
                    camera = kwargs["camera"]

                result = _orig_detect(*args, **kwargs)
                if result is not None:
                    _broadcast_detections(result, camera)
                return result

            _detect_and_visualize.__name__ = "detect_object"
            _detect_and_visualize.__doc__ = _orig_detect.__doc__
            callables["detect_object"] = _detect_and_visualize

        _orig_detect_oneshot = callables.get("detect_objects_oneshot")
        if _orig_detect_oneshot is not None:

            def _detect_oneshot_and_visualize(*args, **kwargs):
                camera = "top"
                if len(args) > 1:
                    camera = args[1]
                elif "camera" in kwargs:
                    camera = kwargs["camera"]
                try:
                    result = _orig_detect_oneshot(*args, **kwargs)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(
                        _broadcast_error(
                            str(e), source="detect_objects_oneshot", detail=str(e)
                        ),
                        loop,
                    )
                    raise
                try:
                    if isinstance(result, dict):
                        merged = []
                        for dets in result.values():
                            if isinstance(dets, list):
                                merged.extend(dets)
                        if merged:
                            _broadcast_detections(merged, camera)
                except Exception as e:
                    logger.debug(f"detect_objects_oneshot broadcast failed: {e}")
                return result

            _detect_oneshot_and_visualize.__name__ = "detect_objects_oneshot"
            _detect_oneshot_and_visualize.__doc__ = _orig_detect_oneshot.__doc__
            callables["detect_objects_oneshot"] = _detect_oneshot_and_visualize

        # Wrap BundleSDF object tracking tools to reuse existing Skill Vis slots:
        #   - segmentation tab shows the SAM3 initialization mask
        #   - detection tab shows the tracked 6-DoF pose / projected box
        _orig_track_object = callables.get("track_object")
        if _orig_track_object is not None:

            def _track_object_and_visualize(*args, **kwargs):
                query = args[0] if args else kwargs.get("query", "")
                camera = "top"
                if len(args) > 1:
                    camera = args[1]
                elif "camera" in kwargs:
                    camera = kwargs["camera"]

                result = _orig_track_object(*args, **kwargs)

                # Reuse the existing segmentation UI slot so we can see the
                # SAM3 initialization mask used to bootstrap BundleSDF/SAM2.
                try:
                    segment_fn = callables.get("segment_object")
                    if segment_fn is not None and query:
                        segment_fn(query=query, media=f"camera:{camera}")
                except Exception as e:
                    logger.debug(f"track_object segmentation broadcast failed: {e}")

                return result

            _track_object_and_visualize.__name__ = "track_object"
            _track_object_and_visualize.__doc__ = _orig_track_object.__doc__
            callables["track_object"] = _track_object_and_visualize

        _orig_get_object_pose = callables.get("get_object_pose")
        if _orig_get_object_pose is not None:

            def _get_object_pose_and_visualize(*args, **kwargs):
                result = _orig_get_object_pose(*args, **kwargs)
                try:
                    pose_tool = registry.get("get_object_pose")
                    camera = getattr(getattr(pose_tool, "_ctx", None), "camera", "top")
                    _broadcast_detections([result], camera)
                except Exception as e:
                    logger.debug(f"get_object_pose detection broadcast failed: {e}")
                return result

            _get_object_pose_and_visualize.__name__ = "get_object_pose"
            _get_object_pose_and_visualize.__doc__ = _orig_get_object_pose.__doc__
            callables["get_object_pose"] = _get_object_pose_and_visualize

        _latest_grasp_viz_payload: dict[str, Any] | None = None

        def _angle_diff_deg(a: float, b: float) -> float:
            diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
            return abs(diff)

        def _match_grasp_row_for_motion(
            payload: dict[str, Any] | None,
            *,
            side: str | None,
            target_pos: list[float] | None,
            target_rpy: list[float] | None,
        ) -> int | None:
            if not payload:
                return None
            rows = payload.get("grasps")
            if not isinstance(rows, list) or not rows or target_pos is None:
                return None

            target_pos_np = np.asarray(target_pos, dtype=np.float64)
            target_rpy_list = (
                [float(x) for x in target_rpy] if target_rpy is not None else None
            )
            best_idx = None
            best_key = None
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                row_pos = row.get("planner_xyz")
                if not isinstance(row_pos, list) or len(row_pos) != 3:
                    continue
                try:
                    pos_err = float(
                        np.linalg.norm(
                            np.asarray(row_pos, dtype=np.float64) - target_pos_np
                        )
                    )
                except Exception:
                    continue
                row_rpy = row.get("planner_rpy")
                if (
                    target_rpy_list is not None
                    and isinstance(row_rpy, list)
                    and len(row_rpy) == 3
                ):
                    rot_err = sum(
                        _angle_diff_deg(a, b) for a, b in zip(row_rpy, target_rpy_list)
                    )
                else:
                    rot_err = 0.0
                key = (pos_err, rot_err, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            return best_idx

        def _broadcast_grasp_row_motion_update(
            payload: dict[str, Any] | None,
            *,
            side: str | None,
            target_pos: list[float] | None,
            target_rpy: list[float] | None,
            motion_payload: dict[str, Any],
        ) -> None:
            nonlocal _latest_grasp_viz_payload
            match_idx = _match_grasp_row_for_motion(
                payload,
                side=side,
                target_pos=target_pos,
                target_rpy=target_rpy,
            )
            if match_idx is None or payload is None:
                return

            rows = payload.get("grasps")
            if not isinstance(rows, list):
                return
            updated_rows = [dict(r) if isinstance(r, dict) else r for r in rows]
            row = dict(updated_rows[match_idx])
            row["motionplanner_side"] = side
            row["motionplanner_status"] = motion_payload.get("status")
            row["motionplanner_preview_only"] = bool(
                motion_payload.get("preview_only", False)
            )
            row["motionplanner_reason"] = motion_payload.get(
                "reason"
            ) or motion_payload.get("error")
            row["final_pos_error_m"] = motion_payload.get("final_pos_error_m")
            row["final_rot_error_deg"] = motion_payload.get("final_rot_error_deg")
            row["final_pose_error"] = motion_payload.get("final_pose_error")
            row["trajectory_steps"] = motion_payload.get("trajectory_steps")
            row["side"] = side
            if bool(motion_payload.get("preview_only", False)):
                row["status"] = (
                    "preview_ok"
                    if not motion_payload.get("error")
                    else "planner_preview_failed"
                )
                row["status_reason"] = (
                    "Planner preview succeeded."
                    if not motion_payload.get("error")
                    else str(
                        motion_payload.get("error") or motion_payload.get("reason")
                    )
                )
            else:
                row["status"] = (
                    "executed" if not motion_payload.get("error") else "execute_failed"
                )
                row["selected"] = not bool(motion_payload.get("error"))
                row["status_reason"] = (
                    "Motion planner executed this grasp."
                    if not motion_payload.get("error")
                    else str(
                        motion_payload.get("error") or motion_payload.get("reason")
                    )
                )
            updated_rows[match_idx] = row

            updated_payload = dict(payload)
            updated_payload["grasps"] = updated_rows
            updated_payload["selected_side"] = side
            updated_payload["selected_rank"] = row.get("rank")
            updated_payload["selection_status"] = (
                "executed"
                if not motion_payload.get("preview_only", False)
                else "previewed"
            )
            _latest_grasp_viz_payload = updated_payload
            asyncio.run_coroutine_threadsafe(
                manager.broadcast("grasp_viz", updated_payload),
                loop,
            )

        def _wrap_grasp_callable(tool_name: str) -> None:
            _orig_grasp = callables.get(tool_name)
            if _orig_grasp is None:
                return

            def _grasp_and_visualize(*args, **kwargs):
                nonlocal _latest_grasp_viz_payload
                camera = "top"
                if len(args) > 1:
                    camera = args[1]
                elif "camera" in kwargs:
                    camera = kwargs["camera"]

                result = _orig_grasp(*args, **kwargs)

                try:
                    from scipy.spatial.transform import Rotation as _GR

                    grasp_tool = registry.get(tool_name)
                    cached_rgb = (
                        getattr(grasp_tool, "last_rgb", None) if grasp_tool else None
                    )
                    cached_mask = (
                        getattr(grasp_tool, "last_mask", None) if grasp_tool else None
                    )

                    img = cached_rgb
                    if img is None:
                        cam_result = registry.call("get_camera_image", camera=camera)
                        if cam_result.success and cam_result.data is not None:
                            img = np.asarray(cam_result.data)

                    if img is not None and img.size > 100:
                        h, w = img.shape[:2]

                        has_grasps = result is not None and len(result) > 0
                        if cached_mask is not None and cached_mask.shape[:2] == (h, w):
                            overlay = img.copy()
                            mask_bool = cached_mask > 0
                            tint = (
                                np.array([0, 180, 0])
                                if has_grasps
                                else np.array([180, 0, 0])
                            )
                            overlay[mask_bool] = (
                                overlay[mask_bool] * 0.6 + tint * 0.4
                            ).astype(np.uint8)
                            img = overlay

                        detect_tool = registry.get("detect_object")
                        cam_pos = cam_rot = intrinsics = None
                        if detect_tool is not None:
                            try:
                                client = detect_tool._get_portal_client()
                                extr = client.get_camera_extrinsics(camera).result()
                                cam_pos = np.asarray(extr["position"], dtype=np.float64)
                                cam_rot = np.asarray(
                                    extr["rotation"], dtype=np.float64
                                ).reshape(3, 3)
                                intr_raw = client.get_camera_intrinsics(camera).result()
                                intrinsics = [float(x) for x in intr_raw]
                            except Exception:
                                pass

                        ui_dets = []
                        if has_grasps:
                            for g in result:
                                quat_xyzw = (
                                    _GR.from_euler("xyz", g.rpy, degrees=True)
                                    .as_quat()
                                    .tolist()
                                )
                                entry: dict = {
                                    "label": f"grasp ({g.score:.0%})",
                                    "confidence": g.score,
                                    "bbox": [],
                                    "position_3d": g.position,
                                    "quaternion_xyzw": quat_xyzw,
                                    "rpy": g.rpy,
                                    "is_grasp": True,
                                }
                                if cam_pos is not None and intrinsics is not None:
                                    origin_px = _project_world_to_pixel(
                                        g.position, cam_pos, cam_rot, intrinsics
                                    )
                                    R = _GR.from_euler(
                                        "xyz", g.rpy, degrees=True
                                    ).as_matrix()
                                    axis_len = VISER_AXIS_LEN_M
                                    pos = np.array(g.position)
                                    axes_px = []
                                    for col in range(3):
                                        tip = (pos + axis_len * R[:, col]).tolist()
                                        axes_px.append(
                                            _project_world_to_pixel(
                                                tip, cam_pos, cam_rot, intrinsics
                                            )
                                        )
                                    if origin_px and all(axes_px):
                                        entry["grasp_axes"] = {
                                            "origin": origin_px,
                                            "x": axes_px[0],
                                            "y": axes_px[1],
                                            "z": axes_px[2],
                                        }
                                ui_dets.append(entry)
                        else:
                            ui_dets.append(
                                {
                                    "label": "NO GRASPS FOUND",
                                    "confidence": 0.0,
                                    "bbox": [],
                                    "position_3d": None,
                                    "is_grasp": False,
                                }
                            )

                        debug_data = {
                            "image": _encode_rgb_jpeg(img),
                            "camera": camera,
                            "imageWidth": w,
                            "imageHeight": h,
                            "detections": ui_dets,
                        }
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast("detection_debug", debug_data), loop
                        )

                        if has_grasps:
                            visualizer.show_grasp_poses(result)

                        grasp_debug = (
                            getattr(grasp_tool, "last_grasp_debug", None) or {}
                        )
                        overlay_jpeg = getattr(grasp_tool, "last_overlay_jpeg", None)
                        native_png = getattr(grasp_tool, "last_native_viz_png", None)
                        if overlay_jpeg is not None or native_png is not None:
                            import base64 as _b64

                            payload = dict(grasp_debug)
                            if overlay_jpeg is not None:
                                payload["overlay_b64"] = _b64.b64encode(
                                    overlay_jpeg
                                ).decode()
                            if native_png is not None:
                                payload["native_viz_b64"] = _b64.b64encode(
                                    native_png
                                ).decode()
                            try:
                                logger.info(
                                    "[grasp_debug] broadcasting grasp_viz backend=%s object=%s n_grasps=%s grasp_rows=%s keys=%s",
                                    payload.get("backend"),
                                    payload.get("object_name"),
                                    payload.get("n_grasps"),
                                    len(payload.get("grasps", []) or []),
                                    sorted(payload.keys()),
                                )
                            except Exception:
                                pass
                            if isinstance(payload.get("grasps"), list):
                                _latest_grasp_viz_payload = payload
                            asyncio.run_coroutine_threadsafe(
                                manager.broadcast("grasp_viz", payload),
                                loop,
                            )
                except Exception as e:
                    logger.debug(f"Grasp broadcast failed ({tool_name}): {e}")
                return result

            _grasp_and_visualize.__name__ = tool_name
            _grasp_and_visualize.__doc__ = _orig_grasp.__doc__
            callables[tool_name] = _grasp_and_visualize

        _wrap_grasp_callable("sample_grasp_pose_anygrasp")
        _wrap_grasp_callable("sample_grasp_pose_2d")
        _wrap_grasp_callable("sample_grasp_pose_3d_bb")

        # Wrap freespace_move to broadcast planner diagnostics to Skill Vis
        motion_tool = registry.get("freespace_move")
        if motion_tool is not None:
            _motion_param_names = [p.name for p in motion_tool.parameters]

            def _freespace_and_broadcast(*args, **kwargs):
                import dataclasses as _dc

                kw = dict(kwargs)
                for i, val in enumerate(args):
                    if i < len(_motion_param_names):
                        kw[_motion_param_names[i]] = val

                result = motion_tool.execute(**kw)
                try:
                    data = result.data
                    payload = {
                        "tool": "freespace_move",
                        "planner_backend": kw.get("planner_backend", "curobo"),
                        "side": (
                            "both"
                            if (
                                kw.get("left_target_pos") is not None
                                or kw.get("left_target_rpy") is not None
                            )
                            and (
                                kw.get("right_target_pos") is not None
                                or kw.get("right_target_rpy") is not None
                            )
                            else (
                                "left"
                                if kw.get("left_target_pos") is not None
                                or kw.get("left_target_rpy") is not None
                                else "right"
                            )
                        ),
                        "preview_only": bool(kw.get("preview_only", False)),
                        "planning_speed": kw.get("planning_speed"),
                        "left_target_pos": kw.get("left_target_pos"),
                        "left_target_rpy": kw.get("left_target_rpy"),
                        "right_target_pos": kw.get("right_target_pos"),
                        "right_target_rpy": kw.get("right_target_rpy"),
                        "error": result.error,
                    }
                    if data is not None:
                        if _dc.is_dataclass(data):
                            payload.update(_dc.asdict(data))
                        elif isinstance(data, dict):
                            payload.update(data)
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast("motion_planner_debug", payload),
                        loop,
                    )
                    if result.error:
                        planner_backend = str(
                            payload.get(
                                "planner_backend", kw.get("planner_backend", "curobo")
                            )
                        )
                        asyncio.run_coroutine_threadsafe(
                            _broadcast_error(
                                result.error,
                                source=f"freespace_move[{planner_backend}]",
                                detail=result.error,
                            ),
                            loop,
                        )
                    target_side = payload.get("side")
                    target_pos = (
                        kw.get("left_target_pos")
                        if target_side == "left"
                        else kw.get("right_target_pos")
                        if target_side == "right"
                        else None
                    )
                    target_rpy = (
                        kw.get("left_target_rpy")
                        if target_side == "left"
                        else kw.get("right_target_rpy")
                        if target_side == "right"
                        else None
                    )
                    _broadcast_grasp_row_motion_update(
                        _latest_grasp_viz_payload,
                        side=target_side,
                        target_pos=target_pos,
                        target_rpy=target_rpy,
                        motion_payload=payload,
                    )
                except Exception as e:
                    logger.debug(f"Motion planner broadcast failed: {e}")

                if not result.success:
                    raise RuntimeError(
                        f"Tool {motion_tool.name} failed: {result.error}"
                    )
                return result.data

            _freespace_and_broadcast.__name__ = "freespace_move"
            _freespace_and_broadcast.__doc__ = motion_tool.description
            callables["freespace_move"] = _freespace_and_broadcast

        # Register detect_object_realtime — continuous detection loop
        from enpire.env.forge.cap.agent.tools.detection import DetectObjectRealtimeTool

        detect_tool = registry.get("detect_object")
        if detect_tool is not None:
            realtime_tool = DetectObjectRealtimeTool(
                detect_tool=detect_tool,
                stop_event=_realtime_stop_event,
                on_detections=_broadcast_detections,
            )
            # Build a callable that matches the tool's parameter schema
            param_names = [p.name for p in realtime_tool.parameters]

            def _realtime_fn(*args, **kw):
                for i, val in enumerate(args):
                    if i < len(param_names):
                        kw[param_names[i]] = val
                _realtime_stop_event.clear()
                result = realtime_tool.execute(**kw)
                if not result.success:
                    raise RuntimeError(f"detect_object_realtime failed: {result.error}")
                return result.data

            _realtime_fn.__name__ = "detect_object_realtime"
            _realtime_fn.__doc__ = realtime_tool.description
            callables["detect_object_realtime"] = _realtime_fn

        # Wrap vlm_query to broadcast prompt+response to UI
        _orig_vlm_query = callables.get("vlm_query")
        if _orig_vlm_query is not None:

            def _vlm_query_and_broadcast(*args, **kwargs):
                result = _orig_vlm_query(*args, **kwargs)
                try:
                    # Extract prompt/backend/camera from args
                    param_names = ["text", "backend", "camera", "image", "model"]
                    kw = dict(kwargs)
                    for i, val in enumerate(args):
                        if i < len(param_names):
                            kw[param_names[i]] = val
                    vlm_data = {
                        "prompt": kw.get("text", ""),
                        "backend": kw.get("backend", DEFAULT_VLM_BACKEND),
                        "camera": kw.get("camera", "top"),
                        "model": kw.get("model"),
                        "response": result if isinstance(result, str) else str(result),
                    }
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast("vlm_result", vlm_data), loop
                    )
                except Exception as e:
                    logger.debug(f"VLM result broadcast failed: {e}")
                return result

            _vlm_query_and_broadcast.__name__ = "vlm_query"
            _vlm_query_and_broadcast.__doc__ = _orig_vlm_query.__doc__
            callables["vlm_query"] = _vlm_query_and_broadcast

        def _publish_grasp_viz(payload: dict | None = None):
            payload = dict(payload or {})
            asyncio.run_coroutine_threadsafe(
                manager.broadcast("grasp_viz", payload),
                loop,
            )
            return payload

        _publish_grasp_viz.__name__ = "publish_grasp_viz"
        _publish_grasp_viz.__doc__ = (
            "Broadcast partial updates to the Skill Vis grasp debug panel. "
            "Useful for attaching per-grasp planner status during custom scripts."
        )
        callables["publish_grasp_viz"] = _publish_grasp_viz

        def _tool_label(log: ExecutionLog) -> str:
            """Return a human-readable label for a log entry."""
            if log.node_type == "for":
                first_line = log.source.split("\n")[0].rstrip(":").strip()
                return first_line[:60] or "for loop"
            if log.node_type == "function_def":
                first_line = log.source.split("\n")[0].strip()
                return first_line[:60] or "def"
            return _source_to_tool_name(log.source)

        def _tool_args_preview(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            preview: dict[str, Any] = {}
            if args:
                preview["args"] = [repr(a)[:120] for a in args[:4]]
            if kwargs:
                preview["kwargs"] = {
                    k: (repr(v)[:120] if not isinstance(v, (int, float, bool, str, type(None))) else v)
                    for k, v in list(kwargs.items())[:8]
                }
            return preview

        def _on_tool_start(*, name: str, call_id: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            entry_id = _make_log_id()
            tool_log_ids[call_id] = entry_id
            entry = _log_entry(
                tool=name,
                args=_tool_args_preview(args, kwargs),
                result=None,
                status="running",
                entry_id=entry_id,
            )
            asyncio.run_coroutine_threadsafe(
                manager.broadcast("execution_log", entry), loop
            )

        def _on_tool_end(*, name: str, call_id: int, result: Any, error: Exception | None, elapsed_ms: float) -> None:
            entry_id = tool_log_ids.pop(call_id, None)
            entry = _log_entry(
                tool=name,
                args={},
                result=(None if error else f"{elapsed_ms:.1f}ms"),
                status=("error" if error else "success"),
                error=(str(error) if error else None),
                entry_id=entry_id,
            )
            asyncio.run_coroutine_threadsafe(
                manager.broadcast("execution_log", entry), loop
            )

        def on_start(log: ExecutionLog) -> None:
            """Called before each statement executes."""
            pass

        def on_log(log: ExecutionLog) -> None:
            """Called after each statement completes."""
            pass

        def on_stdout(text: str) -> None:
            if not text:
                return
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(
                    "stdout_stream",
                    {
                        "text": text,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                ),
                loop,
            )

        from enpire.env.forge.cap.agent.profiler import set_tool_event_hooks
        set_tool_event_hooks(_on_tool_start, _on_tool_end)

        # --- Debug mode: swap motion tool callables with simulated versions ---
        if state.debug_mode:
            sim = _get_sim_robot()

            def _sim_get_robot_state():
                return sim.get_state()

            _sim_get_robot_state.__name__ = "get_robot_state"
            _sim_get_robot_state.__doc__ = callables.get(
                "get_robot_state", lambda: None
            ).__doc__
            callables["get_robot_state"] = _sim_get_robot_state

            def _sim_ik_servo(side, pos, quat, gripper=None, ignore_safety=False):
                return sim._ik_servo(side, pos, quat, gripper, ignore_safety)

            _sim_ik_servo.__name__ = "_ik_servo"
            _sim_ik_servo.__doc__ = callables.get("_ik_servo", lambda: None).__doc__
            callables["_ik_servo"] = _sim_ik_servo

            def _sim_open_gripper(side, vel_limit=None):
                return sim.open_gripper(side, vel_limit)

            _sim_open_gripper.__name__ = "open_gripper"
            _sim_open_gripper.__doc__ = callables.get(
                "open_gripper", lambda: None
            ).__doc__
            callables["open_gripper"] = _sim_open_gripper

            def _sim_close_gripper(side, vel_limit=None):
                return sim.close_gripper(side, vel_limit)

            _sim_close_gripper.__name__ = "close_gripper"
            _sim_close_gripper.__doc__ = callables.get(
                "close_gripper", lambda: None
            ).__doc__
            callables["close_gripper"] = _sim_close_gripper

            def _sim_go_home():
                return sim.go_home()

            _sim_go_home.__name__ = "go_home"
            _sim_go_home.__doc__ = callables.get("go_home", lambda: None).__doc__
            callables["go_home"] = _sim_go_home

            logger.info("Debug mode: using simulated robot callables")

        # Wire up robot callbacks for visualizer UI (grasp rotation tester)
        visualizer.set_robot_callbacks(
            ik_servo_fn=callables.get("_ik_servo"),
            get_state_fn=callables.get("get_robot_state"),
            sample_grasp_fn=callables.get("sample_grasp_pose_anygrasp"),
        )

        # Expose _ik_servo_traj and move_joint_traj as direct sandbox callables.
        # They accept a Python callable (func) which can't cross Portal RPC, so we
        # sample func locally, build numpy arrays, and forward to the keypoints
        # RPC methods. Portal serializes numpy arrays natively; nested Python lists
        # of floats cause serialization failures at large payload sizes.
        from enpire.env.forge.cap.config import (
            CONTROL_PERIOD_S as _CTRL_DT,
        )
        from enpire.env.forge.cap.config import (
            MOVE_EEF_MAX_VEL as _DEFAULT_VEL,
        )

        _kp_eef = callables.get("_ik_servo_keypoints")
        _kp_jnt = callables.get("move_joint_keypoints")

        def _ik_servo_traj(
            side, func, start_time=0.0, stop_time=1.0, max_vel=_DEFAULT_VEL
        ):
            """Track a continuous EEF trajectory defined by func(t) -> (pos, euler).
            pos=[x,y,z], euler=[roll,pitch,yaw] in radians (RPY).
            Samples at CONTROL_PERIOD_S intervals and forwards to _ik_servo_keypoints."""
            import numpy as _np

            n = max(2, round((stop_time - start_time) / _CTRL_DT) + 1)
            rel_times = _np.linspace(0.0, stop_time - start_time, n)
            kps = _np.empty((n, 6), dtype=_np.float64)
            for i, rel_t in enumerate(rel_times):
                pos, euler = func(start_time + float(rel_t))
                kps[i, :3] = _np.asarray(pos, dtype=_np.float64).ravel()
                kps[i, 3:] = _np.asarray(euler, dtype=_np.float64).ravel()[:3]
            return _kp_eef(side, rel_times, kps, max_vel)

        _ik_servo_traj.__name__ = "_ik_servo_traj"
        callables["_ik_servo_traj"] = _ik_servo_traj

        def move_joint_traj(side, func, start_time=0.0, stop_time=1.0):
            """Execute a joint-space trajectory defined by func(t) -> joint_pos (6,).
            Samples at CONTROL_PERIOD_S intervals and forwards to move_joint_keypoints."""
            import numpy as _np

            n = max(2, round((stop_time - start_time) / _CTRL_DT) + 1)
            rel_times = _np.linspace(0.0, stop_time - start_time, n)
            jps = _np.empty((n, 6), dtype=_np.float64)
            for i, rel_t in enumerate(rel_times):
                jps[i] = _np.asarray(
                    func(start_time + float(rel_t)), dtype=_np.float64
                ).ravel()[:6]
            return _kp_jnt(side, rel_times, jps)

        move_joint_traj.__name__ = "move_joint_traj"
        callables["move_joint_traj"] = move_joint_traj

        # --- cuRobo motion planner API ---
        # Expose planner creation and collision world update as sandbox callables
        # so scripts can use cuRobo without importing experimental.* directly.
        # The planner connects to a remote cuRobo server via Portal RPC
        # (configured by CAP_CUROBO_HOST / CAP_CUROBO_PORT env vars).
        _planner_instance = None

        def create_motion_planner(solver_speed="slow", position_threshold=0.01, rotation_threshold=0.1):
            """Create a cuRobo motion planner (connects to remote cuRobo server).
            Returns a planner object with plan_to_pose() method.
            Reuses the same instance across calls.
            Configure remote server via CAP_CUROBO_HOST / CAP_CUROBO_PORT env vars."""
            nonlocal _planner_instance
            if _planner_instance is not None:
                return _planner_instance
            from enpire.env.forge.experimental.portal_motion_planner import PortalMotionPlanner
            raw_port = os.environ.get("CAP_CUROBO_PORT", "").strip()
            host = os.environ.get("CAP_CUROBO_HOST", "127.0.0.1").strip() or "127.0.0.1"
            start_server = raw_port == ""  # auto-start if no port specified
            port = int(raw_port) if raw_port else None
            robot_type = os.environ.get("CAP_ROBOT_TYPE", "yam").strip().lower()
            _planner_instance = PortalMotionPlanner(
                backend="curobo",
                solver_speed=solver_speed,
                host=host,
                port=port,
                start_server=start_server,
                robot_type=robot_type,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
            )
            return _planner_instance

        create_motion_planner.__name__ = "create_motion_planner"
        callables["create_motion_planner"] = create_motion_planner

        def update_planner_world(planner, exclude_body_prefixes=None):
            """Load collision geometry from the sim into the remote cuRobo planner.
            Fetches geom data from cap_server, sends to remote planner via RPC."""
            _client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            kwargs = {}
            if exclude_body_prefixes is not None:
                kwargs["exclude_body_prefixes"] = list(exclude_body_prefixes)
            collision_data = _client.get_collision_geoms(**kwargs).result()
            try:
                n = planner.update_world_from_geoms(collision_data)
            except Exception as e:
                print(f"[update_planner_world] WARNING: collision world update failed ({e}). Proceeding without collision avoidance.")
                n = 0
            base_pos = collision_data["base_pos"]
            base_quat = collision_data["base_quat_xyzw"]
            return {"base_pos": base_pos, "base_quat_xyzw": base_quat, "n_obstacles": n}

        update_planner_world.__name__ = "update_planner_world"
        callables["update_planner_world"] = update_planner_world

        # --- Camera intrinsics/extrinsics (needed for visualization overlays) ---
        def get_camera_intrinsics(camera="top"):
            """Get camera intrinsic parameters [fx, fy, cx, cy]."""
            _client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            return _client.get_camera_intrinsics(camera).result()

        get_camera_intrinsics.__name__ = "get_camera_intrinsics"
        callables["get_camera_intrinsics"] = get_camera_intrinsics

        def get_camera_extrinsics(camera="top"):
            """Get camera extrinsic parameters (rotation, position, needs_optical_flip)."""
            _client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            return _client.get_camera_extrinsics(camera).result()

        get_camera_extrinsics.__name__ = "get_camera_extrinsics"
        callables["get_camera_extrinsics"] = get_camera_extrinsics

        return Executor(
            tool_callables=callables,
            on_log=on_log,
            on_start=on_start,
            on_stdout=on_stdout,
            cancel_event=_exec_cancel,
            go_event=_exec_go,
        )

    # REST endpoints ---

    @app.post("/api/task", response_model=TaskResponse)
    async def submit_task(req: TaskRequest) -> TaskResponse:
        if state.status not in ("idle", "stopped"):
            return TaskResponse(task_id="")

        state.task_counter += 1
        task_id = f"task-{state.task_counter}"
        state.current_task = req.description
        state.status = "generating"
        await _broadcast_status()

        if state.mode == "agent":
            # Generate code via LLM
            try:
                llm = _get_llm()
                context = {
                    "tools": registry.schemas(),
                    "history": state.action_log[-10:],
                }
                code = llm.generate_code(req.description, context)
                state.proposed_code = code
                state.status = "awaiting_approval"
                await manager.broadcast(
                    "code_proposal", {"code": code, "task_id": task_id}
                )
                await _broadcast_status()
            except Exception as e:
                logger.exception("LLM generation failed")
                state.status = "idle"
                await manager.broadcast("error", {"message": str(e)})
                await _broadcast_status()
        else:
            # Oracle mode: just acknowledge, user will send code via /execute
            state.status = "idle"
            await _broadcast_status()

        return TaskResponse(task_id=task_id)

    async def _handle_execution_result(result: ExecutionResult) -> None:
        """Handle execution result, including wait_for_agent pauses and cancellation."""
        if result.cancelled:
            state.paused_namespace = None
            # Keep whatever status was set by the cancel trigger (stopped/idle)
            if state.status == "executing":
                state.status = "idle"
            _log_entry("execution", {}, status="cancelled", error="Execution cancelled")
            logger.info("Execution cancelled")
            await _broadcast_status()
        elif result.paused:
            # Execution paused by wait_for_agent — save namespace, trigger replan
            state.paused_namespace = result.user_namespace
            state.status = "waiting_for_agent"
            _log_entry(
                "wait_for_agent",
                {},
                result=result.pause_message,
                status="paused",
            )
            await _broadcast_status()
            await manager.broadcast(
                "agent_wait",
                {
                    "message": result.pause_message,
                    "stdout": result.stdout,
                    "logs": [
                        {"line": l.line, "source": l.source, "stdout": l.stdout}
                        for l in result.logs
                    ],
                },
            )
            # Auto-trigger replan via bridge
            await _trigger_replan(result.pause_message, result.stdout)
        else:
            state.paused_namespace = None
            state.status = "idle"
            status = "success" if result.success else "error"
            entry = _log_entry(
                "execution",
                {},
                result=result.stdout or None,
                status=status,
                error=result.error,
                stdout=result.stdout or None,
                stderr=result.stderr or None,
            )
            await manager.broadcast("execution_log", entry)
            if result.error:
                await _broadcast_error(
                    result.error, source="execution", detail=result.error
                )
            await _broadcast_status()

    async def _trigger_replan(pause_message: str, stdout: str) -> None:
        """Call the active agent bridge backend to replan after wait_for_agent."""
        import urllib.request

        bridge_url = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/api/chat"
        try:
            from enpire.env.forge.cap.prompt.loader import PromptMemory

            pm = PromptMemory()
            replan_prompt = pm.load(
                "system", "replan",
                pause_message=pause_message,
                stdout=stdout,
            )
        except (FileNotFoundError, Exception):
            replan_prompt = (
                f"[AGENT REPLAN] Code execution paused with wait_for_agent.\n\n"
                f"Message from code: {pause_message}\n\n"
                f"Execution output so far:\n{stdout}\n\n"
                f"Inspect the robot state and cameras, then write continuation "
                f"code. All variables from the previous execution are preserved "
                f"in the namespace. Output your code in a ```python block."
            )
        try:
            data = json.dumps({"message": replan_prompt}).encode("utf-8")
            req = urllib.request.Request(
                bridge_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=5),
            )
            logger.info("Replan triggered via bridge")
        except Exception as e:
            logger.warning("Failed to trigger replan via bridge: %s", e)

    @app.post("/api/approve", response_model=OkResponse)
    async def approve() -> OkResponse:
        if state.status != "awaiting_approval" or not state.proposed_code:
            return OkResponse(ok=False)

        # Reset cancel/pause events for fresh execution
        _exec_cancel.clear()
        _exec_go.set()

        code = state.proposed_code
        state.proposed_code = ""
        state.status = "executing"
        await _broadcast_status()

        loop = asyncio.get_running_loop()
        executor = _make_executor(loop)
        nonlocal _current_executor
        _current_executor = executor
        extra_ns = state.paused_namespace
        try:
            result: ExecutionResult = await loop.run_in_executor(
                None, executor.execute, code, extra_ns
            )
        finally:
            _current_executor = None

        await _handle_execution_result(result)
        return OkResponse(ok=result.success or result.paused)

    @app.post("/api/reject", response_model=OkResponse)
    async def reject(req: RejectRequest) -> OkResponse:
        if state.status != "awaiting_approval":
            return OkResponse(ok=False)
        state.proposed_code = ""
        state.status = "idle"
        _log_entry("reject", {"feedback": req.feedback or ""})
        await _broadcast_status()
        return OkResponse(ok=True)

    @app.post("/api/execute", response_model=OkResponse)
    async def execute_code(req: ExecuteRequest) -> OkResponse:
        if state.status not in ("idle", "stopped", "waiting_for_agent"):
            return OkResponse(ok=False)

        is_resume_from_wait = state.status == "waiting_for_agent"
        if not is_resume_from_wait:
            state.paused_namespace = None
            _reset_runtime_visual_state()
            if not state.debug_mode:
                try:
                    tool = registry.get("get_robot_state")
                    if tool is not None:
                        client = tool._get_client(tool._host, tool._port)
                        client.release_estop().result()
                except Exception as e:
                    logger.warning(f"Failed to auto-release estop before execute: {e}")

        # Reset cancel/pause events for fresh execution
        _exec_cancel.clear()
        _exec_go.set()

        state.status = "executing"
        await _broadcast_status()

        # ── File logging (same as run_script.py) ─────────────────────
        from datetime import datetime
        from pathlib import Path

        from enpire.env.forge.cap.agent.profiler import (
            close_file_logging,
            enable_file_logging,
            set_state_fn,
        )
        from enpire.env.forge.cap.agent.tools._artifact_log import set_artifact_dir

        _logs_root = Path(__file__).resolve().parent.parent.parent / "logs"
        _script_tag = (
            req.code.strip().split("\n")[0].strip("\" #'").replace(" ", "_")[:40]
            or "ui_exec"
        )
        _stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        _run_dir = _logs_root / f"{_stamp}_{_script_tag}"
        _run_dir.mkdir(parents=True, exist_ok=True)

        # Save the exact code submitted from the IDE
        (_run_dir / "code.py").write_text(req.code)

        log_path = enable_file_logging(str(_run_dir))
        set_artifact_dir(_run_dir)
        _get_state = registry.callable_dict().get("get_robot_state")
        if _get_state is not None:
            set_state_fn(_get_state)
        logger.info(f"Execution log dir: {_run_dir}")
        # ─────────────────────────────────────────────────────────────

        loop = asyncio.get_running_loop()
        executor = _make_executor(loop)
        nonlocal _current_executor
        _current_executor = executor
        extra_ns = state.paused_namespace  # Inject preserved vars if resuming
        try:
            result: ExecutionResult = await loop.run_in_executor(
                None, executor.execute, req.code, extra_ns
            )
        finally:
            _current_executor = None
            close_file_logging()

        await _handle_execution_result(result)
        return OkResponse(ok=result.success or result.paused)

    @app.post("/api/mode", response_model=OkResponse)
    async def set_mode(req: ModeRequest) -> OkResponse:
        state.mode = req.mode
        await _broadcast_status()
        return OkResponse(ok=True)

    @app.post("/api/debug_mode", response_model=OkResponse)
    async def set_debug_mode(req: DebugModeRequest) -> OkResponse:
        state.debug_mode = req.enabled
        logger.info(f"Debug mode {'enabled' if req.enabled else 'disabled'}")
        await manager.broadcast("debug_mode_change", {"enabled": req.enabled})
        return OkResponse(ok=True)

    @app.post("/api/camera_streaming", response_model=OkResponse)
    async def set_camera_streaming(req: CameraStreamRequest) -> OkResponse:
        # Accept any camera name — the server will return empty if invalid
        if req.enabled:
            _streaming_cameras.add(req.camera)
        else:
            _streaming_cameras.discard(req.camera)
        logger.info(
            "Camera streaming %s: %s  (active: %s)",
            req.camera,
            "ON" if req.enabled else "OFF",
            _streaming_cameras or "none",
        )
        await manager.broadcast(
            "camera_streaming_change",
            {"camera": req.camera, "enabled": req.enabled},
        )
        return OkResponse(ok=True)

    @app.post("/api/save_cameras", response_model=OkResponse)
    async def save_cameras() -> OkResponse:
        """Save RGB, depth, and intrinsics for each camera to logs/saved_cams/."""
        import json
        from datetime import datetime

        import cv2

        loop = asyncio.get_running_loop()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("logs/saved_cams") / f"cam_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)

        _kacha_client: portal.Client | None = None

        def _save_cam(cam: str) -> None:
            nonlocal _kacha_client
            if _kacha_client is None:
                _kacha_client = portal.Client(f"{cap_server_host}:{cap_server_port}")

            try:
                rgb_result = registry.call("get_camera_image", camera=cam)
                if rgb_result.success and rgb_result.data is not None:
                    img = np.asarray(rgb_result.data)
                    if img.size > 100:
                        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(save_dir / f"{cam}.png"), bgr)
            except Exception:
                logger.warning(f"[kacha] Failed to save RGB for {cam}")

            try:
                depth = np.asarray(_kacha_client.get_camera_depth(cam).result())
                if depth is not None and depth.size > 1:
                    np.save(str(save_dir / f"{cam}_depth.npy"), depth)
            except Exception:
                logger.warning(f"[kacha] Failed to save depth for {cam}")

            try:
                intrinsics = _kacha_client.get_camera_intrinsics(cam).result()
                if intrinsics and float(intrinsics[0]) > 0:
                    intr_dict = {
                        "fx": float(intrinsics[0]),
                        "fy": float(intrinsics[1]),
                        "cx": float(intrinsics[2]),
                        "cy": float(intrinsics[3]),
                    }
                    with open(save_dir / f"{cam}_intrinsics.json", "w") as f:
                        json.dump(intr_dict, f)
            except Exception:
                logger.warning(f"[kacha] Failed to save intrinsics for {cam}")

        for cam in ("top", "left", "right"):
            await loop.run_in_executor(None, _save_cam, cam)

        logger.info(f"Saved camera snapshots to {save_dir}")
        return OkResponse(ok=True)

    @app.post("/api/estop", response_model=OkResponse)
    async def estop() -> OkResponse:
        _realtime_stop_event.set()  # Stop any realtime detection loop
        _exec_cancel.set()  # Signal executor to abort
        _exec_go.set()  # Unblock if paused so it can see cancel
        # Forcibly kill the executor thread (handles while True loops etc.)
        if _current_executor is not None:
            _current_executor.force_cancel()
        state.paused_namespace = None  # Clear any preserved namespace
        if state.debug_mode:
            # In debug mode, just go home in sim — don't touch real robot
            sim = _get_sim_robot()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sim.go_home)
        else:
            # Immediately tell cap_server to freeze the robot
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: registry.call("go_home"))
            # Also trigger the software e-stop on cap_server (holds position)
            try:
                tool = registry.get("get_robot_state")
                if tool is not None:
                    client = tool._get_client(tool._host, tool._port)
                    client.estop().result()
            except Exception as e:
                logger.warning(f"Failed to send estop to cap_server: {e}")
        state.status = "stopped"
        _log_entry("estop", {}, status="error", error="E-STOP triggered")
        await _broadcast_status()
        return OkResponse(ok=True)

    @app.post("/api/pause", response_model=OkResponse)
    async def pause() -> OkResponse:
        _exec_go.clear()  # Executor blocks at next statement boundary
        state.status = "paused"
        await _broadcast_status()
        return OkResponse(ok=True)

    @app.post("/api/resume", response_model=OkResponse)
    async def resume() -> OkResponse:
        # Release the e-stop on cap_server
        try:
            tool = registry.get("get_robot_state")
            if tool is not None:
                client = tool._get_client(tool._host, tool._port)
                client.release_estop().result()
        except Exception as e:
            logger.warning(f"Failed to release estop on cap_server: {e}")
        if state.status == "paused":
            # Unblock the executor so it continues from where it paused
            _exec_cancel.clear()
            _exec_go.set()
            state.status = "executing"
            await _broadcast_status()
        elif state.status == "stopped":
            _exec_cancel.clear()
            _exec_go.set()
            state.status = "idle"
            await _broadcast_status()
        return OkResponse(ok=True)

    @app.post("/api/stop", response_model=OkResponse)
    async def stop() -> OkResponse:
        _realtime_stop_event.set()  # Stop any realtime detection loop
        _exec_cancel.set()  # Signal executor to abort
        _exec_go.set()  # Unblock if paused so it can see cancel
        if _current_executor is not None:
            _current_executor.force_cancel()
        state.status = "idle"
        state.proposed_code = ""
        state.current_task = ""
        state.paused_namespace = None
        _log_entry("stop", {})
        await _broadcast_status()
        return OkResponse(ok=True)

    @app.post("/api/home", response_model=OkResponse)
    async def home() -> OkResponse:
        _realtime_stop_event.set()  # Stop any realtime detection loop
        _exec_cancel.set()  # Signal executor to abort
        _exec_go.set()  # Unblock if paused so it can see cancel
        if _current_executor is not None:
            _current_executor.force_cancel()
        loop = asyncio.get_running_loop()
        if state.debug_mode:
            sim = _get_sim_robot()

            def _do_home_sim():
                sim.open_gripper("left")
                sim.open_gripper("right")
                return sim.go_home()

            ok = await loop.run_in_executor(None, _do_home_sim)
            _log_entry("go_home", {}, status="success")
            return OkResponse(ok=True)
        else:

            def _do_home():
                # RoboCasa exposes reset_env (full episode reset). YAM hardware
                # / non-task envs don't bind it on cap_server, so calling it
                # would surface 'Unknown method reset_env' in server logs even
                # though our except catches it. Gate by robot type instead.
                robot_type = os.environ.get("CAP_ROBOT_TYPE", "yam").strip().lower()
                if robot_type == "robocasa":
                    try:
                        _client = portal.Client(f"{cap_server_host}:{cap_server_port}")
                        result = _client.reset_env().result(timeout=10)
                        if result.get("ok"):
                            logger.info("[Home] Environment reset (new episode)")
                            from enpire.env.forge.cap.agent.tools.base import ToolResult
                            return ToolResult(success=True, data=None)
                    except Exception:
                        pass
                # Hardware / non-task env: open grippers, then go home.
                registry.call("open_gripper", side="left")
                registry.call("open_gripper", side="right")
                return registry.call("go_home")

            result = await loop.run_in_executor(None, _do_home)
            status = "success" if result.success else "error"
            _log_entry("go_home", {}, status=status, error=result.error)
            return OkResponse(ok=result.success)

    # Script management ---

    _scripts_dir = Path(_PROJECT_ROOT) / "cap" / "saved_scripts"
    _scripts_dir.mkdir(parents=True, exist_ok=True)

    _scripts_root = _scripts_dir.resolve()

    def _resolve_script_path(name: str, *, create_parent: bool = False) -> Path:
        """Resolve a user-provided script path under cap/saved_scripts safely."""
        raw = name.strip().replace("\\", "/")
        if not raw:
            raise HTTPException(status_code=400, detail="Script name is required")
        if not raw.endswith(".py"):
            raw += ".py"
        rel = Path(raw)
        if rel.is_absolute() or ".." in rel.parts:
            raise HTTPException(status_code=400, detail="Invalid script path")

        path = (_scripts_dir / rel).resolve()
        try:
            path.relative_to(_scripts_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid script path") from exc

        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @app.get("/api/scripts", response_model=list[ScriptInfo])
    async def list_scripts() -> list[ScriptInfo]:
        scripts = []
        for f in sorted(_scripts_dir.rglob("*.py")):
            stat = f.stat()
            rel = f.relative_to(_scripts_dir).with_suffix("").as_posix()
            scripts.append(
                ScriptInfo(
                    name=rel,
                    size=stat.st_size,
                    modified=time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
                    ),
                )
            )
        return scripts

    @app.post("/api/scripts", response_model=OkResponse)
    async def save_script(req: SaveScriptRequest) -> OkResponse:
        path = _resolve_script_path(req.name, create_parent=True)
        path.write_text(req.code, encoding="utf-8")
        logger.info(f"Script saved: {path}")
        return OkResponse(ok=True)

    @app.get("/api/scripts/{name:path}")
    async def load_script(name: str) -> dict:
        path = _resolve_script_path(name)
        if not path.exists():
            return {"ok": False, "code": "", "error": "not found"}
        return {"ok": True, "code": path.read_text(encoding="utf-8")}

    @app.delete("/api/scripts/{name:path}", response_model=OkResponse)
    async def delete_script(name: str) -> OkResponse:
        path = _resolve_script_path(name)
        if path.exists():
            path.unlink()
            logger.info(f"Script deleted: {path}")
            # Prune now-empty parent directories under saved_scripts.
            parent = path.parent
            while parent != _scripts_dir:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        return OkResponse(ok=True)

    @app.patch("/api/scripts/{name:path}", response_model=OkResponse)
    async def rename_script(name: str, req: RenameScriptRequest) -> OkResponse:
        old_path = _resolve_script_path(name)
        new_path = _resolve_script_path(req.new_name, create_parent=True)
        if not old_path.exists():
            return OkResponse(ok=False)
        if new_path.exists():
            return OkResponse(ok=False)
        old_path.rename(new_path)
        logger.info(f"Script renamed: {old_path} -> {new_path}")
        # Prune empty parent directories left by the move.
        parent = old_path.parent
        while parent != _scripts_dir:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        return OkResponse(ok=True)

    # Read-only state/camera endpoints (for MCP bridge) ---

    @app.get("/api/state")
    async def get_state() -> dict:
        """Return current robot state as JSON (for MCP server / external tools)."""
        loop = asyncio.get_running_loop()
        robot_state = await loop.run_in_executor(None, _fetch_robot_state)
        if robot_state is None:
            return {"ok": False, "error": "Failed to fetch robot state"}
        return {"ok": True, **robot_state}

    @app.get("/api/cameras")
    async def list_cameras() -> dict:
        """Return available camera names from the server."""
        try:
            import portal
            client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            result = client.list_cameras().result()
            return {"ok": True, "cameras": result.get("cameras", [])}
        except Exception as e:
            logger.warning(f"list_cameras failed: {e}")
            return {"ok": True, "cameras": ["top", "left", "right"]}

    @app.get("/api/camera/{name}")
    async def get_camera(name: str) -> dict:
        """Return base64-encoded JPEG for the given camera."""
        # Accept any camera name — server returns empty if invalid
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, _fetch_camera, name)
        if frame is None:
            return {"ok": False, "error": f"Camera {name} unavailable"}
        return {"ok": True, **frame}

    # WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        # Send initial state
        await ws.send_text(
            json.dumps(
                {
                    "type": "status_change",
                    "data": {
                        "mode": state.mode,
                        "status": state.status,
                        "task": state.current_task,
                        "tools": registry.list_tools(),
                        "debug_mode": state.debug_mode,
                    },
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )
        )
        try:
            while True:
                # Keep connection alive; client can also send messages here
                await ws.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(ws)

    # Background state/camera broadcast ---

    def _encode_rgb_jpeg(rgb) -> str:
        """Encode numpy RGB array to base64 JPEG string."""
        from PIL import Image

        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _fetch_robot_state() -> dict | None:
        """Fetch robot state from cap_server (runs in thread). Also updates 3D visualizer."""
        try:
            if state.debug_mode:
                # Use simulated robot state — don't call real cap_server.
                s = _get_sim_robot().get_state()
            else:
                result = registry.call("get_robot_state")
                if not result.success:
                    return None
                s = result.data

                # Update Viser 3D visualization (real mode only)
                visualizer.update_robot_state(
                    left_joint_pos=s.left_joint_pos,
                    right_joint_pos=s.right_joint_pos,
                    left_gripper=s.left_gripper_pos,
                    right_gripper=s.right_gripper_pos,
                )
                visualizer.update_ee_poses(
                    left_pos=s.left_ee_pos,
                    left_quat_xyzw=s.left_ee_quat,
                    right_pos=s.right_ee_pos,
                    right_quat_xyzw=s.right_ee_quat,
                )

            # Build dynamic arm state dict for UI broadcast
            arms_payload = {}
            for arm_name, arm_state in s.arms.items():
                arms_payload[arm_name] = {
                    "joint_positions": arm_state.joint_pos,
                    "gripper": arm_state.gripper_pos,
                    "ee_pose": {
                        "position": arm_state.ee_pos,
                        "quaternion": arm_state.ee_quat,
                    },
                }
            return {
                "arms": arms_payload,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.debug(f"Failed to fetch robot state: {e}")
            return None

    # --- Skill prediction polling (trajectory visualization) ---
    _pred_client: portal.Client | None = None
    _pred_active = False  # track whether we're currently showing predictions
    _policy_ws_active = False
    _last_policy_sig: str | None = None
    _last_policy_error: str | None = None

    def _serialize_policy_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _serialize_policy_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_serialize_policy_value(v) for v in value]
        return value

    def _policy_chunk_horizon(chunk: dict[str, Any]) -> int:
        horizon = 0
        for value in chunk.values():
            arr = np.asarray(value)
            if arr.ndim >= 1:
                horizon = max(horizon, int(arr.shape[0]))
        return horizon

    def _policy_chunk_signature(chunk: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in sorted(chunk):
            arr = np.asarray(chunk[key], dtype=np.float64)
            flat = arr.ravel()
            preview = ",".join(f"{x:.3f}" for x in flat[:4])
            parts.append(f"{key}:{arr.shape}:{preview}")
        return "|".join(parts)

    def _fetch_skill_prediction() -> dict | None:
        """Poll cap_server for the latest action chunk and update visualizer."""
        nonlocal _pred_client, _pred_active, _policy_ws_active, _last_policy_sig, _last_policy_error
        if state.debug_mode:
            if _pred_active:
                visualizer.clear_prediction()
                _pred_active = False
            if _policy_ws_active:
                _policy_ws_active = False
                _last_policy_sig = None
                _last_policy_error = None
                return {"active": False}
            return None
        try:
            if _pred_client is None:
                _pred_client = portal.Client(f"{cap_server_host}:{cap_server_port}")
                logger.info("Prediction client connected to cap_server")
            chunk = _pred_client.get_skill_prediction().result(timeout=POLICY_PREDICTION_RPC_TIMEOUT_S)
            if chunk is not None:
                visualizer.update_prediction(chunk)
                _pred_active = True
                _last_policy_error = None
                sig = _policy_chunk_signature(chunk)
                if sig != _last_policy_sig:
                    _last_policy_sig = sig
                    _policy_ws_active = True
                    return {
                        "active": True,
                        "keys": sorted(chunk.keys()),
                        "horizon": _policy_chunk_horizon(chunk),
                        "chunk": _serialize_policy_value(chunk),
                    }
                _policy_ws_active = True
            else:
                if _pred_active:
                    visualizer.clear_prediction()
                    _pred_active = False
                if _policy_ws_active:
                    _policy_ws_active = False
                    _last_policy_sig = None
                    _last_policy_error = None
                    return {"active": False}
            return None
        except Exception as e:
            error_text = str(e)
            logger.warning("Prediction poll failed: %s", error_text)
            if _pred_active:
                visualizer.clear_prediction()
                _pred_active = False
            should_notify = _policy_ws_active or _last_policy_error != error_text
            _policy_ws_active = False
            _last_policy_sig = None
            _last_policy_error = error_text
            if should_notify:
                return {"active": False, "error": error_text}
            return None

    # --- Scene object sync (object positions → Viser 3D) ---
    _scene_client: portal.Client | None = None

    def _fetch_scene_objects() -> None:
        """Poll cap_server for scene object positions and update Viser."""
        nonlocal _scene_client
        try:
            if _scene_client is None:
                _scene_client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            result = _scene_client.get_object_positions().result()
            if result and result.get("ok"):
                objects = result.get("objects", {})
                if objects:
                    visualizer.update_scene_objects(objects)
                else:
                    visualizer.clear_scene_objects()
        except Exception:
            pass  # cap_server may not support this (non-sim mode)

    _ls_client: portal.Client | None = None

    def _fetch_learn_skill_status() -> dict | None:
        nonlocal _ls_client
        try:
            if _ls_client is None:
                _ls_client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            return _ls_client.get_learn_skill_status().result()
        except Exception:
            return None

    # --- Safety zone visualization polling ---
    _sz_client: portal.Client | None = None
    def _fetch_safety_zone() -> None:
        """Poll cap_server for safety zone config and update Viser."""
        nonlocal _sz_client
        if not SAFETY_VIZ_ENABLED:
            return
        try:
            if _sz_client is None:
                _sz_client = portal.Client(f"{cap_server_host}:{cap_server_port}")
            logger.debug("[safety_viz] polling get_safety_zone...")
            zone_config = _sz_client.get_safety_zone().result()
            logger.debug(
                "[safety_viz] got config: active=%s", zone_config.get("active")
            )
            visualizer.update_safety_zone(zone_config)
            logger.debug("[safety_viz] visualizer updated OK")
        except Exception:
            logger.exception("[safety_viz] _fetch_safety_zone failed")

    _camera_log_once: set[str] = set()

    def _fetch_camera(camera_name: str) -> dict | None:
        """Fetch camera image from cap_server (runs in thread)."""
        try:
            result = registry.call("get_camera_image", camera=camera_name)
            if not result.success or result.data is None:
                if camera_name not in _camera_log_once:
                    logger.warning(
                        f"Camera {camera_name}: tool returned success={result.success}, data={'None' if result.data is None else type(result.data)}"
                    )
                    _camera_log_once.add(camera_name)
                return None
            img = np.asarray(result.data)
            # Skip tiny dummy images (cap_server returns 1x1 when camera unavailable)
            if img.size < 100:
                return None
            if camera_name not in _camera_log_once:
                logger.info(f"Camera {camera_name}: got frame {img.shape} {img.dtype}")
                _camera_log_once.add(camera_name)
            return {
                "camera": camera_name,
                "image": _encode_rgb_jpeg(img),
            }
        except Exception as e:
            if camera_name not in _camera_log_once:
                logger.warning(f"Failed to fetch camera {camera_name}: {e}")
                _camera_log_once.add(camera_name)
            return None

    async def _broadcast_loop() -> None:
        """Periodically broadcast robot state and camera frames to all WS clients."""
        loop = asyncio.get_running_loop()
        _loop_count = 0

        while True:
            if not manager._connections:
                await asyncio.sleep(0.5)
                continue

            _loop_count += 1
            _t_loop = time.time()

            # --- Gather all fetches for this tick concurrently ---
            futures: list[asyncio.Task] = []

            # Scene objects + robot state at ~30Hz (every tick)
            scene_fut = loop.run_in_executor(None, _fetch_scene_objects)
            state_fut = loop.run_in_executor(None, _fetch_robot_state)
            futures.extend([scene_fut, state_fut])

            # Skill prediction at ~10Hz
            pred_fut = None
            if _loop_count % 3 == 1:
                pred_fut = loop.run_in_executor(None, _fetch_skill_prediction)
                futures.append(pred_fut)

            # Safety zone at ~3Hz
            sz_fut = None
            if _loop_count % 10 == 0:
                sz_fut = loop.run_in_executor(None, _fetch_safety_zone)
                futures.append(sz_fut)

            # Learn-skill status at ~10Hz
            ls_fut = None
            if _loop_count % 3 == 2:
                ls_fut = loop.run_in_executor(None, _fetch_learn_skill_status)
                futures.append(ls_fut)

            # Camera streams at ~10Hz
            cam_futs: list[tuple[str, asyncio.Future]] = []
            if _loop_count % 3 == 0:
                for _cam in list(_streaming_cameras):
                    f = loop.run_in_executor(None, _fetch_camera, _cam)
                    cam_futs.append((_cam, f))
                    futures.append(f)

            await asyncio.gather(*futures, return_exceptions=True)

            # --- Broadcast results ---
            robot_state = state_fut.result() if not state_fut.exception() else None
            if robot_state is not None:
                await manager.broadcast("state_update", robot_state)

            if pred_fut is not None and not pred_fut.exception():
                policy_update = pred_fut.result()
                if policy_update is not None:
                    if isinstance(policy_update, dict) and policy_update.get("error"):
                        await manager.broadcast(
                            "error",
                            {
                                "message": "Policy prediction unavailable",
                                "source": "policy_update",
                                "detail": str(policy_update["error"]),
                            },
                        )
                    await manager.broadcast("policy_update", policy_update)

            if sz_fut is not None and sz_fut.exception():
                logger.exception("[broadcast] _fetch_safety_zone failed")

            if ls_fut is not None and not ls_fut.exception():
                ls_status = ls_fut.result()
                if ls_status is not None:
                    await manager.broadcast("learn_skill_update", ls_status)

            for _cam, f in cam_futs:
                if not f.exception():
                    frame = f.result()
                    if frame is not None:
                        await manager.broadcast("camera_frame", frame)

            _loop_ms = (time.time() - _t_loop) * 1000

            # Heartbeat every 150 ticks (~5s) so we know the loop is alive
            if _loop_count % 150 == 0:
                logger.info(
                    "[broadcast] heartbeat tick=%d  loop_ms=%.0f", _loop_count, _loop_ms
                )

            await asyncio.sleep(0.033)  # ~30Hz loop

    @app.on_event("startup")
    async def _start_broadcast():
        asyncio.create_task(_broadcast_loop())

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    host: str = "0.0.0.0"
    port: int = CAP_AGENT_PORT
    cap_server_host: str = "localhost"
    cap_server_port: int = CAP_SERVER_PORT
    detection_host: str = "localhost"
    detection_port: int = DETECTION_SERVER_PORT
    policy_server: str = DEFAULT_POLICY_SERVER
    llm_backend: str = "claude"
    llm_model: str = DEFAULT_LLM_MODEL
    viser_port: int = VISER_PORT
    bundlesdf_host: str | None = None
    bundlesdf_port: int | None = None
    sam3_host: str | None = None
    sam3_port: int | None = None


def main(cfg: AgentConfig | None = None) -> None:
    if cfg is None:
        cfg = tyro.cli(AgentConfig)

    logging.basicConfig(level=logging.INFO)
    app = create_app(
        cap_server_host=cfg.cap_server_host,
        cap_server_port=cfg.cap_server_port,
        detection_host=cfg.detection_host,
        detection_port=cfg.detection_port,
        policy_server=cfg.policy_server,
        llm_backend=cfg.llm_backend,
        llm_model=cfg.llm_model,
        viser_port=cfg.viser_port,
        bundlesdf_host=cfg.bundlesdf_host,
        bundlesdf_port=cfg.bundlesdf_port,
        sam3_host=cfg.sam3_host,
        sam3_port=cfg.sam3_port,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
