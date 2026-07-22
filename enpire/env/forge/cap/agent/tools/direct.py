# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct tool callables that bypass Portal RPC for the CAP server.

``make_direct_callables(server, registry)`` returns a dict of Python callables
matching the same signatures and return types as the Portal-based tools in
``registry.callable_dict()``, but calling the CapServer methods directly.

Also provides ``make_cancel_callables(server)`` for motion cancellation tools
(cancel_motion, estop, release_estop).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from enpire.env.forge.cap.agent.tools import Detection3D, MoveResult, RobotState
from enpire.env.forge.cap.agent.tools.base import ArmState
from enpire.env.forge.cap.config import (
    GRIPPER_SETTLE_TIMEOUT_S,
    MOVE_EEF_MAX_DURATION_S,
    MOVE_EEF_MAX_VEL,
)

if TYPE_CHECKING:
    from enpire.env.forge.cap.server.cap_server import CapServer


def _quat_xyzw_to_rpy(q: list[float]) -> list[float]:
    """Convert quaternion [x, y, z, w] to RPY [roll, pitch, yaw] in radians."""
    qx, qy, qz, qw = q
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return [roll, pitch, yaw]


def make_direct_callables(
    server: CapServer,
    registry: Any,
) -> dict[str, Any]:
    """Build direct-call wrappers that bypass Portal RPC.

    Returns a dict of callables matching the same signatures and return types
    as the Portal-based tools in ``registry.callable_dict()``.
    """

    def get_robot_state() -> RobotState:
        state = server.get_state()
        arms: dict[str, ArmState] = {}
        for key in state:
            if key.endswith("_joint_pos"):
                side = key[: -len("_joint_pos")]
                if f"{side}_ee_pos" not in state:
                    continue
                quat = list(state[f"{side}_ee_quat_xyzw"])
                arms[side] = ArmState(
                    joint_pos=list(state[f"{side}_joint_pos"]),
                    gripper_pos=float(state[f"{side}_gripper_pos"][0]),
                    ee_pos=list(state[f"{side}_ee_pos"]),
                    ee_quat=quat,
                    ee_rpy=_quat_xyzw_to_rpy(quat),
                )
        return RobotState(arms=arms)

    def _ik_servo(
        side: str,
        pos: list[float],
        quat: list[float],
        gripper: float | None = None,
        max_duration_sec: float = MOVE_EEF_MAX_DURATION_S,
        max_vel: float = MOVE_EEF_MAX_VEL,
        check_feasibility: bool = False,
        tol: float = 0.02,
        dry_run_only: bool = False,
        **_: Any,
    ) -> MoveResult:
        result = server._ik_servo(
            side,
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
            gripper,
            float(max_duration_sec),
            float(max_vel),
            bool(check_feasibility),
            float(tol),
            bool(dry_run_only),
        )
        success = bool(result.get("success", False))
        if not success:
            raise RuntimeError(f"Tool _ik_servo failed: {result.get('reason', '')}")
        return MoveResult(
            reached=bool(result.get("moved", success)),
            feasible=bool(result.get("feasible", True)),
            cmd_pos_err=result.get("cmd_pos_err"),
        )

    def set_gripper(
        side: str,
        pos: float,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> None:
        ok = server.set_gripper(
            side, pos, GRIPPER_SETTLE_TIMEOUT_S, vel_limit, torque_limit
        )
        if not ok:
            raise RuntimeError(f"Tool set_gripper failed: side={side}")

    def open_gripper(
        side: str,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> None:
        ok = server.set_gripper(
            side, 1.0, GRIPPER_SETTLE_TIMEOUT_S, vel_limit, torque_limit
        )
        if not ok:
            raise RuntimeError(f"Tool open_gripper failed: side={side}")

    def open_gripper_fast(
        side: str,
        vel_limit: float | None = 30.0,
        torque_limit: float | None = None,
        timeout: float = 0.25,
    ) -> None:
        ok = server.set_gripper(side, 1.0, timeout, vel_limit, torque_limit)
        if not ok:
            raise RuntimeError(f"Tool open_gripper_fast failed: side={side}")

    def close_gripper(
        side: str,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> None:
        ok = server.set_gripper(
            side, 0.0, GRIPPER_SETTLE_TIMEOUT_S, vel_limit, torque_limit
        )
        if not ok:
            raise RuntimeError(f"Tool close_gripper failed: side={side}")

    def go_home() -> None:
        ok = server.go_home()
        if not ok:
            raise RuntimeError("Tool go_home failed")

    def get_camera_image(camera: str) -> np.ndarray:
        return server.get_camera_image(camera)

    # detect_object: oracle mode uses sim ground truth, else Portal-based
    _original_detect = registry.callable_dict().get("detect_object")

    def detect_object(
        query: str,
        camera: str = "top",
        backend: str = "bundlesdf",
    ) -> list[Detection3D]:
        if backend == "oracle":
            result = server.get_object_positions()
            if not result.get("ok"):
                raise RuntimeError("get_object_positions failed (is sim running?)")
            objects: dict[str, dict] = result["objects"]
            if not objects:
                raise RuntimeError("No scene objects found in simulation")
            query_lower = query.lower()
            matches = [
                (name, data)
                for name, data in objects.items()
                if query_lower in name.lower() or name.lower() in query_lower
            ]
            if not matches:
                available = ", ".join(objects.keys())
                raise RuntimeError(
                    f"No object matching '{query}'. Available: {available}"
                )
            detections: list[Detection3D] = []
            for name, data in matches:
                pos_3d = data["pos"]
                quat_wxyz = data["quat"]
                quat_xyzw = quat_wxyz[1:] + quat_wxyz[:1]
                size = data.get("size", [])
                detections.append(
                    Detection3D(
                        label=name,
                        score=1.0,
                        box_2d=[],
                        position_3d=[round(float(x), 4) for x in pos_3d],
                        quaternion_xyzw=[round(float(x), 4) for x in quat_xyzw],
                        half_extents=[round(float(x), 4) for x in size],
                    )
                )
            return detections
        if _original_detect is None:
            raise RuntimeError("detect_object tool not available in registry")
        return _original_detect(query, camera, backend)

    def setup_scene(name: str) -> dict:
        result = server.setup_scene(name)
        if not result.get("ok"):
            raise RuntimeError(f"Tool setup_scene failed: {result}")
        return result

    def clear_table() -> dict:
        result = server.clear_table()
        if not result.get("ok"):
            raise RuntimeError(f"Tool clear_table failed: {result}")
        return result

    def set_body_pose(
        name: str, pos: list, quat_wxyz: list, gravity_comp: bool = True
    ) -> dict:
        result = server.set_body_pose(name, pos, quat_wxyz, gravity_comp)
        if not result.get("ok"):
            raise RuntimeError(f"Tool set_body_pose failed: {result}")
        return result

    def set_safety_zone(
        side: str,
        keyposes: list,
        pos_margin: float,
        ori_margin: float,
    ) -> dict:
        return server.set_safety_zone(side, keyposes, pos_margin, ori_margin)

    def clear_safety_zone(side: str = "") -> dict:
        return server.clear_safety_zone(side)

    def get_safety_zone() -> dict:
        return server.get_safety_zone()

    def learn_skill(
        skill_name: str,
        params: dict | None = None,
    ) -> Any:
        result = server.learn_skill(skill_name, params)
        success = result.get("success", False)
        if not success:
            raise RuntimeError(f"learn_skill failed: {result.get('reason', '')}")

        class SkillResult:
            def __init__(self, d: dict):
                self.success = d.get("success", False)
                self.steps_executed = d.get("steps_executed", 0)
                self.info = d.get("info", {})

        return SkillResult(result)

    def get_task_info() -> dict:
        return server._rpc_get_task_info()

    def reset_env() -> dict:
        """Reset the simulation environment for a fresh episode (sim/RoboCasa only)."""
        if hasattr(server, "_rpc_reset_env"):
            return server._rpc_reset_env()
        return {"ok": False, "reason": "not supported"}

    def reset_to_initial() -> dict:
        """Deterministic reset — same scene, same objects, same positions.

        Restores the initial MuJoCo state without re-randomizing.
        Used by the agent retry loop so each iteration attempts the same task.
        """
        if hasattr(server, "_rpc_reset_to_initial"):
            return server._rpc_reset_to_initial()
        return reset_env()  # fallback

    return {
        "get_robot_state": get_robot_state,
        "_ik_servo": _ik_servo,
        "set_gripper": set_gripper,
        "open_gripper": open_gripper,
        "open_gripper_fast": open_gripper_fast,
        "close_gripper": close_gripper,
        "go_home": go_home,
        "get_camera_image": get_camera_image,
        "detect_object": detect_object,
        "setup_scene": setup_scene,
        "clear_table": clear_table,
        "set_body_pose": set_body_pose,
        "set_safety_zone": set_safety_zone,
        "clear_safety_zone": clear_safety_zone,
        "get_safety_zone": get_safety_zone,
        "learn_skill": learn_skill,
        "get_task_info": get_task_info,
        "reset_env": reset_env,
        "reset_to_initial": reset_to_initial,
    }


def make_cancel_callables(server: CapServer) -> dict[str, Any]:
    """Return cancel_motion, estop, release_estop callables for the namespace."""

    def cancel_motion(side: str) -> None:
        """Cancel an ongoing _ik_servo on the given arm. Thread-safe, non-blocking."""
        if hasattr(server, "cancel_motion"):
            server.cancel_motion(side)

    def estop() -> None:
        """Emergency stop: freeze the entire robot (nuclear option)."""
        server.estop()

    def release_estop() -> None:
        """Release emergency stop."""
        server.release_estop()

    return {
        "cancel_motion": cancel_motion,
        "estop": estop,
        "release_estop": release_estop,
    }
