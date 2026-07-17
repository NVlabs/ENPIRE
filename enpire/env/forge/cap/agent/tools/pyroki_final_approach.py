"""PyRoki-backed straight final approach tool."""

from __future__ import annotations

from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import CAP_SERVER_PORT


DEFAULT_PYROKI_URL = "http://127.0.0.1:9600"
DEFAULT_MAX_FK_POSITION_ERROR_M = 0.02
DEFAULT_MAX_FINAL_POSITION_ERROR_M = 0.03


def _display_rpy_to_quat_xyzw(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    quat = Rotation.from_euler(
        "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
    ).as_quat()
    return [float(x) for x in quat]


def _xyzw_xyz(quat_xyzw: list[float], pos: list[float]) -> list[float]:
    qx, qy, qz, qw = [float(x) for x in quat_xyzw]
    return [qw, qx, qy, qz] + [float(x) for x in pos]


def _side_joint_index(name: str, side: str) -> int | None:
    prefix = f"{side}_joint"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if not suffix.isdigit():
        return None
    idx = int(suffix) - 1
    return idx if 0 <= idx < 6 else None


def _build_start_cfg(joint_names: list[str], state: dict[str, Any]) -> list[float]:
    left = np.asarray(state.get("left_joint_pos", []), dtype=float).reshape(-1)
    right = np.asarray(state.get("right_joint_pos", []), dtype=float).reshape(-1)
    out: list[float] = []
    for name in joint_names:
        idx = _side_joint_index(name, "left")
        if idx is not None and idx < len(left):
            out.append(float(left[idx]))
            continue
        idx = _side_joint_index(name, "right")
        if idx is not None and idx < len(right):
            out.append(float(right[idx]))
            continue
        out.append(0.0)
    return out


def _extract_side_waypoints(
    waypoints: list[list[float]],
    joint_names: list[str],
    side: str,
) -> list[list[float]]:
    indices = [
        i
        for i, name in enumerate(joint_names)
        if _side_joint_index(str(name), side) is not None
    ]
    indices.sort(key=lambda i: _side_joint_index(str(joint_names[i]), side) or 0)
    if len(indices) != 6:
        raise RuntimeError(
            f"PyRoki response did not include six {side} arm joints; "
            f"joint_names={joint_names}"
        )
    return [[float(row[i]) for i in indices] for row in waypoints]


class PyrokiFinalApproachTool(Tool):
    name = "pyroki_final_approach"
    description = (
        "Plan and execute a short straight Cartesian final approach using the "
        "PyRoki HTTP service, then execute the resulting joint keypoints."
    )
    parameters = [
        ToolParameter("side", "str", "Arm side: 'left' or 'right'."),
        ToolParameter("target_pos", "list[float]", "Target EE position xyz in world."),
        ToolParameter("target_rpy", "list[float]", "Target display RPY in degrees."),
        ToolParameter(
            "server_url",
            "str",
            "PyRoki service URL.",
            required=False,
            default=DEFAULT_PYROKI_URL,
        ),
        ToolParameter(
            "timesteps",
            "int",
            "Number of straight-line IK waypoints.",
            required=False,
            default=16,
        ),
        ToolParameter(
            "max_cartesian_speed_mps",
            "float",
            "Max end-effector Cartesian speed used for timing.",
            required=False,
            default=0.15,
        ),
        ToolParameter(
            "max_joint_vel_rad_s",
            "float",
            "Max joint speed used for timing.",
            required=False,
            default=0.8,
        ),
        ToolParameter(
            "min_duration_s",
            "float",
            "Minimum execution duration.",
            required=False,
            default=0.6,
        ),
        ToolParameter(
            "max_fk_position_error_m",
            "float",
            "Reject PyRoki plans whose FK endpoint error exceeds this.",
            required=False,
            default=DEFAULT_MAX_FK_POSITION_ERROR_M,
        ),
        ToolParameter(
            "max_final_position_error_m",
            "float",
            "Reject execution if CAP final EE position is farther than this from target.",
            required=False,
            default=DEFAULT_MAX_FINAL_POSITION_ERROR_M,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import portal

            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    def execute(self, **kwargs: Any) -> ToolResult:
        side = str(kwargs["side"])
        if side not in ("left", "right"):
            return ToolResult(success=False, error=f"invalid side: {side}")

        server_url = str(kwargs.get("server_url") or DEFAULT_PYROKI_URL).rstrip("/")
        timesteps = max(int(kwargs.get("timesteps", 16)), 3)
        max_cart_speed = max(float(kwargs.get("max_cartesian_speed_mps", 0.15)), 1e-3)
        max_joint_vel = max(float(kwargs.get("max_joint_vel_rad_s", 0.8)), 1e-3)
        min_duration = max(float(kwargs.get("min_duration_s", 0.6)), 0.1)
        max_fk_position_error_m = float(
            kwargs.get("max_fk_position_error_m", DEFAULT_MAX_FK_POSITION_ERROR_M)
        )
        max_final_position_error_m = float(
            kwargs.get(
                "max_final_position_error_m",
                DEFAULT_MAX_FINAL_POSITION_ERROR_M,
            )
        )

        try:
            health = requests.get(f"{server_url}/health", timeout=3.0)
            health.raise_for_status()
            health_json = health.json()
            if not bool(health_json.get("ready", False)):
                return ToolResult(success=False, error=f"PyRoki not ready: {health_json}")
            joint_names = [str(x) for x in health_json.get("joint_names", [])]
            if not joint_names:
                return ToolResult(
                    success=False,
                    error="PyRoki health response did not include joint_names.",
                )
            link_names = [str(x) for x in health_json.get("link_names", [])]
            target_link_name = f"{side}_grasp"
            if link_names and target_link_name not in link_names:
                return ToolResult(
                    success=False,
                    error=(
                        f"PyRoki links do not include {target_link_name}; "
                        f"link_names={link_names}"
                    ),
                )

            client = self._get_client()
            state = client.get_state().result()
            cur_pos = [float(x) for x in state[f"{side}_ee_pos"]]
            cur_quat = [float(x) for x in state[f"{side}_ee_quat_xyzw"]]
            target_pos = [float(x) for x in kwargs["target_pos"]]
            target_quat = _display_rpy_to_quat_xyzw(
                [float(x) for x in kwargs["target_rpy"]]
            )
            start_cfg = _build_start_cfg(joint_names, state)

            payload = {
                "start_pose_wxyz_xyz": _xyzw_xyz(cur_quat, cur_pos),
                "end_pose_wxyz_xyz": _xyzw_xyz(target_quat, target_pos),
                "target_link_name": target_link_name,
                "start_cfg": start_cfg,
                "timesteps": timesteps,
            }
            resp = requests.post(f"{server_url}/plan", json=payload, timeout=60.0)
            resp.raise_for_status()
            plan = resp.json()
            response_joint_names = [
                str(x) for x in plan.get("joint_names", joint_names)
            ]
            waypoints = _extract_side_waypoints(
                plan["waypoints"], response_joint_names, side
            )
            if not waypoints:
                return ToolResult(success=False, error="PyRoki returned no waypoints.")

            end_fk_error = float(plan.get("end_pos_error_m", float("inf")))
            max_fk_error = float(plan.get("max_pos_error_m", float("inf")))
            requested_grasp_dist = float(
                np.linalg.norm(np.asarray(target_pos) - np.asarray(cur_pos))
            )
            realized_grasp_dist = float(plan.get("fk_cartesian_distance_m", 0.0))
            if max_fk_error > max_fk_position_error_m:
                return ToolResult(
                    success=False,
                    data=plan,
                    error=(
                        f"PyRoki FK validation failed: max_pos_error_m={max_fk_error:.4f} "
                        f"> {max_fk_position_error_m:.4f}"
                    ),
                )
            if (
                requested_grasp_dist > 0.01
                and realized_grasp_dist < 0.5 * requested_grasp_dist
            ):
                return ToolResult(
                    success=False,
                    data=plan,
                    error=(
                        "PyRoki FK path did not translate enough: "
                        f"requested={requested_grasp_dist:.4f}m "
                        f"realized={realized_grasp_dist:.4f}m"
                    ),
                )

            current_arm = [float(x) for x in state[f"{side}_joint_pos"]]
            waypoints[0] = current_arm
            max_joint_delta = max(
                float(np.max(np.abs(np.asarray(b) - np.asarray(a))))
                for a, b in zip(waypoints[:-1], waypoints[1:])
            )
            cart_dist = float(
                np.linalg.norm(np.asarray(target_pos) - np.asarray(cur_pos))
            )
            duration = max(
                min_duration,
                cart_dist / max_cart_speed,
                max_joint_delta / max_joint_vel,
            )
            timestamps = np.linspace(0.0, duration, len(waypoints)).tolist()

            result = client.move_joint_keypoints(
                side,
                np.asarray(timestamps, dtype=np.float64),
                np.asarray(waypoints, dtype=np.float64),
            ).result()
            if not bool(result.get("success", False)):
                return ToolResult(
                    success=False,
                    data=result,
                    error=result.get("reason", "move_joint_keypoints failed"),
                )

            final_state = client.get_state().result()
            final_pos = np.asarray(final_state[f"{side}_ee_pos"], dtype=float)
            final_err = float(np.linalg.norm(final_pos - np.asarray(target_pos, dtype=float)))
            if final_err > max_final_position_error_m:
                return ToolResult(
                    success=False,
                    data={
                        "move_result": result,
                        "plan": plan,
                        "final_position_error_m": final_err,
                    },
                    error=(
                        f"PyRoki execution ended {final_err:.4f}m from grasp target "
                        f"(limit {max_final_position_error_m:.4f}m)"
                    ),
                )

            data = {
                "side": side,
                "server_url": server_url,
                "target_link_name": target_link_name,
                "timesteps": len(waypoints),
                "duration_s": float(duration),
                "cartesian_distance_m": cart_dist,
                "requested_grasp_distance_m": requested_grasp_dist,
                "realized_grasp_distance_m": realized_grasp_dist,
                "end_fk_error_m": end_fk_error,
                "max_fk_error_m": max_fk_error,
                "final_position_error_m": final_err,
                "max_joint_step_rad": max_joint_delta,
            }
            return ToolResult(success=True, data=data)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
