"""CAP tool registry — dynamic registration and lookup of all tools.

Usage::

    registry = create_default_registry()
    result = registry.call("get_robot_state")
    result = registry.call("detect_object", query="red cup", camera="top")
"""

from __future__ import annotations

from typing import Any

from cap.agent.tools.base import Tool, ToolResult

# Re-export data types for convenience
from cap.agent.tools.base import (  # noqa: F401
    Detection3D,
    FreespaceResult,
    MoveResult,
    NudgeResult,
    RobotState,
    SegmentationResult,
    SkillResult,
    ToolParameter,
)


class ToolRegistry:
    """Container for registered tools.  Supports dynamic add/remove."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(**kwargs)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """Return JSON schemas for all registered tools (for LLM context)."""
        return [t.schema() for t in self._tools.values()]

    def callable_dict(self) -> dict[str, Any]:
        """Return a dict of ``{tool_name: callable}`` for code executor injection.

        Each callable wraps ``tool.execute(**kwargs)`` so that user / LLM code
        can call tools as plain functions::

            state = get_robot_state()  # returns ToolResult
        """

        def _make_fn(tool: Tool):
            # Build ordered param names so positional args can be mapped to kwargs
            param_names = [p.name for p in tool.parameters]
            allowed = set(param_names)

            def fn(*args: Any, **kw: Any) -> Any:
                # Map positional args to keyword args using parameter order
                for i, val in enumerate(args):
                    if i < len(param_names):
                        kw[param_names[i]] = val
                # Reject unknown kwargs loudly — silently dropping them has
                # hidden typos before (e.g. `max_episode_steps` → no-op).
                unknown = set(kw) - allowed
                if unknown:
                    raise TypeError(
                        f"{tool.name}() got unexpected keyword argument(s) "
                        f"{sorted(unknown)}. Known: {sorted(allowed)}."
                    )
                result = tool.execute(**kw)
                if not result.success:
                    raise RuntimeError(f"Tool {tool.name} failed: {result.error}")
                return result.data

            fn.__name__ = tool.name
            fn.__doc__ = tool.description
            return fn

        return {name: _make_fn(tool) for name, tool in self._tools.items()}


def create_default_registry(
    cap_server_host: str = "localhost",
    cap_server_port: int | None = None,
    detection_host: str = "localhost",
    detection_port: int | None = None,
    policy_server: str = "localhost:8964",
    bundlesdf_host: str | None = None,
    bundlesdf_port: int | None = None,
    sam3_host: str | None = None,
    sam3_port: int | None = None,
) -> ToolRegistry:
    """Build a registry with all built-in tools configured for the given endpoints."""
    from cap.config import CAP_SERVER_PORT, DETECTION_SERVER_PORT
    from cap.config import BUNDLESDF_SERVER_HOST, BUNDLESDF_SERVER_PORT
    from cap.config import SAM3_SERVER_HOST, SAM3_SERVER_PORT
    from cap.agent.tools.native import (
        ClearTableTool,
        CloseGripperTool,
        GetCameraImageTool,
        GetObjectPositionsTool,
        GetRobotStateTool,
        GetTaskInfoTool,
        GoHomeTool,
        ListScenesTool,
        LoadTaskTool,
        MoveJointKeypointsTool,
        OpenGripperFastTool,
        OpenGripperTool,
        SetGripperTool,
        SetupRewardTool,
        SetBodyPoseTool,
        SetupSceneTool,
    )
    from cap.agent.tools.detection import DetectObjectTool, DetectObjectsOneshotTool
    from cap.agent.tools.skill import ExecuteSkillTool, LearnSkillTool
    from cap.agent.tools.vlm_query import VlmQueryTool
    from cap.agent.tools.bundlesdf_track import (
        AddDetectionTool,
        EndDetectionTool,
        GetDetectionTool,
        ListDetectionsTool,
    )
    from cap.agent.tools.safety import (
        SetSafetyZoneTool,
        ClearSafetyZoneTool,
        GetSafetyZoneTool,
    )
    from cap.agent.tools.save_image import SaveImageTool
    from cap.agent.tools.object_tracking import (
        GetObjectPoseTool,
        StopTrackingTool,
        TrackObjectTool,
        _TrackingContext,
    )
    from cap.agent.tools.freespace_move import FreespaceMoveTool
    from cap.agent.tools.pyroki_final_approach import PyrokiFinalApproachTool
    from cap.agent.tools.rotate_joint import RotateJointTool
    from cap.agent.tools.nudge import NudgeTool
    from cap.agent.tools.grasp import GraspTool, PlaceTool
    from cap.agent.tools.policy_output import (
        StartPolicyOutputTool,
        StepPolicyOutputTool,
        StopPolicyOutputTool,
        UsePolicyOutputTool,
    )
    from cap.agent.tools.scene_objects import ListSceneObjectsTool
    from cap.agent.tools.segmentation import SegmentAllObjectsTool, SegmentObjectTool
    from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool
    from cap.agent.tools.grasp_2d import SampleGraspPose2DTool
    from cap.agent.tools.grasp_3d_bb import SampleGraspPose3DBBoxTool

    srv_port = cap_server_port or CAP_SERVER_PORT
    det_port = detection_port or DETECTION_SERVER_PORT
    bsdf_host = bundlesdf_host or BUNDLESDF_SERVER_HOST
    bsdf_port = bundlesdf_port or BUNDLESDF_SERVER_PORT
    s3_host = sam3_host or SAM3_SERVER_HOST
    s3_port = sam3_port or SAM3_SERVER_PORT

    registry = ToolRegistry()
    registry.register(GetRobotStateTool(host=cap_server_host, port=srv_port))
    registry.register(MoveJointKeypointsTool(host=cap_server_host, port=srv_port))
    registry.register(SetGripperTool(host=cap_server_host, port=srv_port))
    registry.register(OpenGripperTool(host=cap_server_host, port=srv_port))
    registry.register(OpenGripperFastTool(host=cap_server_host, port=srv_port))
    registry.register(CloseGripperTool(host=cap_server_host, port=srv_port))
    registry.register(GoHomeTool(host=cap_server_host, port=srv_port))
    registry.register(GetCameraImageTool(host=cap_server_host, port=srv_port))
    detect_tool = DetectObjectTool(
        detection_host=detection_host,
        detection_port=det_port,
        cap_server_host=cap_server_host,
        cap_server_port=srv_port,
        bundlesdf_host=bsdf_host,
        bundlesdf_port=bsdf_port,
    )
    registry.register(detect_tool)
    registry.register(DetectObjectsOneshotTool(detect_tool=detect_tool))
    registry.register(
        ExecuteSkillTool(
            host=cap_server_host,
            port=srv_port,
            policy_server=policy_server,
        )
    )
    registry.register(
        StartPolicyOutputTool(
            host=cap_server_host,
            port=srv_port,
        )
    )
    registry.register(
        StepPolicyOutputTool(
            host=cap_server_host,
            port=srv_port,
        )
    )
    registry.register(
        StopPolicyOutputTool(
            host=cap_server_host,
            port=srv_port,
        )
    )
    registry.register(
        UsePolicyOutputTool(
            host=cap_server_host,
            port=srv_port,
        )
    )
    registry.register(
        LearnSkillTool(
            host=cap_server_host,
            port=srv_port,
        )
    )
    registry.register(
        VlmQueryTool(
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
        )
    )
    registry.register(SetupRewardTool(host=cap_server_host, port=srv_port))
    # Scene management (sim-only, no-op errors in real mode)
    registry.register(SetupSceneTool(host=cap_server_host, port=srv_port))
    registry.register(ClearTableTool(host=cap_server_host, port=srv_port))
    registry.register(ListScenesTool(host=cap_server_host, port=srv_port))
    registry.register(GetObjectPositionsTool(host=cap_server_host, port=srv_port))
    registry.register(SetBodyPoseTool(host=cap_server_host, port=srv_port))
    # Task management (RoboCasa, etc.)
    registry.register(GetTaskInfoTool(host=cap_server_host, port=srv_port))
    registry.register(LoadTaskTool(host=cap_server_host, port=srv_port))
    # BundleSDF multi-object tracking tools
    registry.register(
        AddDetectionTool(bundlesdf_host=bsdf_host, bundlesdf_port=bsdf_port)
    )
    registry.register(
        GetDetectionTool(bundlesdf_host=bsdf_host, bundlesdf_port=bsdf_port)
    )
    registry.register(
        EndDetectionTool(bundlesdf_host=bsdf_host, bundlesdf_port=bsdf_port)
    )
    registry.register(
        ListDetectionsTool(bundlesdf_host=bsdf_host, bundlesdf_port=bsdf_port)
    )
    # Safety zone tools
    registry.register(SetSafetyZoneTool(host=cap_server_host, port=srv_port))
    registry.register(ClearSafetyZoneTool(host=cap_server_host, port=srv_port))
    registry.register(GetSafetyZoneTool(host=cap_server_host, port=srv_port))
    # Image save tool
    registry.register(
        SaveImageTool(cap_server_host=cap_server_host, cap_server_port=srv_port)
    )
    # Object tracking (BundleSDF 6-DOF real-time tracking)
    # Shared context carries the active camera name so GetObjectPoseTool
    # can fetch the correct extrinsics for the cam→world transform.
    tracking_ctx = _TrackingContext()
    registry.register(
        TrackObjectTool(
            bundlesdf_host=bsdf_host,
            bundlesdf_port=bsdf_port,
            context=tracking_ctx,
        )
    )
    registry.register(
        GetObjectPoseTool(
            bundlesdf_host=bsdf_host,
            bundlesdf_port=bsdf_port,
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
            context=tracking_ctx,
        )
    )
    registry.register(
        StopTrackingTool(
            bundlesdf_host=bsdf_host,
            bundlesdf_port=bsdf_port,
            context=tracking_ctx,
        )
    )
    # Collision-free free-space movement (cuRobo by default; RRT-Connect optional)
    registry.register(FreespaceMoveTool(host=cap_server_host, port=srv_port))
    # Single-joint rotation used by reorientation skills.
    registry.register(RotateJointTool(host=cap_server_host, port=srv_port))
    # Nudge — small delta EE adjustments
    registry.register(NudgeTool(host=cap_server_host, port=srv_port))
    # PyRoki straight Cartesian final approach for short grasp refinements.
    registry.register(PyrokiFinalApproachTool(host=cap_server_host, port=srv_port))
    # Grasp and place — high-level pick-and-place primitives
    registry.register(GraspTool(host=cap_server_host, port=srv_port))
    registry.register(PlaceTool(host=cap_server_host, port=srv_port))
    # Scene object listing via Qwen3-VL
    registry.register(
        ListSceneObjectsTool(
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
        )
    )
    # SAM3 segmentation tools
    registry.register(
        SegmentObjectTool(
            sam3_host=s3_host,
            sam3_port=s3_port,
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
        )
    )
    registry.register(
        SegmentAllObjectsTool(
            sam3_host=s3_host,
            sam3_port=s3_port,
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
        )
    )
    # AnyGrasp grasp planning
    sam3_url = f"http://{s3_host}:{s3_port}"
    registry.register(
        SampleGraspPoseAnyGraspTool(
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
            sam3_url=sam3_url,
        )
    )
    registry.register(
        SampleGraspPose2DTool(
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
            sam3_url=sam3_url,
        )
    )
    registry.register(
        SampleGraspPose3DBBoxTool(
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
            sam3_url=sam3_url,
        )
    )
    return registry
