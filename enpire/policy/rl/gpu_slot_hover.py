# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
import sys
import types
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.base import ArmState, RobotState
from enpire.env.forge.cap.agent.tools.segmentation import SegmentObjectTool
from enpire.env.forge.paths import FORGE_ROOT
from enpire.env.forge.robot.yam.kinematics import _quat_xyzw_to_rpy_display

DEFAULT_REAL_YAM_SAM3_HOST = "127.0.0.1"
DEFAULT_REAL_YAM_SAM3_PORT = 6767


def check_gpu_slot_hover_dependencies(
    cfg: Any | None = None,
    *,
    timeout_s: float = 1.0,
) -> None:
    """Fail before robot motion if slot-hover perception services are missing."""
    del cfg
    host = _sam3_host()
    port = _sam3_port()
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            if int(getattr(response, "status", 200)) >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(
            "GPU slot-hover reset requires the SAM3 segmentation server, but "
            f"{url} is not reachable ({exc}). Start it from the forge repo with:\n"
            f"  uv run python cap/skills/serve_sam3.py --port {port} --preload\n"
            "or set SAM3_SERVER_HOST/SAM3_SERVER_PORT to the running server. "
            "Set GPU_RL_REQUIRE_SAM3=0 only if you intentionally disabled "
            "GPU slot relocalization."
        ) from exc


def _sam3_host() -> str:
    return os.environ.get("SAM3_SERVER_HOST", DEFAULT_REAL_YAM_SAM3_HOST)


def _sam3_port() -> int:
    return int(os.environ.get("SAM3_SERVER_PORT", str(DEFAULT_REAL_YAM_SAM3_PORT)))


def move_to_gpu_slot_hover(ctx: Any) -> dict[str, Any]:
    """Relocalize the GPU socket and move the left arm to the slot hover pose."""
    env = getattr(ctx.env, "unwrapped", ctx.env)
    _require_gpu_hover_env(env)
    with _hover_env(ctx.cfg):
        helpers = _load_left_slot_hover_helpers(env)
        print(
            "[gpu_rl_reset] Relocalizing slot hover with reset-aligned helper: "
            f"camera={helpers.CAMERA!r} aux_camera={helpers.AUX_CAMERA!r} "
            f"target_socket={int(helpers.TARGET_SOCKET_NUMBER)}",
            flush=True,
        )
        reference_scene = helpers._build_initial_reference_scene(
            camera=helpers.CAMERA
        )
        hover_rpy = helpers._resolve_hover_rpy()
        hover_pos, scene = helpers._acquire_initial_hover(
            reference_scene=reference_scene,
            hover_rpy=hover_rpy,
            camera=helpers.CAMERA,
        )
    result = {
        "target_socket": int(helpers.TARGET_SOCKET_NUMBER),
        "hover_pos": [float(v) for v in hover_pos],
        "hover_rpy": [float(v) for v in hover_rpy],
        "pose_source_camera": scene.get("pose_source_camera"),
        "localization_mode": scene.get("socket_hover_localization_mode", "edge"),
    }
    print(
        "[gpu_rl_reset] Slot hover acquired: "
        f"pos={[round(float(v), 4) for v in result['hover_pos']]} "
        f"rpy={[round(float(v), 1) for v in result['hover_rpy']]} "
        f"mode={result['localization_mode']!r}",
        flush=True,
    )
    return result


def _require_gpu_hover_env(env: Any) -> None:
    required = (
        "render_rgb",
        "render_depth",
        "get_camera_intrinsics",
        "get_camera_extrinsics",
        "get_observations",
        "move_bimanual_joint_keypoints",
    )
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise RuntimeError(
            "GPU slot-hover reset requires YamRealEnv direct camera/motion APIs; "
            f"missing {missing}"
        )


@contextmanager
def _hover_env(cfg: Any):
    env_vars = {
        "GPU_SLOT_DEBUG_CAMERA": getattr(cfg, "gpu_slot_hover_camera", "top"),
        "GPU_SLOT_DEBUG_AUX_CAMERA": getattr(cfg, "gpu_slot_hover_aux_camera", "left_third"),
        "GPU_SLOT_DEBUG_AUX_CAMERA_PREFER_WORLD_POSE": _flag_str(
            getattr(cfg, "gpu_slot_hover_aux_prefer_world_pose", True)
        ),
        "GPU_SLOT_DEBUG_REQUIRE_AUX_CAMERA": _flag_str(
            getattr(cfg, "gpu_slot_hover_aux_required", False)
        ),
        "GPU_SLOT_DEBUG_SAVE_ARTIFACTS": _flag_str(
            getattr(cfg, "gpu_slot_hover_save_artifacts", True)
        ),
        "GPU_SLOT_DEBUG_GO_HOME_ON_START": "0",
        "GPU_SLOT_DEBUG_CLOSE_LEFT_GRIPPER_ON_START": "0",
        "GPU_REACTIVE_SOCKET_HOVER_PERIOD_S": "0",
        "GPU_SLOT_DEBUG_MAX_HOVER_XY_DELTA_M": os.environ.get(
            "GPU_RL_SLOT_HOVER_MAX_XY_DELTA_M",
            "0.24",
        ),
        "GPU_SLOT_DEBUG_AUX_CAMERA_MAX_POSE_DIFF_M": os.environ.get(
            "GPU_RL_SLOT_HOVER_AUX_MAX_POSE_DIFF_M",
            "0.12",
        ),
        "GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_AREA_PX": os.environ.get(
            "GPU_RL_SLOT_HOVER_MIN_MOTHERBOARD_BBOX_AREA_PX",
            "15000",
        ),
        "GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_WIDTH_PX": os.environ.get(
            "GPU_RL_SLOT_HOVER_MIN_MOTHERBOARD_BBOX_WIDTH_PX",
            "110",
        ),
        "GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_HEIGHT_PX": os.environ.get(
            "GPU_RL_SLOT_HOVER_MIN_MOTHERBOARD_BBOX_HEIGHT_PX",
            "100",
        ),
    }
    old = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _flag_str(value: Any) -> str:
    return "1" if bool(value) else "0"


def _load_left_slot_hover_helpers(env: Any) -> types.ModuleType:
    script_path = (
        FORGE_ROOT
        / "cap"
        / "saved_scripts"
        / "gpu"
        / "gpu_debug_left_slot_hover.py"
    )
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    keep = []
    for idx, node in enumerate(tree.body):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            keep.append(node)
        elif (
            idx == 0
            and isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            keep.append(node)

    module_ast = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    helpers = types.ModuleType("gpu_left_slot_hover_helpers_rl")
    helpers.__file__ = str(script_path)
    helpers.__dict__.update({"__builtins__": __builtins__})
    helpers.__dict__.update(_tool_callables(env))
    exec(compile(module_ast, str(script_path), "exec"), helpers.__dict__)  # noqa: S102
    return helpers


def _tool_callables(env: Any) -> dict[str, Any]:
    segment_tool = SegmentObjectTool(
        env=env,
        sam3_host=_sam3_host(),
        sam3_port=_sam3_port(),
    )

    def get_robot_state() -> RobotState:
        arms: dict[str, ArmState] = {}
        for side in ("left", "right"):
            obs = env.get_observations(side)
            quat = np.asarray(obs["ee_quat"], dtype=np.float64).reshape(4)
            arms[side] = ArmState(
                joint_pos=list(np.asarray(obs["joint_pos"], dtype=np.float64).reshape(6)),
                gripper_pos=float(
                    np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(-1)[0]
                ),
                ee_pos=list(np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3)),
                ee_quat=list(quat),
                ee_rpy=list(_quat_xyzw_to_rpy_display(quat)),
            )
        return RobotState(arms=arms)

    def get_camera_image(camera: str) -> np.ndarray:
        return env.render_rgb(camera)

    def render_depth(camera: str) -> np.ndarray | None:
        return env.render_depth(camera)

    def get_camera_intrinsics(camera: str) -> list[float]:
        return env.get_camera_intrinsics(camera)

    def get_camera_extrinsics(camera: str) -> dict:
        return env.get_camera_extrinsics(camera)

    def segment_object(**kwargs: Any) -> Any:
        result = segment_tool.execute(**kwargs)
        if not result.success:
            raise RuntimeError(f"Tool segment_object failed: {result.error}")
        return result.data

    def _unavailable_tool(*_: Any, **__: Any) -> None:
        raise RuntimeError("tool is unavailable in RL slot-hover reset")

    tools = {
        "close_gripper": _unavailable_tool,
        "detect_object": _unavailable_tool,
        "end_detection": _unavailable_tool,
        "freespace_move": _unavailable_tool,
        "get_camera_extrinsics": get_camera_extrinsics,
        "get_camera_image": get_camera_image,
        "get_camera_intrinsics": get_camera_intrinsics,
        "get_robot_state": get_robot_state,
        "go_home": _unavailable_tool,
        "open_gripper": _unavailable_tool,
        "render_depth": render_depth,
        "sample_grasp_pose_3d_bb": _unavailable_tool,
        "segment_object": segment_object,
        "vlm_query": _unavailable_tool,
    }
    for fn in tools.values():
        if callable(fn):
            setattr(fn, "_env", env)

    skill_library = sys.modules.setdefault(
        "skill_library",
        types.ModuleType("skill_library"),
    )
    namespace = sys.modules.setdefault(
        "skill_library.namespace",
        types.ModuleType("skill_library.namespace"),
    )
    for name, fn in tools.items():
        setattr(namespace, name, fn)
    namespace.__all__ = sorted(tools)
    setattr(skill_library, "namespace", namespace)
    return tools
