# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing read-only robot tools to Claude Code.

Runs as a stdio MCP server. Claude Code connects to it via mcp_config.json.
All tools are read-only — they query cap_agent's REST API (via SSH tunnel).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

# cap_agent REST API base URL (through SSH tunnel on laptop)
CAP_AGENT_URL = os.environ.get("CAP_AGENT_URL", "http://localhost:8200")

mcp = FastMCP("cap-robot", instructions="Read-only tools to inspect the YAM bimanual robot.")


def _api_get(path: str) -> dict[str, Any]:
    """Make a GET request to cap_agent REST API."""
    url = f"{CAP_AGENT_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


@mcp.tool()
def get_robot_state() -> str:
    """Get the current robot state including joint positions, end-effector poses, and gripper states for both arms.

    Returns a JSON object with 'left' and 'right' arm states, each containing:
    - joint_positions: list of 6 joint angles in radians
    - gripper: gripper position (-1.0 closed to 1.0 open)
    - ee_pose: {position: [x,y,z], quaternion: [x,y,z,w]}
    """
    data = _api_get("/api/state")
    if not data.get("ok"):
        return f"Error: {data.get('error', 'unknown')}"
    data.pop("ok", None)
    return json.dumps(data, indent=2, default=str)


@mcp.tool()
def get_camera_image(camera: str = "top") -> list:
    """Get a camera image from the robot. Returns a base64-encoded JPEG image.

    Args:
        camera: Camera name - one of "top", "left", "right". Defaults to "top".

    The image shows the robot's workspace from the specified camera angle.
    Use this to understand the scene before writing manipulation code.
    """
    if camera not in ("top", "left", "right"):
        return [{"type": "text", "text": f"Error: unknown camera '{camera}'. Use 'top', 'left', or 'right'."}]

    data = _api_get(f"/api/camera/{camera}")
    if not data.get("ok"):
        return [{"type": "text", "text": f"Error: {data.get('error', 'unknown')}"}]

    b64_jpeg = data.get("image", "")
    return [
        {"type": "text", "text": f"Camera: {camera}"},
        {"type": "image", "data": b64_jpeg, "mimeType": "image/jpeg"},
    ]


@mcp.tool()
def list_saved_scripts() -> str:
    """List all saved robot scripts. Returns names, sizes, and modification dates.

    Use this to check if there's existing code that can be reused or adapted
    for the current task.
    """
    data = _api_get("/api/scripts")
    if not data:
        return "No saved scripts found."
    lines = []
    for s in data:
        lines.append(f"- {s['name']} ({s['size']} bytes, modified {s['modified']})")
    return "\n".join(lines)


@mcp.tool()
def read_saved_script(name: str) -> str:
    """Read the source code of a saved robot script.

    Args:
        name: Script name (without .py extension).

    Returns the Python source code of the script.
    """
    data = _api_get(f"/api/scripts/{name}")
    if not data.get("ok"):
        return f"Error: {data.get('error', 'not found')}"
    return data.get("code", "")


@mcp.tool()
def vlm_query(text: str, camera: str = "top") -> str:
    """Query a vision-language model about the robot's camera view.

    Args:
        text: Text prompt / question about the scene.
        camera: Camera name ("top", "left", "right"). Defaults to "top".

    Returns the VLM's text response.
    Use for scene understanding, counting objects, reading text, or visual reasoning.
    """
    # Get image from cap_agent
    cam_data = _api_get(f"/api/camera/{camera}")
    if not cam_data.get("ok"):
        return f"Error getting camera image: {cam_data.get('error', 'unknown')}"

    b64_jpeg = cam_data.get("image", "")

    from enpire.env.forge.cap.config import QWEN_VL_MODEL, QWEN_VL_URL

    try:
        from openai import OpenAI
        client = OpenAI(base_url=QWEN_VL_URL, api_key="EMPTY", timeout=30.0)
        resp = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}},
                    {"type": "text", "text": text},
                ],
            }],
            max_tokens=512,
            temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"VLM query failed: {e}. Is the SSH tunnel running?"


if __name__ == "__main__":
    mcp.run(transport="stdio")
