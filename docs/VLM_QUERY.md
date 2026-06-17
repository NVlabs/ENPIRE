# VLM Query Tool

`vlm_query` is a unified oracle skill for querying vision-language models (VLMs) from
CAP agent oracle code or scripts. It supports six backends selectable at call time,
all sharing the same API.

**Primary implementation**: `cap/agent/tools/vlm_query.py` (`VlmQueryTool` class)

## Backends

| Backend | Deployment | API Key | Default Model | Best For |
|---------|------------|---------|---------------|----------|
| `qwen` **(default)** | Local vLLM via SSH tunnel (port 8402) | None required | `Qwen3-VL-8B-Instruct` | Strong visual reasoning, scene understanding |
| `smol_vlm` | Local vLLM on lab GPU node (port 8401) | None required | `HuggingFaceTB/SmolVLM-256M-Instruct` | Fast on-robot use, single camera |
| `gemini` | Google cloud | `GEMINI_API_KEY` | `gemini-2.5-flash` | Good quality, multi-camera reasoning |
| `gemini_pro` | Google cloud | `GEMINI_API_KEY` | `gemini-3.1-pro-preview` | Best quality, includes thinking/reasoning |
| `nvidia` | NVIDIA inference gateway | `NVIDIA_API_KEY` or `NVIDIA_API_KEY_1..N` | `gcp/google/gemini-3-flash-preview` | Gateway access to Gemini / Bedrock Claude with optional key rotation |
| `gpt` | OpenAI cloud | `OPENAI_API_KEY` | `gpt-5.4` | Strong reasoning with configurable effort |

The default backend is controlled by `DEFAULT_VLM_BACKEND` in `cap/config.py:248`
(env var: `DEFAULT_VLM_BACKEND`, default: `"qwen"`).

## Parameters

Defined at `cap/agent/tools/vlm_query.py:259-317` (`VlmQueryTool.parameters`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | -- | Text prompt or question (required) |
| `backend` | `str` | `"qwen"` | Which VLM to use: `"qwen"`, `"smol_vlm"`, `"gemini"`, `"gemini_pro"`, `"nvidia"`, `"gpt"` |
| `media` | `list[str]` | `None` | Image sources: `"camera:top"`, `"local:~/img.png"`, `"web:https://..."` |
| `camera` | `str` | `None` | Camera view: `"top"`, `"left"`, `"right"`, or `"all"` (deprecated -- use `media`) |
| `image` | `Any` | `None` | Pre-captured numpy RGB array (deprecated -- use `media`) |
| `model` | `str` | `None` | Override the default model name for the chosen backend |
| `temperature` | `float` | `0.2` | Sampling temperature (0.0 = deterministic, 1.0 = creative). Ignored by `gemini_pro` and auto-omitted for NVIDIA Bedrock Claude models, which reject the field |
| `reasoning_effort` | `str` | `"high"` | Thinking effort for `gpt` and `gemini_pro` backends: `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` (GPT only) |

When no image source is specified (no `media`, `camera`, or `image`), the tool defaults to
capturing from the `"top"` camera (`cap/agent/tools/vlm_query.py:484-489`).

## Usage in Oracle Code

```python
# Describe the scene (Qwen3-VL, default backend)
result = vlm_query("What objects are on the table?")
print(result)

# Specify camera
result = vlm_query("What does the gripper hold?", camera="left")

# Binary task-completion check with all cameras
result = vlm_query("Is the red peg fully inserted? Answer YES or NO.", camera="all")
peg_inserted = result.strip().upper().startswith("YES")

# Use Gemini for higher quality reasoning
result = vlm_query(
    text="What should the robot do next to complete the insertion task?",
    backend="gemini",
    camera="all",
)

# Use Gemini Pro with thinking for complex reasoning
result = vlm_query(
    text="Analyze the workspace layout and propose a grasp strategy.",
    backend="gemini_pro",
    media=["camera:top", "camera:left", "camera:right"],
    reasoning_effort="high",
)

# Use GPT for high-quality visual reasoning
result = vlm_query(
    text="Is the object orientation correct for insertion?",
    backend="gpt",
    media=["camera:left"],
    reasoning_effort="medium",
)

# Media interface — mix cameras, local files, and URLs
result = vlm_query(
    text="Is the grasp aligned like the reference?",
    backend="qwen",
    media=["camera:left", "local:cap/tasks/peg_insertion/aligned_grasp.png"],
)

# Pre-captured image (avoids a second RPC round-trip)
img = get_camera_image(camera="top")
result = vlm_query(text="Is the gripper open?", image=img)

# Adjust temperature for more creative responses
result = vlm_query("Suggest three ways to pick up this object.", temperature=0.8)
```

See `cap/saved_scripts/rl/vlm_query_example.py` for a runnable example demonstrating
all backends and media source types.

## Backward Compatibility

### smol_vlm legacy tool

`vlm_query` is a drop-in replacement for the legacy `SmolVlmTool`
(`cap/agent/tools/smol_vlm.py:30`). The legacy tool is no longer registered in the
default tool registry (`cap/agent/tools/__init__.py`); only `VlmQueryTool` is registered
at line 213.

```python
# Old
result = smol_vlm(text="What do you see?", camera="top")

# New (identical behavior)
result = vlm_query(text="What do you see?", backend="smol_vlm", camera="top")
```

### ask_vlm legacy alias

The code executor (`cap/agent/executor.py:284-324`) injects an `ask_vlm` compatibility
wrapper for older saved scripts that used the historical `ask_vlm(prompt, "top,left,right")`
calling convention. It maps comma-separated camera strings and list-of-cameras into the
current `vlm_query` interface:

```python
# Legacy (still works)
result = ask_vlm("describe the scene", "top,left,right")

# Equivalent modern call
result = vlm_query("describe the scene", media=["camera:top", "camera:left", "camera:right"])
```

## Setup

### Qwen3-VL (recommended, default)

Qwen3-VL runs on a remote GPU node via vLLM, accessed through an SSH tunnel.

The launch script is at `tmux/table_bussing/table_bussing_history/launch_qwen_tunnel.sh`.
It tunnels from local port 8402 to the remote vLLM server on port 8000 at
`LeCAR_4xRTX6000BlackWell_97GB`.

```bash
# On remote GPU node — start vLLM (see third_party/vllm_serving/README.md):
./third_party/vllm_serving/server/run_qwen3_vl_server.sh
# or the lean version for constrained resources:
./third_party/vllm_serving/server/run_qwen3_vl_server_lean.sh

# On robot / dev machine — SSH tunnel (local 8402 → remote 8000):
ssh -N -L 8402:localhost:8000 LeCAR_4xRTX6000BlackWell_97GB

# Or use the provided tmux launch script:
bash tmux/table_bussing/table_bussing_history/launch_qwen_tunnel.sh

# Verify:
curl http://localhost:8402/v1/models
```

Override defaults:
```bash
export QWEN_VL_URL="http://localhost:8402/v1"    # default
export QWEN_VL_MODEL="Qwen3-VL-8B-Instruct"      # default
```

### SmolVLM (local)

SmolVLM is served by a vLLM instance on the lab GPU node (LeCAR-S1). The default URL is
`http://172.26.34.251:8401/v1`.

```bash
# Override URL if the server has moved
export SMOL_VLM_URL="http://<host>:8401/v1"

# Start vLLM yourself on a GPU machine (if the lab server is down)
vllm serve HuggingFaceTB/SmolVLM-256M-Instruct --port 8401
```

### Gemini

```bash
export GEMINI_API_KEY="your-key-here"

# Optional: change the default model
export GEMINI_VL_MODEL="gemini-2.5-flash"  # current default
```

> **Note**: `gemini-2.5-flash` is the current default. For sustained RL training with
> free-tier keys, consider `gemini-2.0-flash` (15 req/min, 1500 req/day).

### Gemini Pro (with thinking)

Uses the same `GEMINI_API_KEY` as the `gemini` backend.

```bash
# Optional: override the default model
export GEMINI_PRO_VL_MODEL="gemini-3.1-pro-preview"  # current default
```

The `reasoning_effort` parameter maps to `thinking_budget` tokens
(`cap/agent/tools/vlm_query.py:530-531`):

| `reasoning_effort` | `thinking_budget` |
|---------------------|-------------------|
| `"none"` | 0 |
| `"low"` | 2048 |
| `"medium"` | 8192 |
| `"high"` | 16384 |

### GPT

```bash
export OPENAI_API_KEY="your-key-here"

# Optional: override the default model
export GPT_VL_MODEL="gpt-5.4"  # current default
```

Uses the OpenAI Responses API (`client.responses.create`) with configurable
`reasoning_effort`: `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`
(`cap/agent/tools/vlm_query.py:177-213`). When `reasoning_effort="none"`, temperature
is used instead of reasoning.

### NVIDIA gateway smoke tests

```bash
export NVIDIA_API_KEY="your-key-here"

# List the models visible to this key.
uv run --no-sync python scripts/test_oai_key.py --list-models

# Text smoke test. Defaults to the Opus 4.7 model string used in this repo.
uv run --no-sync python scripts/test_oai_key.py

# Alternative smoke test through the shared repo backend wrapper.
uv run --no-sync python scripts/test_nvidia_llm.py --model azure/anthropic/claude-opus-4-7
```

For Claude Opus 4.7 on the NVIDIA gateway, this repo currently uses
`azure/anthropic/claude-opus-4-7`. If a request fails with a non-JSON body,
`scripts/test_oai_key.py` now prints the HTTP error instead of raising a local
JSON decode exception.

## Architecture

```
Oracle code / CAP agent
        |
        v
  VlmQueryTool.execute()              # cap/agent/tools/vlm_query.py:440
        |
        |-- backend="qwen" ----------> _query_qwen()           (line 148)
        |   (default)                  POST /v1/chat/completions  (vLLM, OpenAI-compat)
        |                              Host: QWEN_VL_URL (localhost:8402 via SSH tunnel)
        |                              api_key: "EMPTY" (no auth needed)
        |                              Uses OpenAI Python SDK
        |
        |-- backend="smol_vlm" ------> _query_smolvlm()        (line 54)
        |                              POST /v1/chat/completions  (vLLM, OpenAI-compat)
        |                              Host: SMOL_VLM_URL
        |                              Uses urllib.request (no SDK dependency)
        |
        |-- backend="gemini" --------> _query_gemini()          (line 90)
        |                              google-genai SDK
        |                              model: GEMINI_VL_MODEL
        |                              auth:  GEMINI_API_KEY
        |
        |-- backend="gemini_pro" ----> _query_gemini_pro()      (line 119)
        |                              google-genai SDK (with ThinkingConfig)
        |                              model: GEMINI_PRO_VL_MODEL
        |                              auth:  GEMINI_API_KEY
        |
        |-- backend="nvidia" --------> _query_nvidia()          (line 91)
        |                              OpenAI SDK via NVIDIA inference gateway
        |                              model: NVIDIA_VL_MODEL
        |                              auth:  NVIDIA_API_KEY / NVIDIA_API_KEY_1..N
        |                              omits temperature for Bedrock Claude models
        |
        +-- backend="gpt" -----------> _query_gpt()             (line 177)
                                       OpenAI Responses API
                                       model: GPT_VL_MODEL
                                       auth:  OPENAI_API_KEY
```

### Image Capture and Resolution

Image capture (`camera="top"/"left"/"right"/"all"`) is handled via Portal RPC to
`cap_server.get_camera_image()` -- the same mechanism used everywhere in CAP.
Available camera names are resolved from the active station profile
(`cap/config.py:213-222`, `CAMERA_NAMES`).

The `media` parameter (`_resolve_media()` at `cap/agent/tools/vlm_query.py:398-438`)
supports three source types:
- `"camera:<name>"` -- live camera capture via Portal RPC
- `"local:<path>"` -- load from local filesystem (supports `~` expansion and project-root-relative paths)
- `"web:<url>"` -- download from URL

When multiple images are supplied, the tool prepends an `[Images: Image 1: camera:top, Image 2: camera:left]`
label header to the prompt so the model knows which image is which
(`cap/agent/tools/vlm_query.py:498-502`).

### Image Encoding

All backends use shared utilities from `cap/utils/image.py`:
- `encode_image_b64(image)` -- numpy RGB -> base64 JPEG (quality=85). Used by `smol_vlm`, `qwen`, `gpt`.
- `encode_image_jpeg(image)` -- numpy RGB -> raw JPEG bytes (quality=85). Used by `gemini`, `gemini_pro`.

### Tool Registration

`VlmQueryTool` is registered in the default tool registry at
`cap/agent/tools/__init__.py:213-217`. The registry's `callable_dict()` method
(line 57-83) wraps it as a plain function `vlm_query(...)` for oracle code injection.

## Related VLM-Powered Tools

### list_scene_objects

`cap/agent/tools/scene_objects.py` -- `ListSceneObjectsTool`

Uses Qwen3-VL to enumerate objects visible in a camera image, returning a structured
`{"objects": [...], "raw_response": "..."}` dict. The response is parsed from JSON arrays,
numbered lists, bullet lists, or comma-separated text.

```python
objects = list_scene_objects()           # ["red mug", "blue plate", ...]
objects = list_scene_objects(camera="left", prompt="List only food items")
```

Registered in the default tool registry at `cap/agent/tools/__init__.py:259-261`.

### save_image

`cap/agent/tools/save_image.py` -- `SaveImageTool`

Uses the same `camera:`/`local:`/`web:` media source prefix convention as `vlm_query`
to capture and save images to local files.

## VLM-Based Reward Functions

The VLM infrastructure is also used for RL reward computation. Two reward backends
send camera images to VLMs and parse binary YES/NO task-completion judgments:

### Gemini Reward

`cap/reward/gemini_reward.py` -- `vlm_reward(obs)`

- Sends all three camera images (top, left, right) to Gemini
- Uses model `GEMINI_REWARD_MODEL` (env var, default: `gemini-2.5-flash`)
- Rate-limited to one API call every `GEMINI_REWARD_INTERVAL_S` seconds (default: 5.0)
- Prompt template loaded from `cap/prompt/reward_gemini.json` via `cap/utils/prompt_loader.py`
- Returns 1.0 (success) or 0.0 (failure), caches between API calls

### SmolVLM Reward

`cap/reward/smolvlm_reward.py` -- `smolvlm_reward(obs)`

- Sends camera images to local SmolVLM vLLM server
- Uses vLLM `guided_choice: ["YES", "NO"]` for constrained binary output
- Rate-limited to `SMOLVLM_REWARD_INTERVAL_S` seconds (default: 0.5)
- Prompt template loaded from `cap/prompt/reward_smolvlm.json`
- Includes keyword-based fallback parsing for models that ignore guided_choice

### Reward Server

`cap/reward/reward_server.py` -- Portal RPC server exposing `get_reward(obs) -> float`

```bash
uv run cap/reward/reward_server.py --mode gemini    # Gemini VLM reward
uv run cap/reward/reward_server.py --mode smolvlm   # SmolVLM reward
```

Both VLM reward modes are lazily imported at `cap/reward/reward_server.py:111-116`.

## MCP and Bridge Integration

### MCP Server

`cap/bridge/cap_mcp_server.py:102-139` exposes a simplified `vlm_query(text, camera)`
MCP tool for Claude Code. This version uses the Qwen backend only (hardcoded), fetching
the camera image via the cap_agent REST API rather than Portal RPC.

### Agent Bridge

`cap/bridge/agent_bridge.py:172-173` declares `vlm_query(text, backend, camera)` in the
bridge tool signature for external agent backends (Claude Code, OpenAI Codex).

### System Prompt

`cap/bridge/system_prompt.py:90-93` documents `vlm_query` usage in the system prompt
given to LLM agent backends, advising use of `camera="all"` for multi-camera views.

### Claude Code Provider

`cap/bridge/providers/claude_code.py:26` maps the MCP tool name
`mcp__cap-robot__vlm_query` for Claude Code tool-use.

## Configuration Reference

All defaults live in `cap/config.py:237-258` and can be overridden with environment variables:

| Config constant | Env var override | Default | Line |
|----------------|-----------------|---------|------|
| `DEFAULT_VLM_BACKEND` | `DEFAULT_VLM_BACKEND` | `"qwen"` | 248 |
| `QWEN_VL_URL` | `QWEN_VL_URL` | `http://localhost:8402/v1` | 251 |
| `QWEN_VL_MODEL` | `QWEN_VL_MODEL` | `Qwen3-VL-8B-Instruct` | 252 |
| `SMOL_VLM_URL` | `SMOL_VLM_URL` | `http://172.26.34.251:8401/v1` | 240 |
| `SMOL_VLM_MODEL` | -- | `HuggingFaceTB/SmolVLM-256M-Instruct` | 241 |
| `GEMINI_VL_MODEL` | `GEMINI_VL_MODEL` | `gemini-2.5-flash` | 245 |
| `GEMINI_PRO_VL_MODEL` | `GEMINI_PRO_VL_MODEL` | `gemini-3.1-pro-preview` | 255 |
| `GPT_VL_MODEL` | `GPT_VL_MODEL` | `gpt-5.4` | 258 |
| `GEMINI_API_KEY` | `GEMINI_API_KEY` | -- (required for gemini/gemini_pro) | -- |
| `OPENAI_API_KEY` | `OPENAI_API_KEY` | -- (required for gpt) | -- |
| `CAMERA_NAMES` | -- (derived from station profile) | `("top", "left", "right")` | 222 |

## File Reference

| File | Purpose |
|------|---------|
| `cap/agent/tools/vlm_query.py` | Primary unified VLM tool (5 backends) |
| `cap/agent/tools/smol_vlm.py` | Legacy SmolVLM-only tool (not registered, kept for reference) |
| `cap/agent/tools/scene_objects.py` | Qwen3-VL-powered scene object listing |
| `cap/agent/tools/save_image.py` | Image save tool (shares media prefix convention) |
| `cap/agent/tools/base.py` | `Tool`, `ToolParameter`, `ToolResult` base classes |
| `cap/agent/tools/__init__.py` | Tool registry and `VlmQueryTool` registration |
| `cap/agent/executor.py` | Code executor with `ask_vlm` backward-compat wrapper |
| `cap/utils/image.py` | Shared `encode_image_b64()` and `encode_image_jpeg()` |
| `cap/config.py` | All VLM backend URL/model constants |
| `cap/reward/gemini_reward.py` | Gemini-based VLM reward function |
| `cap/reward/smolvlm_reward.py` | SmolVLM-based VLM reward function |
| `cap/reward/reward_server.py` | Portal RPC reward server (modes: `gemini`, `smolvlm`) |
| `cap/prompt/reward_gemini.json` | Gemini reward prompt template |
| `cap/prompt/reward_smolvlm.json` | SmolVLM reward prompt template |
| `cap/utils/prompt_loader.py` | Reward prompt template loader |
| `cap/bridge/cap_mcp_server.py` | MCP server VLM tool for Claude Code |
| `cap/bridge/agent_bridge.py` | Agent bridge VLM tool signature |
| `cap/bridge/system_prompt.py` | System prompt mentioning vlm_query usage |
| `cap/saved_scripts/rl/vlm_query_example.py` | Runnable example script |
| `third_party/vllm_serving/README.md` | vLLM server setup instructions |
| `tmux/table_bussing/table_bussing_history/launch_qwen_tunnel.sh` | SSH tunnel launch script |

## Cross-References

- **CAP Design**: `docs/CAP_DESIGN.md` -- overall CAP system architecture
- **RL Pipeline**: `docs/RL_PIPELINE_DESIGN.md` -- RL training pipeline using VLM rewards
- **Table Bussing Skills**: `docs/TABLE_BUSSING_SKILLS.md` -- task skills that use VLM queries
- **Multi-Camera Config**: `docs/MULTI_CAMERA_CONFIG.md` -- camera setup for VLM image capture
- **Remote Serving**: `docs/remote_serving.md` -- remote model server architecture
- **CAP UI Design**: `docs/CAP_UI_DESIGN.md` -- UI integration with VLM responses
- **BundleSDF Object Detection**: `docs/BUNDLESDF_OBJECT_DETECTION.md` -- related vision pipeline
