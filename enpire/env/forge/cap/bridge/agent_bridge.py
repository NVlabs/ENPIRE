"""Generic CAP agent bridge with switchable Claude Code and OpenAI Codex backends."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enpire.env.forge.cap.bridge.providers import (
    AgentBackend,
    AgentBridgeConfig,
    ChatSession,
    ClaudeCodeBackend,
    OpenAICodexBackend,
    ProviderContext,
)
from enpire.env.forge.cap.bridge.system_prompt import build_system_prompt
from enpire.env.forge.cap.chat import extract_code_blocks
from enpire.env.forge.cap.config import (
    BRIDGE_PORT,
    CAP_AGENT_NAME,
    CAP_AGENT_PORT,
    CAP_VOICE_HOST,
    CAP_VOICE_PORT,
)
from enpire.env.forge.cap.voice import ChatVoiceController, VoiceOutputManager

logger = logging.getLogger(__name__)

CAP_AGENT_URL = f"http://localhost:{CAP_AGENT_PORT}"
VOICE_INPUT_API_URL = os.environ.get(
    "CAP_VOICE_API_URL",
    f"http://{CAP_VOICE_HOST}:{CAP_VOICE_PORT}/api",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_CONFIG = PROJECT_ROOT / "cap" / "bridge" / "mcp_config.json"
CONVERSATION_DIR = PROJECT_ROOT / "conversation"
PROMPT_DIR = PROJECT_ROOT / "cap" / "prompt"
PER_MESSAGE_DIR = PROMPT_DIR / "system"


class ChatRequest(BaseModel):
    message: str
    backend: str | None = None
    model: str | None = None
    reasoning: str | None = None


class SessionConfigRequest(BaseModel):
    backend: str | None = None
    model: str | None = None
    reasoning: str | None = None


class OkResponse(BaseModel):
    ok: bool


class VoiceRequest(BaseModel):
    text: str


class VoiceEnabledRequest(BaseModel):
    enabled: bool


def _fetch_tool_docs() -> str:
    try:
        url = f"{CAP_AGENT_URL}/api/state"
        req = urllib.request.Request(url)
        urllib.request.urlopen(req, timeout=3).close()
    except Exception:
        logger.warning("cap_agent not reachable — using placeholder tool docs")
        return "(Tool documentation unavailable — cap_agent not connected)"

    return _STATIC_TOOL_DOCS


def _stop_voice_input_capture() -> None:
    try:
        req = urllib.request.Request(f"{VOICE_INPUT_API_URL}/stop", method="POST")
        urllib.request.urlopen(req, timeout=3).close()
    except Exception:
        logger.debug(
            "Voice input server unavailable while suppressing mic capture",
            exc_info=True,
        )


_STATIC_TOOL_DOCS = """\
def get_robot_state() -> RobotState:
    \"\"\"Get the current state of both robot arms. Returns a RobotState with
    left/right joint positions, gripper positions, end-effector poses (xyz + rpy in degrees).\"\"\"

def freespace_move(left_target_pos=None, left_target_rpy=None, right_target_pos=None, right_target_rpy=None, left_gripper=None, right_gripper=None, left_gripper_target_width=None, right_gripper_target_width=None, planning_speed=1.5):
    \"\"\"Move arm(s) to target pose via collision-free motion planning (cuRobo by default).
    Provide left_target_pos/rpy and/or right_target_pos/rpy for single or bimanual movement.
    pos=[x,y,z] in metres. rpy=[roll,pitch,yaw] in degrees (home orientation is [0,90,0]).
    left_gripper/right_gripper: 0=closed, 1=open (for collision checking).
    left_gripper_target_width/right_gripper_target_width: optional synchronized gripper
    target widths (0=closed, 1=open) to execute during the same move. Blocking.
    planning_speed: commanded max joint speed in rad/s, clamped to [0.05, 2.0], default 1.5.
    Bimanual moves execute in synchronized lockstep, matching the control-loop UI planner.
    For a single-arm move, omit the inactive arm entirely instead of redundantly
    passing its current pose. If you don't need to go to a specific target rpy,
    keep the active arm's current orientation by passing the current rpy from
    get_robot_state():
        state = get_robot_state()
        freespace_move(right_target_pos=target_pos, right_target_rpy=state.right_ee_rpy)\"\"\"

def nudge(side: str, delta_pos: list[float] = None, delta_rpy: list[float] = None):
    \"\"\"Apply a small delta movement to one arm's end-effector.
    delta_pos=[dx,dy,dz] in metres, delta_rpy=[droll,dpitch,dyaw] in degrees.
    Both in world frame (+X forward, +Y left, +Z up).
    Ideal for fine adjustments after freespace_move — e.g. using wrist camera feedback
    to refine position before grasping.\"\"\"

def set_gripper(side: str = "left", pos: float = 1.0, vel_limit: float | None = None, torque_limit: float | None = None):
    \"\"\"Set gripper position. pos in [-1.0, 1.0], where 1.0=open and -1.0=closed.
    vel_limit is rad/s and torque_limit is Nm. Blocking.\"\"\"

def open_gripper(side: str = "left", vel_limit: float | None = None, torque_limit: float | None = None):
    \"\"\"Open the gripper. side='left' or 'right'. vel_limit in rad/s, torque_limit in Nm.\"\"\"

def close_gripper(side: str = "left", vel_limit: float | None = None, torque_limit: float | None = None):
    \"\"\"Close the gripper. side='left' or 'right'. vel_limit in rad/s, torque_limit in Nm.\"\"\"

def grasp(side: str, position: list[float], rpy: list[float] = None, pre_height: float = 0.10, z_offset: float = 0.05):
    \"\"\"Grasp an object: open gripper, approach from above, descend, close gripper, lift.
    side: 'left' or 'right'. position: [x,y,z] from detect_object. rpy: [roll,pitch,yaw]
    in degrees (default: current arm RPY). pre_height: hover distance above object (m).
    z_offset: safety offset added to Z (m). Blocking.\"\"\"

def place(side: str, position: list[float], rpy: list[float] = None, pre_height: float = 0.15):
    \"\"\"Place a held object: move above target, descend, open gripper, lift away.\"\"\"

def go_home():
    \"\"\"Send both arms to the home (zero) configuration. Blocking.\"\"\"

def get_camera_image(camera: str = "top"):
    \"\"\"Get a camera image. camera='top', 'left', or 'right'. Returns numpy RGB array.\"\"\"

def detect_object(query: str, camera: str = "top", max_retries: int = 3) -> list[Detection3D]:
    \"\"\"Detect objects via BundleSDF 6-DOF pose tracking. Auto-retries on failure.\"\"\"

def execute_skill(skill_name: str, **kwargs):
    \"\"\"Execute a learned skill by name (e.g. a flow-matching policy).\"\"\"

def start_policy_output(model, replan_horizon=30, task_description="", policy_server=""):
    \"\"\"Start a warm external policy-output session but do not move the robot yet.\"\"\"

def step_policy_output(max_steps=None):
    \"\"\"Execute a bounded burst from the active policy-output session.\"\"\"

def stop_policy_output():
    \"\"\"Stop the active policy-output session and clear CAP-side queued policy state.\"\"\"

def use_policy_output(model, replan_horizon=30, task_description="", policy_server=""):
    \"\"\"One-shot wrapper around start_policy_output(), step_policy_output(), and stop_policy_output().\"\"\"

def vlm_query(text: str, backend: str = "qwen", camera: str = "top") -> str:
    \"\"\"Query a vision-language model about the scene.\"\"\"

def wait_for_agent(message: str = "") -> None:
    \"\"\"Pause code execution and yield control back to the agent for replanning.\"\"\"

def setup_scene(name: str) -> dict:
    \"\"\"Load a named scene into the simulation (sim-only).\"\"\"

def clear_table() -> dict:
    \"\"\"Remove all scene objects from the simulation table (sim-only).\"\"\"

def list_scenes() -> dict:
    \"\"\"List available scene files and the currently active scene (sim-only).\"\"\"
"""


def build_provider_registry() -> dict[str, AgentBackend]:
    providers: list[AgentBackend] = [ClaudeCodeBackend(), OpenAICodexBackend()]
    return {provider.spec.backend: provider for provider in providers}


def _normalize_backend_id(backend: str | None) -> str:
    aliases = {
        None: "claude_code",
        "claude": "claude_code",
        "claude_code": "claude_code",
        "openai_codex": "openai_codex",
        "codex": "openai_codex",
    }
    return aliases.get(backend, backend or "claude_code")


@dataclass
class BridgeWSManager:
    _connections: list[WebSocket] = field(default_factory=list)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, event_type: str, data: Any) -> None:
        import json

        payload = json.dumps(
            {"type": event_type, "data": data, "timestamp": time.strftime("%H:%M:%S")}
        )
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._connections.remove(ws)


@dataclass
class BridgeState:
    providers: dict[str, AgentBackend]
    conversation_dir: Path
    active_config: AgentBridgeConfig = field(
        default_factory=lambda: AgentBridgeConfig(
            backend="claude_code",
            model="claude-opus-4-6",
            reasoning="medium",
        )
    )
    sessions: dict[str, ChatSession] = field(default_factory=dict)

    def get_session(self, backend: str) -> ChatSession:
        if backend not in self.sessions:
            self.sessions[backend] = ChatSession(conversation_dir=self.conversation_dir)
        return self.sessions[backend]

    def normalize_config(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> AgentBridgeConfig:
        backend_id = _normalize_backend_id(backend or self.active_config.backend)
        provider = self.providers[backend_id]
        normalized_model, normalized_reasoning = provider.spec.normalize(
            model=model or (self.active_config.model if backend is None else None),
            reasoning=reasoning
            or (self.active_config.reasoning if backend is None else None),
        )
        return AgentBridgeConfig(
            backend=backend_id,
            model=normalized_model,
            reasoning=normalized_reasoning,
        )

    def apply_request(
        self, req: SessionConfigRequest | ChatRequest
    ) -> AgentBridgeConfig:
        config = self.normalize_config(
            backend=req.backend,
            model=req.model,
            reasoning=req.reasoning,
        )
        self.active_config = config
        return config

    def options_payload(self) -> dict[str, Any]:
        return {
            "agent_name": CAP_AGENT_NAME,
            "current": self.active_config.to_dict(),
            "backends": [
                provider.spec.to_dict() for provider in self.providers.values()
            ],
        }


@dataclass
class _TurnWSManager:
    inner: BridgeWSManager
    chat_voice: ChatVoiceController

    async def broadcast(self, event_type: str, data: Any) -> None:
        if event_type == "chat_text_delta":
            text = data.get("text", "") if isinstance(data, dict) else ""
            self.chat_voice.on_text_delta(text if isinstance(text, str) else "")
        elif event_type == "chat_error":
            self.chat_voice.on_turn_error()
        await self.inner.broadcast(event_type, data)


def _compose_prompt(req_message: str) -> str:
    full_message = f"User request: {req_message}"
    # Load per-message API notes from system/per_message_api.md
    api_file = PER_MESSAGE_DIR / "per_message_api.md"
    if api_file.exists():
        content = api_file.read_text(encoding="utf-8").strip()
        full_message += (
            "\n\n---\n\n[Per-message task notes]\n\n"
            f"### {api_file.stem}\n\n{content}"
        )
    return full_message


def create_app() -> FastAPI:
    providers = build_provider_registry()
    state = BridgeState(providers=providers, conversation_dir=CONVERSATION_DIR)
    default_backend = (
        "claude_code" if "claude_code" in providers else next(iter(providers))
    )
    default_provider = providers[default_backend]
    state.active_config = state.normalize_config(
        backend=default_backend,
        model=default_provider.spec.default_model,
        reasoning=default_provider.spec.default_reasoning,
    )
    context = ProviderContext(
        project_root=PROJECT_ROOT,
        cap_agent_url=CAP_AGENT_URL,
        mcp_config_path=MCP_CONFIG,
    )
    ws_manager = BridgeWSManager()
    bridge_voice_enabled = os.environ.get("CAP_BRIDGE_VOICE_ENABLED")
    if bridge_voice_enabled is None and os.environ.get("PYTEST_CURRENT_TEST"):
        bridge_voice_enabled = "0"
    voice_enabled = (bridge_voice_enabled or "1").lower() not in {"0", "false", "no"}
    chat_voice = ChatVoiceController(
        voice_output=VoiceOutputManager(
            enabled=voice_enabled,
            before_speak=_stop_voice_input_capture,
        )
    )
    generating = False
    job_queue: asyncio.Queue = asyncio.Queue()
    worker_task: asyncio.Task | None = None

    tool_docs = _fetch_tool_docs()
    system_prompt = build_system_prompt(tool_docs)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal worker_task
        worker_task = asyncio.create_task(_worker_loop())
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
                worker_task = None
            chat_voice.shutdown()

    app = FastAPI(title="CAP Agent Bridge", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _finalize_turn(
        *,
        session: ChatSession,
        prompt_message: str,
        full_text: str,
    ) -> None:
        for block in extract_code_blocks(full_text):
            await ws_manager.broadcast(
                "chat_code_block",
                {"code": block.code, "language": block.language},
            )
        session.log_user(prompt_message)
        session.log_assistant(full_text)
        session.append_message("user", prompt_message)
        session.append_message("assistant", full_text)
        try:
            chat_voice.on_turn_complete(full_text)
        except Exception:
            logger.exception("Failed to queue assistant text for voice output")
        await ws_manager.broadcast(
            "chat_turn_complete", {"session_id": session.session_id}
        )

    async def _worker_loop() -> None:
        while True:
            job = await job_queue.get()
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Bridge worker job failed")
            finally:
                job_queue.task_done()

    @app.get("/api/agent/options")
    async def agent_options() -> dict[str, Any]:
        return state.options_payload()

    @app.get("/api/voice/status")
    async def voice_status() -> dict[str, Any]:
        status = chat_voice.get_status()
        return {
            "enabled": status.enabled,
            "speaking": status.speaking,
            "last_spoken_text": status.last_spoken_text,
            "last_error": status.last_error,
        }

    @app.post("/api/voice/enabled", response_model=OkResponse)
    async def set_voice_enabled(req: VoiceEnabledRequest) -> OkResponse:
        chat_voice.set_enabled(req.enabled)
        return OkResponse(ok=True)

    @app.post("/api/voice/test", response_model=OkResponse)
    async def voice_test(req: VoiceRequest | None = None) -> OkResponse:
        text = req.text if req is not None else "Hello World"
        return OkResponse(ok=chat_voice.speak_text(text))

    @app.post("/api/voice/speak", response_model=OkResponse)
    async def voice_speak(req: VoiceRequest) -> OkResponse:
        return OkResponse(ok=chat_voice.speak_text(req.text))

    @app.post("/api/voice/stop", response_model=OkResponse)
    async def voice_stop() -> OkResponse:
        chat_voice.stop()
        return OkResponse(ok=True)

    @app.post("/api/chat", response_model=OkResponse)
    async def chat(req: ChatRequest) -> OkResponse:
        nonlocal generating
        if generating:
            return OkResponse(ok=False)

        generating = True
        config = state.apply_request(req)
        provider = state.providers[config.backend]
        session = state.get_session(config.backend)
        full_message = _compose_prompt(req.message)
        logger.info(
            "Chat request via %s model=%s reasoning=%s: %s",
            config.backend,
            config.model,
            config.reasoning,
            full_message[:200] + ("..." if len(full_message) > 200 else ""),
        )

        async def _run() -> None:
            nonlocal generating
            try:
                result = await provider.run_turn(
                    message=full_message,
                    session=session,
                    ws_manager=_TurnWSManager(inner=ws_manager, chat_voice=chat_voice),
                    system_prompt=system_prompt,
                    config=config,
                    context=context,
                )
                await _finalize_turn(
                    session=session,
                    prompt_message=full_message,
                    full_text=result.full_text,
                )
            except Exception as e:
                logger.exception("Bridge turn failed")
                chat_voice.on_turn_error()
                await ws_manager.broadcast("chat_error", {"error": str(e)})
            finally:
                generating = False

        await job_queue.put(_run)
        return OkResponse(ok=True)

    @app.post("/api/chat/reset", response_model=OkResponse)
    async def reset_chat(req: SessionConfigRequest | None = None) -> OkResponse:
        nonlocal generating
        req = req or SessionConfigRequest()
        config = state.apply_request(req)
        state.get_session(config.backend).reset()
        generating = False
        await ws_manager.broadcast("chat_turn_complete", {"session_id": None})
        return OkResponse(ok=True)

    @app.get("/api/chat/status")
    async def chat_status() -> dict[str, Any]:
        session = state.get_session(state.active_config.backend)
        return {
            "generating": generating,
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "backend": state.active_config.backend,
            "model": state.active_config.model,
            "reasoning": state.active_config.reasoning,
        }

    @app.get("/api/prompts")
    async def list_prompts() -> list[dict[str, str]]:
        PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        results = []
        for subdir in ("system", "tools", "embodiment", "task", "heuristics"):
            d = PROMPT_DIR / subdir
            if d.is_dir():
                for f in sorted(d.glob("*.md")):
                    results.append(
                        {"name": f.stem, "filename": f.name, "category": subdir}
                    )
        # Also include any root-level .md files
        for f in sorted(PROMPT_DIR.glob("*.md")):
            results.append({"name": f.stem, "filename": f.name, "category": "root"})
        return results

    @app.get("/api/prompts/{name}")
    async def get_prompt(name: str) -> dict[str, Any]:
        # Search all subfolders then root
        for subdir in ("system", "tools", "embodiment", "task", "heuristics", ""):
            d = PROMPT_DIR / subdir if subdir else PROMPT_DIR
            path = d / f"{name}.md"
            if path.exists():
                return {
                    "ok": True,
                    "content": path.read_text(encoding="utf-8"),
                    "category": subdir or "root",
                }
        return {"ok": False, "error": "not found", "content": ""}

    @app.post("/api/evolve", response_model=OkResponse)
    async def evolve(req: SessionConfigRequest | None = None) -> OkResponse:
        nonlocal generating
        if generating:
            return OkResponse(ok=False)

        req = req or SessionConfigRequest()
        config = state.apply_request(req)
        session = state.get_session(config.backend)
        provider = state.providers[config.backend]

        CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)
        conv_files = sorted(CONVERSATION_DIR.glob("*.md"))
        if not conv_files:
            return OkResponse(ok=False)

        latest = conv_files[-1]
        conv_content = latest.read_text(encoding="utf-8")
        if len(conv_content) > 20000:
            conv_content = conv_content[:20000] + "\n\n... (truncated)"

        evolve_prompt = (
            "[EVOLVE] Analyze the past conversation below and extract useful "
            "task strategies, patterns, failure modes, or lessons that would "
            "help in future robot programming tasks.\n\n"
            "Output a concise markdown document (under 500 words) with the key insights. "
            "Use headings and bullet points. Focus on actionable robot programming "
            "strategies specific to the YAM station.\n\n"
            "After your analysis, output the final prompt document in a "
            "```markdown\n...\n``` fenced block. I will save this to cap/prompt/.\n\n"
            f"---\n\nConversation from {latest.name}:\n\n{conv_content}"
        )

        generating = True

        async def _run() -> None:
            nonlocal generating
            try:
                result = await provider.run_turn(
                    message=evolve_prompt,
                    session=session,
                    ws_manager=_TurnWSManager(inner=ws_manager, chat_voice=chat_voice),
                    system_prompt=system_prompt,
                    config=config,
                    context=context,
                )
                await _finalize_turn(
                    session=session,
                    prompt_message=evolve_prompt,
                    full_text=result.full_text,
                )
                md_blocks = re.findall(
                    r"```markdown\s*\n(.*?)```", result.full_text, re.DOTALL
                )
                if md_blocks:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    heuristics_dir = PROMPT_DIR / "heuristics"
                    heuristics_dir.mkdir(parents=True, exist_ok=True)
                    save_path = heuristics_dir / f"evolved_{ts}.md"
                    save_path.write_text(md_blocks[-1].strip(), encoding="utf-8")
                    logger.info("Evolved prompt saved: %s", save_path)
                    await ws_manager.broadcast(
                        "chat_text_delta",
                        {"text": f"\n\n*Saved to `{save_path.name}`*"},
                    )
            except Exception as e:
                logger.exception("Evolve turn failed")
                chat_voice.on_turn_error()
                await ws_manager.broadcast("chat_error", {"error": str(e)})
            finally:
                generating = False

        await job_queue.put(_run)
        return OkResponse(ok=True)

    @app.websocket("/ws/chat")
    async def chat_ws(ws: WebSocket) -> None:
        await ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    logger.info("CAP Agent Bridge starting on port %s", BRIDGE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT)


if __name__ == "__main__":
    main()
