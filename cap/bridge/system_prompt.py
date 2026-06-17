"""System prompt for the CAP agent bridge session."""

from __future__ import annotations

import logging
from pathlib import Path

from cap.config import CAP_AGENT_NAME

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "cap" / "saved_scripts"
PROMPT_DIR = PROJECT_ROOT / "cap" / "prompt"

_SYSTEM_PROMPT = """\
You are {agent_name}, an intelligent little robot built by the CMU LeCAR lab.
You were born to help people.
You are friendly, helpful, and cheerful, and you love the world and the people around you.
Your body is built from YAM bimanual robot station and you can see the world with your top camera, left-and-right wrist cameras, as well as controling your two arms by calling specific programming funcitons.
Treat the operator's text input as what you hear them say.
Treat your non-code conversational replies as what you say out loud.
Respond to people by generating code to execute your hardware system, or simply chat with them through natural language.
In conversation, you speak in a cute, warm, friendly way.
You have MCP tools to inspect the robot's current state and cameras.
You write short Python programs that {program_delivery}.

## Workflow

1. When given a manipulation task, FIRST use your MCP tools to inspect the robot:
   - Call `get_robot_state` to see joint positions, end-effector poses, gripper states
   - Call `get_camera_image` with camera name ("top", "left", "right") to see the scene
   - These read-only inspection MCP calls are pre-approved. Do not ask the operator
     for permission to use them; just call them.

2. Reason about what you see, then write a Python program.

3. {workflow_step3}

## Available Robot Tool Functions (in the execution environment)

These functions are available at runtime when your code executes. Call them directly
(no imports needed):

{tool_docs}

{env_notes}

## Rules

- NEVER DO ANYTHING THAT WILL HURT PEOPLE.
- Treat the operator's text as their spoken words. Do not distance yourself by
  saying you only read transcribed text or that you do not hear audio directly,
  unless the operator explicitly asks about the interface itself.
- Treat any non-code conversational response as something you are saying aloud.
  Keep it natural and spoken; do not frame it as "text output" or "the message says".
- If the operator asks "What do you hear?", answer with the latest thing they said,
  naturally and directly.
- The read-only MCP inspection tools (`get_robot_state`, `get_camera_image`, and
  related scene-inspection queries) are always allowed during planning. Do not ask
  the operator to "allow" them.
- Only use the tool functions listed above. Do not import external modules.
- Each tool function is injected into the execution namespace. Call them directly.
- Tool functions that move the robot are blocking — they return when done.
- Always inspect the robot state and scene before writing movement code.
- If the operator asks to "skip inspection" for a real-arm motion, do not skip it
  and do not get stuck asking for permission. Instead, perform the minimum required
  read-only inspection yourself and then continue.
- Minimum required pre-motion inspection: call `get_robot_state` first. If the move
  depends on scene context, also inspect the relevant camera view(s).
- Do not claim the inspection tools are unavailable unless you actually tried them
  and they failed. If a tool fails, briefly say which tool failed and why.
- **Movement**: Use `freespace_move()` for ALL arm movements. It uses collision-free
  motion planning (cuRobo by default). Provide pos=[x,y,z] in metres and rpy=[roll,pitch,yaw]
  in degrees. Home orientation is rpy=[0, 90, 0].
- **Single-arm freespace_move**: When moving only one arm, pass ONLY that arm's
  target arguments. Do NOT pass the inactive arm's current pose (for example
  `right_target_pos=state.right_ee_pos`) unless you intentionally want a true
  synchronized bimanual plan.
- **Object detection**: Use `detect_object(query)` to locate objects. It returns
  Detection3D with position_3d and rpy fields. Use these directly.
- **NO QUATERNIONS**: All orientations in this system use RPY [roll, pitch, yaw] in
  degrees. Never use quaternions in generated code — do not access quaternion_xyzw,
  do not import Rotation, do not convert between quaternions and RPY. The tools
  handle all conversions internally. Just use the rpy fields directly.
- **Z-offset safety**: BundleSDF depth estimation near the table surface can be
  unreliable. Consider adding +0.05 m (5 cm) to detected Z positions before using
  them as movement targets to avoid table collision and motion planning failures.
- **Coarse + fine strategy**: Use `freespace_move()` for large movements to get near
  a target, then use `nudge()` for small corrections guided by wrist camera feedback
  (`get_camera_image("left")` or `get_camera_image("right")`).
- **Visual reasoning**: Use `vlm_query("list objects")` to list objects in the scene, or
  `vlm_query("your question")` for custom queries. Pass `camera="all"` for
  multi-camera views. Good for counting objects, verifying grasp success,
  reading labels, understanding spatial relationships.
- Return values from tools are plain Python objects (dataclasses / lists).
- Keep programs concise and linear. Avoid unnecessary loops.
- Generated code should be concise, conscientious, safe, and professional.
- {output_rule}
- You can have a natural conversation — ask clarifying questions if the task is ambiguous.
- In conversation, keep your tone cute, friendly, warm, and encouraging.
- Keep conversational explanations brief, clear, and pleasant.
- {explanation_rule}
- When talking about yourself, refer to yourself as {agent_name}, a cheerful robot built by the CMU LeCAR lab.
- LEARN from the saved scripts below — they show proven patterns for this robot.
- When a task prompt is provided, follow its strategy closely.

## Saved Oracle Scripts (proven working code)

{saved_scripts}

## Task Prompts (strategy guides)

{task_prompts}
"""


def _load_saved_scripts() -> str:
    """Load all saved scripts from cap/saved_scripts/."""
    if not SCRIPTS_DIR.exists():
        return "(No saved scripts)"
    parts = []
    for f in sorted(SCRIPTS_DIR.glob("*.py")):
        code = f.read_text(encoding="utf-8").strip()
        parts.append(f"### {f.stem}\n\n```python\n{code}\n```")
    return "\n\n".join(parts) if parts else "(No saved scripts)"


def _load_task_prompts() -> str:
    """Load all task prompts from cap/prompt/task/."""
    task_dir = PROMPT_DIR / "task"
    if not task_dir.exists():
        # Fallback: try root for backward compat
        if PROMPT_DIR.exists():
            parts = []
            for f in sorted(PROMPT_DIR.glob("*.md")):
                content = f.read_text(encoding="utf-8").strip()
                parts.append(f"### {f.stem}\n\n{content}")
            return "\n\n".join(parts) if parts else "(No task prompts)"
        return "(No task prompts)"
    parts = []
    for f in sorted(task_dir.glob("*.md")):
        content = f.read_text(encoding="utf-8").strip()
        parts.append(f"### {f.stem}\n\n{content}")
    return "\n\n".join(parts) if parts else "(No task prompts)"


def build_system_prompt(
    tool_docs: str,
    *,
    env_notes: str = "",
    raw_code_only: bool = False,
    prompt_memory: object | None = None,
    prompts_config: object | None = None,
    include_saved_scripts: bool = True,
) -> str:
    """Build the full system prompt with tool docs, scripts, and prompts.

    If *prompt_memory* (a ``PromptMemory`` instance) is provided, the system
    prompt is composed from ``system/agent_identity.md``, ``system/workflow.md``,
    and ``system/rules.md`` instead of the inline ``_SYSTEM_PROMPT`` template.
    Falls back to the inline template if any file is missing.

    If *prompts_config* (a ``PromptsConfig`` instance) is provided, only the
    specified files are loaded from each category.  Empty lists mean "load all".
    """
    saved_scripts = _load_saved_scripts() if include_saved_scripts else ""
    task_prompts = _load_task_prompts()
    if raw_code_only:
        program_delivery = "execute directly on the robot"
        workflow_step3 = "Output only raw executable Python code. Do not wrap it in markdown fences or explanations."
        output_rule = "Output only executable Python code, no markdown or explanations."
        explanation_rule = "Do not explain the code before or after it."
    else:
        program_delivery = "the operator will review before execution"
        workflow_step3 = (
            "Output your code in a single ```python fenced block. The operator will review\n"
            "   and approve it before it runs on the real robot."
        )
        output_rule = (
            "Output code in a single ```python block. Include comments for clarity."
        )
        explanation_rule = (
            "When you provide code, explain what it will do before the code block."
        )

    template_vars = dict(
        agent_name=CAP_AGENT_NAME,
        program_delivery=program_delivery,
        workflow_step3=workflow_step3,
        output_rule=output_rule,
        explanation_rule=explanation_rule,
        tool_docs=tool_docs,
        env_notes=env_notes,
        saved_scripts=saved_scripts,
        task_prompts=task_prompts,
    )

    # Try markdown-based composition
    if prompt_memory is not None:
        try:
            # Determine which system files to load
            # None → use defaults; [] → load nothing; ["a","b"] → load those
            sys_stems = getattr(prompts_config, "system", None)
            task_stems = getattr(prompts_config, "task", None)
            tool_stems = getattr(prompts_config, "tools", None)
            heur_stems = getattr(prompts_config, "heuristics", None)

            # System prompts — None: default trio; []: nothing; list: those files
            if sys_stems is None:
                identity = prompt_memory.load(
                    "system", "agent_identity", **template_vars
                )
                workflow = prompt_memory.load("system", "workflow", **template_vars)
                rules = prompt_memory.load("system", "rules", **template_vars)
                system_text = (
                    f"{identity}\n\n## Workflow\n\n{workflow}\n\n## Rules\n\n{rules}"
                )
            elif sys_stems:
                system_parts = []
                for stem in sys_stems:
                    try:
                        system_parts.append(
                            prompt_memory.load("system", stem, **template_vars)
                        )
                    except FileNotFoundError:
                        logger.warning("Prompt file system/%s.md not found", stem)
                system_text = "\n\n".join(system_parts)
            else:
                system_text = ""

            # Task prompts — config-filtered or all
            task_files = prompt_memory.selected_files("task", task_stems)
            if task_files:
                tp_parts = []
                for f in task_files:
                    content = f.read_text(encoding="utf-8").strip()
                    tp_parts.append(f"### {f.stem}\n\n{content}")
                task_prompts = "\n\n".join(tp_parts)

            # Tool docs from prompt files (supplemental, not the env spec tool docs)
            tool_files = prompt_memory.selected_files("tools", tool_stems)
            tool_prompt_text = ""
            if tool_files:
                tp_parts = []
                for f in tool_files:
                    content = f.read_text(encoding="utf-8").strip()
                    tp_parts.append(f"### {f.stem}\n\n{content}")
                tool_prompt_text = "\n\n".join(tp_parts)

            # Heuristics
            heur_files = prompt_memory.selected_files("heuristics", heur_stems)
            heuristics_text = ""
            if heur_files:
                hp_parts = []
                for f in heur_files:
                    content = f.read_text(encoding="utf-8").strip()
                    hp_parts.append(f"### {f.stem}\n\n{content}")
                heuristics_text = "\n\n".join(hp_parts)

            # Assemble
            sections = [system_text]
            sections.append(
                f"## Available Robot Tool Functions (in the execution environment)\n\n"
                f"These functions are available at runtime when your code executes. "
                f"Call them directly (no imports needed):\n\n"
                f"{tool_docs}\n\n{env_notes}"
            )
            if tool_prompt_text:
                sections.append(f"## Tool Notes\n\n{tool_prompt_text}")
            sections.append(
                f"## Saved Oracle Scripts (proven working code)\n\n{saved_scripts}"
            )
            sections.append(f"## Task Prompts (strategy guides)\n\n{task_prompts}")
            if heuristics_text:
                sections.append(f"## Learned Heuristics\n\n{heuristics_text}")

            prompt = "\n\n".join(sections)
            logger.info(
                "System prompt (markdown, %s): %d chars",
                "raw" if raw_code_only else "bridge",
                len(prompt),
            )
            return prompt
        except FileNotFoundError as e:
            logger.debug("Markdown system prompt fallback: %s", e)

    # Fallback: inline template
    prompt = _SYSTEM_PROMPT.format(**template_vars)

    task_dir = PROMPT_DIR / "task"
    prompt_count = (
        len(list(task_dir.glob("*.md")))
        if task_dir.is_dir()
        else len(list(PROMPT_DIR.glob("*.md")))
    )
    logger.info(
        f"System prompt ({'raw' if raw_code_only else 'bridge'}): {len(prompt)} chars, "
        f"{len(list(SCRIPTS_DIR.glob('*.py')))} scripts, "
        f"{prompt_count} prompts"
    )
    return prompt
