# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandboxed code executor for CAP agent.

Executes user/LLM-generated Python code with tool functions injected into
the namespace.  Captures stdout, stderr, and per-statement execution logs.

Supports `wait_for_agent(message)` — a special function that pauses execution
and yields control back to the agent for replanning.
"""

from __future__ import annotations

import ast
import copy
import ctypes
import io
import logging
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from enpire.env.forge.cap.config import DEFAULT_VLM_BACKEND

logger = logging.getLogger(__name__)


class WaitForAgent(Exception):
    """Raised by wait_for_agent() to pause execution and yield to the agent."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ExecutionCancelled(Exception):
    """Raised asynchronously in the executor thread to force cancellation."""

    pass


class _LoopBreak(Exception):
    """Raised when 'break' is encountered in a manually-iterated for loop."""

    pass


class _LoopContinue(Exception):
    """Raised when 'continue' is encountered in a manually-iterated for loop."""

    pass


class _BreakContinueTransformer(ast.NodeTransformer):
    """Replace top-level break/continue with custom exceptions.

    Does NOT recurse into nested for/while loops so their own break/continue
    are left for Python to handle normally.
    """

    def visit_For(self, node: ast.For) -> ast.For:
        return node  # Preserve nested for loops intact

    def visit_While(self, node: ast.While) -> ast.While:
        return node  # Preserve nested while loops intact

    def visit_Break(self, node: ast.Break) -> ast.stmt:
        new = ast.parse("raise __cap_break__()", mode="exec").body[0]
        return ast.copy_location(new, node)

    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        new = ast.parse("raise __cap_continue__()", mode="exec").body[0]
        return ast.copy_location(new, node)


@dataclass
class ExecutionLog:
    """Record of a single statement execution."""

    line: int
    source: str
    result: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
    stdout: str | None = None
    stderr: str | None = None
    # Loop/structure tracking
    node_type: str | None = None  # "for", "function_def", etc.
    parent_id: str | None = None  # parent loop entry ID for child entries
    entry_id: str | None = None  # written back by on_start for child correlation


@dataclass
class ExecutionResult:
    """Aggregate result of executing a code block."""

    success: bool
    logs: list[ExecutionLog] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    # wait_for_agent support
    paused: bool = False
    pause_message: str = ""
    user_namespace: dict[str, Any] | None = None
    # Cancellation support
    cancelled: bool = False


class _StreamingStringIO(io.StringIO):
    def __init__(self, on_write: Callable[[str], None] | None = None):
        """Create a StringIO that forwards each write chunk to ``on_write``."""
        super().__init__()
        self._on_write = on_write

    def write(self, s: str) -> int:
        n = super().write(s)
        if self._on_write and s:
            self._on_write(s)
        return n


class Executor:
    """Execute Python code line-by-line with tool functions available.

    Parameters:
        tool_callables: Dict of ``{name: callable}`` from ToolRegistry.callable_dict().
        on_log: Optional callback invoked after each statement with the ExecutionLog.
    """

    # Modules / builtins allowed in the sandbox
    _ALLOWED_BUILTINS = {
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "list",
        "dict",
        "tuple",
        "set",
        "int",
        "float",
        "str",
        "bool",
        "abs",
        "min",
        "max",
        "sum",
        "round",
        "sorted",
        "reversed",
        "isinstance",
        "type",
        "hasattr",
        "getattr",
        "True",
        "False",
        "None",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "ImportError",
        "io",
    }

    _ALLOWED_IMPORTS = {
        "numpy",
        "scipy",
        "time",
        "threading",
        "concurrent",
        "datetime",
        "math",
        "copy",
        "json",
        "collections",
        "itertools",
        "functools",
        "dataclasses",
        "typing",
        "random",
        "os",
        "re",
        "cv2",
        "cap",
        "skill_library",
    }

    def __init__(
        self,
        tool_callables: dict[str, Any],
        on_log: Callable[[ExecutionLog], None] | None = None,
        on_start: Callable[[ExecutionLog], None] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        go_event: threading.Event | None = None,
    ):
        self._tools = tool_callables
        self._on_log = on_log
        self._on_start = on_start
        self._on_stdout = on_stdout
        self._cancel = cancel_event
        self._go = go_event
        self._thread_id: int | None = None

    def force_cancel(self) -> None:
        """Forcibly raise ExecutionCancelled in the executor thread."""
        tid = self._thread_id
        if tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(ExecutionCancelled),
            )

    def execute(
        self,
        code: str,
        extra_namespace: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Parse and execute *code* statement by statement.

        Args:
            code: Python source code to execute.
            extra_namespace: Optional dict of variables to inject (e.g. from
                a previous paused execution).

        Returns an ExecutionResult with per-statement logs.
        """
        self._thread_id = threading.get_ident()
        try:
            return self._execute_inner(code, extra_namespace)
        finally:
            self._thread_id = None

    def _execute_inner(
        self,
        code: str,
        extra_namespace: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        # Build sandboxed namespace
        safe_builtins = {
            k: __builtins__[k]
            if isinstance(__builtins__, dict)
            else getattr(__builtins__, k)
            for k in self._ALLOWED_BUILTINS
            if (isinstance(__builtins__, dict) and k in __builtins__)
            or (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))
        }
        # Custom import that only allows whitelisted modules
        _real_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Allow submodules of whitelisted packages (e.g. collections.abc)
            top_level = name.split(".")[0]
            if top_level not in self._ALLOWED_IMPORTS:
                raise ImportError(
                    f"Import of '{name}' is not allowed. "
                    f"Allowed: {', '.join(sorted(self._ALLOWED_IMPORTS))}"
                )
            return _real_import(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = _safe_import

        namespace: dict[str, Any] = {"__builtins__": safe_builtins}
        namespace.update(self._tools)

        # Wrap tool callables with timing profiler
        from enpire.env.forge.cap.agent.profiler import wrap_callables_with_timing

        namespace.update(wrap_callables_with_timing(self._tools))

        # Inject live skill_library.namespace into sys.modules so that
        # `from skill_library.namespace import go_home` resolves to real callables.
        import sys as _sys
        import types as _types
        _ns_mod = _types.ModuleType("skill_library.namespace")
        for _k, _v in self._tools.items():
            if not _k.startswith("_") and callable(_v):
                setattr(_ns_mod, _k, _v)
        _ns_mod.__all__ = [k for k in vars(_ns_mod) if not k.startswith("_")]
        _sys.modules["skill_library.namespace"] = _ns_mod

        # Inject wait_for_agent function
        def _wait_for_agent(message: str = "") -> None:
            raise WaitForAgent(message)

        namespace["wait_for_agent"] = _wait_for_agent

        # Backward-compatible VLM alias for older saved scripts / prompts.
        # Historical usage was ask_vlm(prompt, "top,left,right"), while the
        # current interface is vlm_query(text=..., camera=.../media=[...]).
        _vlm_query = namespace.get("vlm_query")
        if callable(_vlm_query):

            def _ask_vlm(
                prompt: str,
                cameras: str | list[str] = "top",
                backend: str = DEFAULT_VLM_BACKEND,
                **kwargs,
            ) -> str:
                """Compatibility wrapper around ``vlm_query`` for legacy scripts."""
                if isinstance(cameras, str):
                    camera_names = [c.strip() for c in cameras.split(",") if c.strip()]
                elif isinstance(cameras, (list, tuple)):
                    camera_names = [str(c).strip() for c in cameras if str(c).strip()]
                else:
                    raise TypeError(
                        "ask_vlm cameras must be a comma-separated string or list of camera names"
                    )

                if not camera_names:
                    camera_names = ["top"]

                normalized = [c.lower() for c in camera_names]
                if normalized == ["all"]:
                    return _vlm_query(prompt, backend=backend, camera="all", **kwargs)
                if len(normalized) == 1:
                    return _vlm_query(
                        prompt, backend=backend, camera=normalized[0], **kwargs
                    )

                media = [f"camera:{cam}" for cam in normalized]
                return _vlm_query(prompt, backend=backend, media=media, **kwargs)

            namespace["_compat_ask_vlm"] = _ask_vlm
            if "ask_vlm" not in namespace and not (
                extra_namespace and "ask_vlm" in extra_namespace
            ):
                namespace["ask_vlm"] = _ask_vlm
            else:
                logger.warning(
                    "Skipping ask_vlm compatibility alias because the sandbox namespace already defines ask_vlm"
                )

        # load_module: import definitions from other saved scripts without running main code.
        from pathlib import Path as _Path

        _scripts_root = _Path(__file__).resolve().parents[1] / "saved_scripts"

        def _load_module(script_name: str) -> dict:
            """Load function definitions from a saved script without executing main code."""
            path = _scripts_root / script_name
            if not path.exists():
                raise RuntimeError(f"load_module: script not found: {script_name}")
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, str(path))
            defs = [
                n
                for n in tree.body
                if isinstance(
                    n,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Assign,
                        ast.AnnAssign,
                        ast.Import,
                        ast.ImportFrom,
                    ),
                )
            ]
            mod = ast.Module(body=defs, type_ignores=[])
            ast.fix_missing_locations(mod)
            compiled = compile(mod, str(path), "exec")
            mod_ns = dict(namespace)
            exec(compiled, mod_ns)
            return mod_ns

        namespace["load_module"] = _load_module

        # load_module: import definitions from other saved scripts without running main code.
        from pathlib import Path as _Path

        _scripts_root = _Path(__file__).resolve().parents[1] / "saved_scripts"

        def _load_module(script_name: str) -> dict:
            """Load function definitions from a saved script without executing main code."""
            path = _scripts_root / script_name
            if not path.exists():
                raise RuntimeError(f"load_module: script not found: {script_name}")
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, str(path))
            defs = [
                n
                for n in tree.body
                if isinstance(
                    n,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Assign,
                        ast.AnnAssign,
                        ast.Import,
                        ast.ImportFrom,
                    ),
                )
            ]
            mod = ast.Module(body=defs, type_ignores=[])
            ast.fix_missing_locations(mod)
            compiled = compile(mod, str(path), "exec")
            mod_ns = dict(namespace)
            exec(compiled, mod_ns)
            return mod_ns

        namespace["load_module"] = _load_module

        # Inject preserved variables from a previous paused execution
        if extra_namespace:
            namespace.update(extra_namespace)

        # Track initial keys so we can extract user-defined variables later
        initial_keys = set(namespace.keys())

        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                error=f"SyntaxError: {e}",
            )

        logs: list[ExecutionLog] = []
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        def _sandbox_print(*args, sep=" ", end="\n", file=None, flush=False):
            """Sandboxed ``print`` that captures stdout and forwards stream updates."""
            target = stdout_capture if file is None else file
            text = sep.join(str(arg) for arg in args) + end
            target.write(text)
            if self._on_stdout is not None and (file is None or file is stdout_capture):
                self._on_stdout(text)
            if flush and hasattr(target, "flush"):
                target.flush()

        safe_builtins["print"] = _sandbox_print

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                for node in ast.iter_child_nodes(tree):
                    # --- cancellation gate ---
                    if self._cancel is not None and self._cancel.is_set():
                        return ExecutionResult(
                            success=False,
                            cancelled=True,
                            logs=logs,
                            stdout=stdout_capture.getvalue(),
                            stderr=stderr_capture.getvalue(),
                            error="Execution cancelled",
                        )
                    # --- pause gate (blocks until go_event is set) ---
                    if self._go is not None:
                        self._go.wait()
                        # Re-check cancel after waking from pause
                        if self._cancel is not None and self._cancel.is_set():
                            return ExecutionResult(
                                success=False,
                                cancelled=True,
                                logs=logs,
                                stdout=stdout_capture.getvalue(),
                                stderr=stderr_capture.getvalue(),
                                error="Execution cancelled",
                            )

                    source = ast.get_source_segment(code, node) or ""
                    line = getattr(node, "lineno", 0)

                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        node_type: str | None = "function_def"
                    else:
                        node_type = None

                    if isinstance(node, ast.For):
                        # --- Per-iteration for loop execution ---
                        loop_start_log = ExecutionLog(
                            line=line, source=source.strip(), node_type="for"
                        )
                        if self._on_start:
                            self._on_start(loop_start_log)
                        loop_parent_id = loop_start_log.entry_id

                        t0 = time.perf_counter()
                        stdout_pos = stdout_capture.tell()
                        try:
                            iter_expr = ast.Expression(body=copy.deepcopy(node.iter))
                            ast.fix_missing_locations(iter_expr)
                            iterable = eval(
                                compile(iter_expr, "<cap>", "eval"), namespace
                            )

                            namespace["__cap_break__"] = _LoopBreak
                            namespace["__cap_continue__"] = _LoopContinue
                            _bct = _BreakContinueTransformer()
                            loop_broken = False

                            for item in iterable:
                                # Assign loop target variable(s)
                                namespace["__cap_item__"] = item
                                target_src = (
                                    f"{ast.unparse(node.target)} = __cap_item__"
                                )
                                target_tree = ast.parse(target_src)
                                exec(compile(target_tree, "<cap>", "exec"), namespace)

                                iter_broken = False
                                for body_stmt in node.body:
                                    # Cancellation check inside loop
                                    if (
                                        self._cancel is not None
                                        and self._cancel.is_set()
                                    ):
                                        return ExecutionResult(
                                            success=False,
                                            cancelled=True,
                                            logs=logs,
                                            stdout=stdout_capture.getvalue(),
                                            stderr=stderr_capture.getvalue(),
                                            error="Execution cancelled",
                                        )

                                    body_source = (
                                        ast.get_source_segment(code, body_stmt) or ""
                                    )
                                    body_line = getattr(body_stmt, "lineno", 0)

                                    # Transform break/continue so we can exec statement alone
                                    transformed = _bct.visit(
                                        ast.Module(
                                            body=[copy.deepcopy(body_stmt)],
                                            type_ignores=[],
                                        )
                                    )
                                    ast.fix_missing_locations(transformed)

                                    child_log = ExecutionLog(
                                        line=body_line,
                                        source=body_source.strip(),
                                        parent_id=loop_parent_id,
                                    )
                                    if self._on_start:
                                        self._on_start(child_log)
                                    child_entry_id = child_log.entry_id

                                    child_stdout_pos = stdout_capture.tell()
                                    stmt_t0 = time.perf_counter()
                                    try:
                                        exec(
                                            compile(transformed, "<cap>", "exec"),
                                            namespace,
                                        )
                                        stmt_elapsed = (
                                            time.perf_counter() - stmt_t0
                                        ) * 1000
                                        child_stdout = (
                                            stdout_capture.getvalue()[
                                                child_stdout_pos:
                                            ].strip()
                                            or None
                                        )

                                        result_log = ExecutionLog(
                                            line=body_line,
                                            source=body_source.strip(),
                                            elapsed_ms=round(stmt_elapsed, 2),
                                            stdout=child_stdout,
                                            parent_id=loop_parent_id,
                                            entry_id=child_entry_id,
                                        )
                                        logs.append(result_log)
                                        if self._on_log:
                                            self._on_log(result_log)

                                    except _LoopContinue:
                                        stmt_elapsed = (
                                            time.perf_counter() - stmt_t0
                                        ) * 1000
                                        result_log = ExecutionLog(
                                            line=body_line,
                                            source=body_source.strip(),
                                            elapsed_ms=round(stmt_elapsed, 2),
                                            parent_id=loop_parent_id,
                                            entry_id=child_entry_id,
                                        )
                                        logs.append(result_log)
                                        if self._on_log:
                                            self._on_log(result_log)
                                        iter_broken = True
                                        break  # go to next iteration

                                    except _LoopBreak:
                                        stmt_elapsed = (
                                            time.perf_counter() - stmt_t0
                                        ) * 1000
                                        result_log = ExecutionLog(
                                            line=body_line,
                                            source=body_source.strip(),
                                            elapsed_ms=round(stmt_elapsed, 2),
                                            parent_id=loop_parent_id,
                                            entry_id=child_entry_id,
                                        )
                                        logs.append(result_log)
                                        if self._on_log:
                                            self._on_log(result_log)
                                        loop_broken = True
                                        iter_broken = True
                                        break

                                    except WaitForAgent as e:
                                        stmt_elapsed = (
                                            time.perf_counter() - stmt_t0
                                        ) * 1000
                                        result_log = ExecutionLog(
                                            line=body_line,
                                            source=body_source.strip(),
                                            elapsed_ms=round(stmt_elapsed, 2),
                                            stdout=f"[wait_for_agent] {e.message}",
                                            parent_id=loop_parent_id,
                                            entry_id=child_entry_id,
                                        )
                                        logs.append(result_log)
                                        if self._on_log:
                                            self._on_log(result_log)
                                        user_vars = {
                                            k: v
                                            for k, v in namespace.items()
                                            if k not in initial_keys
                                        }
                                        return ExecutionResult(
                                            success=True,
                                            logs=logs,
                                            stdout=stdout_capture.getvalue(),
                                            stderr=stderr_capture.getvalue(),
                                            paused=True,
                                            pause_message=e.message,
                                            user_namespace=user_vars,
                                        )

                                    except ExecutionCancelled:
                                        raise

                                    except Exception:
                                        stmt_elapsed = (
                                            time.perf_counter() - stmt_t0
                                        ) * 1000
                                        tb = traceback.format_exc()
                                        result_log = ExecutionLog(
                                            line=body_line,
                                            source=body_source.strip(),
                                            error=tb,
                                            elapsed_ms=round(stmt_elapsed, 2),
                                            parent_id=loop_parent_id,
                                            entry_id=child_entry_id,
                                        )
                                        logs.append(result_log)
                                        if self._on_log:
                                            self._on_log(result_log)
                                        # Update loop entry as error
                                        elapsed = (time.perf_counter() - t0) * 1000
                                        loop_err_log = ExecutionLog(
                                            line=line,
                                            source=source.strip(),
                                            error=tb,
                                            elapsed_ms=round(elapsed, 2),
                                            node_type="for",
                                            entry_id=loop_parent_id,
                                        )
                                        logs.append(loop_err_log)
                                        if self._on_log:
                                            self._on_log(loop_err_log)
                                        return ExecutionResult(
                                            success=False,
                                            logs=logs,
                                            stdout=stdout_capture.getvalue(),
                                            stderr=stderr_capture.getvalue() + tb,
                                            error=tb,
                                        )

                                if loop_broken:
                                    break

                            # Execute the for/else block if loop wasn't broken
                            if not loop_broken and node.orelse:
                                orelse_mod = ast.Module(
                                    body=node.orelse, type_ignores=[]
                                )
                                ast.fix_missing_locations(orelse_mod)
                                exec(compile(orelse_mod, "<cap>", "exec"), namespace)

                            elapsed = (time.perf_counter() - t0) * 1000
                            loop_new_stdout = (
                                stdout_capture.getvalue()[stdout_pos:].strip() or None
                            )
                            loop_end_log = ExecutionLog(
                                line=line,
                                source=source.strip(),
                                elapsed_ms=round(elapsed, 2),
                                stdout=loop_new_stdout,
                                node_type="for",
                                entry_id=loop_parent_id,
                            )
                            logs.append(loop_end_log)
                            if self._on_log:
                                self._on_log(loop_end_log)

                        except (WaitForAgent, ExecutionCancelled):
                            raise

                        except Exception:
                            elapsed = (time.perf_counter() - t0) * 1000
                            tb = traceback.format_exc()
                            loop_err_log = ExecutionLog(
                                line=line,
                                source=source.strip(),
                                error=tb,
                                elapsed_ms=round(elapsed, 2),
                                node_type="for",
                                entry_id=loop_parent_id,
                            )
                            logs.append(loop_err_log)
                            if self._on_log:
                                self._on_log(loop_err_log)
                            return ExecutionResult(
                                success=False,
                                logs=logs,
                                stdout=stdout_capture.getvalue(),
                                stderr=stderr_capture.getvalue() + tb,
                                error=tb,
                            )

                    else:
                        # --- Single statement execution ---
                        mod = ast.Module(body=[node], type_ignores=[])
                        ast.fix_missing_locations(mod)

                        # Notify before execution (for "running" status indicators)
                        if self._on_start:
                            self._on_start(
                                ExecutionLog(
                                    line=line,
                                    source=source.strip(),
                                    node_type=node_type,
                                )
                            )

                        # Track stdout position before this statement
                        stdout_pos = stdout_capture.tell()

                        t0 = time.perf_counter()
                        try:
                            compiled = compile(mod, "<cap>", "exec")
                            exec(compiled, namespace)
                            elapsed = (time.perf_counter() - t0) * 1000

                            # Capture any stdout this statement produced
                            new_stdout = (
                                stdout_capture.getvalue()[stdout_pos:].strip() or None
                            )

                            log_entry = ExecutionLog(
                                line=line,
                                source=source.strip(),
                                elapsed_ms=round(elapsed, 2),
                                stdout=new_stdout,
                                node_type=node_type,
                            )
                            logs.append(log_entry)
                            if self._on_log:
                                self._on_log(log_entry)

                        except WaitForAgent as e:
                            elapsed = (time.perf_counter() - t0) * 1000
                            log_entry = ExecutionLog(
                                line=line,
                                source=source.strip(),
                                elapsed_ms=round(elapsed, 2),
                                stdout=f"[wait_for_agent] {e.message}",
                            )
                            logs.append(log_entry)
                            if self._on_log:
                                self._on_log(log_entry)

                            # Extract user-defined variables
                            user_vars = {
                                k: v
                                for k, v in namespace.items()
                                if k not in initial_keys
                            }

                            return ExecutionResult(
                                success=True,
                                logs=logs,
                                stdout=stdout_capture.getvalue(),
                                stderr=stderr_capture.getvalue(),
                                paused=True,
                                pause_message=e.message,
                                user_namespace=user_vars,
                            )

                        except ExecutionCancelled:
                            raise  # Re-raise to be caught by the outer handler

                        except Exception:
                            elapsed = (time.perf_counter() - t0) * 1000
                            tb = traceback.format_exc()
                            log_entry = ExecutionLog(
                                line=line,
                                source=source.strip(),
                                error=tb,
                                elapsed_ms=round(elapsed, 2),
                                node_type=node_type,
                            )
                            logs.append(log_entry)
                            if self._on_log:
                                self._on_log(log_entry)

                            return ExecutionResult(
                                success=False,
                                logs=logs,
                                stdout=stdout_capture.getvalue(),
                                stderr=stderr_capture.getvalue() + tb,
                                error=tb,
                            )

            except ExecutionCancelled:
                return ExecutionResult(
                    success=False,
                    cancelled=True,
                    logs=logs,
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                    error="Execution forcibly cancelled (E-Stop)",
                )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return ExecutionResult(
            success=True,
            logs=logs,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
        )
