# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_RESET_REQUESTED = 10
_OWNED_TMUX_SESSIONS: set[str] = set()
_CLEANING_UP = False


@dataclass
class SupervisorState:
    phase: str = "idle"
    iteration: int = 0
    start_requested: bool = False
    requested_episode_timeout_s: float | None = None
    active_session: str | None = None
    last_rl_code: int | None = None
    last_reset_code: int | None = None
    last_reset_result: dict = field(default_factory=dict)
    last_error: str | None = None
    last_request: dict = field(default_factory=dict)
    last_started_s: float | None = None
    last_finished_s: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _tmux(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _tmux_session_exists(session: str) -> bool:
    return _tmux(["has-session", "-t", session]).returncode == 0


def _tmux_kill_session(session: str) -> None:
    _tmux(["kill-session", "-t", session])


def _tmux_capture(session: str, lines: int = 240) -> str:
    if not _tmux_session_exists(session):
        return ""
    result = _tmux(["capture-pane", "-pt", session, "-S", f"-{int(lines)}"])
    return result.stdout if result.returncode == 0 else result.stderr


def cleanup_owned_tmux_sessions() -> None:
    global _CLEANING_UP
    if _CLEANING_UP:
        return
    _CLEANING_UP = True
    try:
        for session in sorted(_OWNED_TMUX_SESSIONS):
            if _tmux_session_exists(session):
                print(
                    f"[PushT supervisor] killing owned tmux session {session}",
                    flush=True,
                )
                _tmux_kill_session(session)
    finally:
        _CLEANING_UP = False


def _install_signal_handlers() -> None:
    def _handle_signal(signum, _frame):
        print(
            f"[PushT supervisor] received signal {signum}; cleaning up jobs", flush=True
        )
        cleanup_owned_tmux_sessions()
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _handle_signal)


def cleanup_stale_pusht_processes(stale_age_s: float = 120.0) -> None:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,stat=,etimes=,args="],
            text=True,
        )
    except Exception as exc:
        print(f"[PushT supervisor] stale cleanup skipped: {exc}", flush=True)
        return

    current_pid = os.getpid()
    candidates: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid_s, ppid_s, _stat, etimes_s, args = parts
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
            etimes = float(etimes_s)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        is_rl_runner = (
            "rl/pusht_runner.py" in args and "tasks_config/pusht/pusht.yaml" in args
        )
        is_reset = (
            "run_script.py" in args
            and "script_file=cap/saved_scripts/place_grasped_t_reset.py" in args
        )
        is_debug_ui = "-m cap.debug_ui.app" in args and "place_grasped_t_reset_" in args
        is_orphan_rl_runner = is_rl_runner and ppid != current_pid
        is_stale_reset_child = (is_reset or is_debug_ui) and etimes >= stale_age_s
        if is_orphan_rl_runner or is_stale_reset_child:
            candidates.append(pid)

    if not candidates:
        return
    print(
        f"[PushT supervisor] cleaning stale reset/debug processes: {candidates}",
        flush=True,
    )
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in candidates:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(f"[PushT supervisor] cannot kill pid {pid}: {exc}", flush=True)
        if sig == signal.SIGTERM:
            time.sleep(1.0)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _run_tmux_job(
    *,
    session: str,
    window_name: str,
    cmd: list[str],
    status_path: Path,
    timeout_s: float | None,
    input_group: bool = False,
) -> int:
    if status_path.exists():
        status_path.unlink()
    if _tmux_session_exists(session):
        print(f"[PushT supervisor] replacing tmux session {session}", flush=True)
        _tmux_kill_session(session)

    status_q = shlex.quote(str(status_path))
    quoted_cmd = _quote_cmd(cmd)
    if input_group:
        quoted_cmd = f"sg input -c {shlex.quote(quoted_cmd)}"
    command_text = (
        "set -o pipefail; "
        f"cd {shlex.quote(str(REPO_ROOT))}; "
        f"{quoted_cmd}; "
        "status=$?; "
        f"printf '%s\\n' \"$status\" > {status_q}; "
        "exec bash"
    )
    print(
        f"[PushT supervisor] tmux launch session={session}: {_quote_cmd(cmd)}",
        flush=True,
    )
    _tmux(
        [
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(REPO_ROOT),
            "-n",
            window_name,
            "bash",
            "-lc",
            command_text,
        ],
        check=True,
    )

    start = time.monotonic()
    while not status_path.exists():
        if timeout_s is not None and time.monotonic() - start > timeout_s:
            print(
                f"[PushT supervisor] tmux job {session} timed out after {timeout_s:.1f}s",
                flush=True,
            )
            _tmux_kill_session(session)
            return 124
        if not _tmux_session_exists(session):
            print(
                f"[PushT supervisor] tmux job {session} ended without status file",
                flush=True,
            )
            return 125
        time.sleep(0.2)

    if timeout_s is not None and time.monotonic() - start > timeout_s:
        print(
            f"[PushT supervisor] tmux job {session} timed out after {timeout_s:.1f}s",
            flush=True,
        )
        _tmux_kill_session(session)
        return 124

    try:
        return int(status_path.read_text().strip())
    except Exception as exc:
        print(
            f"[PushT supervisor] could not read tmux status {status_path}: {exc}",
            flush=True,
        )
        return 125


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"success": False, "error": f"could not read {path}: {exc}"}


def _set_state(state: SupervisorState, **updates) -> None:
    with state.lock:
        for key, value in updates.items():
            setattr(state, key, value)


def _snapshot_state(state: SupervisorState, args: argparse.Namespace) -> dict:
    with state.lock:
        payload = {
            "phase": state.phase,
            "iteration": state.iteration,
            "start_requested": state.start_requested,
            "active_session": state.active_session,
            "last_rl_code": state.last_rl_code,
            "last_reset_code": state.last_reset_code,
            "last_reset_result": state.last_reset_result,
            "last_error": state.last_error,
            "last_request": state.last_request,
            "requested_episode_timeout_s": state.requested_episode_timeout_s,
            "last_started_s": state.last_started_s,
            "last_finished_s": state.last_finished_s,
        }
    payload["rl_log_tail"] = _tmux_capture(
        args.rl_tmux_session, lines=args.status_log_lines
    )
    payload["reset_log_tail"] = _tmux_capture(
        args.reset_tmux_session, lines=args.status_log_lines
    )
    payload["home_log_tail"] = _tmux_capture(
        args.home_tmux_session, lines=args.status_log_lines
    )
    return payload


def _ensure_fastapi_port_available(args: argparse.Namespace) -> None:
    if args.fastapi_port == 0:
        return
    try:
        with socket.create_server((args.fastapi_host, args.fastapi_port)):
            pass
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            base_url = f"http://{args.fastapi_host}:{args.fastapi_port}"
            print(
                f"[PushT supervisor] cannot start FastAPI on {base_url}: "
                "address already in use",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[PushT supervisor] check the existing supervisor with: "
                f"curl {base_url}/status",
                file=sys.stderr,
                flush=True,
            )
            print(
                "[PushT supervisor] stop the existing process or pass "
                "--fastapi-port <port> to use a different API port.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1) from exc
        raise


def _start_api(args: argparse.Namespace, state: SupervisorState) -> None:
    app = FastAPI(title="PushT supervisor")
    base_url = f"http://{args.fastapi_host}:{args.fastapi_port}"
    help_payload = {
        "service": "PushT supervisor",
        "public_base_url": base_url,
        "runner_base_url": args.runner_fastapi_url,
        "endpoints": {
            "POST /start": (
                "Start one 2-minute PushT RL run when idle/done/failed; when RL "
                "is already running, proxy to the runner /resume endpoint. "
                "Keyboard 's' uses this."
            ),
            "POST /startlong": (
                "Start one 10-minute PushT RL run when idle/done/failed; when RL "
                "is already running, proxy to the runner /resume endpoint."
            ),
            "POST /startsuperlong": (
                "Start one 20-minute PushT RL run when idle/done/failed; when RL "
                "is already running, proxy to the runner /resume endpoint."
            ),
            "POST /home": (
                "Stop any active PushT RL/reset job, then run the CAP PushT "
                "reset script. Keyboard 'h' uses this."
            ),
            "POST /restart": (
                "Proxy to the active RL runner /restart endpoint to rotate the "
                "recording output directory. Keyboard F5 uses this."
            ),
            "GET /status": (
                "Return supervisor phase, iteration, exit codes, reset/reward "
                "payloads, and recent RL/reset tmux log tails."
            ),
            "GET|POST /help": "Return this help payload.",
        },
        "phases": {
            "idle": "Supervisor is waiting for /start.",
            "queued": "/start accepted; RL tmux job will launch next.",
            "rl_running": "RL runner tmux job is active.",
            "reset_running": "CAP reset tmux job is active after reward threshold.",
            "home_running": "Deprecated phase; /home now uses reset_running.",
            "done": "One RL+reset cycle completed; waiting for the next /start.",
            "failed": "Last RL or reset job failed; /start can try again.",
        },
        "examples": [
            f"curl -X POST {base_url}/start",
            f"curl -X POST {base_url}/startlong",
            f"curl -X POST {base_url}/startsuperlong",
            f"curl -X POST {base_url}/home",
            f"curl -X POST {base_url}/restart",
            f"curl {base_url}/status",
            f"curl {base_url}/help",
        ],
    }

    @app.post("/start")
    def start() -> dict:
        return _queue_start(state, args, episode_timeout_s=120.0, label="/start")

    @app.post("/startlong")
    def startlong() -> dict:
        return _queue_start(state, args, episode_timeout_s=600.0, label="/startlong")

    @app.post("/startsuperlong")
    def startsuperlong() -> dict:
        return _queue_start(
            state,
            args,
            episode_timeout_s=1200.0,
            label="/startsuperlong",
        )

    @app.post("/home")
    def home() -> dict:
        with state.lock:
            phase = state.phase
        killed_sessions: list[str] = []
        if _tmux_session_exists(args.rl_tmux_session):
            _tmux_kill_session(args.rl_tmux_session)
            killed_sessions.append(args.rl_tmux_session)
        if phase == "reset_running" and _tmux_session_exists(args.reset_tmux_session):
            _tmux_kill_session(args.reset_tmux_session)
            killed_sessions.append(args.reset_tmux_session)
        result_path = args.state_dir / "reset_result.json"
        _set_state(
            state,
            phase="reset_running",
            active_session=args.reset_tmux_session,
            last_error=None,
            last_request={"reason": "manual_home_reset", "success": False},
        )
        reset_success, reset_code, reset_result = _run_reset(
            args,
            result_path,
            {"reason": "manual_home_reset", "success": False},
        )
        _set_state(
            state,
            phase="idle" if reset_success else "failed",
            active_session=None,
            last_reset_code=reset_code,
            last_reset_result=reset_result,
            last_error=None
            if reset_success
            else f"manual home reset failed code {reset_code}",
            last_finished_s=time.time(),
        )
        return {
            "accepted": reset_success,
            "phase": phase,
            "killed_sessions": killed_sessions,
            "reset_code": reset_code,
            "reset_result": reset_result,
        }

    @app.post("/restart")
    def restart() -> dict:
        with state.lock:
            phase = state.phase
        if phase == "rl_running":
            proxy = _proxy_runner_post(args, "/restart")
            print(f"[PushT supervisor] /restart proxied to runner: {proxy}", flush=True)
            return {"accepted": bool(proxy.get("ok")), "phase": phase, "proxy": proxy}
        return {"accepted": False, "phase": phase, "reason": "no active RL runner"}

    @app.get("/status")
    def status() -> dict:
        return _snapshot_state(state, args)

    @app.get("/help")
    @app.post("/help")
    def help_() -> dict:
        return help_payload

    config = uvicorn.Config(
        app=app, host=args.fastapi_host, port=args.fastapi_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(
        target=server.run,
        name="pusht-supervisor-fastapi",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    while thread.is_alive() and not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        print(
            f"[PushT supervisor] FastAPI failed to start on {base_url}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    print(
        f"[PushT supervisor] FastAPI listening on {base_url}",
        flush=True,
    )


def _queue_start(
    state: SupervisorState,
    args: argparse.Namespace,
    *,
    episode_timeout_s: float,
    label: str,
) -> dict:
    with state.lock:
        phase = state.phase
        if phase == "rl_running":
            pass
        elif phase not in ("idle", "done", "failed"):
            return {"accepted": False, "phase": state.phase}
        else:
            state.start_requested = True
            state.requested_episode_timeout_s = float(episode_timeout_s)
            state.phase = "queued"
            state.last_error = None
            state.last_started_s = time.time()
            state.last_finished_s = None
            print(
                f"[PushT supervisor] {label} accepted "
                f"episode_timeout_s={episode_timeout_s:.1f}",
                flush=True,
            )
            return {
                "accepted": True,
                "phase": "queued",
                "episode_timeout_s": float(episode_timeout_s),
            }

    proxy = _proxy_runner_post(args, "/resume")
    print(f"[PushT supervisor] {label} proxied to runner: {proxy}", flush=True)
    return {"accepted": bool(proxy.get("ok")), "phase": phase, "proxy": proxy}


def _proxy_runner_post(args: argparse.Namespace, path: str) -> dict:
    url = f"{args.runner_fastapi_url.rstrip('/')}{path}"
    try:
        response = requests.post(url, timeout=1.0)
        payload = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else response.text
        )
        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "url": url,
            "response": payload,
        }
    except requests.RequestException as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def _run_reset(
    args: argparse.Namespace,
    result_path: Path,
    request: dict | None = None,
) -> tuple[bool, int, dict]:
    if result_path.exists():
        result_path.unlink()
    cleanup_stale_pusht_processes(args.stale_process_age_s)
    cmd = [
        "env",
        "CAP_CUROBO_HOST=127.0.0.1",
        "CAP_CUROBO_PORT=8611",
        "CAP_CUROBO_START_SERVER=0",
        "CAP_TOP_CAMERA_BACKEND=realsense",
        "NAIL_PREGRASP_DEBUG_UI=1",
        "NAIL_PREGRASP_DEBUG_UI_BLOCKING=0",
        f"PUSHT_RESET_RESULT_PATH={result_path}",
        f"PUSHT_FINAL_FRAME_DIR={(request or {}).get('episode_dir', '')}",
        f"PUSHT_GOAL_IMAGE={(request or {}).get('goal_image', os.environ.get('PUSHT_GOAL_IMAGE', ''))}",
        f"PUSHT_INITIAL_FRAME={(request or {}).get('initial_frame', '')}",
        sys.executable,
        "run_script.py",
        f"script_file={args.reset_script_file}",
        f"skill_library_path={args.reset_skill_library_path}",
        "env.name=yam-real",
        "robot=real_yam",
        "robot.dashboard=false",
        "robot.await_exit=false",
        "robot.go_home_on_exit=false",
        "execution.record=true",
        "debug_ui.enabled=false",
        "debug_ui.auto_open=false",
        "debug_ui.auto_exit_on_run_end=true",
    ]
    code = _run_tmux_job(
        session=args.reset_tmux_session,
        window_name="reset",
        cmd=cmd,
        status_path=args.state_dir / "reset_exit_code.txt",
        timeout_s=args.reset_timeout_s,
    )
    cleanup_stale_pusht_processes(args.stale_process_age_s)
    result = _load_json(result_path)
    success = bool(result.get("success", False))
    print(
        f"[PushT supervisor] reset finished code={code} success={success} "
        f"result={result}",
        flush=True,
    )
    return success, code, result


def _capture_initial_frame(args: argparse.Namespace, iteration: int) -> str:
    path = (
        args.state_dir
        / f"initial_top_frame_iter_{iteration}_{time.strftime('%Y%m%dT%H%M%S')}.png"
    )
    cmd = [
        "env",
        "CAP_CUROBO_HOST=127.0.0.1",
        "CAP_CUROBO_PORT=8611",
        "CAP_CUROBO_START_SERVER=0",
        "CAP_TOP_CAMERA_BACKEND=realsense",
        "NAIL_PREGRASP_DEBUG_UI=0",
        f"PUSHT_TOP_FRAME_PATH={path}",
        sys.executable,
        "run_script.py",
        f"script_file={args.initial_frame_script_file}",
        f"skill_library_path={args.reset_skill_library_path}",
        "env.name=yam-real",
        "robot=real_yam",
        "robot.dashboard=false",
        "robot.await_exit=false",
        "robot.go_home_on_exit=false",
        "execution.record=false",
        "debug_ui.enabled=false",
        "debug_ui.auto_open=false",
        "debug_ui.auto_exit_on_run_end=true",
    ]
    code = _run_tmux_job(
        session=args.initial_frame_tmux_session,
        window_name="init-frame",
        cmd=cmd,
        status_path=args.state_dir / "initial_frame_exit_code.txt",
        timeout_s=args.initial_frame_timeout_s,
    )
    if code in {0, 134, 139} and path.exists():
        print(f"[PushT supervisor] initial top frame saved: {path}", flush=True)
        return str(path)
    print(
        f"[PushT supervisor] initial top frame capture failed code={code}; path={path}",
        flush=True,
    )
    return ""


def _run_home(args: argparse.Namespace) -> int:
    cleanup_stale_pusht_processes(args.stale_process_age_s)
    cmd = [
        "env",
        "CAP_CUROBO_HOST=127.0.0.1",
        "CAP_CUROBO_PORT=8611",
        "CAP_CUROBO_START_SERVER=0",
        "CAP_TOP_CAMERA_BACKEND=realsense",
        "NAIL_PREGRASP_DEBUG_UI=0",
        sys.executable,
        "run_script.py",
        f"script_file={args.home_script_file}",
        f"skill_library_path={args.reset_skill_library_path}",
        "env.name=yam-real",
        "robot=real_yam",
        "robot.dashboard=false",
        "robot.await_exit=false",
        "robot.go_home_on_exit=false",
        "execution.record=false",
        "debug_ui.enabled=false",
        "debug_ui.auto_open=false",
        "debug_ui.auto_exit_on_run_end=true",
    ]
    return _run_tmux_job(
        session=args.home_tmux_session,
        window_name="home",
        cmd=cmd,
        status_path=args.state_dir / "home_exit_code.txt",
        timeout_s=args.home_timeout_s,
    )


def _run_rl(
    args: argparse.Namespace,
    request_path: Path,
    *,
    episode_timeout_s: float,
) -> int:
    if request_path.exists():
        request_path.unlink()
    cmd = [
        sys.executable,
        "rl/pusht_runner.py",
        "--config-file",
        args.config_file,
        "--data-saving-path",
        args.data_saving_path,
        "--pusht-reset-request-path",
        str(request_path),
        "--pusht-start-on-launch",
        "--no-pusht-keyboard-shortcuts-enabled",
        "--episode-timeout-s",
        str(float(episode_timeout_s)),
        *args.runner_args,
    ]
    return _run_tmux_job(
        session=args.rl_tmux_session,
        window_name="rl",
        cmd=cmd,
        status_path=args.state_dir / "rl_exit_code.txt",
        timeout_s=args.rl_timeout_s,
        input_group=args.runner_use_input_group,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", default="tmux/realworld_rl/tasks_config/pusht/pusht.yaml"
    )
    parser.add_argument("--data-saving-path", required=True)
    parser.add_argument(
        "--reset-script-file", default="cap/saved_scripts/place_grasped_t_reset.py"
    )
    parser.add_argument(
        "--initial-frame-script-file",
        default="cap/saved_scripts/pusht/save_top_frame.py",
    )
    parser.add_argument(
        "--home-script-file", default="cap/saved_scripts/pusht/go_home_fast.py"
    )
    parser.add_argument("--reset-skill-library-path", default="cap/saved_scripts/pusht")
    parser.add_argument("--reset-timeout-s", type=float, default=240.0)
    parser.add_argument("--home-timeout-s", type=float, default=30.0)
    parser.add_argument("--initial-frame-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--home-ok-exit-codes",
        type=int,
        nargs="+",
        default=[0, 134, 139],
        help="Exit codes accepted for the short CAP home job; 134/139 are cleanup abort/segfault after run_script Done.",
    )
    parser.add_argument("--rl-timeout-s", type=float, default=None)
    parser.add_argument("--stale-process-age-s", type=float, default=120.0)
    parser.add_argument("--max-reset-attempts", type=int, default=1)
    parser.add_argument("--rl-tmux-session", default="pusht-rl-job")
    parser.add_argument("--reset-tmux-session", default="pusht-reset-job")
    parser.add_argument("--home-tmux-session", default="pusht-home-job")
    parser.add_argument(
        "--initial-frame-tmux-session", default="pusht-initial-frame-job"
    )
    parser.add_argument(
        "--runner-use-input-group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the RL tmux job through 'sg input -c ...' so keyboard evdev access works even when tmux server lacks the input group.",
    )
    parser.add_argument("--fastapi-host", default="127.0.0.1")
    parser.add_argument("--fastapi-port", type=int, default=8203)
    parser.add_argument("--runner-fastapi-url", default="http://127.0.0.1:8204")
    parser.add_argument("--status-log-lines", type=int, default=240)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    _OWNED_TMUX_SESSIONS.update(
        {
            args.rl_tmux_session,
            args.reset_tmux_session,
            args.home_tmux_session,
            args.initial_frame_tmux_session,
        }
    )
    atexit.register(cleanup_owned_tmux_sessions)
    _install_signal_handlers()

    state_dir = Path(args.data_saving_path).expanduser() / "pusht_supervisor"
    state_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir = state_dir
    request_path = state_dir / "reset_request.json"
    result_path = state_dir / "reset_result.json"
    state = SupervisorState()
    _ensure_fastapi_port_available(args)
    _start_api(args, state)

    iteration = 0
    while True:
        while True:
            with state.lock:
                requested = state.start_requested
                episode_timeout_s = state.requested_episode_timeout_s
            if requested:
                break
            time.sleep(0.2)
        if episode_timeout_s is None:
            episode_timeout_s = 120.0

        iteration += 1
        _set_state(
            state,
            phase="rl_running",
            iteration=iteration,
            start_requested=False,
            active_session=args.rl_tmux_session,
            last_rl_code=None,
            last_reset_code=None,
            last_reset_result={},
            last_request={},
            last_error=None,
            requested_episode_timeout_s=float(episode_timeout_s),
            last_started_s=time.time(),
            last_finished_s=None,
        )
        print(
            f"[PushT supervisor] RL iteration {iteration} start "
            f"episode_timeout_s={float(episode_timeout_s):.1f}",
            flush=True,
        )
        cleanup_stale_pusht_processes(args.stale_process_age_s)
        initial_frame = _capture_initial_frame(args, iteration)
        rl_code = _run_rl(
            args,
            request_path,
            episode_timeout_s=float(episode_timeout_s),
        )
        _set_state(state, last_rl_code=rl_code)
        if rl_code != RL_RESET_REQUESTED:
            maybe_request = _load_json(request_path) if request_path.exists() else {}
            has_reset_request = bool(maybe_request.get("reason"))
            cleanup_crash_after_request = rl_code in {134, 139} and has_reset_request
            if not has_reset_request and not cleanup_crash_after_request:
                _set_state(
                    state,
                    phase="failed" if rl_code != 0 else "done",
                    active_session=args.rl_tmux_session,
                    last_error=None if rl_code == 0 else f"rl exited code {rl_code}",
                    last_finished_s=time.time(),
                )
                print(
                    f"[PushT supervisor] RL exited code={rl_code}; waiting for next /start",
                    flush=True,
                )
                continue
            print(
                f"[PushT supervisor] RL exited with cleanup code {rl_code} "
                "after writing reset request; continuing to reset",
                flush=True,
            )

        request = _load_json(request_path)
        if initial_frame:
            request["initial_frame"] = initial_frame
        _set_state(
            state,
            phase="reset_running",
            active_session=args.reset_tmux_session,
            last_request=request,
        )
        print(f"[PushT supervisor] reset requested: {request}", flush=True)
        reset_success = False
        last_reset_code = None
        last_result = {}
        for attempt in range(1, max(1, args.max_reset_attempts) + 1):
            print(
                f"[PushT supervisor] reset attempt {attempt}/{args.max_reset_attempts}",
                flush=True,
            )
            reset_success, last_reset_code, last_result = _run_reset(
                args,
                result_path,
                request,
            )
            _set_state(
                state,
                last_reset_code=last_reset_code,
                last_reset_result=last_result,
            )
            if reset_success:
                break
        if not reset_success:
            _set_state(
                state,
                phase="failed",
                active_session=args.reset_tmux_session,
                last_error="reset failed",
                last_finished_s=time.time(),
            )
            print(
                "[PushT supervisor] reset failed; waiting for next /start",
                flush=True,
            )
            continue
        _set_state(
            state,
            phase="done",
            active_session=args.rl_tmux_session,
            last_error=None,
            last_finished_s=time.time(),
        )
        print("[PushT supervisor] run complete; waiting for next /start", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
