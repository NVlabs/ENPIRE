"""grootpool server — asyncio ROUTER loop + supervisor + admin.

Run on OSMO inside the ``Isaac-GR00T-benchmark/model_server_venv`` (torch + gr00t)
with ``PYTHONPATH`` pointing at the forge repo root:

    PYTHONPATH=/path/to/forge \\
      /path/to/Isaac-GR00T-benchmark/model_server_venv/bin/python \\
      -m cap.policy.grootpool.server --n-workers 2 --gpu-ids 0,1 \\
      --model-path /path/to/gr00t_n1-5/.../checkpoint-120000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

import zmq
import zmq.asyncio

from enpire.env.forge.cap.policy.grootpool import protocol as P
from enpire.env.forge.cap.policy.grootpool.session_manager import GrootPoolError, RequestDispatcher
from enpire.env.forge.cap.policy.grootpool.supervisor import WorkerConfig, WorkerSupervisor

log = logging.getLogger("grootpool.server")


def _parse_args() -> argparse.Namespace:
    def env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)

    p = argparse.ArgumentParser(description="GR00T inference pool middleware")
    p.add_argument(
        "--listen-port",
        type=int,
        default=int(env("GROOTPOOL_LISTEN_PORT", "7070")),
        help="ZMQ ROUTER port for CAP scripts",
    )
    p.add_argument(
        "--admin-port",
        type=int,
        default=int(env("GROOTPOOL_ADMIN_PORT", "7071")),
        help="FastAPI admin port",
    )
    p.add_argument(
        "--n-workers", type=int, default=int(env("GROOTPOOL_N_WORKERS", "2"))
    )
    p.add_argument(
        "--base-port",
        type=int,
        default=int(env("GROOTPOOL_BASE_PORT", "5555")),
        help="First worker ZMQ port; increments per worker",
    )
    p.add_argument(
        "--gpu-ids",
        type=str,
        default=env("GROOTPOOL_GPU_IDS", "0,1"),
        help="Comma-separated GPU ids; cycled for workers",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=env("GROOTPOOL_MODEL_PATH"),
        help="Path to N1.5 checkpoint (required unless --mock)",
    )
    p.add_argument(
        "--model-version",
        type=str,
        default=env("GROOTPOOL_MODEL_VERSION", "n15"),
        choices=["n15"],
    )
    p.add_argument(
        "--server-python",
        type=str,
        default=env("GROOTPOOL_SERVER_PYTHON"),
        help="Python interpreter for GR00T worker subprocs. "
        "Defaults to GROOT_BENCHMARK_ROOT/model_server_venv/bin/python",
    )
    p.add_argument(
        "--groot-benchmark-root",
        type=str,
        default=env("GROOT_BENCHMARK_ROOT"),
        help="Path to Isaac-GR00T-benchmark clone (used to locate server venv).",
    )
    p.add_argument(
        "--log-dir", type=str, default=env("GROOTPOOL_LOG_DIR", "logs/grootpool")
    )
    p.add_argument("--startup-timeout", type=float, default=300.0)
    p.add_argument("--step-timeout", type=float, default=30.0)
    # open-timeout bounds how long a client waits for an idle worker. With
    # N concurrent clients >> M workers each running a full episode, this has
    # to be at least (N/M)*episode_len — the previous 60s default silently
    # cratered batched evals on a small pool (n_seeds=50, M=3 → most seeds
    # got ERR_NO_WORKER and the arm never moved).
    p.add_argument("--open-timeout", type=float, default=3600.0)
    # session-idle-timeout reaps sessions that stop stepping. Must be
    # larger than any legitimate pause between steps (e.g. sim reset, VLM
    # overlay capture), otherwise live sessions get GC'd mid-rollout.
    p.add_argument("--session-idle-timeout", type=float, default=300.0)
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use in-proc mock workers (no subprocess, no torch, no gr00t).",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="Spawn torch-only stub worker subprocesses instead of real N1.5. "
        "Validates the full subprocess + ZMQ path without building model_server_venv.",
    )
    p.add_argument(
        "--external-workers",
        type=str,
        default=env("GROOTPOOL_EXTERNAL_WORKERS"),
        help="Comma-separated host:port list of pre-running workers "
        "(e.g. from tmux/launch_grootpool.sh). When set, middleware "
        "does not spawn workers — it only connects to them.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_worker_configs(args: argparse.Namespace) -> list[WorkerConfig]:
    # External-workers path: one worker per host:port pair, no spawning.
    if args.external_workers:
        endpoints = [s.strip() for s in args.external_workers.split(",") if s.strip()]
        if not endpoints:
            raise SystemExit("--external-workers is empty")
        configs = []
        for i, ep in enumerate(endpoints):
            if ":" not in ep:
                raise SystemExit(f"bad external worker '{ep}' (expected host:port)")
            host, port_s = ep.rsplit(":", 1)
            configs.append(
                WorkerConfig(
                    worker_id=i,
                    gpu_id=-1,
                    port=int(port_s),
                    host=host,
                    model_path=Path("/dev/null"),
                    server_python=Path("/dev/null"),
                    server_script=Path("/dev/null"),
                    external=True,
                    startup_timeout=args.startup_timeout,
                    call_timeout=args.step_timeout,
                )
            )
        return configs

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise SystemExit("--gpu-ids must be non-empty")

    if args.mock:
        server_python = Path("/dev/null")
        server_script = Path("/dev/null")
        model_path = Path("/dev/null")
    elif args.stub:
        # Stub workers only need torch + zmq — the current interpreter is fine.
        server_python = Path(sys.executable)
        server_script = Path(__file__).resolve().parent / "_servers/stub.py"
        model_path = Path(args.model_path or "/dev/null")
    else:
        if not args.model_path:
            raise SystemExit("--model-path is required unless --mock or --stub")
        model_path = Path(args.model_path)

        if args.server_python:
            server_python = Path(args.server_python)
        elif args.groot_benchmark_root:
            server_python = (
                Path(args.groot_benchmark_root) / "model_server_venv/bin/python"
            )
        else:
            # Fall back to the current interpreter. Useful when the middleware
            # itself runs inside the model_server_venv.
            server_python = Path(sys.executable)

        # Use the standalone N1.5 server script we already ship.
        server_script = (
            Path(__file__).resolve().parents[2]
            / "saved_scripts/robocasa/policy_eval/gr00t/_servers/n15.py"
        )
        if not server_script.exists():
            raise SystemExit(f"missing N1.5 server script: {server_script}")

    log_dir = Path(args.log_dir)

    extra_env = {}
    if args.groot_benchmark_root:
        # N1.5 server script expects gr00t on PYTHONPATH.
        extra_env["PYTHONPATH"] = (
            args.groot_benchmark_root + os.pathsep + os.environ.get("PYTHONPATH", "")
        )

    configs: list[WorkerConfig] = []
    for i in range(args.n_workers):
        gpu = gpu_ids[i % len(gpu_ids)]
        configs.append(
            WorkerConfig(
                worker_id=i,
                gpu_id=gpu,
                port=args.base_port + i,
                model_path=model_path,
                server_python=server_python,
                server_script=server_script,
                embodiment="new_embodiment",
                startup_timeout=args.startup_timeout,
                shutdown_timeout=10.0,
                call_timeout=args.step_timeout,
                log_path=log_dir / f"worker_{i}_gpu{gpu}.log",
                extra_env=extra_env,
            )
        )
    return configs


async def _router_loop(
    socket: zmq.asyncio.Socket,
    dispatcher: RequestDispatcher,
    shutdown: asyncio.Event,
) -> None:
    while not shutdown.is_set():
        try:
            parts = await asyncio.wait_for(socket.recv_multipart(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if len(parts) == 2:
            identity, payload = parts
        elif len(parts) == 3 and parts[1] == b"":
            identity, payload = parts[0], parts[2]
        else:
            log.warning("malformed message (len=%d)", len(parts))
            continue
        asyncio.create_task(_handle_request(socket, identity, payload, dispatcher))


async def _handle_request(
    socket: zmq.asyncio.Socket,
    identity: bytes,
    payload: bytes,
    dispatcher: RequestDispatcher,
) -> None:
    try:
        msg = P.unpack(payload)
    except Exception as exc:
        await _reply(socket, identity, P.error(P.ERR_BAD_REQUEST, f"unpack failed: {exc}"))
        return

    op = msg.get("op")
    try:
        if op == "step":
            action = await dispatcher.step(msg.get("obs", {}))
            await _reply(socket, identity, {"op": "step_ok", "action": action})
        elif op == "ping":
            snap = dispatcher.status_snapshot()
            await _reply(socket, identity, {"op": "pong", **snap})
        else:
            await _reply(socket, identity, P.error(P.ERR_BAD_REQUEST, f"unknown op: {op}"))
    except GrootPoolError as exc:
        await _reply(socket, identity, P.error(exc.code, exc.message))
    except Exception as exc:
        log.exception("internal error handling %s", op)
        await _reply(socket, identity, P.error(P.ERR_INTERNAL, str(exc)))


async def _reply(socket: zmq.asyncio.Socket, identity: bytes, msg: dict) -> None:
    try:
        await socket.send_multipart([identity, P.pack(msg)])
    except Exception:
        log.exception("failed to reply to %s", identity)


async def _run(args: argparse.Namespace) -> int:
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    configs = _build_worker_configs(args)

    dispatcher_holder: dict[str, RequestDispatcher] = {}

    async def on_crash(wid: int) -> None:
        if "d" in dispatcher_holder:
            await dispatcher_holder["d"].on_worker_crash(wid)

    async def on_respawn(wid: int) -> None:
        if "d" in dispatcher_holder:
            await dispatcher_holder["d"].on_worker_respawned(wid)

    sup = WorkerSupervisor(
        configs,
        mock=args.mock,
        respawn=True,
        on_crash=on_crash,
        on_respawn=on_respawn,
    )
    if args.external_workers:
        mode = f"external ({len(configs)} endpoints)"
    elif args.mock:
        mode = "mock"
    elif args.stub:
        mode = "stub"
    else:
        mode = "real"
    log.info("starting %d workers (%s)…", len(configs), mode)
    await sup.start()

    dispatcher = RequestDispatcher(
        sup,
        step_timeout=args.step_timeout,
        no_worker_timeout=args.open_timeout,
    )
    dispatcher_holder["d"] = dispatcher
    await dispatcher.start()

    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(f"tcp://0.0.0.0:{args.listen_port}")
    log.info("listening for CAP clients on tcp://0.0.0.0:%d", args.listen_port)

    from enpire.env.forge.cap.policy.grootpool.admin import run_admin_server

    admin_task = asyncio.create_task(
        run_admin_server(sup, dispatcher, args.admin_port, shutdown)
    )

    router_task = asyncio.create_task(_router_loop(router, dispatcher, shutdown))

    t0 = time.monotonic()
    log.info("grootpool ready in %.1fs", time.monotonic() - t0)

    await shutdown.wait()
    log.info("shutdown signal received, stopping…")

    router_task.cancel()
    try:
        await router_task
    except (asyncio.CancelledError, Exception):
        pass
    router.close(linger=0)
    ctx.term()

    await dispatcher.stop()
    await sup.stop()

    admin_task.cancel()
    try:
        await admin_task
    except (asyncio.CancelledError, Exception):
        pass

    log.info("grootpool stopped")
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
