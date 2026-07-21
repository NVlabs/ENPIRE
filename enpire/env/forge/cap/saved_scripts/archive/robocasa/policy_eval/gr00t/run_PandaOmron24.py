"""Evaluate GR00T N1.6 on PandaOmron tasks via inference_policy(). (GR00TN1.5 currently not supported yet)

Launches one model server per GPU, splits tasks evenly across GPUs,
runs all GPU groups in parallel, merges results.

Usage:
  # All 24 tasks, single GPU:
  python run_PandaOmron24.py --n-eps-per-task 10 --n-parallel-envs 10 --record-video

  # All 24 tasks, 4 GPUs:
  python run_PandaOmron24.py --n-eps-per-task 10 --n-parallel-envs 10 --gpus 0 1 2 3 --record-video

  # Single task:
  python run_PandaOmron24.py --tasks OpenDrawer --n-eps-per-task 10

  # Subset of tasks:
  python run_PandaOmron24.py --tasks OpenDrawer CloseDrawer PnPCounterToCab --gpus 0 1

  # Remote server (skip local launch):
  python run_PandaOmron24.py --server-host 127.0.0.1 --tasks OpenDrawer --n-eps-per-task 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from _common import (
    CLIENT_PYTHON,
    REPO_DIR,
    LOG_ROOT,
    MODEL_PATH,
    ModelServer,
    TeeStream,
)

CLIENT_SCRIPT = Path(__file__).resolve().parent / "_workers" / "pandaomron.py"

# fmt: off
TASKS = [
    ("robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env",          31.0),
    ("robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",          63.5),
    ("robocasa_panda_omron/CoffeePressButton_PandaOmron_Env",       98.5),
    ("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env",          81.5),
    ("robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env",          39.0),
    ("robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env",         96.0),
    ("robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env",         88.5),
    ("robocasa_panda_omron/OpenDrawer_PandaOmron_Env",              81.1),
    ("robocasa_panda_omron/CloseDrawer_PandaOmron_Env",            100.0),
    ("robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env",         91.5),
    ("robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env",        96.0),
    ("robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env",         47.5),
    ("robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env",         41.0),
    ("robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env",        46.0),
    ("robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env",        50.0),
    ("robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env",   19.0),
    ("robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env",   24.5),
    ("robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env",       63.2),
    ("robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env",       54.5),
    ("robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env",        89.0),
    ("robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env",       93.5),
    ("robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env",           87.0),
    ("robocasa_panda_omron/TurnOnStove_PandaOmron_Env",             76.5),
    ("robocasa_panda_omron/TurnOffStove_PandaOmron_Env",            31.0),
]
# fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Returns (lo%, hi%)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, centre - spread) * 100, min(1, centre + spread) * 100


def clopper_pearson_ci(
    successes: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Clopper-Pearson exact interval. Returns (lo%, hi%)."""
    if n == 0:
        return 0.0, 0.0
    lo = (
        beta_dist.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    )
    hi = (
        beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes)
        if successes < n
        else 1.0
    )
    return lo * 100, hi * 100


def short_name(env_name: str) -> str:
    return env_name.split("/")[-1].replace("_PandaOmron_Env", "")


def _tail(path: Path, n_lines: int = 5) -> str:
    """Return the last n_lines of a file, or a placeholder if unreadable."""
    try:
        lines = path.read_text().splitlines()
        return "\n    ".join(lines[-n_lines:])
    except OSError:
        return "<log not available>"


def split_tasks(tasks: list, n_groups: int) -> list[list]:
    """Split tasks into n_groups as evenly as possible."""
    base = len(tasks) // n_groups
    remainder = len(tasks) % n_groups
    groups, idx = [], 0
    for g in range(n_groups):
        size = base + (1 if g < remainder else 0)
        groups.append(tasks[idx : idx + size])
        idx += size
    return groups


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def run_task(
    env_name: str,
    log_dir: Path,
    server_host: str,
    server_port: int,
    n_episodes: int,
    n_envs: int,
    gpu_id: int = 0,
    record_video: bool = False,
) -> dict | None:
    """Run a single task evaluation as a subprocess. Returns parsed results or None."""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_DIR),
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "MUJOCO_EGL_DEVICE_ID": str(gpu_id),
    }
    cmd = [
        str(CLIENT_PYTHON),
        str(CLIENT_SCRIPT),
        "--server-host",
        server_host,
        "--server-port",
        str(server_port),
        "--env-name",
        env_name,
        "--log-dir",
        str(log_dir),
        "--n-eps-per-task",
        str(n_episodes),
        "--n-parallel-envs",
        str(n_envs),
        "--model-name",
        "GR00T-N1.6-3B",
        "--model-path",
        str(MODEL_PATH),
    ]
    if record_video:
        cmd.append("--record-video")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    client_log = log_dir / "client.log"

    with open(client_log, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

    if result.returncode != 0:
        name = short_name(env_name)
        tail = _tail(client_log)
        print(
            f"  [{name}] exit code {result.returncode}. "
            f"Tail of {client_log}:\n    {tail}",
            file=sys.stderr,
        )
        return None

    rfile = log_dir / "task_results.json"
    if rfile.exists():
        return json.loads(rfile.read_text())
    return None


def run_gpu_group(
    gpu_id: int,
    port: int,
    task_slice: list[tuple[str, float]],
    run_dir: Path,
    n_episodes: int,
    n_envs: int,
    record_video: bool,
    results_file: ResultsFile,
    server_host: str | None = None,
) -> None:
    """Run task_slice sequentially, launching a local server unless server_host is set."""
    if server_host:
        tag = "[remote]"
        host = server_host
        server_ctx = nullcontext()
    else:
        tag = f"[GPU {gpu_id}]"
        host = "127.0.0.1"
        log_path = run_dir / f"server_gpu{gpu_id}.log"
        server_ctx = ModelServer(gpu_id, port, log_path)
        print(f"{tag} Starting server on port {port}...")

    with server_ctx:
        if not server_host:
            print(f"{tag} Server ready.", end=" ")
        print(f"{len(task_slice)} tasks.\n")

        for i, (env_name, official) in enumerate(task_slice):
            name = short_name(env_name)
            task_dir = run_dir / name
            print(
                f"{tag}[{i + 1:2d}/{len(task_slice)}] {name:<30s} ... ",
                end="",
                flush=True,
            )

            t0 = time.time()
            res = run_task(
                env_name, task_dir, host, port, n_episodes, n_envs, gpu_id, record_video
            )
            elapsed = time.time() - t0

            results_file.record(env_name, res)

            if res:
                s = res["successes"]
                n_succ, n = sum(s), len(s)
                w_lo, w_hi = wilson_ci(n_succ, n)
                rate = n_succ / n * 100 if n else 0.0
                print(
                    f"{rate:5.1f}%  95%CI [{w_lo:5.1f}, {w_hi:5.1f}]  (official {official:.1f}%)  [{elapsed:.0f}s]"
                )
            else:
                print(f"FAILED  [{elapsed:.0f}s]")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(
    all_results: dict[str, dict | None],
    tasks: list[tuple[str, float]],
    wall_time: float,
) -> None:
    t_min, t_sec = divmod(int(wall_time), 60)
    print(f"\nTotal wall time: {t_min}m{t_sec:02d}s")

    print(f"\n{'=' * 90}")
    print(
        f"  {'Task':<30s} {'Ours':>6s} {'95% CI':>16s} {'Official':>9s} {'Delta':>7s}"
    )
    print(f"{'-' * 90}")

    rates, officials = [], []
    for env_name, official in tasks:
        name = short_name(env_name)
        res = all_results.get(env_name)
        if res and isinstance(res, dict) and "successes" in res:
            s = res["successes"]
            n_succ, n = sum(s), len(s)
            w_lo, w_hi = wilson_ci(n_succ, n)
            rate = n_succ / n * 100 if n else 0.0
            delta = rate - official
            print(
                f"  {name:<30s} {rate:5.1f}%  [{w_lo:5.1f}, {w_hi:5.1f}]  {official:7.1f}%  {delta:+6.1f}"
            )
            rates.append(rate)
            officials.append(official)
        else:
            print(f"  {name:<30s}  FAIL")

    if rates:
        avg = float(np.mean(rates))
        off_avg = float(np.mean(officials))
        std = float(np.std(rates))
        print(f"{'-' * 90}")
        print(
            f"  {'Average':<30s} {avg:5.1f}%  std={std:5.1f}        {off_avg:7.1f}%  {avg - off_avg:+6.1f}"
        )
        print(f"  Tasks completed: {len(rates)}/{len(tasks)}")


class ResultsFile:
    """Thread-safe incremental JSON results writer.

    Writes benchmark_results.json after every task completes so that
    partial results are available even if the run is interrupted.
    """

    def __init__(
        self,
        path: Path,
        tasks: list[tuple[str, float]],
        *,
        timestamp: str,
        n_episodes: int,
        n_envs: int,
        gpus: list[int],
        t_start: float,
    ) -> None:
        self._path = path
        self._tasks = tasks
        self._meta = {
            "timestamp": timestamp,
            "interface": "inference_policy",
            "model_path": str(MODEL_PATH),
            "n_eps_per_task": n_episodes,
            "n_envs": n_envs,
            "gpus": gpus,
        }
        self._t_start = t_start
        self._results: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    @property
    def results(self) -> dict[str, dict | None]:
        with self._lock:
            return dict(self._results)

    def record(self, env_name: str, result: dict | None) -> None:
        """Record a task result and rewrite the JSON file."""
        with self._lock:
            self._results[env_name] = result
            self._write()

    def finalize(self) -> Path:
        """Final write with wall_time. Returns the path."""
        with self._lock:
            self._write()
        return self._path

    def _write(self) -> None:
        summary = {
            **self._meta,
            "wall_time_s": time.time() - self._t_start,
            "tasks_completed": 0,
            "tasks_total": len(self._tasks),
            "per_task": {},
        }
        rates, officials = [], []
        for env_name, official in self._tasks:
            name = short_name(env_name)
            res = self._results.get(env_name)
            if res and isinstance(res, dict) and "successes" in res:
                s = res["successes"]
                n_succ, n = sum(s), len(s)
                rate = n_succ / n * 100 if n else 0.0
                w_lo, w_hi = wilson_ci(n_succ, n)
                cp_lo, cp_hi = clopper_pearson_ci(n_succ, n)
                summary["per_task"][env_name] = {
                    "short_name": name,
                    "success_rate": rate,
                    "successes": n_succ,
                    "total": n,
                    "wilson_ci_95": [round(w_lo, 2), round(w_hi, 2)],
                    "clopper_pearson_ci_95": [round(cp_lo, 2), round(cp_hi, 2)],
                    "official_rate": official,
                    "per_episode": s,
                }
                rates.append(rate)
                officials.append(official)
            elif env_name in self._results:
                summary["per_task"][env_name] = {"short_name": name, "status": "FAILED"}

        summary["tasks_completed"] = len(rates)
        if rates:
            summary["aggregate"] = {
                "mean": round(float(np.mean(rates)), 2),
                "std": round(float(np.std(rates)), 2),
                "official_mean": round(float(np.mean(officials)), 2),
                "n_tasks": len(rates),
            }

        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        tmp.rename(self._path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="GR00T N1.6 -- PandaOmron evaluation")
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task short names to run (default: all 24)",
    )
    p.add_argument("--n-eps-per-task", type=int, default=10)
    p.add_argument(
        "--n-parallel-envs", type=int, default=2, help="Parallel envs per GPU"
    )
    p.add_argument("--gpus", type=int, nargs="+", default=[0], help="CUDA device IDs")
    p.add_argument("--base-port", type=int, default=5555, help="First server port")
    p.add_argument(
        "--server-host", default=None, help="Remote server host (skip local launch)"
    )
    p.add_argument("--record-video", action="store_true", help="Save per-camera videos")
    args = p.parse_args()

    # Filter tasks
    if args.tasks:
        valid = {short_name(env) for env, _ in TASKS}
        bad = [t for t in args.tasks if t not in valid]
        if bad:
            p.error(f"Unknown tasks: {bad}. Valid: {sorted(valid)}")
        selected = set(args.tasks)
        tasks = [(env, off) for env, off in TASKS if short_name(env) in selected]
    else:
        tasks = TASKS

    # Remote server: ignore --gpus, run all tasks in one group
    if args.server_host:
        gpus = [0]
    else:
        gpus = args.gpus

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = LOG_ROOT / f"eval_panda_ip_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to log files in run_dir (real-time)
    tee_out = TeeStream(sys.stdout, run_dir / "stdout.log")
    tee_err = TeeStream(sys.stderr, run_dir / "stderr.log")
    sys.stdout, sys.stderr = tee_out, tee_err

    try:
        server_label = args.server_host or f"local GPUs={gpus}"
        print(
            f"PandaOmron eval: {len(tasks)} tasks, {args.n_eps_per_task} eps/task, "
            f"n_envs={args.n_parallel_envs}, server={server_label}"
        )
        print(f"Log dir: {run_dir}\n")

        t_start = time.time()
        task_groups = split_tasks(tasks, len(gpus))
        results_file = ResultsFile(
            run_dir / "benchmark_results.json",
            tasks,
            timestamp=timestamp,
            n_episodes=args.n_eps_per_task,
            n_envs=args.n_parallel_envs,
            gpus=gpus,
            t_start=t_start,
        )

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {}
            for g, (gpu_id, task_slice) in enumerate(zip(gpus, task_groups)):
                f = pool.submit(
                    run_gpu_group,
                    gpu_id,
                    args.base_port + g,
                    task_slice,
                    run_dir,
                    args.n_eps_per_task,
                    args.n_parallel_envs,
                    args.record_video,
                    results_file,
                    args.server_host,
                )
                futures[f] = gpu_id

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"[GPU {futures[f]}] CRASHED: {e}", file=sys.stderr)

        wall_time = time.time() - t_start
        print_summary(results_file.results, tasks, wall_time)
        path = results_file.finalize()
        print(f"\nFull results: {path}")
    finally:
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()


if __name__ == "__main__":
    main()
