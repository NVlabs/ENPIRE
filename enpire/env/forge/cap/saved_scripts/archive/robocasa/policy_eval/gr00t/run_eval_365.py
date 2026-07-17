"""Evaluate GR00T on the RoboCasa365 benchmark (atomic_seen / composite_seen / composite_unseen).

Supports both GR00T N1.5 (official benchmark) and N1.6 on the 50-task eval suite.
Multi-GPU parallel evaluation with per-task horizon, Wilson + Clopper-Pearson CIs,
video recording, and resume support.

Usage:
  # N1.5 on all 50 benchmark tasks, 4 GPUs (official comparison):
  python run_eval_365.py --model-version n15 \\
    --task-set atomic_seen composite_seen composite_unseen \\
    --gpus 0 1 2 3 --record-video

  # N1.6 on atomic_seen only:
  python run_eval_365.py --model-version n16 \\
    --task-set atomic_seen --gpus 0 1

  # Quick test — 1 task, 2 episodes:
  python run_eval_365.py --model-version n15 \\
    --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1

  # Reuse the shared forge venv as the RoboCasa365 client worker:
  python run_eval_365.py --model-version n15 \\
    --client-python /path/to/forge/.venv/bin/python \\
    --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1

  # Stats only (no eval):
  python run_eval_365.py --stats-only --log-dir <dir>
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

from _common.robocasa365 import (
    CLIENT_SCRIPT_365,
    REPO_DIR,
    LOG_ROOT,
    ModelServer365,
    TeeStream,
    get_client_python_365,
    get_model_config,
)

# Task registry loaded from a pre-extracted JSON file.
# Generated from robocasa365 dataset_registry via a RoboCasa365-capable client env.
# No robocasa import needed in the orchestrator.
_REGISTRY_PATH = Path(__file__).parent / "task_registry_365.json"
if not _REGISTRY_PATH.exists():
    print(
        f"ERROR: {_REGISTRY_PATH} not found. Generate it with a RoboCasa365-capable client env.",
        file=sys.stderr,
    )
    sys.exit(1)
_REG = json.loads(_REGISTRY_PATH.read_text())
TARGET_TASKS: dict[str, list[str]] = _REG["target_tasks"]
TASK_SET_REGISTRY: dict[str, list[str]] = _REG["task_set_registry"]
_HORIZONS: dict[str, int] = _REG["horizons"]


def get_task_horizon(task: str) -> int:
    if task not in _HORIZONS:
        raise ValueError(f"Unknown task: {task}. Not in task_registry_365.json.")
    return _HORIZONS[task]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
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


def _tail(path: Path, n_lines: int = 5) -> str:
    try:
        lines = path.read_text().splitlines()
        return "\n    ".join(lines[-n_lines:])
    except OSError:
        return "<log not available>"


def split_tasks(tasks: list, n_groups: int) -> list[list]:
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
    task_name: str,
    horizon: int,
    log_dir: Path,
    client_python: Path,
    server_host: str,
    server_port: int,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    model_version: str,
    model_name: str,
    model_path: str,
    split: str,
    gpu_id: int = 0,
    record_video: bool = False,
) -> dict | None:
    from _common.robocasa365 import N16_GROOT_ROOT, N15_GROOT_ROOT

    env_name = f"robocasa/{task_name}"
    # Both versions need the benchmark repo for the gymnasium-1.0-compatible
    # MultiStepWrapper. N1.6 also needs the old Isaac-GR00T for its server.
    # Benchmark repo goes FIRST so its MultiStepWrapper wins the import.
    if model_version == "n16":
        pypath = f"{REPO_DIR}:{N15_GROOT_ROOT}:{N16_GROOT_ROOT}"
    else:
        pypath = f"{REPO_DIR}:{N15_GROOT_ROOT}"
    env = {
        **os.environ,
        "PYTHONPATH": pypath,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "MUJOCO_EGL_DEVICE_ID": str(gpu_id),
    }
    cmd = [
        str(client_python),
        str(CLIENT_SCRIPT_365),
        "--server-host",
        server_host,
        "--server-port",
        str(server_port),
        "--env-name",
        env_name,
        "--log-dir",
        str(log_dir),
        "--split",
        split,
        "--model-version",
        model_version,
        "--model-name",
        model_name,
        "--model-path",
        model_path,
        "--n-eps-per-task",
        str(n_episodes),
        "--n-action-steps",
        str(n_action_steps),
        "--max-episode-steps",
        str(horizon),
        "--n-parallel-envs",
        str(n_envs),
    ]
    if record_video:
        cmd.append("--record-video")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    client_log = log_dir / "client.log"

    with open(client_log, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

    if result.returncode != 0:
        tail = _tail(client_log)
        print(
            f"  [{task_name}] exit code {result.returncode}. Tail:\n    {tail}",
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
    task_slice: list[tuple[str, int]],
    run_dir: Path,
    client_python: Path,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    model_version: str,
    model_config: dict,
    split: str,
    record_video: bool,
    results_file: ResultsFile,
    server_host: str | None = None,
) -> None:
    if server_host:
        tag = "[remote]"
        host = server_host
        server_ctx = nullcontext()
    else:
        tag = f"[GPU {gpu_id}]"
        host = "127.0.0.1"
        log_path = run_dir / f"server_gpu{gpu_id}.log"
        server_ctx = ModelServer365(gpu_id, port, log_path, model_config)
        print(f"{tag} Starting server on port {port}...")

    with server_ctx:
        if not server_host:
            print(f"{tag} Server ready.", end=" ")
        print(f"{len(task_slice)} tasks.\n")

        for i, (task_name, horizon) in enumerate(task_slice):
            task_dir = run_dir / task_name

            # Resume: skip tasks with existing results
            rfile = task_dir / "task_results.json"
            if rfile.exists():
                try:
                    res = json.loads(rfile.read_text())
                    results_file.record(task_name, res)
                    s = res["successes"]
                    rate = sum(s) / len(s) * 100 if s else 0.0
                    print(
                        f"{tag}[{i + 1:2d}/{len(task_slice)}] {task_name:<35s} SKIP (cached {rate:.1f}%)"
                    )
                    continue
                except Exception:
                    pass

            print(
                f"{tag}[{i + 1:2d}/{len(task_slice)}] {task_name:<35s} (H={horizon}) ... ",
                end="",
                flush=True,
            )
            t0 = time.time()
            res = run_task(
                task_name,
                horizon,
                task_dir,
                client_python,
                host,
                port,
                n_episodes,
                n_envs,
                n_action_steps,
                model_version,
                model_config["model_name"],
                str(model_config["model_path"]),
                split,
                gpu_id,
                record_video,
            )
            elapsed = time.time() - t0
            results_file.record(task_name, res)

            if res:
                s = res["successes"]
                n_succ, n = sum(s), len(s)
                w_lo, w_hi = wilson_ci(n_succ, n)
                cp_lo, cp_hi = clopper_pearson_ci(n_succ, n)
                rate = n_succ / n * 100 if n else 0.0
                print(
                    f"{rate:5.1f}%  "
                    f"W[{w_lo:5.1f},{w_hi:5.1f}]  "
                    f"CP[{cp_lo:5.1f},{cp_hi:5.1f}]  "
                    f"[{elapsed:.0f}s]"
                )
            else:
                print(f"FAILED  [{elapsed:.0f}s]")


# ---------------------------------------------------------------------------
# Results + reporting
# ---------------------------------------------------------------------------


class ResultsFile:
    """Thread-safe incremental JSON results writer with Wilson + CP intervals."""

    def __init__(
        self,
        path: Path,
        tasks: list[tuple[str, int]],
        *,
        timestamp: str,
        n_episodes: int,
        n_envs: int,
        gpus: list[int],
        model_version: str,
        model_path: str,
        split: str,
        t_start: float,
    ) -> None:
        self._path = path
        self._tasks = tasks
        self._meta = {
            "timestamp": timestamp,
            "model_version": model_version,
            "model_path": model_path,
            "split": split,
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

    def record(self, task_name: str, result: dict | None) -> None:
        with self._lock:
            self._results[task_name] = result
            self._write()

    def finalize(self) -> Path:
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
        rates = []
        for task_name, horizon in self._tasks:
            res = self._results.get(task_name)
            if res and isinstance(res, dict) and "successes" in res:
                s = res["successes"]
                n_succ, n = sum(s), len(s)
                rate = n_succ / n * 100 if n else 0.0
                w_lo, w_hi = wilson_ci(n_succ, n)
                cp_lo, cp_hi = clopper_pearson_ci(n_succ, n)
                summary["per_task"][task_name] = {
                    "success_rate": rate,
                    "successes": n_succ,
                    "total": n,
                    "horizon": horizon,
                    "wilson_ci_95": [round(w_lo, 2), round(w_hi, 2)],
                    "clopper_pearson_ci_95": [round(cp_lo, 2), round(cp_hi, 2)],
                    "per_episode": s,
                }
                rates.append(rate)
            elif task_name in self._results:
                summary["per_task"][task_name] = {"status": "FAILED"}

        summary["tasks_completed"] = len(rates)
        if rates:
            summary["aggregate"] = {
                "mean": round(float(np.mean(rates)), 2),
                "std": round(float(np.std(rates)), 2),
                "n_tasks": len(rates),
            }

        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        tmp.rename(self._path)


def print_summary(
    all_results: dict,
    tasks_by_group: dict[str, list[tuple[str, int]]],
    wall_time: float,
) -> None:
    t_min, t_sec = divmod(int(wall_time), 60)
    print(f"\nTotal wall time: {t_min}m{t_sec:02d}s")

    for group, task_list in tasks_by_group.items():
        print(f"\n{'=' * 100}")
        print(f"  {group.upper()} ({len(task_list)} tasks)")
        print(f"  {'Task':<35s} {'Rate':>6s} {'Wilson 95%':>16s} {'CP 95%':>16s}")
        print(f"{'-' * 100}")

        rates = []
        for task_name, horizon in task_list:
            res = all_results.get(task_name)
            if res and isinstance(res, dict) and "successes" in res:
                s = res["successes"]
                n_succ, n = sum(s), len(s)
                rate = n_succ / n * 100 if n else 0.0
                w_lo, w_hi = wilson_ci(n_succ, n)
                cp_lo, cp_hi = clopper_pearson_ci(n_succ, n)
                print(
                    f"  {task_name:<35s} {rate:5.1f}%  "
                    f"W[{w_lo:5.1f},{w_hi:5.1f}]  "
                    f"CP[{cp_lo:5.1f},{cp_hi:5.1f}]"
                )
                rates.append(rate)
            else:
                print(f"  {task_name:<35s}  FAIL")

        if rates:
            avg = float(np.mean(rates))
            std = float(np.std(rates))
            print(f"{'-' * 100}")
            print(
                f"  {'Average':<35s} {avg:5.1f}%  std={std:.1f}  ({len(rates)}/{len(task_list)} tasks)"
            )


def print_stats_only(log_dir: Path, task_sets: list[str]) -> None:
    """Mimics official get_eval_stats.py but with Wilson + CP intervals."""
    results_file = log_dir / "benchmark_results.json"
    if not results_file.exists():
        print(f"No results at {results_file}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(results_file.read_text())
    print(f"Results from: {results_file}")
    print(f"Model: {data.get('model_version', '?')}, Split: {data.get('split', '?')}")

    for group in task_sets:
        task_names = TARGET_TASKS.get(group, [])
        print(f"\n{'=' * 80}")
        print(f"  {group.upper()} ({len(task_names)} tasks)")
        print(f"{'-' * 80}")
        rates = []
        for name in task_names:
            info = data.get("per_task", {}).get(name)
            if info and "success_rate" in info:
                r = info["success_rate"]
                w = info.get("wilson_ci_95", [0, 0])
                cp = info.get("clopper_pearson_ci_95", [0, 0])
                print(
                    f"  {name:<35s} {r:5.1f}%  W[{w[0]:5.1f},{w[1]:5.1f}]  CP[{cp[0]:5.1f},{cp[1]:5.1f}]"
                )
                rates.append(r)
            else:
                print(f"  {name:<35s}  {'FAIL' if info else 'MISSING'}")
        if rates:
            print(f"{'-' * 80}")
            print(
                f"  {'Average':<35s} {np.mean(rates):5.1f}%  ({len(rates)}/{len(task_names)} tasks)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="RoboCasa365 Benchmark Evaluation")
    p.add_argument("--model-version", choices=["n15", "n16"], default="n15")
    p.add_argument(
        "--task-set",
        nargs="+",
        default=["atomic_seen", "composite_seen", "composite_unseen"],
        help="Task groups to evaluate",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Individual task names (overrides --task-set)",
    )
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--n-eps-per-task", type=int, default=50)
    p.add_argument("--n-parallel-envs", type=int, default=5)
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=16,
        help="Steps executed per policy query (16=official, 8=receding horizon)",
    )
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--base-port", type=int, default=5555)
    p.add_argument("--server-host", default=None)
    p.add_argument(
        "--client-python",
        default=None,
        help=(
            "Python interpreter for RoboCasa365 worker subprocesses. "
            "Defaults to ROBOCASA365_CLIENT_PYTHON, then UV_PROJECT_ENVIRONMENT/bin/python, "
            "then .venv/bin/python, then Isaac-GR00T-benchmark/client_venv/bin/python."
        ),
    )
    p.add_argument("--record-video", action="store_true")
    p.add_argument(
        "--stats-only", action="store_true", help="Print stats from existing results"
    )
    p.add_argument(
        "--log-dir", default=None, help="Log dir (for --stats-only or override)"
    )
    args = p.parse_args()

    # Stats-only mode
    if args.stats_only:
        if not args.log_dir:
            p.error("--stats-only requires --log-dir")
        print_stats_only(Path(args.log_dir), args.task_set)
        return

    model_config = get_model_config(args.model_version)
    client_python = get_client_python_365(args.client_python)
    if not client_python.exists():
        p.error(
            "Client Python not found at "
            f"{client_python}. Pass --client-python, set ROBOCASA365_CLIENT_PYTHON, "
            "or create Isaac-GR00T-benchmark/client_venv."
        )
    if not os.access(client_python, os.X_OK):
        p.error(f"Client Python is not executable: {client_python}")

    # Build task list: [(task_name, horizon), ...]
    tasks_by_group: dict[str, list[tuple[str, int]]] = {}
    if args.tasks:
        # Individual tasks specified
        all_known = set()
        for ts in TASK_SET_REGISTRY.values():
            all_known.update(ts if isinstance(ts, list) else [])
        for t in args.tasks:
            if t not in all_known:
                p.error(f"Unknown task: {t}")
        tasks_by_group["selected"] = [(t, get_task_horizon(t)) for t in args.tasks]
    else:
        for group in args.task_set:
            if group not in TASK_SET_REGISTRY:
                p.error(
                    f"Unknown task set: {group}. Valid: {list(TASK_SET_REGISTRY.keys())}"
                )
            names = TASK_SET_REGISTRY[group]
            tasks_by_group[group] = [(t, get_task_horizon(t)) for t in names]

    # Flatten for scheduling
    all_tasks = []
    seen = set()
    for group_tasks in tasks_by_group.values():
        for t in group_tasks:
            if t[0] not in seen:
                all_tasks.append(t)
                seen.add(t[0])

    if not all_tasks:
        p.error("No tasks to evaluate")

    gpus = [0] if args.server_host else args.gpus

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = (
        Path(args.log_dir)
        if args.log_dir
        else LOG_ROOT / f"{args.model_version}_{args.split}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tee_out = TeeStream(sys.stdout, run_dir / "stdout.log")
    tee_err = TeeStream(sys.stderr, run_dir / "stderr.log")
    sys.stdout, sys.stderr = tee_out, tee_err

    try:
        server_label = args.server_host or f"local GPUs={gpus}"
        print(
            f"RoboCasa365 eval: {len(all_tasks)} tasks, "
            f"{args.n_eps_per_task} eps/task, n_envs={args.n_parallel_envs}, "
            f"n_action_steps={args.n_action_steps}, "
            f"model={model_config['model_name']}, split={args.split}, "
            f"server={server_label}"
        )
        print(f"Client python: {client_python}")
        print(f"Log dir: {run_dir}\n")
        for group, tlist in tasks_by_group.items():
            print(f"  {group}: {len(tlist)} tasks")
        print()

        t_start = time.time()
        task_groups = split_tasks(all_tasks, len(gpus))
        results_file = ResultsFile(
            run_dir / "benchmark_results.json",
            all_tasks,
            timestamp=timestamp,
            n_episodes=args.n_eps_per_task,
            n_envs=args.n_parallel_envs,
            gpus=gpus,
            model_version=args.model_version,
            model_path=str(model_config["model_path"]),
            split=args.split,
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
                    client_python,
                    args.n_eps_per_task,
                    args.n_parallel_envs,
                    args.n_action_steps,
                    args.model_version,
                    model_config,
                    args.split,
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
        print_summary(results_file.results, tasks_by_group, wall_time)
        path = results_file.finalize()
        print(f"\nFull results: {path}")
    finally:
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()


if __name__ == "__main__":
    main()
