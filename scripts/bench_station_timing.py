from __future__ import annotations

import json
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import portal
import tyro

from robot.constants import (
    LEFT_FOLLOWER_PORT,
    LEFT_LEADER_PORT,
    RIGHT_FOLLOWER_PORT,
    RIGHT_LEADER_PORT,
)
from robot.station_profiles import active_station_cameras, resolve_station_key

try:
    from robot.yam.yam_real_env import create_camera as _create_station_camera
    _station_camera_import_error: Exception | None = None
except Exception as exc:
    _create_station_camera = None
    _station_camera_import_error = exc

try:
    from robot.camera_factory import create_camera as _create_named_camera
except Exception:
    _create_named_camera = None


def _open_camera(cfg: Any) -> Any:
    if _create_station_camera is not None:
        return _create_station_camera(cfg)
    if _create_named_camera is not None:
        return _create_named_camera(
            cfg.name,
            resolution=(640, 480),
            fps=60,
            enable_depth=False,
        )
    raise RuntimeError(
        "No camera factory is available for bench_station_timing "
        f"(yam_real_env import error: {_station_camera_import_error!r})"
    )


def _is_retryable_camera_error(exc: Exception) -> bool:
    text = str(exc)
    return "Frame didn't arrive within" in text or "Timed out waiting for first frame" in text


def _camera_get_image(camera: Any, *, attempts: int = 3, retry_sleep_s: float = 0.05) -> Any:
    get_image = getattr(camera, "get_image", None)
    if callable(get_image):
        last_error: Exception | None = None
        for attempt in range(max(int(attempts), 1)):
            try:
                return get_image()
            except Exception as exc:
                last_error = exc
                if not _is_retryable_camera_error(exc) or attempt + 1 >= max(int(attempts), 1):
                    raise
                time.sleep(retry_sleep_s)
        raise RuntimeError("camera get_image retry loop exhausted") from last_error

    read = getattr(camera, "read", None)
    if callable(read):
        last_error: Exception | None = None
        for attempt in range(max(int(attempts), 1)):
            try:
                data = read()
                images = getattr(data, "images", None)
                if isinstance(images, dict) and "rgb" in images:
                    return images["rgb"]
                raise RuntimeError(f"Camera read() returned unsupported payload type: {type(data)!r}")
            except Exception as exc:
                last_error = exc
                if not _is_retryable_camera_error(exc) or attempt + 1 >= max(int(attempts), 1):
                    raise
                time.sleep(retry_sleep_s)
        raise RuntimeError("camera read retry loop exhausted") from last_error

    raise RuntimeError(
        f"Camera object {type(camera)!r} exposes neither get_image() nor read()"
    )


def _camera_close(camera: Any) -> None:
    close = getattr(camera, "close", None)
    if callable(close):
        close()
        return

    stop = getattr(camera, "stop", None)
    if callable(stop):
        stop()


def _prime_camera(camera: Any, name: str) -> None:
    _camera_get_image(camera, attempts=5, retry_sleep_s=0.1)
    print(f"[bench_station_timing] Camera '{name}' primed", flush=True)


@dataclass
class Args:
    host: str = "localhost"
    duration_s: float = 60.0
    warmup_s: float = 3.0
    target_hz: float = 30.0
    include_leaders: bool = True
    include_followers: bool = True
    include_cameras: bool = True
    output_dir: str = "/tmp/station_timing_bench"
    label: str | None = None
    save_raw: bool = False
    startup_timeout_s: float = 10.0
    progress_every_s: float = 5.0


def _timed_portal_call(client: portal.Client, method_name: str) -> tuple[Any, dict[str, float]]:
    submit_t0 = time.perf_counter()
    request = getattr(client, method_name)()
    submit_ms = (time.perf_counter() - submit_t0) * 1000.0
    wait_t0 = time.perf_counter()
    result = request.result()
    wait_ms = (time.perf_counter() - wait_t0) * 1000.0
    return result, {
        "submit_ms": submit_ms,
        "wait_ms": wait_ms,
        "total_ms": submit_ms + wait_ms,
    }


def _wait_for_port(host: str, port: int, label: str, timeout_s: float) -> None:
    deadline = time.time() + max(timeout_s, 0.0)
    last_error = "unknown"
    print(f"[bench_station_timing] Waiting for {label} at {host}:{port}", flush=True)
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                print(f"[bench_station_timing] {label} is reachable", flush=True)
                return
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(
        f"{label} at {host}:{port} was not reachable within {timeout_s:.1f}s "
        f"(last error: {last_error}). Start the robot servers first."
    )


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "std_ms": round(float(arr.std()), 3),
        "min_ms": round(float(arr.min()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(arr.max()), 3),
        "gt2ms": int((arr > 2.0).sum()),
        "gt5ms": int((arr > 5.0).sum()),
        "gt10ms": int((arr > 10.0).sum()),
        "gt20ms": int((arr > 20.0).sum()),
        "n": int(arr.size),
    }


def _top_steps(steps: list[dict[str, Any]], key: str, topn: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(steps, key=lambda step: float(step.get(key, 0.0)), reverse=True)
    result: list[dict[str, Any]] = []
    for step in ranked[:topn]:
        result.append({
            "step": int(step["step"]),
            "wall_time": step["wall_time_iso"],
            key: round(float(step.get(key, 0.0)), 3),
            "loop_total_ms": round(float(step.get("loop_total_ms", 0.0)), 3),
            "policy_phase_ms": round(float(step.get("policy_phase_ms", 0.0)), 3),
            "env_obs_phase_ms": round(float(step.get("env_obs_phase_ms", 0.0)), 3),
            "pacing_ms": round(float(step.get("pacing_ms", 0.0)), 3),
            "pacing_overshoot_ms": round(float(step.get("pacing_overshoot_ms", 0.0)), 3),
            "obs_component_span_ms": round(float(step.get("obs_component_span_ms", 0.0)), 3),
            "left_leader_wait_ms": round(float(step.get("left_leader_wait_ms", 0.0)), 3),
            "right_leader_wait_ms": round(float(step.get("right_leader_wait_ms", 0.0)), 3),
            "left_follower_wait_ms": round(float(step.get("left_follower_wait_ms", 0.0)), 3),
            "right_follower_wait_ms": round(float(step.get("right_follower_wait_ms", 0.0)), 3),
            "top_camera_get_image_ms": round(float(step.get("top_camera_get_image_ms", 0.0)), 3),
            "left_camera_get_image_ms": round(float(step.get("left_camera_get_image_ms", 0.0)), 3),
            "right_camera_get_image_ms": round(float(step.get("right_camera_get_image_ms", 0.0)), 3),
        })
    return result


def _print_high_signal_summary(summary: dict[str, Any]) -> None:
    print("\n[bench_station_timing] Summary")
    print(json.dumps({
        "host": summary["meta"]["hostname"],
        "station_key": summary["meta"]["station_key"],
        "target_hz": summary["meta"]["target_hz"],
        "iterations": summary["meta"]["iterations_recorded"],
        "loop_total_ms": summary["metrics"]["loop_total_ms"],
        "policy_phase_ms": summary["metrics"]["policy_phase_ms"],
        "env_obs_phase_ms": summary["metrics"]["env_obs_phase_ms"],
        "pacing_overshoot_ms": summary["metrics"]["pacing_overshoot_ms"],
        "obs_component_span_ms": summary["metrics"]["obs_component_span_ms"],
        "left_leader_wait_ms": summary["metrics"]["left_leader_wait_ms"],
        "right_leader_wait_ms": summary["metrics"]["right_leader_wait_ms"],
        "left_follower_wait_ms": summary["metrics"]["left_follower_wait_ms"],
        "right_follower_wait_ms": summary["metrics"]["right_follower_wait_ms"],
    }, indent=2))


def main(args: Args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_label = args.label or socket.gethostname().lower()
    summary_path = output_dir / f"station_timing_summary_{run_label}_{run_stamp}.json"
    raw_path = output_dir / f"station_timing_raw_{run_label}_{run_stamp}.jsonl"

    leader_clients: dict[str, portal.Client] = {}
    follower_clients: dict[str, portal.Client] = {}
    cameras: dict[str, Any] = {}

    if args.include_leaders:
        _wait_for_port(args.host, LEFT_LEADER_PORT, "left leader RPC", args.startup_timeout_s)
        _wait_for_port(args.host, RIGHT_LEADER_PORT, "right leader RPC", args.startup_timeout_s)
        leader_clients = {
            "left": portal.Client(f"{args.host}:{LEFT_LEADER_PORT}"),
            "right": portal.Client(f"{args.host}:{RIGHT_LEADER_PORT}"),
        }
    if args.include_followers:
        _wait_for_port(args.host, LEFT_FOLLOWER_PORT, "left follower RPC", args.startup_timeout_s)
        _wait_for_port(args.host, RIGHT_FOLLOWER_PORT, "right follower RPC", args.startup_timeout_s)
        follower_clients = {
            "left": portal.Client(f"{args.host}:{LEFT_FOLLOWER_PORT}"),
            "right": portal.Client(f"{args.host}:{RIGHT_FOLLOWER_PORT}"),
        }
    if args.include_cameras:
        if _create_station_camera is None and _create_named_camera is not None:
            print(
                "[bench_station_timing] Falling back to robot.camera_factory.create_camera "
                f"because robot.yam.yam_real_env could not be imported: "
                f"{_station_camera_import_error}",
                flush=True,
            )
        for cfg in active_station_cameras().cameras:
            print(f"[bench_station_timing] Opening camera '{cfg.name}' ({cfg.type})", flush=True)
            cameras[cfg.name] = _open_camera(cfg)
            _prime_camera(cameras[cfg.name], cfg.name)
            print(f"[bench_station_timing] Camera '{cfg.name}' ready", flush=True)

    raw_file = raw_path.open("w", encoding="utf-8") if args.save_raw else None
    steps: list[dict[str, Any]] = []
    warmup_deadline = time.time() + max(args.warmup_s, 0.0)
    control_period = 1.0 / max(args.target_hz, 0.1)
    last_step_time = time.time()
    step_idx = 0
    recorded_steps = 0
    last_progress_t = time.time()
    warmup_announced = False

    print(
        f"[bench_station_timing] Warmup {args.warmup_s:.1f}s, then benchmark "
        f"{args.duration_s:.1f}s at {args.target_hz:.1f} Hz",
        flush=True,
    )

    try:
        end_time = warmup_deadline + max(args.duration_s, 0.0)
        while time.time() < end_time:
            step_idx += 1
            in_warmup = time.time() < warmup_deadline
            if not in_warmup and not warmup_announced:
                print("[bench_station_timing] Warmup complete, recording samples", flush=True)
                warmup_announced = True
            loop_t0 = time.perf_counter()
            component_ts: dict[str, float] = {}
            step: dict[str, Any] = {
                "step": step_idx,
                "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
            }

            policy_t0 = time.perf_counter()
            for side, client in leader_clients.items():
                _, metrics = _timed_portal_call(client, "get_info")
                step[f"{side}_leader_submit_ms"] = metrics["submit_ms"]
                step[f"{side}_leader_wait_ms"] = metrics["wait_ms"]
                step[f"{side}_leader_total_ms"] = metrics["total_ms"]
                component_ts[f"{side}_leader"] = time.time()
            step["policy_phase_ms"] = (time.perf_counter() - policy_t0) * 1000.0

            env_obs_t0 = time.perf_counter()
            for side, client in follower_clients.items():
                _, metrics = _timed_portal_call(client, "get_observations")
                step[f"{side}_follower_submit_ms"] = metrics["submit_ms"]
                step[f"{side}_follower_wait_ms"] = metrics["wait_ms"]
                step[f"{side}_follower_total_ms"] = metrics["total_ms"]
                component_ts[f"{side}_follower"] = time.time()
            for name, cam in cameras.items():
                cam_t0 = time.perf_counter()
                _ = _camera_get_image(cam)
                step[f"{name}_camera_get_image_ms"] = (time.perf_counter() - cam_t0) * 1000.0
                component_ts[f"{name}_camera"] = time.time()
            step["env_obs_phase_ms"] = (time.perf_counter() - env_obs_t0) * 1000.0

            if component_ts:
                ts_values = list(component_ts.values())
                step["obs_component_span_ms"] = (max(ts_values) - min(ts_values)) * 1000.0
            else:
                step["obs_component_span_ms"] = 0.0

            pacing_t0 = time.perf_counter()
            sleep_end_time = last_step_time + control_period
            now = time.time()
            step["sleep_target_ms"] = max(0.0, (sleep_end_time - now) * 1000.0)
            step["overrun_ms"] = max(0.0, (now - sleep_end_time) * 1000.0)
            while time.time() < sleep_end_time:
                time.sleep(0.0001)
            wake_time = time.time()
            step["pacing_ms"] = (time.perf_counter() - pacing_t0) * 1000.0
            step["pacing_overshoot_ms"] = max(0.0, (wake_time - sleep_end_time) * 1000.0)
            last_step_time = wake_time

            step["loop_total_ms"] = (time.perf_counter() - loop_t0) * 1000.0

            if not in_warmup:
                recorded_steps += 1
                steps.append(step)
                if raw_file is not None:
                    raw_file.write(json.dumps(step, separators=(",", ":")) + "\n")
                if args.progress_every_s > 0 and time.time() - last_progress_t >= args.progress_every_s:
                    elapsed_s = min(max(time.time() - warmup_deadline, 0.0), args.duration_s)
                    print(
                        f"[bench_station_timing] {elapsed_s:.1f}s / {args.duration_s:.1f}s "
                        f"| samples={recorded_steps} "
                        f"| loop={step['loop_total_ms']:.2f}ms "
                        f"| env_obs={step['env_obs_phase_ms']:.2f}ms "
                        f"| pace_overshoot={step['pacing_overshoot_ms']:.2f}ms",
                        flush=True,
                    )
                    last_progress_t = time.time()
    finally:
        if raw_file is not None:
            raw_file.close()
        for cam in cameras.values():
            try:
                _camera_close(cam)
            except Exception:
                pass

    metrics = {
        "loop_total_ms": _summary([step["loop_total_ms"] for step in steps]),
        "policy_phase_ms": _summary([step["policy_phase_ms"] for step in steps]),
        "env_obs_phase_ms": _summary([step["env_obs_phase_ms"] for step in steps]),
        "pacing_ms": _summary([step["pacing_ms"] for step in steps]),
        "pacing_overshoot_ms": _summary([step["pacing_overshoot_ms"] for step in steps]),
        "sleep_target_ms": _summary([step["sleep_target_ms"] for step in steps]),
        "overrun_ms": _summary([step["overrun_ms"] for step in steps]),
        "obs_component_span_ms": _summary([step["obs_component_span_ms"] for step in steps]),
        "left_leader_wait_ms": _summary([step.get("left_leader_wait_ms", 0.0) for step in steps if "left_leader_wait_ms" in step]),
        "right_leader_wait_ms": _summary([step.get("right_leader_wait_ms", 0.0) for step in steps if "right_leader_wait_ms" in step]),
        "left_follower_wait_ms": _summary([step.get("left_follower_wait_ms", 0.0) for step in steps if "left_follower_wait_ms" in step]),
        "right_follower_wait_ms": _summary([step.get("right_follower_wait_ms", 0.0) for step in steps if "right_follower_wait_ms" in step]),
    }

    for name in cameras:
        metrics[f"{name}_camera_get_image_ms"] = _summary(
            [step.get(f"{name}_camera_get_image_ms", 0.0) for step in steps if f"{name}_camera_get_image_ms" in step]
        )

    summary = {
        "meta": {
            "label": run_label,
            "hostname": socket.gethostname(),
            "station_key": resolve_station_key(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "target_hz": args.target_hz,
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "include_leaders": args.include_leaders,
            "include_followers": args.include_followers,
            "include_cameras": args.include_cameras,
            "camera_profile": [asdict(cfg) for cfg in active_station_cameras().cameras] if args.include_cameras else [],
            "iterations_recorded": recorded_steps,
            "summary_path": str(summary_path),
            "raw_path": str(raw_path) if args.save_raw else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": metrics,
        "top_outliers": {
            "loop_total_ms": _top_steps(steps, "loop_total_ms"),
            "policy_phase_ms": _top_steps(steps, "policy_phase_ms"),
            "env_obs_phase_ms": _top_steps(steps, "env_obs_phase_ms"),
            "pacing_overshoot_ms": _top_steps(steps, "pacing_overshoot_ms"),
            "obs_component_span_ms": _top_steps(steps, "obs_component_span_ms"),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _print_high_signal_summary(summary)
    print(f"[bench_station_timing] Wrote summary to {summary_path}")
    if args.save_raw:
        print(f"[bench_station_timing] Wrote raw steps to {raw_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
