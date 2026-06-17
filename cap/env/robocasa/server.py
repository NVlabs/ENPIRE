"""Thin Portal RPC server wrapping RoboCasaEnv.

Exposes the same RPC interface as CapServer so tools like
``viser_curobo_planner`` can connect without a full CapServer.

Usage::

    ROBOCASA_SEED=42 CAP_CUROBO_PORT=8611 \\
        uv run python -m cap.env.robocasa.server \\
        --env robocasa:PickPlaceSinkToCounter --port 18600

    # Then in another terminal:
    CAP_PORT=18600 CAP_CUROBO_PORT=8611 \\
        uv run python tools/viser_curobo_planner.py
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import portal

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class EnvServer:
    """Portal RPC server backed by a RoboCasaEnv (no CapServer)."""

    def __init__(
        self,
        env: Any,
        port: int = 18600,
        namespace: dict | None = None,
        runtime_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._env = env
        self._namespace = namespace or {}
        self._runtime_cfg = dict(runtime_cfg or {})
        self._lock = threading.Lock()
        self._vision_state: dict[str, Any] = {
            "version": 0,
            "detections": [],
            "grasps": [],
            "hover_poses": [],
            "grasp_poses": [],
            "place_poses": [],
        }
        self._vision_lock = threading.Lock()
        self._script_thread: threading.Thread | None = None
        self._script_stop = threading.Event()
        self._wrap_vision_tools()

        self._server = portal.Server(port, logging=False, workers=4)
        self._server.bind("get_state", self._get_state)
        self._server.bind("get_collision_geoms", self._get_collision_geoms)
        self._server.bind("get_camera_image", self._get_camera_image)
        self._server.bind("get_camera_depth", self._get_camera_depth)
        self._server.bind("get_camera_intrinsics", self._get_camera_intrinsics)
        self._server.bind("get_camera_extrinsics", self._get_camera_extrinsics)
        self._server.bind("get_task_info", self._get_task_info)
        self._server.bind("get_oracle_targets", self._get_oracle_targets)
        self._server.bind("reset_env", self._reset_env)
        self._server.bind("reset_to_initial", self._reset_to_initial)
        self._server.bind("next_seed", self._next_seed)
        self._server.bind("open_gripper", self._open_gripper)
        self._server.bind("close_gripper", self._close_gripper)
        self._server.bind("run_saved_script", self._run_saved_script, workers=1)
        self._server.bind("forward_kinematics_batch", self._forward_kinematics_batch)
        self._server.bind("move_joint_keypoints", self._move_joint_keypoints)
        self._server.bind("command_base_action", self._command_base_action)
        self._server.bind("execute_base_trajectory", self._execute_base_trajectory)
        self._server.bind("get_floor_bounds", self._get_floor_bounds)
        self._server.bind("get_vision_state", self._get_vision_state)
        self._server.bind("clear_vision_state", self._clear_vision_state)
        self._server.bind("stop_script", self._stop_script)

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        arm_names = self._env._profile.arm_names if self._env._profile else ("right",)
        state: dict[str, Any] = {}
        for side in arm_names:
            obs = self._env.get_arm_observation(side)
            state[f"{side}_joint_pos"] = obs["joint_pos"]
            state[f"{side}_gripper_pos"] = obs["gripper_pos"]
            state[f"{side}_ee_pos"] = obs.get("ee_pos", np.zeros(3))
            state[f"{side}_ee_quat_xyzw"] = obs.get("ee_quat", np.array([0, 0, 0, 1.0]))
        bp = self._env.get_base_pose()
        if "base_pos" in bp:
            state["base_pos"] = bp["base_pos"]
        if "base_quat" in bp:
            state["base_quat_xyzw"] = bp["base_quat"]
        return state

    def _get_collision_geoms(self, max_dist: float = 1.2) -> dict:
        return self._env.get_collision_geoms(max_dist=max_dist)

    def _get_camera_image(self, camera: str) -> np.ndarray:
        return self._env.render_rgb(camera)

    def _get_camera_depth(self, camera: str) -> np.ndarray:
        return self._env.render_depth(camera)

    def _get_camera_intrinsics(self, camera: str) -> list[float]:
        return self._env.get_camera_intrinsics(camera)

    def _get_camera_extrinsics(self, camera: str) -> dict:
        return self._env.get_camera_extrinsics(camera)

    def _get_task_info(self) -> dict:
        return self._env.get_task_info()

    def _get_oracle_targets(self) -> dict:
        getter = self._namespace.get("get_oracle_targets")
        if callable(getter):
            return getter()
        if hasattr(self._env, "get_oracle_targets"):
            return self._env.get_oracle_targets()
        raise RuntimeError("get_oracle_targets is unavailable in env server")

    def _reset_env(self) -> dict:
        resetter = self._namespace.get("reset_env")
        if callable(resetter):
            return resetter()
        if hasattr(self._env, "reset_env"):
            return self._env.reset_env()
        raise RuntimeError("reset_env is unavailable in env server")

    def _reset_to_initial(self) -> dict:
        resetter = self._namespace.get("reset_env")
        if callable(resetter):
            return resetter()
        if hasattr(self._env, "reset_to_initial"):
            return self._env.reset_to_initial()
        if hasattr(self._env, "reset_env"):
            return self._env.reset_env()
        raise RuntimeError("reset_to_initial is unavailable in env server")

    def _recreate_runtime(self, seed: int) -> dict:
        from cap.env.setup import create_runtime

        if not self._runtime_cfg:
            raise RuntimeError("runtime recreation is unavailable in env server")

        runtime_cfg = dict(self._runtime_cfg)
        runtime_cfg["seed"] = int(seed)
        old_env = self._env

        with self._lock:
            new_env, new_namespace = create_runtime(**runtime_cfg)
            self._env = new_env
            self._namespace = new_namespace
            self._runtime_cfg = runtime_cfg

        try:
            old_env.close()
        except Exception:
            pass

        info = self._env.get_task_info()
        return {
            "ok": True,
            "seed": int(runtime_cfg["seed"]),
            "layout_id": info.get("layout_id"),
            "style_id": info.get("style_id"),
            "task": runtime_cfg.get("env_name"),
        }

    def _next_seed(self) -> dict:
        current_seed = self._runtime_cfg.get("seed")
        if current_seed is None:
            current_seed = getattr(self._env, "_seed", None)
        next_seed = int(0 if current_seed is None else current_seed) + 1
        return self._recreate_runtime(next_seed)

    def _open_gripper(self, side: str) -> dict:
        opener = self._namespace.get("open_gripper")
        if callable(opener):
            opener(side)
            return {"ok": True, "side": side}
        raise RuntimeError("open_gripper is unavailable in env server")

    def _close_gripper(self, side: str) -> dict:
        closer = self._namespace.get("close_gripper")
        if callable(closer):
            closer(side)
            return {"ok": True, "side": side}
        raise RuntimeError("close_gripper is unavailable in env server")

    def _run_saved_script(self, script_file: str) -> dict:
        self._script_stop.clear()
        self._script_thread = threading.current_thread()

        root = Path(_ROOT)
        script_path = Path(script_file).expanduser()
        if not script_path.is_absolute():
            script_path = root / script_path
        if not script_path.exists():
            return {
                "ok": False,
                "error": f"script not found: {script_file}",
            }

        stopped = False
        try:
            code = script_path.read_text(encoding="utf-8")
            stale_modules = [
                name
                for name in list(sys.modules.keys())
                if name == "skill_library.namespace"
                or name.startswith("cap.saved_scripts.robocasa_skill_library.")
            ]
            for name in stale_modules:
                sys.modules.pop(name, None)

            exec_namespace = dict(self._namespace)
            exec_namespace.setdefault("get_state", self._get_state)
            exec_namespace.setdefault("get_collision_geoms", self._get_collision_geoms)
            exec_namespace.setdefault("get_camera_image", self._get_camera_image)
            exec_namespace.setdefault("get_camera_depth", self._get_camera_depth)
            exec_namespace.setdefault("get_camera_intrinsics", self._get_camera_intrinsics)
            exec_namespace.setdefault("get_camera_extrinsics", self._get_camera_extrinsics)
            exec_namespace.setdefault("get_task_info", self._get_task_info)
            exec_namespace.setdefault("get_oracle_targets", self._get_oracle_targets)
            exec_namespace.setdefault("reset_env", self._reset_env)
            exec_namespace.setdefault("forward_kinematics_batch", self._forward_kinematics_batch)
            exec_namespace.setdefault("move_joint_keypoints", self._move_joint_keypoints)
            exec_namespace.setdefault("command_base_action", self._command_base_action)
            exec_namespace.setdefault("execute_base_trajectory", self._execute_base_trajectory)
            exec_namespace.setdefault("get_floor_bounds", self._get_floor_bounds)
            exec_namespace.setdefault("create_motion_planner", self._create_motion_planner)
            exec_namespace.setdefault("set_live_detections", self._set_live_detections)
            exec_namespace.setdefault("set_live_hover_poses", self._set_live_hover_poses)
            exec_namespace.setdefault("set_live_grasp_poses", self._set_live_grasp_poses)
            exec_namespace.setdefault("set_live_place_poses", self._set_live_place_poses)
            self._install_skill_namespace(exec_namespace)
            exec_namespace["__file__"] = str(script_path)
            exec_namespace["__name__"] = "__main__"
            exec_namespace["__builtins__"] = __builtins__
            exec(compile(code, str(script_path), "exec"), exec_namespace)  # noqa: S102
        except KeyboardInterrupt:
            stopped = True
            print(f"[env-server] Script stopped: {script_file}")
            return {"ok": False, "error": "Script stopped by user"}
        except Exception:
            error = traceback.format_exc()
            if self._script_stop.is_set():
                stopped = True
                print(f"[env-server] Script stopped: {script_file}")
                return {"ok": False, "error": "Script stopped by user"}
            print(error)
            return {
                "ok": False,
                "error": error,
            }
        finally:
            self._script_thread = None

        final_info = self._env.get_task_info()
        final_oracle = None
        getter = self._namespace.get("get_oracle_targets")
        if callable(getter):
            try:
                final_oracle = getter()
            except Exception:
                final_oracle = None
        return {
            "ok": True,
            "success": bool(final_info.get("success", False)),
            "reward": float(final_info.get("reward", 0.0)),
            "task_info": final_info,
            "fixture_state": None if final_oracle is None else final_oracle.get("fixture_state"),
        }

    def _install_skill_namespace(self, namespace: dict[str, Any]) -> None:
        import types

        pkg = types.ModuleType("skill_library")
        pkg.__path__ = [  # type: ignore[attr-defined]
            str(Path(_ROOT) / "cap" / "saved_scripts" / "robocasa_skill_library")
        ]
        pkg.__package__ = "skill_library"
        sys.modules["skill_library"] = pkg

        ns_mod = types.ModuleType("skill_library.namespace")
        names = []
        for key, value in namespace.items():
            if key.startswith("_") or not callable(value):
                continue
            setattr(ns_mod, key, value)
            names.append(key)
        ns_mod.__all__ = sorted(names)  # type: ignore[attr-defined]
        sys.modules["skill_library.namespace"] = ns_mod
        setattr(pkg, "namespace", ns_mod)

    def _forward_kinematics_batch(self, side: str, joint_positions: np.ndarray) -> dict:
        import mujoco
        from scipy.spatial.transform import Rotation as ScipyR

        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.ndim == 1:
            joint_positions = joint_positions.reshape(1, -1)
        n_configs = joint_positions.shape[0]
        dof = joint_positions.shape[1]

        empty = {
            "ee_positions": np.zeros((n_configs, 3)),
            "ee_quats_xyzw": np.tile([0, 0, 0, 1.0], (n_configs, 1)),
            "base_pos": np.zeros(3),
            "base_quat_xyzw": np.array([0, 0, 0, 1.0]),
        }

        sim = self._env._env.sim
        model = sim.model._model
        data = sim.data._data

        # Find EE site
        site_id = -1
        robots = getattr(self._env._env, "robots", None)
        if robots and len(robots) > 0:
            sid = getattr(robots[0], "eef_site_id", None)
            if sid is not None:
                site_id = (
                    int(sid)
                    if not isinstance(sid, dict)
                    else int(sid.get(side, sid.get("right", -1)))
                )
        if site_id < 0:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        if site_id < 0:
            return empty

        # Find joint qpos indices
        qpos_addrs = []
        for j in range(1, dof + 1):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{j}"
            )
            if jid < 0:
                return empty
            qpos_addrs.append(model.jnt_qposadr[jid])

        # Compute FK on a data copy (no mutation of live sim)
        fk_data = mujoco.MjData(model)
        fk_data.qpos[:] = data.qpos
        fk_data.qvel[:] = data.qvel

        ee_positions = np.zeros((n_configs, 3))
        ee_quats = np.zeros((n_configs, 4))
        for i in range(n_configs):
            for j, addr in enumerate(qpos_addrs):
                fk_data.qpos[addr] = joint_positions[i, j]
            mujoco.mj_forward(model, fk_data)
            ee_positions[i] = fk_data.site_xpos[site_id].copy()
            ee_quats[i] = ScipyR.from_matrix(
                fk_data.site_xmat[site_id].reshape(3, 3)
            ).as_quat()

        bp = self._env.get_base_pose()
        return {
            "ee_positions": ee_positions,
            "ee_quats_xyzw": ee_quats,
            "base_pos": np.asarray(bp.get("base_pos", np.zeros(3))),
            "base_quat_xyzw": np.asarray(bp.get("base_quat", [0, 0, 0, 1.0])),
        }

    def _command_base_action(self, v_fwd: float, v_side: float, omega: float) -> dict:
        with self._lock:
            self._env.command_base_action(v_fwd, v_side, omega)
        return {"ok": True}

    def _execute_base_trajectory(self, actions: list) -> dict:
        """Execute a full base trajectory in one RPC call.

        actions: list of [v_fwd, v_side, omega] per step.
        Returns: {ok, n_steps, base_pos, base_quat_xyzw, actual_trajectory}
        """
        from scipy.spatial.transform import Rotation as R

        actions_np = np.asarray(actions, dtype=np.float64)
        n = len(actions_np)

        # Record start pose
        bp0 = self._env.get_base_pose()
        start_pos = bp0.get("base_pos", np.zeros(3)).copy()
        start_yaw = float(
            R.from_quat(bp0.get("base_quat", [0, 0, 0, 1])).as_euler("xyz")[2]
        )

        print(f"[exec-base] Starting execution of {n} steps...")
        print(
            f"[exec-base] action_dim={self._env._action_dim}, splits={dict(self._env._action_splits)}"
        )
        print(f"[exec-base] first action={actions_np[0].tolist()}")
        # 6 sim steps per MPPI step — calibrated to match dt_fwd=0.06 at v=1.0
        # (burst gives ~60mm in first 6 steps at v=1.0)
        SIM_PER_MPPI = 6
        with self._lock:
            for i in range(n):
                self._env.command_base_action(
                    float(actions_np[i, 0]),
                    float(actions_np[i, 1]),
                    float(actions_np[i, 2]),
                    n_steps=SIM_PER_MPPI,
                )
                if i % 10 == 0:
                    print(f"[exec-base]   step {i}/{n} ({i * SIM_PER_MPPI} sim steps)")

        bp = self._env.get_base_pose()
        end_pos = bp.get("base_pos", np.zeros(3))
        end_yaw = float(
            R.from_quat(bp.get("base_quat", [0, 0, 0, 1])).as_euler("xyz")[2]
        )
        dist_moved = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))

        print(
            f"[exec-base] {n} steps | "
            f"start=({start_pos[0]:.2f},{start_pos[1]:.2f},yaw={start_yaw:.2f}) | "
            f"end=({end_pos[0]:.2f},{end_pos[1]:.2f},yaw={end_yaw:.2f}) | "
            f"moved={dist_moved:.3f}m | "
            f"action_range=[{actions_np.min():.2f}, {actions_np.max():.2f}]"
        )

        return {
            "ok": True,
            "n_steps": n,
            "base_pos": end_pos.tolist(),
            "base_quat_xyzw": bp.get("base_quat", np.array([0, 0, 0, 1.0])).tolist(),
            "dist_moved": dist_moved,
        }

    def _get_floor_bounds(self) -> dict:
        floor_min, floor_max = self._env.get_floor_bounds()
        return {"min": floor_min.tolist(), "max": floor_max.tolist()}

    def _move_joint_keypoints(
        self,
        side: str,
        timestamps: list[float],
        joint_positions: list,
        gripper_positions: list | None = None,
    ) -> dict:
        move_joint_keypoints = self._namespace.get("move_joint_keypoints")
        if callable(move_joint_keypoints):
            return move_joint_keypoints(
                side,
                timestamps,
                joint_positions,
                gripper_positions,
            )

        raise RuntimeError("move_joint_keypoints tool is unavailable in env server namespace")

    # ------------------------------------------------------------------
    # Script control
    # ------------------------------------------------------------------

    def _stop_script(self) -> dict:
        import ctypes

        t = self._script_thread
        if t is None or not t.is_alive():
            return {"ok": False, "error": "No script running"}
        self._script_stop.set()
        tid = t.ident
        if tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt)
            )
        print("[env-server] Stop signal sent to running script")
        return {"ok": True}

    # ------------------------------------------------------------------
    # Vision state (live debug feed for Viser)
    # ------------------------------------------------------------------

    def _wrap_vision_tools(self) -> None:
        """Wrap namespace vision tools to capture results into _vision_state."""
        import base64 as _b64
        vision_state = self._vision_state
        vision_lock = self._vision_lock

        # Patch the base skill library function to stash the overlay jpeg
        # so we can grab it after sample_grasp_pose_anygrasp returns.
        _overlay_stash: dict[str, bytes | None] = {"jpeg": None}
        try:
            import cap.env.base.skill_library as _base_sl
            _orig_base_grasp = _base_sl.sample_grasp_anygrasp

            def _patched_base_grasp(*args, **kwargs):
                candidates, viz = _orig_base_grasp(*args, **kwargs)
                _overlay_stash["jpeg"] = viz.get("overlay_jpeg")
                return candidates, viz

            _base_sl.sample_grasp_anygrasp = _patched_base_grasp
        except Exception:
            pass

        orig_detect = self._namespace.get("detect_objects_oneshot")
        if callable(orig_detect):
            def _wrapped_detect(*args, **kwargs):
                result = orig_detect(*args, **kwargs)
                detections = []
                for _q, dets in result.items():
                    for det in dets:
                        detections.append({
                            "label": det.label,
                            "pos": [float(x) for x in det.position_3d],
                            "score": float(det.score),
                            "box_2d": list(det.box_2d) if det.box_2d else [],
                        })
                with vision_lock:
                    vision_state["version"] += 1
                    vision_state["detections"] = detections
                return result
            self._namespace["detect_objects_oneshot"] = _wrapped_detect

        orig_grasp = self._namespace.get("sample_grasp_pose_anygrasp")
        if callable(orig_grasp):
            def _wrapped_grasp(*args, **kwargs):
                _overlay_stash["jpeg"] = None
                result = orig_grasp(*args, **kwargs)
                grasps = []
                for g in result:
                    grasps.append({
                        "pos": [float(x) for x in g.position],
                        "rpy": [float(x) for x in g.rpy],
                        "score": float(g.score),
                        "width": float(getattr(g, "width", 0.08)),
                    })
                overlay_b64 = None
                if _overlay_stash["jpeg"]:
                    overlay_b64 = _b64.b64encode(_overlay_stash["jpeg"]).decode()
                with vision_lock:
                    vision_state["version"] += 1
                    vision_state["grasps"] = grasps
                    vision_state["grasp_overlay_b64"] = overlay_b64
                return result
            self._namespace["sample_grasp_pose_anygrasp"] = _wrapped_grasp

    def _get_vision_state(self) -> dict:
        with self._vision_lock:
            return dict(self._vision_state)

    def _clear_vision_state(self) -> dict:
        with self._vision_lock:
            self._vision_state["version"] += 1
            self._vision_state["detections"] = []
            self._vision_state["grasps"] = []
            self._vision_state["hover_poses"] = []
            self._vision_state["grasp_poses"] = []
            self._vision_state["place_poses"] = []
            self._vision_state["grasp_overlay_b64"] = None
        return {"ok": True}

    def _set_live_detections(self, detections: list[dict]) -> dict:
        normalized = []
        for det in detections or []:
            normalized.append(
                {
                    "label": str(det.get("label", det.get("query", "object"))),
                    "pos": [float(x) for x in det.get("pos", [])],
                    "score": float(det.get("score", 0.0)),
                    "box_2d": list(det.get("box_2d", [])),
                }
            )
        with self._vision_lock:
            self._vision_state["version"] += 1
            self._vision_state["detections"] = normalized
        return {"ok": True, "n": len(normalized)}

    def _normalize_live_poses(self, poses: list[dict]) -> list[dict]:
        normalized = []
        for pose in poses or []:
            normalized.append(
                {
                    "pos": [float(x) for x in pose.get("pos", [])],
                    "rpy": [float(x) for x in pose.get("rpy", [])],
                    "score": float(pose.get("score", 0.0)),
                    "rank": int(pose.get("rank", 0)),
                    "source_index": int(pose.get("source_index", -1)),
                }
            )
        return normalized

    def _set_live_hover_poses(self, hover_poses: list[dict]) -> dict:
        normalized = self._normalize_live_poses(hover_poses)
        with self._vision_lock:
            self._vision_state["version"] += 1
            self._vision_state["hover_poses"] = normalized
        return {"ok": True, "n": len(normalized)}

    def _set_live_grasp_poses(self, grasp_poses: list[dict]) -> dict:
        normalized = self._normalize_live_poses(grasp_poses)
        with self._vision_lock:
            self._vision_state["version"] += 1
            self._vision_state["grasp_poses"] = normalized
        return {"ok": True, "n": len(normalized)}

    def _set_live_place_poses(self, place_poses: list[dict]) -> dict:
        normalized = self._normalize_live_poses(place_poses)
        with self._vision_lock:
            self._vision_state["version"] += 1
            self._vision_state["place_poses"] = normalized
        return {"ok": True, "n": len(normalized)}

    # ------------------------------------------------------------------
    # Motion planner (lazy, shared across script runs)
    # ------------------------------------------------------------------

    _planner_instance = None

    def _create_motion_planner(
        self, solver_speed="slow", position_threshold=0.01, rotation_threshold=0.1
    ):
        if self._planner_instance is not None:
            return self._planner_instance
        import os

        from experimental.portal_motion_planner import PortalMotionPlanner

        curobo_host = self._runtime_cfg.get("curobo_host", "127.0.0.1")
        curobo_port = self._runtime_cfg.get("curobo_port", 0)
        start_server = curobo_port == 0
        port = curobo_port if curobo_port != 0 else None
        robot_type = os.environ.get("CAP_ROBOT_TYPE", "panda").strip().lower()
        self._planner_instance = PortalMotionPlanner(
            backend="curobo",
            solver_speed=solver_speed,
            host=curobo_host,
            port=port,
            start_server=start_server,
            robot_type=robot_type,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
        )
        return self._planner_instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        pass


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="RoboCasa env Portal RPC server")
    parser.add_argument(
        "--env", default="robocasa:PickPlaceSinkToCounter", help="Environment spec"
    )
    parser.add_argument("--port", type=int, default=18600, help="Portal RPC port")
    parser.add_argument("--viewer", action="store_true", help="Launch viewer")
    args = parser.parse_args()

    from cap.env.setup import create_runtime

    print(f"[env-server] Creating {args.env} ...")
    seed = int(os.environ["ROBOCASA_SEED"]) if os.environ.get("ROBOCASA_SEED") else None
    layout_id = (
        int(os.environ["ROBOCASA_LAYOUT_ID"])
        if os.environ.get("ROBOCASA_LAYOUT_ID")
        else None
    )
    style_id = (
        int(os.environ["ROBOCASA_STYLE_ID"])
        if os.environ.get("ROBOCASA_STYLE_ID")
        else None
    )
    curobo_host = os.environ.get("CAP_CUROBO_HOST", "127.0.0.1")
    curobo_port = int(os.environ.get("CAP_CUROBO_PORT", "0"))
    runtime_cfg = {
        "env_name": args.env,
        "viewer": args.viewer,
        "seed": seed,
        "layout_id": layout_id,
        "style_id": style_id,
        "curobo_host": curobo_host,
        "curobo_port": curobo_port,
    }
    env, namespace = create_runtime(**runtime_cfg)

    info = env.get_task_info()
    print(
        f"[env-server] Scene: obj={info.get('obj_name')}, container={info.get('container_name')}"
    )

    server = EnvServer(
        env,
        port=args.port,
        namespace=namespace,
        runtime_cfg=runtime_cfg,
    )
    server.start()
    print(f"[env-server] Listening on port {args.port}")
    print("[env-server] Press Ctrl+C to stop")

    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[env-server] Stopped")


if __name__ == "__main__":
    main()
