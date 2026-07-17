"""Build tool namespace for RoboCasa agent runs — no CapServer needed.

Provides the same callable signatures and return types as
``cap.agent.tools.direct.make_direct_callables`` so agent-generated code
works identically whether it runs through CapServer or directly against the
RoboCasa env.

Motion uses cuRobo (served on a remote cloud GPU) for collision-free
trajectory planning, executed via the env's joint_position controller.

Reset tears down and recreates the env from scratch (same constructor args)
so the agent gets an identical scene on every retry.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

import numpy as np

from enpire.env.forge.cap.env.robocasa.env import RoboCasaEnv

logger = logging.getLogger(__name__)

# cuRobo planner defaults — override via env vars
_CUROBO_HOST = os.environ.get("CAP_CUROBO_HOST", "127.0.0.1")
_CUROBO_PORT = int(os.environ.get("CAP_CUROBO_PORT", "0"))

# Optional pyroki fallback for freespace_move when cuRobo isn't available
# (e.g. dev machines without a CUDA 12.8 toolkit). Pyroki does linear SE(3) +
# IK at each waypoint via the HTTP server at scripts/launch_pyroki_server.py;
# no collision avoidance, so trajectories may path through obstacles. Set
# CAP_PLANNER_BACKEND=pyroki to enable.
_PLANNER_BACKEND = os.environ.get("CAP_PLANNER_BACKEND", "curobo").strip().lower()
_PYROKI_HOST = os.environ.get("CAP_PYROKI_HOST", "127.0.0.1")
_PYROKI_PORT = int(os.environ.get("CAP_PYROKI_PORT", "9600"))

# MPPI navigation planner defaults — override via env vars
_MPPI_HOST = os.environ.get("CAP_MPPI_HOST", "127.0.0.1")
_MPPI_PORT = int(os.environ.get("CAP_MPPI_PORT", "0"))

_GEOMETRIC_MAX_JOINT_STEP = 0.03
_PLANNING_SPEED_MIN = 0.05
_PLANNING_SPEED_MAX = 3.0
_DEFAULT_PLANNING_SPEED = 1.5

# OSC-tracking base params. Each OSC tick is capped at a small EE-space
# step so that the commanded delta stays well inside OSC's output bound
# (5 cm pos, 28° rot), avoiding bang-bang acceleration that would shake
# grasped objects free. At 20 Hz control, 0.02 m/tick → 40 cm/s peak EE
# velocity — gentle enough for contact-rich manipulation.
#
# These are *base* values scaled by the per-call ``speed`` multiplier. The
# bounds below stop ``speed`` from producing unsafe saturation even if the
# caller passes a large value.
_OSC_MAX_POS_STEP_M = 0.02  # 2 cm per tick at speed=1
_OSC_MAX_ROT_STEP_RAD = 0.10  # ~6° per tick at speed=1
_OSC_POS_STEP_CEILING_M = 0.04  # never exceed 4 cm/tick (80% of OSC bound)
_OSC_ROT_STEP_CEILING_RAD = 0.25  # never exceed ~14°/tick

# Waypoint downsampling — cuRobo returns ~60 joint waypoints for a typical
# reach; OSC tracks them fine at ~20. The final waypoint is always kept
# regardless of stride.
_OSC_MAX_WAYPOINTS = 24

# Final-settle ticks — budget for OSC to close residual error at the user
# target. Larger than it needs to be on the fast path (exits early when
# within tol) so rare cases with long residual can still converge.
_OSC_FINAL_SETTLE_TICKS = 8
_OSC_FINAL_POS_TOL_M = 0.005  # 5 mm
_OSC_FINAL_ROT_TOL_RAD = 0.05  # ~3°

# Lazy-loaded pyroki robot for in-process FK. One instance per worker process;
# JAX compilation in the first forward_kinematics call takes ~1 s then every
# subsequent call is sub-ms.
_PYROKI_ROBOT = None
_PYROKI_EE_LINK_INDEX: dict[tuple[int, str], int] = {}

# Auto-calibrated rigid transform from pyroki's panda_hand link to robosuite's
# eef site (``robot0_eef_pos``/``robot0_eef_quat``). Measured on the first use
# by comparing FK of the current qpos against the live robosuite obs. This is
# joint-independent (the two bodies are rigidly connected via the hand+gripper
# XML) so a single measurement is reused for all subsequent FK queries.
# Stored as (pos_in_hand_frame, rot_in_hand_frame) — apply with
#     eef_pos_world = hand_pos_world + hand_rot_world @ _EE_POS_OFFSET_HAND
#     eef_rot_world = hand_rot_world * _EE_ROT_OFFSET_HAND
_EE_POS_OFFSET_HAND: np.ndarray | None = None  # (3,)
_EE_ROT_OFFSET_HAND = None  # scipy Rotation


def _get_pyroki_robot():
    """Load pyroki's Panda robot exactly once per process."""
    global _PYROKI_ROBOT
    if _PYROKI_ROBOT is None:
        import pyroki as pk  # type: ignore
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        urdf = load_robot_description("panda_description")
        _PYROKI_ROBOT = pk.Robot.from_urdf(urdf)
        logger.info("Loaded pyroki panda URDF for in-process FK (ee=panda_hand)")
    return _PYROKI_ROBOT


def _fk_hand_pose_base(
    cfg_8dof: np.ndarray, ee_link: str = "panda_hand"
) -> tuple[np.ndarray, np.ndarray]:
    """Pyroki FK → (pos_xyz, quat_xyzw) of the ``ee_link`` in panda_link0 frame."""
    robot = _get_pyroki_robot()
    key = (id(robot), ee_link)
    if key not in _PYROKI_EE_LINK_INDEX:
        _PYROKI_EE_LINK_INDEX[key] = list(robot.links.names).index(ee_link)
    link_idx = _PYROKI_EE_LINK_INDEX[key]

    import jax.numpy as jnp

    cfg = jnp.asarray(cfg_8dof, dtype=jnp.float32)
    poses = robot.forward_kinematics(cfg)  # (link_count, 7) as wxyz_xyz
    ee = np.asarray(poses[link_idx])
    quat_wxyz = ee[:4]
    pos = ee[4:]
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    return pos.astype(np.float64), quat_xyzw


def _hand_pose_world(cfg_8dof: np.ndarray, env):
    """FK of panda_hand in the WORLD frame (accounts for robot base pose)."""
    from scipy.spatial.transform import Rotation as R

    hand_pos_base, hand_quat_xyzw_base = _fk_hand_pose_base(cfg_8dof)
    bp = env.get_base_pose()
    base_pos_world = np.asarray(bp.get("base_pos", np.zeros(3)), dtype=np.float64)
    base_quat_world_xyzw = np.asarray(
        bp.get("base_quat", [0, 0, 0, 1]), dtype=np.float64
    )
    R_base = R.from_quat(base_quat_world_xyzw)
    hand_rot_world = R_base * R.from_quat(hand_quat_xyzw_base)
    hand_pos_world = base_pos_world + R_base.apply(hand_pos_base)
    return hand_pos_world, hand_rot_world


def _calibrate_hand_to_eef(env, side: str) -> None:
    """Measure the rigid hand→eef transform from the current sim state.

    ``panda_hand`` in pyroki's URDF and ``robot0_eef_pos``/``robot0_eef_quat``
    in robosuite point at different spots on the gripper (hand body vs grip
    site) with possibly different axis conventions. The offset is purely
    geometric (rigid), so one measurement at any valid qpos captures it.
    """
    global _EE_POS_OFFSET_HAND, _EE_ROT_OFFSET_HAND
    from scipy.spatial.transform import Rotation as R

    obs = env.get_arm_observation(side)
    jp = np.asarray(obs.get("joint_pos"), dtype=np.float64)
    if jp is None or jp.size < 7:
        return
    gripper_frac = float(obs.get("gripper_pos", [1.0])[0])
    finger = float(np.clip(gripper_frac * 0.04, 0.0, 0.04))
    cfg = np.concatenate([jp[:7], [finger]])

    hand_pos_world, hand_rot_world = _hand_pose_world(cfg, env)

    eef_pos_world = np.asarray(obs.get("ee_pos"), dtype=np.float64)
    eef_quat_world_xyzw = np.asarray(obs.get("ee_quat"), dtype=np.float64)
    eef_rot_world = R.from_quat(eef_quat_world_xyzw)

    # Rigid transform in hand's local frame.
    _EE_POS_OFFSET_HAND = hand_rot_world.inv().apply(eef_pos_world - hand_pos_world)
    _EE_ROT_OFFSET_HAND = hand_rot_world.inv() * eef_rot_world

    logger.info(
        "Calibrated hand→eef rigid offset: pos=%s (hand frame), "
        "rot=%s (hand frame, quat xyzw)",
        np.round(_EE_POS_OFFSET_HAND, 4).tolist(),
        np.round(_EE_ROT_OFFSET_HAND.as_quat(), 4).tolist(),
    )


def _fk_eef_pose_world(
    cfg_8dof: np.ndarray, env, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame EE pose matching robosuite's ``ee_pos``/``ee_quat``.

    Pipeline: pyroki FK (panda_link0 frame) → world frame (compose with robot
    base pose) → hand→eef rigid transform (auto-calibrated on first call).
    """
    if _EE_POS_OFFSET_HAND is None or _EE_ROT_OFFSET_HAND is None:
        _calibrate_hand_to_eef(env, side)

    hand_pos_world, hand_rot_world = _hand_pose_world(cfg_8dof, env)
    eef_pos_world = hand_pos_world + hand_rot_world.apply(_EE_POS_OFFSET_HAND)
    eef_rot_world = hand_rot_world * _EE_ROT_OFFSET_HAND
    return eef_pos_world, eef_rot_world.as_quat()


def _rotation_error_rad(
    quat_cur_xyzw: np.ndarray, quat_target_xyzw: np.ndarray
) -> float:
    """Magnitude of the axis-angle rotation from current to target."""
    from scipy.spatial.transform import Rotation as R

    R_cur = R.from_quat(quat_cur_xyzw)
    R_target = R.from_quat(quat_target_xyzw)
    rotvec = (R_target * R_cur.inv()).as_rotvec()
    return float(np.linalg.norm(rotvec))


def _lerp_quat_xyzw(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """Normalised LERP between two quaternions (xyzw), shortest-path.

    Good enough for small angular deltas — our per-tick angular step is
    bounded to ~6°, where LERP and SLERP differ by <0.001 in each component.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    if float(np.dot(q1, q2)) < 0.0:
        q2 = -q2
    q = (1.0 - alpha) * q1 + alpha * q2
    n = float(np.linalg.norm(q))
    return q / n if n > 0.0 else q1


def _quat_xyzw_to_rpy(q: list[float] | np.ndarray) -> list[float]:
    """Convert quaternion [x, y, z, w] to RPY [roll, pitch, yaw] in degrees."""
    from scipy.spatial.transform import Rotation as R

    try:
        r = R.from_quat(q)
        return r.as_euler("xyz", degrees=True).tolist()
    except Exception:
        return [0.0, 0.0, 0.0]


def _rpy_deg_to_quat_xyzw(rpy_deg: list[float]) -> np.ndarray:
    """Convert RPY [roll, pitch, yaw] in degrees to quaternion [x, y, z, w]."""
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("xyz", rpy_deg, degrees=True).as_quat()


def _densify_waypoints(
    waypoints: np.ndarray,
    *,
    max_joint_step: float = _GEOMETRIC_MAX_JOINT_STEP,
) -> np.ndarray:
    """Densify joint waypoints so max joint delta per step <= max_joint_step."""
    waypoints = np.atleast_2d(np.asarray(waypoints, dtype=np.float64))
    if waypoints.shape[0] <= 1:
        return waypoints
    dense: list[np.ndarray] = [waypoints[0].copy()]
    safe_step = max(1e-6, float(max_joint_step))
    for i in range(waypoints.shape[0] - 1):
        start, end = waypoints[i], waypoints[i + 1]
        max_delta = float(np.max(np.abs(end - start)))
        n_seg = max(1, int(np.ceil(max_delta / safe_step)))
        for s in range(1, n_seg + 1):
            dense.append(start + (s / n_seg) * (end - start))
    return np.asarray(dense, dtype=np.float64)


def make_namespace(
    env: RoboCasaEnv,
    *,
    vlm_backend: str = "gemini",
    curobo_host: str = _CUROBO_HOST,
    curobo_port: int = _CUROBO_PORT,
    sam3_host: str | None = None,
    sam3_port: int | None = None,
    sam3_url: str | None = None,
    anygrasp_host: str | None = None,
    anygrasp_port: int | None = None,
    anygrasp_url: str | None = None,
    pyroki_host: str = _PYROKI_HOST,
    pyroki_port: int = _PYROKI_PORT,
    policy_runner: Any = None,
    cfg: Any = None,
    runtime_role: str = "script",
) -> dict[str, Any]:
    """Build complete tool namespace for agent code execution.

    Returns a dict of callable tools, ready to inject into AgentContext.namespace.
    The callables return the same dataclass types (RobotState, FreespaceResult,
    NudgeResult, Detection3D, etc.) as the CapServer-backed direct callables.

    ``runtime_role`` controls whether eval-time cheese surfaces are exposed:

    - ``"script"`` (default — used by ``run_script.py`` for ``evaluate_code``
      subprocesses): drops ``get_task_info``, ``reset_env``, and the per-fixture
      ``_debug_info`` helpers from the returned dict. Eval ``code.py`` only sees
      perception/actuation/`get_task_description` and cannot reach the oracle
      or k-shot-retry the policy on the same scene.
    - ``"worker"`` / ``"agent"`` / anything else: full surface, including the
      oracle and reset_env. Used by the MCP env_worker subprocess (so the
      `mcp__robot__get_task_info` tool and the vocab-builder agent can reach
      the oracle via RPC) and by the legacy in-process agent loop.

    The control is structural — the cheese keys are absent from the dict, not
    swapped for raising stubs by an env var. Callers cannot opt back in.
    """
    from enpire.env.forge.cap.agent.tools.base import (
        ArmState,
        Detection3D,
        FreespaceResult,
        NudgeResult,
        RobotState,
    )

    _sam3_host = sam3_host or os.environ.get("SAM3_SERVER_HOST", "localhost")
    _sam3_port = int(
        sam3_port
        if sam3_port is not None
        else os.environ.get("SAM3_SERVER_PORT", "6767")
    )
    _sam3_url = (
        sam3_url
        or os.environ.get("SAM3_SERVER_URL")
        or f"http://{_sam3_host}:{_sam3_port}"
    )
    _anygrasp_host = anygrasp_host or os.environ.get(
        "ANYGRASP_SERVER_HOST", "localhost"
    )
    _anygrasp_port = int(
        anygrasp_port
        if anygrasp_port is not None
        else os.environ.get("ANYGRASP_SERVER_PORT", "9300")
    )
    _anygrasp_url = (
        anygrasp_url
        or os.environ.get("ANYGRASP_SERVICE_URL")
        or f"http://{_anygrasp_host}:{_anygrasp_port}"
    )

    # Last COMMANDED gripper fraction per side — written by
    # set_gripper/open_gripper/close_gripper, read by _execute_osc_trajectory
    # so the arm motion preserves grasps. We must not reuse the MEASURED
    # ``gripper_pos`` for this: when the gripper is clamped on an object,
    # gripper_pos settles at the object's width (e.g. 0.55 for a lemon wedge),
    # and passing that to robosuite's GRIP controller would command "open to
    # 0.55", which translates to an incremental-open action every tick and
    # drops the object mid-trajectory.
    _commanded_gripper: dict[str, float] = {}

    # Whether the last close_gripper detected contact (fingers stalled on an
    # object) vs. finished with empty jaws. Set by ``close_gripper`` when
    # ``compliant=True``; stays ``None`` for non-compliant closes (which
    # have no stall-detection phase to report on).
    _gripper_last_contact: dict[str, bool | None] = {}

    # Mutable holder — all closures read _h["env"] so reset_env can swap it.
    _h: dict[str, Any] = {"env": env}

    def _env() -> RoboCasaEnv:
        return _h["env"]

    # ------------------------------------------------------------------
    # Pyroki HTTP /plan client (no collision avoidance)
    # ------------------------------------------------------------------

    def _plan_via_pyroki(
        current_jp: np.ndarray,
        start_pos_b: np.ndarray,
        start_quat_xyzw_b: np.ndarray,
        target_pos_b: np.ndarray,
        target_quat_xyzw_b: np.ndarray,
        timesteps: int = 30,
    ) -> dict[str, Any]:
        """Request a joint trajectory from the pyroki /plan HTTP server.

        Returns the same ``{status, right_positions}`` shape as cuRobo's
        ``plan_to_pose`` so the caller can stay backend-agnostic. Pyroki
        plans linear SE(3) + IK with no collision checking — for tabletop
        free-space moves only.
        """
        import requests

        # Pyroki expects wxyz quat order; we have xyzw.
        def _to_wxyz_xyz(quat_xyzw, pos):
            return [
                float(quat_xyzw[3]),
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            ]

        url = f"http://{pyroki_host}:{pyroki_port}/plan"
        payload = {
            "start_pose_wxyz_xyz": _to_wxyz_xyz(start_quat_xyzw_b, start_pos_b),
            "end_pose_wxyz_xyz": _to_wxyz_xyz(target_quat_xyzw_b, target_pos_b),
            "timesteps": int(timesteps),
            "dt": 0.02,
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {"status": "Error", "status_detail": f"pyroki request failed: {exc}"}

        waypoints = data.get("waypoints") or []
        if not waypoints:
            return {"status": "Planning_Failed", "status_detail": "empty waypoints"}

        # Pyroki's panda_description URDF includes 7 arm + 1-2 finger joints.
        # robocasa's get_arm_observation returns only the 7 arm joints. Pyroki
        # always emits arm joints first (panda_joint1..7), so slice to match.
        wp_arr = np.asarray(waypoints, dtype=np.float64)
        if wp_arr.ndim != 2:
            return {
                "status": "Planning_Failed",
                "status_detail": f"pyroki returned shape {wp_arr.shape}, expected 2D",
            }
        n_arm = current_jp.shape[0]
        if wp_arr.shape[1] < n_arm:
            return {
                "status": "Planning_Failed",
                "status_detail": (
                    f"pyroki returned {wp_arr.shape[1]} DoF, fewer than the "
                    f"{n_arm} arm joints required"
                ),
            }
        if wp_arr.shape[1] > n_arm:
            wp_arr = wp_arr[:, :n_arm]

        return {
            "status": "Success",
            "right_positions": wp_arr,
        }

    # ------------------------------------------------------------------
    # Lazy cuRobo planner connection (cloud-served)
    # ------------------------------------------------------------------

    _planner_holder: dict[str, Any] = {}

    def _get_planner():
        if "planner" not in _planner_holder:
            from enpire.env.forge.experimental.portal_motion_planner import PortalMotionPlanner

            logger.info(
                "Connecting to cuRobo planner at %s:%s", curobo_host, curobo_port
            )
            _planner_holder["planner"] = PortalMotionPlanner(
                backend="curobo",
                host=curobo_host,
                port=curobo_port if curobo_port else None,
                start_server=curobo_port == 0,
                robot_type="panda",
            )
        return _planner_holder["planner"]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_robot_state() -> RobotState:
        e = _env()
        arm_names = e._profile.arm_names if e._profile else ("right",)
        arms: dict[str, ArmState] = {}
        for side in arm_names:
            obs = e.get_arm_observation(side)
            quat = obs.get("ee_quat", np.array([0, 0, 0, 1]))
            arms[side] = ArmState(
                joint_pos=obs["joint_pos"],
                gripper_pos=obs["gripper_pos"],
                ee_pos=obs.get("ee_pos", np.zeros(3)),
                ee_quat=quat,
                ee_rpy=_quat_xyzw_to_rpy(quat),
            )
        return RobotState(arms=arms)

    def get_gripper_info(side: str = "right") -> dict:
        """Detailed gripper state — use after ``close_gripper`` to check grasp.

        Returns a dict with:

        - ``pos``              (float, ``[0, 1]``)  Normalised finger width.
                               0 = fully closed, 1 = fully open.
        - ``qpos_m``           (float, meters)      Raw mean finger qpos
                               (per-finger; total width ≈ 2 × this).
        - ``is_fully_closed``  (bool)  ``pos < 0.02``.
        - ``is_fully_open``    (bool)  ``pos > 0.98``.
        - ``commanded``        (float | None)  Last commanded value from
                               ``set_gripper``/``open_gripper``/``close_gripper``.
                               ``None`` if none has been called yet.
        - ``has_object``       (bool | None)  ``True`` if a compliant close
                               stalled on contact; ``False`` after
                               open_gripper or an empty compliant close;
                               ``None`` if unknown (non-compliant close
                               doesn't perform stall detection).
        - ``actuator_force_N`` (float | None)  Current position-actuator
                               force on one finger (positive = closing on
                               object). Reads from the live MuJoCo data.
                               ``None`` if the actuator can't be located.
        - ``contact_bodies``   (list[str])  Body names currently in contact
                               with a gripper finger geom. Empty if no
                               contact. Best-effort — relies on robosuite's
                               naming conventions.
        """
        e = _env()
        obs = e.get_arm_observation(side)
        pos = float(obs["gripper_pos"][0]) if "gripper_pos" in obs else 0.0
        qpos_m = float(pos * 0.04)  # un-normalise: finger qpos range is [0, 0.04]

        info: dict = {
            "pos": pos,
            "qpos_m": qpos_m,
            "is_fully_closed": pos < 0.02,
            "is_fully_open": pos > 0.98,
            "commanded": _commanded_gripper.get(side),
            "has_object": _gripper_last_contact.get(side),
            "actuator_force_N": None,
            "contact_bodies": [],
        }

        # --- MuJoCo-side introspection (best effort) -----------------------
        try:
            import mujoco as mj

            model = e._env.sim.model._model
            data = e._env.sim.data._data

            # Sum of abs actuator force across the two finger actuators.
            force_sum = 0.0
            n_found = 0
            for act_i in range(model.nu):
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, act_i) or ""
                if "gripper" in name and "finger" in name:
                    force_sum += abs(float(data.actuator_force[act_i]))
                    n_found += 1
            if n_found > 0:
                info["actuator_force_N"] = force_sum / n_found

            # Contact bodies against gripper finger geoms.
            finger_geom_ids: set[int] = set()
            for geom_i in range(model.ngeom):
                gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_i) or ""
                if "finger" in gname and (
                    "gripper" in gname or gname.startswith("gripper0")
                ):
                    finger_geom_ids.add(geom_i)

            contact_body_names: list[str] = []
            for ci in range(data.ncon):
                c = data.contact[ci]
                # One geom is a finger, the other is the thing we touched.
                other = None
                if c.geom1 in finger_geom_ids and c.geom2 not in finger_geom_ids:
                    other = c.geom2
                elif c.geom2 in finger_geom_ids and c.geom1 not in finger_geom_ids:
                    other = c.geom1
                if other is not None:
                    body_id = int(model.geom_bodyid[other])
                    bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
                    # Skip self-contacts (hand body touching finger body).
                    if bname and "gripper" not in bname and "finger" not in bname:
                        contact_body_names.append(bname)
            info["contact_bodies"] = sorted(set(contact_body_names))
        except Exception as exc:
            logger.debug("get_gripper_info MuJoCo introspection failed: %s", exc)

        return info

    # ------------------------------------------------------------------
    # Motion: cuRobo joint waypoints executed via OSC_POSE tick-tracking
    # ------------------------------------------------------------------

    def _execute_osc_trajectory(
        side: str,
        waypoints: np.ndarray,
        gripper_positions: np.ndarray | None = None,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
        final_target_pos: np.ndarray | None = None,
        final_target_quat: np.ndarray | None = None,
    ) -> None:
        """Track a cuRobo joint trajectory via OSC_POSE with bounded per-tick
        EE steps (so OSC never saturates its output and motion stays gentle
        enough to preserve grasped objects).

        Algorithm:
        1. FK each cuRobo joint waypoint → WORLD-frame EE pose
           (auto-calibrated against robosuite's grip-site convention).
        2. For each consecutive (prev, next) EE pose pair, interpolate in
           sub-steps capped at ``_OSC_MAX_POS_STEP_M`` /
           ``_OSC_MAX_ROT_STEP_RAD``. Each sub-step = one OSC tick =
           one env.step(). Peak EE velocity ≈ 0.02 m × 20 Hz = 40 cm/s.
        3. After the final waypoint, run a few "hold-target" ticks so OSC
           can close the residual error within tolerance (PID settling).

        Null-space is no longer pinned by cuRobo's joint plan — OSC picks
        the elbow configuration. Tight-clearance reaches may drift. Accepted
        trade-off for the single-controller architecture.
        """
        e = _env()
        waypoints = np.atleast_2d(np.asarray(waypoints, dtype=np.float64))
        n_wp = waypoints.shape[0]
        if n_wp == 0:
            return

        # Downsample long cuRobo trajectories to ``max_waypoints`` entries.
        # Always preserve the first and last waypoint so start/end accuracy
        # is unaffected. This is the biggest speed win — env.step() dominates
        # wall clock, and we don't need every cuRobo tick to track smoothly
        # via OSC with sub-step interpolation.
        if max_waypoints and n_wp > max_waypoints:
            stride = int(np.ceil(n_wp / max_waypoints))
            idx = list(range(0, n_wp, stride))
            if idx[-1] != n_wp - 1:
                idx.append(n_wp - 1)
            waypoints = waypoints[idx]
            n_wp = waypoints.shape[0]

        # Per-tick step sizes scale with ``speed`` but stay under saturation
        # ceilings so OSC never bangs into its output bound.
        speed = max(0.1, float(speed))
        pos_step = min(_OSC_POS_STEP_CEILING_M, _OSC_MAX_POS_STEP_M * speed)
        rot_step = min(_OSC_ROT_STEP_CEILING_RAD, _OSC_MAX_ROT_STEP_RAD * speed)

        n_arm = waypoints.shape[1]

        # Finger value for FK (doesn't affect panda_hand pose but pyroki
        # wants the full actuated cfg).
        measured_finger_frac = 1.0
        _fobs = e.get_arm_observation(side).get("gripper_pos")
        if _fobs is not None:
            measured_finger_frac = float(_fobs[0])
        finger_val = float(np.clip(measured_finger_frac * 0.04, 0.0, 0.04))

        # Gripper COMMAND — last set_gripper/open/close intent. Preserves
        # grasps; measured gripper_pos would be the half-closed-on-object
        # state which robosuite interprets as "open slightly".
        hold_gripper = _commanded_gripper.get(side, measured_finger_frac)

        # Ensure FK→eef calibration is ready (first freespace_move).
        if _EE_POS_OFFSET_HAND is None:
            _calibrate_hand_to_eef(e, side)

        # Seed "previous target" from the arm's current EE so the first
        # interpolation starts where the arm is, not from waypoint 0.
        obs0 = e.get_arm_observation(side)
        prev_pos = np.asarray(obs0.get("ee_pos", np.zeros(3)), dtype=np.float64)
        prev_quat = np.asarray(obs0.get("ee_quat", [0, 0, 0, 1]), dtype=np.float64)

        for i, arm_wp in enumerate(waypoints):
            if gripper_positions is not None and i < len(gripper_positions):
                gp = float(np.clip(gripper_positions[i], 0.0, 1.0))
            else:
                gp = hold_gripper

            cfg_8dof = (
                np.concatenate([arm_wp[:n_arm], [finger_val]]) if n_arm == 7 else arm_wp
            )
            # For the FINAL waypoint, prefer the user's requested target
            # (exact, in world frame) over pyroki FK of cuRobo's last joint
            # config. pyroki's URDF can drift by cm from robosuite's model
            # as the arm leaves its calibration config — that drift is fine
            # for intermediate trajectory shape but unacceptable at the
            # endpoint where we need to land on the caller's target.
            is_last = i == n_wp - 1
            if is_last and final_target_pos is not None:
                target_pos = np.asarray(final_target_pos, dtype=np.float64)
                target_quat = (
                    np.asarray(final_target_quat, dtype=np.float64)
                    if final_target_quat is not None
                    else _fk_eef_pose_world(cfg_8dof, e, side)[1]
                )
            else:
                target_pos, target_quat = _fk_eef_pose_world(cfg_8dof, e, side)

            # Sub-step count so each tick moves ≤ pos_step and rotates ≤
            # rot_step. Both scale with ``speed``.
            pos_gap = float(np.linalg.norm(target_pos - prev_pos))
            rot_gap = _rotation_error_rad(prev_quat, target_quat)
            n_sub = max(
                1,
                int(np.ceil(pos_gap / pos_step)),
                int(np.ceil(rot_gap / rot_step)),
            )

            for s in range(1, n_sub + 1):
                alpha = s / n_sub
                interp_pos = (1.0 - alpha) * prev_pos + alpha * target_pos
                interp_quat = _lerp_quat_xyzw(prev_quat, target_quat, alpha)
                e.compute_eef_action(
                    side=side,
                    target_pos=interp_pos,
                    target_quat_xyzw=interp_quat,
                    gripper=gp,
                )
                e.step()

            prev_pos = target_pos
            prev_quat = target_quat

        # Final settle: hold the last target so OSC can close residual error.
        for _ in range(_OSC_FINAL_SETTLE_TICKS):
            obs = e.get_arm_observation(side)
            cur_pos = np.asarray(obs.get("ee_pos", np.zeros(3)))
            cur_quat = np.asarray(obs.get("ee_quat", [0, 0, 0, 1]))
            pos_err = float(np.linalg.norm(cur_pos - prev_pos))
            rot_err = _rotation_error_rad(cur_quat, prev_quat)
            if pos_err < _OSC_FINAL_POS_TOL_M and rot_err < _OSC_FINAL_ROT_TOL_RAD:
                break
            e.compute_eef_action(
                side=side,
                target_pos=prev_pos,
                target_quat_xyzw=prev_quat,
                gripper=hold_gripper,
            )
            e.step()

    def move_joint_keypoints(
        side: str,
        timestamps: list[float],
        joint_positions: list,
        gripper_positions: list | None = None,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
    ) -> dict:
        """Execute a joint trajectory via OSC tick-tracking (FK each waypoint).

        Signature kept for CapServer RPC compatibility; timestamps are
        accepted but ignored (direct mode steps synchronously).
        """
        jps = np.asarray(joint_positions, dtype=np.float64)
        if jps.ndim == 1:
            jps = jps.reshape(1, -1)
        gps = None
        if gripper_positions is not None:
            gps = np.asarray(gripper_positions, dtype=np.float64).ravel()
        _execute_osc_trajectory(
            side,
            jps,
            gripper_positions=gps,
            speed=speed,
            max_waypoints=max_waypoints,
        )
        return {"success": True}

    # ------------------------------------------------------------------
    # World → base frame conversion (cuRobo plans in base frame)
    # ------------------------------------------------------------------

    def _world_to_base(pos: np.ndarray, quat_xyzw: np.ndarray):
        """Convert world-frame pos/quat to robot base frame."""
        from scipy.spatial.transform import Rotation as R

        bp = _env().get_base_pose()
        base_pos = np.asarray(bp.get("base_pos", np.zeros(3)), dtype=np.float64)
        base_quat = np.asarray(bp.get("base_quat", [0, 0, 0, 1]), dtype=np.float64)
        R_base = R.from_quat(base_quat)
        pos_b = R_base.inv().apply(np.asarray(pos) - base_pos)
        quat_b = (R_base.inv() * R.from_quat(quat_xyzw)).as_quat()
        return pos_b, quat_b

    # ------------------------------------------------------------------
    # Motion planners are first-class peer tools (move_with_curobo /
    # move_with_pyroki) sharing this _plan_and_move core. cuRobo and pyroki
    # have different capability surfaces — cuRobo does gradient batch with
    # collision avoidance, pyroki does linear SE(3)+IK with no collision
    # avoidance — so the public tools expose only the kwargs each backend
    # actually consumes. ``freespace_move`` is kept as a back-compat alias
    # for existing skill_library/ callers.
    # ------------------------------------------------------------------

    def _plan_and_move(
        backend: str,
        target_pos: list[float] | np.ndarray | None,
        target_quat: list[float] | np.ndarray | None = None,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
        planning_speed: float = _DEFAULT_PLANNING_SPEED,
        exclude_body_prefixes: list[str] | None = None,
    ) -> FreespaceResult:
        """Shared plan + execute core. Robocasa is single-arm (right);
        ``side`` is hardcoded — not exposed on the public peer tools.

        Backend-specific kwargs (``planning_speed``, ``exclude_body_prefixes``)
        are accepted here for the cuRobo path; pyroki ignores them.
        """
        e = _env()
        if target_pos is None:
            return FreespaceResult(status="Invalid", reason="No target pos provided")

        SIDE = "right"
        backend_norm = (backend or "curobo").strip().lower()

        obs = e.get_arm_observation(SIDE)
        current_jp = np.asarray(obs["joint_pos"], dtype=np.float64)

        target_pos_arr = np.asarray(target_pos, dtype=np.float64)
        target_quat_arr = (
            np.asarray(target_quat, dtype=np.float64)
            if target_quat is not None
            else np.asarray(obs.get("ee_quat", [0, 0, 0, 1]), dtype=np.float64)
        )

        # Convert world → base frame (planners operate in base frame)
        target_pos_b, target_quat_b = _world_to_base(target_pos_arr, target_quat_arr)

        if backend_norm == "pyroki":
            # Linear SE(3) + IK via pyroki HTTP server. No collision avoidance
            # — exclude_body_prefixes / collision_data hooks are skipped.
            start_pos_w = np.asarray(obs.get("ee_pos", [0, 0, 0]), dtype=np.float64)
            start_quat_w = np.asarray(
                obs.get("ee_quat", [0, 0, 0, 1]), dtype=np.float64
            )
            start_pos_b, start_quat_b = _world_to_base(start_pos_w, start_quat_w)
            plan_result = _plan_via_pyroki(
                current_jp=current_jp,
                start_pos_b=start_pos_b,
                start_quat_xyzw_b=start_quat_b,
                target_pos_b=target_pos_b,
                target_quat_xyzw_b=target_quat_b,
            )
        elif backend_norm == "curobo":
            planner = _get_planner()

            # Populate the planner's world from the current sim scene. Without
            # this, cuRobo plans against a ground-plane-only world and will
            # happily path through cabinets / counters / walls. Failure to
            # update is logged and downgraded to a warning — we still plan
            # (same behaviour as before this change: empty world), so a single
            # failed update doesn't break the task.
            try:
                kwargs: dict = {}
                if exclude_body_prefixes is not None:
                    kwargs["exclude_body_prefixes"] = list(exclude_body_prefixes)
                collision_data = e.get_collision_geoms(**kwargs)
                n_obstacles = planner.update_world_from_geoms(collision_data)
                logger.debug(
                    "move_with_curobo: world updated with %d obstacles", n_obstacles
                )
            except Exception as exc:
                logger.warning(
                    "move_with_curobo: world update failed (%s); "
                    "planning against stale/empty world",
                    exc,
                )

            # Plan — Panda planner uses right_* convention for single arm
            plan_result = planner.plan_to_pose(
                current_left_jp=np.zeros_like(current_jp),
                current_right_jp=current_jp,
                target_right_pos=target_pos_b,
                target_right_quat_xyzw=target_quat_b,
                side="right",
            )
        else:
            return FreespaceResult(
                status="Invalid",
                reason=f"unknown planner backend {backend!r}; expected 'curobo' or 'pyroki'",
            )

        status = plan_result.get("status", "Error")
        if status != "Success":
            return FreespaceResult(
                status=status,
                reason=plan_result.get("status_detail", "Planning failed"),
            )

        # Extract trajectory waypoints
        waypoints = np.asarray(plan_result["right_positions"], dtype=np.float64)
        if waypoints.ndim < 2 or waypoints.shape[0] == 0:
            return FreespaceResult(status="Planning_Failed", reason="Empty trajectory")

        # Execute — pass the user's requested target (world frame) so the
        # FINAL waypoint tracks to it exactly instead of settling at FK of
        # the planner's final joint config (which can drift from robosuite's
        # actual EE pose as the arm leaves its calibration neighbourhood).
        _execute_osc_trajectory(
            SIDE,
            waypoints,
            speed=speed,
            max_waypoints=max_waypoints,
            final_target_pos=target_pos_arr,
            final_target_quat=target_quat_arr,
        )

        # Check final error
        final_obs = e.get_arm_observation(SIDE)
        final_pos = np.asarray(final_obs["ee_pos"])
        pos_err = float(np.linalg.norm(final_pos - target_pos_arr))

        return FreespaceResult(
            status="Success",
            final_pos_error_m=round(pos_err, 4),
            trajectory_steps=waypoints.shape[0],
            executed=True,
            side=SIDE,
        )

    def move_with_curobo(
        target_pos: list[float] | None = None,
        target_quat: list[float] | None = None,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
        planning_speed: float = _DEFAULT_PLANNING_SPEED,
        exclude_body_prefixes: list[str] | None = None,
    ) -> FreespaceResult:
        """Move arm to target EE pose via cuRobo collision-free planning.

        cuRobo is a gradient-based batch planner with full collision
        avoidance. Always reads the current sim scene before planning.

        Targets are in **world frame**. Single-arm (right) only.

        Args:
            target_pos: World-frame [x, y, z] for the EE.
            target_quat: World-frame [x, y, z, w] for the EE. Defaults to
                current orientation.
            speed: OSC tick speed multiplier. 1.0 ≈ 40 cm/s peak EE
                velocity (safe for grasped objects), 2.0 ≈ 80 cm/s.
            max_waypoints: Downsample cuRobo's trajectory to at most this
                many waypoints before OSC execution.
            planning_speed: cuRobo solver tuning — joint velocity limit.
            exclude_body_prefixes: Body name prefixes to exclude from the
                collision world. Pass [""] to clear all obstacles, ["obj"]
                to ignore grasped objects, etc.
        """
        return _plan_and_move(
            "curobo",
            target_pos=target_pos,
            target_quat=target_quat,
            speed=speed,
            max_waypoints=max_waypoints,
            planning_speed=planning_speed,
            exclude_body_prefixes=exclude_body_prefixes,
        )

    def move_with_pyroki(
        target_pos: list[float] | None = None,
        target_quat: list[float] | None = None,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
    ) -> FreespaceResult:
        """Move arm to target EE pose via pyroki linear SE(3) + IK.

        pyroki interpolates linearly between start and target poses with
        per-waypoint IK via the HTTP server. **No collision avoidance** —
        pyroki has no world model. Use this when ``move_with_curobo``
        returns ``Planning_Failed`` in a narrow corridor where cuRobo's
        gradient solver got stuck in a local minimum, or when you need
        to bypass collision checking deliberately (e.g. nudging against
        a known fixture).

        Targets are in **world frame**. Single-arm (right) only.

        Args:
            target_pos: World-frame [x, y, z] for the EE.
            target_quat: World-frame [x, y, z, w] for the EE. Defaults to
                current orientation.
            speed: OSC tick speed multiplier (see ``move_with_curobo``).
            max_waypoints: Downsample the linear interpolation to at most
                this many waypoints before OSC execution.
        """
        return _plan_and_move(
            "pyroki",
            target_pos=target_pos,
            target_quat=target_quat,
            speed=speed,
            max_waypoints=max_waypoints,
        )

    def freespace_move(
        right_target_pos: list[float] | None = None,
        right_target_quat: list[float] | None = None,
        side: str = "right",
        planning_speed: float = _DEFAULT_PLANNING_SPEED,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
        exclude_body_prefixes: list[str] | None = None,
        **_: Any,
    ) -> FreespaceResult:
        """Back-compat alias for ``move_with_curobo`` / ``move_with_pyroki``.

        Dispatches to whichever backend ``CAP_PLANNER_BACKEND`` selects
        (default ``curobo``). Existing skills under
        ``cap/saved_scripts/robocasa_skill_library/`` and per-run
        ``edit/skill_library/`` still call this name with the legacy
        ``right_target_pos`` / ``side`` kwargs; new code paths should
        prefer the explicit ``move_with_curobo(...)`` / ``move_with_pyroki(...)``
        so the trace shows which backend ran without cross-referencing
        env state.
        """
        return _plan_and_move(
            _PLANNER_BACKEND,
            target_pos=right_target_pos,
            target_quat=right_target_quat,
            speed=speed,
            max_waypoints=max_waypoints,
            planning_speed=planning_speed,
            exclude_body_prefixes=exclude_body_prefixes,
        )

    # ------------------------------------------------------------------
    # Batched grasp candidate ranking via cuRobo
    # ------------------------------------------------------------------

    def select_best_grasp(
        grasp_candidates: list,
        side: str = "right",
        batch_top_k: int = 10,
        planning_speed: float = _DEFAULT_PLANNING_SPEED,
        exclude_body_prefixes: list[str] | None = None,
        augment_yaw_flip: bool = False,
    ) -> FreespaceResult:
        """Rank grasp candidates through cuRobo batch IK+planning in one call.

        Each candidate must have ``.position`` (world xyz) and ``.rpy``
        (display-convention degrees). Returns a ``FreespaceResult`` with
        ``best_candidate`` set to the top feasible grasp (includes
        ``.position``, ``.rpy``, ``.score``, ``.width``). The trajectory
        for the best candidate is **not** cached — call ``freespace_move``
        to execute it afterward.

        Args:
            grasp_candidates: List of GraspCandidate (or anything with
                position/rpy/score/width attrs).
            side: Arm side.
            batch_top_k: Evaluate at most this many candidates (sorted by
                score descending before truncation).
            planning_speed: cuRobo joint velocity limit.
            exclude_body_prefixes: Body name prefixes to exclude from collision
                world. ``None`` = full world (default). ``[""]`` = clear all.
            augment_yaw_flip: If True, double candidates by adding 180°
                yaw-flipped variants (for symmetric parallel-jaw grippers).
        """
        if augment_yaw_flip:
            from enpire.env.forge.cap.env.base.skill_library import GraspCandidate

            augmented = []
            for g in grasp_candidates:
                augmented.append(g)
                flipped_yaw = ((g.rpy[2] + 180.0) + 180.0) % 360.0 - 180.0
                augmented.append(
                    GraspCandidate(
                        position=list(g.position),
                        rpy=[g.rpy[0], g.rpy[1], round(flipped_yaw, 4)],
                        score=g.score * 0.99,
                        width=getattr(g, "width", 0.08),
                    )
                )
            grasp_candidates = augmented
        from scipy.spatial.transform import Rotation as Rot

        from enpire.env.forge.cap.agent.tools.base import FreespaceBatchCandidate

        if not grasp_candidates:
            return FreespaceResult(
                status="Error",
                reason="No grasp candidates provided",
                planning_mode="batch",
                side=side,
            )

        e = _env()
        planner = _get_planner()

        try:
            kwargs: dict = {}
            if exclude_body_prefixes is not None:
                kwargs["exclude_body_prefixes"] = list(exclude_body_prefixes)
            n = planner.update_world_from_geoms(e.get_collision_geoms(**kwargs))
            logger.debug("select_best_grasp: world updated with %d obstacles", n)
        except Exception as exc:
            logger.warning("select_best_grasp: world update failed (%s)", exc)

        obs = e.get_arm_observation(side)
        current_jp = np.asarray(obs["joint_pos"], dtype=np.float64)

        def _display_rpy_to_quat(rpy):
            roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
            return Rot.from_euler(
                "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
            ).as_quat()

        records = []
        for i, c in enumerate(grasp_candidates):
            records.append(
                {
                    "source_index": i,
                    "position": np.asarray(c.position, dtype=np.float64),
                    "rpy": np.asarray(c.rpy, dtype=np.float64),
                    "score": float(c.score),
                    "width": float(getattr(c, "width", 0.08)),
                }
            )
        records.sort(key=lambda r: r["score"], reverse=True)
        records = records[:batch_top_k]

        grasp_xyzs_world = np.array([r["position"] for r in records], dtype=np.float64)
        grasp_quats_world = np.array(
            [_display_rpy_to_quat(r["rpy"]) for r in records], dtype=np.float64
        )

        bp = e.get_base_pose()
        base_pos = np.asarray(bp.get("base_pos", np.zeros(3)), dtype=np.float64)
        base_quat = np.asarray(bp.get("base_quat", [0, 0, 0, 1]), dtype=np.float64)
        R_base = Rot.from_quat(base_quat)

        grasp_xyzs_base = R_base.inv().apply(grasp_xyzs_world - base_pos)
        grasp_quats_base = np.array(
            [(R_base.inv() * Rot.from_quat(q)).as_quat() for q in grasp_quats_world],
            dtype=np.float64,
        )

        try:
            batch_result = planner.plan_batch_to_pose(
                current_left_jp=np.zeros_like(current_jp),
                current_right_jp=current_jp,
                target_right_pos=grasp_xyzs_base,
                target_right_quat_xyzw=grasp_quats_base,
                side="right",
                max_joint_vel=planning_speed,
                validate_trajectory=True,
            )
        except Exception as exc:
            return FreespaceResult(
                status="Error",
                reason=f"Batch planning failed: {exc}",
                planning_mode="batch",
                side=side,
                input_candidate_count=len(grasp_candidates),
            )

        success_mask = np.asarray(
            batch_result.get("success_mask", []), dtype=bool
        ).reshape(-1)
        pos_errors = np.asarray(
            batch_result.get("position_error_m", [np.nan] * len(records)),
            dtype=np.float64,
        ).reshape(-1)
        rot_errors = np.asarray(
            batch_result.get("rotation_error_deg", [np.nan] * len(records)),
            dtype=np.float64,
        ).reshape(-1)

        raw_statuses = batch_result.get("status_by_index")
        if isinstance(raw_statuses, list) and len(raw_statuses) == len(records):
            status_by_index = [None if s is None else str(s) for s in raw_statuses]
        elif success_mask.size >= len(records):
            status_by_index = [
                "Success" if bool(success_mask[i]) else "Planning_Failed"
                for i in range(len(records))
            ]
        else:
            status_by_index = ["Planning_Failed"] * len(records)

        raw_details = batch_result.get("status_detail_by_index")
        if isinstance(raw_details, list) and len(raw_details) == len(records):
            detail_by_index = [None if d is None else str(d) for d in raw_details]
        else:
            detail_by_index = [None] * len(records)

        row_payloads: list[dict[str, object]] = []
        for idx, rec in enumerate(records):
            ok = bool(success_mask[idx]) if idx < len(success_mask) else False
            cand_status = str(status_by_index[idx] or "").strip() or (
                "Success" if ok else "Planning_Failed"
            )
            pe = float(pos_errors[idx]) if idx < len(pos_errors) else float("nan")
            re = float(rot_errors[idx]) if idx < len(rot_errors) else float("nan")

            if not ok:
                if cand_status == "IK_Failed" and np.isfinite(pe):
                    row_payloads.append(
                        {
                            "source_index": rec["source_index"],
                            "position": [round(float(x), 5) for x in rec["position"]],
                            "rpy": [round(float(x), 4) for x in rec["rpy"]],
                            "score": round(rec["score"], 4),
                            "width": round(rec["width"], 5),
                            "ik_error_m": round(pe, 6),
                            "ik_rot_error_deg": round(re, 4)
                            if np.isfinite(re)
                            else None,
                            "within_ik_threshold": False,
                            "planner_status": "IK_Failed",
                            "motion_plan_error": None,
                            "motion_plan_reason": None,
                        }
                    )
                else:
                    row_payloads.append(
                        {
                            "source_index": rec["source_index"],
                            "position": [round(float(x), 5) for x in rec["position"]],
                            "rpy": [round(float(x), 4) for x in rec["rpy"]],
                            "score": round(rec["score"], 4),
                            "width": round(rec["width"], 5),
                            "ik_error_m": None,
                            "ik_rot_error_deg": None,
                            "within_ik_threshold": False,
                            "planner_status": cand_status,
                            "motion_plan_error": True,
                            "motion_plan_reason": detail_by_index[idx] or cand_status,
                        }
                    )
                continue

            row_payloads.append(
                {
                    "source_index": rec["source_index"],
                    "position": [round(float(x), 5) for x in rec["position"]],
                    "rpy": [round(float(x), 4) for x in rec["rpy"]],
                    "score": round(rec["score"], 4),
                    "width": round(rec["width"], 5),
                    "ik_error_m": round(pe, 6) if np.isfinite(pe) else None,
                    "ik_rot_error_deg": round(re, 4) if np.isfinite(re) else None,
                    "within_ik_threshold": bool(np.isfinite(pe) and np.isfinite(re)),
                    "planner_status": "Success",
                    "motion_plan_error": False,
                    "motion_plan_reason": None,
                }
            )

        row_payloads.sort(
            key=lambda r: (
                0
                if r.get("planner_status") == "Success"
                else (1 if r.get("planner_status") == "IK_Failed" else 2),
                -float(r.get("score", 0.0)),
            )
        )

        candidates_out: list[FreespaceBatchCandidate] = []
        best: FreespaceBatchCandidate | None = None

        for rank, row in enumerate(row_payloads, start=1):
            bc = FreespaceBatchCandidate(rank=rank, **row)
            candidates_out.append(bc)
            if bc.is_executable and best is None:
                best = bc

        return FreespaceResult(
            status="Success" if best is not None else "Planning_Failed",
            reason="" if best else f"No feasible grasp out of {len(records)} evaluated",
            planning_mode="batch",
            side=side,
            batch_candidates=candidates_out,
            best_candidate=best,
            input_candidate_count=len(grasp_candidates),
            evaluated_candidate_count=len(records),
        )

    def select_best_grasp_two_step(
        grasp_candidates: list,
        side: str = "right",
        batch_top_k: int = 10,
        pregrasp_offset_m: float = 0.02,
        planning_speed: float = _DEFAULT_PLANNING_SPEED,
        augment_yaw_flip: bool = False,
    ) -> FreespaceResult:
        """Rank AnyGrasp candidates by score after validating pregrasp and descend.

        Step 1 plans from the current arm state to a pregrasp pose with the
        normal collision world enabled. Step 2 plans from that candidate's
        pregrasp terminal joint state to the raw grasp pose with the collision
        world cleared. A candidate is executable only if both plans succeed;
        among executable candidates, the original AnyGrasp score determines
        rank. Pose errors are retained for diagnostics, not ordering.
        """
        from scipy.spatial.transform import Rotation as Rot

        from enpire.env.forge.cap.agent.tools.base import FreespaceBatchCandidate
        from enpire.env.forge.cap.env.base.skill_library import GraspCandidate

        if augment_yaw_flip:
            augmented = []
            for g in grasp_candidates:
                augmented.append(g)
                flipped_yaw = ((g.rpy[2] + 180.0) + 180.0) % 360.0 - 180.0
                augmented.append(
                    GraspCandidate(
                        position=list(g.position),
                        rpy=[g.rpy[0], g.rpy[1], round(flipped_yaw, 4)],
                        score=float(g.score) * 0.99,
                        width=getattr(g, "width", 0.08),
                    )
                )
            grasp_candidates = augmented

        if not grasp_candidates:
            return FreespaceResult(
                status="Error",
                reason="No grasp candidates provided",
                planning_mode="batch",
                side=side,
            )

        e = _env()
        planner = _get_planner()
        obs = e.get_arm_observation(side)
        current_jp = np.asarray(obs["joint_pos"], dtype=np.float64)

        def _display_rpy_to_quat(rpy):
            roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
            return Rot.from_euler(
                "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
            ).as_quat()

        records = []
        for i, c in enumerate(grasp_candidates):
            midpoint_pos = np.asarray(c.position, dtype=np.float64)
            rpy = np.asarray(c.rpy, dtype=np.float64)
            quat = _display_rpy_to_quat(rpy)
            approach_dir = Rot.from_quat(quat).apply([0.0, 0.0, 1.0])
            pregrasp_pos = midpoint_pos - float(pregrasp_offset_m) * approach_dir
            actual_pos = midpoint_pos + float(pregrasp_offset_m) * approach_dir
            records.append(
                {
                    "source_index": i,
                    "midpoint_pos": midpoint_pos,
                    "actual_pos": actual_pos,
                    "pregrasp_pos": pregrasp_pos,
                    "quat": quat,
                    "rpy": rpy,
                    "score": float(c.score),
                    "width": float(getattr(c, "width", 0.08)),
                }
            )
        records.sort(key=lambda r: r["score"], reverse=True)
        records = records[: int(batch_top_k)]

        pre_pos_b = []
        pre_quat_b = []
        actual_pose_b = []
        for rec in records:
            p_b, q_b = _world_to_base(rec["pregrasp_pos"], rec["quat"])
            pre_pos_b.append(p_b)
            pre_quat_b.append(q_b)
            actual_pose_b.append(_world_to_base(rec["actual_pos"], rec["quat"]))

        try:
            n = planner.update_world_from_geoms(e.get_collision_geoms())
            logger.debug(
                "select_best_grasp_two_step: pregrasp world updated with %d obstacles",
                n,
            )
        except Exception as exc:
            logger.warning(
                "select_best_grasp_two_step: pregrasp world update failed (%s)", exc
            )

        try:
            pre_result = planner.plan_batch_to_pose(
                current_left_jp=np.zeros_like(current_jp),
                current_right_jp=current_jp,
                target_right_pos=np.asarray(pre_pos_b, dtype=np.float64),
                target_right_quat_xyzw=np.asarray(pre_quat_b, dtype=np.float64),
                side="right",
                max_joint_vel=planning_speed,
                validate_trajectory=True,
            )
        except Exception as exc:
            return FreespaceResult(
                status="Error",
                reason=f"Pregrasp batch planning failed: {exc}",
                planning_mode="batch",
                side=side,
                input_candidate_count=len(grasp_candidates),
                evaluated_candidate_count=len(records),
            )

        pre_success = np.asarray(
            pre_result.get("success_mask", []), dtype=bool
        ).reshape(-1)
        pre_pos_err = np.asarray(
            pre_result.get("position_error_m", [np.nan] * len(records)),
            dtype=np.float64,
        ).reshape(-1)
        pre_rot_err = np.asarray(
            pre_result.get("rotation_error_deg", [np.nan] * len(records)),
            dtype=np.float64,
        ).reshape(-1)
        pre_status = pre_result.get("status_by_index")
        if not isinstance(pre_status, list) or len(pre_status) != len(records):
            pre_status = [
                "Success"
                if i < len(pre_success) and bool(pre_success[i])
                else "Planning_Failed"
                for i in range(len(records))
            ]
        pre_details = pre_result.get("status_detail_by_index")
        if not isinstance(pre_details, list) or len(pre_details) != len(records):
            pre_details = [None] * len(records)
        right_paths = pre_result.get("right_positions_by_index")
        if not isinstance(right_paths, list) or len(right_paths) != len(records):
            right_paths = [None] * len(records)

        try:
            n = planner.update_world_from_geoms(
                e.get_collision_geoms(exclude_body_prefixes=[""])
            )
            logger.debug(
                "select_best_grasp_two_step: descend world cleared (%d obstacles)",
                n,
            )
        except Exception as exc:
            logger.warning(
                "select_best_grasp_two_step: descend world clear failed (%s)", exc
            )

        row_payloads: list[dict[str, object]] = []
        sort_keys: list[tuple[int, float, int]] = []
        for idx, rec in enumerate(records):
            pre_ok = idx < len(pre_success) and bool(pre_success[idx])
            pre_pe = float(pre_pos_err[idx]) if idx < len(pre_pos_err) else float("nan")
            pre_re = float(pre_rot_err[idx]) if idx < len(pre_rot_err) else float("nan")
            if not pre_ok:
                status = str(pre_status[idx] or "Planning_Failed")
                row_payloads.append(
                    {
                        "source_index": rec["source_index"],
                        "position": [round(float(x), 5) for x in rec["actual_pos"]],
                        "rpy": [round(float(x), 4) for x in rec["rpy"]],
                        "score": round(float(rec["score"]), 4),
                        "width": round(float(rec["width"]), 5),
                        "ik_error_m": round(pre_pe, 6) if np.isfinite(pre_pe) else None,
                        "ik_rot_error_deg": round(pre_re, 4)
                        if np.isfinite(pre_re)
                        else None,
                        "within_ik_threshold": False,
                        "planner_status": status,
                        "motion_plan_error": None if status == "IK_Failed" else True,
                        "motion_plan_reason": f"pregrasp: {pre_details[idx] or status}",
                    }
                )
                sort_keys.append(
                    (
                        1 if status == "IK_Failed" else 2,
                        -float(rec["score"]),
                        int(rec["source_index"]),
                    )
                )
                continue

            raw_path = right_paths[idx]
            if raw_path is None:
                row_payloads.append(
                    {
                        "source_index": rec["source_index"],
                        "position": [round(float(x), 5) for x in rec["actual_pos"]],
                        "rpy": [round(float(x), 4) for x in rec["rpy"]],
                        "score": round(float(rec["score"]), 4),
                        "width": round(float(rec["width"]), 5),
                        "ik_error_m": round(pre_pe, 6) if np.isfinite(pre_pe) else None,
                        "ik_rot_error_deg": round(pre_re, 4)
                        if np.isfinite(pre_re)
                        else None,
                        "within_ik_threshold": False,
                        "planner_status": "Planning_Failed",
                        "motion_plan_error": True,
                        "motion_plan_reason": "pregrasp: no terminal joint path",
                    }
                )
                sort_keys.append((2, -float(rec["score"]), int(rec["source_index"])))
                continue

            terminal_jp = np.asarray(raw_path, dtype=np.float64).reshape(
                -1, len(current_jp)
            )[-1]
            actual_pos_b, actual_quat_b = actual_pose_b[idx]
            try:
                descend_result = planner.plan_to_pose(
                    current_left_jp=np.zeros_like(terminal_jp),
                    current_right_jp=terminal_jp,
                    target_right_pos=actual_pos_b,
                    target_right_quat_xyzw=actual_quat_b,
                    side="right",
                    max_joint_vel=planning_speed,
                    validate_trajectory=True,
                )
            except Exception as exc:
                descend_result = {
                    "status": "Planning_Failed",
                    "status_detail": str(exc),
                }

            descend_status = str(descend_result.get("status", "Planning_Failed"))
            descend_ok = descend_status == "Success"
            descend_pe = float(descend_result.get("position_error_m", np.nan))
            descend_re = float(descend_result.get("rotation_error_deg", np.nan))
            total_pe = (pre_pe if np.isfinite(pre_pe) else 0.0) + (
                descend_pe if np.isfinite(descend_pe) else 0.0
            )
            total_re = (pre_re if np.isfinite(pre_re) else 0.0) + (
                descend_re if np.isfinite(descend_re) else 0.0
            )
            if descend_ok:
                row_payloads.append(
                    {
                        "source_index": rec["source_index"],
                        "position": [round(float(x), 5) for x in rec["actual_pos"]],
                        "rpy": [round(float(x), 4) for x in rec["rpy"]],
                        "score": round(float(rec["score"]), 4),
                        "width": round(float(rec["width"]), 5),
                        "ik_error_m": round(total_pe, 6),
                        "ik_rot_error_deg": round(total_re, 4),
                        "within_ik_threshold": True,
                        "planner_status": "Success",
                        "motion_plan_error": False,
                        "motion_plan_reason": None,
                    }
                )
                sort_keys.append((0, -float(rec["score"]), int(rec["source_index"])))
            else:
                row_payloads.append(
                    {
                        "source_index": rec["source_index"],
                        "position": [round(float(x), 5) for x in rec["actual_pos"]],
                        "rpy": [round(float(x), 4) for x in rec["rpy"]],
                        "score": round(float(rec["score"]), 4),
                        "width": round(float(rec["width"]), 5),
                        "ik_error_m": round(total_pe, 6)
                        if np.isfinite(total_pe)
                        else None,
                        "ik_rot_error_deg": round(total_re, 4)
                        if np.isfinite(total_re)
                        else None,
                        "within_ik_threshold": False,
                        "planner_status": descend_status,
                        "motion_plan_error": None
                        if descend_status == "IK_Failed"
                        else True,
                        "motion_plan_reason": f"descend: {descend_result.get('status_detail', descend_status)}",
                    }
                )
                sort_keys.append(
                    (
                        1 if descend_status == "IK_Failed" else 2,
                        -float(rec["score"]),
                        int(rec["source_index"]),
                    )
                )

        ordered = sorted(zip(sort_keys, row_payloads), key=lambda item: item[0])
        candidates_out: list[FreespaceBatchCandidate] = []
        best: FreespaceBatchCandidate | None = None
        for rank, (_key, row) in enumerate(ordered, start=1):
            bc = FreespaceBatchCandidate(rank=rank, **row)
            candidates_out.append(bc)
            if bc.is_executable and best is None:
                best = bc

        return FreespaceResult(
            status="Success" if best is not None else "Planning_Failed",
            reason=""
            if best
            else f"No two-step feasible grasp out of {len(records)} evaluated",
            planning_mode="batch",
            side=side,
            batch_candidates=candidates_out,
            best_candidate=best,
            input_candidate_count=len(grasp_candidates),
            evaluated_candidate_count=len(records),
        )

    def _step_gripper(side: str, value: float, max_steps: int = 100) -> None:
        """Drive gripper to target by stepping env synchronously.

        In OSC mode, ``command_arm`` reads the last element of ``cmd['pos']``
        as the gripper fraction and dispatches a zero-arm OSC action with
        that gripper value. Arm holds position because the arm OSC delta is
        zero for each step.
        """
        e = _env()
        value = float(np.clip(value, 0.0, 1.0))
        for _ in range(max_steps):
            obs = e.get_arm_observation(side)
            jp = obs["joint_pos"]
            cmd = np.concatenate([jp, [value]])
            e.command_arm(side, {"pos": cmd})
            e.step()
            obs = e.get_arm_observation(side)
            if abs(float(obs["gripper_pos"]) - value) < 0.01:
                return
        # Don't error — partial gripper motion is acceptable

    def _step_gripper_compliant_close(
        side: str,
        max_steps: int = 60,
        stall_eps: float = 0.0005,  # 0.5 mm per tick = "stopped"
        stall_count: int = 3,  # need N consecutive stall ticks
    ) -> tuple[bool, float]:
        """Drive gripper toward closed until either fully closed or stall.

        Returns ``(contacted, final_gripper_pos)``.
        ``contacted=True`` means fingers stopped moving before reaching fully
        closed — an object is in the jaws. ``contacted=False`` = empty close.

        The stall detector runs on the MEASURED ``gripper_pos`` change
        between consecutive sim ticks — works for any object size, doesn't
        need contact-force sensors.
        """
        e = _env()
        prev = None
        stalls = 0
        cur = 1.0
        for _ in range(max_steps):
            obs = e.get_arm_observation(side)
            jp = obs["joint_pos"]
            # Command maximum close each tick (gripper_value=0 → OSC action=+1).
            e.command_arm(side, {"pos": np.concatenate([jp, [0.0]])})
            e.step()
            cur = float(e.get_arm_observation(side)["gripper_pos"][0])
            # Fully closed — empty grasp, no contact.
            if cur < 0.01:
                return False, cur
            if prev is not None and abs(cur - prev) < stall_eps:
                stalls += 1
                if stalls >= stall_count:
                    return True, cur
            else:
                stalls = 0
            prev = cur
        # Timeout — treat as contact if we're far from fully closed.
        return (cur > 0.05), cur

    def set_gripper(
        side: str,
        pos: float,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> None:
        value = float(np.clip(pos, 0.0, 1.0))
        _commanded_gripper[side] = value
        _gripper_last_contact[side] = None
        _step_gripper(side, value)

    def open_gripper(
        side: str,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> None:
        _commanded_gripper[side] = 1.0
        _gripper_last_contact[side] = False
        _step_gripper(side, 1.0)

    def close_gripper(
        side: str,
        compliant: bool = False,
        hold_strength: float = 0.4,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> None:
        """Close gripper.

        Non-compliant (default): close at max force until timeout — tight grip
        (~20 N from the XML forcerange), survives aggressive arm motion but
        can crush fragile objects.

        Compliant: close until fingers stall on an object, then **decay** the
        gripper's internal ``current_action`` away from fully-closed for a
        few ticks so the position-actuator's clamping force drops. During
        subsequent motion we send ``action=0`` which preserves
        ``current_action`` — force stays at the decayed level.

        robosuite's PandaGripper uses ``sign(action)`` (not magnitude) and
        accumulates ``current_action`` internally via
        ``self.current_action += [-0.2, +0.2] * sign(action)``. So the only
        way to vary sustained clamping force is to ramp ``current_action``
        toward the midpoint before motion starts. Each "open" tick moves it
        by 0.2 per finger; we need ``N = round((1 - hold_strength) / 0.2)``
        ticks to land at ``±hold_strength``.

        Args:
            side: Arm side.
            compliant: Enable stall-detect close + force relaxation.
            hold_strength: ``[0, 1]``. 1.0 = full 20 N clamp (same as
                non-compliant). 0.4 ≈ 8 N. 0.0 ≈ 2 N (minimal, held by
                residual position error + friction).
        """
        e = _env()
        if compliant:
            contacted, final_pos = _step_gripper_compliant_close(side)
            _gripper_last_contact[side] = bool(contacted)
            if contacted:
                hs = float(np.clip(hold_strength, 0.0, 1.0))
                # Decay current_action from fully-closed (±1) toward ±hs.
                # robosuite's PandaGripper.speed = 0.2 per tick.
                n_decay = max(0, int(round((1.0 - hs) / 0.2)))
                for _ in range(n_decay):
                    obs = e.get_arm_observation(side)
                    jp = obs["joint_pos"]
                    # Send "open" command so current_action decays.
                    e.command_arm(side, {"pos": np.concatenate([jp, [1.0]])})
                    e.step()
                # During motion: gripper=0.5 → action=0 → sign(0)=0 →
                # current_action unchanged → clamping force stays at the
                # decayed level we just set.
                _commanded_gripper[side] = 0.5
                final_measured = float(e.get_arm_observation(side)["gripper_pos"][0])
                logger.info(
                    "close_gripper(compliant): contact at %.3f, decayed %d "
                    "ticks to hold_strength=%.2f (gripper_pos now %.3f, "
                    "motion command = action=0)",
                    final_pos,
                    n_decay,
                    hs,
                    final_measured,
                )
            else:
                # Empty close: fingers fully closed, keep max clamp so any
                # object that ends up in the jaws later gets a tight grip.
                _commanded_gripper[side] = 0.0
        else:
            _commanded_gripper[side] = 0.0
            # Non-compliant close doesn't distinguish "stalled on object" from
            # "ran full duration empty" — leave last_contact as a None sentinel
            # so get_gripper_info can report "unknown".
            _gripper_last_contact[side] = None
            _step_gripper(side, 0.0)

    def go_home(
        side: str = "right",
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
        exclude_body_prefixes: list[str] | None = None,
        gripper: float | None = None,
    ) -> FreespaceResult:
        """Drive arm back to its initial EE pose via cuRobo freespace planning."""
        home = _env()._initial_ee.get(side)
        if home is None:
            raise RuntimeError(f"No initial EE pose stored for side={side!r}")
        kwargs: dict = dict(
            right_target_pos=home["pos"].tolist(),
            right_target_quat=home["quat"].tolist(),
            side=side,
            speed=speed,
            max_waypoints=max_waypoints,
            exclude_body_prefixes=exclude_body_prefixes,
        )
        if gripper is not None:
            kwargs["gripper"] = float(gripper)
        return freespace_move(**kwargs)

    def nudge(
        side: str,
        delta_pos: list[float] | None = None,
        delta_rpy: list[float] | None = None,
        speed: float = 1.0,
        max_waypoints: int = _OSC_MAX_WAYPOINTS,
    ) -> NudgeResult:
        """Small delta EE move in world frame via cuRobo planning.

        ``delta_pos``: [dx, dy, dz] in metres.
        ``delta_rpy``: [droll, dpitch, dyaw] in degrees.
        """
        from scipy.spatial.transform import Rotation as R

        obs = _env().get_arm_observation(side)
        cur_pos = np.array(obs["ee_pos"])
        cur_quat = np.array(obs["ee_quat"])

        new_pos = cur_pos + np.array(delta_pos) if delta_pos is not None else cur_pos
        if delta_rpy is not None:
            R_cur = R.from_quat(cur_quat)
            R_delta = R.from_euler("xyz", delta_rpy, degrees=True)
            new_quat = (R_delta * R_cur).as_quat()
        else:
            new_quat = cur_quat

        result = freespace_move(
            right_target_pos=new_pos.tolist(),
            right_target_quat=new_quat.tolist(),
            side=side,
            speed=speed,
            max_waypoints=max_waypoints,
        )

        final_obs = _env().get_arm_observation(side)

        return NudgeResult(
            success=result.status == "Success",
            final_pos=final_obs["ee_pos"],
            final_quat=final_obs["ee_quat"],
        )

    def rotate_gripper_in_place(
        side: str = "right",
        angle_deg: float = 90.0,
        direction: str = "clockwise",
        axis: str = "local_z",
        n_steps: int | None = None,
        settle_steps: int = 20,
        gripper: float | None = None,
        max_pos_drift_m: float = 0.025,
        max_rot_error_deg: float = 10.0,
        respect_joint_limits: bool = True,
        allow_partial_rotation: bool = False,
        joint_limit_margin_rad: float = 0.08726646259971647,
        min_executable_angle_deg: float = 1.0,
    ) -> dict:
        """Rotate the EE orientation while holding the current EE position.

        This bypasses cuRobo/freespace_move and directly commands OSC pose
        targets at the starting position. It is intended for post-grasp wrist
        roll/reorientation where small position drift is acceptable but a
        planned free-space path is too much motion.

        Direction convention: clockwise is negative right-hand rotation when
        viewed along the selected positive axis; anticlockwise/counterclockwise
        is positive.

        For local_z / roll rotations, the intended motion is primarily Panda
        joint7. Joint7 is limited, not continuous, so requests beyond the
        remaining joint7 range are rejected before execution by default. Set
        allow_partial_rotation=True to execute the clipped safe portion.
        """
        from scipy.spatial.transform import Rotation as R

        def _axis_vector(axis_name: str) -> tuple[str, np.ndarray]:
            key = str(axis_name).strip().lower().replace("-", "_")
            aliases = {
                "x": "local_x",
                "y": "local_y",
                "z": "local_z",
                "roll": "local_z",
            }
            key = aliases.get(key, key)
            axes = {
                "local_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                "local_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
                "local_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
                "world_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                "world_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
                "world_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            }
            if key not in axes:
                raise ValueError(
                    "axis must be one of local_x/local_y/local_z/world_x/world_y/world_z"
                )
            return key, axes[key]

        def _joint7_range() -> tuple[float, float] | None:
            model = getattr(getattr(_env(), "_env", None), "sim", None)
            model = None if model is None else model.model
            if model is None:
                return None
            for name in ("robot0_joint7", f"{side}_joint7", "joint7"):
                try:
                    joint_id = model.joint_name2id(name)
                except Exception:
                    continue
                joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
                if joint_range.size >= 2:
                    return float(joint_range[0]), float(joint_range[1])
            return None

        dir_key = str(direction).strip().lower().replace("-", "_")
        if dir_key in ("cw", "clockwise"):
            signed_angle_deg = -abs(float(angle_deg))
        elif dir_key in (
            "ccw",
            "counter_clockwise",
            "counterclockwise",
            "anticlockwise",
        ):
            signed_angle_deg = abs(float(angle_deg))
        else:
            raise ValueError("direction must be clockwise or anticlockwise")

        axis_key, axis_vec = _axis_vector(axis)
        requested_signed_angle_deg = float(signed_angle_deg)

        obs0 = _env().get_arm_observation(side)
        start_pos = np.asarray(obs0["ee_pos"], dtype=np.float64)
        start_quat = np.asarray(obs0["ee_quat"], dtype=np.float64)
        start_rot = R.from_quat(start_quat)
        hold_gripper = (
            float(gripper)
            if gripper is not None
            else _commanded_gripper.get(
                side, float(np.asarray(obs0.get("gripper_pos", [0.5]))[0])
            )
        )
        joint_limit_info: dict[str, Any] | None = None
        limited_by_joint_limit = False

        if respect_joint_limits and axis_key == "local_z":
            joint_pos = np.asarray(obs0.get("joint_pos", []), dtype=np.float64)
            joint_range = _joint7_range()
            if joint_pos.size >= 7 and joint_range is not None:
                joint7 = float(joint_pos[-1])
                lower, upper = joint_range
                safe_lower = lower + max(0.0, float(joint_limit_margin_rad))
                safe_upper = upper - max(0.0, float(joint_limit_margin_rad))
                if signed_angle_deg < 0.0:
                    available_deg = max(0.0, np.rad2deg(joint7 - safe_lower))
                    clipped_angle_deg = -min(abs(signed_angle_deg), available_deg)
                else:
                    available_deg = max(0.0, np.rad2deg(safe_upper - joint7))
                    clipped_angle_deg = min(abs(signed_angle_deg), available_deg)

                limited_by_joint_limit = bool(
                    abs(clipped_angle_deg - signed_angle_deg) > 1e-6
                )
                signed_angle_deg = float(clipped_angle_deg)
                joint_limit_info = {
                    "joint": "joint7",
                    "joint_pos_rad": joint7,
                    "range_rad": [lower, upper],
                    "safe_range_rad": [safe_lower, safe_upper],
                    "available_angle_deg": float(available_deg),
                    "margin_rad": float(joint_limit_margin_rad),
                }

        if limited_by_joint_limit and not allow_partial_rotation:
            return {
                "success": False,
                "status": "JointLimit",
                "side": side,
                "axis": axis_key,
                "direction": dir_key,
                "angle_deg": 0.0,
                "requested_angle_deg": requested_signed_angle_deg,
                "clipped_angle_deg": float(signed_angle_deg),
                "limited_by_joint_limit": True,
                "allow_partial_rotation": False,
                "joint_limit": joint_limit_info,
                "n_steps": 0,
                "start_pos": start_pos.tolist(),
                "final_pos": start_pos.tolist(),
                "target_quat": start_quat.tolist(),
                "final_quat": start_quat.tolist(),
                "pos_drift_m": 0.0,
                "rot_error_deg": 0.0,
                "gripper": hold_gripper,
            }

        if abs(signed_angle_deg) < float(min_executable_angle_deg):
            return {
                "success": False,
                "status": "JointLimit" if limited_by_joint_limit else "NoOp",
                "side": side,
                "axis": axis_key,
                "direction": dir_key,
                "angle_deg": float(signed_angle_deg),
                "requested_angle_deg": requested_signed_angle_deg,
                "clipped_angle_deg": float(signed_angle_deg),
                "limited_by_joint_limit": limited_by_joint_limit,
                "allow_partial_rotation": allow_partial_rotation,
                "joint_limit": joint_limit_info,
                "n_steps": 0,
                "start_pos": start_pos.tolist(),
                "final_pos": start_pos.tolist(),
                "target_quat": start_quat.tolist(),
                "final_quat": start_quat.tolist(),
                "pos_drift_m": 0.0,
                "rot_error_deg": 0.0,
                "gripper": hold_gripper,
            }

        steps = (
            max(1, int(n_steps))
            if n_steps is not None
            else max(1, int(np.ceil(abs(signed_angle_deg) / 5.0)))
        )

        target_quat = start_quat
        for i in range(1, steps + 1):
            alpha = i / steps
            delta = R.from_rotvec(axis_vec * np.deg2rad(signed_angle_deg * alpha))
            if axis_key.startswith("world_"):
                target_quat = (delta * start_rot).as_quat()
            else:
                target_quat = (start_rot * delta).as_quat()
            _env().compute_eef_action(
                side=side,
                target_pos=start_pos,
                target_quat_xyzw=target_quat,
                gripper=hold_gripper,
            )
            _env().step()

        for _ in range(max(0, int(settle_steps))):
            _env().compute_eef_action(
                side=side,
                target_pos=start_pos,
                target_quat_xyzw=target_quat,
                gripper=hold_gripper,
            )
            _env().step()

        obs1 = _env().get_arm_observation(side)
        final_pos = np.asarray(obs1["ee_pos"], dtype=np.float64)
        final_quat = np.asarray(obs1["ee_quat"], dtype=np.float64)
        pos_drift = float(np.linalg.norm(final_pos - start_pos))
        rot_err_deg = float(
            (R.from_quat(target_quat).inv() * R.from_quat(final_quat)).magnitude()
            * 180.0
            / np.pi
        )
        success = pos_drift <= float(max_pos_drift_m) and rot_err_deg <= float(
            max_rot_error_deg
        )
        if limited_by_joint_limit:
            success = False

        return {
            "success": success,
            "status": "Success"
            if success
            else ("JointLimit" if limited_by_joint_limit else "Drifted"),
            "side": side,
            "axis": axis_key,
            "direction": dir_key,
            "angle_deg": float(signed_angle_deg),
            "requested_angle_deg": requested_signed_angle_deg,
            "clipped_angle_deg": float(signed_angle_deg),
            "limited_by_joint_limit": limited_by_joint_limit,
            "allow_partial_rotation": allow_partial_rotation,
            "joint_limit": joint_limit_info,
            "n_steps": steps,
            "start_pos": start_pos.tolist(),
            "final_pos": final_pos.tolist(),
            "target_quat": np.asarray(target_quat, dtype=np.float64).tolist(),
            "final_quat": final_quat.tolist(),
            "pos_drift_m": pos_drift,
            "rot_error_deg": rot_err_deg,
            "gripper": hold_gripper,
        }

    def nudge_brutal(
        side: str,
        delta_pos: list[float],
        n_steps: int = 10,
    ) -> NudgeResult:
        """Collision-blind OSC nudge. Bypasses cuRobo entirely.

        Unlike ``nudge``, this does NOT call the motion planner and therefore
        does NOT check for collisions at the start state or along the path.
        The arm will attempt the move regardless of whether it is currently
        in contact with or embedded in an object or surface.

        Use this ONLY when the arm is already in a colliding state (e.g.
        pressed against the sink wall after a grasp) and the planner refuses
        to plan because "Start state is colliding with world". A small upward
        nudge_brutal can extract the arm from the contact before handing back
        to freespace_move or nudge for the main motion.

        Do NOT use for large motions or in open space — no collision avoidance
        means the arm can push through geometry or destabilise grasped objects.

        Args:
            side: Arm side ("right" or "left").
            delta_pos: [dx, dy, dz] displacement in world frame, metres.
            n_steps: Number of OSC ticks to spread the motion over.
                     More steps = smoother but slower. Default 10 (~0.5 s).
        """
        obs = _env().get_arm_observation(side)
        cur_pos = np.array(obs["ee_pos"], dtype=np.float64)
        cur_quat = np.array(obs["ee_quat"], dtype=np.float64)
        target_pos = cur_pos + np.array(delta_pos, dtype=np.float64)
        hold_gripper = _commanded_gripper.get(
            side, float(np.asarray(obs.get("gripper_pos", [0.5]))[0])
        )

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            interp_pos = (1.0 - alpha) * cur_pos + alpha * target_pos
            _env().compute_eef_action(
                side=side,
                target_pos=interp_pos,
                target_quat_xyzw=cur_quat,
                gripper=hold_gripper,
            )
            _env().step()

        final_obs = _env().get_arm_observation(side)
        requested = float(np.linalg.norm(np.array(delta_pos)))
        actual = float(np.linalg.norm(np.array(final_obs["ee_pos"]) - cur_pos))
        success = requested < 1e-6 or actual > requested * 0.3

        return NudgeResult(
            success=success,
            final_pos=final_obs["ee_pos"],
            final_quat=final_obs["ee_quat"],
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def get_camera_image(camera: str) -> np.ndarray:
        return _env().render_rgb(camera)

    def get_camera_depth(camera: str) -> np.ndarray:
        return _env().render_depth(camera)

    def get_camera_intrinsics(camera: str) -> list[float]:
        return _env().get_camera_intrinsics(camera)

    def get_camera_extrinsics(camera: str) -> dict:
        return _env().get_camera_extrinsics(camera)

    def set_debug_markers(markers: list[dict]) -> dict:
        _env().set_debug_markers(markers)
        return {"success": True, "count": len(markers)}

    def clear_debug_markers() -> dict:
        _env().clear_debug_markers()
        return {"success": True}

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def get_task_info() -> dict:
        return _env().get_task_info()

    def get_close_blender_lid_debug_info() -> dict:
        return _env().get_close_blender_lid_debug_info()

    def get_close_fridge_debug_info() -> dict:
        return _env().get_close_fridge_debug_info()

    def get_task_description() -> str:
        """Return the cached natural-language task description from the env.

        Source: robocasa's ``get_ep_meta()["lang"]``, captured at reset time
        (see ``RoboCasaEnv.get_task_description``). Stable within an episode.
        """
        return _env().get_task_description()

    def reset_env() -> dict:
        """Reset env to the initial MuJoCo state (same scene, same objects).

        Restores the sim state snapshot captured after the first reset,
        giving an identical scene on every call. RoboCasa's internal RNG
        makes full teardown+recreate non-deterministic, so we use the
        sim state restore approach instead.
        """
        return _env().reset_to_initial()

    # ------------------------------------------------------------------
    # Planner world
    # ------------------------------------------------------------------

    def update_planner_world(
        planner: Any | None = None, exclude_body_prefixes: list[str] | None = None
    ) -> dict:
        """Load collision geometry from the sim into a cuRobo planner.

        When ``planner`` is omitted, use the branch's shared cuRobo planner so
        scripts can call this tool directly to refresh / clear the collision
        world.
        """
        if planner is None:
            planner = _get_planner()
        kwargs = {}
        if exclude_body_prefixes is not None:
            kwargs["exclude_body_prefixes"] = list(exclude_body_prefixes)
        collision_data = _env().get_collision_geoms(**kwargs)
        try:
            n = planner.update_world_from_geoms(collision_data)
        except Exception as e:
            print(
                f"[update_planner_world] WARNING: collision world update failed ({e}). "
                "Proceeding without collision avoidance."
            )
            n = 0
        return {
            "base_pos": collision_data["base_pos"],
            "base_quat_xyzw": collision_data["base_quat_xyzw"],
            "n_obstacles": n,
        }

    def enable_collision_avoid() -> dict:
        """Refresh the planner world with all scene obstacles enabled."""
        return update_planner_world()

    def disable_collision_avoid() -> dict:
        """Clear the planner world so subsequent plans ignore scene obstacles."""
        return update_planner_world(exclude_body_prefixes=[""])

    def refresh_planner_world(
        exclude_body_prefixes: list[str] | None = None,
    ) -> dict:
        """Script-friendly alias for ``update_planner_world``."""
        return update_planner_world(exclude_body_prefixes=exclude_body_prefixes)

    # ------------------------------------------------------------------
    # VLM query
    # ------------------------------------------------------------------

    def vlm_query(
        text: str,
        camera: str = "top",
        backend: str | None = None,
        image: Any = None,
        media: list[str] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str = "high",
        **kwargs: Any,
    ) -> str:
        """Query a vision-language model with camera image or provided image.

        Returns the model's text response (string).
        """
        e = _env()
        _backend = backend or vlm_backend
        images: list[np.ndarray] = []
        media_labels: list[str] = []

        # Resolve images from media list, direct image, or camera
        if media is not None:
            for src in media:
                if src.startswith("camera:"):
                    cam_name = src[len("camera:") :]
                    img = e.render_rgb(cam_name)
                    if img is not None:
                        images.append(img)
                        media_labels.append(src)
                else:
                    logger.warning(
                        "vlm_query: unsupported media source %r in direct mode", src
                    )
        elif image is not None:
            if isinstance(image, list):
                images = [np.asarray(im) for im in image]
                media_labels = [f"image_{i}" for i in range(len(images))]
            else:
                images = [np.asarray(image)]
                media_labels = ["image (provided directly)"]
            # Override labels if caller provided image_labels
            if "image_labels" in kwargs and kwargs["image_labels"]:
                media_labels = list(kwargs["image_labels"])
        elif camera == "all":
            cam_names = (
                e._profile.camera_names
                if e._profile and hasattr(e._profile, "camera_names")
                else list(e._camera_map.keys())
            )
            for cam in cam_names:
                try:
                    img = e.render_rgb(cam)
                    if img is not None:
                        images.append(img)
                        media_labels.append(f"camera:{cam}")
                except Exception:
                    pass
        else:
            img = e.render_rgb(camera)
            if img is not None:
                images = [img]
                media_labels = [f"camera:{camera}"]

        if not images:
            raise RuntimeError(f"No images available (camera={camera})")

        # Prepend image labels for multi-image queries
        prompt = text
        if len(images) > 1 and media_labels:
            label_lines = ", ".join(
                f"Image {i + 1}: {lbl}" for i, lbl in enumerate(media_labels)
            )
            prompt = f"[Images: {label_lines}]\n{text}"

        # Delegate to the shared transport — single dispatch point across
        # the agent tool, this script adapter, and any future caller.
        # Any backend-specific kwargs (api_key, key_index, reasoning_effort,
        # thinking_budget, api_url, …) flow through **kwargs untouched.
        from enpire.env.forge.cap.agent.tools.vlm import query as _vlm_query

        response = _vlm_query(
            backend=_backend,
            text=prompt,
            images=images,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        logger.info(
            "vlm_query [%s] images=%d response=%s",
            _backend,
            len(images),
            response[:120],
        )
        return response

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_object(
        query: str,
        camera: str = "top",
        backend: str = "oracle",
        max_retries: int = 3,
    ) -> list[Detection3D]:
        """Detect objects — oracle mode uses get_task_info, no external server needed.

        In oracle mode, fuzzy-matches *query* against object position keys
        in the env's task info (RoboCasa provides ``*_pos`` keys for all scene
        objects in the observation dict).
        """
        if backend == "oracle":
            info = _env().get_task_info()
            # RoboCasa task info includes *_pos keys for scene objects
            query_lower = query.lower()
            matches: list[tuple[str, list[float]]] = []
            for k, v in info.items():
                if not k.endswith("_pos"):
                    continue
                if k.startswith("robot0"):
                    continue
                obj_name = k[: -len("_pos")]
                # Check object name from _name key or fuzzy match
                display_name = info.get(f"{obj_name}_name", obj_name)
                if (
                    query_lower in obj_name.lower()
                    or obj_name.lower() in query_lower
                    or query_lower in display_name.lower()
                    or display_name.lower() in query_lower
                ):
                    pos = v if isinstance(v, list) else list(v)
                    matches.append((display_name or obj_name, pos))

            if not matches:
                available = [
                    k[: -len("_pos")]
                    for k in info
                    if k.endswith("_pos") and not k.startswith("robot0")
                ]
                raise RuntimeError(
                    f"No object matching '{query}'. Available: {', '.join(available)}"
                )

            detections: list[Detection3D] = []
            for name, pos_3d in matches:
                detections.append(
                    Detection3D(
                        label=name,
                        score=1.0,
                        box_2d=[],
                        position_3d=[round(float(x), 4) for x in pos_3d],
                    )
                )
            return detections

        # Non-oracle backends would require external servers
        raise NotImplementedError(
            f"Detection backend '{backend}' not yet supported in direct mode. "
            "Use backend='oracle' for RoboCasa ground-truth detection."
        )

    # ------------------------------------------------------------------
    # SAM3 segmentation + depth-based 3D detection (no oracle)
    # ------------------------------------------------------------------

    _MIN_DEPTH_M = 0.1
    _MIN_VALID_POINTS = 30

    def segment_object(
        query: str,
        camera: str = "top",
        score_thresh: float = 0.1,
    ) -> dict:
        """Segment an object using SAM3 text-prompted segmentation.

        Returns dict with keys: mask (H,W int32), bbox_xywh, score, mask_area.
        Raises RuntimeError on failure.
        """
        from enpire.env.forge.cap.env.base.skill_library import segment_object_sam3

        e = _env()
        rgb = e.render_rgb(camera)
        mask, seg_score = segment_object_sam3(rgb, query, _sam3_url)
        mask_area = int((mask > 0).sum())
        if mask_area == 0:
            raise RuntimeError(f"SAM3 returned empty mask for {query!r}")

        ys, xs = np.where(mask > 0)
        bbox_xywh = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min()),
            int(ys.max() - ys.min()),
        ]
        return {
            "mask": mask,
            "bbox_xywh": bbox_xywh,
            "score": seg_score,
            "mask_area": mask_area,
        }

    def segment_object_all(
        query: str,
        camera: str = "top",
        score_thresh: float = 0.1,
        max_results: int = 25,
    ) -> list[dict]:
        """Segment all SAM3 candidates for a text prompt.

        Returns a list of dicts with keys: mask (H,W int32), bbox_xywh, score,
        mask_area. This keeps direct SAM3 service access behind the execution
        namespace for scripts that need more than the top mask.
        """
        import base64
        import json
        import urllib.error
        import urllib.request

        e = _env()
        rgb = e.render_rgb(camera)
        buf = io.BytesIO()
        np.save(buf, np.asarray(rgb))
        payload = json.dumps(
            {
                "text": query,
                "image_b64": base64.b64encode(buf.getvalue()).decode(),
                "score_threshold": float(score_thresh),
                "return_all": True,
                "max_results": int(max_results),
            }
        ).encode()
        req = urllib.request.Request(
            f"{_sam3_url.rstrip('/')}/segment",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=90)
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            raise RuntimeError(f"SAM3 /segment failed ({err.code}): {detail}") from None

        data = json.loads(resp.read())
        masks_payload = data.get("masks") or [
            {
                "mask_b64": data["mask_b64"],
                "bbox_xywh": data.get("bbox_xywh", []),
                "score": data.get("score", data.get("confidence", 0.0)),
            }
        ]
        results = []
        for item in masks_payload:
            score = float(item.get("score", 0.0))
            if score < score_thresh:
                continue
            mask = np.load(io.BytesIO(base64.b64decode(item["mask_b64"]))).astype(
                np.int32
            )
            mask_area = int((mask > 0).sum())
            if mask_area == 0:
                continue
            bbox = item.get("bbox_xywh", [])
            if not bbox:
                ys, xs = np.where(mask > 0)
                bbox = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min()),
                    int(ys.max() - ys.min()),
                ]
            results.append(
                {
                    "mask": mask,
                    "bbox_xywh": [int(x) for x in bbox],
                    "score": score,
                    "mask_area": mask_area,
                }
            )
        return results

    def detect_objects_oneshot(
        query: str | list[str],
        camera: str = "top",
    ) -> dict[str, list[Detection3D]]:
        """One-shot 3D position via SAM3 mask + depth back-projection.

        Captures one shared RGB-D snapshot and segments each query with SAM3
        independently. Returns dict mapping each query to its Detection3D
        results (position_3d from median centroid of masked depth).
        """
        from enpire.env.forge.cap.env.base.skill_library import segment_object_sam3

        queries = [query] if isinstance(query, str) else list(query)
        e = _env()
        rgb = e.render_rgb(camera)
        depth = e.render_depth(camera)
        fx, fy, cx, cy = e.get_camera_intrinsics(camera)
        extr = e.get_camera_extrinsics(camera)
        R_mat = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extr["position"], dtype=np.float64)
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R_mat @ np.diag([-1.0, -1.0, 1.0])
        T_cam_world[:3, 3] = t

        results: dict[str, list[Detection3D]] = {q: [] for q in queries}
        for q in queries:
            try:
                mask, sam_score = segment_object_sam3(rgb, q, _sam3_url)
            except Exception as exc:
                logger.warning("[detect_oneshot] SAM3 failed for %r: %s", q, exc)
                continue
            valid = (mask > 0) & np.isfinite(depth) & (depth > _MIN_DEPTH_M)
            n_valid = int(valid.sum())
            if n_valid < _MIN_VALID_POINTS:
                logger.warning(
                    "[detect_oneshot] %r: only %d valid depth points (need %d)",
                    q,
                    n_valid,
                    _MIN_VALID_POINTS,
                )
                continue
            vs, us = np.where(valid)
            zs = depth[valid].astype(np.float64)
            xs = (us.astype(np.float64) - cx) * zs / fx
            ys = (vs.astype(np.float64) - cy) * zs / fy
            pts_cam = np.stack([xs, ys, zs], axis=1)
            centroid_cam = np.median(pts_cam, axis=0)
            centroid_world = T_cam_world[:3, :3] @ centroid_cam + T_cam_world[:3, 3]

            ys_mask, xs_mask = np.where(mask > 0)
            bbox = [
                int(xs_mask.min()),
                int(ys_mask.min()),
                int(xs_mask.max()),
                int(ys_mask.max()),
            ]
            det = Detection3D(
                label=q,
                score=round(sam_score, 4),
                box_2d=bbox,
                position_3d=[round(float(v), 4) for v in centroid_world],
            )
            results[q] = [det]

            from enpire.env.forge.cap.agent.tools._artifact_log import log_detection, log_mask

            log_mask(rgb, mask, query=q, tag=f"oneshot_mask_{camera}")
            log_detection(rgb, [det], tag=f"oneshot_det_{camera}")

        return results

    # ------------------------------------------------------------------
    # AnyGrasp
    # ------------------------------------------------------------------

    def sample_grasp_pose_anygrasp(
        object_name: str,
        camera: str = "top",
        max_grasps: int = 10,
        top_down_only: bool = False,
        vertical_threshold: float = 0.8,
        object_input_mode: str = "segmented_object_cloud",
        tcp_offset_z_m: float = 0.0,
        disable_planner_z_clipping: bool = False,
    ) -> list:
        """Plan grasp poses using SAM3 + AnyGrasp (remote servers).

        Returns list of GraspCandidate(position, rpy, score, width).
        """
        from enpire.env.forge.cap.env.base.skill_library import sample_grasp_anygrasp

        e = _env()
        rgb = e.render_rgb(camera)
        depth = e.render_depth(camera)
        fx, fy, cx, cy = e.get_camera_intrinsics(camera)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        extr = e.get_camera_extrinsics(camera)
        R_mat = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extr["position"], dtype=np.float64)
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R_mat @ np.diag([-1.0, -1.0, 1.0])
        T_cam_world[:3, 3] = t

        min_z = (
            None
            if disable_planner_z_clipping
            else float(os.environ.get("ANYGRASP_MIN_PLANNER_Z_M", "0.80"))
        )

        candidates, viz = sample_grasp_anygrasp(
            rgb=rgb,
            depth=depth,
            K=K,
            T_cam_world=T_cam_world,
            object_name=object_name,
            max_grasps=max_grasps,
            top_down_only=top_down_only,
            vertical_threshold=vertical_threshold,
            object_input_mode=object_input_mode,
            sam3_url=_sam3_url,
            anygrasp_url=_anygrasp_url,
            tcp_offset_z_m=tcp_offset_z_m,
            min_planner_z_m=min_z,
        )

        from enpire.env.forge.cap.agent.tools._artifact_log import log_grasp, log_mask

        seg_mask = viz.get("mask")
        if seg_mask is not None:
            log_mask(rgb, seg_mask, query=object_name, tag="anygrasp_mask")
        overlay_jpeg = viz.get("overlay_jpeg")
        if overlay_jpeg:
            try:
                from PIL import Image as _PILImage

                overlay_img = np.array(
                    _PILImage.open(io.BytesIO(overlay_jpeg)).convert("RGB")
                )
                log_grasp(
                    overlay_img, None, candidates, query=object_name, tag="anygrasp"
                )
            except Exception:
                log_grasp(rgb, seg_mask, candidates, query=object_name, tag="anygrasp")
        else:
            log_grasp(rgb, seg_mask, candidates, query=object_name, tag="anygrasp")

        return candidates

    # ------------------------------------------------------------------
    # Navigation (MPPI)
    # ------------------------------------------------------------------

    _mppi_holder: dict[str, Any] = {}

    def _get_mppi_client():
        if "client" not in _mppi_holder:
            if _MPPI_PORT == 0:
                logger.warning("CAP_MPPI_PORT not set — move_base unavailable")
                return None
            from enpire.env.forge.cap.env.robocasa.mppi_client import MPPINavigationClient

            logger.info("Connecting to MPPI planner at %s:%s", _MPPI_HOST, _MPPI_PORT)
            _mppi_holder["client"] = MPPINavigationClient(
                host=_MPPI_HOST,
                port=_MPPI_PORT,
                start_server=False,
            )
        return _mppi_holder.get("client")

    def get_base_state() -> dict:
        """Return current base position and yaw angle.

        Returns:
            dict with keys:
                pos: list[float] — [x, y, z] base position in world frame
                yaw: float — heading angle in radians
        """
        from scipy.spatial.transform import Rotation as R

        e = _env()
        bp = e.get_base_pose()
        pos = np.asarray(bp.get("base_pos", np.zeros(3)))
        quat = np.asarray(bp.get("base_quat", [0, 0, 0, 1]))
        yaw = float(R.from_quat(quat).as_euler("xyz")[2])
        return {"pos": pos.tolist(), "yaw": yaw}

    def execute_base_trajectory(actions: list) -> dict:
        """Execute base velocity commands through the direct env wrapper.

        actions: list of [v_fwd, v_side, omega] commands.
        """
        e = _env()
        actions_np = np.asarray(actions, dtype=np.float64)
        for action in actions_np:
            sim_per_step = int(action[3]) if len(action) >= 4 else 6
            e.command_base_action(
                float(action[0]),
                float(action[1]),
                float(action[2]),
                n_steps=sim_per_step,
            )

        bp = e.get_base_pose()
        return {
            "ok": True,
            "n_steps": int(len(actions_np)),
            "base_pos": np.asarray(bp.get("base_pos", np.zeros(3))).tolist(),
            "base_quat_xyzw": np.asarray(bp.get("base_quat", [0, 0, 0, 1.0])).tolist(),
        }

    def move_base(
        target_x: float,
        target_y: float,
        target_yaw: float | None = None,
    ) -> dict:
        """Navigate the mobile base to a target position with collision avoidance.

        Uses a remote MPPI planner to compute a collision-free trajectory,
        then executes it by stepping the sim with base velocity commands.

        Args:
            target_x: Target X position in world frame (meters).
            target_y: Target Y position in world frame (meters).
            target_yaw: Target heading in radians.  None = keep current heading.

        Returns:
            dict with: status, n_steps, final_pos_error, final_yaw_error
        """
        from scipy.spatial.transform import Rotation as R

        e = _env()
        mppi = _get_mppi_client()
        if mppi is None:
            return {
                "status": "Error",
                "reason": "MPPI planner not configured (set CAP_MPPI_PORT)",
            }

        bp = e.get_base_pose()
        current_pos = np.asarray(bp.get("base_pos", np.zeros(3)))
        current_quat = np.asarray(bp.get("base_quat", [0, 0, 0, 1]))
        current_yaw = float(R.from_quat(current_quat).as_euler("xyz")[2])

        if target_yaw is None:
            target_yaw = current_yaw

        collision_data = e.get_collision_geoms(max_dist=10.0)
        floor_bounds = e.get_floor_bounds()
        world_result = mppi.update_world(collision_data, floor_bounds)
        logger.info("MPPI world updated: %s", world_result)

        plan_result = mppi.plan_trajectory(
            current_pos=current_pos[:2],
            current_yaw=current_yaw,
            target_pos=np.array([target_x, target_y]),
            target_yaw=target_yaw,
        )

        status = plan_result.get("status", "Error")
        if status == "Error":
            return plan_result

        actions = np.asarray(plan_result["actions"])
        for i in range(len(actions)):
            e.command_base_action(
                float(actions[i, 0]),
                float(actions[i, 1]),
                float(actions[i, 2]),
            )

        final_bp = e.get_base_pose()
        final_pos = np.asarray(final_bp.get("base_pos", np.zeros(3)))
        final_quat = np.asarray(final_bp.get("base_quat", [0, 0, 0, 1]))
        final_yaw = float(R.from_quat(final_quat).as_euler("xyz")[2])

        pos_err = float(np.linalg.norm(final_pos[:2] - np.array([target_x, target_y])))
        yaw_err = abs(
            np.arctan2(
                np.sin(final_yaw - target_yaw),
                np.cos(final_yaw - target_yaw),
            )
        )

        return {
            "status": status,
            "n_steps": len(actions),
            "final_pos_error": round(pos_err, 4),
            "final_yaw_error": round(yaw_err, 4),
        }

    # ------------------------------------------------------------------
    # External policy rollout (GR00T, …)
    # ------------------------------------------------------------------

    def use_policy_output(**overrides: Any) -> dict:
        """Run a closed-loop VLA policy against this env until success/timeout.

        Pass model choices as kwargs: ``model``, ``max_episode_steps``,
        ``replan_horizon``, ``endpoint``. Returns ``{success, steps,
        task_description, model, profiling}``.
        """
        if policy_runner is None:
            raise RuntimeError(
                "use_policy_output requires a policy_runner. "
                "Pass policy_runner=make_policy_runner(cfg) to make_namespace()."
            )
        td = (
            overrides.pop("task_description", "").strip()
            or _env().get_task_description()
        )
        if not td:
            raise RuntimeError(
                "Task description is empty. Ensure get_ep_meta()['lang'] is populated."
            )
        return policy_runner(_env(), td, **overrides)

    # ------------------------------------------------------------------
    # Shared vision-script utilities
    # ------------------------------------------------------------------

    def display_rpy_to_quat(rpy_deg: list[float]) -> np.ndarray:
        """Convert AnyGrasp display RPY (degrees) to quaternion xyzw.

        Same conversion used internally by ``select_best_grasp``.
        """
        from scipy.spatial.transform import Rotation as Rot

        roll, pitch, yaw = [float(x) for x in rpy_deg]
        return Rot.from_euler(
            "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
        ).as_quat()

    # ------------------------------------------------------------------
    # Assemble namespace
    # ------------------------------------------------------------------

    namespace: dict[str, Any] = {
        # State
        "get_robot_state": get_robot_state,
        "get_gripper_info": get_gripper_info,
        # Motion — peer planner tools (pick one explicitly per call)
        "move_with_curobo": move_with_curobo,
        "move_with_pyroki": move_with_pyroki,
        # Back-compat alias dispatching via CAP_PLANNER_BACKEND env var
        "freespace_move": freespace_move,
        "select_best_grasp": select_best_grasp,
        "select_best_grasp_two_step": select_best_grasp_two_step,
        "move_joint_keypoints": move_joint_keypoints,
        "set_gripper": set_gripper,
        "open_gripper": open_gripper,
        "close_gripper": close_gripper,
        "go_home": go_home,
        "nudge": nudge,
        "rotate_gripper_in_place": rotate_gripper_in_place,
        "nudge_brutal": nudge_brutal,
        # Camera
        "get_camera_image": get_camera_image,
        "get_camera_depth": get_camera_depth,
        "get_camera_intrinsics": get_camera_intrinsics,
        "get_camera_extrinsics": get_camera_extrinsics,
        "set_debug_markers": set_debug_markers,
        "clear_debug_markers": clear_debug_markers,
        # Task — `get_task_info`, `_debug_info` helpers, and `reset_env` are
        # exported here so the MCP server (env_worker.py + cap/mcp/tools/*.py)
        # and the vocab-builder agent can reach them via RPC. They are NOT
        # safe to expose to eval-time `code.py` though: `get_task_info` leaks
        # oracle obj_pos / obj_name / success, the `_debug_info` helpers leak
        # per-fixture oracle state, and `reset_env` enables k-shot policy
        # retry on the same scene (rate-inflation cheese). `run_script.py`
        # swaps these for raising stubs when CAP_DISABLE_TASK_INFO_IN_SCRIPT=1
        # so eval code can't reach them, while keeping the MCP path open.
        "get_task_info": get_task_info,
        "get_close_blender_lid_debug_info": get_close_blender_lid_debug_info,
        "get_close_fridge_debug_info": get_close_fridge_debug_info,
        "get_task_description": get_task_description,
        "reset_env": reset_env,
        # Planner
        "update_planner_world": update_planner_world,
        "refresh_planner_world": refresh_planner_world,
        "enable_collision_avoid": enable_collision_avoid,
        "disable_collision_avoid": disable_collision_avoid,
        # Perception
        "vlm_query": vlm_query,
        "detect_object": detect_object,
        "segment_object": segment_object,
        "segment_object_all": segment_object_all,
        "detect_objects_oneshot": detect_objects_oneshot,
        "sample_grasp_pose_anygrasp": sample_grasp_pose_anygrasp,
        # External policy rollout
        "use_policy_output": use_policy_output,
        # Navigation (MPPI)
        "get_base_state": get_base_state,
        "execute_base_trajectory": execute_base_trajectory,
        "move_base": move_base,
        # Vision script utilities
        "display_rpy_to_quat": display_rpy_to_quat,
        # Convenience: numpy available in agent code
        "np": np,
        "numpy": np,
    }

    # Oracle API — opt-in via oracle_api: true in experiment YAML
    if cfg is not None and getattr(cfg, "oracle_api", False):
        namespace["get_oracle_targets"] = env.get_oracle_targets

    # Eval-script role: drop the cheese surfaces (oracle leaks + k-shot retry).
    # See docstring for the role semantics. The function closures are still
    # present in the enclosing scope so the MCP/worker path can reach them
    # via RPC; eval-time `code.py` simply cannot resolve them as names.
    if runtime_role == "script":
        for _cheese_key in (
            "get_task_info",
            "get_close_blender_lid_debug_info",
            "get_close_fridge_debug_info",
            "reset_env",
        ):
            namespace.pop(_cheese_key, None)
        # `get_oracle_targets` already gated by cfg.oracle_api: true (experiment
        # YAML), but defensively drop it for canonical eval scripts too.
        namespace.pop("get_oracle_targets", None)

    return namespace
