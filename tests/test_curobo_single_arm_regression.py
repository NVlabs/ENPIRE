from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from experimental.motion_planner_curobo import YamMotionPlannerCurobo


class _CollisionFreeValidator:
    def check_collision(self, _left: np.ndarray, _right: np.ndarray) -> bool:
        return False


def test_validate_trajectory_prefers_fixed_inactive_arm_for_single_arm_move() -> None:
    planner = YamMotionPlannerCurobo.__new__(YamMotionPlannerCurobo)
    planner._validator = _CollisionFreeValidator()

    left_positions = np.array([[0.0] * 6, [0.1] * 6], dtype=np.float64)
    right_positions = np.array([[0.0] * 6, [0.2] * 6], dtype=np.float64)
    current_left = np.zeros(6, dtype=np.float64)
    current_right = np.zeros(6, dtype=np.float64)

    out_left, out_right, error = YamMotionPlannerCurobo._validate_trajectory(
        planner,
        side="left",
        left_positions=left_positions,
        right_positions=right_positions,
        current_left_jp=current_left,
        current_right_jp=current_right,
    )

    assert error is None
    np.testing.assert_allclose(out_left, left_positions)
    np.testing.assert_allclose(out_right, np.zeros_like(right_positions))


def test_setup_motion_gen_can_disable_collision_checking() -> None:
    planner = YamMotionPlannerCurobo.__new__(YamMotionPlannerCurobo)
    planner._collision_checking = False
    planner._enable_finetune_trajopt = False
    planner._position_threshold = 0.005
    planner._rotation_threshold = 0.05
    planner._cspace_threshold = 0.05
    planner._tensor_args = object()
    planner._CollisionCheckerType = SimpleNamespace(PRIMITIVE="primitive")
    planner._batch_planner_capacity = None
    planner._solver_preset = {
        "motion_gen": {
            "num_ik_seeds": 8,
            "num_graph_seeds": 1,
            "num_trajopt_seeds": 2,
            "trajopt_tsteps": 32,
            "ik_opt_iters": 96,
            "grad_trajopt_iters": 96,
        },
        "plan": {
            "enable_graph_attempt": 1,
            "max_attempts": 2,
            "timeout": 2.5,
            "time_dilation_factor_single": 0.5,
            "time_dilation_factor_batch": None,
        },
    }
    planner._load_robot_cfg = lambda: {"robot_cfg": "ok"}
    planner._build_world_cfg = lambda: (_ for _ in ()).throw(
        AssertionError(
            "world cfg should not be built when collision checking is disabled"
        )
    )

    captured: dict[str, object] = {}

    class _FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _FakeMotionGen:
        def __init__(self, _cfg) -> None:
            self.kinematics = SimpleNamespace(
                kinematics_config=SimpleNamespace(
                    joint_limits=SimpleNamespace(
                        position=_FakeTensor(np.zeros((2, 12), dtype=np.float64))
                    )
                )
            )

        def warmup(
            self, enable_graph: bool = False, warmup_js_trajopt: bool = False
        ) -> None:
            captured["warmup"] = (enable_graph, warmup_js_trajopt)

    def _fake_load_from_robot_config(robot_cfg, world_cfg, tensor_args, **kwargs):
        captured["robot_cfg"] = robot_cfg
        captured["world_cfg"] = world_cfg
        captured["tensor_args"] = tensor_args
        captured["kwargs"] = kwargs
        return "motion-gen-cfg"

    planner._MotionGenConfig = SimpleNamespace(
        load_from_robot_config=_fake_load_from_robot_config
    )
    planner._MotionGen = _FakeMotionGen
    planner._MotionGenPlanConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    YamMotionPlannerCurobo._setup_motion_gen(planner)

    assert captured["world_cfg"] is None
    assert captured["kwargs"]["collision_checker_type"] is None
    assert captured["kwargs"]["self_collision_check"] is False
    assert captured["kwargs"]["self_collision_opt"] is False
    assert captured["kwargs"]["collision_activation_distance"] is None


def test_portal_motion_planner_server_passes_collision_checking_to_yam_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experimental.motion_planner_curobo as motion_planner_curobo
    from experimental.portal_motion_planner import (
        PortalMotionPlannerConfig,
        PortalMotionPlannerServer,
    )

    captured: dict[str, object] = {}

    class _FakePlanner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        motion_planner_curobo,
        "YamMotionPlannerCurobo",
        _FakePlanner,
    )

    server = PortalMotionPlannerServer.__new__(PortalMotionPlannerServer)
    server._planners = {}
    server.config = PortalMotionPlannerConfig(
        backend="curobo",
        port=0,
        collision_checking=False,
    )

    planner = server._get_planner("yam")

    assert isinstance(planner, _FakePlanner)
    assert captured["collision_checking"] is False
    assert captured["validate_with_mujoco"] is False
