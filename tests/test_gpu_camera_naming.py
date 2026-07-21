from types import SimpleNamespace

from enpire.env.forge.hardware.identify_forge_rl_devices import REALSENSE_ROLES
from enpire.policy.rl.config import load_yaml_defaults
from enpire.env.forge.robot.models.station.paths import needs_optical_flip
from enpire.env.forge.robot.yam._base_yam_env import _BaseYamEnv


def test_realsense_registration_uses_explicit_third_view_and_wrist_names():
    assert REALSENSE_ROLES == ["video_left_third", "video_right", "video_left"]


def test_gpu_insertion_config_names_third_view_and_wrist_separately():
    cfg = load_yaml_defaults("tmux/realworld_rl/tasks_config/gpu_insertion/gpu_insertion.yaml")

    assert cfg.enabled_camera_names == ("top", "left_third", "left_wrist")
    assert cfg.gpu_slot_hover_aux_camera == "left_third"


def test_left_third_and_left_wrist_resolve_to_different_model_frames():
    env = object.__new__(_BaseYamEnv)
    env._spec = SimpleNamespace(
        cameras=[
            SimpleNamespace(name="top_camera_d405"),
            SimpleNamespace(name="top_camera_left_d435"),
            SimpleNamespace(name="left_camera_d405"),
            SimpleNamespace(name="right_camera_d405"),
        ]
    )

    assert env._resolve_model_camera_name("left_third") == "top_camera_left_d435"
    assert env._resolve_model_camera_name("left_wrist") == "left_camera_d405"
    assert env._resolve_model_camera_name("left") == "left_camera_d405"


def test_left_third_is_calibrated_fixed_camera_frame_not_wrist_frame():
    assert needs_optical_flip("left_third") is False
    assert needs_optical_flip("left") is True
    assert needs_optical_flip("left_wrist") is True
