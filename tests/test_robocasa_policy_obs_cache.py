from __future__ import annotations

import importlib
import sys
import types
import unittest


def _install_fake_core_modules() -> None:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = "float32"
    fake_numpy.ndarray = object
    sys.modules["numpy"] = fake_numpy

    fake_mujoco = types.ModuleType("mujoco")
    fake_mujoco.Renderer = object
    fake_mujoco.MjvOption = type(
        "MjvOption",
        (),
        {"__init__": lambda self: setattr(self, "geomgroup", {0: 0})},
    )
    sys.modules["mujoco"] = fake_mujoco


def _install_fake_gym_wrapper() -> None:
    gym_wrapper = types.ModuleType("robocasa.wrappers.gym_wrapper")

    class FakeKeyConverter:
        @classmethod
        def get_camera_config(cls):
            return (
                ["video.robot0_agentview_left"],
                ["robot0_agentview_left"],
                256,
                256,
            )

        @classmethod
        def map_obs(cls, input_obs):
            return {
                "hand.gripper_qpos": input_obs["robot0_gripper_qpos"],
                "body.base_position": input_obs["robot0_base_pos"],
                "body.base_rotation": input_obs["robot0_base_quat"],
                "body.end_effector_position_relative": input_obs[
                    "robot0_base_to_eef_pos"
                ],
                "body.end_effector_rotation_relative": input_obs[
                    "robot0_base_to_eef_quat"
                ],
            }

    class FakeRoboCasaGymEnv:
        def _create_obs_and_action_space(self):
            return None

        def get_observation(self, raw_obs):
            basic_obs = self.get_basic_observation(raw_obs)
            obs = {}
            temp_obs = self.key_converter.map_obs(basic_obs)
            for k, v in temp_obs.items():
                if k.startswith("hand.") or k.startswith("body."):
                    obs["state." + k[5:]] = v
                else:
                    raise ValueError(f"Unknown key: {k}")
            mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
            for mapped_name, camera_name in zip(mapped_names, camera_names):
                obs[mapped_name] = basic_obs[camera_name + "_image"]
            obs["annotation.human.task_description"] = basic_obs["language"]
            return obs

        def step(self, action_dict):
            raise AssertionError("shared view must override step()")

    gym_wrapper.PandaOmronKeyConverter = FakeKeyConverter
    gym_wrapper.RoboCasaGymEnv = FakeRoboCasaGymEnv

    sys.modules["robocasa"] = types.ModuleType("robocasa")
    sys.modules["robocasa.wrappers"] = types.ModuleType("robocasa.wrappers")
    sys.modules["robocasa.wrappers.gym_wrapper"] = gym_wrapper


def _load_env_module():
    _install_fake_core_modules()
    sys.modules.pop("cap.env.robocasa.env", None)
    module = importlib.import_module("cap.env.robocasa.env")
    return importlib.reload(module)


class _FakeInnerEnv:
    def get_ep_meta(self):
        return {"lang": "task description"}


class _FakeCapEnv:
    def __init__(self):
        self._env = _FakeInnerEnv()
        self.CAMERA_WIDTH = 256
        self.CAMERA_HEIGHT = 256
        self._camera_map = {}
        self._recorder = None
        self.render_calls = []
        self.policy_obs_calls = []
        self.policy_basic_obs_calls = []
        self.policy_step_calls = []

    def render_rgb_by_mujoco_name(self, camera_name: str):
        self.render_calls.append(camera_name)
        raise AssertionError("shared view should not render directly")

    def get_policy_observation(
        self,
        *,
        key_converter,
        camera_names,
        render_obs_key,
        raw_obs=None,
        force_refresh=False,
    ):
        self.policy_obs_calls.append(
            {
                "key_converter": key_converter,
                "camera_names": tuple(camera_names),
                "render_obs_key": render_obs_key,
                "raw_obs": raw_obs,
                "force_refresh": force_refresh,
            }
        )
        return {
            "video.robot0_agentview_left": "cached-image",
            "annotation.human.task_description": "task description",
        }

    def get_policy_basic_observation(
        self,
        *,
        camera_names,
        render_obs_key,
        raw_obs=None,
        force_refresh=False,
    ):
        self.policy_basic_obs_calls.append(
            {
                "camera_names": tuple(camera_names),
                "render_obs_key": render_obs_key,
                "raw_obs": raw_obs,
                "force_refresh": force_refresh,
            }
        )
        return {
            "robot0_agentview_left_image": "cached-image",
            "language": "task description",
        }

    def step_policy_action(
        self,
        *,
        action_dict,
        key_converter,
        camera_names,
        render_obs_key,
    ):
        self.policy_step_calls.append(
            {
                "action_dict": action_dict,
                "key_converter": key_converter,
                "camera_names": tuple(camera_names),
                "render_obs_key": render_obs_key,
            }
        )
        return (
            {
                "video.robot0_agentview_left": "step-image",
                "annotation.human.task_description": "task description",
            },
            1.0,
            False,
            False,
            {"success": True},
        )


class SharedViewPolicyCacheTests(unittest.TestCase):
    def test_get_observation_delegates_to_cap_env_policy_observation(self):
        _install_fake_gym_wrapper()
        env_module = _load_env_module()
        view_cls = env_module._build_shared_gym_view_class()
        cap_env = _FakeCapEnv()
        view = view_cls(cap_env)

        raw_obs = {"robot0_gripper_qpos": "ignored"}
        obs = view.get_observation(raw_obs)

        self.assertEqual(cap_env.render_calls, [])
        self.assertEqual(len(cap_env.policy_obs_calls), 1)
        self.assertIs(cap_env.policy_obs_calls[0]["raw_obs"], raw_obs)
        self.assertEqual(len(cap_env.policy_basic_obs_calls), 1)
        self.assertEqual(obs["video.robot0_agentview_left"], "cached-image")

    def test_step_delegates_to_cap_env_policy_step(self):
        _install_fake_gym_wrapper()
        env_module = _load_env_module()
        view_cls = env_module._build_shared_gym_view_class()
        cap_env = _FakeCapEnv()
        view = view_cls(cap_env)

        action = {"action.gripper_close": 1.0}
        obs, reward, done, truncated, info = view.step(action)

        self.assertEqual(cap_env.render_calls, [])
        self.assertEqual(len(cap_env.policy_step_calls), 1)
        self.assertIs(cap_env.policy_step_calls[0]["action_dict"], action)
        self.assertEqual(len(cap_env.policy_basic_obs_calls), 1)
        self.assertEqual(obs["video.robot0_agentview_left"], "step-image")
        self.assertEqual((reward, done, truncated), (1.0, False, False))
        self.assertEqual(info, {"success": True})


if __name__ == "__main__":
    unittest.main()
