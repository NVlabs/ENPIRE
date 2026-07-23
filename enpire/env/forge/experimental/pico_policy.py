# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import numpy as np
import portal
from scipy.spatial.transform import Rotation as R

from enpire.env.forge.robot.constants import PICO_PORT
from enpire.policy.legacy import Action, Info, Observation, Options, Policy

TRIGGER_PRESS_THRESHOLD = 0.5


class PicoClient:
    def __init__(self, host: str = "localhost", port: int = PICO_PORT):
        self.client = portal.Client(f"{host}:{port}")

    def get_info(self):
        return self.client.get_info().result()


class PicoPolicy(Policy):
    """End-effector control policy driven by Pico controller via Portal."""

    def __init__(
        self, host: str = "localhost", port: int = PICO_PORT, env=None, standalone: bool = False
    ):
        self._client = PicoClient(host, port)
        self._lock = threading.Lock()
        self._env = env
        assert self._env is not None, "Environment must be provided"
        self._standalone = standalone
        self._control_mode = (
            self._env.unwrapped.control_mode
        )  # "joint_position" or "cartesian_position"

        self._latest_left_wrist_matrix = np.eye(4, dtype=np.float64)
        self._latest_right_wrist_matrix = np.eye(4, dtype=np.float64)
        self._latest_left_trigger = 0.0
        self._latest_right_trigger = 0.0
        self._latest_left_gripper = 1.0
        self._latest_right_gripper = 1.0
        self._latest_button_x = False
        self._latest_button_y = False
        self._latest_button_a = False
        self._latest_button_b = False

        self._side_state = {
            side: {
                "teleop_init": np.eye(4, dtype=np.float64),
                "arm_init": np.eye(4, dtype=np.float64),
                "target": np.eye(4, dtype=np.float64),
                "trigger_pressed": False,
            }
            for side in ("left", "right")
        }

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while self._running:
            try:
                data = self._client.get_info()
                self._update_state(data)
            except Exception:
                # print(f"Error polling PicoClient: {e}")
                time.sleep(0.01)
            time.sleep(0.001)

    def _update_state(self, data):
        with self._lock:
            self._latest_left_wrist_matrix = np.array(data["left_wrist_pose"], dtype=np.float64)
            self._latest_right_wrist_matrix = np.array(data["right_wrist_pose"], dtype=np.float64)
            buttons = data["buttons"]
            self._latest_left_trigger = float(buttons["left_trigger"])
            self._latest_right_trigger = float(buttons["right_trigger"])
            self._latest_left_gripper = float(buttons["left_gripper"])
            self._latest_right_gripper = float(buttons["right_gripper"])
            self._latest_button_x = bool(buttons["X"])
            self._latest_button_y = bool(buttons["Y"])
            self._latest_button_a = bool(buttons["A"])
            self._latest_button_b = bool(buttons["B"])

    def get_action(
        self, observation: Observation, options: Options | None = None
    ) -> tuple[Action, Info]:
        del options
        action: Action = {}
        info: Info = {}

        with self._lock:
            left_obs_matrix, right_obs_matrix = self._observation_to_matrix(observation)

            left_target_matrix = self._update_side_target(
                "left",
                teleop_matrix=self._latest_left_wrist_matrix,
                obs_matrix=left_obs_matrix,
                trigger_value=self._latest_left_trigger,
            )
            right_target_matrix = self._update_side_target(
                "right",
                teleop_matrix=self._latest_right_wrist_matrix,
                obs_matrix=right_obs_matrix,
                trigger_value=self._latest_right_trigger,
            )

            left_target_pose = self._matrix_to_pose_xyzw(left_target_matrix)
            right_target_pose = self._matrix_to_pose_xyzw(right_target_matrix)

            if self._control_mode == "joint_position":
                left_joint_pos, right_joint_pos = (
                    self._env.unwrapped._kinematics.inverse_kinematics(
                        left_target_pose[:3],
                        left_target_pose[3:],
                        right_target_pose[:3],
                        right_target_pose[3:],
                        seeded=True,
                    )
                )
                action["left_joint_pos"] = left_joint_pos
                action["right_joint_pos"] = right_joint_pos
            elif self._control_mode == "cartesian_position":
                action["left_ee_pos"] = np.asarray(left_target_pose[:3], dtype=np.float32)
                action["left_ee_quat_xyzw"] = np.asarray(left_target_pose[3:], dtype=np.float32)
                action["right_ee_pos"] = np.asarray(right_target_pose[:3], dtype=np.float32)
                action["right_ee_quat_xyzw"] = np.asarray(right_target_pose[3:], dtype=np.float32)
            else:
                raise ValueError(f"Invalid control mode: {self._control_mode}")

            action["left_gripper_pos"] = np.array([self._latest_left_gripper], dtype=np.float32)
            action["right_gripper_pos"] = np.array([self._latest_right_gripper], dtype=np.float32)

            info.update(
                {
                    "left_teleop_wrist_pose": self._matrix_to_pose_wxyz(
                        self._latest_left_wrist_matrix
                    ),
                    "right_teleop_wrist_pose": self._matrix_to_pose_wxyz(
                        self._latest_right_wrist_matrix
                    ),
                    "left_trigger": self._latest_left_trigger,
                    "right_trigger": self._latest_right_trigger,
                    "button_x": self._latest_button_x,
                    "button_y": self._latest_button_y,
                    "button_a": self._latest_button_a,
                    "button_b": self._latest_button_b,
                }
            )

        return action, info

    def _update_side_target(
        self,
        side: str,
        teleop_matrix: np.ndarray,
        obs_matrix: np.ndarray | None,
        trigger_value: float,
    ) -> np.ndarray:
        state = self._side_state[side]
        pressed = trigger_value > TRIGGER_PRESS_THRESHOLD

        if pressed and not state["trigger_pressed"]:  # first time pressed
            if obs_matrix is not None:
                state["arm_init"] = obs_matrix.copy()
            state["teleop_init"] = teleop_matrix.copy()
        # elif not pressed and obs_matrix is not None: # not pressed and obs_matrix is not None
        # state["arm_init"] = obs_matrix.copy()

        state["trigger_pressed"] = pressed

        if pressed and obs_matrix is not None:
            try:
                # delta = teleop_matrix @ np.linalg.inv(state["teleop_init"])
                # target = state["arm_init"] @ delta

                # Calculate delta in the teleop frame (world frame of the controller)
                # delta = current_teleop - init_teleop
                # We want to apply this delta to the arm's init position in the arm's world frame.

                # Decompose teleop movement
                teleop_init_pos = state["teleop_init"][:3, 3]
                teleop_curr_pos = teleop_matrix[:3, 3]
                teleop_init_rot = state["teleop_init"][:3, :3]
                teleop_curr_rot = teleop_matrix[:3, :3]

                # Position delta (World Frame)
                pos_delta = teleop_curr_pos - teleop_init_pos

                # Rotation delta (Relative to init)
                # R_curr = R_delta @ R_init  =>  R_delta = R_curr @ R_init.T
                rot_delta = teleop_curr_rot @ teleop_init_rot.T

                # Apply to Arm
                arm_init_pos = state["arm_init"][:3, 3]
                arm_init_rot = state["arm_init"][:3, :3]

                target = np.eye(4)
                # Apply position delta in World Frame
                target[:3, 3] = arm_init_pos + pos_delta
                # Apply rotation delta: R_target = R_delta @ R_arm_init
                target[:3, :3] = rot_delta @ arm_init_rot
            except np.linalg.LinAlgError:
                target = obs_matrix.copy()
        else:
            target = obs_matrix.copy() if obs_matrix is not None else state["target"].copy()

        state["target"] = target.copy()
        return target

    def _observation_to_matrix(
        self,
        observation: Observation | None,
    ) -> np.ndarray | None:
        if observation is None:
            return None

        # print("observation: ", observation.keys())

        if self._control_mode == "joint_position":
            (left_pos, left_quat_xyzw, right_pos, right_quat_xyzw) = (
                self._env.unwrapped._kinematics.forward_kinematics(
                    observation["left_joint_pos"], observation["right_joint_pos"]
                )
            )
        elif self._control_mode == "cartesian_position":
            left_pos = observation["left_ee_pos"]
            left_quat_xyzw = observation["left_ee_quat_xyzw"]
            right_pos = observation["right_ee_pos"]
            right_quat_xyzw = observation["right_ee_quat_xyzw"]
        else:
            raise ValueError(f"Invalid control mode: {self._control_mode}")

        return (
            self._components_to_matrix(left_pos, left_quat_xyzw),
            self._components_to_matrix(right_pos, right_quat_xyzw),
        )

    def _components_to_matrix(self, pos, quat) -> np.ndarray:
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = R.from_quat(quat_arr).as_matrix()
        matrix[:3, 3] = pos_arr
        return matrix

    def _matrix_to_pose_xyzw(self, matrix: np.ndarray) -> np.ndarray:
        rot = R.from_matrix(matrix[:3, :3])
        quat_xyzw = rot.as_quat(scalar_first=False)
        return np.concatenate((matrix[:3, 3], quat_xyzw))

    def _matrix_to_pose_wxyz(self, matrix: np.ndarray) -> np.ndarray:
        rot = R.from_matrix(matrix[:3, :3])
        quat_wxyz = rot.as_quat(scalar_first=True)
        return np.concatenate((matrix[:3, 3], quat_wxyz))


def visualize_pico_inputs():
    import mujoco
    import mujoco.viewer

    xml = """
    <mujoco>
      <asset>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
          markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texrepeat="5 5"/>
      </asset>
      <worldbody>
        <light directional="true"/>
        <geom name="floor" size="0 0 .05" type="plane" material="groundplane"/>
        <body name="left_target" pos="0 0.5 .5" mocap="true">
          <geom type="box" size=".5 .5 .5" rgba=".6 .3 .3 .5"/>
          <site type="box" size=".01 .01 .01" rgba="1 0 0 1" pos=".05 0 0"/>
          <site type="box" size=".01 .01 .01" rgba="0 1 0 1" pos="0 .05 0"/>
          <site type="box" size=".01 .01 .01" rgba="0 0 1 1" pos="0 0 .05"/>
        </body>
        <body name="right_target" pos="0 -0.5 .5" mocap="true">
          <geom type="box" size=".5 .5 .5" rgba=".3 .6 .3 .5"/>
          <site type="box" size=".01 .01 .01" rgba="1 0 0 1" pos=".05 0 0"/>
          <site type="box" size=".01 .01 .01" rgba="0 1 0 1" pos="0 .05 0"/>
          <site type="box" size=".01 .01 .01" rgba="0 0 1 1" pos="0 0 .05"/>
        </body>
      </worldbody>
    </mujoco>
    """

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    left_mocap_id = m.body("left_target").mocapid[0]
    right_mocap_id = m.body("right_target").mocapid[0]

    policy = PicoPolicy()
    print("PicoPolicy initialized. Visualizing raw inputs...")

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        while viewer.is_running():
            mujoco.mj_step(m, d)

            # Get latest matrices
            left_matrix = policy._latest_left_wrist_matrix
            right_matrix = policy._latest_right_wrist_matrix

            # Update left
            left_pose = policy._matrix_to_pose_xyzw(left_matrix)
            d.mocap_pos[left_mocap_id] = left_pose[:3]
            # xyzw -> wxyz
            d.mocap_quat[left_mocap_id] = np.array(
                [left_pose[6], left_pose[3], left_pose[4], left_pose[5]]
            )

            # Update right
            right_pose = policy._matrix_to_pose_xyzw(right_matrix)
            d.mocap_pos[right_mocap_id] = right_pose[:3]
            d.mocap_quat[right_mocap_id] = np.array(
                [right_pose[6], right_pose[3], right_pose[4], right_pose[5]]
            )

            viewer.sync()
            time.sleep(0.01)


def run_env(real: bool):
    import gymnasium as gym
    from groot.control.envs.yam.yam_sim_env import MujocoViewerWrapper
    from gymnasium.envs.registration import register

    # Environment
    if real:
        register(id="YamReal-v0", entry_point="groot.control.envs.yam.yam_real_env:YamRealEnv")
        env = gym.make("YamReal-v0", control_mode="joint_position")
    else:
        # sim
        register(
            id="YamSim-v0",
            entry_point="groot.control.envs.yam.yam_sim_env:YamSimEnv",
        )
        env = gym.make("YamSim-v0", control_mode="joint_position")
        env = MujocoViewerWrapper(env)

    # Policy
    policy = PicoPolicy(env=env)
    policy.reset()

    # Main loop
    obs, info = env.reset()
    try:
        while True:
            action, _ = policy.get_action(obs)
            obs, _, _, _, _ = env.step(action)

    finally:
        env.close()


def main():
    run_env(real=False)
    # visualize_pico_inputs()


if __name__ == "__main__":
    main()
