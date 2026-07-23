# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import threading

import numpy as np
import portal
from enpire.env.forge.experimental.pico.xr_client import XrClient
from scipy.spatial.transform import Rotation as R

from enpire.env.forge.robot.constants import PICO_PORT

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


class PicoServer:
    def __init__(self):
        self.xr_client = XrClient()
        # self.run_pico_service()

        self.reset_status()

        # Portal server
        self._server = portal.Server(PICO_PORT)
        self._server.bind("get_info", self.get_info)
        self._server_thread = threading.Thread(target=self._server.start, daemon=True)
        self._server_thread.start()
        print(f"PicoServer started on port {PICO_PORT}")

    def get_info(self):
        return self.get()

    def run_pico_service(self):
        # Run the pico service
        self.pico_service_pid = subprocess.Popen(
            ["bash", "/opt/apps/roboticsservice/runService.sh"]
        )
        print(f"Pico service running with pid {self.pico_service_pid.pid}")

    def reset_status(self):
        self.current_base_height = 0.74  # Initial base height, 0.74m (standing height)
        self.compliance = np.array([0.05, 0.05, 0.0])
        self.toggle_policy_action_last = False
        self.toggle_activation_last = False
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False
        self.toggle_locomotion_mode_switch_last = False
        self.reset_env_and_policy_last = False
        self.locomotion_mode = 0  # 0 = slow walk, 1 = fast walk (normal walk)

    def get(self):
        pico_data = self._get_pico_data()

        raw_data = self._generate_unified_raw_data(pico_data)
        return raw_data

    def _get_pico_data(self):
        pico_data = {}

        # Get the pose of the left and right controllers and the headset
        pico_data["left_pose"] = self.xr_client.get_pose_by_name("left_controller")
        pico_data["right_pose"] = self.xr_client.get_pose_by_name("right_controller")
        pico_data["head_pose"] = self.xr_client.get_pose_by_name("headset")

        # Get key value of the left and right controllers
        pico_data["left_trigger"] = self.xr_client.get_key_value_by_name("left_trigger")
        pico_data["right_trigger"] = self.xr_client.get_key_value_by_name("right_trigger")
        pico_data["left_grip"] = self.xr_client.get_key_value_by_name("left_grip")
        pico_data["right_grip"] = self.xr_client.get_key_value_by_name("right_grip")

        # Get button state of the left and right controllers
        pico_data["A"] = self.xr_client.get_button_state_by_name("A")
        pico_data["B"] = self.xr_client.get_button_state_by_name("B")
        pico_data["X"] = self.xr_client.get_button_state_by_name("X")
        pico_data["Y"] = self.xr_client.get_button_state_by_name("Y")
        pico_data["left_menu_button"] = self.xr_client.get_button_state_by_name("left_menu_button")
        pico_data["right_menu_button"] = self.xr_client.get_button_state_by_name(
            "right_menu_button"
        )
        pico_data["left_axis_click"] = self.xr_client.get_button_state_by_name("left_axis_click")
        pico_data["right_axis_click"] = self.xr_client.get_button_state_by_name("right_axis_click")

        # Get the timestamp of the left and right controllers
        pico_data["timestamp"] = self.xr_client.get_timestamp_ns()

        # Get the hand tracking state of the left and right controllers
        pico_data["left_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("left")
        pico_data["right_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("right")

        # Get the joystick state of the left and right controllers
        pico_data["left_joystick"] = self.xr_client.get_joystick_state("left")
        pico_data["right_joystick"] = self.xr_client.get_joystick_state("right")

        # Get the motion tracker data
        pico_data["motion_tracker_data"] = self.xr_client.get_motion_tracker_data()

        # Get the body tracking data
        pico_data["body_tracking_data"] = self.xr_client.get_body_tracking_data()

        return pico_data

    def _generate_unified_raw_data(self, pico_data):
        # Get controller position and orientation in z up world frame
        left_controller_T = self._process_xr_pose(pico_data["left_pose"], pico_data["head_pose"])
        right_controller_T = self._process_xr_pose(pico_data["right_pose"], pico_data["head_pose"])

        # Get base height command
        height_increment = 0.01  # Small step per call when button is pressed
        if pico_data["Y"]:
            self.current_base_height += height_increment
        elif pico_data["X"]:
            self.current_base_height -= height_increment
        self.current_base_height = np.clip(self.current_base_height, 0.2, 0.74)

        # Get activation commands
        toggle_policy_action_tmp = pico_data["left_menu_button"] and (
            pico_data["left_trigger"] > 0.5
        )
        toggle_activation_tmp = pico_data["left_menu_button"] and (pico_data["right_trigger"] > 0.5)

        self.toggle_policy_action_last = toggle_policy_action_tmp
        self.toggle_activation_last = toggle_activation_tmp

        # Get data collection commands
        toggle_data_collection_tmp = pico_data["A"] and not pico_data["left_menu_button"]
        toggle_data_abort_tmp = pico_data["B"] and not pico_data["left_menu_button"]

        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        # Get toggle locomotion mode switch command (click both left and right joysticks)
        toggle_locomotion_mode_switch_tmp = (
            pico_data["left_axis_click"] and pico_data["right_axis_click"]
        )

        # Toggle the locomotion mode state when button is pressed (edge-triggered)
        if self.toggle_locomotion_mode_switch_last != toggle_locomotion_mode_switch_tmp:
            if toggle_locomotion_mode_switch_tmp:
                # Toggle between 0 and 1
                self.locomotion_mode = 1 - self.locomotion_mode
        self.toggle_locomotion_mode_switch_last = toggle_locomotion_mode_switch_tmp

        reset_env_and_policy_tmp = pico_data["left_menu_button"] and pico_data["A"]
        self.reset_env_and_policy_last = reset_env_and_policy_tmp

        return dict(
            left_wrist_pose=left_controller_T,
            right_wrist_pose=right_controller_T,
            buttons=dict(
                X=pico_data["X"],
                Y=pico_data["Y"],
                A=pico_data["A"],
                B=pico_data["B"],
                left_menu_button=pico_data["left_menu_button"],
                right_menu_button=pico_data["right_menu_button"],
                left_axis_click=pico_data["left_axis_click"],
                right_axis_click=pico_data["right_axis_click"],
                left_gripper=1 - pico_data["left_trigger"],
                right_gripper=1 - pico_data["right_trigger"],
                left_trigger=pico_data["left_grip"],
                right_trigger=pico_data["right_grip"],
            ),
        )

    def _process_xr_pose(self, controller_pose, headset_pose):
        # Convert controller pose to x, y, z, w quaternion
        xr_pose_xyz = np.array(controller_pose)[:3]  # x, y, z
        xr_pose_quat = np.array(controller_pose)[3:]  # x, y, z, w

        # Handle all-zero quaternion case by using identity quaternion
        if np.allclose(xr_pose_quat, 0):
            xr_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        xr_pose_xyz = R_HEADSET_TO_WORLD @ xr_pose_xyz
        xr_pose_rotation = R.from_quat(xr_pose_quat).as_matrix()
        xr_pose_rotation = R_HEADSET_TO_WORLD @ xr_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Convert headset pose to x, y, z, w quaternion
        headset_pose_xyz = np.array(headset_pose)[:3]
        headset_pose_quat = np.array(headset_pose)[3:]

        if np.allclose(headset_pose_quat, 0):
            headset_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        headset_pose_xyz = R_HEADSET_TO_WORLD @ headset_pose_xyz
        headset_pose_rotation = R.from_quat(headset_pose_quat).as_matrix()
        headset_pose_rotation = R_HEADSET_TO_WORLD @ headset_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Calculate the delta between the controller and headset positions
        xr_pose_xyz_delta = xr_pose_xyz - headset_pose_xyz

        # Calculate the yaw of the headset
        R_headset_to_world = R.from_matrix(headset_pose_rotation)
        headset_pose_yaw = R_headset_to_world.as_euler("xyz")[2]  # Extract yaw (Z-axis rotation)
        inverse_yaw_rotation = R.from_euler("z", -headset_pose_yaw).as_matrix()

        # Align with headset yaw to controller position delta and rotation
        xr_pose_xyz_delta_compensated = inverse_yaw_rotation @ xr_pose_xyz_delta
        xr_pose_rotation_compensated = inverse_yaw_rotation @ xr_pose_rotation

        xr_pose_T = np.eye(4)
        xr_pose_T[:3, :3] = xr_pose_rotation_compensated
        xr_pose_T[:3, 3] = xr_pose_xyz_delta_compensated

        return xr_pose_T


if __name__ == "__main__":
    # from groot.control.utils.debugger import wait_for_debugger
    # wait_for_debugger()

    server = PicoServer()
    while True:
        raw_data = server.get()

        print("buttons: ", raw_data["buttons"])
        # print(
        #     f"left_wrist: {raw_data['left_wrist_pose']}, right_wrist: {raw_data['right_wrist_pose']}"
        # )
        # print(f"buttons: {raw_data['buttons']}")
