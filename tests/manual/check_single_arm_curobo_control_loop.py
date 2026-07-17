#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from enpire.env.forge.experimental.motion_planner_curobo import YamMotionPlannerCurobo
from enpire.env.forge.experimental.scripted_policy import SafetyLimits, ScriptedPolicy
from enpire.env.forge.robot.yam.yam_sim_env import YamSimEnv


def main() -> int:
    env = YamSimEnv(control_mode="joint_position", policy_control_freq=30.0, enable_cameras=False)
    try:
        zero_state = {
            "left_joint_pos": np.zeros(6, dtype=np.float32),
            "left_gripper_pos": np.ones(1, dtype=np.float32),
            "right_joint_pos": np.zeros(6, dtype=np.float32),
            "right_gripper_pos": np.ones(1, dtype=np.float32),
        }
        observation, _ = env.reset(options={"target_joint_position": zero_state})

        policy = ScriptedPolicy(
            action_space=env.action_space,
            control_hz=30.0,
            safety=SafetyLimits(),
        )
        policy.ensure_initialized(observation)

        planner = YamMotionPlannerCurobo(validate_with_mujoco=True)
        cur_left = np.asarray(observation["left_joint_pos"], dtype=np.float64)
        cur_right = np.asarray(observation["right_joint_pos"], dtype=np.float64)
        cur_l_pos, cur_l_q, _cur_r_pos, _cur_r_q = planner._kin.forward_kinematics(cur_left, cur_right)

        result = planner.plan_to_pose(
            current_left_jp=cur_left,
            current_right_jp=cur_right,
            target_left_pos=np.asarray(cur_l_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.10]),
            target_left_quat_xyzw=np.asarray(cur_l_q, dtype=np.float64),
            side="left",
            left_gripper=float(np.asarray(observation["left_gripper_pos"]).reshape(-1)[0]),
            right_gripper=float(np.asarray(observation["right_gripper_pos"]).reshape(-1)[0]),
        )

        print("planner_status", result["status"])
        print("planner_detail", result.get("status_detail"))
        if result["status"] != "Success":
            return 1

        planned_right = np.asarray(result["right_positions"], dtype=np.float64)
        planner_right_delta = (
            float(np.max(np.abs(planned_right - cur_right[None, :])))
            if len(planned_right)
            else 0.0
        )
        print("planner_right_joint_max_delta", planner_right_delta)

        policy.execute_trajectory(
            np.asarray(result["left_positions"], dtype=np.float64),
            planned_right,
            current_left_joint_pos=cur_left,
            current_right_joint_pos=cur_right,
        )

        max_right_cmd_delta = 0.0
        max_right_obs_delta = 0.0
        steps = max(len(result["left_positions"]) + 5, 10)
        for _ in range(steps):
            action, _ = policy.get_action(observation)
            right_cmd = np.asarray(action["right_joint_pos"], dtype=np.float64)
            max_right_cmd_delta = max(
                max_right_cmd_delta,
                float(np.max(np.abs(right_cmd - cur_right))),
            )
            observation, _, _, _, _ = env.step(action)
            right_obs = np.asarray(observation["right_joint_pos"], dtype=np.float64)
            max_right_obs_delta = max(
                max_right_obs_delta,
                float(np.max(np.abs(right_obs - cur_right))),
            )

        print("control_loop_right_command_max_delta", max_right_cmd_delta)
        print("control_loop_right_observation_max_delta", max_right_obs_delta)

        tol = 1e-4
        if max_right_cmd_delta > tol:
            print(f"FAIL: right arm command drift exceeded tolerance {tol}", file=sys.stderr)
            return 2
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
