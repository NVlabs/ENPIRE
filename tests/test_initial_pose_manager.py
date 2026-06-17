import numpy as np

from rl.config import DataCollectionConfig
from rl.initial_pose_manager import InitialPoseManager


def _cfg(tmp_path, *, enabled_sides="right", enable_oor_check=True):
    config_file = tmp_path / "task.yaml"
    positions_file = tmp_path / "initial_position.yaml"
    config_file.write_text("initial_positions_file: initial_position.yaml\n")
    positions_file.write_text(
        """
position_2:
  left:
    position: [10.0, 10.0, 10.0]
    rpy_deg: [0.0, 0.0, 0.0]
    gripper_pos: [0.0]
  right:
    position: [20.0, 20.0, 20.0]
    rpy_deg: [0.0, 0.0, 0.0]
    gripper_pos: [0.0]
position_1:
  left:
    position: [1.0, 2.0, 3.0]
    rpy_deg: [0.0, 0.0, 0.0]
    gripper_pos: [0.0]
  right:
    position: [4.0, 5.0, 6.0]
    rpy_deg: [0.0, 0.0, 0.0]
    gripper_pos: [0.0]
"""
    )
    return DataCollectionConfig(
        config_file=str(config_file),
        initial_positions_file="initial_position.yaml",
        enabled_sides=enabled_sides,
        randomize_initial_pose=True,
        x_init_lim=(0.1, 0.1),
        y_init_lim=(0.2, 0.2),
        z_init_lim=(-0.3, -0.3),
        enable_oor_check=enable_oor_check,
        x_oor_lim=(-0.5, 0.5),
        y_oor_lim=(-0.5, 0.5),
        z_oor_lim=(-0.5, 0.5),
    )


def test_loads_positions_in_numeric_order_and_wraps_selection(tmp_path):
    manager = InitialPoseManager(_cfg(tmp_path))

    assert manager.count == 2
    assert manager.current_base_pose()["right"]["position"] == [4.0, 5.0, 6.0]

    manager.select_previous()
    assert manager.current_base_pose()["right"]["position"] == [20.0, 20.0, 20.0]

    manager.select_next()
    assert manager.current_base_pose()["right"]["position"] == [4.0, 5.0, 6.0]

    manager.select_initial_position_index(2)
    assert manager.current_base_pose()["right"]["position"] == [20.0, 20.0, 20.0]


def test_randomizes_only_enabled_side_and_keeps_base_pose_stable(tmp_path):
    manager = InitialPoseManager(_cfg(tmp_path, enabled_sides="right"))

    episode_pose = manager.build_episode_start_pose()

    assert episode_pose["left"]["position"] == [1.0, 2.0, 3.0]
    assert episode_pose["right"]["position"] == [4.1, 5.2, 5.7]
    assert np.allclose(manager.last_offset, [0.1, 0.2, -0.3])
    assert manager.current_base_pose()["right"]["position"] == [4.0, 5.0, 6.0]


def test_oor_check_is_relative_to_unrandomized_base_pose(tmp_path):
    manager = InitialPoseManager(_cfg(tmp_path, enabled_sides="right"))
    manager.build_episode_start_pose()

    assert not manager.is_out_of_range({"right_ee_pos": np.array([4.4, 5.0, 6.0])})
    assert manager.is_out_of_range({"right_ee_pos": np.array([4.6, 5.0, 6.0])})
    assert manager.last_out_of_range == {
        "side": "right",
        "axis": "x",
        "delta": 0.5999999999999996,
        "lo": -0.5,
        "hi": 0.5,
    }


def test_boundary_corner_pose_uses_selected_box_and_z(tmp_path):
    manager = InitialPoseManager(_cfg(tmp_path, enabled_sides="right"))

    init_pose = manager.boundary_corner_pose(kind="initial", corner_idx=0, z_idx=1)
    oor_pose = manager.boundary_corner_pose(kind="oor", corner_idx=2, z_idx=0)

    assert init_pose["right"]["position"] == [4.1, 5.2, 5.7]
    assert oor_pose["right"]["position"] == [4.5, 5.5, 5.5]
    assert init_pose["left"]["position"] == [1.0, 2.0, 3.0]


def test_can_capture_enabled_side_base_pose_from_observation(tmp_path):
    manager = InitialPoseManager(_cfg(tmp_path, enabled_sides="left"))

    manager.set_base_pose_from_observation(
        {
            "left_ee_pos": np.array([0.4, 0.1, 0.9], dtype=np.float32),
            "left_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
            "left_gripper_pos": np.array([0.002], dtype=np.float32),
            "right_ee_pos": np.array([9.0, 9.0, 9.0], dtype=np.float32),
            "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        }
    )

    pose = manager.current_base_pose()
    assert np.allclose(pose["left"]["position"], [0.4, 0.1, 0.9])
    assert np.allclose(pose["left"]["rot6d"], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(pose["left"]["gripper_pos"], [0.002])
    assert pose["right"]["position"] == [4.0, 5.0, 6.0]


def test_can_build_lifted_pose_from_observation(tmp_path):
    manager = InitialPoseManager(_cfg(tmp_path, enabled_sides="left"))

    pose = manager.pose_from_observation(
        {
            "left_ee_pos": np.array([0.4, 0.1, 0.9], dtype=np.float32),
            "left_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
            "left_gripper_pos": np.array([0.002], dtype=np.float32),
        },
        z_offset_m=0.03,
    )

    assert np.allclose(pose["left"]["position"], [0.4, 0.1, 0.93])
    assert np.allclose(pose["left"]["rot6d"], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(pose["left"]["gripper_pos"], [0.002])
