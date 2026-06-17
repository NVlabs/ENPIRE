from rl.parking import ParkingNavigator


class _PoseManager:
    def __init__(self):
        self.calls = []

    def boundary_corner_pose(self, *, kind, corner_idx, z_idx):
        self.calls.append((kind, corner_idx, z_idx))
        return {"kind": kind, "corner": corner_idx, "z": z_idx}


def test_parking_defaults_to_initial_high_z_and_cycles_corners():
    navigator = ParkingNavigator()
    manager = _PoseManager()

    pose, label = navigator.next_corner_pose(manager)
    assert pose == {"kind": "initial", "corner": 0, "z": 1}
    assert label == "initial boundary z-high corner 1/4"

    pose, _ = navigator.next_corner_pose(manager)
    assert pose == {"kind": "initial", "corner": 1, "z": 1}


def test_parking_selects_oor_and_low_z_resets_corner_index():
    navigator = ParkingNavigator(corner_idx=3)
    manager = _PoseManager()

    navigator.select_oor_boundary()
    navigator.select_z_low()
    pose, label = navigator.next_corner_pose(manager)

    assert pose == {"kind": "oor", "corner": 0, "z": 0}
    assert label == "oor boundary z-low corner 1/4"


def test_switching_boundary_kind_resets_corner_but_repeated_kind_cycles():
    navigator = ParkingNavigator()
    manager = _PoseManager()

    navigator.select_initial_boundary()
    navigator.next_corner_pose(manager)
    navigator.select_initial_boundary()
    pose, _ = navigator.next_corner_pose(manager)
    assert pose == {"kind": "initial", "corner": 1, "z": 1}

    navigator.select_oor_boundary()
    pose, _ = navigator.next_corner_pose(manager)
    assert pose == {"kind": "oor", "corner": 0, "z": 1}
