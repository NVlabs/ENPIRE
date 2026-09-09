"""Object OBB poses and grasp candidates from the registered SAM3/RGB-D tool.

Import from ``skill_library.sample_grasp_pose_3d_bb`` inside a CaP script, or
invoke this function with run_script.py's script_function/script_kwargs options.
All geometry is in the calibrated world frame, in metres. This is a visible
surface bounding-box estimate, not a validated full-object pose or semantic yaw.
"""

from __future__ import annotations

import math
import time

import numpy as np


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def sample_grasp_pose_3d_bb(
    object_name,
    camera="top",
    *,
    image_bbox=None,
    min_world_z=None,
    max_world_z=None,
    tcp_offset_z_m=None,
    relax=False,
    min_points=30,
    output_dir=None,
    observation_cameras=("top", "left"),
):
    """Return one object_pose plus proposed grasps, without commanding motion.

    Use a specific object description and, for repeated objects, a pixel ROI
    [xmin,ymin,xmax,ymax]. The underlying tool segments one instance; this
    wrapper does not enumerate instances or turn alternative grasps into objects.
    ``object_pose.position`` is the OBB centre; ``grasps[].position`` is an EEF
    target. OBB axis signs/order are ambiguous for symmetric objects. Candidate
    scores rank grasps and are not pose confidence or an accuracy measurement.
    """
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("object_name must be a nonempty string")
    if not isinstance(camera, str) or not camera.strip():
        raise ValueError("camera must be a nonempty string")
    if type(relax) is not bool:
        raise ValueError("relax must be a boolean")
    if type(min_points) is not int or min_points < 4:
        raise ValueError("min_points must be an integer >= 4")
    kwargs = {"object_name": object_name, "camera": camera, "relax": relax}
    if image_bbox is not None:
        box = _array(image_bbox, (4,), "image_bbox")
        if (box < 0).any() or not np.equal(box, np.floor(box)).all() or (box[2:] <= box[:2]).any():
            raise ValueError("image_bbox must have nonnegative integer xyxy bounds with positive area")
        kwargs["image_bbox"] = box.astype(int).tolist()
    for name, value in (("min_world_z", min_world_z), ("max_world_z", max_world_z),
                        ("tcp_offset_z_m", tcp_offset_z_m)):
        if value is not None:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            kwargs[name] = float(value)
    if min_world_z is not None and max_world_z is not None and min_world_z >= max_world_z:
        raise ValueError("min_world_z must be below max_world_z")

    from scipy.spatial.transform import Rotation

    import skill_library.namespace as tools

    started = time.time()
    observation = None
    if output_dir is not None:
        import dataclasses
        import json
        from pathlib import Path

        from PIL import Image

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        state = tools.get_robot_state()
        state = dataclasses.asdict(state) if dataclasses.is_dataclass(state) else state
        observation = {"robot_state": state, "images": {}}
        for name in observation_cameras:
            path = directory / f"{name}.png"
            Image.fromarray(np.asarray(tools.get_camera_image(name), dtype=np.uint8)).save(path)
            observation["images"][name] = str(path)
        observation = json.loads(json.dumps(observation, default=lambda value: value.tolist()))
        (directory / "observation.json").write_text(json.dumps(observation, indent=2))
    candidates = tools.sample_grasp_pose_3d_bb(**kwargs)
    if not candidates:
        raise RuntimeError(f"No 3D bounding-box candidates for {object_name!r}")
    bbox = _field(candidates[0], "bbox_result")
    center = _array(_field(bbox, "obb_center_world"), (3,), "OBB centre")
    axes = _array(_field(bbox, "obb_axes_world"), (3, 3), "OBB axes").copy()
    extents = _array(_field(bbox, "obb_extents"), (3,), "OBB extents")
    n_points = int(_field(bbox, "n_points"))
    top_z = float(_field(bbox, "top_surface_z"))
    if n_points < min_points or (extents <= 0).any() or not math.isfinite(top_z):
        raise RuntimeError("Insufficient depth support or invalid OBB dimensions")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise RuntimeError("OBB axes are not orthonormal")
    # RANSAC's OBB basis can be left handed. Flipping an axis preserves the box,
    # but a quaternion requires a right-handed rotation; never feed a reflection
    # to scipy's implicit rotation correction.
    basis_flipped = bool(np.linalg.det(axes) < 0)
    if basis_flipped:
        axes[:, 2] *= -1
    grasps = []
    for candidate in candidates:
        position = _array(_field(candidate, "position"), (3,), "grasp position")
        rpy = _array(_field(candidate, "rpy"), (3,), "grasp RPY")
        width, score = float(_field(candidate, "width")), float(_field(candidate, "score"))
        if not math.isfinite(width) or width < 0 or not math.isfinite(score):
            raise RuntimeError("Invalid grasp width or ranking score")
        grasps.append({"position": position.tolist(), "rpy": rpy.tolist(),
                       "width_m": width, "score": score})
    return {
        "object_name": object_name, "camera": camera, "frame": "world",
        "observation": observation,
        "source": "sample_grasp_pose_3d_bb:SAM3+RGBD+RANSAC",
        "observation_started_unix_s": started,
        "observation_finished_unix_s": time.time(),
        "object_pose": {"position": center.tolist(),
                        "quaternion_xyzw": Rotation.from_matrix(axes).as_quat().tolist(),
                        "rotation_matrix": axes.tolist()},
        "extents_m": extents.tolist(), "top_surface_z_m": top_z,
        "n_points": n_points, "grasps": grasps,
        "quality": {
            "geometry_valid": True, "accuracy_verified": False,
            "orientation_ambiguous": True, "obb_axis_sign_flipped": basis_flipped,
            "note": "Visible-surface OBB; occlusion and symmetry affect centre, size and orientation. "
                    "Grasps require collision planning and measured execution checks.",
        },
    }
