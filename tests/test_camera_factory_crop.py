from __future__ import annotations

import numpy as np

from robot.camera_factory import center_square_crop, crop_image_region
from robot.yam._base_yam_env import _BaseYamEnv


def test_center_square_crop_640x480_uses_center_480_square() -> None:
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    cropped = center_square_crop(image)
    assert cropped.shape == (480, 480, 3)
    np.testing.assert_array_equal(cropped, image[:, 80:560])


def test_base_env_can_crop_left_and_right_camera_names() -> None:
    env = _BaseYamEnv.__new__(_BaseYamEnv)
    env.crop_camera_names = ("left", "right")
    env.crop_region = ("center",)
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    left = env._crop_camera_image("left", image)
    right = env._crop_camera_image("right", image)
    top = env._crop_camera_image("top", image)

    np.testing.assert_array_equal(left, image[:, 80:560])
    np.testing.assert_array_equal(right, image[:, 80:560])
    assert top is image


def test_crop_image_region_uses_xywh_pixels() -> None:
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    cropped = crop_image_region(image, ("80", "0", "480", "480"))
    assert cropped.shape == (480, 480, 3)
    np.testing.assert_array_equal(cropped, image[:, 80:560])


def test_base_env_accepts_explicit_crop_region() -> None:
    env = _BaseYamEnv.__new__(_BaseYamEnv)
    env.crop_camera_names = ("right",)
    env.crop_region = ("80", "0", "480", "480")
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    cropped = env._crop_camera_image("right", image)

    np.testing.assert_array_equal(cropped, image[:, 80:560])


def test_base_env_crops_both_arms_with_centered_384_xywh_region() -> None:
    env = _BaseYamEnv.__new__(_BaseYamEnv)
    env.crop_camera_names = ("left", "right")
    env.crop_region = ("128", "48", "384", "384")
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    left = env._crop_camera_image("left", image)
    right = env._crop_camera_image("right", image)
    top = env._crop_camera_image("top", image)

    assert left.shape == (384, 384, 3)
    np.testing.assert_array_equal(left, image[48:432, 128:512])
    np.testing.assert_array_equal(right, image[48:432, 128:512])
    assert top is image


def test_base_env_accepts_per_camera_crop_regions() -> None:
    env = _BaseYamEnv.__new__(_BaseYamEnv)
    env.crop_camera_names = ("left", "right")
    env.crop_region = ("left:128,48,384,384", "right:192,112,256,256")
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    left = env._crop_camera_image("left", image)
    right = env._crop_camera_image("right", image)

    assert left.shape == (384, 384, 3)
    assert right.shape == (256, 256, 3)
    np.testing.assert_array_equal(left, image[48:432, 128:512])
    np.testing.assert_array_equal(right, image[112:368, 192:448])
