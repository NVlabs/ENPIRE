# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PolicyBackend adapter for grootpool — carries obs-spec metadata.

Extends ``cap.policy.grootpool.backend.GrootpoolBackend`` with class-level
metadata describing the wire contract:

- ``LANGUAGE_KEY``   — where in the obs dict the backend expects the task
                       description to appear.
- ``CONTROLLER_TYPE`` — the robosuite controller the checkpoint was trained
                       on. ``use_policy_output`` asserts the env matches.
- ``MODEL_VERSION``  — "n15" or "n16" (gr00t server model family).

Registry ``GROOTPOOL_BACKENDS`` maps short names ("n15", "n16") to backend
classes so ``use_policy_output`` can resolve by name from Hydra config.
"""

from __future__ import annotations

from enpire.env.forge.cap.policy.grootpool.backend import GrootpoolBackend


class GrootpoolN15Backend(GrootpoolBackend):
    """GR00T N1.5 via grootpool middleware.

    Obs passes through unchanged — N1.5 consumes robocasa's native gym-wrapper
    obs keys (``video.robot0_agentview_left``, etc.).
    """

    MODEL_VERSION: str = "n15"
    LANGUAGE_KEY: str = "annotation.human.task_description"
    CONTROLLER_TYPE: str = "osc_pose"
    CAMERA_MAP: dict[str, str] = {
        "video.robot0_agentview_left": "side_left",
        "video.robot0_agentview_right": "side_right",
        "video.robot0_eye_in_hand": "wrist",
    }

    def __init__(self, task_description: str, *, endpoint: str | None = None) -> None:
        super().__init__(task_description, model=self.MODEL_VERSION, endpoint=endpoint)


class GrootpoolN16Backend(GrootpoolBackend):
    """GR00T N1.6 via grootpool middleware.

    N1.6 expects different obs keys; callers must feed an env view that emits
    them (see ``_workers/robocasa365.py:_make_env_n16`` for the adapter that
    subclasses ``RoboCasaGymEnv`` to produce N1.6 keys). Adding a shared
    adapter is a follow-up once N1.6 is needed from this path.
    """

    MODEL_VERSION: str = "n16"
    LANGUAGE_KEY: str = "annotation.human.action.task_description"
    CONTROLLER_TYPE: str = "osc_pose"
    CAMERA_MAP: dict[str, str] = {
        "video.res256_image_side_0": "side_left",
        "video.res256_image_side_1": "side_right",
        "video.res256_image_wrist_0": "wrist",
    }

    def __init__(self, task_description: str, *, endpoint: str | None = None) -> None:
        super().__init__(task_description, model=self.MODEL_VERSION, endpoint=endpoint)


GROOTPOOL_BACKENDS: dict[str, type[GrootpoolBackend]] = {
    "n15": GrootpoolN15Backend,
    "n16": GrootpoolN16Backend,
}
