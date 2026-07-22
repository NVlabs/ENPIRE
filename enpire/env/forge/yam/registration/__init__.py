# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe facade over the original Forge station identification logic."""

from enpire.env.forge.yam.registration.rules import (
    RegistrationResult,
    render_camera_aliases,
    render_udev_rules,
)

__all__ = ["RegistrationResult", "render_camera_aliases", "render_udev_rules"]
