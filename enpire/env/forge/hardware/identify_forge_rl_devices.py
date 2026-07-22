# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry for Forge device identification.

The implementation is preserved under ENPIRE's YAM registration package.  It
only discovers identifiers interactively; public commands write resulting
profiles outside the repository.
"""

from enpire.env.forge.yam.registration._legacy_identify import (
    BUTTON_ROLES,
    CAN_ROLES,
    REALSENSE_ROLES,
    REALSENSE_TOP_ROLE,
    TOP_CAMERA_ROLE,
    emit_button_rule,
    emit_can_rule,
    emit_top_rule,
    identify_unplug,
    main,
    scan_realsense_serials,
    scan_usb_serials,
)

__all__ = [
    "BUTTON_ROLES",
    "CAN_ROLES",
    "REALSENSE_ROLES",
    "REALSENSE_TOP_ROLE",
    "TOP_CAMERA_ROLE",
    "emit_button_rule",
    "emit_can_rule",
    "emit_top_rule",
    "identify_unplug",
    "main",
    "scan_realsense_serials",
    "scan_usb_serials",
]


if __name__ == "__main__":
    raise SystemExit(main())
