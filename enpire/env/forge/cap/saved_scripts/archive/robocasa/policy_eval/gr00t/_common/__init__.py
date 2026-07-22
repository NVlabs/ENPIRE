# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-exports from pandaomron and robocasa365 _common modules.

Allows both:
  from _common import REPO_DIR, ModelServer       # pandaomron symbols
  from _common.robocasa365 import get_model_config  # robocasa365 symbols

All external paths (GROOT_ROOT, MODEL_PATH, etc.) are read from environment
variables at import time.  See each module's docstring for the full list.
"""


from _common.pandaomron import (  # noqa: F401
    CLIENT_PYTHON,
    EMBODIMENT,
    GROOT_ROOT,
    REPO_DIR,
    LOG_ROOT,
    MODEL_PATH,
    ModelServer,
    SERVER_PYTHON,
    SERVER_SCRIPT,
    TeeStream,
)

from _common.robocasa365 import (  # noqa: F401
    CLIENT_PYTHON_365,
    CLIENT_SCRIPT_365,
    ModelServer365,
    N15_EMBODIMENT,
    N15_GROOT_ROOT,
    N15_MODEL_PATH,
    N15_SERVER_PYTHON,
    N15_SERVER_SCRIPT,
    N16_EMBODIMENT,
    N16_GROOT_ROOT,
    N16_MODEL_PATH,
    N16_SERVER_PYTHON,
    N16_SERVER_SCRIPT,
    ROBOCASA365_ROOT,
    get_client_python_365,
    get_model_config,
)
# TeeStream is also in robocasa365 but already imported from pandaomron above.
