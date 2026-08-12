# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = REPO_ROOT / "enpire" / "env" / "forge"
RUN_SCRIPT_PATH = FORGE_ROOT / "run_script.py"
EXPECTED_SCRIPT = (
    FORGE_ROOT
    / "cap"
    / "saved_scripts"
    / "examples"
    / "pick_object.py"
).resolve()


def _load_run_script_module() -> types.ModuleType:
    cv2 = types.ModuleType("cv2")
    numpy = types.ModuleType("numpy")

    cap = types.ModuleType("enpire.env.forge.cap")
    cap_agent = types.ModuleType("enpire.env.forge.cap.agent")
    cap_agent_config = types.ModuleType("enpire.env.forge.cap.agent.agent_config")
    cap_agent_tools = types.ModuleType("enpire.env.forge.cap.agent.tools")
    cap_config = types.ModuleType("enpire.env.forge.cap.config")
    cap_server = types.ModuleType("enpire.env.forge.cap.server")
    cap_server_cap_server = types.ModuleType("enpire.env.forge.cap.server.cap_server")
    cap.__path__ = []  # type: ignore[attr-defined]
    cap_agent.__path__ = []  # type: ignore[attr-defined]

    cap.agent = cap_agent
    cap.config = cap_config
    cap.server = cap_server
    cap_agent.agent_config = cap_agent_config
    cap_agent.tools = cap_agent_tools
    cap_server.cap_server = cap_server_cap_server

    dummy_type = type("DummyType", (), {})
    cap_agent_config.register_configs = lambda: None
    cap_agent_tools.create_default_registry = lambda *args, **kwargs: None
    cap_agent_tools.Detection3D = dummy_type
    cap_agent_tools.MoveResult = dummy_type
    cap_agent_tools.RobotState = dummy_type

    cap_config.CAMERA_NAMES = []
    cap_config.CAP_SERVER_PORT = 0
    cap_config.DETECTION_SERVER_PORT = 0
    cap_config.GRIPPER_SETTLE_TIMEOUT_S = 0.0
    cap_config.MOVE_EEF_MAX_DURATION_S = 0.0
    cap_config.MOVE_EEF_MAX_VEL = 0.0

    cap_server_cap_server.CapServer = dummy_type

    stubs = {
        "cv2": cv2,
        "numpy": numpy,
        "enpire.env.forge.cap": cap,
        "enpire.env.forge.cap.agent": cap_agent,
        "enpire.env.forge.cap.agent.agent_config": cap_agent_config,
        "enpire.env.forge.cap.agent.tools": cap_agent_tools,
        "enpire.env.forge.cap.config": cap_config,
        "enpire.env.forge.cap.server": cap_server,
        "enpire.env.forge.cap.server.cap_server": cap_server_cap_server,
    }

    module_name = "run_script_under_test"
    sys.modules.pop(module_name, None)
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(module_name, RUN_SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
    return module


def test_resolve_file_accepts_repo_prefixed_saved_script_path() -> None:
    module = _load_run_script_module()
    resolved = module.resolve_file(
        "enpire/cap/saved_scripts/examples/pick_object.py"
    )
    assert resolved == EXPECTED_SCRIPT


def test_resolve_file_accepts_repo_relative_path_outside_repo_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_run_script_module()
    monkeypatch.chdir(tmp_path)
    resolved = module.resolve_file(
        "cap/saved_scripts/examples/pick_object.py"
    )
    assert resolved == EXPECTED_SCRIPT


def test_resolve_file_rejects_path_traversal() -> None:
    """Path traversal via ../… must not escape saved_scripts/."""
    import pytest

    module = _load_run_script_module()
    with pytest.raises((FileNotFoundError, ValueError)):
        module.resolve_file("cap/saved_scripts/../../../etc/passwd")


def test_resolve_file_rejects_tilde_expansion_to_outside() -> None:
    """~/ paths that resolve outside the repo must not be accepted via saved-scripts."""
    import pytest

    module = _load_run_script_module()
    with pytest.raises(FileNotFoundError):
        module.resolve_file("~/.ssh/id_rsa")


def test_resolve_file_rejects_empty_string() -> None:
    """Empty string must raise FileNotFoundError."""
    import pytest

    module = _load_run_script_module()
    with pytest.raises(FileNotFoundError):
        module.resolve_file("")


def test_resolve_file_rejects_whitespace_only() -> None:
    """Whitespace-only input must raise FileNotFoundError."""
    import pytest

    module = _load_run_script_module()
    with pytest.raises(FileNotFoundError):
        module.resolve_file("   ")
