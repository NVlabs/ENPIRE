# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
YAM_CONFIG_ROOT = REPO_ROOT / "enpire/env/forge/robot/models/station/curobo"
CUROBO_V08_SHA = "4ea77366ca48ee453e7df139e39fa6532af49f3b"


def test_vendored_curobo_declares_v08_apache_release() -> None:
    metadata = tomllib.loads((REPO_ROOT / "third_party/curobo/pyproject.toml").read_text())
    assert metadata["project"]["license"]["text"] == "Apache-2.0"
    tag = subprocess.check_output(
        ["git", "-C", "third_party/curobo", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    assert tag == "v0.8.0"
    commit = subprocess.check_output(
        ["git", "-C", "third_party/curobo", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    assert commit == CUROBO_V08_SHA
    assert "Apache License" in (REPO_ROOT / "third_party/curobo/LICENSE").read_text()


def test_enpire_curobo_legal_metadata_has_no_v07_restriction() -> None:
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    licenses = (REPO_ROOT / "THIRD_PARTY_LICENSES.md").read_text()
    combined = notices + licenses
    assert "v0.8.0" in licenses
    assert "Apache-2.0" in combined
    for obsolete in (
        "v0.7.7",
        "NVIDIA Research License",
        "NVIDIA Platforms",
        "Use Limitation",
    ):
        assert obsolete not in combined


def test_all_yam_curobo_configs_use_v08_schema() -> None:
    config_paths = sorted(YAM_CONFIG_ROOT.glob("yam*.yml"))
    assert len(config_paths) == 5
    legacy_keys = {
        "use_usd_kinematics",
        "usd_path",
        "usd_robot_root",
        "isaac_usd_path",
        "usd_flip_joints",
        "usd_flip_joint_limits",
        "ee_link",
        "link_names",
    }
    for path in config_paths:
        kinematics = yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]
        assert kinematics["format_version"] == 2.0, path
        assert kinematics["tool_frames"] == ["left_grasp", "right_grasp"], path
        assert legacy_keys.isdisjoint(kinematics), path
        cspace = kinematics["cspace"]
        assert "retract_config" not in cspace, path
        assert len(cspace["joint_names"]) == len(cspace["default_joint_position"]), path


def test_enpire_curobo_integrations_do_not_import_removed_v07_modules() -> None:
    source_paths = [
        REPO_ROOT / "enpire/env/forge/experimental/motion_planner_curobo.py",
        REPO_ROOT / "enpire/env/forge/experimental/motion_planner_curobo_panda.py",
        REPO_ROOT / "enpire/env/forge/experimental/curobo_depth_world.py",
        REPO_ROOT / "third_party/pyroki/benchmark/ik_benchmark.py",
    ]
    removed_modules = (
        "curobo.wrap.",
        "curobo.geom.types",
        "curobo.geom.sdf",
        "curobo.types.base",
        "curobo.types.math",
        "curobo.types.state",
        "curobo.types.robot",
        "curobo.util.logger",
    )
    for path in source_paths:
        source = path.read_text()
        for removed_module in removed_modules:
            assert removed_module not in source, (path, removed_module)


def test_cap_ui_sources_have_apache_spdx_headers() -> None:
    source_root = REPO_ROOT / "enpire/env/forge/cap/ui/src"
    source_paths = sorted(
        path for path in source_root.rglob("*") if path.suffix in {".ts", ".tsx", ".css"}
    )
    assert len(source_paths) == 28
    for path in source_paths:
        prefix = path.read_text()[:300]
        assert prefix.startswith("/* SPDX-FileCopyrightText:"), path
        assert "All rights reserved. */\n" in prefix, path
        assert "/* SPDX-License-Identifier: Apache-2.0 */\n" in prefix, path


def test_pyroki_benchmark_has_no_proprietary_distribution_notice() -> None:
    source = (REPO_ROOT / "third_party/pyroki/benchmark/ik_benchmark.py").read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in source[:300]
    assert "distribution without an express license agreement" not in source.lower()
    assert "strictly prohibited" not in source.lower()
