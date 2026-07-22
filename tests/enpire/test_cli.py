# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from enpire.cli import main


def test_tools_list_is_available_without_importing_tool_implementations(capsys):
    assert main(["tools", "list", "--category", "vision"]) == 0
    output = capsys.readouterr().out
    assert "vision.segment" in output
    assert "vision.detect" in output


def test_tools_list_json(capsys):
    assert main(["tools", "list", "--category", "vlm", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "vlm.query"
    assert isinstance(payload[0]["available"], bool)


def test_examples_list(capsys):
    assert main(["examples", "list"]) == 0
    assert "00_hello_environment" in capsys.readouterr().out


def test_examples_run(tmp_path, capsys):
    output = tmp_path / "example-run"
    assert (
        main(
            [
                "examples",
                "run",
                "00_hello_environment",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "success=True" in capsys.readouterr().out
    assert (output / "result.json").is_file()


def test_skills_list(capsys):
    assert main(["skills", "list"]) == 0
    assert "manipulation.pick_and_place" in capsys.readouterr().out


def test_doctor(capsys):
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "enpire: 0.1.0" in output
    assert "hardware: not probed" in output


def test_station_init_and_show(tmp_path, capsys):
    assert (
        main(
            [
                "station",
                "init",
                "--station",
                "yam-test",
                "--config-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "yam-test.yaml").is_file()
    capsys.readouterr()

    assert (
        main(
            [
                "station",
                "show",
                "--station",
                "yam-test",
                "--config-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["station_id"] == "yam-test"
