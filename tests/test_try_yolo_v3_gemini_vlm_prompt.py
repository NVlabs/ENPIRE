# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "golden_prompts/table_bussing/useful_tools/working_well/try_yolo_v3_gemini_vlm.py"
)


def _load_build_prompt():
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source, filename=str(SCRIPT_PATH))
    build_prompt_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_prompt"
    )
    module = ast.Module(body=[build_prompt_node], type_ignores=[])
    namespace = {}
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["build_prompt"]


class BuildPromptTest(unittest.TestCase):
    def test_build_prompt_is_target_centric(self):
        build_prompt = _load_build_prompt()
        prompt = build_prompt(
            [
                ("fruits", "on", "blue plate"),
                ("cups", "in", "cardboard box"),
            ]
        )

        self.assertTrue(
            prompt.startswith("You are given images of a robot table-bussing workspace.")
        )
        self.assertIn("1. Which visible loose fruits should go on the blue plate?", prompt)
        self.assertIn("2. Which visible loose cups should go in the cardboard box?", prompt)
        self.assertIn('"blue plate": ["<object_name>", ...]', prompt)
        self.assertIn('"cardboard box": ["<object_name>", ...]', prompt)
        self.assertIn("STRICT JSON", prompt)
        self.assertNotIn("unsorted_", prompt)
        self.assertIn(
            "Do not include robot parts, containers/targets themselves", prompt
        )


if __name__ == "__main__":
    unittest.main()
