from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from enpire.env.forge.cap.agent.agent_session import AgentRunSession
from enpire.env.forge.cap.agent.skill_library import SkillLibrary


def _skill_source(name: str) -> str:
    return (
        "from skill_library.namespace import *\n"
        "from cap.agent.skill_registry import skill\n\n"
        "@skill\n"
        f"def {name}(side):\n"
        f"    return side\n"
    )


class SnapshotWorkspaceTests(unittest.TestCase):
    def test_keep_and_discard_restore_skill_library_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)

            session = AgentRunSession(run_dir=run_dir, task="task", env_name=None)
            skill_library = SkillLibrary(run_dir)
            skill_library.generate_namespace({})
            session._skill_library = skill_library

            session.initialize_skill_library_baseline()

            skill_library.append_skill("first_skill", _skill_source("first_skill_v1"), 0)
            decision_0 = session.record_skill_library_trial(
                0,
                score=0.2,
                success=True,
            )
            self.assertEqual(decision_0, "keep")
            self.assertTrue((session.baseline_skill_library_dir() / "first_skill.py").exists())

            skill_library.append_skill("second_skill", _skill_source("second_skill_v1"), 1)
            decision_1 = session.record_skill_library_trial(
                1,
                score=0.1,
                success=False,
            )
            self.assertEqual(decision_1, "discard")
            self.assertEqual(skill_library.total_skills, 1)
            self.assertIn("first_skill_v1", skill_library._index)
            self.assertNotIn("second_skill_v1", skill_library._index)
            self.assertTrue((session.trial_skill_library_dir(1) / "second_skill.py").exists())

            state = session.snapshot_state_path().read_text(encoding="utf-8")
            self.assertIn('"baseline_score": 0.2', state)
            self.assertIn('"decision": "keep"', state)
            self.assertIn('"decision": "discard"', state)


if __name__ == "__main__":
    unittest.main()
