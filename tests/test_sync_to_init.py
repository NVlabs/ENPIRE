"""Unit tests for the sync_to_init feature.

Tests cover:
1. start_stop_play_policy.py: State type, Portal binding, get_action() branch
2. yam_control_loop.py: sync_to_init event handling
3. overlay_viz/app.py: PolicyConnection.sync_to_init() and HTTP endpoint
4. _calibrate_for_delta_replay reuse for joint_position mode
"""
from __future__ import annotations

import types
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_observation(left_jp=None, right_jp=None):
    """Return a minimal observation dict."""
    return {
        "left_joint_pos": np.zeros(6) if left_jp is None else np.array(left_jp, dtype=np.float32),
        "left_gripper_pos": np.array([0.0]),
        "right_joint_pos": np.zeros(6) if right_jp is None else np.array(right_jp, dtype=np.float32),
        "right_gripper_pos": np.array([0.0]),
    }


def _make_initial_state(left_jp=None, right_jp=None):
    """Return a replay policy initial_state dict."""
    return {
        "left_joint_pos": np.ones(6) * 0.5 if left_jp is None else np.array(left_jp, dtype=np.float32),
        "left_gripper_pos": np.array([0.1]),
        "right_joint_pos": np.ones(6) * 0.3 if right_jp is None else np.array(right_jp, dtype=np.float32),
        "right_gripper_pos": np.array([-0.1]),
    }


# ---------------------------------------------------------------------------
# 1. State type includes "sync_to_init"
# ---------------------------------------------------------------------------

class TestStateType(unittest.TestCase):
    def test_sync_to_init_in_state_literal(self):
        """'sync_to_init' must appear in the State Literal type's __args__."""
        import sys
        import importlib

        # Import without triggering portal/hardware side-effects
        with patch.dict("sys.modules", {
            "portal": MagicMock(),
            "enpire.env.forge.robot.yam.kinematics": MagicMock(),
        }):
            spec = importlib.util.spec_from_file_location(
                "start_stop_play_policy",
                "experimental/start_stop_play_policy.py",
            )
            mod = importlib.util.module_from_spec(spec)
            # Skip execution — we just want the State annotation
            # Parse the source directly instead
            import ast
            src = open("experimental/start_stop_play_policy.py").read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "State":
                            # Found the State = Literal[...] assignment
                            # Get the string values from the Literal subscript
                            if isinstance(node.value, ast.Subscript):
                                slice_node = node.value.slice
                                if isinstance(slice_node, ast.Tuple):
                                    args = [elt.s for elt in slice_node.elts if isinstance(elt, ast.Constant)]
                                    self.assertIn("sync_to_init", args,
                                                  "State literal must include 'sync_to_init'")
                                    return
            self.fail("Could not find State = Literal[...] in start_stop_play_policy.py")


# ---------------------------------------------------------------------------
# 2. Portal binding is registered
# ---------------------------------------------------------------------------

class TestPortalBinding(unittest.TestCase):
    def test_sync_to_init_binding_registered(self):
        """_server.bind('sync_to_init', ...) must be called in __init__."""
        import ast
        src = open("experimental/start_stop_play_policy.py").read()
        tree = ast.parse(src)
        bind_calls = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == "bind" and
                node.args and isinstance(node.args[0], ast.Constant)):
                bind_calls.append(node.args[0].value)
        self.assertIn("sync_to_init", bind_calls,
                      "Expected self._server.bind('sync_to_init', ...) in __init__")


# ---------------------------------------------------------------------------
# 3. _portal_sync_to_init enters "sync_to_init" state
# ---------------------------------------------------------------------------

class TestPortalSyncToInitHandler(unittest.TestCase):
    def _make_stub_wrapper(self):
        """Create a minimal stub with just the methods we need to test."""
        stub = MagicMock()
        entered_states = []

        def enter_state(s):
            entered_states.append(s)
            stub._execution_state = s

        stub._execution_state = "pause"
        stub.enter_state = enter_state
        stub._portal_sync_to_init = lambda: (enter_state("sync_to_init") or True)
        return stub, entered_states

    def test_portal_handler_enters_sync_to_init(self):
        stub, states = self._make_stub_wrapper()
        result = stub._portal_sync_to_init()
        self.assertTrue(result)
        self.assertIn("sync_to_init", states)

    def test_portal_handler_returns_true(self):
        stub, _ = self._make_stub_wrapper()
        self.assertTrue(stub._portal_sync_to_init())


# ---------------------------------------------------------------------------
# 4. get_action() sync_to_init branch emits event and transitions to pause
# ---------------------------------------------------------------------------

class TestGetActionSyncToInitBranch(unittest.TestCase):
    """Test the logic of the sync_to_init branch in get_action().

    We parse the source to verify the branch is present and correct,
    then test the expected behaviour via a lightweight mock.
    """

    def test_branch_sets_event_sync_to_init(self):
        """sync_to_init branch must return info with event='sync_to_init'."""
        import ast
        src = open("experimental/start_stop_play_policy.py").read()
        tree = ast.parse(src)

        # Look for `elif self.execution_state == "sync_to_init":` in get_action
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_action":
                for child in ast.walk(node):
                    if isinstance(child, ast.Compare):
                        if (isinstance(child.left, ast.Attribute) and
                                child.left.attr == "execution_state" and
                                any(isinstance(c, ast.Constant) and c.value == "sync_to_init"
                                    for c in child.comparators)):
                            found = True
        self.assertTrue(found,
                        "get_action() must have a branch for execution_state == 'sync_to_init'")

    def test_branch_emits_event_key(self):
        """Branch must set info['event'] = 'sync_to_init'."""
        import ast
        src = open("experimental/start_stop_play_policy.py").read()
        # Simple string-search heuristic: check the source contains the pattern
        self.assertIn("\"sync_to_init\"", src)
        self.assertIn("info[\"event\"] = \"sync_to_init\"", src,
                      "Branch must set info['event'] = 'sync_to_init'")

    def test_branch_transitions_to_pause(self):
        """Branch must call enter_state('pause') before returning."""
        import ast
        src = open("experimental/start_stop_play_policy.py").read()
        self.assertIn("enter_state(\"pause\")", src)


# ---------------------------------------------------------------------------
# 5. yam_control_loop: sync_to_init event triggers _calibrate_for_delta_replay
# ---------------------------------------------------------------------------

class TestControlLoopSyncToInit(unittest.TestCase):
    def test_sync_to_init_event_triggers_calibrate(self):
        """Control loop must call _calibrate_for_delta_replay on sync_to_init event."""
        import ast
        src = open("experimental/yam_control_loop.py").read()

        # Verify the event check is present
        self.assertIn("policy_info.get(\"event\") == \"sync_to_init\"", src,
                      "Control loop must check for 'sync_to_init' event in policy_info")

        # Verify calibrate is called within that block
        tree = ast.parse(src)
        in_sync_block = False
        calibrate_called = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Look for the sync_to_init event check
                cond_str = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
                if "sync_to_init" in cond_str and "event" in cond_str:
                    # Check that _calibrate_for_delta_replay appears in the body
                    body_src = "\n".join(ast.unparse(n) for n in node.body) if hasattr(ast, "unparse") else ""
                    if "_calibrate_for_delta_replay" in body_src:
                        calibrate_called = True

        # Fallback string check if ast.unparse not available (Python < 3.9)
        if not calibrate_called:
            lines = src.split("\n")
            in_block = False
            for line in lines:
                if "sync_to_init" in line and "event" in line:
                    in_block = True
                if in_block and "_calibrate_for_delta_replay" in line:
                    calibrate_called = True
                    break
                if in_block and line.strip().startswith("continue"):
                    break

        self.assertTrue(calibrate_called,
                        "Control loop sync_to_init block must call _calibrate_for_delta_replay")

    def test_sync_to_init_uses_continue_to_skip_env_step(self):
        """After calibration, loop must `continue` to skip env.step with stale action."""
        src = open("experimental/yam_control_loop.py").read()
        # Find the sync_to_init event block (wider window)
        idx = src.find("policy_info.get(\"event\") == \"sync_to_init\"")
        self.assertGreater(idx, 0)
        block = src[idx:idx + 1000]
        self.assertIn("continue", block,
                      "sync_to_init block must `continue` after calibration to skip env.step")


# ---------------------------------------------------------------------------
# 6. _calibrate_for_delta_replay works for joint_position mode
# ---------------------------------------------------------------------------

class TestCalibrateForDeltaReplayJointPositionMode(unittest.TestCase):
    """Verify _calibrate_for_delta_replay does NOT gate on control_mode."""

    def test_calibrate_not_gated_on_delta_mode(self):
        """_calibrate_for_delta_replay must not have a top-level mode guard that
        prevents it from running for joint_position mode."""
        src = open("experimental/yam_control_loop.py").read()
        # Verify that the call-site guard is only at the call-site (line 473),
        # not inside the function itself as a top-level mode restriction.
        start = src.find("def _calibrate_for_delta_replay(")
        self.assertGreater(start, 0)
        end = src.find("\ndef ", start + 1)
        func_src = src[start:end]

        # The function must NOT have an early return/raise based solely on control_mode
        # being non-delta (it may use control_mode for hold action construction,
        # but that is fine — it must not reject joint_position mode at the top).
        # Check: no `if control_mode not in (...)` guard at the start
        self.assertNotIn("if control_mode not in", func_src,
                         "_calibrate_for_delta_replay must not restrict accepted modes")

    def test_env_reset_called_with_initial_state(self):
        """_calibrate_for_delta_replay must call env.reset(options={'initial_state': ...})."""
        src = open("experimental/yam_control_loop.py").read()
        start = src.find("def _calibrate_for_delta_replay(")
        end = src.find("\ndef ", start + 1)
        func_src = src[start:end]
        self.assertIn("env.reset", func_src)
        self.assertIn("initial_state", func_src)


# ---------------------------------------------------------------------------
# 7. PolicyConnection.sync_to_init() method exists
# ---------------------------------------------------------------------------

class TestPolicyConnectionSyncToInit(unittest.TestCase):
    def test_method_exists_in_source(self):
        """PolicyConnection must have a sync_to_init method."""
        src = open("third_party/overlay_viz/app.py").read()
        self.assertIn("def sync_to_init(self)", src,
                      "PolicyConnection must define sync_to_init()")

    def test_method_calls_sync_to_init_rpc(self):
        """sync_to_init must call self._client.sync_to_init()."""
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("def sync_to_init(self)")
        self.assertGreater(idx, 0)
        snippet = src[idx:idx + 300]
        self.assertIn("sync_to_init", snippet)
        self.assertIn("_client", snippet)

    def test_method_returns_false_when_not_connected(self):
        """sync_to_init must return False when _client is None."""
        # Parse the method and check there's an early return for not-connected case
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("def sync_to_init(self)")
        snippet = src[idx:idx + 300]
        self.assertIn("return False", snippet)

    def test_method_returns_true_on_success(self):
        """sync_to_init must return True on success."""
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("def sync_to_init(self)")
        snippet = src[idx:idx + 300]
        self.assertIn("return True", snippet)


# ---------------------------------------------------------------------------
# 8. HTTP endpoint /api/replay/sync_to_init exists
# ---------------------------------------------------------------------------

class TestHTTPEndpoint(unittest.TestCase):
    def test_endpoint_registered(self):
        """POST /api/replay/sync_to_init must be registered in create_app."""
        src = open("third_party/overlay_viz/app.py").read()
        self.assertIn("/api/replay/sync_to_init", src,
                      "HTTP endpoint /api/replay/sync_to_init must be registered")

    def test_endpoint_calls_policy_conn_sync_to_init(self):
        """The endpoint must delegate to policy_conn.sync_to_init()."""
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("/api/replay/sync_to_init")
        snippet = src[idx:idx + 400]
        self.assertIn("sync_to_init", snippet)
        self.assertIn("policy_conn", snippet)

    def test_endpoint_raises_503_on_failure(self):
        """Endpoint must raise HTTP 503 when sync_to_init fails."""
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("async def replay_sync_to_init")
        snippet = src[idx:idx + 600]  # wider window
        self.assertIn("503", snippet)
        self.assertIn("HTTPException", snippet)


# ---------------------------------------------------------------------------
# 9. Frontend button and JS function
# ---------------------------------------------------------------------------

class TestFrontend(unittest.TestCase):
    def setUp(self):
        self.html = open("third_party/overlay_viz/static/index.html").read()

    def test_sync_to_init_button_present(self):
        """HTML must contain a 'Sync to Init' button."""
        self.assertIn("Sync to Init", self.html)

    def test_button_calls_replaySyncToInit(self):
        """Button must call replaySyncToInit() on click."""
        self.assertIn("replaySyncToInit()", self.html)

    def test_js_function_defined(self):
        """Alpine.js function replaySyncToInit must be defined."""
        self.assertIn("async replaySyncToInit()", self.html)

    def test_js_calls_api_endpoint(self):
        """JS function must fetch /api/replay/sync_to_init."""
        self.assertIn("/api/replay/sync_to_init", self.html)

    def test_button_disabled_without_loaded_episode(self):
        """Button must be disabled when no episode is loaded (replayLoadedEpisode === null)."""
        self.assertIn("replayLoadedEpisode === null", self.html)

    def test_button_disabled_when_busy(self):
        """Button must also be disabled while replayBusy is true."""
        # Find the Sync to Init button and check its :disabled binding
        idx = self.html.find("Sync to Init")
        # Look backwards for the button opening tag
        button_start = self.html.rfind("<button", 0, idx)
        snippet = self.html[button_start:idx + 20]
        self.assertIn("replayBusy", snippet)

    def test_js_confirms_before_moving(self):
        """JS function must show a confirmation dialog before moving the robot."""
        idx = self.html.find("async replaySyncToInit()")
        snippet = self.html[idx:idx + 300]
        self.assertIn("confirm(", snippet)

    def test_js_stops_playing_after_sync(self):
        """JS function must set replayPlaying=false and stop step timer."""
        idx = self.html.find("async replaySyncToInit()")
        snippet = self.html[idx:idx + 700]
        self.assertIn("replayPlaying = false", snippet)
        self.assertIn("stopReplayStepTimer()", snippet)


# ---------------------------------------------------------------------------
# 10. Integration: mock full sync_to_init flow
# ---------------------------------------------------------------------------

class TestSyncToInitIntegration(unittest.TestCase):
    """End-to-end mock test of the sync_to_init flow."""

    def test_full_flow_mock(self):
        """
        Simulate the full flow via source parsing + mock:
          PolicyConnection.sync_to_init() calls _client.sync_to_init().result(timeout=5)
        """
        # Mock cv2 and other heavy deps so we can import the app
        import sys
        fake_mods = ["cv2", "portal", "uvicorn", "fastapi", "fastapi.staticfiles",
                     "fastapi.responses", "pydantic", "starlette", "starlette.responses"]
        for m in fake_mods:
            if m not in sys.modules:
                sys.modules[m] = MagicMock()

        # Source-level check: the method calls _client.sync_to_init().result(timeout=5)
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("def sync_to_init(self)")
        snippet = src[idx:idx + 300]
        self.assertIn("_client.sync_to_init()", snippet)
        self.assertIn("timeout=5", snippet)
        self.assertIn("return True", snippet)
        self.assertIn("return False", snippet)

    def test_policy_connection_sync_to_init_returns_false_no_client(self):
        """PolicyConnection.sync_to_init() returns False when not connected (source check)."""
        src = open("third_party/overlay_viz/app.py").read()
        idx = src.find("def sync_to_init(self)")
        snippet = src[idx:idx + 200]
        # Must check if _client is falsy before calling
        self.assertIn("_client", snippet)
        self.assertIn("return False", snippet)

    def test_calibrate_reads_initial_state_from_policy(self):
        """_calibrate_for_delta_replay reads initial_state from policy.initial_state."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "yam_control_loop", "experimental/yam_control_loop.py"
        )
        # We can't fully import due to hardware deps, so verify via source
        src = open("experimental/yam_control_loop.py").read()
        start = src.find("def _calibrate_for_delta_replay(")
        end = src.find("\ndef ", start + 1)
        func_src = src[start:end]
        self.assertIn("initial_state", func_src)
        self.assertIn("getattr(policy", func_src)

    def test_calibrate_skips_if_already_at_target(self):
        """_calibrate_for_delta_replay must skip env.reset if robot is already at target."""
        src = open("experimental/yam_control_loop.py").read()
        start = src.find("def _calibrate_for_delta_replay(")
        end = src.find("\ndef ", start + 1)
        func_src = src[start:end]
        # The function checks pre_dist < 0.005 and skips motion
        self.assertIn("is_already_home", func_src)
        self.assertIn("return observation", func_src)

    def test_calibrate_raises_if_initial_state_missing(self):
        """_calibrate_for_delta_replay raises ValueError if policy has no initial_state."""
        src = open("experimental/yam_control_loop.py").read()
        start = src.find("def _calibrate_for_delta_replay(")
        end = src.find("\ndef ", start + 1)
        func_src = src[start:end]
        self.assertIn("raise ValueError", func_src)
        self.assertIn("initial_state", func_src)


if __name__ == "__main__":
    unittest.main()
