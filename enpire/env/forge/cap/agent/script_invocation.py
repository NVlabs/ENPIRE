"""Call a saved CaP function and expose its result through normal run artifacts."""

from __future__ import annotations

import inspect
import json


def validate_invocation(function, kwargs):
    if function is not None and (
        not isinstance(function, str) or not function.isidentifier() or function.startswith("_")
    ):
        raise ValueError("script_function must be a public Python function name")
    if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
        raise ValueError("script_kwargs must be a mapping with string keys")
    if function is None and kwargs:
        raise ValueError("script_kwargs requires script_function")


def invoke_script_function(namespace, function, kwargs, script_path):
    """Invoke only a function defined by this file, with machine-readable output.

    This is an opt-in extension to the existing top-level script execution.
    Exceptions still reach run_script's ordinary error/exit handling.
    """
    validate_invocation(function, kwargs)
    fn = namespace.get(function)
    if not inspect.isfunction(fn) or fn.__code__.co_filename != str(script_path):
        raise ValueError(f"{function!r} must be a function defined in {script_path.name}")
    # Replace default task reward/VLM evaluation with command status. Do this
    # before invocation so a raised exception cannot leave a stale success.
    info = {"success": False, "reward": 0.0, "script_function": function}
    namespace["get_task_info"] = lambda: info
    try:
        result = fn(**kwargs)
        # Reject NaN or unserializable reports instead of silently stringifying
        # arrays into JSON strings. The two atomic skills return plain JSON data.
        json.dumps(result, allow_nan=False)
    except Exception as exc:
        info["error"] = str(exc)
        if hasattr(exc, "report"):
            info["return_value"] = exc.report
        raise
    info.update(success=True, return_value=result)
    return result
