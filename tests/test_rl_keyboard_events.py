import queue
from types import SimpleNamespace

from enpire.policy.rl.config import DataCollectionConfig, load_yaml_defaults
from enpire.policy.rl.keyboard_events import dispatch_keyboard_events


def _ctx(cfg):
    return SimpleNamespace(
        cfg=cfg,
        external_event_queue=queue.Queue(),
        fastapi_base_url="http://127.0.0.1:8203",
        state_machine=SimpleNamespace(state="idle"),
    )


def _keyboard_info(*keys):
    return {"_just_pressed_keys": set(keys), "shift_held": False}


def test_key_a_enters_author_mode_when_enabled():
    ctx = _ctx(DataCollectionConfig(enable_author_mode=True))

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_A"))

    assert ctx.external_event_queue.get_nowait() == ("author", {})


def test_key_a_does_not_enter_author_mode_when_disabled():
    ctx = _ctx(DataCollectionConfig(enable_author_mode=False))

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_A"))

    assert ctx.external_event_queue.empty()


def test_escape_can_still_discard_author_mode_when_author_mode_disabled():
    ctx = _ctx(DataCollectionConfig(enable_author_mode=False))

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_ESC"))

    assert ctx.external_event_queue.get_nowait() == ("discard_author", {})


def test_key_e_starts_auto_eval():
    ctx = _ctx(DataCollectionConfig())

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_E"))

    assert ctx.external_event_queue.get_nowait() == ("auto_eval_start", {})


def test_keyboard_success_and_fail_are_configurable_terminal_labels():
    ctx = _ctx(
        DataCollectionConfig(
            keyboard_success_key="KEY_Y",
            keyboard_fail_key="KEY_N",
        )
    )

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_Y", "KEY_N"))

    assert ctx.external_event_queue.get_nowait() == (
        "success",
        {"source": "keyboard"},
    )
    assert ctx.external_event_queue.get_nowait() == (
        "fail",
        {"source": "keyboard"},
    )


def test_keyboard_success_and_fail_can_be_disabled():
    ctx = _ctx(
        DataCollectionConfig(
            keyboard_success_key="",
            keyboard_fail_key="",
        )
    )

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_ENTER", "KEY_BACKSPACE"))

    assert ctx.external_event_queue.empty()


def test_keyboard_home_start_and_parking_are_configurable():
    ctx = _ctx(
        DataCollectionConfig(
            keyboard_home_key="",
            keyboard_start_key="KEY_Y",
            keyboard_parking_key="",
        )
    )

    dispatch_keyboard_events(ctx, _keyboard_info("KEY_H", "KEY_P", "KEY_Y"))

    assert ctx.external_event_queue.get_nowait() == (
        "start",
        {"source": "keyboard"},
    )
    assert ctx.external_event_queue.empty()


def test_yaml_can_disable_author_mode(tmp_path):
    config_file = tmp_path / "task.yaml"
    config_file.write_text("enable_author_mode: false\n")

    cfg = load_yaml_defaults(str(config_file))

    assert cfg.enable_author_mode is False
