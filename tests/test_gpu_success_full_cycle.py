from types import SimpleNamespace

from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.gpu_success_full_cycle import maybe_request_gpu_success_full_cycle


class _Env:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.last_episode_dir = output_dir / "episode_success"
        self.finalize_calls = []

    def finalize_episode(self, **kwargs):
        self.finalize_calls.append(kwargs)


def _ctx(tmp_path, *, enabled=True, event="success", state="learn"):
    request_path = tmp_path / "request.json"
    env = _Env(tmp_path / "episodes")
    return SimpleNamespace(
        cfg=DataCollectionConfig(
            task_name="gpu_insertion",
            gpu_success_full_cycle_enabled=enabled,
            gpu_success_full_cycle_request_path=str(request_path),
        ),
        env=env,
        state_machine=SimpleNamespace(state=state),
        terminal_event=event,
        timing_log=SimpleNamespace(calls=[], log=lambda *args, **kwargs: None),
    ), request_path


def test_gpu_success_full_cycle_finalizes_episode_and_writes_request(tmp_path):
    ctx, request_path = _ctx(tmp_path)

    handled = maybe_request_gpu_success_full_cycle(
        ctx,
        "success",
        {"source": "spacemouse", "button": 0},
    )

    assert handled is True
    assert ctx.env.finalize_calls == [
        {
            "discard_episode": False,
            "episode_terminal_event": "success",
            "episode_terminal_reward": 1.0,
            "episode_terminal_done": True,
        }
    ]
    text = request_path.read_text()
    assert '"event": "success"' in text
    assert '"task_name": "gpu_insertion"' in text
    assert '"button": 0' in text
    assert ctx.terminal_event is None


def test_gpu_success_full_cycle_disabled_keeps_normal_path(tmp_path):
    ctx, request_path = _ctx(tmp_path, enabled=False)

    handled = maybe_request_gpu_success_full_cycle(ctx, "success")

    assert handled is False
    assert ctx.env.finalize_calls == []
    assert not request_path.exists()


def test_gpu_success_full_cycle_ignores_failure_event(tmp_path):
    ctx, request_path = _ctx(tmp_path, event="fail")

    handled = maybe_request_gpu_success_full_cycle(ctx, "fail")

    assert handled is False
    assert ctx.env.finalize_calls == []
    assert not request_path.exists()
