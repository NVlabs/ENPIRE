from __future__ import annotations

import importlib
import itertools
import json
import os
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

_NVIDIA_ENV_NAMES = [
    "NVIDIA_API_KEY",
    *(f"NVIDIA_API_KEY_{i}" for i in range(1, 101)),
]
_NVIDIA_CONTROL_ENV_NAMES = [
    "CAP_NVIDIA_TELEMETRY_FILE",
    "CAP_NVIDIA_SCHEDULER_DB",
    "CAP_NVIDIA_REQUEST_DELAY_S",
    "CAP_NVIDIA_MAX_CONCURRENT_PER_KEY",
    "CAP_NVIDIA_ACQUIRE_TIMEOUT_S",
    "CAP_NVIDIA_429_COOLDOWN_S",
    "CAP_NVIDIA_PROVIDER_URL",
    "CAP_NVIDIA_GLOBAL_REQUEST_DELAY_S",
    "CAP_NVIDIA_PROVIDER_MAX_ATTEMPTS",
]


def _load_provider(**env: str):
    for name in [*_NVIDIA_ENV_NAMES, *_NVIDIA_CONTROL_ENV_NAMES]:
        os.environ.pop(name, None)
    os.environ.update(env)

    import enpire.env.forge.cap.agent.providers.nvidia as provider

    return importlib.reload(provider)


class NvidiaProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {name: os.environ.get(name) for name in _NVIDIA_ENV_NAMES}
        self._saved_control_env = {
            name: os.environ.get(name) for name in _NVIDIA_CONTROL_ENV_NAMES
        }

    def tearDown(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name, value in self._saved_control_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_key_discovery_prefers_indexed_keys_and_skips_gaps(self) -> None:
        provider = _load_provider(
            NVIDIA_API_KEY="fallback-key",
            NVIDIA_API_KEY_1="indexed-one",
            NVIDIA_API_KEY_3="indexed-three",
        )

        self.assertEqual(provider.list_nvidia_keys(), ["indexed-one", "indexed-three"])
        self.assertEqual(provider.pick_nvidia_key(None), "indexed-one")
        self.assertEqual(provider.pick_nvidia_key(3), "indexed-three")

    def test_auto_pick_key_uses_process_local_round_robin(self) -> None:
        provider = _load_provider(
            NVIDIA_API_KEY_1="indexed-one",
            NVIDIA_API_KEY_2="indexed-two",
        )
        provider._rr_keys = ["indexed-one", "indexed-two"]
        provider._rr_counter = itertools.count()

        self.assertEqual(provider.auto_pick_nvidia_key(), "indexed-one")
        self.assertEqual(provider.auto_pick_nvidia_key(), "indexed-two")
        self.assertEqual(provider.auto_pick_nvidia_key(), "indexed-one")

    def test_auto_pick_key_reports_missing_key(self) -> None:
        provider = _load_provider()

        with self.assertRaisesRegex(RuntimeError, "No NVIDIA key found"):
            provider.auto_pick_nvidia_key()

    def test_model_temperature_support_filters_claude_4_models(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY="key")

        self.assertFalse(
            provider.nvidia_model_supports_temperature(
                "aws/anthropic/bedrock-claude-opus-4-6"
            )
        )
        self.assertFalse(
            provider.nvidia_model_supports_temperature(
                "aws/anthropic/claude-sonnet-4-5"
            )
        )
        self.assertTrue(
            provider.nvidia_model_supports_temperature("azure/openai/gpt-5.1")
        )

    def test_chat_completions_url_accepts_base_or_full_endpoint(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY="key")

        self.assertEqual(
            provider.chat_completions_url("https://inference-api.nvidia.com/v1/"),
            "https://inference-api.nvidia.com/v1/chat/completions",
        )
        self.assertEqual(
            provider.chat_completions_url(
                "https://inference-api.nvidia.com/v1/chat/completions"
            ),
            "https://inference-api.nvidia.com/v1/chat/completions",
        )

    def test_response_text_coerces_openai_content_variants(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY="key")

        self.assertEqual(
            provider.response_text(
                {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}
            ),
            "hi",
        )
        self.assertEqual(
            provider.response_text({"choices": [{"message": {"content": "plain"}}]}),
            "plain",
        )

    def test_post_chat_completions_uses_bearer_key_and_json_payload(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY="key")
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        def fake_urlopen(request, timeout: float):  # noqa: ANN001
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
            response = provider.post_chat_completions(
                {"model": "azure/openai/gpt-5.1", "messages": []},
                api_key="explicit-key",
                base_url="https://example.test/v1",
            )

        self.assertEqual(response, {"choices": [{"message": {"content": "ok"}}]})
        self.assertEqual(captured["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer explicit-key")
        self.assertEqual(
            captured["body"], {"model": "azure/openai/gpt-5.1", "messages": []}
        )
        self.assertEqual(captured["timeout"], 60.0)

    def test_post_chat_completions_raises_rate_limit_error(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY="key")

        class FakeHTTPError(urllib.error.HTTPError):
            def read(self) -> bytes:
                return b'{"error":{"message":"too many requests"}}'

        def fake_urlopen(_request, timeout: float):  # noqa: ANN001, ARG001
            raise FakeHTTPError(
                url="https://example.test/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )

        with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
            with self.assertRaisesRegex(
                provider.NvidiaRateLimitError, "too many requests"
            ):
                provider.post_chat_completions(
                    {"model": "azure/openai/gpt-5.1", "messages": []},
                    api_key="explicit-key",
                    base_url="https://example.test/v1",
                )

    def test_post_chat_completions_uses_provider_server_when_configured(self) -> None:
        provider = _load_provider(
            NVIDIA_API_KEY_1="secret-one",
            CAP_NVIDIA_PROVIDER_URL="http://127.0.0.1:8765",
        )
        captured: dict[str, object] = {}

        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        def fake_urlopen(request, timeout: float):  # noqa: ANN001
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
            response = provider.post_chat_completions(
                {"model": "azure/openai/gpt-5.1", "messages": []},
                api_key="secret-one",
                base_url="https://example.test/v1",
                telemetry_source="llm",
            )

        self.assertEqual(response, {"choices": [{"message": {"content": "ok"}}]})
        self.assertEqual(
            captured["url"], "http://127.0.0.1:8765/v1/chat/completions"
        )
        headers = {str(k).lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(headers["x-cap-nvidia-source"], "llm")
        self.assertNotIn("authorization", headers)
        self.assertEqual(
            captured["body"], {"model": "azure/openai/gpt-5.1", "messages": []}
        )
        self.assertEqual(captured["timeout"], 60.0)

    def test_provider_server_retries_auth_failures_until_good_key(self) -> None:
        provider = _load_provider(
            NVIDIA_API_KEY_1="bad-one",
            NVIDIA_API_KEY_2="bad-two",
            NVIDIA_API_KEY_3="good-three",
        )
        import enpire.env.forge.cap.agent.providers.nvidia_server as server

        calls: list[int] = []

        def fake_direct(*_args, **_kwargs):  # noqa: ANN002, ANN003
            calls.append(len(calls) + 1)
            if len(calls) < 3:
                raise RuntimeError(
                    "NVIDIA chat completions request failed with HTTP 401: "
                    "Authentication Error"
                )
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(server.nvidia, "post_chat_completions_direct", fake_direct):
            with patch.object(server.time, "sleep", lambda _s: None):
                response = server.post_chat_completions_with_key_retry(
                    {"model": "azure/openai/gpt-5.1", "messages": []},
                    base_url="https://example.test/v1",
                    telemetry_source="llm",
                )

        self.assertEqual(response, {"choices": [{"message": {"content": "ok"}}]})
        self.assertEqual(len(calls), 3)
        self.assertIs(server.nvidia, provider)

    def test_provider_server_retries_transient_409_with_bounded_budget(self) -> None:
        _load_provider(
            NVIDIA_API_KEY_1="key-one",
            NVIDIA_API_KEY_2="key-two",
            CAP_NVIDIA_PROVIDER_MAX_ATTEMPTS="3",
        )
        import enpire.env.forge.cap.agent.providers.nvidia_server as server

        calls: list[int] = []

        def fake_direct(*_args, **_kwargs):  # noqa: ANN002, ANN003
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                raise RuntimeError(
                    "NVIDIA chat completions request failed with HTTP 409: conflict"
                )
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(server.nvidia, "post_chat_completions_direct", fake_direct):
            with patch.object(server.time, "sleep", lambda _s: None):
                response = server.post_chat_completions_with_key_retry(
                    {"model": "azure/openai/gpt-5.1", "messages": []},
                    base_url="https://example.test/v1",
                    telemetry_source="llm",
                )

        self.assertEqual(response, {"choices": [{"message": {"content": "ok"}}]})
        self.assertEqual(len(calls), 2)

    def test_provider_server_does_not_retry_bad_request_errors(self) -> None:
        _load_provider(NVIDIA_API_KEY_1="key-one")
        import enpire.env.forge.cap.agent.providers.nvidia_server as server

        calls: list[int] = []

        def fake_direct(*_args, **_kwargs):  # noqa: ANN002, ANN003
            calls.append(len(calls) + 1)
            raise RuntimeError(
                "NVIDIA chat completions request failed with HTTP 400: bad request"
            )

        with patch.object(server.nvidia, "post_chat_completions_direct", fake_direct):
            with self.assertRaises(server.ProviderRequestError) as cm:
                server.post_chat_completions_with_key_retry(
                    {"model": "azure/openai/gpt-5.1", "messages": []},
                    base_url="https://example.test/v1",
                    telemetry_source="llm",
                )

        self.assertEqual(cm.exception.http_status, 400)
        self.assertEqual(len(calls), 1)

    def test_provider_server_health_check_uses_best_predefined_model_status(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="key-one", NVIDIA_API_KEY_2="key-two")
        import enpire.env.forge.cap.agent.providers.nvidia_server as server

        key_one = provider.nvidia_key_label("key-one")
        key_two = provider.nvidia_key_label("key-two")

        def fake_health_check(*, keys, model, base_url, max_workers):  # noqa: ANN001, ARG001
            self.assertEqual(keys, ["key-one", "key-two"])
            if model == "bad-route":
                return [
                    {
                        "key": key_one,
                        "status": "invalid_auth",
                        "http_status": 401,
                        "error": "auth",
                    },
                    {
                        "key": key_two,
                        "status": "invalid_auth",
                        "http_status": 401,
                        "error": "auth",
                    },
                ]
            return [
                {"key": key_one, "status": "healthy", "http_status": 200, "error": ""},
                {
                    "key": key_two,
                    "status": "invalid_auth",
                    "http_status": 401,
                    "error": "auth",
                },
            ]

        with patch.object(server.nvidia, "health_check_nvidia_keys", fake_health_check):
            rows = server.health_check_predefined_models(
                keys=["key-one", "key-two"],
                models=["bad-route", "good-route"],
                base_url="https://example.test/v1",
            )

        statuses = {row["key"]: row["status"] for row in rows}
        self.assertEqual(statuses[key_one], "healthy")
        self.assertEqual(statuses[key_two], "invalid_auth")

    def test_post_chat_completions_writes_redacted_telemetry_events(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="secret-one")

        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return (
                    b'{"choices":[{"message":{"content":"ok"}}],'
                    b'"usage":{"prompt_tokens":11,"completion_tokens":7,'
                    b'"total_tokens":18}}'
                )

        def fake_urlopen(_request, timeout: float):  # noqa: ANN001, ARG001
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nvidia_requests.jsonl"
            os.environ["CAP_NVIDIA_TELEMETRY_FILE"] = str(path)
            with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
                provider.post_chat_completions(
                    {"model": "azure/openai/gpt-5.1", "messages": []},
                    api_key="secret-one",
                    base_url="https://example.test/v1",
                    telemetry_source="phase5_reflection",
                )

            raw = path.read_text(encoding="utf-8")
            events = [json.loads(line) for line in raw.splitlines()]

        self.assertEqual([event["event"] for event in events], ["start", "end"])
        self.assertEqual(events[0]["source"], "phase5_reflection")
        self.assertEqual(events[0]["model"], "azure/openai/gpt-5.1")
        self.assertRegex(events[0]["key"], r"^key01:[0-9a-f]{6}$")
        self.assertEqual(events[1]["request_id"], events[0]["request_id"])
        self.assertEqual(events[1]["status"], "ok")
        self.assertEqual(events[1]["http_status"], 200)
        self.assertGreaterEqual(events[1]["latency_ms"], 0.0)
        self.assertEqual(events[1]["prompt_tokens"], 11)
        self.assertEqual(events[1]["completion_tokens"], 7)
        self.assertEqual(events[1]["total_tokens"], 18)
        self.assertNotIn("secret-one", raw)

    def test_post_chat_completions_writes_telemetry_for_http_errors(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="secret-one")

        class FakeHTTPError(urllib.error.HTTPError):
            def read(self) -> bytes:
                return b'{"error":{"message":"too many requests"}}'

        def fake_urlopen(_request, timeout: float):  # noqa: ANN001, ARG001
            raise FakeHTTPError(
                url="https://example.test/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nvidia_requests.jsonl"
            os.environ["CAP_NVIDIA_TELEMETRY_FILE"] = str(path)
            with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
                with self.assertRaisesRegex(
                    provider.NvidiaRateLimitError, "too many requests"
                ):
                    provider.post_chat_completions(
                        {"model": "azure/openai/gpt-5.1", "messages": []},
                        api_key="secret-one",
                        base_url="https://example.test/v1",
                        telemetry_source="skill_reflection",
                    )
            events = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(events[-1]["event"], "end")
        self.assertEqual(events[-1]["source"], "skill_reflection")
        self.assertEqual(events[-1]["status"], "error")
        self.assertEqual(events[-1]["http_status"], 429)
        self.assertIn("too many requests", events[-1]["error"])
        self.assertNotIn("secret-one", json.dumps(events))

    def test_sanitizes_gateway_errors_before_telemetry_or_raise(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="secret-one")
        gateway_error = (
            "Authentication Error, Invalid proxy server token passed. "
            "Received API Key = sk-testsecretRnvg, Key Hash (Token) "
            "=d191c9b486195f3df59f51337182bba40d3dae8b03c358a7a13c7c871b907617."
        )

        sanitized = provider.sanitize_nvidia_error(gateway_error)

        self.assertIn("Received API Key = [redacted]", sanitized)
        self.assertIn("Key Hash (Token) = [redacted]", sanitized)
        self.assertNotIn("sk-testsecretRnvg", sanitized)
        self.assertNotIn("d191c9b486195f3df59f51337182bba40d3dae8b03c", sanitized)

    def test_health_check_filters_invalid_key_and_scheduler_uses_healthy_key(self) -> None:
        provider = _load_provider(
            NVIDIA_API_KEY_1="bad-key",
            NVIDIA_API_KEY_2="good-key",
        )
        captured_auth: list[str] = []

        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        class FakeHTTPError(urllib.error.HTTPError):
            def read(self) -> bytes:
                return (
                    b'{"error":{"message":"Authentication Error, Received API Key = '
                    b'sk-bad, Key Hash (Token) =abcdefabcdefabcdefabcdefabcdef"}}'
                )

        def fake_urlopen(request, timeout: float):  # noqa: ANN001, ARG001
            auth = dict(request.header_items())["Authorization"]
            captured_auth.append(auth)
            if auth == "Bearer bad-key":
                raise FakeHTTPError(
                    url="https://example.test/v1/chat/completions",
                    code=401,
                    msg="Unauthorized",
                    hdrs=None,
                    fp=None,
                )
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nvidia.sqlite"
            provider.init_nvidia_scheduler(
                db_path,
                keys=provider.list_nvidia_keys(),
                min_interval_s=0.0,
                max_concurrent_per_key=1,
            )
            with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
                results = provider.health_check_nvidia_keys(
                    model="azure/openai/gpt-5.1",
                    base_url="https://example.test/v1",
                    timeout=1.0,
                    max_workers=1,
                )
                provider.apply_nvidia_health_results(results)
                provider.post_chat_completions(
                    {"model": "azure/openai/gpt-5.1", "messages": []},
                    api_key=None,
                    base_url="https://example.test/v1",
                )

        statuses = {r["key"]: r["status"] for r in results}
        bad_label = provider.nvidia_key_label("bad-key")
        good_label = provider.nvidia_key_label("good-key")
        self.assertEqual(statuses[bad_label], "invalid_auth")
        self.assertEqual(statuses[good_label], "healthy")
        self.assertEqual(captured_auth[-1], "Bearer good-key")

    def test_health_check_uses_non_degenerate_probe(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="key-one")
        captured_payload: dict[str, object] = {}

        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        def fake_urlopen(request, timeout: float):  # noqa: ANN001, ARG001
            captured_payload.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
            results = provider.health_check_nvidia_keys(
                keys=["key-one"],
                model="gcp/google/gemini-3.1-pro-preview",
                base_url="https://example.test/v1",
                timeout=1.0,
                max_workers=1,
            )

        self.assertEqual(results[0]["status"], "healthy")
        self.assertEqual(captured_payload["max_tokens"], provider.HEALTH_CHECK_MAX_TOKENS)
        self.assertGreaterEqual(captured_payload["max_tokens"], 16)
        self.assertEqual(
            captured_payload["messages"], [{"role": "user", "content": "Reply only: ok"}]
        )

    def test_health_check_output_limit_error_is_healthy(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="key-one")

        class FakeHTTPError(urllib.error.HTTPError):
            def read(self) -> bytes:
                return (
                    b'{"error":{"message":"Could not finish the message because '
                    b'max_tokens or model output limit was reached."}}'
                )

        def fake_urlopen(request, timeout: float):  # noqa: ANN001, ARG001
            raise FakeHTTPError(
                url="https://example.test/v1/chat/completions",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=None,
            )

        with patch.object(provider.urllib.request, "urlopen", fake_urlopen):
            results = provider.health_check_nvidia_keys(
                keys=["key-one"],
                model="synthetic/output-limit-model",
                base_url="https://example.test/v1",
                timeout=1.0,
                max_workers=1,
            )

        self.assertEqual(results[0]["status"], "healthy")
        self.assertEqual(results[0]["http_status"], 200)
        self.assertEqual(results[0]["error"], "")

    def test_provider_dashboard_ignores_benign_health_probe_output_limit(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="key-one")
        import enpire.env.forge.cap.agent.providers.nvidia_server as server

        label = provider.nvidia_key_label("key-one")
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.jsonl"
            os.environ[provider.TELEMETRY_FILE_ENV] = str(telemetry_path)
            telemetry_path.write_text(
                json.dumps(
                    {
                        "event": "end",
                        "key": label,
                        "source": "health_check",
                        "status": "error",
                        "http_status": 400,
                        "error": "Could not finish because max_tokens or model output limit was reached.",
                        "ts": time.time(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stats = server.build_stats()

        self.assertEqual(stats["summary"]["err"], 0)
        self.assertEqual(stats["summary"]["rpm"], 0)
        self.assertEqual(stats["summary"]["peak_rpm"], 0)
        self.assertEqual(stats["keys"][0]["err"], 0)
        self.assertEqual(stats["keys"][0]["last_error"], "")
        self.assertEqual(stats["recent_errors"], [])

    def test_provider_dashboard_reports_peak_rates_since_start(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="key-one")
        import enpire.env.forge.cap.agent.providers.nvidia_server as server

        label = provider.nvidia_key_label("key-one")
        events = [
            {
                "event": "end",
                "key": label,
                "source": "llm",
                "status": "ok",
                "http_status": 200,
                "total_tokens": 100,
                "ts": 1000.0,
            },
            {
                "event": "end",
                "key": label,
                "source": "llm",
                "status": "ok",
                "http_status": 200,
                "total_tokens": 200,
                "ts": 1010.0,
            },
            {
                "event": "end",
                "key": label,
                "source": "llm",
                "status": "ok",
                "http_status": 200,
                "total_tokens": 400,
                "ts": 1020.0,
            },
            {
                "event": "end",
                "key": label,
                "source": "llm",
                "status": "ok",
                "http_status": 200,
                "total_tokens": 50,
                "ts": 1100.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.jsonl"
            os.environ[provider.TELEMETRY_FILE_ENV] = str(telemetry_path)
            telemetry_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            stats = server.build_stats()

        self.assertEqual(stats["summary"]["peak_rpm"], 3)
        self.assertEqual(stats["summary"]["peak_tpm"], 700)

    def test_scheduler_records_delay_after_release(self) -> None:
        provider = _load_provider(NVIDIA_API_KEY_1="key-one")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nvidia.sqlite"
            provider.init_nvidia_scheduler(
                db_path,
                keys=provider.list_nvidia_keys(),
                min_interval_s=2.0,
                max_concurrent_per_key=1,
            )
            lease = provider.acquire_nvidia_key(request_id="test-request", timeout_s=0.1)
            self.assertEqual(lease.api_key, "key-one")
            before_release = time.time()
            provider.release_nvidia_key(
                lease,
                status="ok",
                http_status=200,
                latency_ms=12.0,
                error=None,
            )
            stats = provider.read_nvidia_scheduler_stats()

        self.assertEqual(stats[0]["key"], provider.nvidia_key_label("key-one"))
        self.assertEqual(stats[0]["active"], 0)
        self.assertGreaterEqual(stats[0]["next_allowed_at"], before_release + 1.9)


if __name__ == "__main__":
    unittest.main()
