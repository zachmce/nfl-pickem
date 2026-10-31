"""Wire-format tests for the LLM vendor switch, the open-path model override and the
5xx retry in ``app.bot.llm_client._post_chat``.

The local dialect (``LLM_API_VENDOR=local``, the default) must stay byte-identical to
what ``tests/test_llm_commentary.py`` pins; the ``openai`` dialect is what api.openai.com
accepted on 2026-09-17 (every rule answers a measured 400). No test touches the network:
``httpx.AsyncClient`` is monkeypatched.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import httpx

from app.bot import llm_client
from app.config import Settings, settings
from app.services import bot_telemetry


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _ScriptedClient:
    """Stand-in for ``httpx.AsyncClient`` that replays a scripted list of responses."""

    responses: list[_FakeResponse] = []
    posted: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url, *, json=None, headers=None):
        type(self).posted.append(dict(json or {}))
        return type(self).responses[min(len(type(self).posted) - 1, len(type(self).responses) - 1)]


_OK = _FakeResponse(200, {"choices": [{"message": {"content": "voiced line"}}]})


def _configured(**overrides):
    values = {
        "llm_api_server": "http://llm:8000/v1",
        "llm_api_model": "gemma",
        "llm_api_key": "secret-key",
        "llm_api_vendor": "local",
        "llm_api_open_model": None,
    }
    values.update(overrides)
    return mock.patch.multiple(settings, **values)


def _run(coro):
    return asyncio.run(coro)


def _script(*responses: _FakeResponse):
    _ScriptedClient.responses = list(responses)
    _ScriptedClient.posted = []
    return mock.patch.object(httpx, "AsyncClient", _ScriptedClient)


class VendorTranslationTests(unittest.TestCase):
    def test_local_vendor_body_is_unchanged(self) -> None:
        with _configured(), _script(_OK):
            _run(llm_client.phrase("fact", system_prompt="sys"))
        body = _ScriptedClient.posted[0]
        self.assertEqual(body["model"], "gemma")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("max_tokens", body)
        self.assertIn("temperature", body)
        self.assertIn("top_p", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("reasoning_effort", body)

    def test_openai_vendor_translates_the_rejected_fields(self) -> None:
        with _configured(llm_api_vendor="openai", llm_api_model="gpt-5.6-luna"), _script(_OK):
            out = _run(llm_client.phrase("fact", system_prompt="sys"))
        self.assertEqual(out, "voiced line")
        body = _ScriptedClient.posted[0]
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertNotIn("chat_template_kwargs", body)
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], llm_client._MAX_TOKENS)
        self.assertEqual(body["reasoning_effort"], "none")
        # The message shape is not the translator's business.
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertTrue(body["messages"][0]["content"].startswith("sys"))

    def test_openai_vendor_keeps_tools_on_the_open_path(self) -> None:
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        with _configured(llm_api_vendor="openai"), _script(_OK):
            _run(
                llm_client.open_chat(
                    [{"role": "user", "content": "q"}], system_prompt="s", tools=tools
                )
            )
        body = _ScriptedClient.posted[0]
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["reasoning_effort"], "none")
        # Issue #341: a whole-slate fan-out needs more room than the 600 prose cap.
        self.assertEqual(body["max_completion_tokens"], llm_client._OPEN_TOOLS_MAX_TOKENS)

    def test_openai_vendor_keeps_the_prose_cap_on_the_tools_free_close(self) -> None:
        with _configured(llm_api_vendor="openai"), _script(_OK):
            _run(llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s"))
        self.assertEqual(
            _ScriptedClient.posted[0]["max_completion_tokens"], llm_client._OPEN_MAX_TOKENS
        )


class _FakeMessage:
    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump(self, mode: str = "python") -> dict:
        return self._data


class _FakeAnthropic:
    """Stand-in for the SDK client: records ``messages.create`` kwargs, replays replies."""

    def __init__(self, *replies: object) -> None:
        self.replies = list(replies) or [
            {
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": "voiced line"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            }
        ]
        self.created: list[dict] = []
        self.messages = self
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float):
        self.timeouts.append(timeout)
        return self

    async def create(self, **kwargs) -> _FakeMessage:
        self.created.append(kwargs)
        reply = self.replies[min(len(self.created) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, dict)
        return _FakeMessage(reply)


def _anthropic(fake: _FakeAnthropic):
    return mock.patch.object(llm_client, "_anthropic_client", lambda: fake)


_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look it up.",
        "parameters": {"type": "object", "properties": {"team": {"type": "string"}}},
    },
}


class AnthropicNativeTests(unittest.TestCase):
    """The anthropic vendor speaks the native Messages API: the compatible endpoint
    does not cache prompts (Anthropic's docs, read 2026-10-31)."""

    def test_phrase_sends_system_with_a_breakpoint_and_no_sampling_knobs(self) -> None:
        fake = _FakeAnthropic()
        with _configured(llm_api_vendor="anthropic", llm_api_model="claude-haiku-5-5"):
            with _anthropic(fake):
                out = _run(llm_client.phrase("fact", system_prompt="sys"))
        self.assertEqual(out, "voiced line")
        sent = fake.created[0]
        self.assertEqual(sent["model"], "claude-haiku-5-5")
        self.assertEqual(sent["max_tokens"], llm_client._MAX_TOKENS)
        self.assertEqual(sent["messages"], [{"role": "user", "content": "fact"}])
        self.assertTrue(sent["system"][0]["text"].startswith("sys"))
        self.assertEqual(sent["system"][0]["cache_control"], {"type": "ephemeral"})
        # A one-shot call ends in unique text: an automatic breakpoint there is a pure write.
        for absent in ("cache_control", "tools", "temperature", "top_p", "chat_template_kwargs"):
            self.assertNotIn(absent, sent)

    def test_thinking_follows_each_call_model(self) -> None:
        # Measured 2026-10-22: Haiku 5.5 400s on between_tools, Sonnet 5.5 on disabled.
        fake = _FakeAnthropic()
        with (
            _configured(
                llm_api_vendor="anthropic",
                llm_api_model="claude-haiku-5-5",
                llm_api_open_model="claude-sonnet-5-5",
            ),
            _anthropic(fake),
        ):
            _run(llm_client.phrase("fact", system_prompt="sys"))
            _run(llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s"))
        self.assertEqual(
            [c["thinking"] for c in fake.created],
            [{"type": "disabled"}, {"type": "between_tools"}],
        )

    def test_a_tools_round_caches_the_tools_the_system_and_the_tail(self) -> None:
        fake = _FakeAnthropic()
        with (
            _configured(llm_api_vendor="anthropic", llm_api_open_model="claude-sonnet-5-5"),
            _anthropic(fake),
        ):
            _run(
                llm_client.open_chat(
                    [{"role": "user", "content": "q"}], system_prompt="s", tools=[_TOOL_SPEC]
                )
            )
        sent = fake.created[0]
        self.assertEqual(
            sent["tools"],
            [
                {
                    "name": "lookup",
                    "description": "Look it up.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"team": {"type": "string"}},
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
        self.assertEqual(sent["tool_choice"], {"type": "auto"})
        self.assertEqual(sent["cache_control"], {"type": "ephemeral"})
        self.assertEqual(sent["max_tokens"], llm_client._OPEN_TOOLS_MAX_TOKENS)
        self.assertEqual(fake.timeouts, [llm_client._OPEN_TIMEOUT_SECONDS])

    def test_tool_turns_replay_as_tool_use_and_one_tool_result_turn(self) -> None:
        fake = _FakeAnthropic()
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "toolu_1", "function": {"name": "a", "arguments": '{"team": "DEN"}'}},
                    {"id": "toolu_2", "function": {"name": "b", "arguments": "not json"}},
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "name": "a", "content": '{"x": 1}'},
            {"role": "tool", "tool_call_id": "toolu_2", "name": "b", "content": '"none"'},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "follow-up"},
        ]
        with _configured(llm_api_vendor="anthropic"), _anthropic(fake):
            _run(llm_client.open_chat(messages, system_prompt="s"))
        self.assertEqual(
            fake.created[0]["messages"],
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "a",
                            "input": {"team": "DEN"},
                        },
                        {"type": "tool_use", "id": "toolu_2", "name": "b", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"x": 1}'},
                        {"type": "tool_result", "tool_use_id": "toolu_2", "content": '"none"'},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "earlier answer"}]},
                {"role": "user", "content": "follow-up"},
            ],
        )

    def test_the_reply_reads_back_as_a_chat_completions_message(self) -> None:
        fake = _FakeAnthropic(
            {
                "model": "claude-sonnet-5-5",
                "content": [
                    {"type": "thinking", "thinking": "I'll check.", "signature": "x"},
                    {"type": "tool_use", "id": "toolu_9", "name": "lookup", "input": {"t": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 7000,
                    "cache_creation_input_tokens": 300,
                },
            }
        )

        async def _go() -> tuple[dict | None, bot_telemetry.AnswerTrace]:
            trace = bot_telemetry.start("q", conversation_key=None, asker_name=None)
            message = await llm_client.open_chat(
                [{"role": "user", "content": "q"}], system_prompt="s", tools=[_TOOL_SPEC]
            )
            return message, trace

        with _configured(llm_api_vendor="anthropic"), _anthropic(fake):
            message, trace = _run(_go())
        bot_telemetry._CURRENT.set(None)
        self.assertEqual(
            message,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_9",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"t": 1}'},
                    }
                ],
                llm_client.COMPLETION_TOKENS_KEY: 20,
            },
        )
        self.assertEqual(
            trace.llm_calls,
            [
                {
                    "kind": "open",
                    "model": "claude-sonnet-5-5",
                    "prompt": 7350,
                    "completion": 20,
                    "cached": 7000,
                    "cache_write": 300,
                }
            ],
        )

    def test_max_tokens_reads_as_a_length_finish(self) -> None:
        fake = _FakeAnthropic(
            {"content": [{"type": "text", "text": "cut"}], "stop_reason": "max_tokens"}
        )
        with _configured(llm_api_vendor="anthropic"), _anthropic(fake):
            message = _run(
                llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s")
            )
        assert message is not None
        self.assertTrue(message[llm_client.TRUNCATED_KEY])

    def test_an_api_error_returns_none(self) -> None:
        fake = _FakeAnthropic(RuntimeError("boom"))
        with _configured(llm_api_vendor="anthropic"), _anthropic(fake):
            self.assertIsNone(_run(llm_client.phrase("fact", system_prompt="sys")))

    def test_the_client_strips_the_compatible_endpoint_path(self) -> None:
        async def _go():
            return llm_client._anthropic_client()

        with _configured(
            llm_api_vendor="anthropic", llm_api_server="https://api.anthropic.com/v1/"
        ):
            client = _run(_go())
        self.assertEqual(str(client.base_url).rstrip("/"), "https://api.anthropic.com")


class OpenModelOverrideTests(unittest.TestCase):
    def test_open_path_uses_the_open_model_when_set(self) -> None:
        with _configured(llm_api_open_model="gpt-5.6-terra"), _script(_OK):
            _run(llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s"))
        self.assertEqual(_ScriptedClient.posted[0]["model"], "gpt-5.6-terra")

    def test_open_path_falls_back_to_the_default_model(self) -> None:
        with _configured(), _script(_OK):
            _run(llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s"))
        self.assertEqual(_ScriptedClient.posted[0]["model"], "gemma")

    def test_classify_takes_a_per_call_model(self) -> None:
        with _configured(), _script(_OK):
            _run(llm_client.classify("q", system_prompt="sys", model="big"))
            _run(llm_client.classify("q", system_prompt="sys", model=None))
        self.assertEqual([b["model"] for b in _ScriptedClient.posted], ["big", "gemma"])

    def test_phrase_and_classify_ignore_the_open_model(self) -> None:
        with _configured(llm_api_open_model="gpt-5.6-terra"), _script(_OK):
            _run(llm_client.phrase("fact", system_prompt="sys"))
            _run(llm_client.classify("q", system_prompt="sys"))
        self.assertEqual([b["model"] for b in _ScriptedClient.posted], ["gemma", "gemma"])


class ResponseModelTests(unittest.TestCase):
    """Issue #354: the trace keeps the model and the usage the API RETURNED."""

    def test_each_call_records_the_answering_model_and_usage(self) -> None:
        reply = _FakeResponse(
            200,
            {
                "model": "claude-sonnet-5-5",
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 40,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
                "choices": [{"message": {"content": "ok"}}],
            },
        )

        async def _go() -> bot_telemetry.AnswerTrace:
            trace = bot_telemetry.start("q", conversation_key=None, asker_name=None)
            await llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s")
            await llm_client.classify("q", system_prompt="sys")
            return trace

        with _configured(llm_api_open_model="claude-sonnet-5-5"), _script(reply, _OK):
            trace = _run(_go())
        bot_telemetry._CURRENT.set(None)
        self.assertEqual(
            trace.llm_calls,
            [
                {
                    "kind": "open",
                    "model": "claude-sonnet-5-5",
                    "prompt": 900,
                    "completion": 40,
                    "cached": 800,
                    "cache_write": 0,
                },
                {
                    "kind": "classify",
                    "model": None,
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "cache_write": 0,
                },
            ],
        )

    def test_a_failed_call_records_nothing(self) -> None:
        async def _go() -> bot_telemetry.AnswerTrace:
            trace = bot_telemetry.start("q", conversation_key=None, asker_name=None)
            await llm_client.phrase("fact", system_prompt="sys")
            return trace

        with _configured(), _script(_FakeResponse(400, {})):
            trace = _run(_go())
        bot_telemetry._CURRENT.set(None)
        self.assertEqual(trace.llm_calls, [])


class RetryTests(unittest.TestCase):
    def test_a_5xx_is_retried_once_and_the_retry_answer_is_used(self) -> None:
        with _configured(), _script(_FakeResponse(500, {}), _OK):
            out = _run(llm_client.phrase("fact", system_prompt="sys"))
        self.assertEqual(out, "voiced line")
        self.assertEqual(len(_ScriptedClient.posted), 2)

    def test_two_5xx_in_a_row_return_none(self) -> None:
        with _configured(), _script(_FakeResponse(503, {}), _FakeResponse(502, {})):
            out = _run(llm_client.phrase("fact", system_prompt="sys"))
        self.assertIsNone(out)
        self.assertEqual(len(_ScriptedClient.posted), 2)

    def test_a_4xx_is_not_retried(self) -> None:
        with _configured(), _script(_FakeResponse(400, {}), _OK):
            out = _run(llm_client.phrase("fact", system_prompt="sys"))
        self.assertIsNone(out)
        self.assertEqual(len(_ScriptedClient.posted), 1)


class VendorSettingTests(unittest.TestCase):
    """``model_validate`` runs the field validators without reading .env or the process env."""

    def test_vendor_defaults_to_local_and_normalizes_case(self) -> None:
        self.assertEqual(Settings.model_validate({}).llm_api_vendor, "local")
        self.assertEqual(
            Settings.model_validate({"llm_api_vendor": "OpenAI"}).llm_api_vendor, "openai"
        )
        self.assertEqual(
            Settings.model_validate({"llm_api_vendor": "Anthropic"}).llm_api_vendor, "anthropic"
        )
        self.assertEqual(Settings.model_validate({"llm_api_vendor": ""}).llm_api_vendor, "local")

    def test_unknown_vendor_fails_at_startup(self) -> None:
        with self.assertRaises(ValueError):
            Settings.model_validate({"llm_api_vendor": "gemini"})

    def test_blank_open_model_means_unset(self) -> None:
        self.assertIsNone(Settings.model_validate({"llm_api_open_model": "  "}).llm_api_open_model)


class SharedClientTests(unittest.TestCase):
    def test_one_client_serves_every_call_in_a_loop(self) -> None:
        built: list[object] = []

        class _CountingClient(_ScriptedClient):
            def __init__(self, *args, **kwargs) -> None:
                built.append(kwargs)

        async def _two_calls() -> None:
            await llm_client.phrase("fact", system_prompt="sys")
            await llm_client.phrase("fact", system_prompt="sys")

        _ScriptedClient.responses = [_OK]
        _ScriptedClient.posted = []
        with _configured(), mock.patch.object(httpx, "AsyncClient", _CountingClient):
            _run(_two_calls())
            _run(_two_calls())
        self.assertEqual(len(_ScriptedClient.posted), 4)
        self.assertEqual(len(built), 2)  # one per event loop, not one per call


if __name__ == "__main__":
    unittest.main()
