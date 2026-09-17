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

    async def post(self, url, *, json=None, headers=None):  # noqa: A002
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
        self.assertEqual(body["max_completion_tokens"], llm_client._OPEN_MAX_TOKENS)


class OpenModelOverrideTests(unittest.TestCase):
    def test_open_path_uses_the_open_model_when_set(self) -> None:
        with _configured(llm_api_open_model="gpt-5.6-terra"), _script(_OK):
            _run(llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s"))
        self.assertEqual(_ScriptedClient.posted[0]["model"], "gpt-5.6-terra")

    def test_open_path_falls_back_to_the_default_model(self) -> None:
        with _configured(), _script(_OK):
            _run(llm_client.open_chat([{"role": "user", "content": "q"}], system_prompt="s"))
        self.assertEqual(_ScriptedClient.posted[0]["model"], "gemma")

    def test_phrase_and_classify_ignore_the_open_model(self) -> None:
        with _configured(llm_api_open_model="gpt-5.6-terra"), _script(_OK):
            _run(llm_client.phrase("fact", system_prompt="sys"))
            _run(llm_client.classify("q", system_prompt="sys"))
        self.assertEqual([b["model"] for b in _ScriptedClient.posted], ["gemma", "gemma"])


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
        self.assertEqual(Settings.model_validate({"llm_api_vendor": ""}).llm_api_vendor, "local")

    def test_unknown_vendor_fails_at_startup(self) -> None:
        with self.assertRaises(ValueError):
            Settings.model_validate({"llm_api_vendor": "anthropic"})

    def test_blank_open_model_means_unset(self) -> None:
        self.assertIsNone(Settings.model_validate({"llm_api_open_model": "  "}).llm_api_open_model)


if __name__ == "__main__":
    unittest.main()
