"""Offline unit tests for the best-effort LLM client (260627-nef).

These tests NEVER touch a live LLM endpoint: ``httpx.AsyncClient`` is monkeypatched
with a fake whose ``post`` returns a canned response (or raises). They assert the
HARD wire-format rule (``chat_template_kwargs.enable_thinking == False`` — without
it the served gemma reasoning model returns empty content) + bearer auth, and the
best-effort contract: a 200 returns the stripped content; a timeout / non-200 /
empty content returns ``None`` and NEVER raises.

Run with: ``backend/.venv/bin/python -m unittest tests.test_llm_commentary -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import httpx

from app.bot import llm_client
from app.config import settings


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Stand-in for ``httpx.AsyncClient`` recording the last POST it received."""

    last_url: str | None = None
    last_json: dict | None = None
    last_headers: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url, *, json=None, headers=None):  # noqa: A002
        type(self).last_url = url
        type(self).last_json = json
        type(self).last_headers = headers
        return self._response

    # set per-test
    _response: object = None


def _configured():
    """Patch the three LLM_* settings so the client is 'configured'."""
    return mock.patch.multiple(
        settings,
        llm_api_server="http://llm:8000/v1",
        llm_api_model="gemma",
        llm_api_key="secret-key",
    )


def _run(coro):
    return asyncio.run(coro)


class PhrasePatternTests(unittest.TestCase):
    def test_success_returns_stripped_content_and_sends_wire_format(self) -> None:
        _FakeAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": "  KC again? bold. 🔒  "}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            out = _run(llm_client.phrase_pattern("Alice has taken KC OVER 3 weeks running"))
        self.assertEqual(out, "KC again? bold. 🔒")
        # HARD RULE: enable_thinking must be False or content comes back empty.
        body = _FakeAsyncClient.last_json
        assert body is not None
        self.assertIn("chat_template_kwargs", body)
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)
        # bearer auth + chat/completions endpoint + model wired through.
        self.assertEqual(_FakeAsyncClient.last_headers["Authorization"], "Bearer secret-key")
        self.assertTrue(_FakeAsyncClient.last_url.endswith("/chat/completions"))
        self.assertEqual(body["model"], "gemma")
        # Widened phrasing sampling (read the constants so a future retune stays honest).
        self.assertEqual(body["top_p"], llm_client._TOP_P)
        self.assertEqual(body["temperature"], llm_client._TEMPERATURE)

    def test_non_200_returns_none(self) -> None:
        _FakeAsyncClient._response = _FakeResponse(500, {})
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            out = _run(llm_client.phrase_pattern("fact"))
        self.assertIsNone(out)

    def test_empty_content_returns_none(self) -> None:
        _FakeAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": "   "}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            out = _run(llm_client.phrase_pattern("fact"))
        self.assertIsNone(out)

    def test_timeout_returns_none_never_raises(self) -> None:
        class _RaisingClient(_FakeAsyncClient):
            async def post(self, url, *, json=None, headers=None):  # noqa: A002
                raise httpx.TimeoutException("slow")

        with _configured(), mock.patch.object(httpx, "AsyncClient", _RaisingClient):
            out = _run(llm_client.phrase_pattern("fact"))
        self.assertIsNone(out)

    def test_unconfigured_returns_none_without_calling_http(self) -> None:
        """Missing any of the three LLM_* settings -> disabled, returns None."""
        with (
            mock.patch.multiple(
                settings,
                llm_api_server=None,
                llm_api_model=None,
                llm_api_key=None,
            ),
            mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient),
        ):
            out = _run(llm_client.phrase_pattern("fact"))
        self.assertIsNone(out)


class PhraseTests(unittest.TestCase):
    """The generalized ``phrase(fact, *, system_prompt)`` core: same wire-format
    and best-effort contract as ``phrase_pattern``, but the system prompt is a
    parameter so any event can phrase its own bot-supplied fact."""

    def test_success_sends_supplied_prompt_and_fact_and_wire_format(self) -> None:
        _FakeAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": "  let's go! 🏈  "}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            out = _run(
                llm_client.phrase("Week 3 picks are open", system_prompt="You are a hype bot")
            )
        self.assertEqual(out, "let's go! 🏈")
        body = _FakeAsyncClient.last_json
        assert body is not None
        # The SUPPLIED prompt LEADS the system message (still verbatim, still first),
        # with the closer-variety directive appended AFTER it; the fact is the user msg.
        messages = {m["role"]: m["content"] for m in body["messages"]}
        self.assertIn("You are a hype bot", messages["system"])
        self.assertTrue(messages["system"].startswith("You are a hype bot"))
        self.assertIn("Vary how you sign off", messages["system"])
        self.assertEqual(messages["user"], "Week 3 picks are open")
        # HARD RULE: enable_thinking must be False or content comes back empty.
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)
        # bearer auth + chat/completions endpoint + model wired through.
        self.assertEqual(_FakeAsyncClient.last_headers["Authorization"], "Bearer secret-key")
        self.assertTrue(_FakeAsyncClient.last_url.endswith("/chat/completions"))
        self.assertEqual(body["model"], "gemma")
        # Widened phrasing sampling (read the constants so a future retune stays honest).
        self.assertEqual(body["top_p"], llm_client._TOP_P)
        self.assertEqual(body["temperature"], llm_client._TEMPERATURE)

    def test_phrase_appends_closer_variety_directive_and_preserves_prompt(self) -> None:
        """The caller's guard-bearing prompt survives verbatim (and leads) while the
        closer-variety directive is appended — proving every event path (which all
        flow through phrase()) gets the anti-repetition style nudge."""
        sentinel = "SENTINEL_GUARD_BEARING_PROMPT_zx9"
        _FakeAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": "ok"}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            _run(llm_client.phrase("some fact", system_prompt=sentinel))
        body = _FakeAsyncClient.last_json
        assert body is not None
        system_content = {m["role"]: m["content"] for m in body["messages"]}["system"]
        self.assertIn(sentinel, system_content)
        self.assertIn("Vary how you sign off", system_content)
        # The prompt still LEADS (facts-first ordering preserved).
        self.assertTrue(system_content.startswith(sentinel))

    def test_non_200_returns_none(self) -> None:
        _FakeAsyncClient._response = _FakeResponse(503, {})
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            out = _run(llm_client.phrase("fact", system_prompt="prompt"))
        self.assertIsNone(out)

    def test_empty_content_returns_none(self) -> None:
        _FakeAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": "  \n "}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
            out = _run(llm_client.phrase("fact", system_prompt="prompt"))
        self.assertIsNone(out)

    def test_timeout_returns_none_never_raises(self) -> None:
        class _RaisingClient(_FakeAsyncClient):
            async def post(self, url, *, json=None, headers=None):  # noqa: A002
                raise httpx.TimeoutException("slow")

        with _configured(), mock.patch.object(httpx, "AsyncClient", _RaisingClient):
            out = _run(llm_client.phrase("fact", system_prompt="prompt"))
        self.assertIsNone(out)

    def test_unconfigured_returns_none_without_calling_http(self) -> None:
        with (
            mock.patch.multiple(
                settings,
                llm_api_server=None,
                llm_api_model=None,
                llm_api_key=None,
            ),
            mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient),
        ):
            out = _run(llm_client.phrase("fact", system_prompt="prompt"))
        self.assertIsNone(out)


class ConfigLlmSettingsTests(unittest.TestCase):
    def test_blank_env_coerces_to_none(self) -> None:
        from app.config import Settings

        s = Settings(llm_api_server="", llm_api_model="  ", llm_api_key="")  # type: ignore[call-arg]
        self.assertIsNone(s.llm_api_server)
        self.assertIsNone(s.llm_api_model)
        self.assertIsNone(s.llm_api_key)


if __name__ == "__main__":
    unittest.main()
