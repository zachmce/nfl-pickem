"""Offline unit tests for the ESPN "extras" injuries adapter (260709-u0z Task 1).

Two layers are exercised, both fully OFFLINE (no network socket, no Redis socket):

* the PURE :func:`app.services.espn_extra.parse_injuries` against a captured
  ``summary`` fixture (multi-player team, empty-injuries team, team-filtering) plus
  inline malformed inputs — it must never raise and must return the right
  list / ``[]`` / ``None`` distinction;
* the IMPURE :func:`app.services.espn_extra.fetch_injuries` with ``httpx`` and the
  ``_redis_client`` seam monkeypatched (mirroring the capturing-client style of
  ``tests/test_qa_classifier.py``) to prove cache HIT (no HTTP), cache MISS
  (HTTP + cache write), and best-effort ``None`` on a fetch/Redis failure.

One OPTIONAL live ESPN smoke test is SKIPPED unless ``RUN_ESPN_LIVE`` is set (mirrors
``tests/test_scoreboard_espn.py``).

Run with: ``backend/.venv/bin/python -m unittest tests.test_espn_extra -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import httpx

from app.services import espn_extra

_FIXTURE = Path(__file__).parent / "fixtures" / "espn_summary_injuries.json"


def _run(coro):
    return asyncio.run(coro)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


# --------------------------------------------------------------------------- #
# Fake outbound seams (never open a real socket).
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _CapturingAsyncClient:
    """A stand-in for ``httpx.AsyncClient`` that records the GET and returns a canned response."""

    last_url: str | None = None
    calls: int = 0
    _response: object = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get(self, url, *, headers=None):
        type(self).calls += 1
        type(self).last_url = url
        return self._response


class _RaisingAsyncClient:
    """An ``httpx.AsyncClient`` stand-in that FAILS if any HTTP is attempted."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get(self, url, *, headers=None):
        raise AssertionError("HTTP must not be called on a cache hit")


class _FakeRedis:
    """A minimal async Redis stand-in recording get/set, with a seeded store."""

    def __init__(self, store: dict | None = None) -> None:
        self.store = store or {}
        self.sets: list[tuple[str, str, int | None]] = []

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, *, ex=None):
        self.sets.append((key, value, ex))
        self.store[key] = value

    async def aclose(self):
        return None


def _redis_returns(client: _FakeRedis):
    return mock.patch.object(espn_extra, "_redis_client", lambda: client)


def _redis_raises():
    def _boom():
        raise RuntimeError("redis down")

    return mock.patch.object(espn_extra, "_redis_client", _boom)


# --------------------------------------------------------------------------- #
# Pure parser.
# --------------------------------------------------------------------------- #


class ParseInjuriesTests(unittest.TestCase):
    def test_multi_player_team_returns_full_list(self) -> None:
        players = espn_extra.parse_injuries(_load_fixture(), "KC")
        assert players is not None
        self.assertEqual(len(players), 3)
        first = players[0]
        self.assertEqual(first["display_name"], "Isiah Pacheco")
        self.assertEqual(first["status"], "Out")
        self.assertEqual(first["position"], "RB")
        self.assertEqual(first["body_part"], "Knee")
        self.assertEqual(first["return_date"], "2026-01-19")
        self.assertEqual(first["date"], "2026-01-05T18:00Z")
        # A player without a returnDate carries None there (not invented).
        questionable = players[1]
        self.assertEqual(questionable["status"], "Questionable")
        self.assertIsNone(questionable["return_date"])

    def test_case_insensitive_team_match(self) -> None:
        self.assertIsNotNone(espn_extra.parse_injuries(_load_fixture(), "kc"))

    def test_team_present_with_no_injuries_returns_empty_list(self) -> None:
        # LAC's block is present with an empty injuries[] -> [] (distinct from None).
        players = espn_extra.parse_injuries(_load_fixture(), "LAC")
        self.assertEqual(players, [])

    def test_team_filtering_excludes_the_other_team(self) -> None:
        # Asking for KC never returns LAC's block (and vice-versa). KC has 3 players;
        # none of them belong to LAC's (empty) block.
        kc = espn_extra.parse_injuries(_load_fixture(), "KC")
        assert kc is not None
        names = {p["display_name"] for p in kc}
        self.assertEqual(names, {"Isiah Pacheco", "Rashee Rice", "Nick Bolton"})

    def test_absent_team_block_returns_none(self) -> None:
        # A team that is NOT one of the two blocks -> None (degrade, never "no injuries").
        self.assertIsNone(espn_extra.parse_injuries(_load_fixture(), "SF"))

    def test_non_dict_payload_returns_none(self) -> None:
        for bad in (None, "garbage", 42, ["injuries"]):
            self.assertIsNone(espn_extra.parse_injuries(bad, "KC"))

    def test_injuries_not_a_list_returns_none(self) -> None:
        self.assertIsNone(espn_extra.parse_injuries({"injuries": "nope"}, "KC"))
        self.assertIsNone(espn_extra.parse_injuries({}, "KC"))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "injuries": [
                "garbage",
                {"team": "not-a-dict", "injuries": []},
                {
                    "team": {"abbreviation": "KC"},
                    "injuries": [
                        "nope",
                        {},  # empty entry -> all-None fact dict, still counted
                        {
                            "status": "Out",
                            "athlete": {"displayName": "Real Player"},
                        },
                    ],
                },
            ]
        }
        players = espn_extra.parse_injuries(payload, "KC")
        assert players is not None
        # The string "nope" is skipped; the {} and the real entry are parsed.
        self.assertEqual(len(players), 2)
        self.assertEqual(players[1]["display_name"], "Real Player")
        self.assertIsNone(players[0]["display_name"])

    def test_status_falls_back_to_type_name(self) -> None:
        payload = {
            "injuries": [
                {
                    "team": {"abbreviation": "KC"},
                    "injuries": [
                        {"type": {"name": "Questionable"}, "athlete": {"displayName": "X"}}
                    ],
                }
            ]
        }
        players = espn_extra.parse_injuries(payload, "KC")
        assert players is not None
        self.assertEqual(players[0]["status"], "Questionable")


# --------------------------------------------------------------------------- #
# Impure fetch (cache + HTTP), fully monkeypatched.
# --------------------------------------------------------------------------- #


class FetchInjuriesTests(unittest.TestCase):
    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_fixture()
        fake = _FakeRedis({espn_extra._cache_key(555): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(espn_extra.fetch_injuries(555))
        self.assertEqual(out, payload)  # served from cache
        self.assertEqual(fake.sets, [])  # nothing re-written on a hit

    def test_cache_miss_fetches_and_writes_cache(self) -> None:
        payload = _load_fixture()
        fake = _FakeRedis()  # empty -> miss
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient._response = _FakeResponse(200, payload)
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_injuries(777))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)  # exactly one GET on a miss
        assert _CapturingAsyncClient.last_url is not None
        self.assertIn("event=777", _CapturingAsyncClient.last_url)
        # The raw payload was written to the cache under the event key + TTL.
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._cache_key(777))
        self.assertEqual(json.loads(value), payload)
        self.assertEqual(ex, espn_extra.INJURIES_CACHE_TTL_SECONDS)

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            out = _run(espn_extra.fetch_injuries(1))
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])  # nothing cached on a failed fetch

    def test_non_200_degrades_to_none(self) -> None:
        fake = _FakeRedis()
        _CapturingAsyncClient._response = _FakeResponse(503, {})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_injuries(2))
        self.assertIsNone(out)

    def test_redis_error_still_allows_live_fetch(self) -> None:
        # A Redis outage on the read must FAIL OPEN -> the live fetch still happens.
        payload = _load_fixture()
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient._response = _FakeResponse(200, payload)
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_injuries(9))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)


@unittest.skipUnless(
    os.environ.get("RUN_ESPN_LIVE"),
    "live ESPN smoke test (set RUN_ESPN_LIVE=1 to enable; performs a real network GET)",
)
class EspnExtraLiveSmokeTest(unittest.TestCase):
    """OPTIONAL, network-gated smoke test — skipped by default (offline suite)."""

    def test_fetch_injuries_returns_a_dict(self) -> None:
        # A recent regular-season event id; the endpoint returns a large summary dict.
        out = _run(espn_extra.fetch_injuries(401547001))
        self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
