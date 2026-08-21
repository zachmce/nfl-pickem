"""Offline unit tests for the live-odds seam's PURE core (260710-mpw Task 1).

These tests NEVER touch the network. They exercise
:func:`app.services.live_odds.select_live_odds_for_event` — the pure indexer that
reuses ``app.scoreboard.espn.select_odds_item`` + ``normalize_odds`` to turn a
site-scoreboard payload into per-event normalized odds — proving it resolves a real
DraftKings inline shape, indexes by event id, and degrades to ``None`` on a missing
event / non-dict / odds-less payload (never raises).

They also exercise the IMPURE :func:`app.services.live_odds.fetch_live_odds` with ``httpx``
and the ``_redis_client`` seam monkeypatched (mirroring ``tests/test_espn_extra.py``): a
cache HIT with zero HTTP, a cache MISS with the exact week key and TTL, NO custom
User-Agent on the ESPN host, both degrade paths, and Redis fail-open on the read.

Run with: ``backend/.venv/bin/python -m unittest tests.test_live_odds -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

import httpx

from app.scoreboard.espn import SITE_SCOREBOARD_URL
from app.services import live_odds
from app.services.live_odds import select_live_odds_for_event


def _payload_one_event() -> dict:
    """A synthetic site scoreboard with one event carrying a DraftKings inline line.

    The shape mirrors the LIVE site path: odds live on ``competitions[0].odds[]`` with a
    ``provider.name``, a signed home-relative ``spread``, ``overUnder``, and inline
    ``home/awayTeamOdds`` favorite/underdog flags (see espn.py ``normalize_odds``).
    """
    return {
        "events": [
            {
                "id": "401547001",
                "competitions": [
                    {
                        "odds": [
                            {
                                "provider": {"name": "DraftKings", "priority": 1},
                                "spread": -3.5,
                                "overUnder": 47.5,
                                "awayTeamOdds": {"team": {"id": "1"}, "underdog": True},
                                "homeTeamOdds": {"team": {"id": "2"}, "favorite": True},
                            }
                        ]
                    }
                ],
            }
        ]
    }


class SelectLiveOddsTests(unittest.TestCase):
    def test_resolves_matching_event_via_normalize_odds(self) -> None:
        odds = select_live_odds_for_event(_payload_one_event(), 401547001)
        self.assertIsNotNone(odds)
        assert odds is not None  # narrow for the type checker
        self.assertEqual(odds.provider, "DraftKings")
        self.assertEqual(odds.spread, -3.5)
        self.assertEqual(odds.total, 47.5)
        self.assertEqual(odds.favorite_team_id, "2")
        self.assertEqual(odds.underdog_team_id, "1")

    def test_missing_event_returns_none(self) -> None:
        self.assertIsNone(select_live_odds_for_event(_payload_one_event(), 999999))

    def test_non_dict_payload_degrades_to_none(self) -> None:
        for bad in (None, "nope", 42, []):
            self.assertIsNone(select_live_odds_for_event(bad, 401547001))

    def test_oddsless_event_degrades_to_none(self) -> None:
        payload = {"events": [{"id": "5", "competitions": [{}]}]}
        self.assertIsNone(select_live_odds_for_event(payload, 5))

    def test_event_id_is_matched_as_string(self) -> None:
        # Our DB event id is an int; ESPN reports ids as strings — the lookup coerces.
        odds = select_live_odds_for_event(_payload_one_event(), 401547001)
        self.assertIsNotNone(odds)


# --------------------------------------------------------------------------- #
# Fake outbound seams (never open a real socket) — mirror tests/test_espn_extra.py.
# --------------------------------------------------------------------------- #


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


_NEVER_CALLED = "<never called>"


class _CapturingAsyncClient:
    """A stand-in for ``httpx.AsyncClient`` that records the GET and returns a canned response."""

    last_url: str | None = None
    # Sentinel-initialised so "headers was None" is distinguishable from "get never ran".
    last_headers: object = _NEVER_CALLED
    last_init_kwargs: dict | None = None
    calls: int = 0
    _response: object = None

    def __init__(self, *args, **kwargs) -> None:
        type(self).last_init_kwargs = dict(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get(self, url, *, headers=None):
        type(self).calls += 1
        type(self).last_url = url
        type(self).last_headers = headers
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
    return mock.patch.object(live_odds, "_redis_client", lambda: client)


def _redis_raises():
    def _boom():
        raise RuntimeError("redis down")

    return mock.patch.object(live_odds, "_redis_client", _boom)


# --------------------------------------------------------------------------- #
# Impure fetch (cache + HTTP), fully monkeypatched.
# --------------------------------------------------------------------------- #


class FetchLiveOddsTests(unittest.TestCase):
    _SEASON = 2025
    _WEEK = 7
    _EVENT_ID = 401547001

    def _fetch(self):
        return live_odds.fetch_live_odds(self._SEASON, self._WEEK, self._EVENT_ID)

    def _key(self) -> str:
        return live_odds._cache_key(self._SEASON, self._WEEK)

    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_cache_hit_returns_parsed_odds_without_http(self) -> None:
        fake = _FakeRedis({self._key(): json.dumps(_payload_one_event())})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            odds = _run(self._fetch())
        self.assertIsNotNone(odds)
        assert odds is not None  # narrow for the type checker
        self.assertEqual(odds.provider, "DraftKings")  # parsed, not the raw page
        self.assertEqual(odds.spread, -3.5)
        self.assertEqual(fake.sets, [])  # nothing re-written on a hit

    def test_cache_miss_fetches_once_and_writes_the_week_key_and_ttl(self) -> None:
        payload = _payload_one_event()
        fake = _FakeRedis()  # empty -> miss
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            odds = _run(self._fetch())
        self.assertIsNotNone(odds)
        self.assertEqual(_CapturingAsyncClient.calls, 1)  # exactly one GET on a miss
        # The URL carries ONLY the int season/week — never user text (T-mpw-01).
        self.assertEqual(
            _CapturingAsyncClient.last_url,
            SITE_SCOREBOARD_URL.format(season=self._SEASON, week=self._WEEK),
        )
        # One key per season+week, so one cached page serves every game that week.
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, self._key())
        self.assertEqual(json.loads(value), payload)  # the RAW page is cached
        self.assertEqual(ex, live_odds.LIVE_ODDS_CACHE_TTL_SECONDS)

    def test_no_custom_user_agent_and_explicit_timeout(self) -> None:
        # Invariant 1: this is the SAME ESPN edge that 403s a branded User-Agent (PR #171).
        # Invariant 2: a hung ESPN response must not be able to block the bot loop.
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, _payload_one_event()))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(self._fetch())
        self.assertEqual(_CapturingAsyncClient.calls, 1)  # the sentinel was really replaced
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": live_odds.DEFAULT_TIMEOUT}
        )

    def test_non_200_degrades_to_none_and_caches_nothing(self) -> None:
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(503, {}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            odds = _run(self._fetch())
        self.assertIsNone(odds)
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none_and_caches_nothing(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            odds = _run(self._fetch())
        self.assertIsNone(odds)
        self.assertEqual(fake.sets, [])

    def test_successful_fetch_without_the_asked_event_returns_none(self) -> None:
        # A live page that simply does not carry our event degrades — never invents a line.
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, {"events": []}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            odds = _run(self._fetch())
        self.assertIsNone(odds)
        self.assertEqual(len(fake.sets), 1)  # the page itself was still cached

    def test_redis_outage_still_serves_a_live_fetch(self) -> None:
        # FAIL-OPEN on the read: a cache outage degrades to the live fetch, never blocks it.
        self._arm_client(_FakeResponse(200, _payload_one_event()))
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            odds = _run(self._fetch())
        self.assertIsNotNone(odds)
        self.assertEqual(_CapturingAsyncClient.calls, 1)


if __name__ == "__main__":
    unittest.main()
