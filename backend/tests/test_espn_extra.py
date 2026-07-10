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
_NEWS_FIXTURE = Path(__file__).parent / "fixtures" / "espn_news.json"

# The DISTINCTIVE KC headline the no-rephrasing regression asserts survives byte-for-byte.
_KC_HEADLINE = "Patrick Mahomes throws for 5 touchdowns as Chiefs storm past Bills 38-20"


def _run(coro):
    return asyncio.run(coro)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _load_news_fixture() -> dict:
    return json.loads(_NEWS_FIXTURE.read_text())


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


# --------------------------------------------------------------------------- #
# Pure news parser (verbatim relay + client-side team filter).
# --------------------------------------------------------------------------- #


class ParseNewsTests(unittest.TestCase):
    def test_league_parse_relays_headlines_verbatim(self) -> None:
        # team_filter=None: every article is returned, headline byte-for-byte.
        articles = espn_extra.parse_news(_load_news_fixture(), limit=10)
        assert articles is not None
        self.assertEqual(len(articles), 4)
        first = articles[0]
        # VERBATIM: the exact headline string survives with no re-casing/truncation.
        self.assertEqual(first["headline"], _KC_HEADLINE)
        self.assertEqual(
            first["description"],
            "Kansas City's offense was unstoppable in a statement road win over Buffalo.",
        )
        self.assertEqual(first["published"], "2026-01-05T18:00Z")
        self.assertEqual(first["link"], "https://www.espn.com/nfl/story/_/id/1001/chiefs-bills")

    def test_limit_is_honored_top_first(self) -> None:
        articles = espn_extra.parse_news(_load_news_fixture(), limit=2)
        assert articles is not None
        self.assertEqual(len(articles), 2)
        # Top-first in payload order.
        self.assertEqual(articles[0]["headline"], _KC_HEADLINE)
        self.assertEqual(
            articles[1]["headline"], "Bills sign veteran cornerback ahead of playoff push"
        )

    def test_team_filter_keeps_only_matching_articles(self) -> None:
        # A KC filter keeps BOTH KC-tagged headlines and excludes the Bills + league ones.
        articles = espn_extra.parse_news(
            _load_news_fixture(), team_filter=("KC", "KANSAS CITY CHIEFS"), limit=10
        )
        assert articles is not None
        headlines = [a["headline"] for a in articles]
        self.assertIn(_KC_HEADLINE, headlines)
        self.assertIn("Chiefs place starting guard on injured reserve", headlines)
        # The non-KC (Bills) and league-wide articles are filtered out client-side.
        self.assertNotIn("Bills sign veteran cornerback ahead of playoff push", headlines)
        self.assertNotIn("NFL announces 2026 international games slate", headlines)

    def test_team_filter_matches_via_display_name_substring(self) -> None:
        # The canonical display name matching an ESPN category description also works
        # (name_upper in descriptor), not only an exact abbreviation match.
        articles = espn_extra.parse_news(
            _load_news_fixture(), team_filter=("BUF", "BUFFALO BILLS"), limit=10
        )
        assert articles is not None
        self.assertEqual(
            [a["headline"] for a in articles],
            ["Bills sign veteran cornerback ahead of playoff push"],
        )

    def test_team_filter_matching_nothing_returns_empty_list(self) -> None:
        # A real team with no matching article -> [] (distinct from the None failure).
        articles = espn_extra.parse_news(
            _load_news_fixture(), team_filter=("SF", "SAN FRANCISCO 49ERS"), limit=10
        )
        self.assertEqual(articles, [])

    def test_empty_articles_returns_empty_list(self) -> None:
        self.assertEqual(espn_extra.parse_news({"articles": []}, limit=10), [])

    def test_missing_link_and_published_carry_none(self) -> None:
        # The 4th fixture article has no links/link and the parser must carry None,
        # never invent one.
        articles = espn_extra.parse_news(_load_news_fixture(), limit=10)
        assert articles is not None
        ir = next(
            a for a in articles if a["headline"] == "Chiefs place starting guard on injured reserve"
        )
        self.assertIsNone(ir["link"])
        # An article with only lastModified (no published) uses it as the as-of stamp.
        intl = next(
            a for a in articles if a["headline"] == "NFL announces 2026 international games slate"
        )
        self.assertEqual(intl["published"], "2026-01-03T12:00Z")

    def test_non_dict_payload_returns_none(self) -> None:
        for bad in (None, "garbage", 42, ["articles"]):
            self.assertIsNone(espn_extra.parse_news(bad, limit=10))

    def test_articles_not_a_list_returns_none(self) -> None:
        self.assertIsNone(espn_extra.parse_news({"articles": "nope"}, limit=10))
        self.assertIsNone(espn_extra.parse_news({}, limit=10))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "articles": [
                "garbage",
                42,
                {"description": "no headline here"},  # missing headline -> skipped
                {"headline": ""},  # blank headline -> skipped
                {"headline": "Real Headline", "links": "not-a-list", "link": 42},
            ]
        }
        articles = espn_extra.parse_news(payload, limit=10)
        assert articles is not None
        # Only the one real, headline-bearing article survives; nothing fabricated.
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["headline"], "Real Headline")
        self.assertIsNone(articles[0]["link"])


# --------------------------------------------------------------------------- #
# Impure news fetch (cache + HTTP), fully monkeypatched.
# --------------------------------------------------------------------------- #


class FetchNewsTests(unittest.TestCase):
    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_news_fixture()
        limit = espn_extra.NEWS_FETCH_LIMIT
        fake = _FakeRedis({espn_extra._news_cache_key(limit): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(espn_extra.fetch_news())
        self.assertEqual(out, payload)  # served from cache
        self.assertEqual(fake.sets, [])  # nothing re-written on a hit

    def test_cache_miss_fetches_and_writes_cache(self) -> None:
        payload = _load_news_fixture()
        fake = _FakeRedis()  # empty -> miss
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient._response = _FakeResponse(200, payload)
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_news(limit=25))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)  # exactly one GET on a miss
        assert _CapturingAsyncClient.last_url is not None
        # The URL is the league news endpoint carrying the limit — never a team param.
        self.assertIn("news?limit=25", _CapturingAsyncClient.last_url)
        self.assertNotIn("team=", _CapturingAsyncClient.last_url)
        # The raw payload was written to the cache under the news key + TTL.
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._news_cache_key(25))
        self.assertEqual(json.loads(value), payload)
        self.assertEqual(ex, espn_extra.NEWS_CACHE_TTL_SECONDS)

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            out = _run(espn_extra.fetch_news())
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])  # nothing cached on a failed fetch

    def test_non_200_degrades_to_none(self) -> None:
        fake = _FakeRedis()
        _CapturingAsyncClient._response = _FakeResponse(503, {})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_news())
        self.assertIsNone(out)

    def test_redis_error_still_allows_live_fetch(self) -> None:
        # A Redis outage on the read must FAIL OPEN -> the live fetch still happens.
        payload = _load_news_fixture()
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient._response = _FakeResponse(200, payload)
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_news())
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
