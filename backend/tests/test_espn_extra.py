"""Offline unit tests for the ESPN "extras" adapter (260709-u0z, 260820-oym).

Three layers are exercised, all fully OFFLINE (no network socket, no Redis socket):

* the PURE :func:`app.services.espn_extra.parse_injuries` against a captured
  ``summary`` fixture (multi-player team, empty-injuries team, team-filtering) plus
  inline malformed inputs — it must never raise and must return the right
  list / ``[]`` / ``None`` distinction;
* the IMPURE per-endpoint fetches (:func:`fetch_injuries`, :func:`fetch_news`,
  :func:`fetch_team_roster`) with ``httpx`` and the ``_redis_client`` seam monkeypatched
  (mirroring the capturing-client style of ``tests/test_qa_classifier.py``) to prove
  cache HIT (no HTTP), cache MISS (HTTP + cache write), and best-effort ``None`` on a
  fetch/Redis failure — plus, for the roster, that a team outside the canonical 32
  performs ZERO HTTP (T-oym-01);
* the generic shell :func:`_fetch_cached` they all delegate to, driven with an arbitrary
  url/key/TTL, which is where the no-User-Agent and explicit-timeout invariants are
  pinned.

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
from typing import Any
from unittest import mock

import httpx

from app.services import espn_extra

_FIXTURE = Path(__file__).parent / "fixtures" / "espn_summary_injuries.json"
_NEWS_FIXTURE = Path(__file__).parent / "fixtures" / "espn_news.json"
_ROSTER_FIXTURE = Path(__file__).parent / "fixtures" / "espn_team_roster.json"
_STATS_FIXTURE = Path(__file__).parent / "fixtures" / "espn_athlete_stats.json"

# The DISTINCTIVE KC headline the no-rephrasing regression asserts survives byte-for-byte.
_KC_HEADLINE = "Patrick Mahomes throws for 5 touchdowns as Chiefs storm past Bills 38-20"


def _run(coro):
    return asyncio.run(coro)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _load_news_fixture() -> dict:
    return json.loads(_NEWS_FIXTURE.read_text())


def _load_roster_fixture() -> dict:
    return json.loads(_ROSTER_FIXTURE.read_text())


def _load_stats_fixture() -> dict:
    """Matthew Stafford's real career table, trimmed to passing + defensive, 2024-2025.

    ``defensive`` is kept BECAUSE its labels collide (``YDS`` twice) — the trim would be
    pointless without the category the collision test needs.
    """
    return json.loads(_STATS_FIXTURE.read_text())


# The LAR roster is built here rather than captured because the live one carries 93
# players and the resolver only needs the awkward names. Every id, spelling and suffix
# below was read off the live LAR roster on 2026-08-20 — Kyren and Mario Williams really
# do share a surname on this team, so the ambiguity case is not contrived.
def _lar_roster() -> dict:
    def athlete(athlete_id: str, first: str, last: str, position: str) -> dict:
        return {
            "id": athlete_id,
            "firstName": first,
            "lastName": last,
            "displayName": f"{first} {last}",
            "position": {"abbreviation": position, "displayName": position},
        }

    return {
        "team": {"abbreviation": "LAR", "displayName": "Los Angeles Rams"},
        "athletes": [
            {
                "position": "offense",
                "items": [
                    athlete("12483", "Matthew", "Stafford", "QB"),
                    athlete("4430737", "Kyren", "Williams", "RB"),
                    athlete("4431618", "Mario", "Williams", "WR"),
                    athlete("4426512", "Warren", "McClendon Jr.", "OT"),
                    athlete("4259553", "Stetson", "Bennett IV", "QB"),
                ],
            },
            {
                "position": "defense",
                "items": [
                    athlete("4690606", "Al'zillion", "Hamilton", "CB"),
                    athlete("4432266", "Nikhai", "Hill-Green", "LB"),
                ],
            },
        ],
    }


# --------------------------------------------------------------------------- #
# Fake outbound seams (never open a real socket).
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
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
        self.gets: list[str] = []

    async def get(self, key):
        self.gets.append(key)
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


class FetchCachedTests(unittest.TestCase):
    """The ONE generic shell, driven with an arbitrary url/key/TTL — endpoint-independent."""

    _URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/anything"
    _KEY = "qa:generic:probe"
    _TTL = 42

    def _fetch(self):
        return espn_extra._fetch_cached(
            self._URL, cache_key=self._KEY, ttl_seconds=self._TTL, label="generic"
        )

    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = {"probe": True}
        fake = _FakeRedis({self._KEY: json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(self._fetch())
        self.assertEqual(out, payload)  # served from cache
        self.assertEqual(fake.sets, [])  # nothing re-written on a hit

    def test_cache_miss_fetches_once_and_writes_the_exact_key_and_ttl(self) -> None:
        payload = {"probe": True}
        fake = _FakeRedis()  # empty -> miss
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(self._fetch())
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)  # exactly one GET on a miss
        self.assertEqual(_CapturingAsyncClient.last_url, self._URL)  # verbatim, unmodified
        # Written under the key and TTL the CALLER passed, not any endpoint's own.
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, self._KEY)
        self.assertEqual(json.loads(value), payload)
        self.assertEqual(ex, self._TTL)

    def test_no_custom_user_agent_and_explicit_timeout(self) -> None:
        # Invariant 1: ESPN's edge 403s branded User-Agents, so no headers may be sent.
        # Invariant 2: a hung ESPN response must not be able to block the bot loop.
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, {"probe": True}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(self._fetch())
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": espn_extra.DEFAULT_TIMEOUT}
        )

    def test_non_200_degrades_to_none_and_caches_nothing(self) -> None:
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(503, {}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(self._fetch())
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none_and_caches_nothing(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            out = _run(self._fetch())
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])

    def test_non_dict_payload_degrades_to_none_and_caches_nothing(self) -> None:
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, ["not", "a", "dict"]))  # pyright: ignore[reportArgumentType]
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(self._fetch())
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])

    def test_redis_outage_still_serves_a_live_fetch(self) -> None:
        # FAIL-OPEN on the read: a cache outage degrades to the live fetch, never blocks it.
        payload = {"probe": True}
        self._arm_client(_FakeResponse(200, payload))
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(self._fetch())
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)


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

    def test_link_extraction_handles_espn_dict_shape_and_fallbacks(self) -> None:
        # ESPN's real shape is links as a DICT keyed by surface: prefer web.href,
        # fall back to mobile.href, then a list of {href}, then a singular link.href.
        cases = [
            ({"links": {"web": {"href": "W"}, "mobile": {"href": "M"}}}, "W"),
            ({"links": {"mobile": {"href": "M"}}}, "M"),  # no web -> mobile fallback
            ({"links": [{"href": "L1"}, {"href": "L2"}]}, "L1"),  # list tolerance
            ({"link": {"href": "S"}}, "S"),  # singular link fallback
            ({"links": {"web": {}}}, None),  # web present but no href -> None
            ({}, None),  # nothing -> None, never fabricated
        ]
        for extra, expected in cases:
            payload = {"articles": [{"headline": "H", **extra}]}
            articles = espn_extra.parse_news(payload, limit=10)
            assert articles is not None
            self.assertEqual(articles[0]["link"], expected, msg=str(extra))

    def test_filter_news_by_subject_narrows_matches_and_fallbacks(self) -> None:
        articles = [
            {
                "headline": "What is the Chiefs' ceiling this season?",
                "description": "A look at Kansas City's upside.",
                "teams": ["KC", "KANSAS CITY CHIEFS", "PATRICK MAHOMES"],  # athlete tag
            },
            {
                "headline": "Chiefs sign a veteran guard",
                "description": "Line depth for the playoff run.",
                "teams": ["KC", "KANSAS CITY CHIEFS"],
            },
        ]
        # A specific athlete matches only the article tagged with him (via descriptors,
        # even though his name is NOT in that headline).
        hit = espn_extra.filter_news_by_subject(articles, "Patrick Mahomes")
        assert hit is not None
        self.assertEqual([a["headline"] for a in hit], ["What is the Chiefs' ceiling this season?"])
        # ALL meaningful tokens must appear — a subject no article satisfies -> [] (the
        # caller then falls back to the full feed, never empty).
        self.assertEqual(espn_extra.filter_news_by_subject(articles, "Travis Kelce"), [])
        # An all-generic subject carries no signal -> None (NO narrowing applied).
        self.assertIsNone(espn_extra.filter_news_by_subject(articles, "recent news"))
        self.assertIsNone(espn_extra.filter_news_by_subject(articles, None))
        self.assertIsNone(espn_extra.filter_news_by_subject(articles, "the latest"))

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


# --------------------------------------------------------------------------- #
# Pure roster parser (compact facts, no starter claim).
# --------------------------------------------------------------------------- #


class ParseTeamRosterTests(unittest.TestCase):
    def test_position_query_returns_that_position_including_the_injured_one(self) -> None:
        facts = espn_extra.parse_team_roster(_load_roster_fixture(), position="QB")
        assert facts is not None
        self.assertEqual(facts["team"], "Chicago Bears")
        self.assertEqual(facts["season_year"], 2026)
        self.assertEqual(facts["season_type"], "Preseason")
        names = [p["display_name"] for p in facts["players"]]
        # The injured-reserve QB is a real roster fact and is NOT silently dropped.
        self.assertEqual(names, ["Caleb Williams", "Tyson Bagent", "Case Keenum"])
        starter = facts["players"][0]
        self.assertEqual(starter["position"], "QB")
        self.assertEqual(starter["jersey"], "18")
        self.assertEqual(starter["experience_years"], 3)
        self.assertEqual(starter["status"], "Active")
        self.assertEqual(starter["group"], "offense")
        injured = facts["players"][2]
        self.assertEqual(injured["status"], "Day-To-Day")
        self.assertEqual(injured["group"], "injured reserve or out")

    def test_no_position_returns_counts_only(self) -> None:
        facts = espn_extra.parse_team_roster(_load_roster_fixture())
        assert facts is not None
        self.assertEqual(facts["players"], [])  # counts alone are the answer
        self.assertEqual(facts["position_counts"], {"QB": 3, "WR": 1, "CB": 1, "PK": 1})
        self.assertFalse(facts["truncated"])

    def test_unplayed_position_returns_empty_players_with_counts_intact(self) -> None:
        facts = espn_extra.parse_team_roster(_load_roster_fixture(), position="LS")
        assert facts is not None
        self.assertEqual(facts["players"], [])  # a VALID empty answer, not a failure
        self.assertEqual(facts["position_counts"], {"QB": 3, "WR": 1, "CB": 1, "PK": 1})

    def test_position_matching_is_case_insensitive_and_accepts_the_full_name(self) -> None:
        for asked in ("qb", " Qb ", "quarterback", "Quarterback"):
            with self.subTest(position=asked):
                facts = espn_extra.parse_team_roster(_load_roster_fixture(), position=asked)
                assert facts is not None
                self.assertEqual(len(facts["players"]), 3)

    def test_caveat_is_the_module_constant_verbatim(self) -> None:
        facts = espn_extra.parse_team_roster(_load_roster_fixture(), position="QB")
        assert facts is not None
        self.assertEqual(facts["caveat"], espn_extra.ROSTER_CAVEAT)
        # The sentence the model is most likely to voice must SAY it does not know.
        self.assertIn("does not", espn_extra.ROSTER_CAVEAT)
        self.assertIn("start", espn_extra.ROSTER_CAVEAT)

    def test_unusable_top_level_shapes_return_none(self) -> None:
        for bad in (None, "garbage", 42, ["athletes"]):
            self.assertIsNone(espn_extra.parse_team_roster(bad))
        self.assertIsNone(espn_extra.parse_team_roster({"athletes": "nope"}))
        self.assertIsNone(espn_extra.parse_team_roster({}))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "athletes": [
                "garbage",
                {"position": "offense", "items": "not-a-list"},
                {
                    "position": "offense",
                    "items": [
                        "nope",
                        {"jersey": "99"},  # no displayName -> never invented
                        {"displayName": "No Position"},  # no position block -> skipped
                        {
                            "displayName": "Real Player",
                            "position": {"abbreviation": "QB", "name": "Quarterback"},
                        },
                    ],
                },
            ]
        }
        facts = espn_extra.parse_team_roster(payload, position="QB")
        assert facts is not None
        self.assertEqual([p["display_name"] for p in facts["players"]], ["Real Player"])
        self.assertEqual(facts["position_counts"], {"QB": 1})
        # A missing field degrades to None rather than being invented.
        self.assertIsNone(facts["players"][0]["jersey"])
        self.assertIsNone(facts["players"][0]["status"])
        self.assertIsNone(facts["team"])

    def test_players_are_capped_and_the_truncated_flag_reports_it(self) -> None:
        cap = espn_extra.ROSTER_MAX_PLAYERS
        items = [
            {"displayName": f"Player {i}", "position": {"abbreviation": "QB"}}
            for i in range(cap + 5)
        ]
        payload = {"athletes": [{"position": "offense", "items": items}]}
        facts = espn_extra.parse_team_roster(payload, position="QB")
        assert facts is not None
        self.assertEqual(len(facts["players"]), cap)
        self.assertTrue(facts["truncated"])
        # The COUNT still reports the true roster size — only the name list is capped.
        self.assertEqual(facts["position_counts"], {"QB": cap + 5})


# --------------------------------------------------------------------------- #
# Impure roster fetch — the ALLOWLIST is the security assertion of this section.
# --------------------------------------------------------------------------- #


class FetchTeamRosterTests(unittest.TestCase):
    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient._response = response

    def test_lowercase_and_padded_abbreviations_normalize_into_the_url(self) -> None:
        payload = _load_roster_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_team_roster("  chi "))
        self.assertEqual(out, payload)
        assert _CapturingAsyncClient.last_url is not None
        self.assertIn("/teams/CHI/roster", _CapturingAsyncClient.last_url)

    def test_a_team_outside_the_canonical_32_performs_zero_http(self) -> None:
        # T-oym-01: the allowlist runs BEFORE the URL is formatted. A regression that
        # formats first and validates second fails here, loudly.
        fake = _FakeRedis()
        for bogus in ("", "ZZZ", "../../etc/passwd", "CHI/../DAL", "chi;rm -rf /"):
            with self.subTest(team=bogus):
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_team_roster(bogus)))
        self.assertEqual(fake.gets, [])  # Redis is not touched either
        self.assertEqual(fake.sets, [])

    def test_a_non_string_argument_does_not_raise(self) -> None:
        fake = _FakeRedis()
        for bogus in (None, 42, ["CHI"]):
            with self.subTest(team=bogus):
                bad: Any = bogus
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_team_roster(bad)))

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_roster_fixture()
        fake = _FakeRedis({espn_extra._roster_cache_key("CHI"): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(espn_extra.fetch_team_roster("CHI"))
        self.assertEqual(out, payload)
        self.assertEqual(fake.sets, [])

    def test_cache_miss_writes_the_roster_key_and_ttl(self) -> None:
        payload = _load_roster_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_team_roster("DAL"))
        self.assertEqual(out, payload)
        self.assertEqual(len(fake.sets), 1)
        key, _value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._roster_cache_key("DAL"))
        self.assertEqual(ex, espn_extra.ROSTER_CACHE_TTL_SECONDS)

    def test_every_canonical_abbreviation_is_accepted(self) -> None:
        # The allowlist must be the canonical table itself, never a drifted copy.
        self.assertEqual(len(espn_extra.NFL_TEAM_ABBRS), 32)
        self.assertIn("WSH", espn_extra.NFL_TEAM_ABBRS)
        self.assertIn("JAX", espn_extra.NFL_TEAM_ABBRS)


# --------------------------------------------------------------------------- #
# Name -> athlete id resolution (pure, over the RAW roster payload).
# --------------------------------------------------------------------------- #


class FindRosterAthletesTests(unittest.TestCase):
    def test_a_shortened_first_name_resolves_to_the_full_roster_name(self) -> None:
        matches = espn_extra.find_roster_athletes(_lar_roster(), "Matt Stafford")
        assert matches is not None
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["athlete_id"], "12483")
        self.assertEqual(matches[0]["display_name"], "Matthew Stafford")
        self.assertEqual(matches[0]["position"], "QB")


# --------------------------------------------------------------------------- #
# Pure season-selecting stats parser.
# --------------------------------------------------------------------------- #


class ParseAthleteStatsTests(unittest.TestCase):
    def test_no_season_argument_selects_the_newest_season_in_the_payload(self) -> None:
        # D-1 (LOCKED): the default season comes from ESPN's OWN table, never a DB read.
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture())
        assert facts is not None
        self.assertEqual(facts["season"], 2025)
        self.assertEqual(facts["available_seasons"], [2024, 2025])
        self.assertEqual(facts["stats"]["Passing"]["Passing Yards"], "4,707")

    def test_an_explicit_season_overrides_the_default(self) -> None:
        # ESPN ignores ``?season=``, so ONE cached payload has to answer any season.
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture(), season=2024)
        assert facts is not None
        self.assertEqual(facts["season"], 2024)
        self.assertEqual(facts["stats"]["Passing"]["Passing Yards"], "3,762")


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
