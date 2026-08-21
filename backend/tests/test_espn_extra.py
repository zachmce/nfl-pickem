"""Offline unit tests for the ESPN "extras" adapter (260709-u0z, 260820-oym, 260820-s5y).

Three layers are exercised, all fully OFFLINE (no network socket, no Redis socket):

* the PURE :func:`app.services.espn_extra.parse_injuries` against a captured
  ``summary`` fixture (multi-player team, empty-injuries team, team-filtering) plus
  inline malformed inputs — it must never raise and must return the right
  list / ``[]`` / ``None`` distinction;
* the PURE :func:`find_roster_athletes` name resolver and :func:`parse_athlete_stats`
  season selector, against a captured roster and a captured career table — the label
  collision, the shared surname and the suffixed surname are the cases worth pinning;
* the IMPURE per-endpoint fetches (:func:`fetch_injuries`, :func:`fetch_news`,
  :func:`fetch_team_roster`, :func:`fetch_athlete_stats`) with ``httpx`` and the
  ``_redis_client`` seam monkeypatched
  (mirroring the capturing-client style of ``tests/test_qa_classifier.py``) to prove
  cache HIT (no HTTP), cache MISS (HTTP + cache write), and best-effort ``None`` on a
  fetch/Redis failure — plus, for the roster, that a team outside the canonical 32
  performs ZERO HTTP (T-oym-01), and for the stats, that a non-digit athlete id does
  the same (T-s5y-01);
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
_SEARCH_FIXTURE = Path(__file__).parent / "fixtures" / "espn_athlete_search.json"
_SCHEDULE_FIXTURE = Path(__file__).parent / "fixtures" / "espn_team_schedule.json"
_GAME_LEADERS_FIXTURE = Path(__file__).parent / "fixtures" / "espn_game_leaders.json"
_SUPER_BOWL_FIXTURE = Path(__file__).parent / "fixtures" / "espn_postseason_super_bowl.json"

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


def _load_schedule_fixture() -> dict:
    """KC's REAL 2025 regular-season schedule, captured 2026-08-21 and trimmed to weeks
    1, 9, 11 and 18 (9 and 11 straddle KC's real week-10 bye) plus ONE hand-added
    synthetic week-19 event that is not completed, so the not-yet-played branch has a
    case without a second capture. The misleading top-level ``season`` block is kept
    VERBATIM as a decoy: it claims 2026 Preseason on a request that returned 2025."""
    return json.loads(_SCHEDULE_FIXTURE.read_text())


def _load_game_leaders_fixture() -> dict:
    """The REAL summary leaders for KC at LV, 2025 week 18, captured 2026-08-21 and cut
    to the keys the parser reads. The two scores are replaced with the SENTINELS 9991
    and 9992, which is how D-2's never-read rule becomes an assertion rather than a
    comment."""
    return json.loads(_GAME_LEADERS_FIXTURE.read_text())


def _load_super_bowl_fixture() -> dict:
    """The REAL 2025 postseason week 5 — Super Bowl LX, captured 2026-08-21 and cut to the
    keys the parser reads. The two scores are replaced with the SENTINELS 9991 and 9992,
    which is how "the score is never read" becomes an assertion rather than a comment."""
    return json.loads(_SUPER_BOWL_FIXTURE.read_text())


def _unplayed_super_bowl() -> dict:
    """The MEASURED shape of a postseason still ahead of the calendar (``dates=2026``).

    Built here rather than captured because it will stop being true in February 2027: ESPN
    schedules the whole bracket with ``TBD`` competitors, which is exactly why the caller
    must never relay the games on this branch.
    """
    return {
        "season": {"type": 3, "year": 2026},
        "week": {"number": 5},
        "events": [
            {
                "id": "401856001",
                "shortName": "TBD VS TBD",
                "date": "2027-02-14T23:30Z",
                "competitions": [
                    {
                        "notes": [{"type": "event", "headline": "Super Bowl LXI"}],
                        "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False}},
                        "competitors": [
                            {"winner": None, "score": "0", "team": {"displayName": "TBD"}},
                            {"winner": None, "score": "0", "team": {"displayName": "TBD"}},
                        ],
                    }
                ],
            }
        ],
    }


def _conference_championships() -> dict:
    """The REAL 2025 postseason week 3, read live 2026-08-21 and trimmed by hand.

    Two games under two different ESPN headlines is the case a single-game fixture cannot
    cover, and the scores are the same sentinels for the same reason.
    """

    def _game(headline: str, winner: str, loser: str, date: str) -> dict:
        return {
            "id": "401772980",
            "shortName": f"{winner} @ {loser}",
            "date": date,
            "competitions": [
                {
                    "notes": [{"type": "event", "headline": headline}],
                    "status": {"type": {"name": "STATUS_FINAL", "completed": True}},
                    "competitors": [
                        {"winner": False, "score": "9991", "team": {"displayName": loser}},
                        {"winner": True, "score": "9992", "team": {"displayName": winner}},
                    ],
                }
            ],
        }

    return {
        "season": {"type": 3, "year": 2025},
        "week": {"number": 3},
        "events": [
            _game(
                "AFC Championship",
                "New England Patriots",
                "Denver Broncos",
                "2026-01-25T20:00Z",
            ),
            _game(
                "NFC Championship",
                "Seattle Seahawks",
                "Los Angeles Rams",
                "2026-01-25T23:30Z",
            ),
        ],
    }


def _load_search_fixture() -> dict:
    """The REAL "josh allen" search page, captured 2026-08-20 and kept verbatim.

    Every trap issue #183 records for Route B is in this one page: a Luton Town soccer
    player, two college programmes, a second NFL Josh Allen on Arizona, and the athlete
    id sitting in ``uid`` while the top-level ``id`` is a GUID.
    """
    return json.loads(_SEARCH_FIXTURE.read_text())


def _pacheco_search() -> dict:
    """The measured single-NFL-match page — the case that motivated the fallback.

    Read off the live endpoint on 2026-08-20: asked about on KC, ESPN has him on DET.
    """
    return {
        "results": [
            {
                "type": "player",
                "contents": [
                    {
                        "id": "b80fe7c4-00cc-27e8-a8a5-e4f408778fb0",
                        "uid": "s:20~l:28~a:4361529",
                        "type": "player",
                        "displayName": "Isiah Pacheco",
                        "subtitle": "Detroit Lions",
                    }
                ],
            }
        ]
    }


def _league_root() -> dict:
    """The ESPN league root, trimmed to the block :func:`league_season_year` reads.

    Measured live 2026-08-21: HTTP 200, 11,969 bytes, an explicit top-level
    ``season.year`` of 2026. The ``$ref`` is kept and deliberately points at a DIFFERENT
    year from the field beside it, so a test can tell "read the integer" apart from
    "parsed the URL" — which no live payload could, since the two always agree there.
    """
    return {
        "id": "28",
        "name": "National Football League",
        "season": {
            "$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025",
            "year": 2026,
            "startDate": "2026-08-06T07:00Z",
            "endDate": "2027-02-16T07:59Z",
            "displayName": "2026",
            "type": {"id": "1", "name": "Preseason", "year": 2026},
        },
    }


def _load_stats_fixture() -> dict:
    """Matthew Stafford's real career table, trimmed to passing + defensive, 2024-2025.

    ``defensive`` is kept BECAUSE its labels collide (``YDS`` twice) — the trim would be
    pointless without the category the collision test needs.
    """
    return json.loads(_STATS_FIXTURE.read_text())


def _split_season_stats() -> dict:
    """The captured table with 2025 split between two clubs, in ESPN's measured shape.

    Measured 2026-08-20 on Davante Adams (2024, LV then NYJ) and Diontae Johnson (2024,
    three clubs): each club gets its own row carrying ``teamId`` and a resolvable
    ``teamSlug``, and a FINAL combined row carries no ``teamId`` and slugs itself
    "2024 Totals", which no ``teams`` entry resolves.
    """
    payload = _load_stats_fixture()
    payload["teams"]["kansas-city-chiefs"] = {
        "id": "12",
        "abbreviation": "KC",
        "displayName": "Kansas City Chiefs",
    }
    for category in payload["categories"]:
        rows = category["statistics"]
        combined = json.loads(json.dumps(rows[-1]))
        moved = json.loads(json.dumps(rows[-1]))
        moved["teamId"] = "12"
        moved["teamSlug"] = "kansas-city-chiefs"
        del combined["teamId"]
        combined["teamSlug"] = "2025 Totals"
        combined["displayName"] = "2025  Totals"
        # Games Played tells the three rows apart, so a test can name which one was read.
        rows[-1]["stats"][0], moved["stats"][0], combined["stats"][0] = "9", "8", "17"
        rows.extend([moved, combined])
    return payload


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

    def test_a_full_first_name_resolves_a_shortened_roster_name(self) -> None:
        # Prefix runs BOTH ways, so a roster carrying the short form still resolves.
        payload = {
            "athletes": [
                {
                    "items": [
                        {
                            "id": "12483",
                            "firstName": "Matt",
                            "lastName": "Stafford",
                            "displayName": "Matt Stafford",
                            "position": {"abbreviation": "QB"},
                        }
                    ]
                }
            ]
        }
        matches = espn_extra.find_roster_athletes(payload, "Matthew Stafford")
        assert matches is not None
        self.assertEqual(len(matches), 1)

    def test_two_players_sharing_a_surname_both_come_back(self) -> None:
        # Kyren Williams and Mario Williams are BOTH on the live LAR roster today, so
        # this is a real ambiguous question and not a contrived one. The caller turns
        # two matches into a question back to the member, never a silent pick (D-5).
        matches = espn_extra.find_roster_athletes(_lar_roster(), "Williams")
        assert matches is not None
        self.assertEqual(
            sorted(m["display_name"] for m in matches), ["Kyren Williams", "Mario Williams"]
        )

    def test_a_first_name_disambiguates_a_shared_surname(self) -> None:
        matches = espn_extra.find_roster_athletes(_lar_roster(), "Kyren Williams")
        assert matches is not None
        self.assertEqual([m["athlete_id"] for m in matches], ["4430737"])

    def test_a_name_on_no_roster_entry_returns_an_empty_list_not_none(self) -> None:
        # [] means "the roster read fine and nobody matches" — a FACT the caller
        # reports. None would collapse to the no-data payload and send the model back
        # to its own stale memory, which is what this tool exists to remove.
        matches = espn_extra.find_roster_athletes(_lar_roster(), "Tom Brady")
        self.assertEqual(matches, [])

    def test_a_surname_carrying_a_suffix_matches_the_bare_surname(self) -> None:
        # ESPN puts the suffix INSIDE lastName (measured: 'McClendon Jr.').
        for asked in ("McClendon", "Warren McClendon", "Warren McClendon Jr.", "mcclendon jr"):
            with self.subTest(name=asked):
                matches = espn_extra.find_roster_athletes(_lar_roster(), asked)
                assert matches is not None
                self.assertEqual([m["athlete_id"] for m in matches], ["4426512"])

    def test_punctuated_surnames_match_their_unpunctuated_spellings(self) -> None:
        for asked, expected in (
            ("Nikhai Hill-Green", "4432266"),
            ("Nikhai HillGreen", "4432266"),
            ("Hill-Green", "4432266"),
            ("Al'zillion Hamilton", "4690606"),
            ("Alzillion Hamilton", "4690606"),
        ):
            with self.subTest(name=asked):
                matches = espn_extra.find_roster_athletes(_lar_roster(), asked)
                assert matches is not None
                self.assertEqual([m["athlete_id"] for m in matches], [expected])

    def test_an_athlete_without_a_digit_id_is_skipped(self) -> None:
        # Only a value that could LEGALLY reach the stats URL is ever returned
        # (T-s5y-01) — an id the guard would reject never leaves this function.
        payload = {
            "athletes": [
                {
                    "items": [
                        {"displayName": "No Id At All", "position": {"abbreviation": "QB"}},
                        {"id": "../../etc/passwd", "displayName": "Bad Id"},
                        {"id": 12483, "displayName": "Numeric Id"},  # not a string
                    ]
                }
            ]
        }
        for asked in ("No Id At All", "Bad Id", "Numeric Id"):
            with self.subTest(name=asked):
                self.assertEqual(espn_extra.find_roster_athletes(payload, asked), [])

    def test_an_empty_or_unusable_query_matches_nobody(self) -> None:
        for asked in ("", "   ", None, 42, ["Stafford"]):
            with self.subTest(name=asked):
                bad: Any = asked
                self.assertEqual(espn_extra.find_roster_athletes(_lar_roster(), bad), [])

    def test_unusable_top_level_shapes_return_none(self) -> None:
        for bad in (None, "garbage", 42, ["athletes"]):
            self.assertIsNone(espn_extra.find_roster_athletes(bad, "Stafford"))
        self.assertIsNone(espn_extra.find_roster_athletes({"athletes": "nope"}, "Stafford"))
        self.assertIsNone(espn_extra.find_roster_athletes({}, "Stafford"))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "athletes": [
                "garbage",
                {"items": "not-a-list"},
                {
                    "items": [
                        "nope",
                        {"id": "1"},  # no display name -> never invented
                        {"id": "2", "displayName": "Real Player", "position": "not-a-dict"},
                    ]
                },
            ]
        }
        matches = espn_extra.find_roster_athletes(payload, "Real Player")
        assert matches is not None
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["athlete_id"], "2")
        self.assertIsNone(matches[0]["position"])  # degraded, not invented


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

    def test_a_colliding_label_loses_no_statistic(self) -> None:
        # MEASURED: the defensive labels repeat YDS (fumble-recovery yards, then
        # interception yards), so ``dict(zip(labels, stats))`` silently drops one.
        # Keying on displayNames (D-3) keeps both. The COUNT is asserted as well as the
        # keys, so a future key-fallback change that starts overwriting is caught here.
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture())
        assert facts is not None
        defense = facts["stats"]["Defense"]
        self.assertEqual(len(defense), 17)
        self.assertIn("Fumbles Recovered Yards", defense)
        self.assertIn("Interception Yards", defense)
        # Independent values either side of the collision, not one value written twice.
        self.assertEqual(defense["Total Tackles"], "1")
        self.assertEqual(defense["Games Played"], "17")

    def test_values_are_relayed_verbatim_with_their_thousands_separator(self) -> None:
        # D-2: "4,707" reaches the model as "4,707". Normalising would mean parsing
        # "65.0", "4,707" and "-" through one coercion that can fail, inside a function
        # contracted never to raise, for no gain.
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture())
        assert facts is not None
        passing = facts["stats"]["Passing"]
        self.assertEqual(passing["Passing Yards"], "4,707")
        self.assertNotEqual(passing["Passing Yards"], "4707")
        self.assertIsInstance(passing["Passing Yards"], str)
        self.assertEqual(passing["Completion Percentage"], "65.0")
        self.assertEqual(passing["Adjusted QBR"], "71.2")

    def test_a_season_the_table_does_not_carry_returns_none_with_the_seasons_intact(
        self,
    ) -> None:
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture(), season=2009)
        assert facts is not None
        self.assertIsNone(facts["season"])  # the caller attaches the note; this never phrases
        self.assertEqual(facts["stats"], {})
        self.assertEqual(facts["available_seasons"], [2024, 2025])

    def test_an_all_zero_category_is_dropped(self) -> None:
        # D-6. Games Played is excluded from the test, so a category the player only
        # appeared in is still dropped rather than shown as a wall of zeroes.
        payload = {
            "categories": [
                {
                    "displayName": "Kicking",
                    "displayNames": ["Games Played", "Field Goals Made", "Points"],
                    "statistics": [
                        {"season": {"year": 2025}, "stats": ["17", "0", "-"]},
                    ],
                },
                {
                    "displayName": "Passing",
                    "displayNames": ["Games Played", "Passing Yards"],
                    "statistics": [
                        {"season": {"year": 2025}, "stats": ["17", "4,707"]},
                    ],
                },
            ]
        }
        facts = espn_extra.parse_athlete_stats(payload)
        assert facts is not None
        self.assertEqual(list(facts["stats"]), ["Passing"])

    def test_the_category_count_is_capped(self) -> None:
        cap = espn_extra.STATS_MAX_CATEGORIES
        payload = {
            "categories": [
                {
                    "displayName": f"Category {i}",
                    "displayNames": ["Games Played", "Yards"],
                    "statistics": [{"season": {"year": 2025}, "stats": ["17", "100"]}],
                }
                for i in range(cap + 3)
            ]
        }
        facts = espn_extra.parse_athlete_stats(payload)
        assert facts is not None
        self.assertEqual(len(facts["stats"]), cap)

    def test_caveat_is_the_module_constant_verbatim(self) -> None:
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture())
        assert facts is not None
        self.assertEqual(facts["caveat"], espn_extra.STATS_CAVEAT)
        # The sentence the model is most likely to voice must name the SEASON, and must
        # forbid the relative-time relabelling that is the measured failure (D-1).
        self.assertIn("season's year", espn_extra.STATS_CAVEAT)
        self.assertIn("last year's", espn_extra.STATS_CAVEAT)

    def test_a_key_falls_back_to_names_then_labels(self) -> None:
        payload = {
            "categories": [
                {
                    "displayName": "Passing",
                    "names": ["gamesPlayed", "passingYards"],
                    "labels": ["GP", "YDS"],
                    "statistics": [{"season": {"year": 2025}, "stats": ["17", "4,707"]}],
                },
                {
                    "name": "rushing",
                    "labels": ["GP", "YDS"],
                    "statistics": [{"season": {"year": 2025}, "stats": ["17", "50"]}],
                },
            ]
        }
        facts = espn_extra.parse_athlete_stats(payload)
        assert facts is not None
        self.assertEqual(facts["stats"]["Passing"], {"gamesPlayed": "17", "passingYards": "4,707"})
        self.assertEqual(facts["stats"]["rushing"], {"GP": "17", "YDS": "50"})

    def test_unusable_top_level_shapes_return_none(self) -> None:
        for bad in (None, "garbage", 42, ["categories"]):
            self.assertIsNone(espn_extra.parse_athlete_stats(bad))
        self.assertIsNone(espn_extra.parse_athlete_stats({"categories": "nope"}))
        self.assertIsNone(espn_extra.parse_athlete_stats({}))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "categories": [
                "garbage",
                {"displayName": "No Rows", "statistics": "not-a-list"},
                {"displayNames": ["Yards"], "statistics": []},  # no usable label
                {
                    "displayName": "Passing",
                    "displayNames": ["Games Played", None, "Passing Yards"],
                    "statistics": [
                        "nope",
                        {"season": "not-a-dict", "stats": ["1"]},
                        {"season": {"year": "2025"}, "stats": ["1"]},  # year not an int
                        {"season": {"year": 2025}, "stats": ["17", {"a": 1}, "4,707"]},
                    ],
                },
            ]
        }
        facts = espn_extra.parse_athlete_stats(payload)
        assert facts is not None
        self.assertEqual(facts["available_seasons"], [2025])
        # The unusable value at index 1 is skipped WITHOUT shifting the ones after it —
        # index 2 still pairs with displayNames[2], so nothing is mislabelled.
        self.assertEqual(
            facts["stats"]["Passing"], {"Games Played": "17", "Passing Yards": "4,707"}
        )

    def test_an_empty_category_list_returns_no_season(self) -> None:
        facts = espn_extra.parse_athlete_stats({"categories": []})
        assert facts is not None
        self.assertIsNone(facts["season"])
        self.assertEqual(facts["available_seasons"], [])
        self.assertEqual(facts["stats"], {})

    def test_the_season_team_is_the_one_that_seasons_rows_name(self) -> None:
        # THE LIVE DEFECT (2026-08-20): the figures are a PAST season's and the only team
        # name in hand was the player's CURRENT one, so the model said Pacheco played for
        # Detroit in 2025. He played for Kansas City; his 2025 row says so.
        facts = espn_extra.parse_athlete_stats(_load_stats_fixture())
        assert facts is not None
        self.assertEqual(facts["season_teams"], ["Los Angeles Rams"])

    def test_an_unresolvable_team_slug_leaves_the_season_team_unstated(self) -> None:
        # Degrade, never guess: a missing slug, an absent ``teams`` block and a slug the
        # block does not carry all leave the season's team unsaid rather than borrowing
        # one from somewhere else in the payload.
        for mangle in ("drop-block", "drop-slug", "unknown-slug"):
            with self.subTest(mangle=mangle):
                payload = _load_stats_fixture()
                if mangle == "drop-block":
                    del payload["teams"]
                for category in payload["categories"]:
                    for row in category["statistics"]:
                        if mangle == "drop-slug":
                            del row["teamSlug"]
                        elif mangle == "unknown-slug":
                            row["teamSlug"] = "not-a-real-club"
                facts = espn_extra.parse_athlete_stats(payload)
                assert facts is not None
                self.assertEqual(facts["season_teams"], [])
                self.assertEqual(facts["stats"]["Passing"]["Passing Yards"], "4,707")

    def test_a_malformed_teams_block_never_raises(self) -> None:
        for teams in ("garbage", 42, ["los-angeles-rams"], {"los-angeles-rams": "nope"}):
            with self.subTest(teams=teams):
                payload = _load_stats_fixture()
                payload["teams"] = teams
                facts = espn_extra.parse_athlete_stats(payload)
                assert facts is not None
                self.assertEqual(facts["season_teams"], [])

    def test_a_season_split_between_clubs_names_both_and_reads_the_combined_row(
        self,
    ) -> None:
        # Measured live: one row per club plus a combined row. Taking the first match
        # would report one club's PARTIAL figures as the season's own AND name only that
        # club — the same misattribution, one layer down.
        facts = espn_extra.parse_athlete_stats(_split_season_stats())
        assert facts is not None
        self.assertEqual(facts["season"], 2025)
        self.assertEqual(facts["season_teams"], ["Los Angeles Rams", "Kansas City Chiefs"])
        self.assertEqual(facts["stats"]["Passing"]["Games Played"], "17")

    def test_a_split_season_with_no_combined_row_reports_no_figures_for_it(self) -> None:
        # Defensive branch: ESPN has always emitted the combined row in every split season
        # measured, so this shape is unverified against live data. One club's partial is
        # never relayed as the season's total, and both clubs are still named.
        payload = _split_season_stats()
        for category in payload["categories"]:
            category["statistics"] = [row for row in category["statistics"] if "teamId" in row]
        facts = espn_extra.parse_athlete_stats(payload)
        assert facts is not None
        self.assertEqual(facts["season_teams"], ["Los Angeles Rams", "Kansas City Chiefs"])
        self.assertEqual(facts["stats"], {})

    def test_the_season_teams_survive_the_category_cap(self) -> None:
        # Teams are gathered in their own pass, so the club a player moved to is named
        # even when the category recording the move falls outside the relayed ones.
        payload = _split_season_stats()
        payload["categories"] = payload["categories"][:1] + [
            {
                "displayName": f"Category {i}",
                "displayNames": ["Games Played", "Yards"],
                "statistics": [{"season": {"year": 2025}, "stats": ["17", "100"], "teamId": "14"}],
            }
            for i in range(espn_extra.STATS_MAX_CATEGORIES + 3)
        ]
        facts = espn_extra.parse_athlete_stats(payload)
        assert facts is not None
        self.assertEqual(len(facts["stats"]), espn_extra.STATS_MAX_CATEGORIES)
        self.assertIn("Kansas City Chiefs", facts["season_teams"])


# --------------------------------------------------------------------------- #
# Impure stats fetch — the DIGIT GUARD is the security assertion of this section.
# --------------------------------------------------------------------------- #


class FetchAthleteStatsTests(unittest.TestCase):
    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_a_non_digit_id_performs_zero_http_and_zero_redis(self) -> None:
        # T-s5y-01: the guard runs BEFORE the format string. A regression that formats
        # first and validates second fails here, loudly.
        fake = _FakeRedis()
        for bogus in ("", "   ", "../../etc/passwd", "12483/../99", "12483;rm -rf /", "abc"):
            with self.subTest(athlete_id=bogus):
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_athlete_stats(bogus)))
        self.assertEqual(fake.gets, [])  # Redis is not touched either
        self.assertEqual(fake.sets, [])

    def test_a_non_string_argument_does_not_raise(self) -> None:
        fake = _FakeRedis()
        for bogus in (None, 12483, ["12483"]):
            with self.subTest(athlete_id=bogus):
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_athlete_stats(bogus)))
        self.assertEqual(fake.gets, [])

    def test_an_over_long_digit_string_is_rejected(self) -> None:
        fake = _FakeRedis()
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            self.assertIsNone(_run(espn_extra.fetch_athlete_stats("1" * 13)))
        self.assertEqual(fake.gets, [])

    def test_cache_miss_issues_one_get_on_the_web_api_host(self) -> None:
        payload = _load_stats_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_athlete_stats(" 12483 "))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)
        url = _CapturingAsyncClient.last_url
        assert url is not None
        # The NEW subdomain is still an ESPN edge, so the no-UA and explicit-timeout
        # invariants hold here exactly as they do on the old one.
        self.assertTrue(url.startswith("https://site.web.api.espn.com/"))
        self.assertTrue(url.endswith("/athletes/12483/stats"))
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": espn_extra.DEFAULT_TIMEOUT}
        )

    def test_cache_miss_writes_the_stats_key_and_ttl(self) -> None:
        payload = _load_stats_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(espn_extra.fetch_athlete_stats("12483"))
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._athlete_stats_cache_key("12483"))
        self.assertEqual(json.loads(value), payload)
        self.assertEqual(ex, espn_extra.ATHLETE_STATS_CACHE_TTL_SECONDS)

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_stats_fixture()
        fake = _FakeRedis({espn_extra._athlete_stats_cache_key("12483"): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(espn_extra.fetch_athlete_stats("12483"))
        self.assertEqual(out, payload)
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            out = _run(espn_extra.fetch_athlete_stats("12483"))
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])


class ParseAthleteSearchTests(unittest.TestCase):
    def test_a_single_nfl_match_returns_the_id_from_uid_and_the_team_name(self) -> None:
        # The motivating live case: asked about on KC, ESPN has him on DET, and the data
        # was always answerable — only the id resolution failed.
        found = espn_extra.parse_athlete_search(_pacheco_search())
        self.assertEqual(
            found,
            [
                {
                    "athlete_id": "4361529",
                    "display_name": "Isiah Pacheco",
                    "team_name": "Detroit Lions",
                }
            ],
        )

    def test_the_id_never_comes_from_the_top_level_guid(self) -> None:
        found = espn_extra.parse_athlete_search(_pacheco_search())
        assert found is not None
        self.assertNotIn("-", found[0]["athlete_id"])  # the GUID is hyphenated; the id is not

    def test_the_cross_sport_and_college_results_are_filtered_out(self) -> None:
        # Measured: "josh allen" returns a Luton Town SOCCER player and two college
        # programmes alongside the NFL players. Only the 32 club subtitles survive.
        found = espn_extra.parse_athlete_search(_load_search_fixture())
        assert found is not None
        self.assertEqual(
            [(one["display_name"], one["team_name"]) for one in found],
            [
                ("Josh Allen", "Buffalo Bills"),
                ("Josh Allen", "Arizona Cardinals"),
                ("Josh Hines-Allen", "Jacksonville Jaguars"),
            ],
        )

    def test_a_nonsense_query_page_returns_an_empty_list_not_none(self) -> None:
        # Measured: a nonsense query answers 200 with no ``results`` key at all. That is
        # "nobody by that name", which the caller reports — not a fetch failure.
        self.assertEqual(espn_extra.parse_athlete_search({"totalFound": None}), [])

    def test_a_non_dict_payload_returns_none(self) -> None:
        for bogus in (None, [], "results", 7):
            with self.subTest(payload=bogus):
                self.assertIsNone(espn_extra.parse_athlete_search(bogus))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "results": [
                None,
                "player",
                {"contents": "nope"},
                {
                    "contents": [
                        None,
                        {"type": "team", "displayName": "Buffalo Bills", "uid": "s:20~l:28~t:2"},
                        {"type": "player", "subtitle": "Buffalo Bills", "uid": "s:20~l:28~a:1"},
                        {"type": "player", "displayName": "No Team", "uid": "s:20~l:28~a:2"},
                        {"type": "player", "displayName": "No Uid", "subtitle": "Buffalo Bills"},
                        {
                            "type": "player",
                            "displayName": "Bad Uid",
                            "subtitle": "Buffalo Bills",
                            "uid": "s:20~l:28~a:notdigits",
                        },
                        {
                            "type": "player",
                            "displayName": "Real Player",
                            "subtitle": "buffalo bills",
                            "uid": "s:20~l:28~a:99",
                        },
                    ]
                },
            ]
        }
        found = espn_extra.parse_athlete_search(payload)
        self.assertEqual(
            found,
            [{"athlete_id": "99", "display_name": "Real Player", "team_name": "buffalo bills"}],
        )


class FetchAthleteSearchTests(unittest.TestCase):
    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_an_empty_or_over_long_query_performs_zero_http_and_zero_redis(self) -> None:
        # T-s5y-08: the cap and the encoding run BEFORE the format string, so a rejected
        # query costs nothing and reaches nothing.
        fake = _FakeRedis()
        bogus: tuple[Any, ...] = ("", "   ", "x" * 41, None, 12483, ["josh allen"])
        for query in bogus:
            with self.subTest(query=query):
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_athlete_search(query)))
        self.assertEqual(fake.gets, [])
        self.assertEqual(fake.sets, [])

    def test_url_metacharacters_cannot_add_or_displace_a_query_parameter(self) -> None:
        # The query is model-influenced text in a URL. ``quote(safe="")`` is the whole
        # guarantee, so this pins it on the characters that would break out.
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, {"results": []}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(espn_extra.fetch_athlete_search("a&type=team#/../x?b"))
        url = _CapturingAsyncClient.last_url
        assert url is not None
        base, _, query_string = url.partition("?")
        self.assertEqual(base, "https://site.web.api.espn.com/apis/search/v2")
        self.assertEqual(query_string, "query=a%26type%3Dteam%23%2F..%2Fx%3Fb&limit=6&type=player")
        self.assertEqual(len(query_string.split("&")), 3)

    def test_cache_miss_issues_one_get_on_the_web_api_host(self) -> None:
        payload = _load_search_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_athlete_search("  Josh   Allen "))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)
        url = _CapturingAsyncClient.last_url
        assert url is not None
        # Whitespace collapses and the query lower-cases, because ESPN's search is
        # case-insensitive (measured) and one cache entry should serve either spelling.
        self.assertIn("query=josh%20allen&", url)
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": espn_extra.DEFAULT_TIMEOUT}
        )

    def test_cache_miss_writes_the_search_key_and_ttl(self) -> None:
        payload = _load_search_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(espn_extra.fetch_athlete_search("Josh Allen"))
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._athlete_search_cache_key("josh%20allen"))
        self.assertEqual(json.loads(value), payload)
        self.assertEqual(ex, espn_extra.ATHLETE_SEARCH_CACHE_TTL_SECONDS)

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_search_fixture()
        fake = _FakeRedis(
            {espn_extra._athlete_search_cache_key("josh%20allen"): json.dumps(payload)}
        )
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(espn_extra.fetch_athlete_search("josh allen"))
        self.assertEqual(out, payload)
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            self.assertIsNone(_run(espn_extra.fetch_athlete_search("josh allen")))
        self.assertEqual(fake.sets, [])


class LeagueSeasonYearTests(unittest.TestCase):
    def test_the_explicit_year_field_is_read_and_the_ref_url_is_not_parsed(self) -> None:
        # The league root is the ONE source of the current season: the stats payload
        # carries none and D-1 forbids a database read. The ``$ref`` beside the year
        # encodes 2025 here ON PURPOSE — reading it would be a guess about ESPN's path
        # shape, and this is the value that must never be guessed.
        self.assertEqual(espn_extra.league_season_year(_league_root()), 2026)

    def test_a_payload_without_a_usable_season_block_returns_none(self) -> None:
        for bogus in (
            None,
            [],
            {},
            {"season": "2026"},
            {"season": {"year": "2026"}},
            {"season": {"year": True}},
            {"season": {"$ref": "http://x/seasons/2026?lang=en"}},
        ):
            with self.subTest(payload=bogus):
                self.assertIsNone(espn_extra.league_season_year(bogus))


class FetchLeagueTests(unittest.TestCase):
    """The league-root fetch: one constant URL through the shared cache-and-fetch shell."""

    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_cache_miss_issues_one_get_on_the_constant_core_api_url(self) -> None:
        payload = _league_root()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_league())
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)
        # No format string, no parameter, no model-influenced segment — the whole URL is
        # the constant, so the request target cannot vary at all.
        self.assertEqual(
            _CapturingAsyncClient.last_url,
            "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl",
        )
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": espn_extra.DEFAULT_TIMEOUT}
        )

    def test_cache_miss_writes_the_league_key_and_the_long_ttl(self) -> None:
        payload = _league_root()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(espn_extra.fetch_league())
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._LEAGUE_CACHE_KEY)
        self.assertEqual(json.loads(value), payload)
        # Hours, not the ten minutes the news and injury feeds take: the year behind this
        # key moves once a season, so a per-question refetch would be pure waste.
        self.assertEqual(ex, espn_extra.LEAGUE_CACHE_TTL_SECONDS)
        self.assertGreaterEqual(espn_extra.LEAGUE_CACHE_TTL_SECONDS, 3600)

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _league_root()
        fake = _FakeRedis({espn_extra._LEAGUE_CACHE_KEY: json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            self.assertEqual(_run(espn_extra.fetch_league()), payload)
        self.assertEqual(fake.sets, [])

    def test_a_non_200_degrades_to_none(self) -> None:
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(503, {}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            self.assertIsNone(_run(espn_extra.fetch_league()))
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            self.assertIsNone(_run(espn_extra.fetch_league()))
        self.assertEqual(fake.sets, [])

    def test_a_redis_outage_fails_open_on_both_the_read_and_the_write(self) -> None:
        payload = _league_root()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            self.assertEqual(_run(espn_extra.fetch_league()), payload)


class GamesPlayedTests(unittest.TestCase):
    def test_games_played_is_found_under_any_of_the_three_spellings(self) -> None:
        for key in ("Games Played", "gamesPlayed", "GP"):
            with self.subTest(key=key):
                self.assertEqual(espn_extra.games_played({"Passing": {key: "13"}}), "13")

    def test_stats_without_games_played_return_none(self) -> None:
        for bogus in (None, [], {}, {"Passing": "nope"}, {"Passing": {"Passing Yards": "462"}}):
            with self.subTest(stats=bogus):
                self.assertIsNone(espn_extra.games_played(bogus))


# --------------------------------------------------------------------------- #
# Team schedule + game leaders (Route D of issue #183).
# --------------------------------------------------------------------------- #


class ParseTeamScheduleTests(unittest.TestCase):
    def test_no_week_selects_the_most_recent_completed_game(self) -> None:
        out = espn_extra.parse_team_schedule(_load_schedule_fixture())
        assert out is not None
        assert out["game"] is not None
        self.assertEqual(out["game"]["event_id"], "401772957")
        self.assertEqual(out["game"]["week"], 18)
        self.assertEqual(out["game"]["name"], "Kansas City Chiefs at Las Vegas Raiders")
        self.assertTrue(out["game"]["completed"])
        self.assertEqual(out["team"], "Kansas City Chiefs")

    def test_the_season_comes_from_requested_season_and_never_from_the_decoy(self) -> None:
        # The fixture's own top-level ``season`` block says 2026 Preseason on a payload
        # that carries 2025's games. ``requestedSeason`` is the authoritative echo.
        fixture = _load_schedule_fixture()
        self.assertEqual(fixture["season"]["year"], 2026)
        out = espn_extra.parse_team_schedule(fixture)
        assert out is not None
        self.assertEqual(out["season"], 2025)

    def test_an_explicit_week_selects_that_same_game(self) -> None:
        out = espn_extra.parse_team_schedule(_load_schedule_fixture(), week=18)
        assert out is not None
        assert out["game"] is not None
        self.assertEqual(out["game"]["event_id"], "401772957")
        self.assertEqual(out["game"]["week"], 18)

    def test_a_preseason_payload_can_never_be_parsed_as_a_regular_season(self) -> None:
        # ``seasontype=2`` is pinned on the URL, so a type-1 reply means the contract this
        # parser reads was not honoured — and preseason week numbers disagree with their
        # own week text, so guessing from one is worse than declining.
        fixture = _load_schedule_fixture()
        fixture["requestedSeason"]["type"] = 1
        self.assertIsNone(espn_extra.parse_team_schedule(fixture))

    def test_the_real_bye_week_returns_no_game_and_reports_the_bye(self) -> None:
        out = espn_extra.parse_team_schedule(_load_schedule_fixture(), week=10)
        assert out is not None
        self.assertIsNone(out["game"])
        self.assertEqual(out["bye_week"], 10)

    def test_a_week_the_payload_does_not_carry_returns_no_game(self) -> None:
        out = espn_extra.parse_team_schedule(_load_schedule_fixture(), week=5)
        assert out is not None
        self.assertIsNone(out["game"])
        self.assertNotEqual(out["bye_week"], 5)

    def test_an_unplayed_week_comes_back_flagged_incomplete_never_dropped(self) -> None:
        # The synthetic week-19 entry. The caller needs the game to name its date, so it
        # must come back rather than be filtered out as "not a real game".
        out = espn_extra.parse_team_schedule(_load_schedule_fixture(), week=19)
        assert out is not None
        assert out["game"] is not None
        self.assertFalse(out["game"]["completed"])
        self.assertEqual(out["game"]["date"], "2026-01-11T21:25Z")

    def test_the_highest_completed_week_wins_even_when_the_events_are_shuffled(self) -> None:
        # Selection is by week NUMBER, not by list position, so a reordered payload
        # cannot pick the wrong game.
        fixture = _load_schedule_fixture()
        fixture["events"] = list(reversed(fixture["events"]))
        out = espn_extra.parse_team_schedule(fixture)
        assert out is not None
        assert out["game"] is not None
        self.assertEqual(out["game"]["week"], 18)

    def test_a_season_with_nothing_finished_reports_any_completed_false(self) -> None:
        # D-7's fallback predicate, which is the whole offseason for the default path.
        fixture = _load_schedule_fixture()
        for event in fixture["events"]:
            event["competitions"][0]["status"]["type"]["completed"] = False
        out = espn_extra.parse_team_schedule(fixture)
        assert out is not None
        self.assertFalse(out["any_completed"])
        self.assertIsNone(out["game"])

    def test_unusable_top_level_shapes_return_none(self) -> None:
        for bogus in (None, [], "nope", {}, {"events": "nope"}, {"events": []}):
            with self.subTest(payload=bogus):
                self.assertIsNone(espn_extra.parse_team_schedule(bogus))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "requestedSeason": {"year": 2025, "type": 2},
            "byeWeek": "ten",
            "team": "not a dict",
            "events": [
                None,
                "nope",
                {},
                {"id": "abc", "week": {"number": 3}},  # non-digit id
                {"id": "1" * 13, "week": {"number": 3}},  # over-long id
                {"id": "401", "week": "nope"},  # unusable week block
                {"id": "402", "week": {"number": True}},  # a bool is not a week number
                {"id": "403", "week": {"number": 4}},  # no competitions at all
                {"id": "404", "week": {"number": 5}, "competitions": "nope"},
                {"id": "405", "week": {"number": 6}, "competitions": []},
                {
                    "id": "406",
                    "week": {"number": 7},
                    "competitions": [{"status": {"type": {"completed": 1}}}],
                },
            ],
        }
        out = espn_extra.parse_team_schedule(payload)
        assert out is not None
        self.assertIsNone(out["bye_week"])  # "ten" is not an int
        self.assertIsNone(out["team"])
        self.assertFalse(out["any_completed"])  # a truthy 1 is not ESPN's own True
        self.assertIsNone(out["game"])
        weeks = [espn_extra.parse_team_schedule(payload, week=n) for n in (4, 5, 6, 7)]
        for week_out in weeks:
            assert week_out is not None
            assert week_out["game"] is not None
            self.assertFalse(week_out["game"]["completed"])
            self.assertIsNone(week_out["game"]["name"])


class ParseGameLeadersTests(unittest.TestCase):
    def test_both_clubs_come_back_keyed_by_full_display_name(self) -> None:
        out = espn_extra.parse_game_leaders(_load_game_leaders_fixture())
        assert out is not None
        leaders = out["leaders"]
        self.assertEqual(sorted(leaders), ["Kansas City Chiefs", "Las Vegas Raiders"])
        kc_passing = next(
            r for r in leaders["Kansas City Chiefs"] if r["category"] == "Passing Yards"
        )
        self.assertEqual(kc_passing["player"], "Shane Buechele")
        self.assertEqual(kc_passing["stat_line"], "7/14, 88 YDS")
        self.assertEqual(kc_passing["position"], "QB")
        lv_rushing = next(
            r for r in leaders["Las Vegas Raiders"] if r["category"] == "Rushing Yards"
        )
        self.assertEqual(lv_rushing["player"], "Ashton Jeanty")
        self.assertEqual(lv_rushing["stat_line"], "26 CAR, 87 YDS")
        self.assertEqual(out["winner"], "Las Vegas Raiders")


class ParseGameLeadersDefensiveTests(unittest.TestCase):
    def test_a_payload_with_no_leaders_key_returns_none(self) -> None:
        # The MEASURED shape of a game that has not been played: 200, 109 KB, and no
        # ``leaders`` key at all. This is the second barrier behind the completed-gate.
        fixture = _load_game_leaders_fixture()
        del fixture["leaders"]
        self.assertIsNone(espn_extra.parse_game_leaders(fixture))

    def test_a_truthy_winner_that_is_not_true_names_no_winner(self) -> None:
        # Identity, not truthiness: guessing a winner is worse than reporting none.
        for bogus in (1, "true", "LV"):
            with self.subTest(winner=bogus):
                fixture = _load_game_leaders_fixture()
                for competitor in fixture["header"]["competitions"][0]["competitors"]:
                    competitor["winner"] = bogus if competitor["winner"] else False
                out = espn_extra.parse_game_leaders(fixture)
                assert out is not None
                self.assertIsNone(out["winner"])

    def test_the_score_is_never_read_onto_any_returned_field(self) -> None:
        # D-2, asserted on the runtime VALUES: the fixture's sentinels are in the payload
        # the parser was handed, and must be nowhere in what it returns.
        out = espn_extra.parse_game_leaders(_load_game_leaders_fixture())
        assert out is not None
        rendered = json.dumps(out)
        self.assertNotIn("9991", rendered)
        self.assertNotIn("9992", rendered)

    def test_unusable_top_level_shapes_return_none(self) -> None:
        for bogus in (None, [], "nope", {}, {"leaders": "nope"}):
            with self.subTest(payload=bogus):
                self.assertIsNone(espn_extra.parse_game_leaders(bogus))

    def test_malformed_entries_are_skipped_without_raising(self) -> None:
        payload = {
            "header": {"competitions": [{"competitors": "nope"}]},
            "leaders": [
                None,
                "nope",
                {},
                {"team": {"displayName": "No Categories"}, "leaders": "nope"},
                {
                    "team": {"displayName": "Kansas City Chiefs"},
                    "leaders": [
                        None,
                        {},  # no label at all
                        {"displayName": "Empty", "leaders": []},
                        {"displayName": "Not A List", "leaders": "nope"},
                        {"displayName": "No Value", "leaders": [{"athlete": {}}]},
                        {"displayName": "No Athlete", "leaders": [{"displayValue": "5"}]},
                        {
                            "name": "sacks",  # falls back to ``name`` when displayName is absent
                            "leaders": [
                                {"displayValue": "2", "athlete": {"displayName": "Chris Jones"}}
                            ],
                        },
                    ],
                },
            ],
        }
        out = espn_extra.parse_game_leaders(payload)
        assert out is not None
        self.assertIsNone(out["winner"])
        # A club whose categories are unusable is OMITTED rather than emitted with an
        # empty list, which would read to the model as "this club led nothing".
        self.assertNotIn("No Categories", out["leaders"])
        rows = out["leaders"]["Kansas City Chiefs"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "sacks")
        self.assertIsNone(rows[0]["position"])  # missing, never invented

    def test_an_empty_header_never_raises(self) -> None:
        for header in (None, "nope", {}, {"competitions": "nope"}, {"competitions": []}):
            with self.subTest(header=header):
                out = espn_extra.parse_game_leaders({"leaders": [], "header": header})
                assert out is not None
                self.assertIsNone(out["winner"])


class FetchTeamScheduleTests(unittest.TestCase):
    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_a_team_outside_the_canonical_32_performs_zero_http_and_zero_redis(self) -> None:
        # T-f0s-01: the allowlist runs BEFORE the URL is formatted. A regression that
        # formats first and validates second fails here, loudly.
        fake = _FakeRedis()
        for bogus in ("", "   ", "zzz", "ZZZ", "../../etc/passwd", "KC/../DAL", "kc;rm -rf /"):
            with self.subTest(team=bogus):
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_team_schedule(bogus)))
        for bogus_any in (None, 42, ["KC"]):
            with self.subTest(team=bogus_any):
                bad: Any = bogus_any
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_team_schedule(bad)))
        self.assertEqual(fake.gets, [])
        self.assertEqual(fake.sets, [])

    def test_an_unusable_season_performs_zero_http_and_zero_redis(self) -> None:
        # T-f0s-02: same discipline, applied to an integer. A bool is rejected explicitly
        # because ``True`` IS an int in Python and would otherwise format as "True".
        fake = _FakeRedis()
        for bogus in ("2025", 2025.0, True, False, 1919, 2101, -2025):
            with self.subTest(season=bogus):
                bad: Any = bogus
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_team_schedule("KC", season=bad)))
        self.assertEqual(fake.gets, [])
        self.assertEqual(fake.sets, [])

    def test_no_season_issues_one_get_pinned_to_the_regular_season(self) -> None:
        payload = _load_schedule_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_team_schedule(" kc "))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)
        url = _CapturingAsyncClient.last_url
        assert url is not None
        self.assertIn("/teams/KC/schedule", url)
        # Without seasontype=2 the endpoint serves the PRESEASON, whose week numbers
        # disagree with their own week text.
        self.assertIn("seasontype=2", url)
        self.assertNotIn("season=", url.replace("seasontype=", ""))
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": espn_extra.DEFAULT_TIMEOUT}
        )
        self.assertEqual(len(fake.sets), 1)
        key, _value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._schedule_cache_key("KC", None))
        self.assertEqual(ex, espn_extra.SCHEDULE_CACHE_TTL_SECONDS)

    def test_an_explicit_season_carries_both_parameters_and_its_own_key(self) -> None:
        payload = _load_schedule_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_team_schedule("KC", season=2024))
        self.assertEqual(out, payload)
        url = _CapturingAsyncClient.last_url
        assert url is not None
        self.assertIn("season=2024", url)
        self.assertIn("seasontype=2", url)
        key, _value, _ex = fake.sets[0]
        self.assertEqual(key, espn_extra._schedule_cache_key("KC", 2024))
        # A season-less request must NOT be served a specific season's entry.
        self.assertNotEqual(key, espn_extra._schedule_cache_key("KC", None))

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_schedule_fixture()
        fake = _FakeRedis({espn_extra._schedule_cache_key("KC", 2025): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(espn_extra.fetch_team_schedule("KC", season=2025))
        self.assertEqual(out, payload)
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            self.assertIsNone(_run(espn_extra.fetch_team_schedule("KC")))
        self.assertEqual(fake.sets, [])


class FetchGameSummaryTests(unittest.TestCase):
    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient._response = response

    def test_a_non_digit_event_id_performs_zero_http_and_zero_redis(self) -> None:
        # T-f0s-03: the guard runs BEFORE the format string, exactly as the athlete id's
        # does, even though the event id is read out of a schedule payload.
        fake = _FakeRedis()
        for bogus in ("", "   ", "abc", "401/../99", "401;rm -rf /", "1" * 13, True, False, None):
            with self.subTest(event_id=bogus):
                bad: Any = bogus
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_game_summary(bad)))
        self.assertEqual(fake.gets, [])
        self.assertEqual(fake.sets, [])

    def test_the_injuries_path_and_the_game_path_share_one_cache_entry(self) -> None:
        # D-8, the entire premise of Route D: whichever name asks first warms the other,
        # so the 635 KB summary is fetched ONCE and served twice.
        payload = _load_game_leaders_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            first = _run(espn_extra.fetch_injuries(401772957))
        self.assertEqual(first, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)
        self.assertEqual(fake.sets[0][0], espn_extra._cache_key(401772957))

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            second = _run(espn_extra.fetch_game_summary("401772957"))
        self.assertEqual(second, payload)  # served from the SAME entry, zero HTTP

    def test_the_key_is_identical_whether_the_id_arrives_as_an_int_or_a_string(self) -> None:
        self.assertEqual(espn_extra._cache_key(777), espn_extra._cache_key("777"))

    def test_a_digit_string_event_id_reaches_the_summary_url(self) -> None:
        payload = _load_game_leaders_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_game_summary(" 401772957 "))
        self.assertEqual(out, payload)
        url = _CapturingAsyncClient.last_url
        assert url is not None
        self.assertIn("event=401772957", url)
        _key, _value, ex = fake.sets[0]
        self.assertEqual(ex, espn_extra.INJURIES_CACHE_TTL_SECONDS)


class LeagueSeasonPhaseTests(unittest.TestCase):
    def test_the_phase_is_read_from_the_season_type_and_lower_cased(self) -> None:
        # Lower-cased because a capitalised token handed to the model comes back out of
        # its mouth capitalised, mid-sentence (memory: qa-phrasing-inversion).
        self.assertEqual(espn_extra.league_season_phase(_league_root()), "preseason")

    def test_a_payload_without_a_usable_type_name_returns_none(self) -> None:
        for bogus in (
            None,
            [],
            {},
            {"season": "2026"},
            {"season": {"year": 2026}},
            {"season": {"type": "Preseason"}},
            {"season": {"type": {"id": "1"}}},
            {"season": {"type": {"name": "   "}}},
        ):
            with self.subTest(payload=bogus):
                self.assertIsNone(espn_extra.league_season_phase(bogus))


class PostseasonRoundWeekTests(unittest.TestCase):
    """The round resolver: the ONLY producer of a week number that reaches the URL."""

    def test_every_spelling_of_the_four_rounds_resolves(self) -> None:
        for spelling, week in (
            ("super bowl", 5),
            ("Super Bowl", 5),
            ("the Super Bowl LX", 5),
            ("superbowl", 5),
            ("wild card", 1),
            ("wild-card round", 1),
            ("AFC Wild Card Playoffs", 1),
            ("divisional", 2),
            ("divisional round", 2),
            ("NFC Divisional Playoffs", 2),
            ("conference championships", 3),
            ("AFC Championship", 3),
            ("championship game", 3),
        ):
            with self.subTest(round=spelling):
                self.assertEqual(espn_extra.postseason_round_week(spelling), week)

    def test_no_round_name_can_ever_select_the_pro_bowl_week(self) -> None:
        # THE trap of this endpoint: postseason week 4 is the Pro Bowl, an exhibition
        # game, and reporting it as a playoff result is the failure worth pinning.
        for spelling in ("pro bowl", "Pro Bowl", "the Pro-Bowl Games", "probowl", "PRO BOWL"):
            with self.subTest(round=spelling):
                self.assertIsNone(espn_extra.postseason_round_week(spelling))
                self.assertTrue(espn_extra.asked_for_the_pro_bowl(spelling))
        self.assertNotIn(espn_extra.PRO_BOWL_WEEK, espn_extra.POSTSEASON_WEEKS)
        self.assertNotIn(espn_extra.PRO_BOWL_WEEK, espn_extra.POSTSEASON_ROUND_LABELS)

    def test_an_unusable_or_unknown_round_resolves_to_nothing(self) -> None:
        for bogus in (None, 5, [], "", "   ", "the quarterfinals", "week 4", "regular season"):
            bad: Any = bogus
            with self.subTest(round=bogus):
                self.assertIsNone(espn_extra.postseason_round_week(bad))
                self.assertFalse(espn_extra.asked_for_the_pro_bowl(bad))

    def test_the_resolver_can_only_ever_produce_a_labelled_playoff_week(self) -> None:
        # Whatever the model writes, the value that reaches the URL is one of four
        # literals out of this module's own table.
        for spelling in ("super bowl", "wild card", "divisional", "conference championships"):
            week = espn_extra.postseason_round_week(spelling)
            with self.subTest(round=spelling):
                self.assertIn(week, espn_extra.POSTSEASON_WEEKS)
                self.assertIn(week, espn_extra.POSTSEASON_ROUND_LABELS)


class ParsePostseasonRoundTests(unittest.TestCase):
    def test_the_super_bowl_resolves_to_its_winner_and_its_espn_headline(self) -> None:
        facts = espn_extra.parse_postseason_round(_load_super_bowl_fixture())
        assert facts is not None
        self.assertEqual(facts["season"], 2025)
        self.assertEqual(facts["week"], 5)
        self.assertTrue(facts["any_completed"])
        self.assertEqual(len(facts["games"]), 1)
        game = facts["games"][0]
        self.assertEqual(game["game"], "Super Bowl LX")
        self.assertEqual(game["winner"], "Seattle Seahawks")
        self.assertEqual(sorted(game["teams"]), ["New England Patriots", "Seattle Seahawks"])
        self.assertEqual(game["date"], "2026-02-08T23:30Z")
        self.assertTrue(game["completed"])

    def test_neither_score_is_read_on_any_path(self) -> None:
        # D-2, on the RUNTIME value: OPEN_OWNERSHIP_CLAUSE forbids the model stating a
        # score, and a field it can see is a field it may voice. The fixture's two scores
        # are the sentinels 9991 and 9992 precisely so this is an assertion.
        rendered = json.dumps(espn_extra.parse_postseason_round(_load_super_bowl_fixture()))
        self.assertNotIn("9991", rendered)
        self.assertNotIn("9992", rendered)

    def test_a_round_nobody_has_played_carries_no_winner_and_no_completion(self) -> None:
        facts = espn_extra.parse_postseason_round(_unplayed_super_bowl())
        assert facts is not None
        self.assertFalse(facts["any_completed"])
        self.assertIsNone(facts["games"][0]["winner"])
        self.assertFalse(facts["games"][0]["completed"])

    def test_both_conference_championships_come_back_under_their_own_headlines(self) -> None:
        facts = espn_extra.parse_postseason_round(_conference_championships())
        assert facts is not None
        self.assertEqual(
            [(game["game"], game["winner"]) for game in facts["games"]],
            [
                ("AFC Championship", "New England Patriots"),
                ("NFC Championship", "Seattle Seahawks"),
            ],
        )

    def test_a_payload_that_is_not_the_postseason_is_refused(self) -> None:
        # The season echo is the discriminator: a regular-season or preseason payload is
        # not the thing this parser contracts to read, so it degrades rather than
        # relaying weeks whose numbers mean something else entirely.
        for season in (None, {}, {"year": 2025}, {"type": 2, "year": 2025}, {"type": "3"}):
            payload = {**_load_super_bowl_fixture(), "season": season}
            with self.subTest(season=season):
                self.assertIsNone(espn_extra.parse_postseason_round(payload))

    def test_a_malformed_payload_never_raises(self) -> None:
        for bogus in (
            None,
            [],
            "nope",
            {},
            {"events": "nope"},
            {"season": {"type": 3}, "events": []},
            {"season": {"type": 3, "year": "2025"}, "week": "5", "events": [None, 7, {}]},
            {"season": {"type": 3}, "events": [{"competitions": "nope"}]},
            {"season": {"type": 3}, "events": [{"competitions": [{"competitors": [None]}]}]},
        ):
            with self.subTest(payload=bogus):
                facts = espn_extra.parse_postseason_round(bogus)
                self.assertTrue(facts is None or isinstance(facts, dict))

    def test_a_game_naming_no_club_at_all_is_dropped_never_half_emitted(self) -> None:
        payload = _load_super_bowl_fixture()
        for competitor in payload["events"][0]["competitions"][0]["competitors"]:
            competitor["team"] = {}
        facts = espn_extra.parse_postseason_round(payload)
        assert facts is not None
        self.assertEqual(facts["games"], [])
        self.assertFalse(facts["any_completed"])

    def test_a_winner_flag_that_is_not_the_boolean_true_names_nobody(self) -> None:
        # Identity, not truthiness: guessing a winner is worse than reporting none.
        payload = _load_super_bowl_fixture()
        for competitor in payload["events"][0]["competitions"][0]["competitors"]:
            competitor["winner"] = "true"
        facts = espn_extra.parse_postseason_round(payload)
        assert facts is not None
        self.assertIsNone(facts["games"][0]["winner"])


class FetchPostseasonScoreboardTests(unittest.TestCase):
    def _arm_client(self, response: object) -> None:
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient.last_headers = _NEVER_CALLED
        _CapturingAsyncClient.last_init_kwargs = None
        _CapturingAsyncClient._response = response

    def test_an_unusable_season_performs_zero_http_and_zero_redis(self) -> None:
        # The same discipline the schedule holds, against the SAME bounds: the season is
        # the one model-influenced value here, and a guard applied after the format string
        # is not a guard.
        fake = _FakeRedis()
        for bogus in ("2025", 2025.0, True, False, 1919, 2101, -2025, None):
            with self.subTest(season=bogus):
                bad: Any = bogus
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_postseason_scoreboard(bad, 5)))
        self.assertEqual(fake.gets, [])
        self.assertEqual(fake.sets, [])

    def test_the_pro_bowl_week_can_never_be_requested(self) -> None:
        # Week 4 is the Pro Bowl. It is not in POSTSEASON_ROUND_LABELS, so no round name
        # produces it, and this guard is the second barrier: even a direct call for it
        # performs no HTTP at all.
        fake = _FakeRedis()
        for bogus in (espn_extra.PRO_BOWL_WEEK, 0, 6, 18, -1, "5", 5.0, True, None):
            with self.subTest(week=bogus):
                bad: Any = bogus
                with (
                    _redis_returns(fake),
                    mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient),
                ):
                    self.assertIsNone(_run(espn_extra.fetch_postseason_scoreboard(2025, bad)))
        self.assertEqual(fake.gets, [])
        self.assertEqual(fake.sets, [])

    def test_cache_miss_issues_one_get_pinned_to_the_postseason(self) -> None:
        payload = _load_super_bowl_fixture()
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(espn_extra.fetch_postseason_scoreboard(2025, 5))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)
        url = _CapturingAsyncClient.last_url
        assert url is not None
        self.assertIn("dates=2025", url)
        # Without seasontype=3 this endpoint serves the regular season, whose week 5 is a
        # different set of games entirely.
        self.assertIn("seasontype=3", url)
        self.assertIn("week=5", url)
        self.assertIsNone(_CapturingAsyncClient.last_headers)
        self.assertEqual(
            _CapturingAsyncClient.last_init_kwargs, {"timeout": espn_extra.DEFAULT_TIMEOUT}
        )
        key, _value, ex = fake.sets[0]
        self.assertEqual(key, espn_extra._postseason_cache_key(2025, 5))
        self.assertEqual(ex, espn_extra.POSTSEASON_CACHE_TTL_SECONDS)

    def test_each_season_and_round_keeps_its_own_cache_entry(self) -> None:
        keys = {
            espn_extra._postseason_cache_key(season, week)
            for season in (2024, 2025)
            for week in espn_extra.POSTSEASON_WEEKS
        }
        self.assertEqual(len(keys), 2 * len(espn_extra.POSTSEASON_WEEKS))

    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_super_bowl_fixture()
        fake = _FakeRedis({espn_extra._postseason_cache_key(2025, 5): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            self.assertEqual(_run(espn_extra.fetch_postseason_scoreboard(2025, 5)), payload)
        self.assertEqual(fake.sets, [])

    def test_a_non_200_degrades_to_none(self) -> None:
        fake = _FakeRedis()
        self._arm_client(_FakeResponse(503, {}))
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            self.assertIsNone(_run(espn_extra.fetch_postseason_scoreboard(2025, 5)))
        self.assertEqual(fake.sets, [])

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            self.assertIsNone(_run(espn_extra.fetch_postseason_scoreboard(2025, 5)))
        self.assertEqual(fake.sets, [])

    def test_a_redis_outage_fails_open_on_both_the_read_and_the_write(self) -> None:
        payload = _load_super_bowl_fixture()
        self._arm_client(_FakeResponse(200, payload))
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            self.assertEqual(_run(espn_extra.fetch_postseason_scoreboard(2025, 5)), payload)


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
