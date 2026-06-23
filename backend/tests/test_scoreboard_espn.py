"""Offline unit tests for the ESPN scoreboard adapter's PURE normalization.

These exercise only the pure normalize functions in :mod:`app.scoreboard.espn`
against small, realistic INLINE site-scoreboard payloads (modeled on
``.planning/notes/espn-ingestion-strategy.md`` "Odds object shape" and the
shapes the fixture generator parses). They make **NO network calls** — the
default ``python -m unittest`` run stays fully offline.

One OPTIONAL live ESPN smoke test is provided but is SKIPPED unless the
``RUN_ESPN_LIVE`` environment variable is set (it performs a real outbound GET
and is therefore excluded from the default offline suite). Enable it explicitly::

    RUN_ESPN_LIVE=1 .venv/bin/python -m unittest tests.test_scoreboard_espn -v

Run the default (offline) suite from ``backend/``::

    cd backend && python -m unittest tests.test_scoreboard_espn -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import os
import unittest

from app.models import GameStatus
from app.scoreboard.espn import (
    EspnScoreboardSource,
    SITE_SCOREBOARD_URL,
    normalize_odds,
    normalize_scoreboard,
    select_odds_item,
)
from app.scoreboard.port import ScoreboardSource


def _scheduled_event_with_dk_odds() -> dict:
    """An upcoming game carrying inline DraftKings odds (and a second book)."""
    return {
        "id": "401547001",
        "date": "2026-09-10T00:20Z",
        "competitions": [
            {
                "date": "2026-09-10T00:20Z",
                "status": {"type": {"state": "pre", "name": "STATUS_SCHEDULED"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"id": "26", "abbreviation": "SEA"},
                        "score": "",
                    },
                    {
                        "homeAway": "away",
                        "team": {"id": "17", "abbreviation": "NE"},
                        "score": "",
                    },
                ],
                "odds": [
                    {
                        "provider": {"id": "999", "name": "SomeOtherBook", "priority": 2},
                        "spread": -2.5,
                        "overUnder": 43.0,
                        "homeTeamOdds": {
                            "favorite": True,
                            "underdog": False,
                            "team": {"id": "26"},
                        },
                        "awayTeamOdds": {
                            "favorite": False,
                            "underdog": True,
                            "team": {"id": "17"},
                        },
                    },
                    {
                        "provider": {"id": "100", "name": "DraftKings", "priority": 1},
                        "spread": -3.5,
                        "overUnder": 44.5,
                        "homeTeamOdds": {
                            "favorite": True,
                            "underdog": False,
                            "team": {"id": "26", "abbreviation": "SEA"},
                        },
                        "awayTeamOdds": {
                            "favorite": False,
                            "underdog": True,
                            "team": {"id": "17", "abbreviation": "NE"},
                        },
                    },
                ],
            }
        ],
    }


def _final_event_no_odds() -> dict:
    """A completed game with scores and no odds (odds disappear once final)."""
    return {
        "id": "401772510",
        "date": "2025-09-05T00:20Z",
        "competitions": [
            {
                "date": "2025-09-05T00:20Z",
                "status": {"type": {"state": "post", "name": "STATUS_FINAL"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"id": "21", "abbreviation": "PHI"},
                        "score": "24",
                    },
                    {
                        "homeAway": "away",
                        "team": {"id": "6", "abbreviation": "DAL"},
                        "score": "20",
                    },
                ],
                # no "odds" key at all
            }
        ],
    }


def _in_progress_event_by_state() -> dict:
    """A live game whose status NAME is absent but state == 'in'."""
    return {
        "id": "401547999",
        "date": "2026-09-10T17:00Z",
        "competitions": [
            {
                "date": "2026-09-10T17:00Z",
                "status": {"type": {"state": "in"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"id": "1", "abbreviation": "AAA"},
                        "score": "10",
                    },
                    {
                        "homeAway": "away",
                        "team": {"id": "2", "abbreviation": "BBB"},
                        "score": "7",
                    },
                ],
            }
        ],
    }


def _payload(*events: dict) -> dict:
    return {"events": list(events)}


class NormalizeScoreboardTest(unittest.TestCase):
    def test_scheduled_game_with_dk_odds(self) -> None:
        games = normalize_scoreboard(
            _payload(_scheduled_event_with_dk_odds()), season=2026, week=1
        )
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(game.espn_event_id, "401547001")
        self.assertEqual(game.season, 2026)
        self.assertEqual(game.week, 1)
        self.assertEqual(game.status, GameStatus.SCHEDULED)
        # scores withheld until final
        self.assertIsNone(game.home.score)
        self.assertIsNone(game.away.score)
        self.assertEqual(game.home.espn_team_id, "26")
        self.assertEqual(game.away.espn_team_id, "17")
        # kickoff parsed tz-aware
        self.assertIsNotNone(game.kickoff_at)
        self.assertIsNotNone(game.kickoff_at.tzinfo)
        # odds populated, DraftKings preferred over the other book
        self.assertIsNotNone(game.odds)
        self.assertEqual(game.odds.provider, "DraftKings")
        self.assertEqual(game.odds.spread, -3.5)
        self.assertEqual(game.odds.total, 44.5)
        self.assertEqual(game.odds.favorite_team_id, "26")
        self.assertEqual(game.odds.underdog_team_id, "17")

    def test_final_game_with_scores_no_odds(self) -> None:
        games = normalize_scoreboard(
            _payload(_final_event_no_odds()), season=2025, week=1
        )
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(game.status, GameStatus.FINAL)
        self.assertEqual(game.home.score, 24)
        self.assertEqual(game.away.score, 20)
        self.assertEqual(game.home.espn_team_id, "21")
        self.assertEqual(game.away.espn_team_id, "6")
        self.assertIsNone(game.odds)

    def test_status_mapping_by_state_in(self) -> None:
        games = normalize_scoreboard(
            _payload(_in_progress_event_by_state()), season=2026, week=1
        )
        self.assertEqual(games[0].status, GameStatus.IN_PROGRESS)

    def test_status_mapping_by_name_in_progress(self) -> None:
        event = _in_progress_event_by_state()
        event["competitions"][0]["status"]["type"] = {
            "state": "post",
            "name": "STATUS_IN_PROGRESS",
        }
        games = normalize_scoreboard(_payload(event), season=2026, week=1)
        # Name takes precedence: STATUS_IN_PROGRESS maps to IN_PROGRESS even
        # though the (contrived) state says 'post'.
        self.assertEqual(games[0].status, GameStatus.IN_PROGRESS)

    def test_multiple_events_and_empty_payload(self) -> None:
        games = normalize_scoreboard(
            _payload(_final_event_no_odds(), _scheduled_event_with_dk_odds()),
            season=2025,
            week=1,
        )
        self.assertEqual(len(games), 2)
        self.assertEqual(normalize_scoreboard({}, season=2025, week=1), [])
        self.assertEqual(
            normalize_scoreboard({"events": "garbage"}, season=2025, week=1), []
        )


class ProviderSelectionTest(unittest.TestCase):
    def test_prefers_draftkings_over_other_book(self) -> None:
        items = _scheduled_event_with_dk_odds()["competitions"][0]["odds"]
        chosen = select_odds_item(items)
        self.assertEqual(chosen["provider"]["name"], "DraftKings")

    def test_falls_back_to_priority_one(self) -> None:
        items = [
            {"provider": {"id": "5", "name": "BookA", "priority": 3}, "spread": -1.0},
            {"provider": {"id": "6", "name": "BookB", "priority": 1}, "spread": -2.0},
        ]
        chosen = select_odds_item(items)
        self.assertEqual(chosen["provider"]["name"], "BookB")

    def test_falls_back_to_first_present(self) -> None:
        items = [
            {"provider": {"id": "5", "name": "BookA", "priority": 7}, "spread": -1.0},
            {"provider": {"id": "6", "name": "BookB", "priority": 9}, "spread": -2.0},
        ]
        chosen = select_odds_item(items)
        self.assertEqual(chosen["provider"]["name"], "BookA")

    def test_no_items_returns_none(self) -> None:
        self.assertIsNone(select_odds_item(None))
        self.assertIsNone(select_odds_item([]))


class NormalizeOddsDefensiveTest(unittest.TestCase):
    def test_missing_fields_degrade_to_none_without_raising(self) -> None:
        odds = normalize_odds({"provider": {"name": "DraftKings"}})
        self.assertIsNotNone(odds)
        self.assertEqual(odds.provider, "DraftKings")
        self.assertIsNone(odds.spread)
        self.assertIsNone(odds.total)
        self.assertIsNone(odds.favorite_team_id)
        self.assertIsNone(odds.underdog_team_id)

    def test_non_dict_returns_none(self) -> None:
        self.assertIsNone(normalize_odds(None))
        self.assertIsNone(normalize_odds("garbage"))

    def test_raw_signed_spread_preserved(self) -> None:
        odds = normalize_odds(
            {"provider": {"name": "DraftKings"}, "spread": -7.5, "overUnder": 47.5}
        )
        # raw home-relative signed value, not abs()
        self.assertEqual(odds.spread, -7.5)
        self.assertEqual(odds.total, 47.5)


class AdapterShapeTest(unittest.TestCase):
    def test_adapter_satisfies_port(self) -> None:
        self.assertIsInstance(EspnScoreboardSource(), ScoreboardSource)

    def test_url_template_is_per_week_regular_season(self) -> None:
        url = SITE_SCOREBOARD_URL.format(season=2026, week=3)
        self.assertIn("dates=2026", url)
        self.assertIn("seasontype=2", url)
        self.assertIn("week=3", url)


@unittest.skipUnless(
    os.environ.get("RUN_ESPN_LIVE"),
    "live ESPN smoke test (set RUN_ESPN_LIVE=1 to enable; performs a real network GET)",
)
class EspnLiveSmokeTest(unittest.TestCase):
    """OPTIONAL, network-gated smoke test — skipped by default (offline suite)."""

    def test_fetch_week_returns_games(self) -> None:
        source = EspnScoreboardSource()
        games = source.fetch_week(2024, 1)
        self.assertGreater(len(games), 0)
        for game in games:
            self.assertIsNotNone(game.espn_event_id)


if __name__ == "__main__":
    unittest.main()
