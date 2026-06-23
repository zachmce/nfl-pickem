"""Offline unit tests for the demo scoreboard adapter and pure derive_status.

These exercise :mod:`app.scoreboard.demo`. The pure :func:`derive_status`
boundary tests inject a FIXED ``now`` (no clock read). The
:class:`~app.scoreboard.demo.Demo2025Source` tests read the REAL packaged 2025
fixture via ``fixture_2025.FIXTURE_PATH`` (exactly like
``test_import_fixture_2025``) and rely on a large offset so the outcome is
deterministic against the real clock — there is NO network and NO database.

Run from the ``backend/`` directory::

    cd backend && python -m unittest tests.test_scoreboard_demo -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.models import GameStatus
from app.scoreboard.demo import (
    GAME_DURATION,
    Demo2025Source,
    derive_status,
)
from app.scoreboard.espn import EspnScoreboardSource
from app.scoreboard.port import ScoreboardSource

KICKOFF = datetime(2025, 9, 5, 0, 20, tzinfo=timezone.utc)
DURATION = timedelta(hours=3, minutes=30)


class DeriveStatusTest(unittest.TestCase):
    def test_before_kickoff_is_scheduled_no_reveal(self) -> None:
        now = KICKOFF - timedelta(seconds=1)
        status, reveal = derive_status(KICKOFF, now, duration=DURATION)
        self.assertEqual(status, GameStatus.SCHEDULED)
        self.assertFalse(reveal)

    def test_exactly_at_kickoff_is_in_progress_no_reveal(self) -> None:
        status, reveal = derive_status(KICKOFF, KICKOFF, duration=DURATION)
        self.assertEqual(status, GameStatus.IN_PROGRESS)
        self.assertFalse(reveal)

    def test_mid_window_is_in_progress_no_reveal(self) -> None:
        now = KICKOFF + timedelta(hours=1)
        status, reveal = derive_status(KICKOFF, now, duration=DURATION)
        self.assertEqual(status, GameStatus.IN_PROGRESS)
        self.assertFalse(reveal)

    def test_exactly_at_end_is_final_reveal(self) -> None:
        now = KICKOFF + DURATION
        status, reveal = derive_status(KICKOFF, now, duration=DURATION)
        self.assertEqual(status, GameStatus.FINAL)
        self.assertTrue(reveal)

    def test_after_end_is_final_reveal(self) -> None:
        now = KICKOFF + DURATION + timedelta(hours=2)
        status, reveal = derive_status(KICKOFF, now, duration=DURATION)
        self.assertEqual(status, GameStatus.FINAL)
        self.assertTrue(reveal)

    def test_naive_now_raises(self) -> None:
        naive = datetime(2025, 9, 5, 0, 20)
        with self.assertRaises(ValueError):
            derive_status(KICKOFF, naive, duration=DURATION)

    def test_naive_kickoff_raises(self) -> None:
        naive_kickoff = datetime(2025, 9, 5, 0, 20)
        with self.assertRaises(ValueError):
            derive_status(naive_kickoff, KICKOFF, duration=DURATION)

    def test_default_duration_is_game_duration(self) -> None:
        # exercise the default-arg path without passing duration explicitly
        status, _ = derive_status(KICKOFF, KICKOFF + GAME_DURATION)
        self.assertEqual(status, GameStatus.FINAL)


class Demo2025SourceTest(unittest.TestCase):
    def test_large_negative_offset_finals_week1(self) -> None:
        # Push 2025 ~10 years into the past so every game is well past FINAL now.
        source = Demo2025Source(offset=timedelta(days=-3650))
        games = source.fetch_week(2025, 1)
        self.assertEqual(len(games), 16)
        for game in games:
            self.assertEqual(game.status, GameStatus.FINAL)
            # scores revealed when FINAL
            self.assertIsNotNone(game.home.score)
            self.assertIsNotNone(game.away.score)
            # identity preserved
            self.assertIsNotNone(game.espn_event_id)
            self.assertEqual(game.season, 2025)
            self.assertEqual(game.week, 1)

    def test_revealed_scores_match_fixture(self) -> None:
        source = Demo2025Source(offset=timedelta(days=-3650))
        games = {g.espn_event_id: g for g in source.fetch_week(2025, 1)}
        # The PHI/DAL opener: PHI 24, DAL 20 (home/away from the fixture).
        opener = games["401772510"]
        self.assertEqual(opener.home.espn_team_id, "21")
        self.assertEqual(opener.home.score, 24)
        self.assertEqual(opener.away.espn_team_id, "6")
        self.assertEqual(opener.away.score, 20)
        # odds present on this game, carried raw (signed home-relative)
        self.assertIsNotNone(opener.odds)
        self.assertEqual(opener.odds.provider, "ESPN BET")
        self.assertEqual(opener.odds.spread, -7.5)
        self.assertEqual(opener.odds.favorite_team_id, "21")

    def test_kickoff_at_is_positioned_by_offset(self) -> None:
        offset = timedelta(days=-3650)
        source = Demo2025Source(offset=offset)
        games = {g.espn_event_id: g for g in source.fetch_week(2025, 1)}
        opener = games["401772510"]
        expected = datetime(2025, 9, 5, 0, 20, tzinfo=timezone.utc) + offset
        self.assertEqual(opener.kickoff_at, expected)

    def test_large_positive_offset_scheduled_scores_withheld(self) -> None:
        # Push 2025 ~10 years into the future so every game is SCHEDULED now.
        source = Demo2025Source(offset=timedelta(days=3650))
        games = source.fetch_week(2025, 1)
        self.assertEqual(len(games), 16)
        for game in games:
            self.assertEqual(game.status, GameStatus.SCHEDULED)
            self.assertIsNone(game.home.score)
            self.assertIsNone(game.away.score)

    def test_non_matching_season_returns_empty(self) -> None:
        source = Demo2025Source(offset=timedelta(days=-3650))
        self.assertEqual(source.fetch_week(2099, 1), [])


class PortConformanceTest(unittest.TestCase):
    def test_both_adapters_satisfy_port(self) -> None:
        self.assertIsInstance(Demo2025Source(), ScoreboardSource)
        self.assertIsInstance(EspnScoreboardSource(), ScoreboardSource)


if __name__ == "__main__":
    unittest.main()
