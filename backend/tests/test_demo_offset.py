"""Offline boundary tests for the demo offset-positioning helper.

These exercise :mod:`app.demo.offset` against the real packaged 2025 fixture,
fully offline — no DB, no network, no ``app.db`` import, and
:class:`~app.scoreboard.espn.EspnScoreboardSource` is NEVER instantiated. The
only source used is :class:`~app.scoreboard.demo.Demo2025Source`, positioned by
the computed offset, with status derived against the REAL clock.

What is proven, for weeks 1, 2, 3:

* ``DemoPhase.WINDOW_OPEN_FOR_WEEK`` positions so that target week N derives
  SCHEDULED (its earliest game is in the future), the previous week N-1 (if any)
  is fully FINAL, and ``compute_window`` + ``is_pick_open`` report the window OPEN
  at the real now.
* ``DemoPhase.ALL_WEEK_FINAL`` positions so every game of target week N derives
  FINAL against the real now.
* Naive ``now`` or an empty/missing week raises a deliberate ``ValueError``
  (mirroring ``pick_window._require_aware``).

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_demo_offset -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.demo.offset import (
    DemoPhase,
    DEFAULT_MARGIN,
    compute_offset,
    load_fixture_kickoffs,
)
from app.models import GameStatus
from app.scoreboard.demo import Demo2025Source, derive_status
from app.services.pick_window import compute_window, is_pick_open

SEASON = 2025


class LoadKickoffsTests(unittest.TestCase):
    """The fixture-kickoff loader returns tz-aware kickoffs per week."""

    def test_loads_weeks_1_to_3_tz_aware(self) -> None:
        weeks = load_fixture_kickoffs()
        for wk in (1, 2, 3):
            self.assertIn(wk, weeks)
            self.assertEqual(len(weeks[wk]), 16)
            for ko in weeks[wk]:
                self.assertIsNotNone(ko.tzinfo)
                self.assertIsNotNone(ko.utcoffset())


class ComputeOffsetGuardTests(unittest.TestCase):
    """Deliberate ValueErrors for naive/empty inputs."""

    def setUp(self) -> None:
        self.weeks = load_fixture_kickoffs()

    def test_naive_now_raises(self) -> None:
        naive = datetime(2026, 6, 23, 12, 0, 0)  # no tzinfo
        with self.assertRaises(ValueError):
            compute_offset(
                naive,
                target_week=1,
                phase=DemoPhase.WINDOW_OPEN_FOR_WEEK,
                weeks_kickoffs=self.weeks,
            )

    def test_empty_target_week_raises(self) -> None:
        now = datetime.now(timezone.utc)
        with self.assertRaises(ValueError):
            compute_offset(
                now,
                target_week=99,  # not in the fixture
                phase=DemoPhase.ALL_WEEK_FINAL,
                weeks_kickoffs=self.weeks,
            )

    def test_naive_kickoff_in_week_raises(self) -> None:
        now = datetime.now(timezone.utc)
        bad = dict(self.weeks)
        bad[1] = [datetime(2025, 9, 5, 0, 20, 0)]  # naive
        with self.assertRaises(ValueError):
            compute_offset(
                now,
                target_week=1,
                phase=DemoPhase.WINDOW_OPEN_FOR_WEEK,
                weeks_kickoffs=bad,
            )


class WindowOpenPositioningTests(unittest.TestCase):
    """WINDOW_OPEN positions target week SCHEDULED, predecessor FINAL, open."""

    def setUp(self) -> None:
        self.weeks = load_fixture_kickoffs()

    def _assert_window_open(self, target_week: int) -> None:
        now = datetime.now(timezone.utc)
        offset = compute_offset(
            now,
            target_week=target_week,
            phase=DemoPhase.WINDOW_OPEN_FOR_WEEK,
            weeks_kickoffs=self.weeks,
        )
        source = Demo2025Source(offset=offset)

        # Re-fetch now: the source derives against its own real now() inside
        # fetch_week, so use a fresh now for the expectations too.
        target_games = source.fetch_week(SEASON, target_week)
        self.assertTrue(target_games)

        # Every target-week game is SCHEDULED (its earliest kickoff is future).
        for g in target_games:
            status, _ = derive_status(g.kickoff_at, datetime.now(timezone.utc))
            self.assertEqual(
                status,
                GameStatus.SCHEDULED,
                f"week {target_week} game {g.espn_event_id} not SCHEDULED",
            )

        # Predecessor (if any) is fully FINAL.
        if target_week > 1:
            prev_games = source.fetch_week(SEASON, target_week - 1)
            self.assertTrue(prev_games)
            for g in prev_games:
                status, _ = derive_status(
                    g.kickoff_at, datetime.now(timezone.utc)
                )
                self.assertEqual(
                    status,
                    GameStatus.FINAL,
                    f"prev week {target_week - 1} game {g.espn_event_id} "
                    "not FINAL",
                )

        # compute_window + is_pick_open over the positioned weeks report OPEN.
        from app.models import Game

        def _to_games(scoreboard_games):
            return [
                Game(
                    espn_event_id=int(sg.espn_event_id),
                    week_id=0,
                    season=sg.season,
                    week=sg.week,
                    home_team_id=0,
                    away_team_id=0,
                    kickoff_at=sg.kickoff_at,
                    status=sg.status,
                )
                for sg in scoreboard_games
            ]

        prev = (
            _to_games(source.fetch_week(SEASON, target_week - 1))
            if target_week > 1
            else None
        )
        window = compute_window(_to_games(target_games), prev)
        self.assertTrue(
            is_pick_open(window, datetime.now(timezone.utc)),
            f"window for week {target_week} should be open",
        )

    def test_week_1_open(self) -> None:
        self._assert_window_open(1)

    def test_week_2_open(self) -> None:
        self._assert_window_open(2)

    def test_week_3_open(self) -> None:
        self._assert_window_open(3)


class AllFinalPositioningTests(unittest.TestCase):
    """ALL_WEEK_FINAL positions every target-week game FINAL."""

    def setUp(self) -> None:
        self.weeks = load_fixture_kickoffs()

    def _assert_all_final(self, target_week: int) -> None:
        now = datetime.now(timezone.utc)
        offset = compute_offset(
            now,
            target_week=target_week,
            phase=DemoPhase.ALL_WEEK_FINAL,
            weeks_kickoffs=self.weeks,
        )
        source = Demo2025Source(offset=offset)
        games = source.fetch_week(SEASON, target_week)
        self.assertTrue(games)
        for g in games:
            status, reveal = derive_status(
                g.kickoff_at, datetime.now(timezone.utc)
            )
            self.assertEqual(
                status,
                GameStatus.FINAL,
                f"week {target_week} game {g.espn_event_id} not FINAL",
            )
            self.assertTrue(reveal)

    def test_week_1_all_final(self) -> None:
        self._assert_all_final(1)

    def test_week_2_all_final(self) -> None:
        self._assert_all_final(2)

    def test_week_3_all_final(self) -> None:
        self._assert_all_final(3)


class MarginIsTimedeltaTest(unittest.TestCase):
    """The default safety margin is a small positive timedelta."""

    def test_default_margin(self) -> None:
        self.assertIsInstance(DEFAULT_MARGIN, timedelta)
        self.assertGreater(DEFAULT_MARGIN, timedelta(0))


if __name__ == "__main__":
    unittest.main()
