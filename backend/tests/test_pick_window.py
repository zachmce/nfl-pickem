"""Offline unit tests for the pure pick-window service.

These tests exercise :mod:`app.services.pick_window` with hand-built ``Game``
model instances (no database needed for the synthetic cases — they are plain
SQLModel objects), covering the window open/close boundaries, the week-1
unbounded-open special case, the per-game lock predicate, timezone-aware vs
naive handling, and the empty/kickoffless current-week programmer errors. One
additional *ground-truth* test imports the real 2025 season into an in-memory
SQLite db and checks that the week-2 window opens after week-1's last kickoff
and before week-2's first kickoff, fully offline.

Everything runs offline:

* the synthetic tests touch no database at all,
* the ground-truth test uses an in-memory SQLite engine
  (``create_engine("sqlite://")``) and never imports the Postgres engine module,
* there is no network access of any kind, and the service never reads the clock
  (``now`` is always injected by the caller).

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_pick_window -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m unittest ...`` or the venv
> interpreter ``.venv/bin/python -m unittest ...``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus
from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.pick_window import (
    DEFAULT_GAME_DURATION,
    PickWindow,
    compute_window,
    is_game_locked,
    is_pick_open,
)

# Small constant team ids for the synthetic games. The service only reads
# kickoff_at; it never loads Team rows, so these need not exist in any database.
HOME = 1
AWAY = 2


def _dt(*args: int) -> datetime:
    """Build a tz-aware UTC datetime, e.g. ``_dt(2025, 9, 7, 17, 0)``."""
    return datetime(*args, tzinfo=timezone.utc)


def _game(*, kickoff_at: datetime | None = None, game_id: int = 100) -> Game:
    """Build a synthetic ``Game`` with just the fields the service reads."""
    return Game(
        id=game_id,
        espn_event_id=game_id,
        week_id=1,
        season=2025,
        week=1,
        home_team_id=HOME,
        away_team_id=AWAY,
        kickoff_at=kickoff_at,
        status=GameStatus.SCHEDULED,
    )


class ComputeWindowTests(unittest.TestCase):
    """close_at = earliest current kickoff; open_at = latest prev + duration."""

    def test_close_at_is_earliest_current_kickoff(self) -> None:
        # Current week games handed in out of order; close = the earliest.
        week = [
            _game(kickoff_at=_dt(2025, 9, 14, 20, 0), game_id=1),
            _game(kickoff_at=_dt(2025, 9, 14, 17, 0), game_id=2),
            _game(kickoff_at=_dt(2025, 9, 15, 0, 15), game_id=3),
        ]
        window = compute_window(week, None)
        self.assertEqual(window.close_at, _dt(2025, 9, 14, 17, 0))

    def test_open_at_is_latest_prev_kickoff_plus_duration(self) -> None:
        prev = [
            _game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=10),
            _game(kickoff_at=_dt(2025, 9, 8, 0, 20), game_id=11),  # latest
            _game(kickoff_at=_dt(2025, 9, 7, 20, 5), game_id=12),
        ]
        week = [_game(kickoff_at=_dt(2025, 9, 14, 17, 0), game_id=1)]
        window = compute_window(week, prev)
        self.assertEqual(
            window.open_at, _dt(2025, 9, 8, 0, 20) + DEFAULT_GAME_DURATION
        )

    def test_returns_pickwindow_instance(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 14, 17, 0), game_id=1)]
        self.assertIsInstance(compute_window(week, None), PickWindow)

    def test_does_not_mutate_inputs(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 14, 17, 0), game_id=1)]
        prev = [_game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=10)]
        week_copy = list(week)
        prev_copy = list(prev)
        compute_window(week, prev)
        self.assertEqual(week, week_copy)
        self.assertEqual(prev, prev_copy)
        self.assertEqual(week[0].kickoff_at, _dt(2025, 9, 14, 17, 0))


class IsPickOpenBoundaryTests(unittest.TestCase):
    """Half-open window: includes open_at, excludes close_at."""

    def setUp(self) -> None:
        self.window = PickWindow(
            open_at=_dt(2025, 9, 9, 0, 0), close_at=_dt(2025, 9, 14, 17, 0)
        )

    def test_before_open_is_closed(self) -> None:
        self.assertFalse(is_pick_open(self.window, _dt(2025, 9, 8, 23, 59)))

    def test_exactly_at_open_is_open(self) -> None:
        self.assertTrue(is_pick_open(self.window, _dt(2025, 9, 9, 0, 0)))

    def test_mid_window_is_open(self) -> None:
        self.assertTrue(is_pick_open(self.window, _dt(2025, 9, 11, 12, 0)))

    def test_exactly_at_close_is_closed(self) -> None:
        self.assertFalse(is_pick_open(self.window, _dt(2025, 9, 14, 17, 0)))

    def test_after_close_is_closed(self) -> None:
        self.assertFalse(is_pick_open(self.window, _dt(2025, 9, 14, 17, 1)))


class WeekOneTests(unittest.TestCase):
    """Week 1 (no previous week) is unbounded-open: open_at is None."""

    def test_none_prev_yields_open_at_none(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=1)]
        self.assertIsNone(compute_window(week, None).open_at)

    def test_empty_prev_yields_open_at_none(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=1)]
        self.assertIsNone(compute_window(week, []).open_at)

    def test_prev_with_only_none_kickoffs_yields_open_at_none(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=1)]
        prev = [_game(kickoff_at=None, game_id=10)]
        self.assertIsNone(compute_window(week, prev).open_at)

    def test_open_for_any_now_before_close(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=1)]
        window = compute_window(week, None)
        # A very early time is still inside the unbounded-open lower boundary.
        self.assertTrue(is_pick_open(window, _dt(2000, 1, 1, 0, 0)))
        self.assertTrue(is_pick_open(window, _dt(2025, 9, 7, 16, 59)))

    def test_week_one_closes_at_first_kickoff(self) -> None:
        week = [_game(kickoff_at=_dt(2025, 9, 7, 17, 0), game_id=1)]
        window = compute_window(week, None)
        self.assertFalse(is_pick_open(window, _dt(2025, 9, 7, 17, 0)))


class IsGameLockedTests(unittest.TestCase):
    """Per-game lock: now >= kickoff; kickoff None is never locked."""

    def test_before_kickoff_not_locked(self) -> None:
        game = _game(kickoff_at=_dt(2025, 9, 14, 17, 0))
        self.assertFalse(is_game_locked(game, _dt(2025, 9, 14, 16, 59)))

    def test_exactly_at_kickoff_locked(self) -> None:
        game = _game(kickoff_at=_dt(2025, 9, 14, 17, 0))
        self.assertTrue(is_game_locked(game, _dt(2025, 9, 14, 17, 0)))

    def test_after_kickoff_locked(self) -> None:
        game = _game(kickoff_at=_dt(2025, 9, 14, 17, 0))
        self.assertTrue(is_game_locked(game, _dt(2025, 9, 14, 20, 0)))

    def test_kickoff_none_never_locked(self) -> None:
        game = _game(kickoff_at=None)
        self.assertFalse(is_game_locked(game, _dt(2025, 9, 14, 20, 0)))

    def test_later_game_locks_after_week_close(self) -> None:
        # The week closes at the earliest kickoff, but a later game in the same
        # week is still unlocked until its own kickoff.
        early = _game(kickoff_at=_dt(2025, 9, 14, 17, 0), game_id=1)
        late = _game(kickoff_at=_dt(2025, 9, 15, 0, 15), game_id=2)
        window = compute_window([early, late], None)
        now = _dt(2025, 9, 14, 18, 0)  # after week close, before late kickoff
        self.assertFalse(is_pick_open(window, now))
        self.assertTrue(is_game_locked(early, now))
        self.assertFalse(is_game_locked(late, now))


class TimezoneHandlingTests(unittest.TestCase):
    """Naive datetimes raise a deliberate ValueError, not a bare TypeError."""

    def test_is_pick_open_naive_now_raises(self) -> None:
        window = PickWindow(open_at=None, close_at=_dt(2025, 9, 14, 17, 0))
        with self.assertRaises(ValueError):
            is_pick_open(window, datetime(2025, 9, 11, 12, 0))

    def test_is_game_locked_naive_now_raises(self) -> None:
        game = _game(kickoff_at=_dt(2025, 9, 14, 17, 0))
        with self.assertRaises(ValueError):
            is_game_locked(game, datetime(2025, 9, 14, 16, 0))

    def test_is_game_locked_naive_kickoff_raises(self) -> None:
        game = _game(kickoff_at=datetime(2025, 9, 14, 17, 0))  # naive kickoff
        with self.assertRaises(ValueError):
            is_game_locked(game, _dt(2025, 9, 14, 18, 0))

    def test_compute_window_naive_kickoff_raises(self) -> None:
        week = [_game(kickoff_at=datetime(2025, 9, 14, 17, 0), game_id=1)]
        with self.assertRaises(ValueError):
            compute_window(week, None)


class ProgrammerErrorTests(unittest.TestCase):
    """Empty / kickoffless current week is a deliberate ValueError."""

    def test_empty_current_week_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_window([], None)

    def test_current_week_only_none_kickoff_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_window([_game(kickoff_at=None, game_id=1)], None)


class GroundTruthRealSeasonTests(unittest.TestCase):
    """Real 2025 week-1/week-2 window ordering, fully offline."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_week2_window_ordering(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)

            week1 = session.exec(
                select(Game).where(Game.season == 2025, Game.week == 1)
            ).all()
            week2 = session.exec(
                select(Game).where(Game.season == 2025, Game.week == 2)
            ).all()
            self.assertTrue(week1, "no week-1 games imported")
            self.assertTrue(week2, "no week-2 games imported")

            # SQLite has no tz-aware column type, so DateTime(timezone=True)
            # round-trips kickoffs back as naive datetimes (Postgres preserves
            # the tz). The fixture's source times are UTC, so re-attach UTC here
            # to mirror what production hands the (correctly tz-strict) service.
            for g in (*week1, *week2):
                if g.kickoff_at is not None and g.kickoff_at.tzinfo is None:
                    g.kickoff_at = g.kickoff_at.replace(tzinfo=timezone.utc)

            window = compute_window(week2, week1)

            week1_kickoffs = [g.kickoff_at for g in week1 if g.kickoff_at is not None]
            week2_kickoffs = [g.kickoff_at for g in week2 if g.kickoff_at is not None]
            self.assertTrue(week1_kickoffs and week2_kickoffs)

            self.assertIsNotNone(window.open_at)
            # open_at is strictly after week 1's last kickoff (latest + duration).
            self.assertGreater(window.open_at, max(week1_kickoffs))
            # close_at is week 2's earliest kickoff.
            self.assertEqual(window.close_at, min(week2_kickoffs))
            # And the window is non-degenerate: opens before it closes.
            self.assertLess(window.open_at, window.close_at)


if __name__ == "__main__":
    unittest.main()
