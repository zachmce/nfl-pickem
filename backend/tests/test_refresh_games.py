"""Offline tests for the ``refresh_games`` reconciliation service.

These exercise :func:`app.services.refresh.refresh_games` end-to-end against the
real packaged 2025 fixture (seeded via :mod:`app.seeds.fixture_2025`) into an
in-memory SQLite db, driven exclusively by
:class:`~app.scoreboard.demo.Demo2025Source` so status/scores derive from the
real clock plus a constructor offset — no clock monkeypatching.

Everything runs OFFLINE:

* an in-memory SQLite engine is constructed inside the test (no Postgres),
* :mod:`app.db` is deliberately NOT imported (it builds a Postgres engine at
  import time),
* :class:`~app.scoreboard.espn.EspnScoreboardSource` is NEVER instantiated for
  any path that fetches — there is no network access of any kind. The default
  source is positioned via the offset: a large NEGATIVE offset finals the whole
  season; a large POSITIVE offset keeps it SCHEDULED.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_refresh_games -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, Week
from app.scoreboard.demo import Demo2025Source
from app.scoreboard.port import ScoreboardFetchError
from app.scoreboard.types import ScoreboardGame
from app.seeds.fixture_2025 import FIXTURE_PATH, import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.pick_window import compute_window
from app.services.refresh import RefreshResult, refresh_games

# A large negative offset positions the entire 2025 fixture well in the past so
# every game derives as FINAL; a large positive offset keeps it all SCHEDULED.
FINAL_OFFSET = timedelta(days=-3650)
FUTURE_OFFSET = timedelta(days=3650)


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


class _RecordingSource:
    """Wraps a :class:`Demo2025Source`, recording each fetched (season, week)."""

    def __init__(self, inner: Demo2025Source) -> None:
        self._inner = inner
        self.fetched: list[tuple[int, int]] = []

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        self.fetched.append((season, week))
        return self._inner.fetch_week(season, week)


class _FailingWeekSource:
    """Delegates to an inner source but raises for one specific (season, week)."""

    def __init__(self, inner: Demo2025Source, fail: tuple[int, int]) -> None:
        self._inner = inner
        self._fail = fail
        self.fetched: list[tuple[int, int]] = []

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        if (season, week) == self._fail:
            raise ScoreboardFetchError(f"injected failure for {(season, week)}")
        self.fetched.append((season, week))
        return self._inner.fetch_week(season, week)


class RefreshGamesTests(unittest.TestCase):
    """Offline reconciliation tests over the real 2025 fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cls.season = int(cls.fixture["metadata"]["season"])

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)
            # The packaged 2025 fixture imports as a COMPLETED season (all rows
            # FINAL with real scores). Reset every row to a fresh pre-game state
            # (SCHEDULED, no scores) so refresh_games has non-FINAL weeks to
            # poll — i.e. simulate the schedule being ingested before kickoff.
            for g in session.exec(select(Game)).all():
                g.status = GameStatus.SCHEDULED
                g.home_score = None
                g.away_score = None
                session.add(g)
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    # -- helpers -----------------------------------------------------------

    def _games_for_week(self, session: Session, week: int) -> list[Game]:
        return list(
            session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == week
                )
            ).all()
        )

    def _week_row(self, session: Session, week: int) -> Week:
        row = session.exec(
            select(Week).where(Week.season == self.season, Week.week == week)
        ).first()
        assert row is not None, f"week {week} missing"
        return row

    @staticmethod
    def _aware(dt: datetime | None) -> datetime | None:
        if dt is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    # -- tests -------------------------------------------------------------

    def test_status_transition_and_score_reveal(self) -> None:
        """A SCHEDULED row becomes FINAL with the fixture's real scores."""
        source = Demo2025Source(offset=FINAL_OFFSET)
        with Session(self.engine) as session:
            # Rows default SCHEDULED with no scores after import.
            before = self._games_for_week(session, 1)
            self.assertTrue(all(g.status == GameStatus.SCHEDULED for g in before))

            result = refresh_games(session, source)
            session.commit()

            self.assertIsInstance(result, RefreshResult)
            self.assertGreater(result.games_updated, 0)

            after = self._games_for_week(session, 1)
            self.assertTrue(all(g.status == GameStatus.FINAL for g in after))
            # Scores were revealed (final games carry real scores).
            self.assertTrue(all(g.home_score is not None for g in after))
            self.assertTrue(all(g.away_score is not None for g in after))

    def test_idempotent_second_run_writes_nothing(self) -> None:
        """A second run against unchanged source state dirties zero rows."""
        source = Demo2025Source(offset=FINAL_OFFSET)
        with Session(self.engine) as session:
            refresh_games(session, source)
            session.commit()

            second = refresh_games(session, source)
            # No row/window should be dirty before commit.
            self.assertEqual(second.games_updated, 0)
            self.assertEqual(second.windows_stamped, 0)
            self.assertFalse(session.dirty, session.dirty)
            self.assertFalse(session.new, session.new)

    def test_only_polls_weeks_that_need_it(self) -> None:
        """A fully-FINAL week is never refetched."""
        with Session(self.engine) as session:
            # Mark all of week 1 FINAL up front.
            for g in self._games_for_week(session, 1):
                g.status = GameStatus.FINAL
                g.home_score = 0
                g.away_score = 0
                session.add(g)
            session.commit()

            spy = _RecordingSource(Demo2025Source(offset=FUTURE_OFFSET))
            refresh_games(session, spy)
            session.commit()

            polled_weeks = {wk for _s, wk in spy.fetched}
            self.assertNotIn(1, polled_weeks)
            # Other (non-final) weeks were still polled.
            self.assertTrue(polled_weeks)

    def test_window_stamping_closes_and_opens(self) -> None:
        """closes_at = first kickoff; successor opens_at = compute_window().open_at."""
        source = Demo2025Source(offset=FINAL_OFFSET)
        with Session(self.engine) as session:
            refresh_games(session, source)
            session.commit()

            wk1_games = self._games_for_week(session, 1)
            wk2_games = self._games_for_week(session, 2)

            # Re-attach UTC (SQLite drops tz) before computing expectations.
            for g in wk1_games + wk2_games:
                g.kickoff_at = self._aware(g.kickoff_at)

            expected_close_wk1 = compute_window(wk1_games, None).close_at
            expected_open_wk2 = compute_window(wk2_games, wk1_games).open_at
            self.assertIsNotNone(expected_open_wk2)

            wk1 = self._week_row(session, 1)
            wk2 = self._week_row(session, 2)

            self.assertEqual(
                self._aware(wk1.window_closes_at), expected_close_wk1
            )
            # Week 1 has no predecessor -> open stays None.
            self.assertIsNone(wk1.window_opens_at)
            # Week 1 is fully FINAL, so week 2's open is stamped.
            self.assertEqual(
                self._aware(wk2.window_opens_at), expected_open_wk2
            )

    def test_window_stamping_is_idempotent(self) -> None:
        """Re-running does not re-stamp already-correct windows."""
        source = Demo2025Source(offset=FINAL_OFFSET)
        with Session(self.engine) as session:
            refresh_games(session, source)
            session.commit()

            second = refresh_games(session, source)
            self.assertEqual(second.windows_stamped, 0)

    def test_in_progress_does_not_null_present_score(self) -> None:
        """A withheld (None) source score never overwrites a present score."""
        with Session(self.engine) as session:
            # Seed week 1 with a present score, status SCHEDULED.
            wk1 = self._games_for_week(session, 1)
            target = wk1[0]
            target.home_score = 21
            target.away_score = 17
            session.add(target)
            session.commit()
            target_id = target.id
            event_id = target.espn_event_id

        # A source that reports this game IN_PROGRESS with withheld scores.
        class _InProgressSource:
            def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
                from app.scoreboard.types import ScoreboardTeam

                if week != 1:
                    return []
                return [
                    ScoreboardGame(
                        espn_event_id=str(event_id),
                        season=season,
                        week=week,
                        kickoff_at=None,
                        status=GameStatus.IN_PROGRESS,
                        home=ScoreboardTeam(None, None, score=None),
                        away=ScoreboardTeam(None, None, score=None),
                    )
                ]

        with Session(self.engine) as session:
            refresh_games(session, _InProgressSource())
            session.commit()

            row = session.get(Game, target_id)
            assert row is not None
            self.assertEqual(row.status, GameStatus.IN_PROGRESS)
            # Present scores preserved, NOT nulled.
            self.assertEqual(row.home_score, 21)
            self.assertEqual(row.away_score, 17)

    def test_fetch_error_one_week_does_not_abort_others(self) -> None:
        """A ScoreboardFetchError on one week is recorded; others still update."""
        failing = _FailingWeekSource(
            Demo2025Source(offset=FINAL_OFFSET), fail=(self.season, 3)
        )
        with Session(self.engine) as session:
            result = refresh_games(session, failing)
            session.commit()

            self.assertIn((self.season, 3), result.failed_weeks)
            self.assertGreater(result.games_updated, 0)

            # The failed week was not written; another week was.
            wk3 = self._games_for_week(session, 3)
            self.assertTrue(all(g.status == GameStatus.SCHEDULED for g in wk3))
            wk1 = self._games_for_week(session, 1)
            self.assertTrue(all(g.status == GameStatus.FINAL for g in wk1))


if __name__ == "__main__":
    unittest.main()
