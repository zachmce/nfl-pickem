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

    def test_kickoff_reconciled_when_source_differs(self) -> None:
        """The positioned source kickoff is reconciled onto the row (flex move).

        NFL flex scheduling moves kickoffs; the persisted kickoff drives the
        pick window/lock, so refresh must reconcile it from the source. The
        Demo2025Source positions every kickoff by its offset, so after a refresh
        each row's kickoff_at must equal the SOURCE's positioned kickoff, not the
        original fixture kickoff. A second identical run is a no-op (idempotent).
        """
        source = Demo2025Source(offset=FINAL_OFFSET)
        with Session(self.engine) as session:
            # Capture the source's positioned week-1 kickoffs (by event id).
            positioned = {
                sg.espn_event_id: sg.kickoff_at
                for sg in source.fetch_week(self.season, 1)
            }
            self.assertTrue(positioned)

            # Pre-refresh: the persisted kickoff is the ORIGINAL fixture kickoff
            # (no offset), so it differs from the positioned source kickoff.
            before = self._games_for_week(session, 1)
            for g in before:
                src_ko = positioned[str(g.espn_event_id)]
                self.assertNotEqual(
                    self._aware(g.kickoff_at),
                    src_ko,
                    "precondition: persisted kickoff should differ pre-refresh",
                )

            refresh_games(session, source)
            session.commit()

            # Post-refresh: each row's kickoff equals the SOURCE's positioned one.
            after = self._games_for_week(session, 1)
            for g in after:
                src_ko = positioned[str(g.espn_event_id)]
                self.assertEqual(
                    self._aware(g.kickoff_at),
                    src_ko,
                    f"game {g.espn_event_id} kickoff not reconciled to source",
                )

            # Idempotent: a second identical run dirties nothing.
            second = refresh_games(session, source)
            self.assertEqual(second.games_updated, 0)
            self.assertFalse(session.dirty, session.dirty)

    def test_kickoff_unchanged_is_a_noop(self) -> None:
        """A source kickoff equal to the persisted value never dirties the row."""
        from app.scoreboard.types import ScoreboardTeam

        with Session(self.engine) as session:
            target = self._games_for_week(session, 1)[0]
            target_id = target.id
            event_id = target.espn_event_id
            # Pin the row to a known tz-aware kickoff and FINAL state.
            fixed_kickoff = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
            target.kickoff_at = fixed_kickoff
            target.status = GameStatus.FINAL
            target.home_score = 21
            target.away_score = 17
            session.add(target)
            session.commit()

        class _SameKickoffSource:
            """Reports week 1's target game with the SAME kickoff, FINAL, scores."""

            def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
                if week != 1:
                    return []
                return [
                    ScoreboardGame(
                        espn_event_id=str(event_id),
                        season=season,
                        week=week,
                        kickoff_at=fixed_kickoff,
                        status=GameStatus.FINAL,
                        home=ScoreboardTeam(None, None, score=21),
                        away=ScoreboardTeam(None, None, score=17),
                    )
                ]

        with Session(self.engine) as session:
            # Make week 1 the only needy week by finalizing every other row up
            # front is unnecessary — needy weeks are those with a non-FINAL row;
            # the source above returns [] for them so they simply do not update.
            row = session.get(Game, target_id)
            assert row is not None
            row.status = GameStatus.SCHEDULED  # make week 1 needy so it's polled
            session.add(row)
            session.commit()

            result = refresh_games(session, _SameKickoffSource())
            # Only the status flips back to FINAL; kickoff is an unchanged no-op.
            session.commit()
            refreshed = session.get(Game, target_id)
            assert refreshed is not None
            self.assertEqual(self._aware(refreshed.kickoff_at), fixed_kickoff)

            # A second run with everything already matching dirties nothing.
            second = refresh_games(session, _SameKickoffSource())
            self.assertEqual(second.games_updated, 0)
            self.assertFalse(session.dirty, session.dirty)

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


class RefreshEdgeDetectionTests(unittest.TestCase):
    """In-cycle edge collection: game.final + week.recap fire once-on-transition.

    Drives two reconcile cycles over the real 2025 fixture (reset to a fresh
    pre-game SCHEDULED state) and asserts the edges appear EXACTLY on the cycle
    that transitions a game to FINAL and are ABSENT on a steady-state re-poll
    (the fully-final week is not even re-fetched).
    """

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
            for g in session.exec(select(Game)).all():
                g.status = GameStatus.SCHEDULED
                g.home_score = None
                g.away_score = None
                session.add(g)
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_game_final_recorded_once_then_empty_on_repoll(self) -> None:
        """A finalizing cycle records finalized_games; the re-poll records none."""
        with Session(self.engine) as session:
            spy = _RecordingSource(Demo2025Source(offset=FINAL_OFFSET))

            first = refresh_games(session, spy)
            session.commit()
            # At least one game finalized this cycle -> recorded once each.
            self.assertGreater(len(first.finalized_games), 0)
            # Every finalized entry carries display data: week + abbrs + scores.
            for fg in first.finalized_games:
                self.assertEqual(len(fg), 5)
                week, away, home, away_score, home_score = fg
                self.assertIsInstance(week, int)
                self.assertIsInstance(away, str)
                self.assertIsInstance(home, str)
                self.assertIsInstance(away_score, int)
                self.assertIsInstance(home_score, int)

            spy.fetched.clear()
            second = refresh_games(session, spy)
            session.commit()
            # Steady-state: nothing transitions, nothing re-fetched.
            self.assertEqual(second.finalized_games, ())
            self.assertEqual(spy.fetched, [])

    def test_week_recap_recorded_once_then_empty_on_repoll(self) -> None:
        """The week whose last game finals this cycle appears once in recap_weeks."""
        with Session(self.engine) as session:
            spy = _RecordingSource(Demo2025Source(offset=FINAL_OFFSET))

            first = refresh_games(session, spy)
            session.commit()
            # Every week finals this cycle -> each fully-final week recaps once.
            self.assertGreater(len(first.recap_weeks), 0)
            # Recap weeks are unique (no double-add) and are real week numbers.
            self.assertEqual(len(first.recap_weeks), len(set(first.recap_weeks)))

            second = refresh_games(session, spy)
            session.commit()
            self.assertEqual(second.recap_weeks, ())

    def test_recap_only_when_last_game_finals_this_cycle(self) -> None:
        """A week with a still-non-final game does NOT recap yet."""
        from app.scoreboard.types import ScoreboardTeam

        with Session(self.engine) as session:
            wk1 = self._games_for_week(session, 1)
            self.assertGreater(len(wk1), 1)  # need at least two games
            final_target = wk1[0]
            event_id = final_target.espn_event_id

            class _OneGameFinalSource:
                """Finals exactly ONE week-1 game; the rest stay SCHEDULED."""

                def fetch_week(self, season: int, week: int):
                    if week != 1:
                        return []
                    return [
                        ScoreboardGame(
                            espn_event_id=str(event_id),
                            season=season,
                            week=week,
                            kickoff_at=None,
                            status=GameStatus.FINAL,
                            home=ScoreboardTeam(None, None, score=24),
                            away=ScoreboardTeam(None, None, score=17),
                        )
                    ]

            result = refresh_games(session, _OneGameFinalSource())
            session.commit()
            # One game finalized, but week 1 is NOT fully final -> no recap.
            self.assertEqual(len(result.finalized_games), 1)
            self.assertNotIn(1, result.recap_weeks)

    def _games_for_week(self, session: Session, week: int) -> list[Game]:
        return list(
            session.exec(
                select(Game).where(Game.season == self.season, Game.week == week)
            ).all()
        )


class RefreshWindowEdgeTests(unittest.TestCase):
    """In-cycle window.opened / window.closed edges fire once on the crossing.

    These build a tiny two-week DB directly (no fixture) and a synthetic source
    with explicit per-week kickoffs so a single reconcile crosses a window
    boundary at an injected ``now``. Proves the crossing is recorded ONCE and a
    steady-state re-poll (same kickoffs both ends) records neither.
    """

    SEASON = 2025

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        from app.models import Team

        with Session(self.engine) as session:
            session.add_all(
                [
                    Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                    for i in range(1, 5)
                ]
            )
            session.commit()
            self.team_ids = [
                t.id for t in session.exec(select(Team)).all() if t.id is not None
            ]

    def tearDown(self) -> None:
        self.engine.dispose()

    def _seed_week_game(
        self, session: Session, *, week: int, event_id: int, kickoff: datetime
    ) -> None:
        wk = Week(season=self.SEASON, week=week)
        session.add(wk)
        session.commit()
        session.refresh(wk)
        session.add(
            Game(
                espn_event_id=event_id,
                week_id=wk.id,
                season=self.SEASON,
                week=week,
                home_team_id=self.team_ids[0],
                away_team_id=self.team_ids[1],
                kickoff_at=kickoff,
                status=GameStatus.SCHEDULED,
            )
        )
        session.commit()

    def test_window_closed_edge_once_then_none_on_repoll(self) -> None:
        """A week whose earliest kickoff moves into the past closes the window."""
        from app.scoreboard.types import ScoreboardTeam

        now = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)
        future = now + timedelta(days=2)
        past = now - timedelta(hours=2)

        with Session(self.engine) as session:
            # Persisted kickoff is in the FUTURE -> window OPEN at `now`.
            self._seed_week_game(session, week=1, event_id=5001, kickoff=future)

        # Source repositions the game's kickoff to the PAST and finals it ->
        # earliest kickoff now in the past -> window CLOSED at `now`.
        def _src(season, week):
            if week != 1:
                return []
            return [
                ScoreboardGame(
                    espn_event_id="5001",
                    season=season,
                    week=week,
                    kickoff_at=past,
                    status=GameStatus.FINAL,
                    home=ScoreboardTeam(None, None, score=21),
                    away=ScoreboardTeam(None, None, score=17),
                )
            ]

        class _Src:
            fetch_week = staticmethod(_src)

        with Session(self.engine) as session:
            first = refresh_games(session, _Src(), now=now)
            session.commit()
            self.assertIn(1, first.windows_closed)
            self.assertNotIn(1, first.windows_opened)

            # Steady-state re-poll: week 1 is now fully FINAL -> not refetched, and
            # the window boolean is the same both ends -> no edge.
            second = refresh_games(session, _Src(), now=now)
            self.assertEqual(second.windows_closed, ())
            self.assertEqual(second.windows_opened, ())

    def test_window_opened_edge_once_then_none_on_repoll(self) -> None:
        """Week 2 opens once week 1's last game (its kickoff) moves into the past."""
        from app.scoreboard.types import ScoreboardTeam

        now = datetime(2025, 9, 18, 12, 0, tzinfo=timezone.utc)
        wk1_future = now + timedelta(days=5)  # before: wk1 far future -> wk2 closed
        wk1_past = now - timedelta(hours=6)  # after: wk1 past -> wk2 open_at past
        wk2_future = now + timedelta(days=3)  # wk2 close stays future -> still open

        with Session(self.engine) as session:
            self._seed_week_game(session, week=1, event_id=6001, kickoff=wk1_future)
            self._seed_week_game(session, week=2, event_id=6002, kickoff=wk2_future)

        def _src(season, week):
            if week == 1:
                return [
                    ScoreboardGame(
                        espn_event_id="6001",
                        season=season,
                        week=week,
                        kickoff_at=wk1_past,
                        status=GameStatus.FINAL,
                        home=ScoreboardTeam(None, None, score=10),
                        away=ScoreboardTeam(None, None, score=7),
                    )
                ]
            if week == 2:
                # Week 2 stays SCHEDULED with its future kickoff (window stays open).
                return [
                    ScoreboardGame(
                        espn_event_id="6002",
                        season=season,
                        week=week,
                        kickoff_at=wk2_future,
                        status=GameStatus.SCHEDULED,
                        home=ScoreboardTeam(None, None, score=None),
                        away=ScoreboardTeam(None, None, score=None),
                    )
                ]
            return []

        class _Src:
            fetch_week = staticmethod(_src)

        with Session(self.engine) as session:
            first = refresh_games(session, _Src(), now=now)
            session.commit()
            self.assertIn(2, first.windows_opened)
            self.assertNotIn(2, first.windows_closed)

            # Re-poll: week 1 is fully final (not refetched), week 2 unchanged ->
            # the window boolean is stable both ends -> no new edge.
            second = refresh_games(session, _Src(), now=now)
            self.assertEqual(second.windows_opened, ())


class BeatWiringRegressionTests(unittest.TestCase):
    """The scores beat is preserved byte-for-byte after the scheduler refactor.

    The beat_schedule is now DERIVED from the polling-job registry
    (:data:`app.services.scheduler.POLLING_JOBS`) rather than a hard-coded literal.
    This regression test pins the observable result for the scores poller: the
    registry must still produce the exact historical entry.
    """

    def test_refresh_games_beat_entry_is_unchanged(self) -> None:
        from app.celery_app import celery_app

        entry = celery_app.conf.beat_schedule["refresh-games-poller"]
        self.assertEqual(
            entry,
            {"task": "app.tasks.refresh_games", "schedule": 60.0},
        )


if __name__ == "__main__":
    unittest.main()
