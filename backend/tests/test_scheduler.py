"""Offline tests for the source-agnostic scheduler/polling-job seam.

These exercise :mod:`app.services.scheduler` — the new seam that factors the
single ``refresh-games-poller`` beat into a first-class ``PollingJob`` unit, a
``POLLING_JOBS`` registry, and a shared per-week fetch helper, so a future odds
job can join as a sibling without disturbing the scores poller.

Everything runs OFFLINE, mirroring :mod:`tests.test_refresh_games`:

* an in-memory SQLite engine is constructed inside the test (no Postgres),
* :mod:`app.db` is deliberately NOT imported,
* :class:`~app.scoreboard.espn.EspnScoreboardSource` is NEVER instantiated — the
  fetch is driven by :class:`~app.scoreboard.demo.Demo2025Source`.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_scheduler -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
import unittest
from datetime import timedelta
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus
from app.scoreboard.demo import Demo2025Source
from app.scoreboard.port import ScoreboardFetchError
from app.scoreboard.types import ScoreboardGame
from app.seeds.fixture_2025 import FIXTURE_PATH, import_fixture_2025
from app.seeds.teams import seed_teams
from app.services import scheduler
from app.services.refresh import group_games_by_week

# A large negative offset positions the entire 2025 fixture in the past so every
# game derives FINAL; a large positive offset keeps it all SCHEDULED.
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


def _scores_job() -> scheduler.PollingJob:
    """The single registered scores job (fail loud if the registry shape drifts)."""
    jobs = [
        j
        for j in scheduler.POLLING_JOBS
        if j.beat_name == "refresh-games-poller"
    ]
    assert len(jobs) == 1, f"expected exactly one scores job, got {jobs}"
    return jobs[0]


class SchedulerRegistryTests(unittest.TestCase):
    """The registry holds the scores job with the exact beat wiring."""

    def test_registry_holds_only_the_scores_job_with_expected_wiring(self) -> None:
        # In THIS task the registry holds exactly one job: the scores job.
        self.assertEqual(len(scheduler.POLLING_JOBS), 1)
        job = scheduler.POLLING_JOBS[0]
        self.assertIsInstance(job, scheduler.PollingJob)
        self.assertEqual(job.beat_name, "refresh-games-poller")
        self.assertEqual(job.task_name, "app.tasks.refresh_games")
        self.assertEqual(job.schedule_seconds, 60.0)
        # The cadence is a float (Celery beat schedule contract).
        self.assertIsInstance(job.schedule_seconds, float)


class SchedulerNeedyPredicateTests(unittest.TestCase):
    """The scores job's needy predicate selects non-FINAL weeks only."""

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
            # Reset every row to SCHEDULED (the fixture imports a COMPLETED
            # season) so there are non-FINAL weeks to select.
            for g in session.exec(select(Game)).all():
                g.status = GameStatus.SCHEDULED
                g.home_score = None
                g.away_score = None
                session.add(g)
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_needy_selects_non_final_weeks_and_excludes_final_weeks(self) -> None:
        job = _scores_job()
        with Session(self.engine) as session:
            # Mark week 1 fully FINAL up front; the rest stay SCHEDULED (needy).
            wk1 = session.exec(
                select(Game).where(
                    Game.season == self.season, Game.week == 1
                )
            ).all()
            for g in wk1:
                g.status = GameStatus.FINAL
                g.home_score = 0
                g.away_score = 0
                session.add(g)
            session.commit()

            by_week = group_games_by_week(
                list(session.exec(select(Game)).all())
            )
            needy = job.needy(by_week)

            needy_set = set(needy)
            # Week 1 (fully FINAL) is excluded.
            self.assertNotIn((self.season, 1), needy_set)
            # Other weeks (with non-FINAL rows) are selected.
            self.assertIn((self.season, 2), needy_set)
            # Result is sorted.
            self.assertEqual(list(needy), sorted(needy))

    def test_needy_empty_when_all_weeks_final(self) -> None:
        job = _scores_job()
        with Session(self.engine) as session:
            for g in session.exec(select(Game)).all():
                g.status = GameStatus.FINAL
                g.home_score = 0
                g.away_score = 0
                session.add(g)
            session.commit()

            by_week = group_games_by_week(
                list(session.exec(select(Game)).all())
            )
            self.assertEqual(job.needy(by_week), [])


class SchedulerFetchHelperTests(unittest.TestCase):
    """The shared fetch helper issues one fetch per week + per-week failure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cls.season = int(cls.fixture["metadata"]["season"])

    def test_one_fetch_per_needy_week(self) -> None:
        spy = _RecordingSource(Demo2025Source(offset=FUTURE_OFFSET))
        weeks = [(self.season, 1), (self.season, 2), (self.season, 3)]

        fetched, failed = scheduler.fetch_needy_weeks(spy, weeks)

        # Exactly one fetch per requested week, no duplicates.
        self.assertEqual(spy.fetched, weeks)
        self.assertEqual(len(spy.fetched), len(set(spy.fetched)))
        # All weeks fetched successfully.
        self.assertEqual(set(fetched.keys()), set(weeks))
        self.assertEqual(failed, set())
        # Each value is the source's per-week games list.
        for key in weeks:
            self.assertIsInstance(fetched[key], list)

    def test_per_week_fetch_error_is_recorded_not_raised(self) -> None:
        failing = _FailingWeekSource(
            Demo2025Source(offset=FUTURE_OFFSET), fail=(self.season, 2)
        )
        weeks = [(self.season, 1), (self.season, 2), (self.season, 3)]

        # The failing week is recorded, not raised, and other weeks still fetch.
        fetched, failed = scheduler.fetch_needy_weeks(failing, weeks)

        self.assertIn((self.season, 2), failed)
        self.assertNotIn((self.season, 2), fetched)
        # The other two weeks fetched successfully.
        self.assertIn((self.season, 1), fetched)
        self.assertIn((self.season, 3), fetched)
        # The successful weeks were actually fetched on the inner source.
        self.assertIn((self.season, 1), failing.fetched)
        self.assertIn((self.season, 3), failing.fetched)
        self.assertNotIn((self.season, 2), failing.fetched)


class SchedulerSourceAgnosticGuardTests(unittest.TestCase):
    """scheduler.py must not couple to config / ESPN / the demo gate."""

    def test_scheduler_module_is_source_agnostic(self) -> None:
        source = Path(scheduler.__file__).read_text(encoding="utf-8")
        self.assertNotIn("app.config", source)
        self.assertNotIn("scoreboard.espn", source)
        self.assertNotIn("IS_DEMO_DATA", source)


if __name__ == "__main__":
    unittest.main()
