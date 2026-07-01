"""Offline tests for the shared ``active_season`` selector (260630-h96).

``app.services.standings.active_season`` is the ONE active-season selector that
the three formerly-divergent call sites (current-week endpoint, the poller recap
twin, the chat twin) now delegate to. The rule is the newest persisted season is
active (``max(Game.season)``), ``None`` only on an empty DB — deterministic, with
no clock or game-status dependency (spec: ``.planning/notes/active-season-model.md``).

This also pins ``notifications_read.current_season`` (the public twin that
``app/bot/db_bridge.py`` calls in 8 places) — previously untested — to the same
max-season behavior on a multi-season DB.

Everything runs OFFLINE on an in-memory SQLite engine (``StaticPool``); no
Postgres, no network. Run from ``backend/``::

    .venv/bin/python -m unittest tests.test_active_season -v
"""

from __future__ import annotations

import unittest

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.models import Game, GameStatus, Team, Week
from app.services.notifications_read import current_season
from app.services.standings import active_season


def _memory_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _seed_season_game(session: Session, *, season: int, week: int = 1) -> None:
    """Seed one minimal Week + Game row for ``season`` (enough to count a season)."""
    week_row = Week(season=season, week=week)
    session.add(week_row)
    session.commit()
    session.refresh(week_row)
    assert week_row.id is not None
    session.add(
        Game(
            espn_event_id=season * 1000 + week,
            week_id=week_row.id,
            season=season,
            week=week,
            home_team_id=1,
            away_team_id=2,
            status=GameStatus.SCHEDULED,
        )
    )
    session.commit()


class ActiveSeasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _memory_engine()
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as s:
            # Two FK-target teams the seeded games reference.
            s.add_all(
                [
                    Team(espn_team_id=1, abbreviation="AAA", display_name="Team A"),
                    Team(espn_team_id=2, abbreviation="BBB", display_name="Team B"),
                ]
            )
            s.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_empty_db_returns_none(self) -> None:
        with Session(self.engine) as s:
            self.assertIsNone(active_season(s))

    def test_single_season_returns_that_season(self) -> None:
        with Session(self.engine) as s:
            _seed_season_game(s, season=2025)
        with Session(self.engine) as s:
            self.assertEqual(active_season(s), 2025)

    def test_multi_season_returns_the_max(self) -> None:
        with Session(self.engine) as s:
            _seed_season_game(s, season=2024)
            _seed_season_game(s, season=2025)
        with Session(self.engine) as s:
            self.assertEqual(active_season(s), 2025)

    def test_current_season_twin_returns_max_on_multi_season(self) -> None:
        # notifications_read.current_season delegates to active_season; on a
        # two-season DB it must resolve the newer season (was: None, untested).
        with Session(self.engine) as s:
            _seed_season_game(s, season=2024)
            _seed_season_game(s, season=2025)
        with Session(self.engine) as s:
            self.assertEqual(current_season(s), 2025)

    def test_current_season_twin_returns_none_on_empty(self) -> None:
        with Session(self.engine) as s:
            self.assertIsNone(current_season(s))


if __name__ == "__main__":
    unittest.main()
