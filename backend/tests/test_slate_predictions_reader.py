"""Offline seeded-session tests for the whole-slate predictions reader (260713-k6z).

Exercises :func:`app.services.notifications_read.get_slate_predictions_for_week`
against an in-memory SQLite database (no Postgres, no network — ``app.db`` is
deliberately NOT imported because it builds a Postgres engine at import time),
combining the seeding style of ``tests/test_ratings.py`` (``HistoricalGame`` rows so
``compute_ratings`` yields non-uniform ratings) with the ``Game`` line-field seeding
style of ``tests/test_slate_api.py``.

The reader is DISPLAY-ONLY (T-k6z-01): the leak assertion pins the per-game dict key
set to the display fields and proves NO pick/user key ever crosses the boundary.

Run from the ``backend/`` directory with the standard-library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_slate_predictions_reader -v

(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, HistoricalGame, Team, Week
from app.seeds.teams import seed_teams
from app.services.notifications_read import get_slate_predictions_for_week

SEASON = 2025
WEEK = 5

# The exact display-only key set every per-game dict must carry — the leak guard.
_EXPECTED_GAME_KEYS = {"away", "home", "favorite", "underdog", "spread", "model_margin"}


class SlatePredictionsReaderTests(unittest.TestCase):
    """The reader pairs a per-game model margin with the frozen line, display-only."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            seed_teams(session)
            ids = sorted(t.id for t in session.exec(select(Team)).all() if t.id is not None)
            # Four distinct real Team.id values: two per current-week game.
            self.home_fav_id: int = ids[0]
            self.away_dog_id: int = ids[1]
            self.home_noline_id: int = ids[2]
            self.away_noline_id: int = ids[3]
            self.abbr_by_id = {
                t.id: t.abbreviation for t in session.exec(select(Team)).all() if t.id is not None
            }

    def tearDown(self) -> None:
        self.engine.dispose()

    # --- seeding helpers ---------------------------------------------------
    def _hist(
        self,
        *,
        season: int,
        week: int,
        gameday: date,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                HistoricalGame(
                    nflverse_game_id=f"{season}_{week:02d}_{home_team_id}_{away_team_id}",
                    season=season,
                    week=week,
                    game_type="REG",
                    gameday=gameday,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_score=home_score,
                    away_score=away_score,
                    result=home_score - away_score,
                    spread_line=Decimal("0"),
                )
            )
            session.commit()

    def _week_id(self, session: Session, *, season: int, week: int) -> int:
        existing = session.exec(
            select(Week).where(Week.season == season, Week.week == week)
        ).first()
        if existing is not None:
            assert existing.id is not None
            return existing.id
        row = Week(season=season, week=week)
        session.add(row)
        session.commit()
        session.refresh(row)
        assert row.id is not None
        return row.id

    def _game(
        self,
        *,
        espn_event_id: int,
        home_team_id: int,
        away_team_id: int,
        kickoff_at: datetime,
        spread: Decimal | None = None,
        favorite_team_id: int | None = None,
        underdog_team_id: int | None = None,
    ) -> None:
        with Session(self.engine) as session:
            week_id = self._week_id(session, season=SEASON, week=WEEK)
            session.add(
                Game(
                    espn_event_id=espn_event_id,
                    week_id=week_id,
                    season=SEASON,
                    week=WEEK,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    kickoff_at=kickoff_at,
                    status=GameStatus.SCHEDULED,
                    spread=spread,
                    favorite_team_id=favorite_team_id,
                    underdog_team_id=underdog_team_id,
                )
            )
            session.commit()

    def _seed(self) -> None:
        # A handful of prior-season HISTORICAL games so compute_ratings yields
        # non-uniform ratings (the home_fav team wins big; the away team loses).
        for i in range(3):
            self._hist(
                season=2011,
                week=i + 1,
                gameday=date(2011, 9, 11 + i),
                home_team_id=self.home_fav_id,
                away_team_id=self.away_dog_id,
                home_score=31,
                away_score=10,
            )
        # Current-week game 1: a posted HOME-favorite line.
        self._game(
            espn_event_id=5001,
            home_team_id=self.home_fav_id,
            away_team_id=self.away_dog_id,
            kickoff_at=datetime(2025, 10, 5, 17, 0, tzinfo=timezone.utc),
            spread=Decimal("3.5"),
            favorite_team_id=self.home_fav_id,
            underdog_team_id=self.away_dog_id,
        )
        # Current-week game 2: NO line posted (favorite/underdog/spread all None).
        self._game(
            espn_event_id=5002,
            home_team_id=self.home_noline_id,
            away_team_id=self.away_noline_id,
            kickoff_at=datetime(2025, 10, 5, 20, 0, tzinfo=timezone.utc),
        )

    # --- (a) one dict per game with the right favorite/underdog/spread ----
    def test_reader_pairs_model_margin_with_the_frozen_line(self) -> None:
        self._seed()
        with Session(self.engine) as session:
            slate = get_slate_predictions_for_week(session, SEASON, WEEK)

        self.assertEqual(slate["week"], WEEK)
        games = slate["games"]
        self.assertEqual(len(games), 2)

        by_home = {g["home"]: g for g in games}
        fav_abbr = self.abbr_by_id[self.home_fav_id]
        dog_abbr = self.abbr_by_id[self.away_dog_id]
        posted = by_home[fav_abbr]
        # Favorite/underdog/spread pair straight off the seeded Game row.
        self.assertEqual(posted["favorite"], fav_abbr)
        self.assertEqual(posted["underdog"], dog_abbr)
        self.assertEqual(posted["spread"], "3.5")
        self.assertEqual(posted["away"], dog_abbr)

    # --- (b) each game carries a numeric model_margin ---------------------
    def test_every_game_carries_a_numeric_model_margin(self) -> None:
        self._seed()
        with Session(self.engine) as session:
            slate = get_slate_predictions_for_week(session, SEASON, WEEK)
        for g in slate["games"]:
            self.assertIsInstance(g["model_margin"], float)
        # The home-favorite team beat the away team three times historically, so its
        # home-relative model margin is a strong positive (home favored by the model).
        fav_abbr = self.abbr_by_id[self.home_fav_id]
        posted = next(g for g in slate["games"] if g["home"] == fav_abbr)
        self.assertGreater(posted["model_margin"], 0.0)

    # --- (c) the no-line game still carries a model_margin ----------------
    def test_unposted_line_game_still_has_model_margin(self) -> None:
        self._seed()
        with Session(self.engine) as session:
            slate = get_slate_predictions_for_week(session, SEASON, WEEK)
        noline_home = self.abbr_by_id[self.home_noline_id]
        unposted = next(g for g in slate["games"] if g["home"] == noline_home)
        self.assertIsNone(unposted["favorite"])
        self.assertIsNone(unposted["underdog"])
        self.assertIsNone(unposted["spread"])
        # No line, but the model still has a number for the matchup.
        self.assertIsInstance(unposted["model_margin"], float)

    # --- (d) LEAK ASSERTION: display-only key set, no pick/user data ------
    def test_no_pick_or_user_data_crosses_the_boundary(self) -> None:
        self._seed()
        with Session(self.engine) as session:
            slate = get_slate_predictions_for_week(session, SEASON, WEEK)
        # Top-level shape carries no user/pick key.
        self.assertEqual(set(slate.keys()), {"week", "close_at", "pick_open", "games"})
        # Each per-game dict carries EXACTLY the display set — nothing more.
        for g in slate["games"]:
            self.assertEqual(set(g.keys()), _EXPECTED_GAME_KEYS)


if __name__ == "__main__":
    unittest.main()
