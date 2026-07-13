"""Offline unit tests for the bot's deterministic Elo rating engine.

The five locked behaviors are covered against an in-memory SQLite database (no
Postgres, no network — ``app.db`` is deliberately NOT imported because it builds
a Postgres engine at import time), mirroring the pattern in
``test_historical_games_seed``:

(a) a home win moves the winner above 1500 and the loser symmetrically below;
(b) equal ratings give the home side the HFA edge (2.2 pt), and a hand-built
    ratings dict yields the exact ``(diff + HFA) / ELO_PER_POINT`` margin;
(c) the between-season regression pulls a rating one-third toward 1500;
(d) BOTH tables feed one walk-forward stream (a later-season FINAL ``Game``
    changes the outcome of a history-only snapshot);
(e) a non-FINAL (or score-missing) ``Game`` never feeds the stream.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_ratings -v

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, HistoricalGame, Team, Week
from app.seeds.teams import seed_teams
from app.services.ratings import (
    ELO_PER_POINT,
    HFA_ELO,
    MEAN,
    compute_ratings,
    estimate,
    expected_margin,
    regress_toward_mean,
    win_probability,
)


class RatingEngineTests(unittest.TestCase):
    """Deterministic Elo behavior against an in-memory SQLite database."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            seed_teams(session)
            # Grab three distinct real Team.id values (home / away / third).
            teams = session.exec(select(Team)).all()
            ids = sorted(t.id for t in teams if t.id is not None)
            self.home_id: int = ids[0]
            self.away_id: int = ids[1]
            self.third_id: int = ids[2]

    def tearDown(self) -> None:
        self.engine.dispose()

    # --- row-insert helpers ------------------------------------------------
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
        """Return the ``Week.id`` for ``(season, week)``, creating the row once."""
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
        season: int,
        week: int,
        home_team_id: int,
        away_team_id: int,
        kickoff_at: datetime,
        status: GameStatus,
        home_score: int | None,
        away_score: int | None,
    ) -> None:
        with Session(self.engine) as session:
            week_id = self._week_id(session, season=season, week=week)
            session.add(
                Game(
                    espn_event_id=espn_event_id,
                    week_id=week_id,
                    season=season,
                    week=week,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    kickoff_at=kickoff_at,
                    status=status,
                    home_score=home_score,
                    away_score=away_score,
                )
            )
            session.commit()

    # --- (a) winner rises / loser falls -----------------------------------
    def test_home_win_moves_ratings_symmetrically(self) -> None:
        self._hist(
            season=2011,
            week=1,
            gameday=date(2011, 9, 11),
            home_team_id=self.home_id,
            away_team_id=self.away_id,
            home_score=27,
            away_score=10,
        )
        with Session(self.engine) as session:
            ratings = compute_ratings(session)
        self.assertGreater(ratings[self.home_id], MEAN)
        self.assertLess(ratings[self.away_id], MEAN)
        # The delta is applied +home / -away, so both moves are symmetric.
        self.assertAlmostEqual(ratings[self.home_id] - MEAN, MEAN - ratings[self.away_id])

    # --- (b) HFA edge on equal ratings ------------------------------------
    def test_expected_margin_hfa_and_diff(self) -> None:
        # Empty DB -> no games -> empty ratings; equal (defaulted) ratings give
        # the home side exactly the HFA edge.
        with Session(self.engine) as session:
            ratings = compute_ratings(session)
        self.assertEqual(ratings, {})
        self.assertAlmostEqual(
            expected_margin(self.home_id, self.away_id, ratings),
            HFA_ELO / ELO_PER_POINT,
        )
        self.assertAlmostEqual(expected_margin(self.home_id, self.away_id, {}), 2.2)

        # A hand-built non-equal ratings dict: (1700 - 1500 + 55) / 25 == 10.2.
        hand = {self.home_id: 1700.0, self.away_id: 1500.0}
        self.assertAlmostEqual(expected_margin(self.home_id, self.away_id, hand), 10.2)
        est = estimate(self.home_id, self.away_id, hand)
        self.assertAlmostEqual(est.expected_margin, 10.2)
        self.assertAlmostEqual(est.home_rating, 1700.0)
        self.assertAlmostEqual(est.away_rating, 1500.0)

    # --- (b2) outright win probability ------------------------------------
    def test_win_probability_edge_gap_and_bounds(self) -> None:
        # Equal (defaulted) ratings: the home side carries the HFA edge, so its
        # outright win probability is strictly above a coin flip.
        equal = win_probability(self.home_id, self.away_id, {})
        self.assertGreater(equal, 0.5)
        self.assertLess(equal, 1.0)

        # A heavily higher-rated home team is nearly certain to win outright (but
        # never exactly 1.0 — the logistic is open at both ends).
        lopsided = {self.home_id: 2100.0, self.away_id: 1300.0}
        big = win_probability(self.home_id, self.away_id, lopsided)
        self.assertGreater(big, 0.95)
        self.assertLess(big, 1.0)

        # A heavily lower-rated home team is nearly certain to lose, but still > 0.
        weak = win_probability(
            self.home_id, self.away_id, {self.home_id: 1300.0, self.away_id: 2100.0}
        )
        self.assertGreater(weak, 0.0)
        self.assertLess(weak, 0.05)

    def test_win_probability_sign_consistent_with_expected_margin(self) -> None:
        # A positive expected home margin must imply a home win probability > 0.5, and a
        # negative expected home margin < 0.5 — the two read off the same elo_diff.
        fav = {self.home_id: 1700.0, self.away_id: 1500.0}
        self.assertGreater(expected_margin(self.home_id, self.away_id, fav), 0.0)
        self.assertGreater(win_probability(self.home_id, self.away_id, fav), 0.5)

        # Away much stronger than home, enough to overcome HFA -> negative home margin.
        dog = {self.home_id: 1400.0, self.away_id: 1800.0}
        self.assertLess(expected_margin(self.home_id, self.away_id, dog), 0.0)
        self.assertLess(win_probability(self.home_id, self.away_id, dog), 0.5)

    def test_estimate_populates_home_win_prob(self) -> None:
        hand = {self.home_id: 1700.0, self.away_id: 1500.0}
        est = estimate(self.home_id, self.away_id, hand)
        # The estimate carries the win probability off the SAME snapshot.
        self.assertAlmostEqual(est.home_win_prob, win_probability(self.home_id, self.away_id, hand))
        self.assertGreater(est.home_win_prob, 0.5)
        self.assertLess(est.home_win_prob, 1.0)

    # --- (c) cross-season regression toward 1500 --------------------------
    def test_regress_toward_mean(self) -> None:
        regressed = regress_toward_mean(1600.0)
        self.assertAlmostEqual(regressed, 1500.0 + 100.0 * (1.0 - 1.0 / 3.0))
        self.assertLess(regressed, 1600.0)
        self.assertGreater(regressed, 1500.0)
        # A mean rating is unchanged.
        self.assertAlmostEqual(regress_toward_mean(MEAN), MEAN)

    # --- (d) union is walk-forward; both tables feed the stream -----------
    def test_union_stream_walks_both_tables_forward(self) -> None:
        # Earlier-season historical game: A (home) beats B.
        self._hist(
            season=2011,
            week=1,
            gameday=date(2011, 9, 11),
            home_team_id=self.home_id,
            away_team_id=self.away_id,
            home_score=27,
            away_score=10,
        )
        with Session(self.engine) as session:
            rating_after_hist = compute_ratings(session)[self.home_id]
        self.assertGreater(rating_after_hist, MEAN)

        # Later-season FINAL live Game: A (home) wins again.
        self._game(
            espn_event_id=1001,
            season=2012,
            week=1,
            home_team_id=self.home_id,
            away_team_id=self.away_id,
            kickoff_at=datetime(2012, 9, 9, 17, 0, tzinfo=timezone.utc),
            status=GameStatus.FINAL,
            home_score=24,
            away_score=17,
        )
        with Session(self.engine) as session:
            final_rating = compute_ratings(session)[self.home_id]

        # The FINAL live Game fed the stream and changed the outcome: A's rating
        # differs from the history-only snapshot, and A stays above the mean
        # after winning both games (even across the season-boundary regression).
        self.assertNotAlmostEqual(final_rating, rating_after_hist)
        self.assertGreater(final_rating, MEAN)

    # --- (e) non-FINAL / score-missing games are excluded -----------------
    def test_non_final_game_does_not_feed_stream(self) -> None:
        self._hist(
            season=2011,
            week=1,
            gameday=date(2011, 9, 11),
            home_team_id=self.home_id,
            away_team_id=self.away_id,
            home_score=27,
            away_score=10,
        )
        with Session(self.engine) as session:
            baseline = compute_ratings(session)

        # A SCHEDULED game with no scores must not feed the stream.
        self._game(
            espn_event_id=2001,
            season=2011,
            week=2,
            home_team_id=self.home_id,
            away_team_id=self.third_id,
            kickoff_at=datetime(2011, 9, 18, 17, 0, tzinfo=timezone.utc),
            status=GameStatus.SCHEDULED,
            home_score=None,
            away_score=None,
        )
        # A FINAL game MISSING a score must likewise be excluded.
        self._game(
            espn_event_id=2002,
            season=2011,
            week=3,
            home_team_id=self.away_id,
            away_team_id=self.third_id,
            kickoff_at=datetime(2011, 9, 25, 17, 0, tzinfo=timezone.utc),
            status=GameStatus.FINAL,
            home_score=None,
            away_score=14,
        )
        with Session(self.engine) as session:
            after = compute_ratings(session)

        self.assertEqual(after, baseline)


if __name__ == "__main__":
    unittest.main()
