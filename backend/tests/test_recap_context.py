"""Offline tests for the read-only recap-context reader (260627-tfb).

These mirror :mod:`tests.test_results_api`'s setUp: a single shared in-memory
SQLite connection (``StaticPool``), Teams + a Week + FINAL Games with real
scores/odds + Users + Picks seeded directly (reads do not gate on the window).
They pin the Tier-2 recap contract:

* :func:`app.services.notifications_read.get_recap_context` returns a plain dict
  ``{week, weekly_scores, season_standings}`` built over the EXISTING
  ``season_standings`` + ``week_results`` services — no scoring/standings math
  re-implemented.
* ``weekly_scores`` is ``{display_name, weekly_score}`` high->low, matching
  ``week_results``; ``season_standings`` is ``{display_name, season_total, rank,
  gap_to_leader}`` matching ``season_standings``.
* The output is DISPLAY-ONLY: no ``user_id`` key ever appears (T-tfb-01).
* An empty week/season yields empty lists (no raise).

Run with: ``backend/.venv/bin/python -m unittest tests.test_recap_context -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.models import Game, GameStatus, Pick, PickType, Team, User, Week
from app.services.auth import hash_password
from app.services.notifications_read import get_recap_context
from app.services.standings import season_standings, week_results

SEASON = 2025
WEEK = 1

# The scored week is in the PAST and FINAL (reads do not gate on the window).
_PAST = timedelta(days=2)


def _walk(obj) -> list:
    """Flatten every dict key reachable in a nested dict/list structure."""
    keys: list = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(k)
            keys.extend(_walk(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_walk(item))
    return keys


class GetRecapContextTests(unittest.TestCase):
    """Service-level coverage for the recap-context reader."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            teams = [
                Team(espn_team_id=i, abbreviation=f"T{i}", display_name=f"Team {i}")
                for i in range(1, 5)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = [t.id for t in teams]

            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            self.week_id = week.id

            # Favorite (home, tid[0]) favored by 3.5, wins by 7 -> favorite COVERS.
            game_fav = Game(
                espn_event_id=1001,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[0],
                away_team_id=tid[1],
                kickoff_at=now - _PAST,
                status=GameStatus.FINAL,
                home_score=24,
                away_score=17,
                spread=Decimal("3.5"),
                total=Decimal("44.5"),
                favorite_team_id=tid[0],
                underdog_team_id=tid[1],
            )
            # Combined 20+14 = 34 < total 41 -> UNDER wins, OVER loses.
            game_total = Game(
                espn_event_id=1002,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid[2],
                away_team_id=tid[3],
                kickoff_at=now - _PAST + timedelta(hours=3),
                status=GameStatus.FINAL,
                home_score=20,
                away_score=14,
                spread=Decimal("6.5"),
                total=Decimal("41.0"),
                favorite_team_id=tid[2],
                underdog_team_id=tid[3],
            )
            session.add_all([game_fav, game_total])
            session.commit()
            session.refresh(game_fav)
            session.refresh(game_total)
            self.game_fav_id = game_fav.id
            self.game_total_id = game_total.id

            pw = hash_password("correct horse battery staple")
            # Distinct discord_ids: the one-null-discord_id invariant (260629-n59)
            # caps NULL discord_ids at one.
            user_a = User(display_name="alice", password_hash=pw, is_active=True, discord_id=1)
            user_b = User(display_name="bob", password_hash=pw, is_active=True, discord_id=2)
            session.add_all([user_a, user_b])
            session.commit()
            session.refresh(user_a)
            session.refresh(user_b)
            self.user_a_id = user_a.id
            self.user_b_id = user_b.id

            # alice nails both (favorite cover + under); bob misses both.
            session.add_all(
                [
                    Pick(
                        user_id=user_a.id,
                        game_id=game_fav.id,
                        week_id=week.id,
                        pick_type=PickType.FAVORITE_COVER,
                    ),
                    Pick(
                        user_id=user_a.id,
                        game_id=game_total.id,
                        week_id=week.id,
                        pick_type=PickType.UNDER,
                    ),
                    Pick(
                        user_id=user_b.id,
                        game_id=game_fav.id,
                        week_id=week.id,
                        pick_type=PickType.UNDERDOG_COVER,
                    ),
                    Pick(
                        user_id=user_b.id,
                        game_id=game_total.id,
                        week_id=week.id,
                        pick_type=PickType.OVER,
                    ),
                ]
            )
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return Session(self.engine)

    def test_returns_dict_with_expected_top_level_keys(self) -> None:
        with self._session() as session:
            ctx = get_recap_context(session, SEASON, WEEK)
        self.assertIsInstance(ctx, dict)
        self.assertEqual(set(ctx.keys()), {"week", "weekly_scores", "season_standings"})
        self.assertEqual(ctx["week"], WEEK)

    def test_weekly_scores_match_week_results_high_to_low(self) -> None:
        with self._session() as session:
            ctx = get_recap_context(session, SEASON, WEEK)
            expected = week_results(session, season=SEASON, week=WEEK)

        self.assertEqual(
            ctx["weekly_scores"],
            [
                {"display_name": r.display_name, "weekly_score": r.weekly_score}
                for r in expected
            ],
        )
        # Ordering is high->low (week_results already orders by -weekly_score).
        scores = [row["weekly_score"] for row in ctx["weekly_scores"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # alice (both right) leads bob (both wrong).
        self.assertEqual(ctx["weekly_scores"][0]["display_name"], "alice")
        self.assertGreater(
            ctx["weekly_scores"][0]["weekly_score"],
            ctx["weekly_scores"][-1]["weekly_score"],
        )

    def test_season_standings_carry_rank_and_gap_to_leader(self) -> None:
        with self._session() as session:
            ctx = get_recap_context(session, SEASON, WEEK)
            expected = season_standings(session, season=SEASON)[0].results

        standings = ctx["season_standings"]
        self.assertEqual(len(standings), len(expected))
        # Values + order match the service.
        for row, exp in zip(standings, expected):
            self.assertEqual(row["display_name"], exp.display_name)
            self.assertEqual(row["season_total"], exp.season_total)

        # rank is 1-based dense over the service ordering.
        self.assertEqual([r["rank"] for r in standings], list(range(1, len(standings) + 1)))
        # gap_to_leader: leader is 0, others are leader_total - own_total.
        leader_total = expected[0].season_total
        self.assertEqual(standings[0]["gap_to_leader"], 0)
        for row, exp in zip(standings, expected):
            self.assertEqual(row["gap_to_leader"], leader_total - exp.season_total)

    def test_output_is_display_only_no_user_id_anywhere(self) -> None:
        with self._session() as session:
            ctx = get_recap_context(session, SEASON, WEEK)
        self.assertNotIn("user_id", _walk(ctx))

    def test_empty_week_and_season_yield_empty_lists(self) -> None:
        with self._session() as session:
            ctx = get_recap_context(session, SEASON, week=99)  # no picks that week
            empty_season = get_recap_context(session, season=1999, week=1)

        self.assertEqual(ctx["weekly_scores"], [])
        self.assertEqual(empty_season["weekly_scores"], [])
        self.assertEqual(empty_season["season_standings"], [])


if __name__ == "__main__":
    unittest.main()
