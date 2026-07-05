"""Offline tests for the week.recap "closing ceremony" context reader (260705-kuv).

These mirror :mod:`tests.test_recap_context`'s setUp: a single shared in-memory
SQLite connection (``StaticPool``), Teams + Weeks + FINAL Games with real
scores/spreads + Users + Picks seeded directly (reads do not gate on the window).
They pin the enriched recap contract that feeds the marquee Discord embed:

* :func:`app.services.notifications_read.get_week_recap_context` returns a plain
  display-only dict ``{standings, best_call, biggest_bust, mortal_locks}`` built
  over the EXISTING ``get_recap_context`` + ``grade_pick`` — no scoring/standings
  math re-implemented.
* ``standings`` rows carry ``rank`` / ``display_name`` / ``season_total`` and a
  ``week_delta`` matching ``week_results``' weekly score.
* ``best_call`` selects the largest-``Game.spread`` UNDERDOG_COVER win; ``biggest_bust``
  the largest-spread FAVORITE_COVER loss with a mortal-lock breaking a spread tie.
* ``mortal_locks`` reports hit/miss + points from ``grade_pick``, omitting players
  with no lock.
* The whole structure is DISPLAY-ONLY: no ``user_id`` key ever appears (T-kuv-01).
* An empty/ambiguous week or season yields the all-empty shape (no raise).

Run with: ``backend/.venv/bin/python -m unittest tests.test_week_recap_context -v``
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
from app.services.notifications_read import get_week_recap_context
from app.services.standings import season_standings, week_results

SEASON = 2025
WEEK = 1
# A second FINAL week where every pick is a PUSH / non-lock, so best_call /
# biggest_bust are None and the board is [] even though FINAL games exist.
QUIET_WEEK = 2

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


class GetWeekRecapContextTests(unittest.TestCase):
    """Service-level coverage for the enriched week.recap context reader."""

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
                for i in range(1, 9)
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = {t.abbreviation: t.id for t in teams if t.id is not None}
            self.tid = tid

            pw = hash_password("correct horse battery staple")
            alice = User(display_name="alice", password_hash=pw, is_active=True, discord_id=1)
            bob = User(display_name="bob", password_hash=pw, is_active=True, discord_id=2)
            carol = User(display_name="carol", password_hash=pw, is_active=True, discord_id=3)
            session.add_all([alice, bob, carol])
            session.commit()
            for u in (alice, bob, carol):
                session.refresh(u)
            assert alice.id is not None and bob.id is not None and carol.id is not None

            def _game(
                *,
                week_id: int,
                week: int,
                eid: int,
                home: str,
                away: str,
                home_score: int,
                away_score: int,
                spread: str,
                offset_h: int,
            ) -> Game:
                # Favorite is always the HOME team here; underdog the away team.
                return Game(
                    espn_event_id=eid,
                    week_id=week_id,
                    season=SEASON,
                    week=week,
                    home_team_id=tid[home],
                    away_team_id=tid[away],
                    kickoff_at=now - _PAST + timedelta(hours=offset_h),
                    status=GameStatus.FINAL,
                    home_score=home_score,
                    away_score=away_score,
                    spread=Decimal(spread),
                    total=Decimal("40.0"),
                    favorite_team_id=tid[home],
                    underdog_team_id=tid[away],
                )

            # --- Week 1: the rich fixture. ---
            week1 = Week(season=SEASON, week=WEEK)
            session.add(week1)
            session.commit()
            session.refresh(week1)
            assert week1.id is not None

            # Big underdog win (spread 10): home favorite loses 10-20 -> UNDERDOG covers.
            game_big_dog = _game(
                week_id=week1.id,
                week=WEEK,
                eid=101,
                home="T1",
                away="T2",
                home_score=10,
                away_score=20,
                spread="10.0",
                offset_h=0,
            )
            # Small underdog win (spread 3.5): home fav loses 14-17 -> UNDERDOG covers.
            game_small_dog = _game(
                week_id=week1.id,
                week=WEEK,
                eid=102,
                home="T3",
                away="T4",
                home_score=14,
                away_score=17,
                spread="3.5",
                offset_h=3,
            )
            # Big favorite bust (spread 7.5): home fav wins by 4 (< 7.5) -> FAVORITE loses.
            game_big_bust = _game(
                week_id=week1.id,
                week=WEEK,
                eid=103,
                home="T5",
                away="T6",
                home_score=24,
                away_score=20,
                spread="7.5",
                offset_h=6,
            )
            # Tie-spread favorite bust (spread 7.5): home fav wins by 1 -> FAVORITE loses.
            game_tie_bust = _game(
                week_id=week1.id,
                week=WEEK,
                eid=104,
                home="T7",
                away="T8",
                home_score=21,
                away_score=20,
                spread="7.5",
                offset_h=9,
            )
            session.add_all([game_big_dog, game_small_dog, game_big_bust, game_tie_bust])
            session.commit()
            for g in (game_big_dog, game_small_dog, game_big_bust, game_tie_bust):
                session.refresh(g)
            assert (
                game_big_dog.id is not None
                and game_small_dog.id is not None
                and game_big_bust.id is not None
                and game_tie_bust.id is not None
            )

            session.add_all(
                [
                    # alice: the largest-spread underdog win (best_call) ...
                    Pick(
                        user_id=alice.id,
                        game_id=game_big_dog.id,
                        week_id=week1.id,
                        pick_type=PickType.UNDERDOG_COVER,
                    ),
                    # ... and her mortal lock BUSTS on a spread-7.5 favorite (amplified bust).
                    Pick(
                        user_id=alice.id,
                        game_id=game_tie_bust.id,
                        week_id=week1.id,
                        pick_type=PickType.FAVORITE_COVER,
                        is_mortal_lock=True,
                    ),
                    # bob: a mortal-lock underdog win at a SMALLER spread (loses best_call
                    # to alice on magnitude despite the lock) ...
                    Pick(
                        user_id=bob.id,
                        game_id=game_small_dog.id,
                        week_id=week1.id,
                        pick_type=PickType.UNDERDOG_COVER,
                        is_mortal_lock=True,
                    ),
                    # ... and a NON-lock favorite bust at spread 7.5 (ties alice's spread,
                    # but her lock wins the amplification tie-break).
                    Pick(
                        user_id=bob.id,
                        game_id=game_big_bust.id,
                        week_id=week1.id,
                        pick_type=PickType.FAVORITE_COVER,
                    ),
                    # carol: a small favorite bust, NO mortal lock (omitted from board).
                    Pick(
                        user_id=carol.id,
                        game_id=game_small_dog.id,
                        week_id=week1.id,
                        pick_type=PickType.FAVORITE_COVER,
                    ),
                ]
            )
            session.commit()

            # --- Week 2: FINAL game but nothing qualifies (a PUSH, no lock). ---
            week2 = Week(season=SEASON, week=QUIET_WEEK)
            session.add(week2)
            session.commit()
            session.refresh(week2)
            assert week2.id is not None
            # Home favorite wins by EXACTLY the spread -> PUSH (not a win or a loss).
            game_push = _game(
                week_id=week2.id,
                week=QUIET_WEEK,
                eid=201,
                home="T1",
                away="T2",
                home_score=20,
                away_score=17,
                spread="3.0",
                offset_h=0,
            )
            session.add(game_push)
            session.commit()
            session.refresh(game_push)
            assert game_push.id is not None
            session.add(
                Pick(
                    user_id=alice.id,
                    game_id=game_push.id,
                    week_id=week2.id,
                    pick_type=PickType.FAVORITE_COVER,
                )
            )
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return Session(self.engine)

    # ------------------------------------------------------------------ #
    # Shape + standings                                                  #
    # ------------------------------------------------------------------ #

    def test_returns_expected_top_level_keys(self) -> None:
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, WEEK)
        self.assertEqual(
            set(ctx.keys()), {"standings", "best_call", "biggest_bust", "mortal_locks"}
        )

    def test_standings_carry_rank_name_total_and_week_delta(self) -> None:
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, WEEK)
            expected_standings = season_standings(session, season=SEASON)[0].results
            week_rows = week_results(session, season=SEASON, week=WEEK)

        weekly_by_name = {r.display_name: r.weekly_score for r in week_rows}
        standings = ctx["standings"]
        self.assertEqual(len(standings), len(expected_standings))
        # rank is 1-based dense over the season_standings ordering.
        self.assertEqual([r["rank"] for r in standings], list(range(1, len(standings) + 1)))
        for row, exp in zip(standings, expected_standings):
            self.assertEqual(row["display_name"], exp.display_name)
            self.assertEqual(row["season_total"], exp.season_total)
            # week_delta == that player's weekly_score this week (0 when absent).
            self.assertEqual(row["week_delta"], weekly_by_name.get(exp.display_name, 0))
        # bob (mortal-lock underdog win = +2) leads the week and the season.
        self.assertEqual(standings[0]["display_name"], "bob")
        self.assertEqual(standings[0]["week_delta"], 2)

    # ------------------------------------------------------------------ #
    # best_call / biggest_bust upset-magnitude ranking                   #
    # ------------------------------------------------------------------ #

    def test_best_call_is_largest_spread_underdog_win(self) -> None:
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, WEEK)
        best = ctx["best_call"]
        self.assertIsNotNone(best)
        assert best is not None
        # alice's spread-10 underdog win beats bob's spread-3.5 mortal-lock win.
        self.assertEqual(best["display_name"], "alice")
        self.assertEqual(best["spread"], "10.0")
        self.assertFalse(best["is_mortal_lock"])
        self.assertEqual(best["team_abbr"], "T2")  # the underdog (away) team
        self.assertIn("Underdog", best["side_label"])

    def test_biggest_bust_amplifies_mortal_lock_on_spread_tie(self) -> None:
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, WEEK)
        bust = ctx["biggest_bust"]
        self.assertIsNotNone(bust)
        assert bust is not None
        # alice + bob both bust a spread-7.5 favorite; alice's mortal lock wins the tie.
        self.assertEqual(bust["display_name"], "alice")
        self.assertEqual(bust["spread"], "7.5")
        self.assertTrue(bust["is_mortal_lock"])
        self.assertEqual(bust["team_abbr"], "T7")  # the favorite (home) team
        self.assertIn("Favorite", bust["side_label"])

    # ------------------------------------------------------------------ #
    # mortal-lock board                                                  #
    # ------------------------------------------------------------------ #

    def test_mortal_lock_board_matches_grade_pick_and_omits_lockless(self) -> None:
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, WEEK)
        board = ctx["mortal_locks"]
        # alice (miss) + bob (hit); carol used no mortal lock -> omitted.
        names = [row["display_name"] for row in board]
        self.assertEqual(names, ["alice", "bob"])  # sorted by display_name
        by_name = {row["display_name"]: row for row in board}
        # alice: busted FAVORITE_COVER mortal lock -> hit False, points -1.
        self.assertFalse(by_name["alice"]["hit"])
        self.assertEqual(by_name["alice"]["points"], -1)
        # bob: hit UNDERDOG_COVER mortal lock -> hit True, points +2.
        self.assertTrue(by_name["bob"]["hit"])
        self.assertEqual(by_name["bob"]["points"], 2)

    # ------------------------------------------------------------------ #
    # omit-empty + display-only + empty/ambiguous                        #
    # ------------------------------------------------------------------ #

    def test_quiet_week_yields_none_blocks_and_empty_board(self) -> None:
        # FINAL games exist but every pick is a PUSH / non-lock -> nothing qualifies.
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, QUIET_WEEK)
        self.assertIsNone(ctx["best_call"])
        self.assertIsNone(ctx["biggest_bust"])
        self.assertEqual(ctx["mortal_locks"], [])
        # standings still populate (season-cumulative), so the card is never blank.
        self.assertTrue(ctx["standings"])

    def test_output_is_display_only_no_user_id_anywhere(self) -> None:
        with self._session() as session:
            ctx = get_week_recap_context(session, SEASON, WEEK)
        self.assertNotIn("user_id", _walk(ctx))

    def test_empty_week_and_ambiguous_season_yield_all_empty_shape(self) -> None:
        with self._session() as session:
            no_games_week = get_week_recap_context(session, SEASON, week=99)
            empty_season = get_week_recap_context(session, season=1999, week=1)

        # A week with no games: no upset blocks, but season standings still present.
        self.assertIsNone(no_games_week["best_call"])
        self.assertIsNone(no_games_week["biggest_bust"])
        self.assertEqual(no_games_week["mortal_locks"], [])
        # An ambiguous season yields the fully-empty shape (never raises).
        self.assertEqual(
            empty_season,
            {"standings": [], "best_call": None, "biggest_bust": None, "mortal_locks": []},
        )


if __name__ == "__main__":
    unittest.main()
