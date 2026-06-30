"""Offline tests for the three embellished-chat context readers (260627-vpc).

These mirror :mod:`tests.test_recap_context`'s setUp: one shared in-memory SQLite
connection (``StaticPool``), Teams + a Week + FINAL Games with real scores/odds +
Users + Picks seeded directly (reads do not gate on the window). They pin the
enriched-chat contract for the three embellished events:

* :func:`app.services.notifications_read.get_game_final_context` resolves THE game
  by ``(season, week, away_abbr, home_abbr)`` and returns a display-only dict
  carrying the abbrs + scores, the spread result (favorite + frozen spread +
  did_cover), the total result (frozen total + went_over), and ``pick_impacts``
  graded via the EXISTING scoring engine — NO cover/over-under math re-implemented.
* :func:`app.services.notifications_read.get_roster_complete_context` returns the
  actor's public rank + season total and the week completion as a COUNT — NEVER
  the names of who is outstanding and NEVER any pick content.
* :func:`app.services.notifications_read.get_leaders_context` returns the season
  leader (+ runner-up + gap) from the top of ``season_standings``.
* Every returned dict is DISPLAY-ONLY: no ``user_id`` key ever appears.
* Unknown/ambiguous abbrs + empty season/week yield graceful empties (no raise).

Run with: ``backend/.venv/bin/python -m unittest tests.test_chat_context -v``
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
from app.services.notifications_read import (
    get_game_final_context,
    get_leaders_context,
    get_roster_complete_context,
)
from app.services.scoring import GradeOutcome, grade_pick
from app.services.standings import season_standings

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


class ChatContextTests(unittest.TestCase):
    """Service-level coverage for the three embellished-chat context readers."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            # Teams T1..T4 with friendly abbreviations KC/LAC/SF/SEA.
            teams = [
                Team(espn_team_id=1, abbreviation="KC", display_name="Chiefs"),
                Team(espn_team_id=2, abbreviation="LAC", display_name="Chargers"),
                Team(espn_team_id=3, abbreviation="SF", display_name="49ers"),
                Team(espn_team_id=4, abbreviation="SEA", display_name="Seahawks"),
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = {t.abbreviation: t.id for t in teams}
            self.tid = tid

            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            self.week_id = week.id

            # game_fav: KC (home, favorite by 3.5) beats LAC 24-17 -> favorite
            # margin 7 > 3.5 -> favorite COVERS. Combined 41 vs total 44.5 -> UNDER.
            game_fav = Game(
                espn_event_id=1001,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid["KC"],
                away_team_id=tid["LAC"],
                kickoff_at=now - _PAST,
                status=GameStatus.FINAL,
                home_score=24,
                away_score=17,
                spread=Decimal("3.5"),
                total=Decimal("44.5"),
                favorite_team_id=tid["KC"],
                underdog_team_id=tid["LAC"],
            )
            # game_total: SF (home, fav by 6.5) wins 20-14. Combined 34 < 41 ->
            # UNDER wins, OVER loses. Favorite margin 6 < 6.5 -> favorite did NOT
            # cover (underdog covers).
            game_total = Game(
                espn_event_id=1002,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid["SF"],
                away_team_id=tid["SEA"],
                kickoff_at=now - _PAST + timedelta(hours=3),
                status=GameStatus.FINAL,
                home_score=20,
                away_score=14,
                spread=Decimal("6.5"),
                total=Decimal("41.0"),
                favorite_team_id=tid["SF"],
                underdog_team_id=tid["SEA"],
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

            # alice nails the KC game as a MORTAL LOCK favorite-cover (dramatic
            # hit); bob takes the underdog there as a mortal lock (dramatic bust).
            # On game_total alice is UNDER (win), bob OVER (loss).
            session.add_all(
                [
                    Pick(
                        user_id=user_a.id,
                        game_id=game_fav.id,
                        week_id=week.id,
                        pick_type=PickType.FAVORITE_COVER,
                        is_mortal_lock=True,
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
                        is_mortal_lock=True,
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

    # ---- get_game_final_context ------------------------------------------ #

    def test_game_final_resolves_game_and_echoes_scores(self) -> None:
        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="LAC", home_abbr="KC")
        self.assertTrue(ctx["found"])
        self.assertEqual(ctx["away"], "LAC")
        self.assertEqual(ctx["home"], "KC")
        self.assertEqual(ctx["away_score"], 17)
        self.assertEqual(ctx["home_score"], 24)

    def test_game_final_spread_result_matches_scoring(self) -> None:
        # KC favored by 3.5 wins by 7 -> favorite covers (per scoring._spread_outcome).
        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="LAC", home_abbr="KC")
        spread = ctx["spread_result"]
        self.assertIsNotNone(spread)
        self.assertEqual(spread["favorite_abbr"], "KC")
        self.assertEqual(Decimal(str(spread["spread"])), Decimal("3.5"))
        self.assertTrue(spread["did_cover"])

    def test_game_final_underdog_cover_is_did_cover_false(self) -> None:
        # SF favored by 6.5 wins by 6 -> favorite did NOT cover.
        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="SEA", home_abbr="SF")
        spread = ctx["spread_result"]
        self.assertEqual(spread["favorite_abbr"], "SF")
        self.assertFalse(spread["did_cover"])

    def test_game_final_total_result_matches_scoring(self) -> None:
        # KC game combined 41 vs total 44.5 -> UNDER (went_over False).
        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="LAC", home_abbr="KC")
        total = ctx["total_result"]
        self.assertIsNotNone(total)
        self.assertEqual(Decimal(str(total["total"])), Decimal("44.5"))
        self.assertFalse(total["went_over"])

    def test_game_final_pick_impacts_grade_this_game(self) -> None:
        # alice's mortal-lock FAVORITE_COVER on KC HITS; bob's mortal-lock
        # UNDERDOG_COVER BUSTS — both must surface as impacts, graded via the
        # scoring engine, named by display_name only.
        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="LAC", home_abbr="KC")
        impacts = ctx["pick_impacts"]
        by_name = {i["display_name"]: i for i in impacts}
        self.assertIn("alice", by_name)
        self.assertIn("bob", by_name)

        self.assertTrue(by_name["alice"]["is_mortal_lock"])
        self.assertEqual(by_name["alice"]["outcome"], GradeOutcome.WIN.value)
        self.assertTrue(by_name["bob"]["is_mortal_lock"])
        self.assertEqual(by_name["bob"]["outcome"], GradeOutcome.LOSS.value)

        # side_label uses the favorite/underdog convention, never raw enum names.
        self.assertIn("KC", by_name["alice"]["side_label"])
        self.assertIn("LAC", by_name["bob"]["side_label"])

    def test_game_final_pick_impacts_match_grade_pick(self) -> None:
        # The reader must NOT re-derive outcomes — each impact's outcome must equal
        # what grade_pick returns for the SAME pick/game.
        from sqlmodel import select

        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="LAC", home_abbr="KC")
            game = session.get(Game, self.game_fav_id)
            orm_picks = session.exec(select(Pick).where(Pick.game_id == self.game_fav_id)).all()
            names = {self.user_a_id: "alice", self.user_b_id: "bob"}
            expected = {names[p.user_id]: grade_pick(game, p).outcome.value for p in orm_picks}

        for impact in ctx["pick_impacts"]:
            self.assertEqual(impact["outcome"], expected[impact["display_name"]])

    def test_game_final_unknown_abbrs_return_not_found(self) -> None:
        with self._session() as session:
            ctx = get_game_final_context(session, SEASON, WEEK, away_abbr="ZZZ", home_abbr="YYY")
        self.assertFalse(ctx["found"])
        self.assertIsNone(ctx["spread_result"])
        self.assertIsNone(ctx["total_result"])
        self.assertEqual(ctx["pick_impacts"], [])

    def test_game_final_empty_season_does_not_raise(self) -> None:
        with self._session() as session:
            ctx = get_game_final_context(
                session, season=1999, week=1, away_abbr="LAC", home_abbr="KC"
            )
        self.assertFalse(ctx["found"])

    # ---- get_roster_complete_context ------------------------------------- #

    def test_roster_complete_reports_rank_and_total(self) -> None:
        with self._session() as session:
            ctx = get_roster_complete_context(session, SEASON, WEEK, actor="alice")
            standings = season_standings(session, season=SEASON)[0].results
        by_name = {r.display_name: r for r in standings}
        self.assertEqual(ctx["actor"], "alice")
        self.assertEqual(ctx["season_total"], by_name["alice"].season_total)
        self.assertIsNotNone(ctx["rank"])

    def test_roster_complete_reports_counts_only(self) -> None:
        with self._session() as session:
            ctx = get_roster_complete_context(session, SEASON, WEEK, actor="alice")
        # Pool is the 2 players who have any pick this season. Neither holds a full
        # standard card in this seed (each has a mortal lock + one base pick, far
        # short of all four base types), so the completion count is 0 and both are
        # outstanding.
        self.assertEqual(ctx["total_players"], 2)
        self.assertEqual(ctx["completed_count"], 0)
        self.assertEqual(ctx["outstanding_count"], 2)
        self.assertEqual(ctx["outstanding_count"], ctx["total_players"] - ctx["completed_count"])

    def test_roster_complete_counts_a_full_roster(self) -> None:
        # Give bob all four BASE slots; with his existing mortal lock that makes a
        # full standard card, so he should then count as completed.
        from sqlmodel import select

        with self._session() as session:
            bob = session.exec(select(User).where(User.display_name == "bob")).one()
            # bob already has UNDERDOG_COVER(ML) on KC + OVER on SF — the mortal
            # lock that the full-standard-card predicate also requires. Add the
            # three missing base types so his base set is {FAV, UNDERDOG, OVER,
            # UNDER} and, with the mortal lock, the card is complete.
            session.add_all(
                [
                    Pick(
                        user_id=bob.id,
                        game_id=self.game_fav_id,
                        week_id=self.week_id,
                        pick_type=PickType.FAVORITE_COVER,
                    ),
                    Pick(
                        user_id=bob.id,
                        game_id=self.game_fav_id,
                        week_id=self.week_id,
                        pick_type=PickType.UNDERDOG_COVER,
                    ),
                    Pick(
                        user_id=bob.id,
                        game_id=self.game_total_id,
                        week_id=self.week_id,
                        pick_type=PickType.UNDER,
                    ),
                ]
            )
            session.commit()
            ctx = get_roster_complete_context(session, SEASON, WEEK, actor="bob")

        self.assertEqual(ctx["total_players"], 2)
        self.assertEqual(ctx["completed_count"], 1)
        self.assertEqual(ctx["outstanding_count"], 1)

    def test_roster_complete_absent_actor_has_none_rank(self) -> None:
        with self._session() as session:
            ctx = get_roster_complete_context(session, SEASON, WEEK, actor="nobody")
        self.assertIsNone(ctx["rank"])

    def test_roster_complete_carries_no_names_of_outstanding(self) -> None:
        # The ONLY player-identifying field is the actor; no other display_name
        # (which could name an outstanding player) may appear anywhere.
        with self._session() as session:
            ctx = get_roster_complete_context(session, SEASON, WEEK, actor="alice")
        # bob is the other player; the count-only contract means bob is never named.
        self.assertNotIn("bob", str(ctx))
        # No pick-content keys leak either.
        keys = set(_walk(ctx))
        for forbidden in ("pick", "side", "pick_impacts", "outstanding_names"):
            self.assertNotIn(forbidden, keys)

    # ---- get_leaders_context --------------------------------------------- #

    def test_leaders_reports_leader_and_runner_up(self) -> None:
        with self._session() as session:
            ctx = get_leaders_context(session, SEASON)
            standings = season_standings(session, season=SEASON)[0].results
        self.assertEqual(ctx["leader"], standings[0].display_name)
        self.assertEqual(ctx["leader_total"], standings[0].season_total)
        self.assertEqual(ctx["runner_up"], standings[1].display_name)
        self.assertEqual(ctx["runner_up_total"], standings[1].season_total)
        self.assertEqual(ctx["gap"], standings[0].season_total - standings[1].season_total)

    def test_leaders_empty_season_has_none_leader(self) -> None:
        with self._session() as session:
            ctx = get_leaders_context(session, season=1999)
        self.assertIsNone(ctx["leader"])
        self.assertIsNone(ctx["runner_up"])
        self.assertIsNone(ctx["gap"])

    # ---- display-only boundary ------------------------------------------- #

    def test_all_contexts_have_no_user_id_key(self) -> None:
        with self._session() as session:
            game_ctx = get_game_final_context(
                session, SEASON, WEEK, away_abbr="LAC", home_abbr="KC"
            )
            roster_ctx = get_roster_complete_context(session, SEASON, WEEK, actor="alice")
            leaders_ctx = get_leaders_context(session, SEASON)
        self.assertNotIn("user_id", _walk(game_ctx))
        self.assertNotIn("user_id", _walk(roster_ctx))
        self.assertNotIn("user_id", _walk(leaders_ctx))


if __name__ == "__main__":
    unittest.main()
