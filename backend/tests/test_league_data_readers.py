"""Offline SQLite tests for the three app-data readers behind the open path's tools
(2026-09-18): ``get_league_picks`` (the ONE hard rule: no pick leaves while the week's
window is open), ``get_pick_completion`` (names, not a count) and
``get_standings_table``. Same seeding style as :mod:`tests.test_chat_context`.

Run with: ``backend/.venv/bin/python -m unittest tests.test_league_data_readers -v``
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
    get_league_picks,
    get_pick_completion,
    get_standings_table,
    get_week_scores,
)

SEASON = 2026
WEEK = 2


class _ReaderTestCase(unittest.TestCase):
    """Two games in one week; ``kickoff_offset`` decides whether the window is open."""

    kickoff_offset = timedelta(days=-1)  # closed by default

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        SQLModel.metadata.create_all(self.engine)
        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            teams = [
                Team(espn_team_id=1, abbreviation="BUF", display_name="Bills"),
                Team(espn_team_id=2, abbreviation="DET", display_name="Lions"),
                Team(espn_team_id=3, abbreviation="KC", display_name="Chiefs"),
                Team(espn_team_id=4, abbreviation="IND", display_name="Colts"),
            ]
            session.add_all(teams)
            session.commit()
            for t in teams:
                session.refresh(t)
            tid = {t.abbreviation: t.id for t in teams if t.id is not None}
            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            final = Game(
                espn_event_id=1,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid["BUF"],
                away_team_id=tid["DET"],
                kickoff_at=now + self.kickoff_offset,
                status=GameStatus.FINAL
                if self.kickoff_offset < timedelta(0)
                else GameStatus.SCHEDULED,
                home_score=41 if self.kickoff_offset < timedelta(0) else None,
                away_score=31 if self.kickoff_offset < timedelta(0) else None,
                spread=Decimal("5.5"),
                total=Decimal("51.5"),
                favorite_team_id=tid["BUF"],
                underdog_team_id=tid["DET"],
            )
            later = Game(
                espn_event_id=2,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=tid["KC"],
                away_team_id=tid["IND"],
                kickoff_at=now + self.kickoff_offset + timedelta(days=3),
                status=GameStatus.SCHEDULED,
                spread=Decimal("7.0"),
                total=Decimal("47.0"),
                favorite_team_id=tid["KC"],
                underdog_team_id=tid["IND"],
            )
            session.add_all([final, later])
            session.commit()
            session.refresh(final)
            session.refresh(later)
            pw = hash_password("correct horse battery staple")
            alice = User(display_name="alice", password_hash=pw, is_active=True, discord_id=1)
            bob = User(display_name="bob", password_hash=pw, is_active=True, discord_id=2)
            carol = User(display_name="carol", password_hash=pw, is_active=True, discord_id=3)
            root = User(
                display_name="root",
                password_hash=pw,
                is_active=True,
                is_protected=True,
                discord_id=None,
            )
            session.add_all([alice, bob, carol, root])
            session.commit()
            for u in (alice, bob, carol):
                session.refresh(u)
            assert alice.id and bob.id and final.id and later.id
            # alice: a FULL card (four base types + a mortal lock); bob: one pick.
            session.add_all(
                [
                    Pick(
                        user_id=alice.id,
                        game_id=final.id,
                        week_id=week.id,
                        pick_type=PickType.FAVORITE_COVER,
                        is_mortal_lock=True,
                    ),
                    Pick(
                        user_id=alice.id,
                        game_id=final.id,
                        week_id=week.id,
                        pick_type=PickType.FAVORITE_COVER,
                    ),
                    Pick(
                        user_id=alice.id,
                        game_id=later.id,
                        week_id=week.id,
                        pick_type=PickType.UNDERDOG_COVER,
                    ),
                    Pick(
                        user_id=alice.id, game_id=final.id, week_id=week.id, pick_type=PickType.OVER
                    ),
                    Pick(
                        user_id=alice.id,
                        game_id=later.id,
                        week_id=week.id,
                        pick_type=PickType.UNDER,
                    ),
                    Pick(
                        user_id=alice.id,
                        game_id=later.id,
                        week_id=week.id,
                        pick_type=PickType.MISC,
                        misc_text="Mahomes throws 4 TDs",
                    ),
                    Pick(
                        user_id=bob.id,
                        game_id=final.id,
                        week_id=week.id,
                        pick_type=PickType.UNDERDOG_COVER,
                        is_mortal_lock=True,
                    ),
                ]
            )
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return Session(self.engine)


class LeaguePicksWindowOpenTests(_ReaderTestCase):
    kickoff_offset = timedelta(days=1)

    def test_no_pick_leaves_while_the_window_is_open(self) -> None:
        with self._session() as session:
            out = get_league_picks(session, SEASON, WEEK)
        self.assertFalse(out["picks_locked"])
        self.assertIsNotNone(out["close_at"])
        self.assertEqual(sorted(m["display_name"] for m in out["members"]), ["alice", "bob"])
        for member in out["members"]:
            self.assertEqual(member["picks"], [])
        self.assertNotIn("Mahomes", str(out))
        self.assertNotIn("user_id", str(out))

    def test_completion_names_the_outstanding_members_while_open(self) -> None:
        with self._session() as session:
            out = get_pick_completion(session, SEASON, WEEK)
        self.assertTrue(out["pick_open"])
        self.assertEqual(out["complete"], ["alice"])
        # carol has no picks, bob has one; the protected root account is not in the pool.
        self.assertEqual(out["outstanding"], ["bob", "carol"])
        self.assertEqual(out["total_players"], 3)


class LeaguePicksWindowClosedTests(_ReaderTestCase):
    def test_every_pick_is_revealed_with_its_label_and_grade(self) -> None:
        with self._session() as session:
            out = get_league_picks(session, SEASON, WEEK)
        self.assertTrue(out["picks_locked"])
        by_name = {m["display_name"]: m for m in out["members"]}
        alice = by_name["alice"]
        labels = {p["pick"] for p in alice["picks"]}
        self.assertIn("BUF to cover as the favorite (5.5)", labels)
        self.assertIn("IND to cover as the underdog (7.0)", labels)
        self.assertIn("the over on DET at BUF (51.5)", labels)
        self.assertIn("the under on IND at KC (47.0)", labels)
        self.assertIn("misc call on IND at KC: Mahomes throws 4 TDs", labels)
        lock = next(p for p in alice["picks"] if p["mortal_lock"])
        self.assertEqual(lock["game"], "DET at BUF")
        # BUF won by 10 against a 5.5 spread: the favorite covered.
        self.assertEqual(lock["outcome"], "WIN")
        self.assertGreater(lock["points"], 0)
        # The later game is not final, so its picks are ungradeable.
        self.assertEqual(
            {p["outcome"] for p in alice["picks"] if p["game"] == "IND at KC"}, {"UNGRADEABLE"}
        )
        self.assertEqual(by_name["bob"]["picks"][0]["outcome"], "LOSS")
        # Ordered by weekly score, best first.
        self.assertEqual([m["display_name"] for m in out["members"]], ["alice", "bob"])
        self.assertNotIn("user_id", str(out))

    def test_completion_after_the_close_reads_as_missed_deadline(self) -> None:
        with self._session() as session:
            out = get_pick_completion(session, SEASON, WEEK)
        self.assertFalse(out["pick_open"])
        self.assertEqual(out["outstanding"], ["bob", "carol"])

    def test_standings_table_ranks_with_shared_ranks_on_ties(self) -> None:
        with self._session() as session:
            out = get_standings_table(session, SEASON)
        self.assertEqual(out["season"], SEASON)
        names = [e["display_name"] for e in out["entries"]]
        self.assertEqual(names, ["alice", "bob"])
        self.assertEqual(out["entries"][0]["rank"], 1)
        self.assertEqual(out["entries"][0]["last_week"], WEEK)
        self.assertEqual(out["entries"][0]["weeks_played"], 1)
        self.assertGreater(out["entries"][0]["season_total"], out["entries"][1]["season_total"])
        self.assertEqual(out["entries"][1]["rank"], 2)
        self.assertNotIn("user_id", str(out))

    def test_week_scores_narrow_to_the_asked_team(self) -> None:
        with self._session() as session:
            both = get_week_scores(session, SEASON, WEEK)
            bills = get_week_scores(session, SEASON, WEEK, team_abbr="BILLS")
            chiefs = get_week_scores(session, SEASON, WEEK, team_abbr="KC")
        self.assertEqual(len(both["games"]), 1)  # the KC game is not started
        self.assertEqual(bills["games"], both["games"])
        self.assertEqual(chiefs["games"], [])

    def test_an_empty_season_is_empty_not_a_raise(self) -> None:
        with self._session() as session:
            self.assertEqual(get_standings_table(session, 1999), {"season": 1999, "entries": []})
            self.assertEqual(get_league_picks(session, SEASON, 9)["members"], [])
            self.assertEqual(get_pick_completion(session, SEASON, 9)["complete"], [])


if __name__ == "__main__":
    unittest.main()
