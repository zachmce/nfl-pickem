"""DB-level ON DELETE CASCADE for pick.user_id (offline, real FK enforcement).

Proves that deleting a User removes that user's Picks via the database FK
cascade — the foundation for the admin "delete user" action (locked decision 4
in .planning/notes/admin-area-design.md). The cascade is declared on the model
(``Pick.user_id`` ``ondelete="CASCADE"``) and in migration 0006; this test
exercises the MODEL declaration by building the schema from
``SQLModel.metadata.create_all`` — so the same ``ondelete`` the app ships is what
is enforced here.

CRITICAL: SQLite does NOT enforce foreign keys (or cascades) unless
``PRAGMA foreign_keys = ON`` is issued per DBAPI connection. Without the connect
event listener below, SQLite would silently ignore the FK and the cascade, so
test 1 would FALSELY PASS even if the cascade were missing (the orphan picks
would simply remain, and a no-cascade FK would not even error). The listener is
what makes this a REAL cascade test.

Fully offline: in-memory SQLite, no Postgres, no network.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); run from repo root with
> ``backend/.venv/bin/python -m unittest`` (or from backend/ with
> ``.venv/bin/python -m unittest``). This repo uses unittest, NOT pytest.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import (
    Game,
    GameStatus,
    Pick,
    PickType,
    Team,
    User,
    Week,
)

SEASON = 2025
WEEK = 1


class PickUserCascadeTests(unittest.TestCase):
    """Real DB-level cascade: deleting a user removes that user's picks."""

    user_a_id: int
    user_b_id: int
    week_id: int
    game_id: int

    def setUp(self) -> None:
        # One shared in-memory connection (StaticPool) so every Session sees the
        # SAME database — mirrors the offline harness in test_picks_api.py.
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # CRITICAL: turn FK enforcement ON for every new DBAPI connection. SQLite
        # ignores foreign keys (and cascades) by default — without this PRAGMA the
        # cascade test would FALSELY PASS even with no cascade declared. Register
        # BEFORE create_all so the schema-building connection is covered too.
        @event.listens_for(self.engine, "connect")
        def _enable_sqlite_fks(dbapi_connection, _connection_record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            # --- Two teams (FK targets for game home/away) -------------------
            team_home = Team(espn_team_id=1, abbreviation="HOM", display_name="Home")
            team_away = Team(espn_team_id=2, abbreviation="AWY", display_name="Away")
            session.add_all([team_home, team_away])
            session.commit()
            session.refresh(team_home)
            session.refresh(team_away)

            # --- Week --------------------------------------------------------
            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            self.week_id = week.id

            # --- Game (satisfy NOT NULL FKs + columns) -----------------------
            game = Game(
                espn_event_id=1001,
                week_id=week.id,
                season=SEASON,
                week=WEEK,
                home_team_id=team_home.id,
                away_team_id=team_away.id,
                kickoff_at=now,
                status=GameStatus.SCHEDULED,
            )
            session.add(game)
            session.commit()
            session.refresh(game)
            assert game.id is not None
            self.game_id = game.id

            # --- Two users (password_hash None is valid per the model) -------
            user_a = User(display_name="userA", is_active=True)
            user_b = User(display_name="userB", is_active=True)
            session.add_all([user_a, user_b])
            session.commit()
            session.refresh(user_a)
            session.refresh(user_b)
            assert user_a.id is not None and user_b.id is not None
            self.user_a_id = user_a.id
            self.user_b_id = user_b.id

            # --- Picks: one for each user (distinct so unique indexes are OK) -
            session.add_all(
                [
                    Pick(
                        user_id=user_a.id,
                        game_id=game.id,
                        week_id=week.id,
                        pick_type=PickType.UNDERDOG_COVER,
                    ),
                    Pick(
                        user_id=user_b.id,
                        game_id=game.id,
                        week_id=week.id,
                        pick_type=PickType.OVER,
                    ),
                ]
            )
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_delete_user_with_picks_cascades(self) -> None:
        """Deleting a user with picks succeeds and removes that user's picks."""
        with Session(self.engine) as session:
            user_a = session.get(User, self.user_a_id)
            assert user_a is not None
            session.delete(user_a)
            session.commit()  # must NOT raise a FK violation

            orphans = session.exec(
                select(Pick).where(Pick.user_id == self.user_a_id)
            ).all()
            self.assertEqual(orphans, [], "user_a's picks should be cascade-deleted")
            self.assertIsNone(session.get(User, self.user_a_id))

    def test_delete_user_without_picks_still_deletes(self) -> None:
        """A pick-less user deletes cleanly, affecting nothing else."""
        with Session(self.engine) as session:
            user_c = User(display_name="userC", is_active=True)
            session.add(user_c)
            session.commit()
            session.refresh(user_c)
            user_c_id = user_c.id

            session.delete(user_c)
            session.commit()  # must NOT raise

            self.assertIsNone(session.get(User, user_c_id))
            # Existing picks for the other users are untouched.
            self.assertEqual(
                len(session.exec(select(Pick)).all()),
                2,
                "deleting a pick-less user must not affect other picks",
            )

    def test_unrelated_users_picks_untouched(self) -> None:
        """Deleting user_a leaves user_b's pick(s) intact and unchanged."""
        with Session(self.engine) as session:
            before = session.exec(
                select(Pick).where(Pick.user_id == self.user_b_id)
            ).all()
            before_ids = sorted(p.id for p in before)
            self.assertTrue(before_ids, "fixture should give user_b a pick")

            user_a = session.get(User, self.user_a_id)
            session.delete(user_a)
            session.commit()

            after = session.exec(
                select(Pick).where(Pick.user_id == self.user_b_id)
            ).all()
            after_ids = sorted(p.id for p in after)
            self.assertEqual(after_ids, before_ids, "user_b's picks must be untouched")


if __name__ == "__main__":
    unittest.main()
