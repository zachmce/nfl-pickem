"""DB-level ON DELETE CASCADE for pick_edit_audit's user FKs (offline).

Proves that deleting a User removes the ``PickEditAudit`` rows that reference them
— on BOTH user FKs (``admin_user_id`` and ``target_user_id``) — via the database
FK cascade. This REVERSES the prior "the audit survives the user / permanent
record" decision (per .planning/notes/admin-hardening-pre-stakeholder.md decision
6). The cascade is declared on the model (``PickEditAudit.admin_user_id`` /
``target_user_id`` ``ondelete="CASCADE"``, mirroring ``Pick.user_id``) and on the
Postgres path by migration 0013; this test exercises the MODEL declaration by
building the schema from ``SQLModel.metadata.create_all`` — so the same
``ondelete`` the app ships is what is enforced here.

POSTGRES NUANCE: SQLite tests use ``create_all`` and never run Alembic, so the
live Postgres cascade is enforced by migration 0013's FK drop/recreate (which must
be applied via ``alembic upgrade head``). This test covers the model-level
``ondelete`` under the SQLite FK pragma; 0013 covers the migrated Postgres schema.

CRITICAL: SQLite does NOT enforce foreign keys (or cascades) unless
``PRAGMA foreign_keys = ON`` is issued per DBAPI connection. Without the connect
event listener below (registered BEFORE ``create_all``), SQLite would silently
ignore the FK and the cascade, so the cascade tests would FALSELY PASS even if the
cascade were missing. The listener is what makes this a REAL cascade test.

Fully offline: in-memory SQLite, no Postgres, no network.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
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
    PickEditAudit,
    PickType,
    Team,
    User,
    Week,
)

SEASON = 2025
WEEK = 1


class PickEditAuditCascadeTests(unittest.TestCase):
    """Real DB-level cascade: deleting a user removes the audit rows for them."""

    admin_id: int
    target_id: int
    week_id: int
    game_id: int
    audit_id: int

    def setUp(self) -> None:
        # One shared in-memory connection (StaticPool) so every Session sees the
        # SAME database — mirrors test_pick_user_cascade.py.
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # CRITICAL: turn FK enforcement ON for every new DBAPI connection BEFORE
        # create_all — otherwise SQLite ignores the cascade and these tests would
        # falsely pass with no cascade declared.
        @event.listens_for(self.engine, "connect")
        def _enable_sqlite_fks(dbapi_connection, _connection_record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        SQLModel.metadata.create_all(self.engine)

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            # Two teams (FK targets for game home/away).
            team_home = Team(espn_team_id=1, abbreviation="HOM", display_name="Home")
            team_away = Team(espn_team_id=2, abbreviation="AWY", display_name="Away")
            session.add_all([team_home, team_away])
            session.commit()
            session.refresh(team_home)
            session.refresh(team_away)

            week = Week(season=SEASON, week=WEEK)
            session.add(week)
            session.commit()
            session.refresh(week)
            assert week.id is not None
            self.week_id = week.id

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

            # Admin + target users. Distinct discord_ids: the one-null-discord_id
            # invariant (260629-n59) caps NULL discord_ids at one.
            admin = User(display_name="admin", is_admin=True, is_active=True, discord_id=1)
            target = User(display_name="target", is_active=True, discord_id=2)
            session.add_all([admin, target])
            session.commit()
            session.refresh(admin)
            session.refresh(target)
            assert admin.id is not None and target.id is not None
            self.admin_id = admin.id
            self.target_id = target.id

            audit = PickEditAudit(
                admin_user_id=admin.id,
                target_user_id=target.id,
                game_id=game.id,
                week_id=week.id,
                action="set",
                before_existed=False,
                after_pick_type=PickType.FAVORITE_COVER,
                after_is_mortal_lock=False,
                game_was_final=False,
            )
            session.add(audit)
            session.commit()
            session.refresh(audit)
            assert audit.id is not None
            self.audit_id = audit.id

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_delete_target_user_cascades_audit(self) -> None:
        """Deleting the TARGET user removes the audit row (target_user_id FK)."""
        with Session(self.engine) as session:
            target = session.get(User, self.target_id)
            assert target is not None
            session.delete(target)
            session.commit()  # must NOT raise a FK violation

            remaining = session.exec(select(PickEditAudit)).all()
            self.assertEqual(
                remaining, [], "the target's audit row should cascade away"
            )
            self.assertIsNone(session.get(User, self.target_id))

    def test_delete_admin_user_cascades_audit(self) -> None:
        """Deleting the ACTING ADMIN removes the audit row (admin_user_id FK)."""
        with Session(self.engine) as session:
            admin = session.get(User, self.admin_id)
            assert admin is not None
            session.delete(admin)
            session.commit()  # must NOT raise a FK violation

            remaining = session.exec(select(PickEditAudit)).all()
            self.assertEqual(
                remaining, [], "the admin's audit row should cascade away"
            )
            self.assertIsNone(session.get(User, self.admin_id))


if __name__ == "__main__":
    unittest.main()
