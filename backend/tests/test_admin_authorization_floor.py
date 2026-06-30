"""Offline unit tests for the admin authorization floor (quick task 260629-n59).

Fully OFFLINE (in-memory SQLite via ``StaticPool``, no Postgres, no network),
mirroring the harness of :mod:`tests.test_admin_api`: a ``StaticPool`` engine, a
``@event.listens_for(engine, "connect")`` ``PRAGMA foreign_keys=ON`` handler
registered BEFORE ``create_all`` (so the schema-building connection is covered),
and ``SQLModel.metadata.create_all``.

Proves the four locked invariants of the admin authorization floor:

* DECISION-1 — at most ONE NULL ``discord_id`` (the model-level partial unique
  index ``uq_users_one_null_discord_id`` is emitted by ``create_all`` so the
  SQLite test DB enforces it; the 0012 migration mirrors it on Postgres). A
  second null insert raises ``IntegrityError``. The pre-existing non-null
  uniqueness still holds.
* DECISION-2 — the bot seed assigns deterministic small-int discord_ids; a fresh
  admin + bot seed leaves exactly one null.
* DECISION-3 — ``is_protected`` blocks delete / revoke / deactivate with a stable
  leading ``protected`` code, while grant / reactivate stay allowed.
* DECISION-4 — ``is_admin_by_discord_id`` requires ``is_admin AND is_active``.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest

import sqlalchemy.exc
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import User
from app.seeds.admins import seed_admin
from app.seeds.bots import BOT_ACCOUNTS, seed_bots
from app.services import admin as admin_service
from app.services.auth import is_admin_by_discord_id


def _enable_sqlite_fks(dbapi_connection, _connection_record):  # noqa: ANN001
    """Connect listener: turn SQLite FK (and cascade) enforcement ON."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class _FloorTestBase(unittest.TestCase):
    """Shared StaticPool SQLite engine with FK enforcement + the model schema."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        # Register the FK-enforcement listener BEFORE create_all (parity with
        # tests/test_admin_api) so the schema-building connection is covered too.
        event.listen(self.engine, "connect", _enable_sqlite_fks)
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()


# --------------------------------------------------------------------------- #
# DECISION-1 / DECISION-2 — one-null invariant + deterministic seed ids
# --------------------------------------------------------------------------- #
class OneNullDiscordIdTests(_FloorTestBase):
    """The model-level partial unique index caps NULL discord_ids at one."""

    def test_second_null_discord_id_raises_integrity_error(self) -> None:
        with Session(self.engine) as session:
            session.add(User(display_name="first_null", discord_id=None))
            session.commit()
            session.add(User(display_name="second_null", discord_id=None))
            with self.assertRaises(sqlalchemy.exc.IntegrityError):
                session.commit()

    def test_non_null_discord_id_still_unique(self) -> None:
        # Regression guard: the pre-existing plain UNIQUE on discord_id survives.
        with Session(self.engine) as session:
            session.add(User(display_name="a", discord_id=42))
            session.commit()
            session.add(User(display_name="b", discord_id=42))
            with self.assertRaises(sqlalchemy.exc.IntegrityError):
                session.commit()

    def test_seed_bots_assigns_deterministic_ids(self) -> None:
        with Session(self.engine) as session:
            seed_bots(session)
            expected = {name: did for name, _pw, did in BOT_ACCOUNTS}
            rows = session.exec(select(User).where(User.display_name.in_(list(expected)))).all()
            self.assertEqual(len(rows), len(BOT_ACCOUNTS))
            for row in rows:
                self.assertEqual(row.discord_id, expected[row.display_name])
            # Distinct ids — no collisions.
            ids = [r.discord_id for r in rows]
            self.assertEqual(len(ids), len(set(ids)))

            # Rerun is idempotent: same ids, no new rows.
            seed_bots(session)
            rerun = session.exec(select(User).where(User.display_name.in_(list(expected)))).all()
            self.assertEqual(len(rerun), len(BOT_ACCOUNTS))
            for row in rerun:
                self.assertEqual(row.discord_id, expected[row.display_name])

    def test_seed_admin_protected_and_null(self) -> None:
        with Session(self.engine) as session:
            seed_admin(session, username="root_admin", password="pw-secret-1")
            user = session.exec(select(User).where(User.display_name == "root_admin")).one()
            self.assertIsNone(user.discord_id)
            self.assertTrue(user.is_protected)

    def test_one_null_after_admin_plus_bots(self) -> None:
        with Session(self.engine) as session:
            seed_admin(session, username="root_admin", password="pw-secret-1")
            seed_bots(session)
            nulls = session.exec(select(User).where(User.discord_id.is_(None))).all()
            self.assertEqual(len(nulls), 1)
            self.assertEqual(nulls[0].display_name, "root_admin")


# --------------------------------------------------------------------------- #
# DECISION-3 — protected-row service guards (delete / revoke / deactivate)
# --------------------------------------------------------------------------- #
class ProtectedRowGuardTests(_FloorTestBase):
    """delete / revoke / deactivate refuse a protected row with "protected".

    grant / reactivate are NOT guarded (they cannot strand the system). The
    service treats ``caller_id`` as a pure parameter, so a synthetic distinct
    caller exercises the protected branch without tripping the self-guard.
    """

    CALLER = 10_000_000

    def _add(self, **kwargs) -> int:
        with Session(self.engine) as session:
            user = User(**kwargs)
            session.add(user)
            session.commit()
            session.refresh(user)
            assert user.id is not None
            return user.id

    @staticmethod
    def _leading_token(exc: ValueError) -> str:
        # Codes are formatted "code: human message"; the stable token is the
        # leading whitespace-delimited word with its trailing colon stripped.
        return str(exc.args[0]).split()[0].rstrip(":")

    def test_delete_protected_raises(self) -> None:
        # A second admin exists so the last-admin guard does NOT mask "protected".
        protected = self._add(
            display_name="bootstrap",
            discord_id=None,
            is_admin=True,
            is_active=True,
            is_protected=True,
        )
        self._add(display_name="other", discord_id=1, is_admin=True, is_active=True)
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.delete_user(session, caller_id=self.CALLER, user_id=protected)
            self.assertEqual(self._leading_token(ctx.exception), "protected")
            self.assertIsNotNone(session.get(User, protected))

    def test_revoke_protected_raises(self) -> None:
        protected = self._add(
            display_name="bootstrap",
            discord_id=None,
            is_admin=True,
            is_active=True,
            is_protected=True,
        )
        self._add(display_name="other", discord_id=1, is_admin=True, is_active=True)
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.revoke_admin(session, caller_id=self.CALLER, user_id=protected)
            self.assertEqual(self._leading_token(ctx.exception), "protected")
            self.assertTrue(session.get(User, protected).is_admin)

    def test_deactivate_protected_raises(self) -> None:
        protected = self._add(
            display_name="bootstrap",
            discord_id=None,
            is_admin=True,
            is_active=True,
            is_protected=True,
        )
        self._add(display_name="other", discord_id=1, is_admin=True, is_active=True)
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.deactivate_user(session, caller_id=self.CALLER, user_id=protected)
            self.assertEqual(self._leading_token(ctx.exception), "protected")
            self.assertTrue(session.get(User, protected).is_active)

    def test_grant_and_reactivate_protected_allowed(self) -> None:
        # grant_admin on an already-admin protected row raises ONLY "already_admin"
        # (proves the protected guard is ABSENT in grant); reactivate of an
        # inactive protected row succeeds (proves protected does not block it).
        protected_admin = self._add(
            display_name="bootstrap",
            discord_id=None,
            is_admin=True,
            is_active=True,
            is_protected=True,
        )
        with Session(self.engine) as session:
            with self.assertRaises(ValueError) as ctx:
                admin_service.grant_admin(session, user_id=protected_admin)
            self.assertEqual(self._leading_token(ctx.exception), "already_admin")

        inactive_protected = self._add(
            display_name="bootstrap_inactive",
            discord_id=2,
            is_admin=True,
            is_active=False,
            is_protected=True,
        )
        with Session(self.engine) as session:
            row = admin_service.reactivate_user(session, user_id=inactive_protected)
            self.assertTrue(row.is_active)


# --------------------------------------------------------------------------- #
# DECISION-4 — Discord admin gate requires is_active
# --------------------------------------------------------------------------- #
class DiscordActiveGateTests(_FloorTestBase):
    """is_admin_by_discord_id requires is_admin AND is_active."""

    def test_is_admin_by_discord_id_requires_active(self) -> None:
        with Session(self.engine) as session:
            user = User(
                display_name="discord_admin",
                discord_id=99,
                is_admin=True,
                is_active=False,
            )
            session.add(user)
            session.commit()
            # Deactivated admin -> not authorized.
            self.assertFalse(is_admin_by_discord_id(session, 99))
            # Reactivate -> authorized.
            user.is_active = True
            session.add(user)
            session.commit()
            self.assertTrue(is_admin_by_discord_id(session, 99))

    def test_bot_admin_command_refuses_deactivated(self) -> None:
        # The bot admin path (app/bot/db_bridge.py is_admin_async) is a thin
        # wrapper over is_admin_by_discord_id run in a thread; exercising the gate
        # directly with the SAME call the bridge makes avoids async test machinery
        # while proving the bot path refuses a deactivated admin.
        with Session(self.engine) as session:
            session.add(
                User(
                    display_name="bot_gated_admin",
                    discord_id=123,
                    is_admin=True,
                    is_active=False,
                )
            )
            session.commit()
            self.assertFalse(is_admin_by_discord_id(session, 123))


if __name__ == "__main__":
    unittest.main()
