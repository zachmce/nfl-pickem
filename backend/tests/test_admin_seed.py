"""Offline unit tests for the idempotent admin-bootstrap seed.

These tests exercise :mod:`app.seeds.admins` against an in-memory SQLite engine —
no Postgres, no network, no ``app.db`` import. The seed is driven through the
``username=`` / ``password=`` keyword seam (never process env), so the tests do
not touch the lru_cache'd Settings singleton. They prove the seed:

* creates exactly one canonical admin from an empty users table,
* gives it a real argon2 credential that verifies via
  :func:`app.services.auth.verify_password` (never a hand-rolled hash),
* no-ops when either credential is unset, and
* skips an existing ``display_name`` without overwriting its password.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_admin_seed -v
"""

from __future__ import annotations

import unittest

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import User
from app.seeds.admins import seed_admin
from app.services.auth import verify_password


class AdminSeedTests(unittest.TestCase):
    """Create / skip-unset / skip-exists-no-overwrite for the admin seed."""

    def setUp(self) -> None:
        # Fresh in-memory db per test; no Postgres, no app.db import.
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _admins(self, session: Session) -> list[User]:
        return list(
            session.exec(
                select(User).where(User.display_name == "root_admin")
            ).all()
        )

    def test_creates_when_set_and_absent(self) -> None:
        with Session(self.engine) as session:
            created = seed_admin(
                session, username="root_admin", password="pw-secret-1"
            )
            self.assertTrue(created)
            self.assertEqual(len(self._admins(session)), 1)

    def test_created_user_is_canonical(self) -> None:
        with Session(self.engine) as session:
            seed_admin(session, username="root_admin", password="pw-secret-1")
            user = session.exec(
                select(User).where(User.display_name == "root_admin")
            ).one()
            self.assertTrue(user.is_admin)
            self.assertTrue(user.is_active)
            self.assertIsNone(user.discord_id)
            self.assertIsNotNone(user.password_hash)
            # A real argon2 hash: verifies the right password, rejects a wrong one.
            self.assertTrue(verify_password(user.password_hash, "pw-secret-1"))
            self.assertFalse(verify_password(user.password_hash, "wrong"))

    def test_skips_when_username_unset(self) -> None:
        with Session(self.engine) as session:
            created = seed_admin(
                session, username=None, password="pw-secret-1"
            )
            self.assertFalse(created)
            self.assertEqual(len(self._admins(session)), 0)

    def test_skips_when_password_unset(self) -> None:
        with Session(self.engine) as session:
            created = seed_admin(session, username="root_admin", password=None)
            self.assertFalse(created)
            self.assertEqual(len(self._admins(session)), 0)

    def test_skips_when_display_name_exists_no_overwrite(self) -> None:
        with Session(self.engine) as session:
            self.assertTrue(
                seed_admin(
                    session, username="root_admin", password="pw-secret-1"
                )
            )
            original_hash = session.exec(
                select(User).where(User.display_name == "root_admin")
            ).one().password_hash

            # Second run with a DIFFERENT password must skip and not overwrite.
            created = seed_admin(
                session, username="root_admin", password="a-different-password"
            )
            self.assertFalse(created)
            self.assertEqual(len(self._admins(session)), 1)

            stored_hash = session.exec(
                select(User).where(User.display_name == "root_admin")
            ).one().password_hash
            self.assertEqual(stored_hash, original_hash)
            # The second password never took effect.
            self.assertFalse(
                verify_password(stored_hash, "a-different-password")
            )
            self.assertTrue(verify_password(stored_hash, "pw-secret-1"))


if __name__ == "__main__":
    unittest.main()
