"""Offline unit tests for the idempotent demo/bot-user seed.

These tests exercise :mod:`app.seeds.bots` against an in-memory SQLite engine —
no Postgres, no network, no ``app.db`` import. They prove the seed:

* creates exactly N clearly-labeled bot users from an empty users table,
* gives each a real argon2 credential that verifies via
  :func:`app.services.auth.verify_password` (never a hand-rolled hash),
* sets the labeled-account fields (``is_active``, ``is_admin``, ``discord_id``),
* and is idempotent — re-running leaves exactly N bot users with no error.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_bot_seed -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``.venv/bin/python -m unittest ...``.
"""

from __future__ import annotations

import unittest

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import User
from app.seeds.bots import BOT_ACCOUNTS, seed_bots
from app.services.auth import verify_password


class BotSeedTests(unittest.TestCase):
    """Idempotency + credential validity for the bot-user seed."""

    def setUp(self) -> None:
        # Fresh in-memory db per test; no Postgres, no app.db import.
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _bots(self, session: Session) -> list[User]:
        return list(
            session.exec(
                select(User).where(
                    User.display_name.in_([name for name, *_ in BOT_ACCOUNTS])
                )
            ).all()
        )

    def test_seed_creates_exactly_n_bots_from_empty(self) -> None:
        with Session(self.engine) as session:
            count = seed_bots(session)
            self.assertEqual(count, len(BOT_ACCOUNTS))
            self.assertEqual(len(self._bots(session)), len(BOT_ACCOUNTS))

    def test_at_least_four_bots_authored(self) -> None:
        # The plan calls for 4 or 5 bots (5 chosen for richer standings).
        self.assertGreaterEqual(len(BOT_ACCOUNTS), 4)

    def test_each_bot_is_clearly_labeled(self) -> None:
        for display_name, *_ in BOT_ACCOUNTS:
            self.assertTrue(
                display_name.startswith("bot_"),
                f"{display_name!r} is not clearly a demo/bot account",
            )
        # Display names must be unique (the natural key the seeder owns).
        names = [name for name, *_ in BOT_ACCOUNTS]
        self.assertEqual(len(names), len(set(names)))

    def test_credentials_verify_via_auth(self) -> None:
        with Session(self.engine) as session:
            seed_bots(session)
            for display_name, plaintext, _discord_id in BOT_ACCOUNTS:
                user = session.exec(
                    select(User).where(User.display_name == display_name)
                ).one()
                self.assertIsNotNone(user.password_hash)
                # The stored hash must verify against the known plaintext...
                self.assertTrue(verify_password(user.password_hash, plaintext))
                # ...and reject a wrong password (proves it is a real hash).
                self.assertFalse(verify_password(user.password_hash, "wrong"))

    def test_bot_fields_are_canonical(self) -> None:
        with Session(self.engine) as session:
            seed_bots(session)
            # Each bot carries its deterministic small-int discord_id (1..5) so
            # the one-null-discord_id invariant leaves only the bootstrap admin
            # null; bots are never protected.
            expected_ids = {name: did for name, _pw, did in BOT_ACCOUNTS}
            for user in self._bots(session):
                self.assertTrue(user.is_active)
                self.assertFalse(user.is_admin)
                self.assertFalse(user.is_protected)
                self.assertEqual(user.discord_id, expected_ids[user.display_name])
            # All bot discord_ids are distinct.
            ids = [u.discord_id for u in self._bots(session)]
            self.assertEqual(len(ids), len(set(ids)))

    def test_seed_is_idempotent(self) -> None:
        with Session(self.engine) as session:
            seed_bots(session)
            first = self._bots(session)
            first_hashes = {u.display_name: u.password_hash for u in first}

            # Re-run on the same session: no duplicates, no error.
            count = seed_bots(session)
            self.assertEqual(count, len(BOT_ACCOUNTS))

            second = self._bots(session)
            self.assertEqual(len(second), len(BOT_ACCOUNTS))
            # Existing rows are NOT re-hashed (hashing is nondeterministic; a
            # re-hash on rerun would silently dirty idempotency).
            for u in second:
                self.assertEqual(u.password_hash, first_hashes[u.display_name])


if __name__ == "__main__":
    unittest.main()
