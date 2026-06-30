"""Offline behavioral proof: a password change invalidates existing sessions.

Fully OFFLINE, mirroring tests/test_change_password_api.py:

* an in-memory SQLite engine (StaticPool — one pinned connection so every
  Session, including the one get_current_user opens, sees the SAME database),
* the app's DB dependency (:func:`app.db.get_session`) replaced via
  ``app.dependency_overrides`` so importing :mod:`app.main` never opens Postgres,
* no network of any kind.

The contract under test (Codex audit Theme 2 MEDIUM, quick task 260630-p0q):
the signed session cookie carries the user's ``session_version`` as ``sv``;
:func:`app.api.deps.get_current_user` rejects a cookie whose ``sv`` no longer
matches the column (treated exactly like a bad signature → 401). A password
change bumps the column, so an old cookie is logged out while a freshly-issued
cookie still works. A LEGACY cookie carrying no ``sv`` key decodes to 0 and still
authenticates a version-0 user (no force-logout on deploy).

> Note: on this machine the interpreter is ``python3`` (no bare ``python`` on
> ``PATH``); run with ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from app.db import get_session
from app.main import app
from app.models import User
from app.services.auth import _serializer, create_session_cookie, hash_password

_CURRENT_PASSWORD = "correct horse battery staple"
_NEW_PASSWORD = "a brand new passphrase 99"


class SessionInvalidationTests(unittest.TestCase):
    """Offline TestClient proof that session_version gates the session cookie."""

    user_id: int

    def setUp(self) -> None:
        # A single shared in-memory connection (StaticPool) so every Session —
        # including get_current_user's own Depends — sees the SAME database.
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        with Session(self.engine) as session:
            user = User(
                display_name="userA",
                password_hash=hash_password(_CURRENT_PASSWORD),
                is_active=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            assert user.id is not None
            self.user_id = user.id

        def _override_get_session():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_session] = _override_get_session
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_session, None)
        self.client.close()
        self.engine.dispose()

    # -- helpers -----------------------------------------------------------

    def _set_session_cookie(self, value: str) -> None:
        self.client.cookies.clear()
        self.client.cookies.set("session", value)

    def _csrf_pair(self) -> dict[str, str]:
        """Set the csrftoken cookie and return the matching X-CSRF-Token header."""
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        return {CSRF_HEADER_NAME: csrf_value}

    def _me(self) -> int:
        return self.client.get("/api/auth/me").status_code

    def _bump_version_in_db(self, new_version: int) -> None:
        with Session(self.engine) as session:
            user = session.get(User, self.user_id)
            assert user is not None
            user.session_version = new_version
            session.add(user)
            session.commit()

    def _read_version(self) -> int:
        with Session(self.engine) as session:
            user = session.get(User, self.user_id)
            assert user is not None
            return user.session_version

    # -- tests -------------------------------------------------------------

    def test_stale_cookie_rejected_after_version_bump(self) -> None:
        """An old cookie (sv=0) authenticates, then is 401 once the DB version bumps."""
        self._set_session_cookie(create_session_cookie(self.user_id, 0))
        self.assertEqual(self._me(), 200)

        # Simulate a password change in another session bumping the version.
        self._bump_version_in_db(1)

        # The SAME old cookie is now stale -> 401.
        self._set_session_cookie(create_session_cookie(self.user_id, 0))
        self.assertEqual(self._me(), 401)

    def test_fresh_cookie_works_after_version_bump(self) -> None:
        """A cookie carrying the bumped sv authenticates."""
        self._bump_version_in_db(1)
        self._set_session_cookie(create_session_cookie(self.user_id, 1))
        self.assertEqual(self._me(), 200)

    def test_legacy_cookie_with_no_sv_authenticates_version_zero_user(self) -> None:
        """A legacy token signed WITHOUT an sv key authenticates a v0 user (backward-compat)."""
        # Mint a token the OLD way — uid only, no sv — using the same serializer.
        legacy = _serializer().dumps({"uid": self.user_id})
        self._set_session_cookie(legacy)
        self.assertEqual(self._read_version(), 0)
        self.assertEqual(self._me(), 200)

    def test_user_never_changing_password_keeps_session(self) -> None:
        """A user whose version never changes keeps a working session."""
        self._set_session_cookie(create_session_cookie(self.user_id, 0))
        self.assertEqual(self._me(), 200)
        # Repeat probe — still authenticated, nothing invalidated it.
        self.assertEqual(self._me(), 200)

    def test_end_to_end_change_password_invalidates_old_cookie(self) -> None:
        """Real endpoint: change-password 200, OLD cookie -> 401, bumped cookie -> 200."""
        # Cookie-auth with the current (v0) cookie + the double-submit CSRF pair.
        self._set_session_cookie(create_session_cookie(self.user_id, 0))
        headers = self._csrf_pair()
        resp = self.client.post(
            "/api/auth/change-password",
            json={"current_password": _CURRENT_PASSWORD, "new_password": _NEW_PASSWORD},
            headers=headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        # The change bumped the version in the DB.
        bumped = self._read_version()
        self.assertEqual(bumped, 1)

        # The OLD pre-change cookie (sv=0) is now logged out.
        self._set_session_cookie(create_session_cookie(self.user_id, 0))
        self.assertEqual(self._me(), 401)

        # A cookie carrying the bumped sv authenticates again.
        self._set_session_cookie(create_session_cookie(self.user_id, bumped))
        self.assertEqual(self._me(), 200)


if __name__ == "__main__":
    unittest.main()
