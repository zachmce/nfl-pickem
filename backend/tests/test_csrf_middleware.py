"""Offline HTTP tests for the pure-ASGI CSRFMiddleware enforcement contract.

Fully OFFLINE, mirroring tests/test_change_password_api.py:

* an in-memory SQLite engine (StaticPool — one pinned connection so every
  Session, including the one get_current_user opens, sees the SAME database),
* the app's DB dependency (:func:`app.db.get_session`) replaced via
  ``app.dependency_overrides`` so importing :mod:`app.main` never opens Postgres,
* no network of any kind.

These prove the double-submit CSRF contract is byte-identical after the
``BaseHTTPMiddleware`` -> pure-ASGI ``CSRFMiddleware`` rewrite: cookie-auth
enforcement, the exact 403 envelope, the bearer exemption, exempt paths, and
safe methods.

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
from app.services.auth import create_session_cookie, hash_password

_CURRENT_PASSWORD = "correct horse battery staple"
_NEW_PASSWORD = "a brand new passphrase 99"


class CSRFMiddlewareTests(unittest.TestCase):
    """Offline TestClient coverage for the CSRF enforcement contract."""

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

    # -- tests -------------------------------------------------------------

    def test_cookie_post_without_csrf_returns_exact_403_envelope(self) -> None:
        """(a) Cookie-auth POST (session cookie only, NO csrftoken cookie, NO
        header) to a non-exempt route -> 403 with the EXACT envelope."""
        self.client.cookies.clear()
        self.client.cookies.set("session", create_session_cookie(self.user_id))
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": _CURRENT_PASSWORD,
                "new_password": _NEW_PASSWORD,
            },
        )
        self.assertEqual(resp.status_code, 403, resp.text)
        self.assertEqual(
            resp.json(),
            {"error": {"code": "csrf_failed", "message": "CSRF token missing or invalid"}},
        )

    def test_cookie_post_with_matching_pair_is_not_csrf_blocked(self) -> None:
        """(b) Cookie-auth POST WITH matching csrftoken cookie + X-CSRF-Token
        header passes CSRF (a valid change returns 200)."""
        self.client.cookies.clear()
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set("session", create_session_cookie(self.user_id))
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": _CURRENT_PASSWORD,
                "new_password": _NEW_PASSWORD,
            },
            headers={CSRF_HEADER_NAME: csrf_value},
        )
        self.assertNotEqual(resp.status_code, 403, resp.text)
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_bearer_post_without_csrf_is_not_blocked(self) -> None:
        """(c) Bearer-auth POST (no session cookie, no CSRF pair) is NOT CSRF
        blocked — the bearer exemption is preserved."""
        self.client.cookies.clear()
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": _CURRENT_PASSWORD,
                "new_password": _NEW_PASSWORD,
            },
            headers={"Authorization": f"Bearer {create_session_cookie(self.user_id)}"},
        )
        self.assertNotEqual(resp.status_code, 403, resp.text)

    def test_exempt_path_with_session_cookie_is_not_blocked(self) -> None:
        """(d) An exempt path (login) with a session cookie present but no CSRF
        pair is NOT CSRF blocked."""
        self.client.cookies.clear()
        self.client.cookies.set("session", create_session_cookie(self.user_id))
        resp = self.client.post(
            "/api/auth/login",
            json={"display_name": "userA", "password": _CURRENT_PASSWORD},
        )
        self.assertNotEqual(resp.status_code, 403, resp.text)

    def test_safe_method_get_with_session_cookie_is_not_blocked(self) -> None:
        """(e) A safe method (GET) with a session cookie but no CSRF pair is
        never CSRF blocked."""
        self.client.cookies.clear()
        self.client.cookies.set("session", create_session_cookie(self.user_id))
        resp = self.client.get("/api/auth/me")
        self.assertNotEqual(resp.status_code, 403, resp.text)


if __name__ == "__main__":
    unittest.main()
