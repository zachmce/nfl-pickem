"""Offline HTTP tests for POST /api/auth/change-password.

Fully OFFLINE, mirroring tests/test_picks_api.py:

* an in-memory SQLite engine (StaticPool — one pinned connection so every
  Session, including the one get_current_user opens, sees the SAME database),
* the app's DB dependency (:func:`app.db.get_session`) replaced via
  ``app.dependency_overrides`` so importing :mod:`app.main` never opens Postgres,
* no network of any kind.

Auth is exercised on the cookie path with the double-submit CSRF pair (session
cookie + ``csrftoken`` cookie + matching ``X-CSRF-Token`` header), exactly as the
SPA does and as :mod:`app.csrf` enforces. The unauthenticated case clears all
cookies and sends nothing.

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
from app.services.auth import create_session_cookie, hash_password, verify_password

_CURRENT_PASSWORD = "correct horse battery staple"
_NEW_PASSWORD = "a brand new passphrase 99"


class ChangePasswordApiTests(unittest.TestCase):
    """Offline TestClient coverage for POST /api/auth/change-password."""

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

    def _cookie_auth_headers(self, user_id: int) -> dict[str, str]:
        """Set the signed session + CSRF cookies and return the CSRF header.

        Mirrors the SPA double-submit contract: a ``csrftoken`` cookie whose
        value is echoed in ``X-CSRF-Token``.
        """
        csrf_value = "test-csrf-token-value"
        self.client.cookies.set("session", create_session_cookie(user_id))
        self.client.cookies.set(CSRF_COOKIE_NAME, csrf_value)
        return {CSRF_HEADER_NAME: csrf_value}

    def _bearer_headers(self, user_id: int) -> dict[str, str]:
        """Bearer auth (CSRF-exempt) — used only for the unauth-contrast read."""
        return {"Authorization": f"Bearer {create_session_cookie(user_id)}"}

    def _clear_auth(self) -> None:
        self.client.cookies.clear()

    def _stored_hash(self) -> str:
        with Session(self.engine) as session:
            user = session.get(User, self.user_id)
            assert user is not None and user.password_hash is not None
            return user.password_hash

    @staticmethod
    def _assert_envelope(body: dict) -> dict:
        assert "error" in body, f"expected an error envelope, got: {body}"
        err = body["error"]
        assert "code" in err, f"envelope missing 'code': {err}"
        return err

    # -- tests -------------------------------------------------------------

    def test_change_password_success_then_relogin(self) -> None:
        """Valid change -> 200; the stored hash rotates so the NEW password logs in
        and the OLD one is rejected."""
        headers = self._cookie_auth_headers(self.user_id)
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": _CURRENT_PASSWORD,
                "new_password": _NEW_PASSWORD,
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"message": "password_changed"})

        # The stored hash now verifies the NEW password and rejects the OLD one.
        new_hash = self._stored_hash()
        self.assertTrue(verify_password(new_hash, _NEW_PASSWORD))
        self.assertFalse(verify_password(new_hash, _CURRENT_PASSWORD))

        # And a fresh login with the new password succeeds; the old one is 401.
        self._clear_auth()
        ok = self.client.post(
            "/api/auth/login",
            json={"display_name": "userA", "password": _NEW_PASSWORD},
        )
        self.assertEqual(ok.status_code, 200, ok.text)

        self._clear_auth()
        bad = self.client.post(
            "/api/auth/login",
            json={"display_name": "userA", "password": _CURRENT_PASSWORD},
        )
        self.assertEqual(bad.status_code, 401, bad.text)

    def test_wrong_current_password_returns_401(self) -> None:
        """Authenticated + WRONG current password -> 401 (invalid_credentials), NOT 500;
        the stored hash is unchanged."""
        before = self._stored_hash()
        headers = self._cookie_auth_headers(self.user_id)
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": "not the current password",
                "new_password": _NEW_PASSWORD,
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 401, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("code"), "invalid_credentials")
        # Hash untouched.
        self.assertEqual(self._stored_hash(), before)

    def test_unauthenticated_returns_401(self) -> None:
        """No session cookie, no bearer -> 401 (unauthorized)."""
        self._clear_auth()
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": _CURRENT_PASSWORD,
                "new_password": _NEW_PASSWORD,
            },
        )
        self.assertEqual(resp.status_code, 401, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("code"), "unauthorized")

    def test_cookie_post_without_csrf_returns_403(self) -> None:
        """Cookie-authenticated POST with NO csrftoken cookie / no header -> 403
        (csrf_failed). Proves the route lives inside CSRF protection."""
        self._clear_auth()
        # Session cookie ONLY — no csrftoken cookie, no X-CSRF-Token header.
        self.client.cookies.set("session", create_session_cookie(self.user_id))
        resp = self.client.post(
            "/api/auth/change-password",
            json={
                "current_password": _CURRENT_PASSWORD,
                "new_password": _NEW_PASSWORD,
            },
        )
        self.assertEqual(resp.status_code, 403, resp.text)
        err = self._assert_envelope(resp.json())
        self.assertEqual(err.get("code"), "csrf_failed")

    def test_me_exposes_created_at_and_no_password_hash(self) -> None:
        """GET /api/auth/me returns created_at and never leaks password_hash."""
        resp = self.client.get(
            "/api/auth/me", headers=self._bearer_headers(self.user_id)
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("created_at", body)
        self.assertIsInstance(body["created_at"], str)
        self.assertNotIn("password_hash", body)


if __name__ == "__main__":
    unittest.main()
