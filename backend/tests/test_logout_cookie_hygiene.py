"""Offline proof: logout emits attribute-matched cookie-deletion Set-Cookie headers.

Fully OFFLINE, mirroring tests/test_session_invalidation.py:

* an in-memory SQLite engine (StaticPool — one pinned connection),
* the app's DB dependency (:func:`app.db.get_session`) replaced via
  ``app.dependency_overrides`` so importing :mod:`app.main` never opens Postgres,
* no network of any kind (and no real HTTPS — Secure is asserted purely from the
  emitted Set-Cookie attribute string).

The contract under test (quick task 260708-h7w, brief t11-session-cookie-hygiene):
``POST /api/auth/logout`` must delete BOTH the session cookie and the ``csrftoken``
cookie with a Set-Cookie whose attributes are MATCHED to the cookies' setters
(``_set_session_cookie`` in app.api.auth and ``set_csrf_cookie`` in app.csrf).
Browsers increasingly require an attribute-matched overwrite to actually clear a
cookie, so under ``session_cookie_secure=true`` a bare deletion may leave the
cookie in place. We therefore assert the deletion headers carry:

* session: HttpOnly, SameSite=Lax, Path=/, a deletion marker (Max-Age=0 or a
  past Expires), and Secure IFF ``session_cookie_secure`` is on;
* csrftoken: SameSite=Lax, Path=/, a deletion marker, NOT HttpOnly, and Secure
  IFF ``session_cookie_secure`` is on.

TestClient auto-processes Set-Cookie into its own cookie jar, so we assert on the
RAW response headers (``resp.headers.get_list("set-cookie")``), not the jar.
logout is CSRF-exempt and needs no auth (idempotent), so a bare POST suffices.

> Note: on this machine the interpreter is ``python3`` (no bare ``python`` on
> ``PATH``) and pytest is not installed; run with
> ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api import auth as auth_module
from app.config import settings
from app.csrf import CSRF_COOKIE_NAME
from app.db import get_session
from app.main import app


class LogoutCookieHygieneTests(unittest.TestCase):
    """Offline TestClient proof that logout deletes both cookies attribute-matched."""

    def setUp(self) -> None:
        # A single shared in-memory connection (StaticPool) so importing app.main
        # never reaches for Postgres; logout itself has no DB dependency.
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

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

    def _logout(self):
        """POST logout and return the response (no auth/CSRF needed — exempt)."""
        return self.client.post("/api/auth/logout")

    @staticmethod
    def _find_set_cookie(resp, name_prefix: str) -> str:
        """Return the lower-cased Set-Cookie line whose cookie name matches, or fail."""
        for line in resp.headers.get_list("set-cookie"):
            if line.startswith(name_prefix + "="):
                return line.lower()
        raise AssertionError(
            f"no Set-Cookie for {name_prefix!r}; got {resp.headers.get_list('set-cookie')!r}"
        )

    @staticmethod
    def _has_deletion_marker(cookie_line_lower: str) -> bool:
        """A deletion Set-Cookie carries Max-Age=0 OR an Expires in the past (1970)."""
        return "max-age=0" in cookie_line_lower or "expires=thu, 01 jan 1970" in cookie_line_lower

    # -- tests -------------------------------------------------------------

    def test_logout_status_and_body_unchanged(self) -> None:
        """logout still returns 200 with body message == 'logged_out'."""
        resp = self._logout()
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json().get("message"), "logged_out")

    def test_session_cookie_deletion_is_attribute_matched(self) -> None:
        """Session deletion carries HttpOnly + SameSite=Lax + Path=/ + a deletion marker, no Secure."""
        resp = self._logout()
        line = self._find_set_cookie(resp, settings.session_cookie_name)
        self.assertIn("httponly", line)
        self.assertIn("samesite=lax", line)
        self.assertIn("path=/", line)
        self.assertTrue(self._has_deletion_marker(line), line)
        # session_cookie_secure defaults False in tests -> no Secure attribute.
        self.assertNotIn("secure", line)

    def test_csrf_cookie_deletion_is_attribute_matched(self) -> None:
        """csrftoken deletion carries SameSite=Lax + Path=/ + a deletion marker, NOT HttpOnly, no Secure."""
        resp = self._logout()
        line = self._find_set_cookie(resp, CSRF_COOKIE_NAME)
        self.assertIn("samesite=lax", line)
        self.assertIn("path=/", line)
        self.assertTrue(self._has_deletion_marker(line), line)
        self.assertNotIn("httponly", line)
        self.assertNotIn("secure", line)

    def test_both_deletions_carry_secure_when_secure_enabled(self) -> None:
        """With session_cookie_secure on, BOTH deletion headers carry Secure (matched to secure setters)."""
        original = auth_module.settings.session_cookie_secure
        auth_module.settings.session_cookie_secure = True
        try:
            resp = self._logout()
            session_line = self._find_set_cookie(resp, settings.session_cookie_name)
            csrf_line = self._find_set_cookie(resp, CSRF_COOKIE_NAME)
            self.assertIn("secure", session_line)
            self.assertIn("secure", csrf_line)
            # Attribute-match sanity: the other flags still hold under secure.
            self.assertIn("httponly", session_line)
            self.assertNotIn("httponly", csrf_line)
            self.assertTrue(self._has_deletion_marker(session_line), session_line)
            self.assertTrue(self._has_deletion_marker(csrf_line), csrf_line)
        finally:
            auth_module.settings.session_cookie_secure = original


if __name__ == "__main__":
    unittest.main()
