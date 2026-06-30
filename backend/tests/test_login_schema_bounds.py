"""Pure-schema bounds tests for UserLoginRequest (offline, no DB).

Pins the max-only Field bounds added to UserLoginRequest so an oversized login
payload is rejected as a 422 at request validation BEFORE any Argon2 hashing
runs (T-hsb-01: brute-force / CPU-exhaustion guard), instead of doing pointless
hashing work. These instantiate the Pydantic model directly and assert
``ValidationError`` — no DB or TestClient needed (faster than driving the full
API), and the same field bounds are what FastAPI surfaces as a 422 envelope at
the /login route boundary.

Covered bounds:
* UserLoginRequest.display_name: max_length=100.
* UserLoginRequest.password: max_length=128 (argon2id has no bcrypt 72-byte cap).

At-bound values (100-char display_name, 128-char password) are ACCEPTED.

Deliberately NOT bounded: there is NO min_length on the login password, so a
short password is accepted at the schema level — login must not enforce or leak
a password policy, and pre-existing short-password accounts must still
authenticate.

> Run from backend/ with ``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.schemas.auth import UserLoginRequest

_DISPLAY_NAME_MAX = 100
_PASSWORD_MAX = 128


class UserLoginRequestBoundsTests(unittest.TestCase):
    """display_name (100) and password (128) are capped; no min_length anywhere."""

    def test_at_bound_values_ok(self) -> None:
        req = UserLoginRequest(
            display_name="a" * _DISPLAY_NAME_MAX,
            password="a" * _PASSWORD_MAX,
        )
        self.assertEqual(len(req.display_name), _DISPLAY_NAME_MAX)
        self.assertEqual(len(req.password), _PASSWORD_MAX)

    def test_over_length_display_name_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserLoginRequest(
                display_name="a" * (_DISPLAY_NAME_MAX + 1),
                password="a" * _PASSWORD_MAX,
            )

    def test_over_length_password_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserLoginRequest(
                display_name="a" * _DISPLAY_NAME_MAX,
                password="a" * (_PASSWORD_MAX + 1),
            )

    def test_short_password_accepted_no_min_length(self) -> None:
        """A 1-char password is ACCEPTED — login enforces no password policy."""
        req = UserLoginRequest(display_name="userA", password="x")
        self.assertEqual(req.password, "x")


if __name__ == "__main__":
    unittest.main()
