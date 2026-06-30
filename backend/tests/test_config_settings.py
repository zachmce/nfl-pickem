"""Offline tests for the APP_ENV fail-closed production guard (260630-lkf).

Construct :class:`app.config.Settings` DIRECTLY with explicit kwargs and pass
``_env_file=None`` on every instantiation so a developer's real ``.env`` (or the
repo ``.env.example``) can never bleed into these assertions — the guard's whole
point is to react to the supplied config, so the inputs must be hermetic.

The idiom mirrors :class:`tests.test_llm_commentary.ConfigLlmSettingsTests`
(Settings built with kwargs), hardened here with ``_env_file=None``.

> Note: on this machine there is no bare ``python`` on ``PATH``; run with
> ``backend/.venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest

from app.config import Settings

# A real-looking secure production baseline reused by the "otherwise secure"
# negative cases so each test isolates the ONE setting it is exercising.
_SECURE_PROD = dict(
    app_env="production",
    secret_key="x" * 40,
    session_cookie_secure=True,
    is_demo_data=False,
    cors_allowed_origins=["https://picks.example.com"],
    _env_file=None,
)


class ConfigFailClosedTests(unittest.TestCase):
    """Cover the _prod_fail_closed model_validator end to end."""

    def test_default_dev_env_constructs(self) -> None:
        """Default (development) constructs with no raise; dev defaults intact."""
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        self.assertEqual(s.app_env, "development")
        self.assertEqual(
            s.secret_key, "dev-only-insecure-secret-key-change-me-in-production"
        )
        self.assertFalse(s.session_cookie_secure)
        self.assertFalse(s.is_demo_data)

    def test_production_with_dev_secret_raises(self) -> None:
        """Production carrying the dev defaults refuses to construct."""
        with self.assertRaises(ValueError) as ctx:
            Settings(app_env="production", _env_file=None)  # type: ignore[call-arg]
        self.assertIn("secret", str(ctx.exception).lower())

    def test_production_insecure_cookie_raises(self) -> None:
        """An otherwise-secure prod config with insecure cookies raises."""
        cfg = {**_SECURE_PROD, "session_cookie_secure": False}
        with self.assertRaises(ValueError) as ctx:
            Settings(**cfg)  # type: ignore[call-arg]
        self.assertIn("cookie", str(ctx.exception).lower())

    def test_production_demo_data_raises(self) -> None:
        """An otherwise-secure prod config with demo data on raises."""
        cfg = {**_SECURE_PROD, "is_demo_data": True}
        with self.assertRaises(ValueError) as ctx:
            Settings(**cfg)  # type: ignore[call-arg]
        self.assertIn("demo", str(ctx.exception).lower())

    def test_production_localhost_cors_raises(self) -> None:
        """An otherwise-secure prod config with a localhost CORS origin raises."""
        cfg = {**_SECURE_PROD, "cors_allowed_origins": ["http://localhost:5173"]}
        with self.assertRaises(ValueError) as ctx:
            Settings(**cfg)  # type: ignore[call-arg]
        msg = str(ctx.exception).lower()
        self.assertTrue("cors" in msg or "origin" in msg, msg)

    def test_production_fully_secure_constructs(self) -> None:
        """The positive case: a fully-secure prod config constructs cleanly."""
        s = Settings(**_SECURE_PROD)  # type: ignore[call-arg]
        self.assertEqual(s.app_env, "production")
        self.assertEqual(s.secret_key, "x" * 40)

    def test_production_collects_all_failures(self) -> None:
        """All four checks failing at once surface in a single ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Settings(
                app_env="production",
                secret_key="short",
                session_cookie_secure=False,
                is_demo_data=True,
                cors_allowed_origins=["http://localhost:5173"],
                _env_file=None,
            )  # type: ignore[call-arg]
        msg = str(ctx.exception).lower()
        self.assertIn("secret", msg)
        self.assertIn("cookie", msg)
        self.assertIn("demo", msg)
        self.assertTrue("cors" in msg or "origin" in msg, msg)


if __name__ == "__main__":
    unittest.main()
