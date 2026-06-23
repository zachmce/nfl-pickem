"""Offline tests for the gated scoreboard source seam (PROD-LEAK-GUARD).

These exercise :func:`app.config.default_scoreboard_source` — the SINGLE source
seam — offline. No network is touched: the ESPN adapter constructs without
fetching (we never call ``fetch_week`` on it), and the demo branch reads an
in-memory ``DemoState`` row.

Proven (T-sf0-01):

* flag OFF (default prod path): returns an :class:`~app.scoreboard.espn.EspnScoreboardSource`
  and performs NO DemoState read (the demo machinery is never even imported/used);
* flag ON with a stored anchor: returns a :class:`~app.scoreboard.demo.Demo2025Source`
  whose offset equals ``offset_from_anchor(stored_anchor)`` (the seed/poller share);
* flag ON with no anchor: raises a clear ``RuntimeError`` (run the seed first).

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_demo_source_seam -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from sqlmodel import Session, SQLModel, create_engine

import app.config as config
from app.config import default_scoreboard_source
from app.demo.anchor import offset_from_anchor, store_demo_anchor
from app.scoreboard.demo import Demo2025Source
from app.scoreboard.espn import EspnScoreboardSource


class _SeamTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)
        # Remember the real flag so each test restores it (the module-level
        # ``settings`` is the one the seam reads).
        self._orig_flag = config.settings.is_demo_data

    def tearDown(self) -> None:
        config.settings.is_demo_data = self._orig_flag
        self.engine.dispose()


class FlagOffTests(_SeamTestBase):
    """OFF -> ESPN adapter, zero demo read (prod path byte-for-byte)."""

    def test_returns_espn_and_does_not_read_demo_state(self) -> None:
        config.settings.is_demo_data = False

        # A session whose .exec must NEVER be called on the prod path. If the seam
        # touched DemoState it would call session.exec and trip this guard.
        class _GuardSession:
            def exec(self, *_a, **_k):  # pragma: no cover - must not be called
                raise AssertionError(
                    "prod path must not read DemoState (PROD-LEAK-GUARD)"
                )

        source = default_scoreboard_source(_GuardSession())
        self.assertIsInstance(source, EspnScoreboardSource)

    def test_off_ignores_session_none(self) -> None:
        config.settings.is_demo_data = False
        source = default_scoreboard_source()
        self.assertIsInstance(source, EspnScoreboardSource)


class FlagOnTests(_SeamTestBase):
    """ON -> Demo2025Source built from the shared persisted anchor."""

    def test_on_with_anchor_returns_demo_source_with_shared_offset(self) -> None:
        config.settings.is_demo_data = True
        anchor = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
        with Session(self.engine) as session:
            store_demo_anchor(session, anchor)
            session.commit()

            source = default_scoreboard_source(session)
            self.assertIsInstance(source, Demo2025Source)
            # The offset must equal the SAME formula the seed used (shared anchor).
            self.assertEqual(source._offset, offset_from_anchor(anchor))

    def test_on_without_anchor_raises_runtime_error(self) -> None:
        config.settings.is_demo_data = True
        with Session(self.engine) as session:
            with self.assertRaises(RuntimeError):
                default_scoreboard_source(session)


if __name__ == "__main__":
    unittest.main()
