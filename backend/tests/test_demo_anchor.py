"""Offline tests for the shared demo anchor/offset module.

Exercise :mod:`app.demo.anchor` against an in-memory SQLite engine — no Postgres,
no network, no ``app.db`` import, and :class:`~app.scoreboard.espn.EspnScoreboardSource`
is never constructed. They prove the determinism crux:

* :func:`offset_from_anchor` is pure/deterministic and positions week-1's earliest
  fixture kickoff at exactly ``anchor + DEMO_KICKOFF_BUFFER``;
* a naive anchor raises a deliberate ``ValueError``;
* ``store_demo_anchor`` -> ``load_demo_anchor`` round-trips tz-aware and is a
  single-row idempotent upsert (a second store leaves exactly one row).

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_demo_anchor -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel, create_engine, select

from app.demo.anchor import (
    DEMO_KICKOFF_BUFFER,
    load_demo_anchor,
    offset_from_anchor,
    store_demo_anchor,
)
from app.demo.offset import load_fixture_kickoffs
from app.models import DemoState


class OffsetFromAnchorTests(unittest.TestCase):
    """The pure offset formula is deterministic and lands week-1 at anchor+buffer."""

    def setUp(self) -> None:
        self.weeks = load_fixture_kickoffs()

    def test_positions_week1_earliest_at_anchor_plus_buffer(self) -> None:
        anchor = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
        offset = offset_from_anchor(anchor, self.weeks)
        earliest = min(self.weeks[1])
        # week-1 earliest positioned kickoff == anchor + buffer.
        self.assertEqual(earliest + offset, anchor + DEMO_KICKOFF_BUFFER)

    def test_deterministic_across_calls(self) -> None:
        anchor = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
        # Same anchor -> byte-identical offset whether or not the fixture is passed
        # (the default loads the same static packaged fixture).
        self.assertEqual(
            offset_from_anchor(anchor, self.weeks),
            offset_from_anchor(anchor),
        )

    def test_custom_buffer_honored(self) -> None:
        anchor = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
        offset = offset_from_anchor(anchor, self.weeks, buffer=timedelta(hours=48))
        self.assertEqual(min(self.weeks[1]) + offset, anchor + timedelta(hours=48))

    def test_naive_anchor_raises(self) -> None:
        naive = datetime(2026, 6, 23, 12, 0, 0)  # no tzinfo
        with self.assertRaises(ValueError):
            offset_from_anchor(naive, self.weeks)

    def test_default_buffer_is_24h(self) -> None:
        self.assertEqual(DEMO_KICKOFF_BUFFER, timedelta(hours=24))


class StoreLoadAnchorTests(unittest.TestCase):
    """The DB shells round-trip tz-aware and upsert a single row idempotently."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_round_trips_tz_aware(self) -> None:
        anchor = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
        with Session(self.engine) as session:
            store_demo_anchor(session, anchor)
            session.commit()
            loaded = load_demo_anchor(session)
        self.assertIsNotNone(loaded)
        self.assertIsNotNone(loaded.tzinfo)  # re-attached UTC after SQLite read
        self.assertEqual(loaded, anchor)

    def test_load_returns_none_when_unseeded(self) -> None:
        with Session(self.engine) as session:
            self.assertIsNone(load_demo_anchor(session))

    def test_store_is_single_row_idempotent_upsert(self) -> None:
        first = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
        second = datetime(2026, 7, 1, 9, 30, 0, tzinfo=timezone.utc)
        with Session(self.engine) as session:
            store_demo_anchor(session, first)
            session.commit()
            store_demo_anchor(session, second)  # re-stamp, not duplicate
            session.commit()

            rows = list(session.exec(select(DemoState)).all())
            self.assertEqual(len(rows), 1)
            self.assertEqual(load_demo_anchor(session), second)


if __name__ == "__main__":
    unittest.main()
