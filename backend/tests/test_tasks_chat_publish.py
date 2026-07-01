"""Offline regression tests for the QT-3 pickem-CHAT publish glue in ``app.tasks``.

The refresh-task layer (``_publish_refresh_chat_edges`` + ``_active_refresh_season``)
turns the in-cycle edges that ``refresh_games`` collects into published chat events.
It was previously untested (no test imported ``app.tasks``), which let a real bug
ship: ``_active_refresh_season`` did ``for (s,) in session.exec(select(Game.season))``,
but ``session.exec`` yields scalar ints for a single-column select — so it raised
``TypeError: cannot unpack non-iterable int object`` whenever ``recap_weeks`` was
non-empty (i.e. exactly when a week.recap should fire). Surfaced by the live
demo-anchor walkthrough; these tests pin the fix.

Everything runs OFFLINE: an in-memory SQLite engine (no Postgres — ``app.db``
builds its engine lazily, so importing ``app.tasks`` never connects) and
``publish_event`` is monkeypatched to capture events (never touches Redis).

Run from ``backend/``::

    .venv/bin/python -m unittest tests.test_tasks_chat_publish -v
"""

from __future__ import annotations

import unittest
from unittest import mock

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.refresh import RefreshResult
from app.tasks import _active_refresh_season, _publish_refresh_chat_edges


def _memory_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


class ActiveRefreshSeasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _memory_engine()
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as s:
            seed_teams(s)
            import_fixture_2025(s)  # 272 games, all season 2025
            s.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_returns_the_single_season(self) -> None:
        # Direct regression: the old `for (s,) in ...` raised TypeError here.
        with Session(self.engine) as s:
            self.assertEqual(_active_refresh_season(s), 2025)

    def test_returns_none_when_no_games(self) -> None:
        engine = _memory_engine()
        SQLModel.metadata.create_all(engine)
        try:
            with Session(engine) as s:
                self.assertIsNone(_active_refresh_season(s))
        finally:
            engine.dispose()


class PublishRefreshChatEdgesTests(unittest.TestCase):
    """Drive the real publish glue with a recap week present — the path that
    crashed before the fix — and assert it runs end-to-end."""

    def setUp(self) -> None:
        self.engine = _memory_engine()
        SQLModel.metadata.create_all(self.engine)
        with Session(self.engine) as s:
            seed_teams(s)
            import_fixture_2025(s)
            s.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_recap_week_publishes_finals_and_windows_without_unpack_error(self) -> None:
        # recap_weeks is non-empty -> reaches _active_refresh_season (the old crash
        # site). With no graded picks there is no winner, so the recap line is
        # gracefully skipped, but the finals/window publishes still fire and the
        # season resolution must NOT raise.
        recorded: list[dict] = []
        result = RefreshResult(
            finalized_games=((2, "WSH", "GB", 18, 27),),
            windows_opened=(3,),
            windows_closed=(2,),
            recap_weeks=(2,),
        )
        with mock.patch("app.tasks.publish_event", side_effect=recorded.append):
            with Session(self.engine) as s:
                _publish_refresh_chat_edges(s, result)

        types = [e.get("type") for e in recorded]
        # Finals + both window crossings published; recap skipped (no winner) —
        # crucially, NO TypeError reaching the recap block.
        self.assertEqual(types, ["game.final", "window.opened", "window.closed"])

    def test_no_recap_block_when_recap_weeks_empty(self) -> None:
        recorded: list[dict] = []
        result = RefreshResult(windows_closed=(2,))
        with mock.patch("app.tasks.publish_event", side_effect=recorded.append):
            with Session(self.engine) as s:
                _publish_refresh_chat_edges(s, result)
        self.assertEqual([e.get("type") for e in recorded], ["window.closed"])


if __name__ == "__main__":
    unittest.main()
