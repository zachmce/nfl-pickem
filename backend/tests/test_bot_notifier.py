"""Offline unit tests for the bot-side channel resolver (QT-1).

These tests NEVER touch a live Redis or a live Discord gateway. The resilient
``redis.asyncio`` subscriber loop is exercised by the manual/live confirm step in
the SUMMARY; here we unit-test the pure seam — ``resolve_channel`` — which is the
guild-scoping guard (T-kd8-03): it searches ONLY the passed guild's channels and
matches by numeric id OR by name.

Run with: ``backend/.venv/bin/python -m unittest tests.test_bot_notifier -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.bot.notifier import _render, resolve_channel
from app.services.notifications import (
    admin_pick_cleared_event,
    admin_pick_set_event,
    freeze_week_event,
    ingest_season_event,
    login_event,
    pick_cleared_event,
    pick_event,
    player_registered_event,
)


@dataclass
class _FakeChannel:
    id: int
    name: str


class _FakeGuild:
    """A guild-like object exposing only the ``.channels`` iterable the resolver
    reads — proves the resolver needs no real Discord object."""

    def __init__(self, channels: list[_FakeChannel]) -> None:
        self.channels = channels


def _guild() -> _FakeGuild:
    return _FakeGuild(
        [
            _FakeChannel(id=123, name="pickem-logger"),
            _FakeChannel(id=456, name="pickem-chat"),
        ]
    )


class ResolveChannelTests(unittest.TestCase):
    def test_match_by_numeric_id(self) -> None:
        ch = resolve_channel(_guild(), "123")
        self.assertIsNotNone(ch)
        self.assertEqual(ch.id, 123)
        self.assertEqual(ch.name, "pickem-logger")

    def test_match_by_name(self) -> None:
        ch = resolve_channel(_guild(), "pickem-logger")
        self.assertIsNotNone(ch)
        self.assertEqual(ch.id, 123)

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(resolve_channel(_guild(), "nonexistent"))

    def test_none_setting_returns_none(self) -> None:
        self.assertIsNone(resolve_channel(_guild(), None))

    def test_blank_setting_returns_none(self) -> None:
        self.assertIsNone(resolve_channel(_guild(), "   "))

    def test_numeric_id_not_present_returns_none(self) -> None:
        # An int that matches no channel id must NOT fall back to a name match.
        self.assertIsNone(resolve_channel(_guild(), "999"))

    def test_none_guild_returns_none(self) -> None:
        # get_guild() can return None (bot not in guild yet) — must not raise.
        self.assertIsNone(resolve_channel(None, "pickem-logger"))


class RenderTests(unittest.TestCase):
    """Feed each builder's output through ``_render`` and assert the exact line.

    The bot does NO resolution — it only string-joins the structured fields the
    QT-2 builders emit (the resolved side/team is already in ``detail``).
    """

    def test_render_login(self) -> None:
        self.assertEqual(_render(login_event("alice")), "alice logged in")

    def test_render_pick_created(self) -> None:
        event = pick_event("pick.created", actor="bob", week=3, detail="OVER KC")
        self.assertEqual(_render(event), "bob pick · Week 3 · OVER KC")

    def test_render_pick_changed(self) -> None:
        event = pick_event("pick.changed", actor="bob", week=3, detail="Favorite (KC)")
        self.assertEqual(_render(event), "bob pick · Week 3 · Favorite (KC)")

    def test_render_pick_cleared(self) -> None:
        event = pick_cleared_event(actor="bob", week=3, detail="OVER KC")
        self.assertEqual(_render(event), "bob cleared · Week 3 · OVER KC")

    def test_render_admin_pick_set(self) -> None:
        event = admin_pick_set_event(target="alice", week=3, detail="Favorite (KC)")
        self.assertEqual(_render(event), "admin set alice · Week 3 · Favorite (KC)")

    def test_render_admin_pick_cleared(self) -> None:
        event = admin_pick_cleared_event(target="alice", week=3, slot="FAVORITE_COVER")
        self.assertEqual(
            _render(event), "admin cleared alice · Week 3 · FAVORITE_COVER"
        )

    def test_render_player_registered(self) -> None:
        self.assertEqual(_render(player_registered_event("newbie")), "new player: newbie")

    def test_render_ingest_season(self) -> None:
        event = ingest_season_event(season=2026, weeks=18, games=272, failed=1)
        self.assertEqual(_render(event), "ingested 2026 · 18 wk / 272 games (1 failed)")

    def test_render_freeze_week(self) -> None:
        self.assertEqual(_render(freeze_week_event(week=3)), "Week 3 lines frozen")

    def test_render_unknown_type_returns_none(self) -> None:
        self.assertIsNone(_render({"v": 1, "type": "totally.unknown"}))


if __name__ == "__main__":
    unittest.main()
