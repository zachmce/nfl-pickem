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

import asyncio
import json
import unittest
from dataclasses import dataclass
from unittest import mock

from app.bot.notifier import _render, resolve_channel, run_notifier
from app.services.notifications import (
    EVENTS_CHANNEL,
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


class _SendableChannel:
    """A channel exposing the ``.id``/``.name`` the resolver reads plus an async
    ``.send`` that records the lines posted — no real Discord object needed."""

    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name
        self.sent: list[str] = []

    async def send(self, line: str) -> None:
        self.sent.append(line)


class _SendableGuild:
    def __init__(self, channels: list[_SendableChannel]) -> None:
        self.channels = channels


class _FakeClient:
    def __init__(self, guild: _SendableGuild) -> None:
        self._guild = guild

    def get_guild(self, _guild_id):  # noqa: ANN001 - mirrors discord.Client
        return self._guild


class _FakeSettings:
    redis_url = "redis://fake:6379/0"
    discord_guild_id = 999
    discord_chat_log_channel = "pickem-logger"


class _FakePubSub:
    """Async-generator-backed pubsub. ``script`` is either the string ``"drop"``
    (subscribe succeeds, then ``listen`` raises a connection error mid-stream) or a
    list of message frames to yield before raising ``CancelledError`` to end the
    test cleanly (the real bot-shutdown signal)."""

    def __init__(self, script) -> None:  # noqa: ANN001
        self.script = script
        self.subscribed: list[str] = []
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)

    async def listen(self):
        if self.script == "drop":
            raise ConnectionError("redis connection lost")
            yield  # pragma: no cover - unreachable; makes this an async generator
        for message in self.script:
            yield message
        raise asyncio.CancelledError()

    async def aclose(self) -> None:
        self.closed = True


class _FakeRedis:
    def __init__(self, pubsub: _FakePubSub) -> None:
        self._pubsub = pubsub
        self.closed = False

    def pubsub(self) -> _FakePubSub:
        return self._pubsub

    async def aclose(self) -> None:
        self.closed = True


class RunNotifierReconnectTests(unittest.IsolatedAsyncioTestCase):
    """Prove the subscriber RECONNECTS after a Redis connection drop instead of
    silently dying — the bug where one Redis restart killed all notifications
    until the bot was manually restarted (0 subscribers on ``pickem:events``)."""

    async def test_reconnects_after_connection_drop(self) -> None:
        login_frame = {"type": "message", "data": json.dumps(login_event("ohai"))}
        subscribe_frame = {"type": "subscribe", "data": 1}

        channel = _SendableChannel(id=123, name="pickem-logger")
        client = _FakeClient(_SendableGuild([channel]))

        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            attempt = len(created)
            if attempt == 0:
                pubsub = _FakePubSub("drop")  # 1st connection: drops mid-listen
            elif attempt == 1:
                # 2nd connection: re-subscribes and delivers the held-back event.
                pubsub = _FakePubSub([subscribe_frame, login_frame])
            else:  # pragma: no cover - guards against an unbounded reconnect loop
                raise AssertionError("run_notifier reconnected more than once")
            redis_client = _FakeRedis(pubsub)
            created.append(redis_client)
            return redis_client

        with (
            mock.patch("redis.asyncio.from_url", fake_from_url),
            mock.patch("app.bot.notifier.get_settings", lambda: _FakeSettings()),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_START", 0),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_MAX", 0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_notifier(client)

        # Reconnected exactly once (drop -> reconnect), proving the outer loop.
        self.assertEqual(len(created), 2)
        # The event published while "reconnecting" was delivered after re-subscribe.
        self.assertEqual(channel.sent, ["ohai logged in"])
        # Both connections were re-subscribed and cleaned up.
        self.assertEqual(created[0]._pubsub.subscribed, [EVENTS_CHANNEL])
        self.assertEqual(created[1]._pubsub.subscribed, [EVENTS_CHANNEL])
        self.assertTrue(created[0]._pubsub.closed)
        self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
