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

from app.bot.notifier import _render, render_chat, resolve_channel, run_notifier
from app.services.notifications import (
    EVENTS_CHANNEL,
    admin_pick_cleared_event,
    admin_pick_set_event,
    freeze_week_event,
    game_final_event,
    ingest_season_event,
    login_event,
    misc_graded_event,
    misc_picked_event,
    pick_cleared_event,
    pick_event,
    player_registered_event,
    roster_complete_event,
    week_recap_event,
    window_closed_event,
    window_opened_event,
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
        self.assertEqual(_render(event), "admin cleared alice · Week 3 · FAVORITE_COVER")

    def test_render_player_registered(self) -> None:
        self.assertEqual(_render(player_registered_event("newbie")), "new player: newbie")

    def test_render_ingest_season(self) -> None:
        event = ingest_season_event(season=2026, weeks=18, games=272, failed=1)
        self.assertEqual(_render(event), "ingested 2026 · 18 wk / 272 games (1 failed)")

    def test_render_freeze_week(self) -> None:
        self.assertEqual(_render(freeze_week_event(week=3)), "Week 3 lines frozen")

    def test_render_unknown_type_returns_none(self) -> None:
        self.assertIsNone(_render({"v": 1, "type": "totally.unknown"}))


class RenderChatTests(unittest.TestCase):
    """``render_chat`` renders the five player-facing chat lines (the LLM seam)."""

    def test_render_roster_complete(self) -> None:
        line = render_chat(roster_complete_event(actor="Bob", week=3))
        self.assertIsNotNone(line)
        self.assertIn("Bob", line)
        self.assertIn("3", line)

    def test_render_roster_complete_no_locked_in_and_no_lock_emoji(self) -> None:
        """The reworded roster line drops "locked in" framing AND the lock emoji."""
        line = render_chat(roster_complete_event(actor="Bob", week=3))
        self.assertIsNotNone(line)
        self.assertNotIn("locked in", line)
        self.assertNotIn("\U0001f512", line)  # 🔒 lock emoji

    def test_render_misc_picked_is_leak_safe(self) -> None:
        """misc.picked renders a content-free line from actor + week only."""
        line = render_chat(misc_picked_event(actor="Bob", week=3))
        self.assertIsNotNone(line)
        self.assertIn("Bob", line)
        self.assertIn("3", line)

    def test_render_window_opened(self) -> None:
        line = render_chat(window_opened_event(week=3))
        self.assertIsNotNone(line)
        self.assertIn("3", line)

    def test_render_window_closed(self) -> None:
        line = render_chat(window_closed_event(week=3))
        self.assertIsNotNone(line)
        self.assertIn("3", line)

    def test_render_game_final(self) -> None:
        line = render_chat(
            game_final_event(week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27)
        )
        self.assertIsNotNone(line)
        self.assertIn("KC", line)
        self.assertIn("LAC", line)
        self.assertIn("27", line)
        self.assertIn("20", line)

    def test_render_week_recap(self) -> None:
        line = render_chat(
            week_recap_event(week=3, winner="Carol", winner_score=6, leader="Dave", leader_score=18)
        )
        self.assertIsNotNone(line)
        self.assertIn("Carol", line)
        self.assertIn("Dave", line)
        self.assertIn("6", line)

    def test_render_misc_graded_correct_signed_points(self) -> None:
        line = render_chat(
            misc_graded_event(
                actor="Bob",
                week=3,
                prediction="Mahomes throws 4 TDs",
                verdict="correct",
                points=3,
            )
        )
        self.assertIsNotNone(line)
        self.assertIn("Bob", line)
        self.assertIn("3", line)
        self.assertIn("Mahomes throws 4 TDs", line)
        self.assertIn("correct", line)
        # Points are SIGNED — a positive shows a leading plus.
        self.assertIn("+3", line)

    def test_render_misc_graded_incorrect_negative_points(self) -> None:
        line = render_chat(
            misc_graded_event(
                actor="Carol",
                week=5,
                prediction="a bold call",
                verdict="incorrect",
                points=-2,
            )
        )
        self.assertIsNotNone(line)
        self.assertIn("Carol", line)
        self.assertIn("a bold call", line)
        self.assertIn("incorrect", line)
        self.assertIn("-2", line)

    def test_render_chat_unknown_type_returns_none(self) -> None:
        self.assertIsNone(render_chat({"v": 1, "type": "totally.unknown"}))


class _SendableChannel:
    """A channel exposing the ``.id``/``.name`` the resolver reads plus an async
    ``.send`` that records the lines posted — no real Discord object needed.

    Accepts BOTH the text form (``send(line, ...)``) and the embed form
    (``send(embed=..., ...)``): text lines land in ``sent`` and embeds in
    ``embeds`` so a test can assert on either. ``send_kwargs`` records every call's
    keyword args (e.g. ``allowed_mentions``)."""

    def __init__(self, id: int, name: str) -> None:  # noqa: A002
        self.id = id
        self.name = name
        self.sent: list[str] = []
        self.embeds: list = []
        self.send_kwargs: list[dict] = []

    async def send(self, line: str | None = None, *, embed=None, **kwargs) -> None:  # noqa: ANN001
        if embed is not None:
            self.embeds.append(embed)
        elif line is not None:
            self.sent.append(line)
        self.send_kwargs.append(kwargs)


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
    discord_chat_channel = "pickem-chat"


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


class RunNotifierRoutingTests(unittest.IsolatedAsyncioTestCase):
    """``run_notifier`` routes by the event's ``targets``: chat-targeted events go
    to ``discord_chat_channel``; logger-targeted events go to
    ``discord_chat_log_channel`` — both within ``DISCORD_GUILD_ID``."""

    async def test_chat_and_logger_events_route_to_their_channels(self) -> None:
        subscribe_frame = {"type": "subscribe", "data": 1}
        chat_frame = {
            "type": "message",
            "data": json.dumps(window_opened_event(week=3)),
        }
        logger_frame = {
            "type": "message",
            "data": json.dumps(login_event("ohai")),
        }

        logger_channel = _SendableChannel(id=123, name="pickem-logger")
        chat_channel = _SendableChannel(id=456, name="pickem-chat")
        client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))

        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            if created:  # pragma: no cover - guards an unbounded reconnect loop
                raise AssertionError("run_notifier reconnected unexpectedly")
            pubsub = _FakePubSub([subscribe_frame, chat_frame, logger_frame])
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

        # The chat event landed in pickem-chat; the logger event in pickem-logger.
        self.assertEqual(len(chat_channel.sent), 1)
        self.assertEqual(len(logger_channel.sent), 1)
        self.assertIn("3", chat_channel.sent[0])  # window.opened for week 3
        self.assertEqual(logger_channel.sent, ["ohai logged in"])


class RunNotifierEmbellishTests(unittest.IsolatedAsyncioTestCase):
    """The three Tier-1 chat events route through ``embellish_chat``: an LLM line
    is posted when the client is configured, the deterministic line on a None, and
    exactly ONE line lands per event. ``window.closed`` is untouched (no embellish
    call) and the chat send suppresses mass mentions."""

    async def _drive(self, frames, phrase_value):
        """Drive run_notifier over ``frames`` with chat_personality.llm_client.phrase
        patched to return ``phrase_value`` (a str, None, or a raising side effect)."""
        from app.bot import chat_personality

        subscribe_frame = {"type": "subscribe", "data": 1}
        logger_channel = _SendableChannel(id=123, name="pickem-logger")
        chat_channel = _SendableChannel(id=456, name="pickem-chat")
        client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))

        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            if created:  # pragma: no cover - guards an unbounded reconnect loop
                raise AssertionError("run_notifier reconnected unexpectedly")
            pubsub = _FakePubSub([subscribe_frame, *frames])
            redis_client = _FakeRedis(pubsub)
            created.append(redis_client)
            return redis_client

        if isinstance(phrase_value, BaseException):

            async def _phrase(fact_text, *, system_prompt):
                raise phrase_value
        else:

            async def _phrase(fact_text, *, system_prompt):
                return phrase_value

        with (
            mock.patch("redis.asyncio.from_url", fake_from_url),
            mock.patch("app.bot.notifier.get_settings", lambda: _FakeSettings()),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_START", 0),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_MAX", 0),
            mock.patch.object(chat_personality.llm_client, "phrase", _phrase),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_notifier(client)
        return logger_channel, chat_channel

    async def test_window_opened_posts_llm_line_when_configured(self) -> None:
        frame = {"type": "message", "data": json.dumps(window_opened_event(week=3))}
        _, chat = await self._drive([frame], "WEEK 3 IS LIVE 🏈")
        self.assertEqual(chat.sent, ["WEEK 3 IS LIVE 🏈"])

    async def test_window_opened_falls_back_to_deterministic_on_none(self) -> None:
        event = window_opened_event(week=3)
        frame = {"type": "message", "data": json.dumps(event)}
        _, chat = await self._drive([frame], None)
        self.assertEqual(chat.sent, [render_chat(event)])

    async def test_game_final_posts_embed_with_llm_quip_in_description(self) -> None:
        # game.final (260703-piv) posts a RICH EMBED, not a text line: the LLM quip
        # rides in the embed description after the deterministic score line.
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        frame = {"type": "message", "data": json.dumps(event)}
        _, chat = await self._drive([frame], "KC takes it 🔥")
        self.assertEqual(chat.sent, [])  # no plain text line
        self.assertEqual(len(chat.embeds), 1)
        self.assertIn("KC takes it 🔥", chat.embeds[0].description or "")
        # Mention hygiene still applies on the embed send.
        import discord

        am = chat.send_kwargs[0]["allowed_mentions"]
        self.assertIsInstance(am, discord.AllowedMentions)
        self.assertFalse(am.everyone)

    async def test_roster_complete_falls_back_to_deterministic_on_none(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        frame = {"type": "message", "data": json.dumps(event)}
        _, chat = await self._drive([frame], None)
        self.assertEqual(chat.sent, [render_chat(event)])

    async def test_llm_raise_does_not_kill_loop_and_posts_deterministic(self) -> None:
        event = window_opened_event(week=3)
        frame = {"type": "message", "data": json.dumps(event)}
        _, chat = await self._drive([frame], RuntimeError("llm exploded"))
        # The loop survived (CancelledError still raised) and the deterministic
        # line still posted exactly once.
        self.assertEqual(chat.sent, [render_chat(event)])

    async def test_chat_send_suppresses_mass_mentions(self) -> None:
        import discord

        frame = {"type": "message", "data": json.dumps(window_opened_event(week=3))}
        _, chat = await self._drive([frame], "WEEK 3 IS LIVE 🏈")
        self.assertEqual(len(chat.send_kwargs), 1)
        am = chat.send_kwargs[0].get("allowed_mentions")
        # AllowedMentions has no __eq__, so compare the suppressing flags directly.
        self.assertIsInstance(am, discord.AllowedMentions)
        self.assertFalse(am.everyone)
        self.assertFalse(am.users)
        self.assertFalse(am.roles)

    async def test_window_closed_unchanged_and_no_embellish_call(self) -> None:
        from app.bot import chat_personality

        event = window_closed_event(week=3)
        frame = {"type": "message", "data": json.dumps(event)}

        called = {"embellish": False}
        real_embellish = chat_personality.embellish_chat

        async def _spy(ev):
            called["embellish"] = True
            return await real_embellish(ev)

        logger_channel = _SendableChannel(id=123, name="pickem-logger")
        chat_channel = _SendableChannel(id=456, name="pickem-chat")
        client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))
        subscribe_frame = {"type": "subscribe", "data": 1}
        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            if created:  # pragma: no cover
                raise AssertionError("reconnected unexpectedly")
            pubsub = _FakePubSub([subscribe_frame, frame])
            redis_client = _FakeRedis(pubsub)
            created.append(redis_client)
            return redis_client

        async def _no_commentary(week):  # build_lock_commentary stub (no db)
            return ["streak line"]

        with (
            mock.patch("redis.asyncio.from_url", fake_from_url),
            mock.patch("app.bot.notifier.get_settings", lambda: _FakeSettings()),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_START", 0),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_MAX", 0),
            mock.patch("app.bot.chat_personality.embellish_chat", _spy),
            mock.patch("app.bot.commentary.build_lock_commentary", _no_commentary),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_notifier(client)

        # Deterministic lock line + the commentary line posted; embellish NOT called.
        self.assertEqual(chat_channel.sent, [render_chat(event), "streak line"])
        self.assertFalse(called["embellish"])


class RunNotifierRecapTests(unittest.IsolatedAsyncioTestCase):
    """The ``week.recap`` event routes through the Tier-2 ``build_week_recap``
    orchestrator: the LLM-narrated column is posted when the orchestrator yields one,
    and the deterministic ``render_chat`` one-liner on fallback — exactly ONE line
    lands per event and mention hygiene is unchanged."""

    async def _drive(self, frames, build_recap):
        """Drive run_notifier over ``frames`` with app.bot.recap.build_week_recap
        patched to ``build_recap`` (an async fn)."""
        subscribe_frame = {"type": "subscribe", "data": 1}
        logger_channel = _SendableChannel(id=123, name="pickem-logger")
        chat_channel = _SendableChannel(id=456, name="pickem-chat")
        client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))

        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            if created:  # pragma: no cover - guards an unbounded reconnect loop
                raise AssertionError("run_notifier reconnected unexpectedly")
            pubsub = _FakePubSub([subscribe_frame, *frames])
            redis_client = _FakeRedis(pubsub)
            created.append(redis_client)
            return redis_client

        with (
            mock.patch("redis.asyncio.from_url", fake_from_url),
            mock.patch("app.bot.notifier.get_settings", lambda: _FakeSettings()),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_START", 0),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_MAX", 0),
            mock.patch("app.bot.recap.build_week_recap", build_recap),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_notifier(client)
        return logger_channel, chat_channel

    def _event(self):
        return week_recap_event(
            week=3, winner="alice", winner_score=9, leader="alice", leader_score=30
        )

    async def test_posts_llm_column_when_configured(self) -> None:
        frame = {"type": "message", "data": json.dumps(self._event())}

        async def _build(event):
            return "Big week for alice; she leads the season 🏈"

        _, chat = await self._drive([frame], _build)
        self.assertEqual(chat.sent, ["Big week for alice; she leads the season 🏈"])

    async def test_falls_back_to_deterministic_one_liner(self) -> None:
        event = self._event()
        frame = {"type": "message", "data": json.dumps(event)}

        # The orchestrator's own fallback path: return render_chat(event) itself.
        async def _build(ev):
            return render_chat(ev)

        _, chat = await self._drive([frame], _build)
        self.assertEqual(chat.sent, [render_chat(event)])

    async def test_recap_send_suppresses_mass_mentions(self) -> None:
        import discord

        frame = {"type": "message", "data": json.dumps(self._event())}

        async def _build(event):
            return "recap column"

        _, chat = await self._drive([frame], _build)
        self.assertEqual(len(chat.send_kwargs), 1)
        am = chat.send_kwargs[0].get("allowed_mentions")
        self.assertIsInstance(am, discord.AllowedMentions)
        self.assertFalse(am.everyone)
        self.assertFalse(am.users)
        self.assertFalse(am.roles)


class RunNotifierDecorateTests(unittest.IsolatedAsyncioTestCase):
    """The notifier decorates ONLY chat-channel lines with team-logo emojis
    (260627-wt5): a chat line with a team abbreviation gains its logo before send;
    the logger line is verbatim; an empty emoji cache leaves chat lines unchanged."""

    def setUp(self) -> None:
        from app.bot import team_emoji

        team_emoji.reset_emoji_cache()

    def tearDown(self) -> None:
        from app.bot import team_emoji

        team_emoji.reset_emoji_cache()

    async def _drive(self, frames):
        subscribe_frame = {"type": "subscribe", "data": 1}
        logger_channel = _SendableChannel(id=123, name="pickem-logger")
        chat_channel = _SendableChannel(id=456, name="pickem-chat")
        client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))

        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            if created:  # pragma: no cover - guards an unbounded reconnect loop
                raise AssertionError("run_notifier reconnected unexpectedly")
            pubsub = _FakePubSub([subscribe_frame, *frames])
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
        return logger_channel, chat_channel

    @staticmethod
    class _FakeEmoji:
        def __init__(self, name, id):  # noqa: A002
            self.name = name
            self.id = id

        def __str__(self):
            return f"<:{self.name}:{self.id}>"

    async def test_chat_line_decorated_with_team_logo(self) -> None:
        from app.bot import team_emoji

        team_emoji.populate_emoji_cache([self._FakeEmoji("chiefs", 7)])
        # game.final (260703-piv) posts an EMBED: the deterministic score line in the
        # description carries the KC logo directly via resolve_logo.
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=27, home_score=20
        )
        frame = {"type": "message", "data": json.dumps(event)}
        _, chat = await self._drive([frame])
        self.assertEqual(len(chat.embeds), 1)
        # The KC token is followed by the chiefs logo string in the embed body.
        self.assertIn("KC <:chiefs:7>", chat.embeds[0].description or "")

    async def test_logger_line_not_decorated(self) -> None:
        from app.bot import team_emoji

        # A KC entry would decorate IF the logger path ran through the decorator;
        # it must not. Use a logger event whose line happens to contain an abbr.
        team_emoji.populate_emoji_cache([self._FakeEmoji("chiefs", 7)])
        event = pick_event("pick.created", actor="bob", week=3, detail="OVER KC")
        frame = {"type": "message", "data": json.dumps(event)}
        logger, _ = await self._drive([frame])
        self.assertEqual(logger.sent, ["bob pick · Week 3 · OVER KC"])
        self.assertNotIn("<:chiefs:7>", logger.sent[0])

    async def test_empty_cache_leaves_chat_line_unchanged(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=27, home_score=20
        )
        frame = {"type": "message", "data": json.dumps(event)}
        _, chat = await self._drive([frame])
        # Embed still built with an empty cache — just no logo tokens in the body.
        self.assertEqual(len(chat.embeds), 1)
        self.assertNotIn("<:", chat.embeds[0].description or "")

    async def test_window_closed_lock_line_and_commentary_decorated(self) -> None:
        from app.bot import team_emoji

        team_emoji.populate_emoji_cache([self._FakeEmoji("vikingslogo", 9)])

        async def _commentary(week):
            return ["MIN are streaking"]

        event = window_closed_event(week=3)
        frame = {"type": "message", "data": json.dumps(event)}

        with mock.patch("app.bot.commentary.build_lock_commentary", _commentary):
            subscribe_frame = {"type": "subscribe", "data": 1}
            logger_channel = _SendableChannel(id=123, name="pickem-logger")
            chat_channel = _SendableChannel(id=456, name="pickem-chat")
            client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))
            created: list[_FakeRedis] = []

            def fake_from_url(_url):  # noqa: ANN001
                if created:  # pragma: no cover
                    raise AssertionError("reconnected unexpectedly")
                pubsub = _FakePubSub([subscribe_frame, frame])
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

        # The commentary extra line carrying MIN gains the vikings logo.
        self.assertEqual(len(chat_channel.sent), 2)
        self.assertIn("MIN <:vikingslogo:9>", chat_channel.sent[1])


class RunNotifierGameFinalFallbackTests(unittest.IsolatedAsyncioTestCase):
    """game.final embed is BEST-EFFORT (T-piv-02): if the embed build raises for
    ANY reason, the notifier still posts the text line and the loop never dies."""

    async def test_embed_build_failure_falls_back_to_text_send(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        frame = {"type": "message", "data": json.dumps(event)}
        subscribe_frame = {"type": "subscribe", "data": 1}

        logger_channel = _SendableChannel(id=123, name="pickem-logger")
        chat_channel = _SendableChannel(id=456, name="pickem-chat")
        client = _FakeClient(_SendableGuild([logger_channel, chat_channel]))
        created: list[_FakeRedis] = []

        def fake_from_url(_url):  # noqa: ANN001
            if created:  # pragma: no cover - guards an unbounded reconnect loop
                raise AssertionError("reconnected unexpectedly")
            pubsub = _FakePubSub([subscribe_frame, frame])
            redis_client = _FakeRedis(pubsub)
            created.append(redis_client)
            return redis_client

        def _boom(*_a, **_k):
            raise RuntimeError("embed construction blew up")

        with (
            mock.patch("redis.asyncio.from_url", fake_from_url),
            mock.patch("app.bot.notifier.get_settings", lambda: _FakeSettings()),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_START", 0),
            mock.patch("app.bot.notifier._RECONNECT_BACKOFF_MAX", 0),
            mock.patch("app.bot.game_final_embed.build_game_final_embed", _boom),
        ):
            # The loop still ends only via the shutdown CancelledError — it survived.
            with self.assertRaises(asyncio.CancelledError):
                await run_notifier(client)

        # No embed landed; a TEXT line did (the best-effort fallback).
        self.assertEqual(chat_channel.embeds, [])
        self.assertEqual(len(chat_channel.sent), 1)
        # Fallback send still suppresses mass mentions.
        import discord

        am = chat_channel.send_kwargs[-1]["allowed_mentions"]
        self.assertIsInstance(am, discord.AllowedMentions)
        self.assertFalse(am.everyone)


if __name__ == "__main__":
    unittest.main()
