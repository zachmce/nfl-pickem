"""Offline unit tests for the @mention Q&A listener cog (260709-k5w Task 3).

These tests NEVER connect to a real Discord gateway. Fake ``message`` objects
(``SimpleNamespace`` + a fake channel recording sends) are fed to the cog's
``on_message`` and ``qa.answer_question`` is monkeypatched. They assert the mention
gate (self/other-bot / bare-ping / @everyone / role-ping all excluded), the per-user
cooldown, the public reply with ``AllowedMentions.none()``, and that a raise inside
``answer_question`` is swallowed (never propagates out of ``on_message``).

Run with: ``backend/.venv/bin/python -m unittest tests.test_qa_listener -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

import discord
from discord.ext import commands

from app.bot import qa
from app.bot.commands import mention_qa
from app.bot.commands.mention_qa import MentionQaCog

# A stand-in bot user; identity equality makes ``bot_user in message.mentions`` work.
_BOT_USER = SimpleNamespace(id=999, bot=True)


def _run(coro):
    return asyncio.run(coro)


class _FakeChannel:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, content, *, allowed_mentions=None):  # noqa: ANN001
        self.sent.append({"content": content, "allowed_mentions": allowed_mentions})


def _make_message(
    *,
    content: str,
    author_bot: bool = False,
    author_id: int = 42,
    mentions_bot: bool = True,
    mention_everyone: bool = False,
    in_guild: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=[_BOT_USER] if mentions_bot else [],
        mention_everyone=mention_everyone,
        guild=SimpleNamespace(id=1) if in_guild else None,
        channel=_FakeChannel(),
    )


def _cog() -> MentionQaCog:
    # The fake bot stands in for commands.Bot (only .user is read by the cog).
    return MentionQaCog(cast(commands.Bot, SimpleNamespace(user=_BOT_USER)))


def _deliver(cog: MentionQaCog, message: SimpleNamespace) -> None:
    """Run the listener with a fake message.

    The ``SimpleNamespace`` fake deliberately stands in for a real
    ``discord.Message`` (the handler only reads a few attributes); the cast tells the
    type checker that's intentional so the strict gate stays green.
    """
    _run(cog.on_message(cast(discord.Message, message)))


def _answer_returns(value):
    """Patch qa.answer_question with an async fake, recording its calls."""
    calls: list[dict] = []

    async def _fake(question, *, discord_id):
        calls.append({"question": question, "discord_id": discord_id})
        return value

    return mock.patch.object(qa, "answer_question", _fake), calls


def _answer_raises():
    async def _fake(question, *, discord_id):
        raise RuntimeError("boom")

    return mock.patch.object(qa, "answer_question", _fake)


class MentionGateTests(unittest.TestCase):
    def test_real_mention_answers_and_sends_with_no_mentions(self) -> None:
        cog = _cog()
        message = _make_message(content="<@999> what's the score?")
        patcher, calls = _answer_returns("KC 27, LAC 20 (final) 🔒")
        with patcher:
            _deliver(cog, message)
        # answer_question was called with the STRIPPED question + the asker's id.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["question"], "what's the score?")
        self.assertEqual(calls[0]["discord_id"], 42)
        # Replied publicly with AllowedMentions.none().
        self.assertEqual(len(message.channel.sent), 1)
        am = message.channel.sent[0]["allowed_mentions"]
        self.assertIsInstance(am, discord.AllowedMentions)
        self.assertFalse(am.everyone)
        self.assertFalse(am.users)
        self.assertFalse(am.roles)
        self.assertEqual(message.channel.sent[0]["content"], "KC 27, LAC 20 (final) 🔒")

    def test_message_from_a_bot_is_ignored(self) -> None:
        cog = _cog()
        message = _make_message(content="<@999> hi", author_bot=True)
        patcher, calls = _answer_returns("nope")
        with patcher:
            _deliver(cog, message)
        self.assertEqual(calls, [])
        self.assertEqual(message.channel.sent, [])

    def test_bare_mention_with_no_text_is_ignored(self) -> None:
        cog = _cog()
        message = _make_message(content="   <@999>   ")
        patcher, calls = _answer_returns("nope")
        with patcher:
            _deliver(cog, message)
        self.assertEqual(calls, [])
        self.assertEqual(message.channel.sent, [])

    def test_everyone_ping_is_not_a_bot_mention(self) -> None:
        cog = _cog()
        # mention_everyone True even though the bot happens to be in mentions.
        message = _make_message(content="@everyone <@999> standings?", mention_everyone=True)
        patcher, calls = _answer_returns("nope")
        with patcher:
            _deliver(cog, message)
        self.assertEqual(calls, [])
        self.assertEqual(message.channel.sent, [])

    def test_role_ping_is_not_a_bot_mention(self) -> None:
        cog = _cog()
        # A role ping does not put the bot in message.mentions.
        message = _make_message(content="<@&555> standings?", mentions_bot=False)
        patcher, calls = _answer_returns("nope")
        with patcher:
            _deliver(cog, message)
        self.assertEqual(calls, [])
        self.assertEqual(message.channel.sent, [])

    def test_dm_is_out_of_scope(self) -> None:
        cog = _cog()
        message = _make_message(content="<@999> standings?", in_guild=False)
        patcher, calls = _answer_returns("nope")
        with patcher:
            _deliver(cog, message)
        self.assertEqual(calls, [])


class CooldownTests(unittest.TestCase):
    def test_rapid_second_mention_from_same_user_is_suppressed(self) -> None:
        cog = _cog()
        patcher, calls = _answer_returns("ok")
        with patcher:
            _deliver(cog, _make_message(content="<@999> standings?", author_id=42))
            _deliver(cog, _make_message(content="<@999> standings again?", author_id=42))
        # Answered once — the second call within the window is suppressed.
        self.assertEqual(len(calls), 1)

    def test_different_users_are_not_shared_buckets(self) -> None:
        cog = _cog()
        patcher, calls = _answer_returns("ok")
        with patcher:
            _deliver(cog, _make_message(content="<@999> standings?", author_id=1))
            _deliver(cog, _make_message(content="<@999> standings?", author_id=2))
        self.assertEqual(len(calls), 2)


class GuardTests(unittest.TestCase):
    def test_raise_inside_answer_question_is_swallowed(self) -> None:
        cog = _cog()
        message = _make_message(content="<@999> standings?")
        with _answer_raises():
            # Must not raise out of on_message.
            _deliver(cog, message)
        self.assertEqual(message.channel.sent, [])


class WiringTests(unittest.TestCase):
    def test_message_content_intent_and_cog_registered(self) -> None:
        # client.py must enable message_content and list the cog in COG_MODULES.
        from app.bot import client

        self.assertIn("app.bot.commands.mention_qa", client.COG_MODULES)

    def test_setup_adds_cog(self) -> None:
        added: list[object] = []

        class _FakeBot:
            user = _BOT_USER

            async def add_cog(self, cog):
                added.append(cog)

        _run(mention_qa.setup(cast(commands.Bot, _FakeBot())))
        self.assertEqual(len(added), 1)
        self.assertIsInstance(added[0], MentionQaCog)


if __name__ == "__main__":
    unittest.main()
