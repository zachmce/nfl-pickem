"""Offline unit tests for the @mention Q&A listener cog (260709-k5w Task 3).

These tests NEVER connect to a real Discord gateway. Fake ``message`` objects
(``SimpleNamespace`` + a fake channel recording sends) are fed to the cog's
``on_message`` and ``qa.answer_question`` is monkeypatched. They assert the mention
gate (self/other-bot / bare-ping / @everyone / role-ping all excluded), the per-user
cooldown, the public reply with ``AllowedMentions.none()``, and that a raise inside
``answer_question`` is swallowed (never propagates out of ``on_message``).

260820-lw6 extends the harness with a channel id, an optional message reference and an
author display name, and adds the READ-THE-ROOM cases: an explicit @mention (and a
Discord reply to one of the bot's own messages) is answered WITHOUT ever consulting the
model gate, a cold channel never consults it either, and a gate-rejected message must
not burn the asker's cooldown bucket. ``qa_room.is_addressed`` is patched with a SPY
that records call counts, so "the gate was never called" is a real assertion.

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

from app.bot import qa, qa_room
from app.bot.commands import mention_qa
from app.bot.commands.mention_qa import MentionQaCog

# A stand-in bot user; identity equality makes ``bot_user in message.mentions`` work.
_BOT_USER = SimpleNamespace(id=999, bot=True, display_name="Pick'em Bot")


def _run(coro):
    return asyncio.run(coro)


class _FakeTyping:
    """A minimal async context manager standing in for ``channel.typing()``.

    Records enter/exit on the owning channel so a test can assert the answer + send ran
    INSIDE the typing indicator (the production path uses ``async with
    message.channel.typing():``). Added to the fake rather than loosening production code.
    """

    def __init__(self, channel: "_FakeChannel") -> None:
        self._channel = channel

    async def __aenter__(self) -> "_FakeTyping":
        self._channel.typing_entered = True
        # The send must happen while typing is active — nothing sent yet at enter.
        self._channel.sent_at_typing_enter = len(self._channel.sent)
        return self

    async def __aexit__(self, *exc) -> None:  # noqa: ANN002
        self._channel.typing_exited = True


class _FakeChannel:
    def __init__(self, channel_id: int = 77) -> None:
        self.id = channel_id
        self.sent: list[dict] = []
        self.typing_entered = False
        self.typing_exited = False
        self.sent_at_typing_enter: int | None = None

    def typing(self) -> "_FakeTyping":
        return _FakeTyping(self)

    async def send(self, content, *, allowed_mentions=None, suppress_embeds=False):  # noqa: ANN001
        self.sent.append(
            {
                "content": content,
                "allowed_mentions": allowed_mentions,
                "suppress_embeds": suppress_embeds,
            }
        )


def _make_message(
    *,
    content: str,
    author_bot: bool = False,
    author_id: int = 42,
    author_name: str = "Ada",
    mentions_bot: bool = True,
    mention_everyone: bool = False,
    in_guild: bool = True,
    channel_id: int = 77,
    reply_to_author_id: int | None = None,
) -> SimpleNamespace:
    """Build a fake message.

    ``reply_to_author_id`` populates a Discord ``reference`` whose ``resolved`` message
    was authored by that id — the reply-to-the-bot fast path resolves it defensively
    (the reference may be absent, unresolved, or point at a deleted message).
    """
    reference = None
    if reply_to_author_id is not None:
        reference = SimpleNamespace(
            resolved=SimpleNamespace(author=SimpleNamespace(id=reply_to_author_id))
        )
    return SimpleNamespace(
        content=content,
        author=SimpleNamespace(id=author_id, bot=author_bot, display_name=author_name),
        mentions=[_BOT_USER] if mentions_bot else [],
        mention_everyone=mention_everyone,
        guild=SimpleNamespace(id=1) if in_guild else None,
        channel=_FakeChannel(channel_id),
        reference=reference,
    )


def _gate(verdict: bool = True):
    """Patch ``qa_room.is_addressed`` with a SPY recording every call.

    Returns ``(patcher, calls)``; ``len(calls)`` is what makes the never-called
    assertions real assertions instead of inferences from the send list.
    """
    calls: list[dict] = []

    async def _fake(turns, *, bot_name):
        calls.append({"turns": list(turns), "bot_name": bot_name})
        return verdict

    return mock.patch.object(qa_room, "is_addressed", _fake), calls


def _gate_raises():
    async def _fake(turns, *, bot_name):
        raise RuntimeError("boom")

    return mock.patch.object(qa_room, "is_addressed", _fake)


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

    async def _fake(question, *, discord_id, history=(), conversation_key=None, asker_name=None):
        calls.append(
            {
                "question": question,
                "discord_id": discord_id,
                "history": list(history),
                "conversation_key": conversation_key,
                "asker_name": asker_name,
            }
        )
        return value

    return mock.patch.object(qa, "answer_question", _fake), calls


def _answer_raises():
    async def _fake(question, *, discord_id, history=(), conversation_key=None, asker_name=None):
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
        # Link embeds are suppressed so news source links don't unfurl into a card wall.
        self.assertTrue(message.channel.sent[0]["suppress_embeds"])

    def test_answer_and_send_run_inside_the_typing_indicator(self) -> None:
        cog = _cog()
        message = _make_message(content="<@999> who wins the Chiefs game?")
        patcher, _ = _answer_returns("Pick: KC to cover.")
        with patcher:
            _deliver(cog, message)
        # The typing indicator wrapped the work: entered, sent while active, then exited.
        self.assertTrue(message.channel.typing_entered)
        self.assertTrue(message.channel.typing_exited)
        self.assertEqual(message.channel.sent_at_typing_enter, 0)  # nothing sent before enter
        self.assertEqual(len(message.channel.sent), 1)  # the reply landed inside the block

    def test_guarded_out_message_never_shows_typing(self) -> None:
        # A bare ping short-circuits BEFORE any typing/LLM work (the indicator only shows
        # for a real answer).
        cog = _cog()
        message = _make_message(content="   <@999>   ")
        patcher, calls = _answer_returns("nope")
        with patcher:
            _deliver(cog, message)
        self.assertEqual(calls, [])
        self.assertFalse(message.channel.typing_entered)

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


class ReadTheRoomTests(unittest.TestCase):
    """The model gate replaces the individual-mention check — but it is consulted
    ONLY for a message that is neither an explicit mention nor a reply to the bot,
    and only in a channel the bot has spoken in recently."""

    def test_explicit_mention_answers_and_never_consults_the_gate(self) -> None:
        cog = _cog()
        message = _make_message(content="<@999> standings?")
        answer_patch, calls = _answer_returns("Ada leads by 7.")
        gate_patch, gate_calls = _gate(verdict=False)  # would say NO if asked
        with answer_patch, gate_patch:
            _deliver(cog, message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(gate_calls, [])  # the gate was NEVER called

    def test_reply_to_a_bot_message_answers_and_never_consults_the_gate(self) -> None:
        cog = _cog()
        message = _make_message(
            content="how long has he started?", mentions_bot=False, reply_to_author_id=999
        )
        answer_patch, calls = _answer_returns("Since 2024.")
        gate_patch, gate_calls = _gate(verdict=False)
        with answer_patch, gate_patch:
            _deliver(cog, message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(gate_calls, [])

    def test_reply_to_another_human_is_not_a_fast_path(self) -> None:
        cog = _cog()
        message = _make_message(content="agreed", mentions_bot=False, reply_to_author_id=1234)
        answer_patch, calls = _answer_returns("nope")
        gate_patch, gate_calls = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, message)
        # Cold channel: no bot reply yet, so the pre-filter stops it BEFORE the gate.
        self.assertEqual(calls, [])
        self.assertEqual(gate_calls, [])

    def test_cold_channel_non_mention_is_ignored_without_a_gate_call(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("nope")
        gate_patch, gate_calls = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="anyone watching?", mentions_bot=False))
        self.assertEqual(calls, [])
        self.assertEqual(gate_calls, [])  # the cheap deterministic pre-filter ran first

    def test_non_mention_after_a_recent_bot_reply_consults_the_gate_and_answers(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("Since 2024.")
        gate_patch, gate_calls = _gate(verdict=True)
        with answer_patch, gate_patch:
            # 1) a mention opens the conversation (records the bot's reply)
            _deliver(cog, _make_message(content="<@999> who starts at QB?", author_id=1))
            # 2) a bare follow-up from someone else
            _deliver(
                cog,
                _make_message(content="how long has he started?", mentions_bot=False, author_id=2),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(gate_calls), 1)  # consulted exactly once, for the follow-up
        self.assertEqual(gate_calls[0]["bot_name"], "Pick'em Bot")

    def test_gate_false_stays_silent(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("should not land")
        gate_patch, gate_calls = _gate(verdict=False)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> who starts at QB?", author_id=1))
            _deliver(
                cog,
                _make_message(content="I think the Bears stink", mentions_bot=False, author_id=2),
            )
        self.assertEqual(len(calls), 1)  # only the mention was answered
        self.assertEqual(len(gate_calls), 1)

    def test_a_raise_in_the_gate_never_escapes_on_message(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("nope")
        with answer_patch, _gate_raises():
            _deliver(cog, _make_message(content="<@999> hi", author_id=1))
            # Must not raise out of on_message.
            _deliver(cog, _make_message(content="follow up", mentions_bot=False, author_id=2))
        self.assertEqual(len(calls), 1)


class KeptGatesOnTheNonMentionPathTests(unittest.TestCase):
    """The four gates the design says to KEEP still hold now that a message no
    longer needs an @mention to reach the handler."""

    def _warm(self, cog, gate_patch) -> None:
        """Open the channel (a bot reply) so the pre-filter is not what stops us."""
        answer_patch, _ = _answer_returns("opening line")
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> hi", author_id=1))

    def test_bot_author_is_ignored(self) -> None:
        cog = _cog()
        gate_patch, gate_calls = _gate(verdict=True)
        self._warm(cog, gate_patch)
        answer_patch, calls = _answer_returns("nope")
        with answer_patch, gate_patch:
            _deliver(
                cog,
                _make_message(
                    content="beep boop", mentions_bot=False, author_bot=True, author_id=5
                ),
            )
        self.assertEqual(calls, [])
        self.assertEqual(len(gate_calls), 0)

    def test_dm_is_still_out_of_scope(self) -> None:
        cog = _cog()
        gate_patch, gate_calls = _gate(verdict=True)
        answer_patch, calls = _answer_returns("nope")
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="hello", mentions_bot=False, in_guild=False))
        self.assertEqual(calls, [])
        self.assertEqual(gate_calls, [])

    def test_everyone_ping_is_still_ignored(self) -> None:
        cog = _cog()
        gate_patch, gate_calls = _gate(verdict=True)
        self._warm(cog, gate_patch)
        answer_patch, calls = _answer_returns("nope")
        with answer_patch, gate_patch:
            _deliver(
                cog,
                _make_message(content="@everyone look", mentions_bot=False, mention_everyone=True),
            )
        self.assertEqual(calls, [])
        self.assertEqual(len(gate_calls), 0)

    def test_empty_text_is_still_ignored(self) -> None:
        cog = _cog()
        gate_patch, gate_calls = _gate(verdict=True)
        self._warm(cog, gate_patch)
        answer_patch, calls = _answer_returns("nope")
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="   ", mentions_bot=False))
        self.assertEqual(calls, [])
        self.assertEqual(len(gate_calls), 0)


class CooldownOrderingTests(unittest.TestCase):
    """``update_rate_limit`` MUTATES the bucket, so it must run AFTER the room gate —
    otherwise ordinary channel chatter burns the asker's bucket and their real
    question is dropped seconds later."""

    def test_gate_rejected_message_does_not_consume_the_bucket(self) -> None:
        cog = _cog()
        # Warm the channel with a DIFFERENT author so user 42's bucket is untouched.
        warm_patch, _ = _answer_returns("opening line")
        open_gate, _ = _gate(verdict=True)
        with warm_patch, open_gate:
            _deliver(cog, _make_message(content="<@999> hi", author_id=1))

        answer_patch, calls = _answer_returns("real answer")
        closed_gate, _ = _gate(verdict=False)
        with answer_patch, closed_gate:
            _deliver(cog, _make_message(content="chatter", mentions_bot=False, author_id=42))
        self.assertEqual(calls, [])

        # Immediately after the rejection, the SAME author's real question lands.
        with answer_patch, open_gate:
            _deliver(cog, _make_message(content="real question", mentions_bot=False, author_id=42))
        self.assertEqual(len(calls), 1)

    def test_an_answered_message_does_consume_the_bucket(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("answer")
        gate_patch, _ = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> hi", author_id=42))
            _deliver(cog, _make_message(content="again", mentions_bot=False, author_id=42))
        self.assertEqual(len(calls), 1)


class ChannelHistoryTests(unittest.TestCase):
    def test_answer_question_receives_history_excluding_the_current_question(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("Caleb Williams.")
        gate_patch, _ = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> who starts at QB?", author_id=1))
            _deliver(cog, _make_message(content="how long?", mentions_bot=False, author_id=2))
        # First answer: nothing said before it.
        self.assertEqual(calls[0]["history"], [])
        # Second answer: the prior question + the bot's own reply, current one EXCLUDED.
        history = calls[1]["history"]
        self.assertEqual([role for role, _ in history], ["user", "assistant"])
        self.assertEqual(history[0][1], "Ada: who starts at QB?")
        self.assertEqual(history[1][1], "Caleb Williams.")
        self.assertNotIn("how long?", [text for _, text in history])
        # The asker is named, so a follow-up resolves against THEIR earlier turns.
        self.assertEqual([call["asker_name"] for call in calls], ["Ada", "Ada"])
        # The channel id names the conversation the open path keeps its grounding under.
        self.assertEqual(calls[0]["conversation_key"], calls[1]["conversation_key"])
        self.assertIsInstance(calls[0]["conversation_key"], str)
        self.assertTrue(calls[0]["conversation_key"])

    def test_the_bots_own_reply_is_recorded_into_the_channel_transcript(self) -> None:
        cog = _cog()
        answer_patch, _ = _answer_returns("Caleb Williams.")
        gate_patch, gate_calls = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> who starts at QB?", author_id=1))
            _deliver(cog, _make_message(content="how long?", mentions_bot=False, author_id=2))
        turns = gate_calls[0]["turns"]
        # Without this stamp the pre-filter never opens and the gate never fires at all.
        self.assertIn(("Pick'em Bot", "Caleb Williams."), turns)
        # The transcript ends with the message being judged.
        self.assertEqual(turns[-1][1], "how long?")

    def test_history_is_per_channel(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("answer")
        gate_patch, _ = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> a?", author_id=1, channel_id=1))
            _deliver(cog, _make_message(content="<@999> b?", author_id=2, channel_id=2))
        self.assertEqual(calls[0]["history"], [])
        self.assertEqual(calls[1]["history"], [])  # channel 2 knows nothing of channel 1


class ConcurrentQuestionTests(unittest.TestCase):
    """Live 2026-09-20: two questions in one minute got the FIRST one answered twice."""

    def test_a_question_that_arrives_mid_answer_sees_that_answer_in_its_history(self) -> None:
        cog = _cog()
        calls: list[dict] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def _fake(
            question, *, discord_id, history=(), conversation_key=None, asker_name=None
        ):
            calls.append({"question": question, "history": list(history)})
            if question == "did Shough throw for 250?":
                first_started.set()
                await release_first.wait()
                return "Yes, 252 yards."
            return "It is on NBC."

        async def _scenario() -> None:
            first = asyncio.create_task(
                cog.on_message(
                    cast(
                        discord.Message,
                        _make_message(content="<@999> did Shough throw for 250?", author_id=1),
                    )
                )
            )
            await first_started.wait()
            second = asyncio.create_task(
                cog.on_message(
                    cast(
                        discord.Message,
                        _make_message(content="<@999> look up the channel number", author_id=2),
                    )
                )
            )
            # Let the second handler run up to the lock before the first one finishes.
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertEqual(len(calls), 1)  # the second answer has NOT started
            release_first.set()
            await asyncio.gather(first, second)

        with mock.patch.object(qa, "answer_question", _fake):
            _run(_scenario())
        self.assertEqual(
            calls[1]["history"],
            [("user", "Ada: did Shough throw for 250?"), ("assistant", "Yes, 252 yards.")],
        )

    def test_history_excludes_the_question_by_identity_not_by_text(self) -> None:
        memory = mention_qa._ChannelMemory()
        memory.record(1, "Ada", "who wins?")
        turn = memory.record(1, "Bo", "who wins?")
        self.assertEqual(memory.history(1, exclude=turn), [("user", "Ada: who wins?")])

    def test_channel_eviction_drops_the_lock(self) -> None:
        memory = mention_qa._ChannelMemory()
        memory.answer_lock(0)
        for channel_id in range(1, mention_qa._MEMORY_MAX_CHANNELS + 1):
            memory.record(channel_id, "Ada", "hi")
        self.assertNotIn(0, memory._locks)

    def test_channel_eviction_keeps_a_held_lock(self) -> None:
        memory = mention_qa._ChannelMemory()
        lock = memory.answer_lock(0)
        asyncio.run(lock.acquire())
        for channel_id in range(1, mention_qa._MEMORY_MAX_CHANNELS + 1):
            memory.record(channel_id, "Ada", "hi")
        self.assertIs(memory.answer_lock(0), lock)

    def test_forget_drops_one_turn_by_identity(self) -> None:
        memory = mention_qa._ChannelMemory()
        memory.record(1, "Ada", "who wins?")
        turn = memory.record(1, "Ada", "who wins?")
        memory.forget(1, turn)
        self.assertEqual(memory.history(1), [("user", "Ada: who wins?")])

    def test_a_rate_limited_question_leaves_no_turn_in_history(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("answer")
        with answer_patch:
            _deliver(cog, _make_message(content="<@999> first?", author_id=1))
            _deliver(cog, _make_message(content="<@999> second?", author_id=1))
            _deliver(cog, _make_message(content="<@999> third?", author_id=2))
        self.assertEqual(len(calls), 2)
        self.assertNotIn(("user", "Ada: second?"), calls[1]["history"])
        self.assertFalse(any("second?" in text for _role, text in calls[1]["history"]))

    def test_other_member_mentions_reach_the_answer_as_names(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("answer")
        with answer_patch:
            message = _make_message(content="<@999> what did <@!555> pick?", author_id=1)
            message.mentions = [_BOT_USER, SimpleNamespace(id=555, display_name="Austin")]
            _deliver(cog, message)
        self.assertEqual(calls[0]["question"], "what did @Austin pick?")


class MentionOfAnotherMemberTests(unittest.TestCase):
    def test_a_message_that_mentions_another_member_never_reaches_the_gate(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("answer")
        gate_patch, gate_calls = _gate(verdict=True)
        with answer_patch, gate_patch:
            _deliver(cog, _make_message(content="<@999> who starts?", author_id=1))
            other = _make_message(content="<@555> look at this", mentions_bot=False, author_id=2)
            other.mentions = [SimpleNamespace(id=555, bot=False)]
            _deliver(cog, other)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(gate_calls), 0)
        self.assertEqual(other.channel.sent, [])

    def test_a_message_that_mentions_the_bot_and_another_member_is_answered(self) -> None:
        cog = _cog()
        answer_patch, calls = _answer_returns("answer")
        gate_patch, gate_calls = _gate(verdict=False)
        with answer_patch, gate_patch:
            both = _make_message(content="<@999> tell <@555> who starts", author_id=1)
            both.mentions = [_BOT_USER, SimpleNamespace(id=555, bot=False)]
            _deliver(cog, both)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(gate_calls), 0)


class ChannelMemoryTests(unittest.TestCase):
    """The memory lives for the life of a long-running gateway process, so it is
    bounded in BOTH dimensions."""

    def test_turns_per_channel_are_bounded(self) -> None:
        memory = mention_qa._ChannelMemory()
        for i in range(mention_qa._MEMORY_MAX_TURNS + 10):
            memory.record(1, f"user{i}", f"message {i}")
        turns = memory.transcript(1)
        self.assertEqual(len(turns), mention_qa._MEMORY_MAX_TURNS)
        self.assertEqual(turns[-1][1], f"message {mention_qa._MEMORY_MAX_TURNS + 9}")

    def test_channel_count_is_bounded_with_least_recently_touched_eviction(self) -> None:
        memory = mention_qa._ChannelMemory()
        for channel_id in range(mention_qa._MEMORY_MAX_CHANNELS + 5):
            memory.record(channel_id, "Ada", "hi")
        self.assertLessEqual(len(memory._turns), mention_qa._MEMORY_MAX_CHANNELS)
        self.assertEqual(memory.transcript(0), [])  # the oldest channel was evicted
        self.assertNotEqual(memory.transcript(mention_qa._MEMORY_MAX_CHANNELS + 4), [])

    def test_spoke_recently_is_false_before_any_bot_reply(self) -> None:
        memory = mention_qa._ChannelMemory()
        memory.record(1, "Ada", "hi")
        self.assertFalse(memory.spoke_recently(1, now=0.0))

    def test_spoke_recently_expires_after_the_window(self) -> None:
        memory = mention_qa._ChannelMemory()
        memory.record_bot_reply(1, "Bot", "hi", now=1000.0)
        self.assertTrue(memory.spoke_recently(1, now=1000.0 + mention_qa._ROOM_RECENT_SECONDS - 1))
        self.assertFalse(memory.spoke_recently(1, now=1000.0 + mention_qa._ROOM_RECENT_SECONDS + 1))

    def test_history_maps_bot_turns_to_the_assistant_role(self) -> None:
        memory = mention_qa._ChannelMemory()
        memory.record(1, "Ada", "who starts?")
        memory.record_bot_reply(1, "Bot", "Caleb Williams.", now=0.0)
        self.assertEqual(
            memory.history(1), [("user", "Ada: who starts?"), ("assistant", "Caleb Williams.")]
        )


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


class SplitForDiscordTests(unittest.TestCase):
    """`_split_for_discord` keeps every chunk within Discord's 2000-char cap without
    cutting a line (nor a logo token inside one) mid-way."""

    def test_short_text_is_a_single_unchanged_chunk(self) -> None:
        text = "one line\nsecond line"
        self.assertEqual(mention_qa._split_for_discord(text), [text])

    def test_long_body_splits_on_line_boundaries_each_within_limit(self) -> None:
        # 40 lines of ~100 chars => ~4000 chars => must become >1 chunk.
        lines = [f"GAME {i:02d}: " + "x" * 90 for i in range(40)]
        text = "\n".join(lines)
        chunks = mention_qa._split_for_discord(text, limit=2000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 2000)
        # No line was split: rejoining all chunks reproduces the original exactly, and
        # every original line survives intact in some chunk.
        self.assertEqual("\n".join(chunks), text)
        for original_line in lines:
            self.assertTrue(any(original_line in c for c in chunks))

    def test_single_overlong_line_is_hard_sliced(self) -> None:
        # A pathological single line longer than the limit is sliced, never emitted whole.
        chunks = mention_qa._split_for_discord("z" * 4500, limit=2000)
        self.assertTrue(all(len(c) <= 2000 for c in chunks))
        self.assertEqual("".join(chunks), "z" * 4500)


class LongAnswerChunkingTests(unittest.TestCase):
    """End-to-end: a whole-slate answer that exceeds 2000 chars is delivered as multiple
    valid messages instead of 400-ing the send (regression for the slate_predictions
    over-length crash)."""

    def test_over_length_answer_sends_as_multiple_capped_messages(self) -> None:
        big = "\n".join(f"GAME {i:02d}: " + "y" * 90 for i in range(40))  # ~4000 chars
        patch, _calls = _answer_returns(big)
        cog = _cog()
        msg = _make_message(content="what are your picks this week?")
        with patch:
            _deliver(cog, msg)
        sent = msg.channel.sent
        self.assertGreater(len(sent), 1)  # split into multiple messages
        for entry in sent:
            self.assertLessEqual(len(entry["content"]), 2000)
            # Each chunk keeps the safe posting flags (no-ping + no link unfurl).
            self.assertIsInstance(entry["allowed_mentions"], discord.AllowedMentions)
            self.assertTrue(entry["suppress_embeds"])
        # Reassembling the chunks reproduces the full decorated answer (nothing dropped).
        self.assertEqual("\n".join(e["content"] for e in sent), big)


if __name__ == "__main__":
    unittest.main()


class ChannelTranscriptTests(unittest.TestCase):
    """Issue #252: every chat-channel message becomes one transcript entry."""

    def setUp(self) -> None:
        from app.services import bot_telemetry

        self.stored: list[dict] = []
        patchers = [
            mock.patch.object(bot_telemetry, "_store", self.stored.append),
            mock.patch.object(mention_qa.settings, "discord_chat_channel", "77"),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_an_answer_is_one_entry_with_its_message_and_decision(self) -> None:
        cog = _cog()
        patcher, _ = _answer_returns("KC 27, LAC 20")
        with patcher:
            _deliver(cog, _make_message(content="<@999> score?", author_name="Ada"))
        (entry,) = self.stored
        self.assertEqual(entry["decision"], "answered")
        self.assertEqual(entry["addressed_by"], "mention")
        self.assertEqual((entry["author"], entry["question"]), ("Ada", "score?"))
        self.assertEqual(entry["answer"], "KC 27, LAC 20")

    def test_chat_channel_chatter_is_logged_with_why_it_was_skipped(self) -> None:
        cog = _cog()
        patcher, calls = _answer_returns("x")
        with patcher:
            _deliver(cog, _make_message(content="anyone watching?", mentions_bot=False))
        self.assertEqual(calls, [])
        (entry,) = self.stored
        self.assertEqual(entry["decision"], "not_addressed:bot_not_recently_active")
        self.assertEqual(entry["question"], "anyone watching?")

    def test_chatter_in_another_channel_is_never_stored(self) -> None:
        cog = _cog()
        patcher, _ = _answer_returns("x")
        with patcher:
            _deliver(cog, _make_message(content="private talk", mentions_bot=False, channel_id=5))
        self.assertEqual(self.stored, [])

    def test_a_bot_event_post_is_logged_but_its_own_reply_is_not_twice(self) -> None:
        cog = _cog()
        post = _make_message(content="Week 3 is locked.", author_bot=True, author_id=999)
        post.id = 501
        _deliver(cog, post)
        cog._reply_ids.append(502)
        echo = _make_message(content="KC 27", author_bot=True, author_id=999)
        echo.id = 502
        _deliver(cog, echo)
        (entry,) = self.stored
        self.assertEqual((entry["kind"], entry["content"]), ("bot_post", "Week 3 is locked."))

    def test_a_cooldown_drop_is_logged(self) -> None:
        cog = _cog()
        patcher, _ = _answer_returns("ok")
        with patcher:
            _deliver(cog, _make_message(content="<@999> one"))
            _deliver(cog, _make_message(content="<@999> two"))
        self.assertEqual([e["decision"] for e in self.stored], ["answered", "cooldown"])
