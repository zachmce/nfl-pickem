"""MentionQaCog — the inbound @mention Q&A listener (Path A v1, 260709-k5w).

A thin Discord surface (mirrors :mod:`app.bot.commands.register`): this is the ONLY
Q&A module that imports ``discord``. All the brains live in the Discord-free
:mod:`app.bot.qa`; this cog only decides whether a message is a genuine user->bot
mention, enforces a per-user cooldown, hands the stripped question to
:func:`app.bot.qa.answer_question`, and posts the reply PUBLICLY with
``discord.AllowedMentions.none()`` so the LLM-authored text can never ping anyone.

Locked posture:
  - Bots/self ignored, @everyone/@here excluded, a bare ping (no text after stripping
    the mention) ignored, DMs out of scope (guild messages only). These are the FOUR
    kept gates.
  - READ THE ROOM (260820-lw6): the individual-mention requirement is replaced by a
    model judgement. An explicit @mention — and a Discord reply to one of the bot's
    own messages — is ALWAYS answered and never consults the gate. Anything else must
    first clear a cheap deterministic pre-filter (has the bot spoken in this channel
    within ``_ROOM_RECENT_SECONDS``?) and then :func:`app.bot.qa_room.is_addressed`,
    which fails closed to False.
  - Per-user ``CooldownMapping`` (BucketType.user) gates every ANSWER — deliberately
    AFTER the room gate, because ``update_rate_limit`` mutates the bucket.
  - A bounded per-channel memory carries the recent transcript, both to feed the gate
    and to give ``qa.answer_question`` conversation history for the open path.
  - Answers in one channel run ONE AT A TIME (a per-channel lock), so a question that
    arrives while an answer is in flight sees that answer in its history.
  - The whole handler body is guarded (structlog + swallow): one bad message must
    never crash the gateway loop (``qa.answer_question`` and ``qa_room.is_addressed``
    are themselves best-effort, but the send / decorate path is guarded here too).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from collections.abc import Sequence

import discord
import structlog
from discord.ext import commands

from app.bot import qa, qa_room
from app.bot.team_emoji import decorate_team_logos
from app.config import settings
from app.services import bot_telemetry

logger = structlog.get_logger(__name__)

# Per-user cooldown window. Each answer triggers several local-Gemma calls, so it is
# still gated. Lighter than /register's 300s (this is a chat query) but still
# throttles a spammer to one answer per window.
_COOLDOWN_SECONDS = 10.0

# Channel-memory bounds. The first two bound memory in a LONG-RUNNING gateway process
# (the cog outlives every conversation in it), so neither the channel count nor the
# per-channel turn count may grow without limit (T-lw6-05).
_MEMORY_MAX_CHANNELS = 64
# 8 -> 12 for issue #220: the served model has a 131k context, and a follow-up such as
# "in that game" needs the bot's own earlier answer to still be in the transcript.
_MEMORY_MAX_TURNS = 12
# The cheap DETERMINISTIC pre-filter in front of the model gate. The gate runs on
# messages nobody sent to the bot, so at ~200ms and ~120 prompt tokens per call it is
# affordable at league volume and NOT affordable at arbitrary volume. Requiring that
# the bot has actually spoken in this channel recently bounds that cost if channel
# volume grows, and it costs nothing when it says no.
_ROOM_RECENT_SECONDS = 300.0


def _strip_bot_mention(content: str, bot_id: int) -> str:
    """Strip the bot's mention token(s) from ``content`` and return the remainder.

    Discord serializes a user mention as ``<@id>`` or ``<@!id>`` (the nickname
    form). Both are replaced with a space and the result is collapsed/stripped, so a
    bare ping yields ``""`` (which the caller treats as "not a question").
    """
    stripped = content
    for token in (f"<@{bot_id}>", f"<@!{bot_id}>"):
        stripped = stripped.replace(token, " ")
    return " ".join(stripped.split())


def _name_member_mentions(content: str, mentions: Sequence[object]) -> str:
    """Replace each OTHER member's ``<@id>`` token with ``@display name``.

    The model cannot resolve a raw id, so "what did <@123> pick?" had no member name to
    look up. A mention whose member is not in ``mentions`` stays as it is.
    """
    named = content
    for member in mentions:
        member_id = getattr(member, "id", None)
        name = getattr(member, "display_name", None) or getattr(member, "name", None)
        if member_id is None or not name:
            continue
        for token in (f"<@{member_id}>", f"<@!{member_id}>"):
            named = named.replace(token, f"@{name}")
    return named


# Discord rejects any message body over 2000 chars with a 400 (error 50035). The
# whole-slate answers (e.g. slate_predictions over a full 16-game week, each line
# further inflated by team-logo <:name:id> tokens from decorate_team_logos) can blow
# past that, which previously crashed the send. Splitting is done here, AFTER logo
# decoration, so every emitted chunk is guaranteed within the real posted length.
_DISCORD_MAX_CHARS = 2000


def _split_for_discord(text: str, *, limit: int = _DISCORD_MAX_CHARS) -> list[str]:
    """Split a (already logo-decorated) reply into Discord-sendable chunks.

    Splits on NEWLINE boundaries so no per-game line — nor a ``<:name:id>`` logo token
    inside one — is ever cut mid-way; whole lines are greedily packed into each chunk.
    A single line longer than ``limit`` (not expected for these one-line-per-game
    bodies) is hard-sliced as a last resort so a chunk can never exceed ``limit``.
    Short replies (the common case) return a single-element list unchanged.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # Defensive: a single line over the limit is emitted in limit-sized slices.
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


class _ChannelMemory:
    """A bounded per-channel transcript plus the bot's last-reply timestamp.

    Turns are ``(speaker, text, is_bot)`` oldest-first in a ``deque`` capped at
    :data:`_MEMORY_MAX_TURNS`; channels live in an ``OrderedDict`` capped at
    :data:`_MEMORY_MAX_CHANNELS` with least-recently-touched eviction. ``is_bot`` is
    stored explicitly rather than inferred by comparing the speaker to the bot's
    display name, so a member who renames themselves after the bot cannot have their
    messages relabelled as the bot's.
    """

    def __init__(self) -> None:
        self._turns: OrderedDict[int, deque[tuple[str, str, bool]]] = OrderedDict()
        self._last_bot_reply: dict[int, float] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def _touch(self, channel_id: int) -> deque[tuple[str, str, bool]]:
        """Return ``channel_id``'s turn deque, creating + evicting as needed."""
        existing = self._turns.get(channel_id)
        if existing is None:
            existing = deque(maxlen=_MEMORY_MAX_TURNS)
            self._turns[channel_id] = existing
        self._turns.move_to_end(channel_id)
        while len(self._turns) > _MEMORY_MAX_CHANNELS:
            evicted, _ = self._turns.popitem(last=False)
            self._last_bot_reply.pop(evicted, None)
            # A held lock stays: dropping it lets a new question start a second answer
            # in that channel while the first still runs.
            lock = self._locks.get(evicted)
            if lock is None or not lock.locked():
                self._locks.pop(evicted, None)
        return existing

    def record(
        self, channel_id: int, speaker: str, text: str, *, is_bot: bool = False
    ) -> tuple[str, str, bool]:
        """Append one turn to ``channel_id``'s transcript and return that turn."""
        turn = (speaker, text, is_bot)
        self._touch(channel_id).append(turn)
        return turn

    def forget(self, channel_id: int, turn: tuple[str, str, bool]) -> None:
        """Drop ``turn`` BY IDENTITY, for a question the bot will not answer.

        An unanswered question left in the transcript reads, in the next answer's
        history, as a question still waiting for a reply (the 2026-09-20 double answer).
        """
        turns = self._turns.get(channel_id)
        if turns is None:
            return
        kept = [kept_turn for kept_turn in turns if kept_turn is not turn]
        if len(kept) != len(turns):
            turns.clear()
            turns.extend(kept)

    def answer_lock(self, channel_id: int) -> asyncio.Lock:
        """The lock that makes answers in ``channel_id`` run one at a time."""
        self._touch(channel_id)
        return self._locks.setdefault(channel_id, asyncio.Lock())

    def record_bot_reply(self, channel_id: int, speaker: str, text: str, *, now: float) -> None:
        """Append the bot's OWN reply and stamp the last-reply time.

        Without this stamp the deterministic pre-filter never opens, so the room gate
        would never fire at all and read-the-room would be dead code.
        """
        self.record(channel_id, speaker, text, is_bot=True)
        self._last_bot_reply[channel_id] = now

    def transcript(self, channel_id: int) -> list[tuple[str, str]]:
        """The ``(speaker, text)`` turns for the room gate, oldest-first."""
        return [(speaker, text) for speaker, text, _ in self._turns.get(channel_id, ())]

    def history(
        self, channel_id: int, *, exclude: tuple[str, str, bool] | None = None
    ) -> list[tuple[str, str]]:
        """The ``(role, text)`` turns for ``qa.answer_question``, oldest-first.

        ``exclude`` drops one turn BY IDENTITY (the question being answered), so a
        second message with the same text stays in the history. A member's turn carries
        the speaker's name, because every member shares the one ``user`` role and a
        follow-up belongs to the member who wrote it. The bot's turn stays bare: the
        open path matches it byte-for-byte to replay its tool turns.
        """
        return [
            ("assistant", turn[1]) if turn[2] else ("user", f"{turn[0]}: {turn[1]}")
            for turn in self._turns.get(channel_id, ())
            if turn is not exclude
        ]

    def spoke_recently(self, channel_id: int, *, now: float) -> bool:
        """Whether the bot replied in ``channel_id`` within the recency window."""
        last = self._last_bot_reply.get(channel_id)
        return last is not None and (now - last) <= _ROOM_RECENT_SECONDS


def _is_chat_channel(channel: object) -> bool:
    """Whether ``channel`` is the league chat channel (DISCORD_CHAT_CHANNEL, id or name).

    Every message there goes into the transcript (issue #252); elsewhere only messages
    addressed to the bot do, so a private channel's chatter is never stored.
    """
    setting = (settings.discord_chat_channel or "").strip()
    if not setting:
        return False
    if setting.isdigit():
        return getattr(channel, "id", None) == int(setting)
    name = getattr(channel, "name", None)
    return isinstance(name, str) and name.casefold() == setting.lstrip("#").casefold()


def _message_meta(message: discord.Message, *, mentioned: bool, reply_to_bot: bool) -> dict:
    return {
        "channel": getattr(message.channel, "name", None),
        "channel_id": str(getattr(message.channel, "id", "")),
        "message_id": str(getattr(message, "id", "")),
        "author": getattr(message.author, "display_name", None) or "someone",
        "mentioned": mentioned,
        "reply_to_bot": reply_to_bot,
    }


def _post_text(message: discord.Message) -> str:
    """A bot post's text: its content, else its embeds' titles and descriptions."""
    parts = [message.content] if message.content else []
    for embed in getattr(message, "embeds", None) or []:
        for part in (getattr(embed, "title", None), getattr(embed, "description", None)):
            if part:
                parts.append(part)
    return " | ".join(parts)


def _is_reply_to_bot(message: discord.Message, bot_user_id: int) -> bool:
    """Whether ``message`` is a Discord reply to one of the BOT's own messages.

    Resolved defensively with ``getattr`` at every hop: the reference may be absent,
    unresolved (not in the cache), or point at a deleted message, and any of those must
    read as "not a reply to the bot" rather than raise. T-lw6-06 ACCEPTS that a member
    can reply to an old bot message to skip the gate — the blast radius is one answer,
    still behind the per-user cooldown, and treating a reply as an addressing signal is
    exactly the point.
    """
    reference = getattr(message, "reference", None)
    resolved = getattr(reference, "resolved", None)
    author_id = getattr(getattr(resolved, "author", None), "id", None)
    return author_id is not None and author_id == bot_user_id


class MentionQaCog(commands.Cog):
    """Answers a genuine user->bot @mention with a public in-voice line."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # Per-user cooldown mapping — keyed by message.author.id (BucketType.user).
        self._cooldown = commands.CooldownMapping.from_cooldown(
            1, _COOLDOWN_SECONDS, commands.BucketType.user
        )
        # Bounded per-channel transcript — feeds both the room gate and the open
        # path's conversation history.
        self._memory = _ChannelMemory()
        # The bot's own Q&A reply ids: the answer entry already holds that text, so the
        # echo through on_message is not logged a second time.
        self._reply_ids: deque[int] = deque(maxlen=200)

    def _is_rate_limited(self, message: discord.Message) -> bool:
        """Whether ``message``'s author is over the per-user cooldown right now.

        Passes an explicit ``current`` timestamp so the mapping never has to read
        ``message.created_at`` (keeps the handler testable with a lightweight fake
        message). Returns True when the bucket is exhausted (the mention is skipped).
        """
        retry_after = self._cooldown.update_rate_limit(message, time.time())
        return retry_after is not None

    def _bot_name(self) -> str:
        """The bot's display name for the transcript (falls back to a fixed label)."""
        return getattr(self.bot.user, "display_name", None) or "the bot"

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Answer a message addressed to the bot; ignore everything else. Never raises.

        The ORDER below is load-bearing:
          1-4. the four KEPT gates (bot author, guild-only, @everyone, empty text) —
               cheap, deterministic, and unchanged from the mention-only version.
          5.   record this message, so the gate judges a transcript ENDING in it.
          6.   addressing: an explicit @mention or a reply to the bot passes with NO
               gate call; a message that @mentions other members and not the bot is
               dropped with NO gate call; anything else needs the recent-bot-reply
               pre-filter and then the model gate.
          7.   the per-user cooldown — AFTER the gate on purpose. ``update_rate_limit``
               MUTATES the bucket, so running it first would burn the asker's bucket on
               ordinary channel chatter and drop their real question seconds later.
          8.   the per-channel lock, then the history snapshot EXCLUDING this message,
               then typing indicator + answer + decorate + split + send.
          9.   record the bot's own reply (what opens the pre-filter next time) —
               still inside the lock, so the next waiting answer reads it.
        """
        trace = None
        try:
            # (2) Guild messages only — DMs are out of scope, and never logged.
            if message.guild is None:
                return
            in_chat = _is_chat_channel(message.channel)
            bot_user = self.bot.user
            mentioned = bot_user is not None and bot_user in message.mentions
            reply_to_bot = bot_user is not None and _is_reply_to_bot(message, bot_user.id)
            meta = _message_meta(message, mentioned=mentioned, reply_to_bot=reply_to_bot)

            def skip(decision: str, text: str | None = None) -> None:
                if in_chat or mentioned or reply_to_bot:
                    bot_telemetry.log_message(
                        kind="member", **meta, question=text or message.content, decision=decision
                    )

            # (1) Ignore messages from bots / the bot itself (its event posts are logged).
            if message.author.bot:
                if in_chat and getattr(message, "id", None) not in self._reply_ids:
                    own = bot_user is not None and message.author.id == bot_user.id
                    bot_telemetry.log_message(
                        kind="bot_post" if own else "other_bot", **meta, content=_post_text(message)
                    )
                return
            # (3) @everyone / @here is never addressed to the bot in particular.
            if message.mention_everyone:
                skip("ignored:everyone")
                return
            if bot_user is None:
                return

            # (4) A bare ping (or an empty body) is not a question.
            question = _strip_bot_mention(message.content, bot_user.id)
            if not question:
                skip("ignored:empty")
                return
            question = _name_member_mentions(question, message.mentions)

            # (5) Record first so the gate's transcript ends in this message.
            channel_id = message.channel.id
            speaker = getattr(message.author, "display_name", None) or "someone"
            turn = self._memory.record(channel_id, speaker, question)

            # (6) Is this addressed to the bot?
            addressed_by = "mention" if mentioned else "reply" if reply_to_bot else "room_gate"
            if not (mentioned or reply_to_bot):
                if message.mentions:
                    # addressed to another member — the gate is never consulted
                    skip("not_addressed:mentions_another_member", question)
                    return
                if not self._memory.spoke_recently(channel_id, now=time.time()):
                    # cold channel — the gate is never consulted
                    skip("not_addressed:bot_not_recently_active", question)
                    return
                addressed = await qa_room.is_addressed(
                    self._memory.transcript(channel_id), bot_name=self._bot_name()
                )
                # Logged both ways: a silent drop was undiagnosable from a transcript.
                logger.info("mention_qa_gate_decision", channel_id=channel_id, addressed=addressed)
                if not addressed:
                    skip("not_addressed:room_gate_said_no", question)
                    return

            # (7) Per-user cooldown — only ever spent on a message we would answer.
            if self._is_rate_limited(message):
                logger.info("mention_qa_cooldown_dropped", channel_id=channel_id)
                self._memory.forget(channel_id, turn)
                skip("cooldown", question)
                return

            # (8) Show the "Pick'em Bot is typing…" indicator for the whole answer + send.
            # Clean here (unlike a slash command) because this is an on_message listener
            # with NO 3s interaction ACK deadline; discord.py auto-refreshes the indicator
            # every ~10s until the block exits — covering the Gemma calls + any live
            # fetches a prediction makes (#117 / the prediction-intent design).
            # 2026-09-20: two questions in one minute got the FIRST one answered twice —
            # the second answer's history held the first question with no reply to it.
            async with self._memory.answer_lock(channel_id), message.channel.typing():
                history = self._memory.history(channel_id, exclude=turn)
                answered = False
                trace = bot_telemetry.start(
                    question,
                    conversation_key=str(channel_id),
                    asker_name=speaker,
                    message={**meta, "addressed_by": addressed_by},
                )
                line = await qa.answer_question(
                    question,
                    discord_id=message.author.id,
                    history=history,
                    conversation_key=str(channel_id),
                    asker_name=speaker,
                )
                decorated = decorate_team_logos(line)
                # suppress_embeds: a news reply carries source links (masked links) —
                # without this Discord unfurls EVERY link into a wall of rich preview
                # cards below the clean headline list. The Q&A replies are plain text
                # lines, so suppressing link embeds is always the right call here.
                # Split so a long whole-slate answer (>2000 chars after logo tokens)
                # sends as multiple messages instead of 400-ing the gateway send.
                try:
                    for chunk in _split_for_discord(decorated):
                        sent = await message.channel.send(
                            chunk,
                            allowed_mentions=discord.AllowedMentions.none(),
                            suppress_embeds=True,
                        )
                        sent_id = getattr(sent, "id", None)
                        if sent_id is not None:
                            self._reply_ids.append(sent_id)
                        answered = True
                finally:
                    if not answered:
                        self._memory.forget(channel_id, turn)
                    bot_telemetry.finish(
                        trace, line, decision="answered" if answered else "send_failed"
                    )
                    trace = None
                # (9) Stamp the bot's own reply — this is what opens the pre-filter so a
                # bare follow-up in this channel can reach the gate at all.
                self._memory.record_bot_reply(channel_id, self._bot_name(), line, now=time.time())
        except Exception:
            # One bad message must never crash the gateway loop (mirrors the notifier
            # per-message guard). answer_question and is_addressed are best-effort too,
            # but guard the send / decorate path here as well.
            logger.warning("mention_qa_on_message_failed", exc_info=True)
            if trace is not None:
                bot_telemetry.finish(trace, None, decision="error")


async def setup(bot: commands.Bot) -> None:
    """Required by load_extension — registers MentionQaCog with the bot."""
    await bot.add_cog(MentionQaCog(bot))
