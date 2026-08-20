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
  - The whole handler body is guarded (structlog + swallow): one bad message must
    never crash the gateway loop (``qa.answer_question`` and ``qa_room.is_addressed``
    are themselves best-effort, but the send / decorate path is guarded here too).
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque

import discord
import structlog
from discord.ext import commands

from app.bot import qa, qa_room
from app.bot.team_emoji import decorate_team_logos

logger = structlog.get_logger(__name__)

# Per-user cooldown window. Each answer triggers several local-Gemma calls, so it is
# still gated. Lighter than /register's 300s (this is a chat query) but still
# throttles a spammer to one answer per window.
_COOLDOWN_SECONDS = 10.0

# Channel-memory bounds. The first two bound memory in a LONG-RUNNING gateway process
# (the cog outlives every conversation in it), so neither the channel count nor the
# per-channel turn count may grow without limit (T-lw6-05).
_MEMORY_MAX_CHANNELS = 64
_MEMORY_MAX_TURNS = 8
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
        return existing

    def record(self, channel_id: int, speaker: str, text: str, *, is_bot: bool = False) -> None:
        """Append one turn to ``channel_id``'s transcript."""
        self._touch(channel_id).append((speaker, text, is_bot))

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

    def history(self, channel_id: int) -> list[tuple[str, str]]:
        """The ``(role, text)`` turns for ``qa.answer_question``, oldest-first."""
        return [
            ("assistant" if is_bot else "user", text)
            for _, text, is_bot in self._turns.get(channel_id, ())
        ]

    def spoke_recently(self, channel_id: int, *, now: float) -> bool:
        """Whether the bot replied in ``channel_id`` within the recency window."""
        last = self._last_bot_reply.get(channel_id)
        return last is not None and (now - last) <= _ROOM_RECENT_SECONDS


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
          5.   snapshot the history for the answer BEFORE recording, then record this
               message (so the gate judges a transcript ENDING in this message while
               ``answer_question`` gets the history EXCLUDING it).
          6.   addressing: an explicit @mention or a reply to the bot passes with NO
               gate call; anything else needs the recent-bot-reply pre-filter and then
               the model gate.
          7.   the per-user cooldown — AFTER the gate on purpose. ``update_rate_limit``
               MUTATES the bucket, so running it first would burn the asker's bucket on
               ordinary channel chatter and drop their real question seconds later.
          8.   typing indicator + answer + decorate + split + send.
          9.   record the bot's own reply (what opens the pre-filter next time).
        """
        try:
            # (1) Ignore messages from bots / the bot itself.
            if message.author.bot:
                return
            # (2) Guild messages only — DMs are out of scope.
            if message.guild is None:
                return
            # (3) @everyone / @here is never addressed to the bot in particular.
            if message.mention_everyone:
                return
            if self.bot.user is None:
                return

            # (4) A bare ping (or an empty body) is not a question.
            question = _strip_bot_mention(message.content, self.bot.user.id)
            if not question:
                return

            # (5) Snapshot BEFORE recording so the answer's history excludes this turn.
            channel_id = message.channel.id
            history = self._memory.history(channel_id)
            speaker = getattr(message.author, "display_name", None) or "someone"
            self._memory.record(channel_id, speaker, question)

            # (6) Is this addressed to the bot?
            mentioned = self.bot.user in message.mentions
            if not (mentioned or _is_reply_to_bot(message, self.bot.user.id)):
                if not self._memory.spoke_recently(channel_id, now=time.time()):
                    return  # cold channel — the gate is never consulted
                addressed = await qa_room.is_addressed(
                    self._memory.transcript(channel_id), bot_name=self._bot_name()
                )
                if not addressed:
                    return

            # (7) Per-user cooldown — only ever spent on a message we would answer.
            if self._is_rate_limited(message):
                return

            # (8) Show the "Pick'em Bot is typing…" indicator for the whole answer + send.
            # Clean here (unlike a slash command) because this is an on_message listener
            # with NO 3s interaction ACK deadline; discord.py auto-refreshes the indicator
            # every ~10s until the block exits — covering the Gemma calls + any live
            # fetches a prediction makes (#117 / the prediction-intent design).
            async with message.channel.typing():
                line = await qa.answer_question(
                    question, discord_id=message.author.id, history=history
                )
                decorated = decorate_team_logos(line)
                # suppress_embeds: a news reply carries source links (masked links) —
                # without this Discord unfurls EVERY link into a wall of rich preview
                # cards below the clean headline list. The Q&A replies are plain text
                # lines, so suppressing link embeds is always the right call here.
                # Split so a long whole-slate answer (>2000 chars after logo tokens)
                # sends as multiple messages instead of 400-ing the gateway send.
                for chunk in _split_for_discord(decorated):
                    await message.channel.send(
                        chunk,
                        allowed_mentions=discord.AllowedMentions.none(),
                        suppress_embeds=True,
                    )
            # (9) Stamp the bot's own reply — this is what opens the pre-filter so a
            # bare follow-up in this channel can reach the gate at all.
            self._memory.record_bot_reply(channel_id, self._bot_name(), line, now=time.time())
        except Exception:
            # One bad message must never crash the gateway loop (mirrors the notifier
            # per-message guard). answer_question and is_addressed are best-effort too,
            # but guard the send / decorate path here as well.
            logger.warning("mention_qa_on_message_failed", exc_info=True)


async def setup(bot: commands.Bot) -> None:
    """Required by load_extension — registers MentionQaCog with the bot."""
    await bot.add_cog(MentionQaCog(bot))
