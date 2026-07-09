"""MentionQaCog — the inbound @mention Q&A listener (Path A v1, 260709-k5w).

A thin Discord surface (mirrors :mod:`app.bot.commands.register`): this is the ONLY
Q&A module that imports ``discord``. All the brains live in the Discord-free
:mod:`app.bot.qa`; this cog only decides whether a message is a genuine user->bot
mention, enforces a per-user cooldown, hands the stripped question to
:func:`app.bot.qa.answer_question`, and posts the reply PUBLICLY with
``discord.AllowedMentions.none()`` so the LLM-authored text can never ping anyone.

Locked posture:
  - Fires ONLY on a real user->bot mention: bots/self ignored, @everyone/@here and
    role pings excluded, a bare ping (no text after stripping the mention) ignored,
    DMs out of scope for v1 (guild messages only).
  - Per-user ``CooldownMapping`` (BucketType.user) gates EVERY mention BEFORE any
    LLM work — each answer costs two local-Gemma calls (classify + phrase).
  - The whole handler body is guarded (structlog + swallow): one bad message must
    never crash the gateway loop (``qa.answer_question`` is itself best-effort, but
    the send / decorate path is guarded here too).
"""

from __future__ import annotations

import time

import discord
import structlog
from discord.ext import commands

from app.bot import qa
from app.bot.team_emoji import decorate_team_logos

logger = structlog.get_logger(__name__)

# Per-user cooldown window. Each mention triggers two local-Gemma calls, so gate
# before any LLM work. Lighter than /register's 300s (this is a chat query) but
# still throttles a spammer to one answer per window.
_COOLDOWN_SECONDS = 20.0


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


class MentionQaCog(commands.Cog):
    """Answers a genuine user->bot @mention with a public in-voice line."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # Per-user cooldown mapping — keyed by message.author.id (BucketType.user).
        self._cooldown = commands.CooldownMapping.from_cooldown(
            1, _COOLDOWN_SECONDS, commands.BucketType.user
        )

    def _is_rate_limited(self, message: discord.Message) -> bool:
        """Whether ``message``'s author is over the per-user cooldown right now.

        Passes an explicit ``current`` timestamp so the mapping never has to read
        ``message.created_at`` (keeps the handler testable with a lightweight fake
        message). Returns True when the bucket is exhausted (the mention is skipped).
        """
        retry_after = self._cooldown.update_rate_limit(message, time.time())
        return retry_after is not None

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Answer a real user->bot mention; ignore everything else. Never raises."""
        try:
            # Ignore messages from bots / the bot itself.
            if message.author.bot:
                return
            # Guild messages only — DMs are out of scope for v1.
            if message.guild is None:
                return
            # @everyone / @here is never a real bot mention.
            if message.mention_everyone:
                return
            # Require the bot to be INDIVIDUALLY mentioned (a role ping puts the role
            # in message.role_mentions, NOT the bot in message.mentions).
            if self.bot.user is None or self.bot.user not in message.mentions:
                return

            question = _strip_bot_mention(message.content, self.bot.user.id)
            if not question:
                return  # a bare ping is not a question

            # Per-user cooldown BEFORE any LLM work (each answer costs two Gemma calls).
            if self._is_rate_limited(message):
                return

            line = await qa.answer_question(question, discord_id=message.author.id)
            decorated = decorate_team_logos(line)
            await message.channel.send(
                decorated, allowed_mentions=discord.AllowedMentions.none()
            )
        except Exception:
            # One bad message must never crash the gateway loop (mirrors the notifier
            # per-message guard). answer_question is best-effort too, but guard the
            # send / decorate path here as well.
            logger.warning("mention_qa_on_message_failed", exc_info=True)


async def setup(bot: commands.Bot) -> None:
    """Required by load_extension — registers MentionQaCog with the bot."""
    await bot.add_cog(MentionQaCog(bot))
