"""Discord-free READ-THE-ROOM gate (Path C, 260820-lw6).

The bot already receives every message in every channel it can see — ``guild_messages``
and the privileged ``message_content`` intent are both already on in production, or the
current @mention Q&A could not work at all. What message it ANSWERS is pure policy, and
until now that policy was "the bot is individually in ``message.mentions``". This module
replaces that one check with a model judgement: given the recent channel transcript, is
the LAST message addressed to the bot?

Posture:

* **No ``discord`` import** — the cog :mod:`app.bot.commands.mention_qa` stays the only
  discord-importing module for this feature.
* **Best-effort, and it FAILS CLOSED to ``False``.** :func:`is_addressed` never raises;
  every failure mode (a dead server, unparseable output, a non-dict payload, a
  non-boolean value) answers ``False``. The prompt leans false too. This asymmetry is
  deliberate: a false positive makes the bot INTERRUPT a human conversation, which is
  more socially costly than missing a follow-up the member can just repeat.
* **Untrusted text crosses the model boundary ONLY via
  :func:`app.bot.chat_personality._fence_untrusted`** — the message bodies AND the
  speaker names, because a member picks their own display name.
* **No new ``llm_client`` function.** The gate reuses the existing deterministic JSON
  extraction seam :func:`app.bot.llm_client.classify` (greedy decode, no closer-variety
  chat directive, JSON-sized token cap).

The 2026-08-20 probe measured the gate at ~200 ms and ~120 prompt tokens per call over
five hand-written cases, all correct. Five hand-written cases are a FEASIBILITY signal,
not a validation set — the gate needs a real check against actual channel traffic before
it runs unattended.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import structlog

from app.bot import chat_personality, llm_client

logger = structlog.get_logger(__name__)

# How much channel context the gate sees. The served model reports ``n_ctx: 114688``, so
# this is a LATENCY and COST bound, not a memory bound — the gate runs on messages nobody
# sent to the bot, so every token it reads is paid for on ordinary chatter.
_MAX_TURNS = 8
_MAX_TURN_CHARS = 200

# JSON-ONLY, ONE decision. Deliberately enumerates the FALSE cases concretely (the local
# Gemma handles concrete enumerations far better than an abstract rule) and ends on an
# explicit lean-false instruction.
ROOM_GATE_SYSTEM_PROMPT = (
    "You read a short transcript of recent messages from a Discord channel and decide "
    "exactly ONE thing: whether the LAST message in that transcript is addressed to the "
    "bot named at the top of the transcript. "
    "Reply with ONLY a compact JSON object and NOTHING else — no prose, no code fence, "
    'no explanation. The object has exactly one key: "addressed", whose value is the '
    "boolean true or the boolean false. "
    "Answer true when the last message is a question or a request aimed at the bot, "
    "including a follow-up that continues something the bot just said. "
    "Answer false when two people are talking to each other, when the last message is a "
    "reaction to or a comment ABOUT the bot rather than a question TO it, when it is "
    "unrelated channel chatter, and in every case where you are not sure. "
    "When in doubt, answer false."
)


def _render_transcript(turns: Sequence[tuple[str, str]], *, bot_name: str) -> str:
    """Render ``turns`` as a fenced, capped, oldest-first Discord transcript. Pure.

    A header line names the bot (so the prompt's "the bot named at the top of the
    transcript" resolves), then one ``speaker: body`` line per turn. BOTH the speaker
    and the body — and the bot name itself — go through
    :func:`app.bot.chat_personality._fence_untrusted`, which strips control characters
    (so no smuggled newline can forge its own transcript line) and the fence markers, and
    length-caps each piece at :data:`_MAX_TURN_CHARS`. Only the last :data:`_MAX_TURNS`
    turns are rendered.
    """
    safe_bot = chat_personality._fence_untrusted(bot_name, limit=_MAX_TURN_CHARS) or "the bot"
    lines = [
        f'The bot in this channel is named "{safe_bot}". '
        "Here are the most recent messages in the channel, oldest first:"
    ]
    for speaker, body in list(turns)[-_MAX_TURNS:]:
        safe_speaker = (
            chat_personality._fence_untrusted(speaker, limit=_MAX_TURN_CHARS) or "someone"
        )
        safe_body = chat_personality._fence_untrusted(body, limit=_MAX_TURN_CHARS)
        lines.append(f"{safe_speaker}: {safe_body}")
    return "\n".join(lines)


async def is_addressed(turns: Sequence[tuple[str, str]], *, bot_name: str) -> bool:
    """Whether the LAST turn in ``turns`` is addressed to ``bot_name``. Never raises.

    Renders the fenced transcript, runs it through the deterministic JSON extraction
    seam, and returns whether the payload's ``addressed`` key is the boolean ``True`` BY
    IDENTITY. Everything else — a missing key, a null, the string ``"true"``, a ``1``, a
    non-dict payload, unparseable output, a ``None`` from the client, or a surprise raise
    — returns ``False``. FAIL CLOSED: silence is always the safe answer here.
    """
    try:
        transcript = _render_transcript(turns, bot_name=bot_name)
        raw = await llm_client.classify(transcript, system_prompt=ROOM_GATE_SYSTEM_PROMPT)
        if raw is None:
            return False
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return False
        return parsed.get("addressed") is True
    except Exception:
        # Any failure at all leans silent — the bot never interrupts on a hiccup.
        logger.warning("qa_room_gate_failed", exc_info=True)
        return False
