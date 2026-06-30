"""Best-effort Tier-2 weekly-recap orchestrator (260627-tfb).

This upgrades the ``week.recap`` chat event from a one-line render into a 2-3
sentence LLM-narrated "column" built from the FULL week's per-player scores + the
current season standings. It keeps the v1/Tier-1 safety contract intact: the BOT
owns a deterministic, display-only FACT block and the LLM ONLY narrates it; on ANY
db OR LLM failure the existing deterministic :func:`app.bot.notifier.render_chat`
one-liner is returned, so the event always produces exactly one line.

Best-effort posture mirrors :mod:`app.bot.chat_personality`: structlog only, NO
``discord`` import, and the whole body of :func:`build_week_recap` is wrapped so any
error falls back to the deterministic line — it NEVER raises into the notifier loop
(T-tfb-03). The fact block carries display-only data only (display_name, integer
scores, ranks) — never user_id/secrets (T-tfb-01).
"""

from __future__ import annotations

import structlog

from app.bot import llm_client
from app.bot.personality import DEFAULT_PERSONALITY_ID, PERSONALITIES, compose_prompt

logger = structlog.get_logger(__name__)

# The Tier-2 recap ROLE line: the event-specific context (writing the weekly recap
# column). It is NOT the voice — the leading voice sentence is supplied by the
# active personality at compose time (260627-xbb).
RECAP_ROLE = (
    "You are writing a short weekly recap column. Given this week's final per-player "
    "scores and the current season standings, reply with 2-3 short sentences that "
    "recap THIS week's results AND the current season picture."
)

# The INVARIANT recap guard. It MUST instruct the model to narrate ONLY the supplied
# numbers and NOT invent movement/trends — no prior-week standings are supplied, so
# the model has no basis to claim anyone "jumped" or "climbed" (T-tfb-02). This
# stays byte-identical across every personality (the no-prior-standings clause is a
# correctness guarantee, not voice flavor — 260627-xbb).
RECAP_GUARD = (
    "Use ONLY the numbers you are given. Do NOT invent movement, trends, jumps, "
    "climbs, streaks, or any stat that is not in the data — you were given no "
    "prior-week standings, so never claim anyone rose or fell. Use at most one or "
    "two emoji."
)

# Back-compat: the composed default (sarcastic) recap prompt. ``build_week_recap``
# composes with the ACTIVE voice resolved via the db_bridge seam; this constant is
# the unset/default composition (active voice == sarcastic) so callers/tests that
# reference RECAP_PROMPT still see the default prompt.
RECAP_PROMPT = compose_prompt(PERSONALITIES[DEFAULT_PERSONALITY_ID], RECAP_ROLE, RECAP_GUARD)


async def _recap_context(week: int) -> dict:
    """Fetch the display-only recap context for ``week``.

    Thin async indirection over :func:`app.bot.db_bridge.get_recap_context_async`
    so tests can monkeypatch it cleanly without a real db. Lazy import mirrors
    :mod:`app.bot.chat_personality`'s import posture and keeps this module's
    top-level import graph light.
    """
    from app.bot.db_bridge import get_recap_context_async

    return await get_recap_context_async(week)


async def _active_voice() -> str:
    """Resolve the active personality's voice preamble via the db_bridge seam.

    The DB read happens INSIDE db_bridge's async/thread seam; falls back to the
    sarcastic voice on any miss/unset/raise. Lazy import mirrors ``_recap_context``
    so tests can monkeypatch this seam without a real db.
    """
    from app.bot.db_bridge import resolve_active_voice_async

    return await resolve_active_voice_async()


def _recap_fact(context: dict) -> str | None:
    """Build the deterministic recap fact string from the context, or ``None``.

    Returns ``None`` when there are no weekly scores (nothing to narrate). Otherwise
    builds a deterministic, multi-line fact from the context numbers ONLY: a
    "This week (Week N):" section listing each ``{display_name}: {weekly_score}``
    high->low, and a "Season standings:" section listing each ``{display_name} —
    {season_total} (rank {rank}, {gap_to_leader} back)`` in order. This is the LLM
    input; it carries display-only fields only (never user_id).
    """
    weekly_scores = context.get("weekly_scores") or []
    if not weekly_scores:
        return None

    week = context.get("week")
    lines = [f"This week (Week {week}):"]
    for row in weekly_scores:
        lines.append(f"- {row['display_name']}: {row['weekly_score']}")

    standings = context.get("season_standings") or []
    if standings:
        lines.append("Season standings:")
        for row in standings:
            lines.append(
                f"- {row['display_name']} — {row['season_total']} "
                f"(rank {row['rank']}, {row['gap_to_leader']} back)"
            )

    return "\n".join(lines)


async def build_week_recap(event: dict) -> str:
    """Return the LLM-narrated recap column for a ``week.recap`` ``event``.

    Fetches the display-only context for ``event["week"]``, builds the deterministic
    fact, and asks :func:`app.bot.llm_client.phrase` to narrate it under
    :data:`RECAP_PROMPT`. Returns the phrased column when non-None; otherwise (a
    None phrase, an empty context, OR any exception) returns the existing
    deterministic :func:`app.bot.notifier.render_chat` one-liner.

    Best-effort (T-tfb-03): the entire body is guarded so ANY db OR LLM error falls
    back to the deterministic line. Unlike ``chat_personality.embellish_chat`` this
    is only ever called for ``week.recap``, so it ALWAYS returns a string (never
    None) — the notifier swaps its one-liner for this call. NEVER raises.
    """
    # Lazy import to avoid the notifier import cycle (mirrors embellish_chat).
    from app.bot.notifier import render_chat

    try:
        week = event.get("week")
        context = await _recap_context(week)
        fact = _recap_fact(context)
        if fact is None:
            return render_chat(event)  # nothing to narrate -> deterministic line
        # Compose the active voice (resolved in the db_bridge seam) + the recap
        # ROLE + the byte-identical recap guard. Unset/unreadable -> sarcastic, so
        # this equals RECAP_PROMPT by default.
        active_voice = await _active_voice()
        system_prompt = compose_prompt(active_voice, RECAP_ROLE, RECAP_GUARD)
        phrased = await llm_client.phrase(fact, system_prompt=system_prompt)
        if phrased is not None:
            return phrased
        return render_chat(event)
    except Exception:
        # A db/LLM/render hiccup must never escape into the notifier loop — fall
        # back to the deterministic one-liner.
        logger.warning("recap_failed", event_type=event.get("type"), exc_info=True)
        return render_chat(event)
