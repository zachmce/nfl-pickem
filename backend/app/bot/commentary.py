"""Commentary orchestrator for the pickem-chat personality layer (260627-nef).

This is the glue between the deterministic scanner and the LLM: on a window
close it reads each player's locked slate + recent history (via the Discord-free
:mod:`app.bot.db_bridge` wrappers), runs the pure
:func:`app.services.pick_patterns.scan_streak` to find a FACT, builds a
deterministic fact sentence, and asks :func:`app.bot.llm_client.phrase_pattern`
to phrase it — falling back to the deterministic sentence on any LLM failure.

Best-effort end to end (T-nef-03): each per-player step is wrapped so one bad
player is logged and skipped, and NO exception escapes ``build_lock_commentary``
— the notifier loop must survive an LLM/db hiccup. It does NOT post to Discord
(the notifier owns the channel); it returns plain strings. Only display-only
data (display_name, team_abbr, side, streak) ever crosses to the LLM or the
returned lines — never a user_id, never an open-window pick (the caller fires
this only on ``window.closed``).
"""

from __future__ import annotations

import structlog

from app.bot import llm_client
from app.bot.db_bridge import (
    get_pick_history_async,
    get_week_picks_async,
    resolve_active_voice_async,
)
from app.services.pick_patterns import scan_streak

logger = structlog.get_logger(__name__)


def _fact_sentence(display_name: str, fact: dict) -> str:
    """Deterministic, hallucination-proof fact line — BOTH the LLM input and the
    fallback. Carries only display-only fields (name, team_abbr, side, streak)."""
    side = fact["side"].lower()
    return (
        f"{display_name} has taken {fact['team_abbr']} {side} "
        f"{fact['streak_weeks']} weeks running."
    )


async def build_lock_commentary(week: int) -> list[str]:
    """Return the ordered personality lines for a just-closed ``week`` (may be empty).

    For each player with a qualifying streak, phrase the deterministic fact via
    the LLM (falling back to the fact sentence when the LLM returns ``None``).
    Players with no fact contribute nothing. Best-effort: any per-player error is
    logged and skipped; the whole function never raises.
    """
    lines: list[str] = []
    try:
        slates = await get_week_picks_async(week)
        histories = await get_pick_history_async(week)
        # Resolve the active voice ONCE inside the db_bridge seam and thread it into
        # the pure phrase layer (which never reads the DB itself); sarcastic on any
        # miss/raise.
        active_voice = await resolve_active_voice_async()
    except Exception:
        # db hiccup — produce no commentary rather than break the caller.
        logger.warning("commentary_read_failed", week=week, exc_info=True)
        return lines

    for display_name, slate_keys in slates.items():
        try:
            history_keys = histories.get(display_name, [])
            fact = scan_streak(week, slate_keys, history_keys)
            if fact is None:
                continue
            sentence = _fact_sentence(display_name, fact)
            phrased = await llm_client.phrase_pattern(sentence, voice=active_voice)
            lines.append(phrased if phrased is not None else sentence)
        except Exception:
            # One bad player must never abort the batch.
            logger.warning(
                "commentary_player_failed", display_name=display_name, exc_info=True
            )
            continue

    return lines
