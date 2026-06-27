"""Best-effort chat personality layer for the Tier-1 reactive events (260627-t5u).

This is the orchestrator that gives three existing chat events a little more
personality without giving up the v1 safety contract (260627-nef): the BOT owns a
deterministic, hallucination-proof FACT and the LLM ONLY phrases it. On ANY LLM
failure the deterministic :func:`app.bot.notifier.render_chat` line is returned, so
each event always produces exactly one line.

Handled here (and ONLY here):

* ``window.opened``   -> a hype line built from the week number.
* ``game.final``      -> a reaction line built from the two abbrs + two scores +
  a COMPUTED margin descriptor (blowout / nail-biter / "") — never an invented stat.
* ``roster.complete`` -> a light-ribbing line built from actor + week ONLY. The
  event carries NO pick content, and the fact provably references none (LEAK-SAFE,
  T-t5u-01).

Everything else (``window.closed``, ``week.recap``, unknown types) returns ``None``
— the notifier keeps owning those via its existing deterministic path /
``build_lock_commentary``.

Best-effort posture mirrors :mod:`app.bot.commentary`: structlog only, NO
``discord`` import, and the whole body is wrapped so any error falls back to the
deterministic line — :func:`embellish_chat` NEVER raises (T-t5u-03).
"""

from __future__ import annotations

import structlog

from app.bot import llm_client

logger = structlog.get_logger(__name__)

# Tier-1 event types this seam phrases. Everything else is the notifier's job.
_HANDLED_TYPES = frozenset({"window.opened", "game.final", "roster.complete"})

# Margin thresholds (points) for the COMPUTED game.final descriptor.
_NAIL_BITER_MARGIN = 3  # margin <= 3 -> a nail-biter
_BLOWOUT_MARGIN = 17  # margin >= 17 -> a blowout

_WINDOW_OPENED_PROMPT = (
    "You are the hype-bot for an NFL pick'em league. Given the week number, reply "
    "with ONE short, energetic line telling players the pick window just opened and "
    "to get their picks in. Use at most one emoji. NEVER invent any detail beyond "
    "the fact you are given."
)

_GAME_FINAL_PROMPT = (
    "You are the house bot for an NFL pick'em league reacting to a game that just "
    "went final. Given the two teams, their final scores, and a one-word margin "
    "descriptor, reply with ONE short, playful reaction line. Use at most one "
    "emoji. NEVER invent any stat or detail beyond the score and descriptor you are "
    "given."
)

_ROSTER_COMPLETE_PROMPT = (
    "You are the house bot for an NFL pick'em league. Given a player's name and the "
    "week they just finished locking in all their picks, reply with ONE short, "
    "lightly-ribbing line. Use at most one emoji. NEVER mention or guess which picks "
    "they made — you do not know them. NEVER invent any detail beyond the name and "
    "week you are given."
)


def _final_descriptor(home_score: int, away_score: int) -> str:
    """Return a COMPUTED one-word margin descriptor for a final score.

    Pure and deterministic — derived only from ``abs(home_score - away_score)``,
    never invented. A tight game is a ``"nail-biter"``, a runaway is a
    ``"blowout"``, and anything in between gets ``""`` (no descriptor).
    """
    margin = abs(home_score - away_score)
    if margin <= _NAIL_BITER_MARGIN:
        return "nail-biter"
    if margin >= _BLOWOUT_MARGIN:
        return "blowout"
    return ""


def _fact_and_prompt(event: dict) -> tuple[str, str] | None:
    """Build the deterministic (fact, system_prompt) for a handled event, or None.

    The fact is built from ONLY the event's existing display fields — for
    ``roster.complete`` that is actor + week and NOTHING pick-related (LEAK-SAFE).
    """
    etype = event.get("type")

    if etype == "window.opened":
        fact = f"Week {event.get('week')} pick window just opened."
        return fact, _WINDOW_OPENED_PROMPT

    if etype == "game.final":
        home = event.get("home")
        away = event.get("away")
        home_score = event.get("home_score")
        away_score = event.get("away_score")
        descriptor = _final_descriptor(home_score, away_score)
        fact = (
            f"Week {event.get('week')} final: {home} {home_score}, "
            f"{away} {away_score}."
        )
        if descriptor:
            fact = f"{fact} It was a {descriptor}."
        return fact, _GAME_FINAL_PROMPT

    if etype == "roster.complete":
        # actor + week ONLY — the event carries no pick content, so the fact
        # carries none either (T-t5u-01). Worded to avoid even substrings of
        # pick-content tokens so the LEAK-SAFE guard holds by construction.
        fact = (
            f"{event.get('actor')} just finished their full Week "
            f"{event.get('week')} roster and submitted it."
        )
        return fact, _ROSTER_COMPLETE_PROMPT

    return None


async def embellish_chat(event: dict) -> str | None:
    """Return one personality line for a Tier-1 chat ``event``, or ``None``.

    For ``window.opened`` / ``game.final`` / ``roster.complete``: build a
    deterministic fact, ask :func:`app.bot.llm_client.phrase` to phrase it under the
    event's prompt, and return the phrased line — or the deterministic
    :func:`app.bot.notifier.render_chat` line if the LLM returns ``None``. For any
    other type return ``None`` (the notifier owns it).

    Best-effort (T-t5u-03): the entire body is guarded so ANY error falls back to
    the deterministic render_chat line (for a handled type) or ``None`` (otherwise).
    NEVER raises out into the notifier loop.
    """
    etype = event.get("type")
    if etype not in _HANDLED_TYPES:
        return None

    # Lazy import to avoid an import cycle (notifier imports this module's seam).
    from app.bot.notifier import render_chat

    try:
        built = _fact_and_prompt(event)
        if built is None:  # pragma: no cover - guarded by the _HANDLED_TYPES check
            return None
        fact, system_prompt = built
        phrased = await llm_client.phrase(fact, system_prompt=system_prompt)
        if phrased is not None:
            return phrased
        return render_chat(event)
    except Exception:
        # An LLM/render hiccup must never escape into the notifier loop — fall back
        # to the deterministic line.
        logger.warning("embellish_chat_failed", event_type=etype, exc_info=True)
        return render_chat(event)
