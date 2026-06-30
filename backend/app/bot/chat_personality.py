"""Best-effort chat personality layer for the Tier-1 reactive events (260627-t5u,
enriched 260627-vpc).

This is the orchestrator that gives three existing chat events real personality
without giving up the v1 safety contract (260627-nef): the BOT owns a
deterministic, hallucination-proof FACT and the LLM ONLY phrases it. On ANY db OR
LLM failure the deterministic :func:`app.bot.notifier.render_chat` line is
returned, so each event always produces exactly one line.

Enriched read->fact->phrase->fallback (260627-vpc)
--------------------------------------------------
For the three handled events the FACT is no longer pure flavor: it is built from
REAL, DISPLAY-ONLY DB context read through thin async seams over the Tier-2
:mod:`app.bot.db_bridge` wrappers (mirroring :mod:`app.bot.recap`). On any read
miss/empty OR a raised read the fact degrades to a basic event-field fact, then to
``render_chat`` — embellish_chat NEVER raises and always posts exactly ONE line.

Handled here (and ONLY here):

* ``window.opened``   -> a hype line STATING the season leader (+ runner-up + gap).
* ``game.final``      -> a reaction line STATING teams + final score + the line
  result (spread cover + over/under vs the FROZEN line) + a notable pick impact
  (mortal-lock hit/bust by display_name). The game is FINAL/public.
* ``roster.complete`` -> a light-ribbing line STATING the actor's public rank +
  season total and the week completion COUNT. The window is OPEN, so the fact
  carries ONLY the COUNT of who is outstanding — NEVER their names, NEVER any pick
  content (LEAK-SAFE, T-vpc / T-t5u-01).

Everything else (``window.closed``, ``week.recap``, unknown types) returns ``None``
— the notifier keeps owning those via its existing deterministic path /
``build_lock_commentary`` / ``build_week_recap``.

Best-effort posture mirrors :mod:`app.bot.recap`: structlog only, NO ``discord``
import, and the whole body is wrapped so any error falls back to the deterministic
line — :func:`embellish_chat` NEVER raises (T-t5u-03 / T-vpc).
"""

from __future__ import annotations

import structlog

from app.bot import llm_client
from app.bot.personality import compose_prompt

logger = structlog.get_logger(__name__)

# Tier-1 event types this seam phrases. Everything else is the notifier's job.
_HANDLED_TYPES = frozenset(
    {"window.opened", "game.final", "roster.complete", "misc.graded", "misc.picked"}
)

# Margin thresholds (points) for the COMPUTED game.final descriptor.
_NAIL_BITER_MARGIN = 3  # margin <= 3 -> a nail-biter
_BLOWOUT_MARGIN = 17  # margin >= 17 -> a blowout

# The shared STATE-FACTS-FIRST + anti-hallucination guard every embellished prompt
# composes LAST. It mirrors recap.RECAP_PROMPT's discipline: say the news (the
# supplied numbers/names) FIRST, never let flavor replace it, and invent NOTHING
# beyond the facts handed in (no stats, no line values, no movement).
#
# INVARIANT (260627-xbb): this guard — and the per-event ROLE leak/verdict clauses
# below — are byte-identical for EVERY personality. They stay OUT of the swappable
# voice preamble (see app.bot.personality): a personality may ONLY change the
# leading voice sentence, never weaken or relocate these guarantees. Each event's
# system prompt is built at call time as
# ``compose_prompt(<active voice>, <ROLE line>, _FACTS_FIRST_GUARD)``.
_FACTS_FIRST_GUARD = (
    "STATE THE CONCRETE FACTS FIRST (scores, teams, names, numbers), THEN add a "
    "little personality — flavor must NEVER replace the news. Use ONLY the facts "
    "you are given: do NOT invent any stat, line value, point spread, total, or "
    "movement that is not in the facts. Reply with ONE short line and at most one "
    "or two emoji."
)

# Per-event ROLE lines: the event-specific CONTEXT (not the voice). The
# roster.complete leak clause and the misc.graded verdict-preservation clause live
# here and are byte-identical across every personality (they are correctness
# guarantees, not flavor). The leading voice sentence is supplied separately by the
# active personality at compose time.
_WINDOW_OPENED_ROLE = "You are announcing that the pick window just opened."

_GAME_FINAL_ROLE = "You are reacting to a game that just went final."

_ROSTER_COMPLETE_ROLE = (
    "You are reacting to a player making/setting their full roster of picks for "
    "the week — they got their picks in (the SYSTEM locks them at kickoff; the "
    "player does NOT 'lock in', so never use that phrase). You are told their "
    "public standing and a COUNT of how many players still have not submitted — "
    "you do NOT know who they are or what anyone picked, so NEVER name another "
    "player and NEVER guess any pick."
)

_MISC_GRADED_ROLE = (
    "You are reacting to an admin grading a player's MISC prediction. State the "
    "player, their prediction, whether it was correct or incorrect, and the points "
    "FIRST — then add a little personality. The prediction text is the player's own "
    "words; quote it as given and do NOT alter the verdict or the points."
)

_MISC_PICKED_ROLE = (
    "You are announcing that a player just got their MISC call/prediction in for "
    "the week. You know ONLY that they made the call and the week — you do NOT "
    "know the prediction text and it is hidden until the window closes, so NEVER "
    "guess, invent, hint at, or state what they predicted, and NEVER name another "
    "player."
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


# --------------------------------------------------------------------------- #
# Thin async DB-context seams (260627-vpc). Lazy-import + delegate to the Tier-2
# db_bridge wrappers, mirroring recap._recap_context, so tests monkeypatch them at
# the chat_personality seam without a real db.
# --------------------------------------------------------------------------- #


async def _game_final_context(event: dict) -> dict:
    """Display-only game.final context for ``event`` via the db_bridge wrapper."""
    from app.bot.db_bridge import get_game_final_context_async

    return await get_game_final_context_async(
        event.get("week"), event.get("away"), event.get("home")
    )


async def _roster_complete_context(event: dict) -> dict:
    """Display-only roster.complete context for ``event`` via the db_bridge wrapper."""
    from app.bot.db_bridge import get_roster_complete_context_async

    return await get_roster_complete_context_async(event.get("week"), event.get("actor"))


async def _leaders_context(event: dict) -> dict:
    """Display-only season-leaders context via the db_bridge wrapper."""
    from app.bot.db_bridge import get_leaders_context_async

    return await get_leaders_context_async()


async def _active_voice() -> str:
    """Resolve the active personality's voice preamble via the db_bridge seam.

    The DB read happens INSIDE db_bridge's async/thread seam (never from the pure
    ``llm_client.phrase`` layer); falls back to the sarcastic voice on any
    miss/unset/raise. Lazy import mirrors the context seams above so tests can
    monkeypatch this seam without a real db.
    """
    from app.bot.db_bridge import resolve_active_voice_async

    return await resolve_active_voice_async()


# --------------------------------------------------------------------------- #
# Basic event-field facts (the fallback when a context read misses or raises).
# --------------------------------------------------------------------------- #


def _basic_game_final_fact(event: dict) -> str:
    """A deterministic game.final fact from the event fields ONLY (no db).

    Teams + final scores + the COMPUTED margin descriptor — the same shape the
    t5u layer used, kept as the safe fallback.
    """
    home = event.get("home")
    away = event.get("away")
    home_score = event.get("home_score")
    away_score = event.get("away_score")
    descriptor = _final_descriptor(home_score, away_score)
    fact = f"Week {event.get('week')} final: {home} {home_score}, {away} {away_score}."
    if descriptor:
        fact = f"{fact} It was a {descriptor}."
    return fact


def _basic_roster_complete_fact(event: dict) -> str:
    """A deterministic roster.complete fact from the event fields ONLY (LEAK-SAFE).

    actor + week and NOTHING pick-related — worded to avoid even substrings of the
    pick-content leak tokens so the LEAK-SAFE guard holds by construction.
    """
    return (
        f"{event.get('actor')} just finished their full Week "
        f"{event.get('week')} roster and submitted it."
    )


def _basic_window_opened_fact(event: dict) -> str:
    """A deterministic window.opened fact from the event fields ONLY (week)."""
    return f"Week {event.get('week')} pick window just opened."


def _basic_misc_graded_fact(event: dict) -> str:
    """A STATE-FACTS-FIRST misc.graded fact from the event fields ONLY (no db).

    This event carries ALL its facts in the payload (unlike the enriched Tier-1
    events), so there is NO db read / context seam: actor + week + the quoted
    prediction + the verdict word + the SIGNED points, stated before any flavor.
    """
    return (
        f"Week {event.get('week')}: {event.get('actor')}'s MISC prediction "
        f'"{event.get("prediction")}" was graded {event.get("verdict")} '
        f"for {event.get('points'):+d} points."
    )


# --------------------------------------------------------------------------- #
# Enriched facts built from the DB context (None -> use the basic fallback).
# --------------------------------------------------------------------------- #


def _basic_misc_picked_fact(event: dict) -> str:
    """A deterministic misc.picked fact from the event fields ONLY (LEAK-SAFE).

    actor + week and NOTHING prediction-related — this event carries no enriched
    context (and no db read), and the prediction text is hidden until the window
    closes, so the fact STATES only that the player got their misc call in.
    """
    return f"{event.get('actor')} just submitted their Week {event.get('week')} misc call."


def _enriched_game_final_fact(event: dict, context: dict) -> str | None:
    """Build the STATE-FACTS-FIRST game.final fact, or ``None`` to fall back.

    States teams + final score, the line result (spread cover + over/under vs the
    FROZEN line), and the most notable pick impact (a mortal-lock hit/bust by
    display_name when present, else the first impact). The game is FINAL/public, so
    naming pick winners/losers is fine. Returns ``None`` when the context did not
    resolve THIS game (caller uses the basic fact).
    """
    if not context.get("found"):
        return None

    home = context.get("home")
    away = context.get("away")
    home_score = context.get("home_score")
    away_score = context.get("away_score")
    descriptor = _final_descriptor(home_score, away_score)

    parts = [f"Week {event.get('week')} final: {home} {home_score}, {away} {away_score}."]
    if descriptor:
        parts.append(f"It was a {descriptor}.")

    spread = context.get("spread_result")
    if spread:
        fav = spread.get("favorite_abbr")
        line = spread.get("spread")
        if spread.get("did_cover"):
            parts.append(f"{fav} covered the {line}.")
        else:
            parts.append(f"{fav} (favored by {line}) failed to cover.")

    total = context.get("total_result")
    if total:
        ou = "over" if total.get("went_over") else "under"
        parts.append(f"Combined points went {ou} the {total.get('total')}.")

    impacts = context.get("pick_impacts") or []
    notable = next((i for i in impacts if i.get("is_mortal_lock")), None)
    if notable is None and impacts:
        notable = impacts[0]
    if notable is not None:
        name = notable.get("display_name")
        outcome = notable.get("outcome")
        ml = " mortal-lock" if notable.get("is_mortal_lock") else ""
        if outcome == "WIN":
            parts.append(f"{name}'s{ml} call on it hit.")
        elif outcome == "LOSS":
            parts.append(f"{name}'s{ml} call on it busted.")

    return " ".join(parts)


def _enriched_roster_complete_fact(event: dict, context: dict) -> str | None:
    """Build the STATE-FACTS-FIRST, LEAK-SAFE roster.complete fact, or ``None``.

    States the actor's public rank + season total and the week completion as a
    COUNT — "first to lock in" when ``completed_count == 1`` else "N still on the
    clock". HARD LEAK RULE (window OPEN): only the COUNT crosses — never the names
    of who is outstanding, never any pick content. Returns ``None`` when the
    context carries no usable counts (caller uses the basic fact).
    """
    total_players = context.get("total_players")
    completed_count = context.get("completed_count")
    if not total_players:
        return None

    actor = context.get("actor") or event.get("actor")
    rank = context.get("rank")
    season_total = context.get("season_total")

    # Worded to avoid even substrings of the pick-content leak tokens (over, under,
    # favorite, underdog, spread, cover, moneyline, mortal, lock, slot, pick) so the
    # LEAK-SAFE guard holds by construction — the window is OPEN.
    parts = [f"{actor} just finished their full Week {event.get('week')} roster."]
    if rank is not None:
        parts.append(f"They sit at #{rank} on the season with {season_total}.")

    outstanding = context.get("outstanding_count")
    if completed_count == 1:
        parts.append("They were the first to lock in.")
    elif outstanding and outstanding > 0:
        parts.append(f"{outstanding} player(s) still have not submitted.")
    else:
        parts.append("Everyone has now submitted.")

    return " ".join(parts)


def _enriched_window_opened_fact(event: dict, context: dict) -> str | None:
    """Build the STATE-FACTS-FIRST window.opened fact, or ``None`` to fall back.

    States the season leader (+ runner-up + gap when present) by display_name and
    total. Returns ``None`` when there is no leader yet (caller uses the basic
    week-number fact).
    """
    leader = context.get("leader")
    if not leader:
        return None

    leader_total = context.get("leader_total")
    runner_up = context.get("runner_up")
    gap = context.get("gap")

    parts = [f"Week {event.get('week')} pick window is open."]
    if runner_up and gap == 0:
        # A zero gap means they are CO-LEADERS — phrase it as a tie for the lead,
        # not "leader" + "runner-up is 0 back in second" (which reads as a false
        # first/second split and led the model to say "tied for second").
        parts.append(f"{leader} and {runner_up} are tied for the lead with {leader_total}.")
    else:
        parts.append(f"{leader} leads the season with {leader_total}.")
        if runner_up:
            parts.append(f"{runner_up} is {gap} back in second.")
    return " ".join(parts)


async def _enriched_fact_and_prompt(event: dict, active_voice: str) -> tuple[str, str] | None:
    """Build the (fact, system_prompt) for a handled event, reading DB context.

    Reads the matching display-only context through the async seam, builds the
    enriched STATE-FACTS-FIRST fact, and degrades to the basic event-field fact
    when the context misses/empties OR the read raises. The system prompt is
    composed at call time as ``compose_prompt(active_voice, <ROLE>,
    _FACTS_FIRST_GUARD)`` — the swappable ``active_voice`` (resolved upstream in
    the db_bridge seam) leads, the per-event ROLE + the byte-identical guard
    follow. Returns ``None`` only for a non-handled type (guarded upstream).
    """
    etype = event.get("type")

    if etype == "window.opened":
        try:
            context = await _leaders_context(event)
            fact = _enriched_window_opened_fact(event, context)
        except Exception:
            logger.warning("window_opened_context_failed", exc_info=True)
            fact = None
        prompt = compose_prompt(active_voice, _WINDOW_OPENED_ROLE, _FACTS_FIRST_GUARD)
        return (fact or _basic_window_opened_fact(event)), prompt

    if etype == "game.final":
        try:
            context = await _game_final_context(event)
            fact = _enriched_game_final_fact(event, context)
        except Exception:
            logger.warning("game_final_context_failed", exc_info=True)
            fact = None
        prompt = compose_prompt(active_voice, _GAME_FINAL_ROLE, _FACTS_FIRST_GUARD)
        return (fact or _basic_game_final_fact(event)), prompt

    if etype == "roster.complete":
        try:
            context = await _roster_complete_context(event)
            fact = _enriched_roster_complete_fact(event, context)
        except Exception:
            logger.warning("roster_complete_context_failed", exc_info=True)
            fact = None
        prompt = compose_prompt(active_voice, _ROSTER_COMPLETE_ROLE, _FACTS_FIRST_GUARD)
        return (fact or _basic_roster_complete_fact(event)), prompt

    if etype == "misc.graded":
        # This event carries all its facts in the payload — no db read / context
        # seam. Build the STATE-FACTS-FIRST fact directly from the event fields.
        prompt = compose_prompt(active_voice, _MISC_GRADED_ROLE, _FACTS_FIRST_GUARD)
        return _basic_misc_graded_fact(event), prompt

    if etype == "misc.picked":
        # Like misc.graded, no db/context seam — but LEAK-SAFE: the fact carries
        # only actor + week (the prediction is hidden until the window closes).
        prompt = compose_prompt(active_voice, _MISC_PICKED_ROLE, _FACTS_FIRST_GUARD)
        return _basic_misc_picked_fact(event), prompt

    return None


async def embellish_chat(event: dict) -> str | None:
    """Return one personality line for a Tier-1 chat ``event``, or ``None``.

    For ``window.opened`` / ``game.final`` / ``roster.complete``: read the
    display-only DB context, build a STATE-FACTS-FIRST fact (degrading to a basic
    event-field fact on any read miss/failure), ask
    :func:`app.bot.llm_client.phrase` to phrase it under the event's prompt, and
    return the phrased line — or the deterministic
    :func:`app.bot.notifier.render_chat` line if the LLM returns ``None``. For any
    other type return ``None`` (the notifier owns it).

    Best-effort (T-t5u-03 / T-vpc): the entire body is guarded so ANY db OR LLM
    error falls back to the deterministic render_chat line (for a handled type) or
    ``None`` (otherwise). NEVER raises out into the notifier loop, NEVER returns
    None for a handled type, posts exactly ONE line per event.
    """
    etype = event.get("type")
    if etype not in _HANDLED_TYPES:
        return None

    # Lazy import to avoid an import cycle (notifier imports this module's seam).
    from app.bot.notifier import render_chat

    try:
        # Resolve the active voice INSIDE the async/db seam (never from the pure
        # phrase layer); falls back to the sarcastic voice on any miss/raise.
        active_voice = await _active_voice()
        built = await _enriched_fact_and_prompt(event, active_voice)
        if built is None:  # pragma: no cover - guarded by the _HANDLED_TYPES check
            return None
        fact, system_prompt = built
        phrased = await llm_client.phrase(fact, system_prompt=system_prompt)
        if phrased is not None:
            return phrased
        return render_chat(event)
    except Exception:
        # A db/LLM/render hiccup must never escape into the notifier loop — fall
        # back to the deterministic line.
        logger.warning("embellish_chat_failed", event_type=etype, exc_info=True)
        return render_chat(event)
