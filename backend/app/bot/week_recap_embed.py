"""Pure, Discord-send-free builder for the ``week.recap`` embed card (260705-kuv).

This module renders the marquee weekly "closing ceremony" :class:`discord.Embed` —
the message people screenshot. Like :mod:`app.bot.game_final_embed` /
:mod:`app.bot.misc_graded_embed` it MAY import ``discord`` to construct the embed,
but it takes NO client and performs NO send — mirroring the notifier's pure/impure
split so the whole thing is unit-testable without a live gateway. The notifier wraps
:func:`build_week_recap_embed` in a best-effort try/except and, on ANY failure, falls
back to the existing text send so the notifier loop can never die.

Design decisions (LOCKED in CONTEXT.md / the task PLAN.md):

* **Amethyst color** — :data:`RECAP_EMBED_COLOR` (``0x9B59B6``), a distinct marquee
  color, NOT reusing the game.final winner colors, misc green/red, freeze gold, or
  the window colors.
* **LLM narration is the description** — the caller passes the already-decorated
  ``build_week_recap`` narration verbatim; the builder does NOT re-embellish and does
  NOT touch the anti-hallucination guard.
* **Structured fields BELOW the narration, omit-empty** — Week Winner, Best Call,
  Biggest Bust, Mortal Locks, Standings. Any block with no data is OMITTED, so a
  minimal (un-enriched) payload still renders a valid card (title + color + narration
  + winner headline).
* **Upset-magnitude framing** — Best Call names the gutsiest RIGHT call (the biggest
  underdog covered) and Biggest Bust the worst miss (the biggest favorite that
  busted), each with a 🔒 marker when it was a mortal lock.
* **Field caps** — standings/board rows are capped with a ``+k more`` tail (mirroring
  :data:`app.bot.game_final_embed.IMPACT_NAME_CAP`) so a growing league never
  overflows Discord's 1024-char-per-field limit.

Custom app-emoji (team logos) appear ONLY in the description (the caller decorates
it) — never the title or field NAMES, where Discord will not render them.
"""

from __future__ import annotations

import discord

from app.bot.game_final_embed import MORTAL_LOCK_MARKER
from app.bot.misc_graded_embed import HIT_MARKER, MISS_MARKER

# Amethyst / royal purple — the marquee recap color, distinct from every other card.
RECAP_EMBED_COLOR = 0x9B59B6

# Cap the standings / mortal-lock board at this many rows before collapsing the tail
# into a "+k more" line, so a growing league never blows past Discord's 1024-char
# field limit (mirrors game_final_embed.IMPACT_NAME_CAP's spirit).
RECAP_ROW_CAP = 12


def _cap_rows(lines: list[str]) -> str:
    """Join ``lines`` with newlines, capped at :data:`RECAP_ROW_CAP` + a ``+k more`` tail."""
    shown = lines[:RECAP_ROW_CAP]
    remaining = len(lines) - RECAP_ROW_CAP
    if remaining > 0:
        shown.append(f"+{remaining} more")
    return "\n".join(shown)


def _format_delta(points: int) -> str:
    """Signed weekly delta — ``+3`` / ``-2`` for non-zero, plain ``0`` for zero.

    Zero is deliberately NOT signed (no awkward ``+0``), mirroring
    :func:`app.bot.misc_graded_embed._format_points`.
    """
    if points == 0:
        return "0"
    return f"{points:+d}"


def build_winner_field(event: dict) -> tuple[str, str] | None:
    """The ``Week Winner`` headline from the event's ``winner`` / ``winner_score``.

    Omitted (``None``) when ``winner`` is falsy — a minimal payload with no winner
    renders zero fields.
    """
    winner = event.get("winner")
    if not winner:
        return None
    return ("Week Winner", f"{winner} — {event.get('winner_score')}")


def _format_upset(impact: dict) -> str:
    """One-line best-call / biggest-bust value: name — side_label (+spread) [🔒]."""
    display_name = impact.get("display_name")
    side_label = impact.get("side_label")
    spread = impact.get("spread")
    line = f"{display_name} — {side_label} (+{spread})"
    if impact.get("is_mortal_lock"):
        line = f"{line} {MORTAL_LOCK_MARKER}"
    return line


def build_best_call_field(event: dict) -> tuple[str, str] | None:
    """The ``Best Call`` field from ``event["best_call"]``; omitted when ``None``."""
    best_call = event.get("best_call")
    if not best_call:
        return None
    return ("Best Call", _format_upset(best_call))


def build_bust_field(event: dict) -> tuple[str, str] | None:
    """The ``Biggest Bust`` field from ``event["biggest_bust"]``; omitted when ``None``."""
    biggest_bust = event.get("biggest_bust")
    if not biggest_bust:
        return None
    return ("Biggest Bust", _format_upset(biggest_bust))


def build_mortal_lock_field(event: dict) -> tuple[str, str] | None:
    """The ``Mortal Locks`` board from ``event["mortal_locks"]``; omitted when empty.

    One line per row — ``✅``/``❌`` by ``hit``, the signed ``points``, and the
    ``side_label`` — capped with a ``+k more`` tail.
    """
    rows = event.get("mortal_locks") or []
    if not rows:
        return None
    lines = [
        f"{HIT_MARKER if row.get('hit') else MISS_MARKER} "
        f"{row.get('display_name')} ({_format_delta(row.get('points', 0))}) "
        f"— {row.get('side_label')}"
        for row in rows
    ]
    return ("Mortal Locks", _cap_rows(lines))


def build_standings_field(event: dict) -> tuple[str, str] | None:
    """The ``Standings`` field from ``event["standings"]``; omitted when empty.

    One line per row — ``{rank}. {display_name} — {season_total} ({+week_delta}
    this wk)`` (zero delta rendered plainly, no ``+0``) — capped with a ``+k more``
    tail so the value stays within Discord's 1024-char field limit.
    """
    rows = event.get("standings") or []
    if not rows:
        return None
    lines = [
        f"{row.get('rank')}. {row.get('display_name')} — {row.get('season_total')} "
        f"({_format_delta(row.get('week_delta', 0))} this wk)"
        for row in rows
    ]
    return ("Standings", _cap_rows(lines))


def build_week_recap_embed(event: dict, narration: str) -> discord.Embed:
    """Assemble the amethyst ``week.recap`` "closing ceremony" embed card.

    * plain title = ``Week {week} - Recap`` (no custom emoji);
    * color = :data:`RECAP_EMBED_COLOR` (amethyst);
    * description = ``narration`` verbatim (the already-decorated LLM column the
      caller passes in — the builder does NOT re-embellish and does NOT touch the
      guard);
    * omit-empty fields BELOW the narration, in order: Week Winner, Best Call,
      Biggest Bust, Mortal Locks, Standings.

    Every block reads ONLY the settled event keys, so a minimal (un-enriched) payload
    renders a valid card (title + color + narration + winner headline). Pure:
    constructs and returns the embed, performs NO send; never raises on a minimal
    payload.
    """
    week = event.get("week")
    embed = discord.Embed(
        title=f"Week {week} - Recap",
        description=narration,
        color=RECAP_EMBED_COLOR,
    )
    for builder in (
        build_winner_field,
        build_best_call_field,
        build_bust_field,
        build_mortal_lock_field,
        build_standings_field,
    ):
        field = builder(event)
        if field is not None:
            embed.add_field(name=field[0], value=field[1], inline=False)
    return embed
