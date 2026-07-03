"""Pure, Discord-send-free builder for the ``game.final`` embed card (260703-piv).

This module renders a :class:`discord.Embed` for a finished game. It MAY import
``discord`` to construct the embed, but it takes NO client and performs NO send —
mirroring the notifier's pure/impure split so the whole thing is unit-testable
without a live gateway. The notifier wraps :func:`build_game_final_embed` in a
best-effort try/except and, on ANY failure, falls back to the existing text send
so the notifier loop can never die.

Design decisions (LOCKED in the task CONTEXT.md):

* **D-01** — plain title (Discord does not render custom app-emoji in embed titles
  or field NAMES); description line 1 = a deterministic away→home logo'd score built
  from the event's own teams/scores via :func:`app.bot.team_emoji.resolve_logo`
  (order fixed by us, independent of the LLM); description line 2+ = the voiced quip.
* **D-02** — color bar = the winning team's primary color via
  :func:`app.bot.team_emoji.resolve_team_color`; a tie or an unresolvable color uses
  :data:`app.bot.team_emoji.NEUTRAL_EMBED_COLOR`.
* **D-03** — Busted/Cashed fields built from ``event["impacts"]`` (the 260703-oye
  contract) — an empty side is omitted, no fields at all when ``impacts == []``,
  mortal locks marked in the VALUE, long lists capped.

Custom app-emoji appear ONLY in the description and field VALUES — never the title
or field names, where Discord will not render them.
"""

from __future__ import annotations

import discord

from app.bot.team_emoji import NEUTRAL_EMBED_COLOR, resolve_logo, resolve_team_color

# Cap each Busted/Cashed side at this many names before collapsing the tail into a
# "+k more" suffix, so a heavily-picked game does not blow out the embed (D-03).
IMPACT_NAME_CAP = 6

# Marker appended to a mortal-lock pick in a field value (D-03).
MORTAL_LOCK_MARKER = "\U0001f512"  # 🔒

# Human wording for the two point-bearing outcomes (D-03). Field NAMES carry no
# custom emoji (Discord will not render them there).
_OUTCOME_FIELD_NAME: dict[str, str] = {
    "busted": "Busted",
    "cashed": "Cashed",
}


def _side(abbr: str, score: int) -> str:
    """Render one team's ``ABBR :logo: score`` fragment; bare abbr if uncached."""
    logo = resolve_logo(abbr)
    if logo:
        return f"{abbr} {logo} {score}"
    return f"{abbr} {score}"


def build_score_line(event: dict) -> str:
    """Deterministic away→home logo'd score line (D-01).

    Built from the event's own ``away``/``home`` abbreviations and integer scores
    (NOT the LLM's wording). A team whose logo is uncached renders its bare abbr —
    :func:`resolve_logo` returns ``None`` and never raises.
    """
    away = _side(event["away"], event["away_score"])
    home = _side(event["home"], event["home_score"])
    return f"{away} · {home}"


def select_winner_color(event: dict) -> int:
    """Winning team's primary color, else the neutral fallback (D-02).

    Higher score wins → that team's :func:`resolve_team_color`. A tie, or a winner
    whose color can not be resolved, uses :data:`NEUTRAL_EMBED_COLOR`.
    """
    away_score = event["away_score"]
    home_score = event["home_score"]
    if away_score > home_score:
        winner = event["away"]
    elif home_score > away_score:
        winner = event["home"]
    else:
        return NEUTRAL_EMBED_COLOR
    color = resolve_team_color(winner)
    return color if color is not None else NEUTRAL_EMBED_COLOR


def _format_side(items: list[dict]) -> str:
    """Format one outcome group into a field value: names (mortal locks marked),
    capped at :data:`IMPACT_NAME_CAP` with a ``+k more`` tail."""
    shown: list[str] = []
    for impact in items[:IMPACT_NAME_CAP]:
        name = str(impact.get("username", ""))
        if impact.get("was_mortal_lock"):
            name = f"{name} {MORTAL_LOCK_MARKER}"
        shown.append(name)
    text = ", ".join(shown)
    remaining = len(items) - IMPACT_NAME_CAP
    if remaining > 0:
        text += f", +{remaining} more"
    return text


def build_impact_fields(impacts: list[dict]) -> list[tuple[str, str]]:
    """Group ``event["impacts"]`` into (field_name, field_value) pairs (D-03).

    A "Busted" field for ``outcome == "busted"`` and a "Cashed" field for
    ``outcome == "cashed"``; an empty side is OMITTED; ``impacts == []`` yields NO
    fields at all. Order (mortal-lock-first, then by display name) is preserved from
    the input contract. Never raises.
    """
    if not impacts:
        return []
    fields: list[tuple[str, str]] = []
    for outcome, field_name in _OUTCOME_FIELD_NAME.items():
        side = [i for i in impacts if i.get("outcome") == outcome]
        if side:
            fields.append((field_name, _format_side(side)))
    return fields


def build_game_final_embed(event: dict, quip: str) -> discord.Embed:
    """Assemble the ``game.final`` embed card (D-01/D-02/D-03/D-04).

    * plain title (e.g. ``Week 3 · Final``) — no custom emoji;
    * description = deterministic score line, then the voiced ``quip`` verbatim (D-04);
    * color = the winning team's primary color (neutral on tie/unresolved);
    * Busted/Cashed fields from ``event["impacts"]`` (omit-empty / capped / marked).

    Pure: constructs and returns the embed, performs NO send.
    """
    week = event.get("week")
    title = f"Week {week} · Final"
    description = f"{build_score_line(event)}\n{quip}"
    embed = discord.Embed(
        title=title,
        description=description,
        color=select_winner_color(event),
    )
    for name, value in build_impact_fields(event.get("impacts") or []):
        embed.add_field(name=name, value=value, inline=False)
    return embed
