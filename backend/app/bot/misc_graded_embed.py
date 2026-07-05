"""Pure, Discord-send-free builder for the ``misc.graded`` embed card (260705-if1).

This module renders a LIGHT :class:`discord.Embed` for a graded MISC prediction —
a genuine OUTCOME (a verdict with a point swing), so per the LOCKED design record
(``.planning/notes/discord-chat-message-types.md``) it earns a rich card rather than
a nudge. It is deliberately LIGHTER than the ``game.final`` card: a binary hit/miss
color, a single marker line, and three compact single-player fields — NO team-color
computation and NO multi-name Busted/Cashed lists (there is exactly one player).

Like :mod:`app.bot.game_final_embed` it MAY import ``discord`` to construct the embed,
but it takes NO client and performs NO send — mirroring the notifier's pure/impure
split so the whole thing is unit-testable without a live gateway. The notifier wraps
:func:`build_misc_graded_embed` in a best-effort try/except and, on ANY failure, falls
back to the existing text send so the notifier loop can never die.

Design decisions (LOCKED in the task PLAN.md):

* **Hit vs miss is by POINTS SIGN, not the verdict word** — ``is_hit = points > 0``;
  ``points <= 0`` (including 0) is a miss. The ``verdict`` word is displayed VERBATIM
  from the payload, but the COLOR + MARKER are chosen from the points sign.
* **Binary color from module constants** — :data:`HIT_COLOR` (green) / :data:`MISS_COLOR`
  (red), NOT a team-color lookup.
* **The quip is passed through VERBATIM** — it is the already-embellished, already-
  decorated notifier ``line`` (the ``embellish_chat`` output whose anti-hallucination
  guard is left byte-identical); the builder does NOT re-embellish.
* **Omit-empty on the free-text prediction field** — mirrors game_final's omit-empty
  discipline; the other two fields (Player, Verdict) always render.

The builder consumes ONLY the real ``misc_graded_event`` keys: ``actor``, ``week``,
``prediction``, ``verdict``, ``points``. It reads no other key and never re-derives
the verdict from points.
"""

from __future__ import annotations

import discord

# Binary hit/miss colors — this card is intentionally binary-colored, NOT team-colored.
HIT_COLOR = 0x2ECC71  # green — a hit (points > 0)
MISS_COLOR = 0xE74C3C  # red — a miss (points <= 0)

# Single hit/miss markers (this card carries one marker line, not a score line).
HIT_MARKER = "✅"  # ✅
MISS_MARKER = "❌"  # ❌


def is_hit(event: dict) -> bool:
    """Whether the graded prediction is a hit.

    The SIGN of ``points`` — NOT the verdict word — drives color/marker per the design:
    ``points > 0`` is a hit; ``points <= 0`` (including 0) is a miss.
    """
    return event["points"] > 0


def select_color(event: dict) -> int:
    """:data:`HIT_COLOR` when :func:`is_hit`, else :data:`MISS_COLOR`."""
    return HIT_COLOR if is_hit(event) else MISS_COLOR


def build_marker_line(event: dict) -> str:
    """The single marker line: ``✅ Cashed`` on a hit, else ``❌ Busted``."""
    if is_hit(event):
        return f"{HIT_MARKER} Cashed"
    return f"{MISS_MARKER} Busted"


def build_verdict_value(event: dict) -> str:
    """``{verdict} ({points:+d})`` — the verdict word carried VERBATIM plus signed points.

    A positive point swing shows a leading ``+`` (e.g. ``correct (+3)``); a negative one
    shows its own sign (e.g. ``incorrect (-2)``).
    """
    return f"{event['verdict']} ({event['points']:+d})"


def build_misc_graded_embed(event: dict, quip: str) -> discord.Embed:
    """Assemble the LIGHT ``misc.graded`` embed card.

    * plain title (e.g. ``Week 3 - MISC Graded``) — no custom emoji;
    * description = the marker line, then the voiced ``quip`` verbatim;
    * color = binary hit/miss (:data:`HIT_COLOR` / :data:`MISS_COLOR`) by points sign;
    * fields = Player (actor) + Verdict (verdict word + signed points), and a Prediction
      field ONLY when the free-text prediction is non-empty (omit-empty discipline).

    Pure: constructs and returns the embed, performs NO send.
    """
    week = event.get("week")
    title = f"Week {week} - MISC Graded"
    description = f"{build_marker_line(event)}\n{quip}"
    embed = discord.Embed(
        title=title,
        description=description,
        color=select_color(event),
    )
    embed.add_field(name="Player", value=event.get("actor"), inline=True)
    embed.add_field(name="Verdict", value=build_verdict_value(event), inline=True)
    prediction = str(event.get("prediction") or "").strip()
    if prediction:
        embed.add_field(name="Prediction", value=prediction, inline=False)
    return embed
