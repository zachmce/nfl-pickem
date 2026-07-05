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

Design decisions (LOCKED in the task PLAN.md + the post-live-fire redesign):

* **Hit vs miss is by POINTS SIGN, not the verdict word** — ``is_hit = points > 0``;
  ``points <= 0`` (including 0) is a miss. The ``verdict`` word is displayed VERBATIM
  from the payload, but the COLOR + MARKER are chosen from the points sign.
* **Binary color from module constants** — :data:`HIT_COLOR` (green) / :data:`MISS_COLOR`
  (red), NOT a team-color lookup.
* **Player in the TITLE** — ``Week {week} - MISC Graded · {actor}`` (middot before the
  player), keeping the ``Week X - <Event>`` parallel with game.final. No Player field.
* **Two inline columns** — ``Result`` (``✅ Cashed`` / ``❌ Busted``) and ``Verdict``
  (``{verdict} ({pts})``). ``pts`` is signed (``+3`` / ``-2``) EXCEPT zero renders as a
  plain ``0`` (no awkward ``+0``): ``correct (+3)`` / ``incorrect (-2)`` / ``incorrect (0)``.
* **Quip at the BOTTOM** — Discord renders ``description`` ABOVE fields, so the voiced
  quip is the LAST full-width field with a zero-width-space name (only the text shows);
  omitted when blank. It is the already-embellished, already-decorated notifier ``line``
  (the ``embellish_chat`` output whose anti-hallucination guard is byte-identical); the
  builder does NOT re-embellish. The quip already references the prediction, so there is
  no separate Prediction field.
* **Grader footer** — when the event carries a truthy ``grader`` (the admin's display
  name), ``Graded by {grader}`` renders as the embed footer; omitted otherwise. The
  builder renders fine for older/synthetic events with NO ``grader`` key.

The builder consumes ONLY the real ``misc_graded_event`` keys: ``actor``, ``week``,
``verdict``, ``points``, and the optional ``grader``. It reads no other key and never
re-derives the verdict from points.
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


def build_result_value(event: dict) -> str:
    """The ``Result`` field value: ``✅ Cashed`` on a hit, else ``❌ Busted``."""
    if is_hit(event):
        return f"{HIT_MARKER} Cashed"
    return f"{MISS_MARKER} Busted"


def _format_points(points: int) -> str:
    """Signed points display — ``+3`` / ``-2`` for non-zero, plain ``0`` for zero.

    Zero is deliberately NOT signed (no awkward ``+0`` on an ``incorrect (0)`` grade).
    """
    if points == 0:
        return "0"
    return f"{points:+d}"


def build_verdict_value(event: dict) -> str:
    """``{verdict} ({pts})`` — the verdict word carried VERBATIM plus the points.

    ``pts`` is signed for non-zero swings (``correct (+3)`` / ``incorrect (-2)``) and a
    plain ``0`` when the graded points are zero (``incorrect (0)``).
    """
    return f"{event['verdict']} ({_format_points(event['points'])})"


# Zero-width space — used as the name of the quip field so ONLY the quip text renders
# (Discord requires a non-empty field name, but this shows as nothing).
_ZERO_WIDTH_SPACE = "​"


def build_misc_graded_embed(event: dict, quip: str) -> discord.Embed:
    """Assemble the LIGHT ``misc.graded`` embed card (post-live-fire layout).

    * plain title = ``Week {week} - MISC Graded · {actor}`` (player in the title);
    * color = binary hit/miss (:data:`HIT_COLOR` / :data:`MISS_COLOR`) by points sign;
    * two inline fields — ``Result`` (✅ Cashed / ❌ Busted) and ``Verdict``
      (verdict word + points, zero unsigned);
    * a final full-width quip field (zero-width name) so the voiced ``quip`` sits at the
      BOTTOM below the columns — omitted when the quip is blank;
    * a ``Graded by {grader}`` footer when the event carries a truthy ``grader``.

    No description, no Player field, no Prediction field (the quip references the
    prediction). Pure: constructs and returns the embed, performs NO send.
    """
    week = event.get("week")
    actor = event.get("actor")
    title = f"Week {week} - MISC Graded · {actor}"
    embed = discord.Embed(title=title, color=select_color(event))
    embed.add_field(name="Result", value=build_result_value(event), inline=True)
    embed.add_field(name="Verdict", value=build_verdict_value(event), inline=True)
    quip_text = str(quip or "").strip()
    if quip_text:
        # Discord renders description above fields; put the quip LAST as a full-width
        # field with a zero-width name so only the text shows, keeping it at the bottom.
        embed.add_field(name=_ZERO_WIDTH_SPACE, value=quip, inline=False)
    grader = event.get("grader")
    if grader:
        embed.set_footer(text=f"Graded by {grader}")
    return embed
