"""Pure, Discord-send-free builder for the ``freeze.week`` embed card (260705-jo9).

This module renders a LIGHT :class:`discord.Embed` for the promoted ``freeze.week``
"lines locked" state change — a week's point spreads freezing. Per the LOCKED design
record (``.planning/notes/discord-chat-message-types.md``) the freeze is a noteworthy
state change worth a chat card, but a DELIBERATELY LIGHT one (like the pick-window
cards in :mod:`app.bot.window_embed`): just a title, a color bar, and a single
deterministic body line — NO fields, NO footer, and NO LLM quip.

The event is DUAL-DISPATCHED (260705-jo9): it keeps its terse ops-log line
(``_render`` -> "Week N lines frozen") AND newly posts this light chat card, because
``freeze_week_event`` now targets ``["logger", "chat"]``.

Like :mod:`app.bot.window_embed` it MAY import ``discord`` to construct the embed, but
it takes NO client and performs NO send — mirroring the notifier's pure/impure split so
the whole thing is unit-testable without a live gateway. The notifier wraps
:func:`build_freeze_week_embed` in a best-effort try/except and, on ANY failure, falls
back to the existing text send so the notifier loop can never die.

The builder consumes ONLY the ``week`` (and ``type``) key of the real
``freeze_week_event`` payload; it reads no other key and invents nothing.
"""

from __future__ import annotations

import discord

# Gold — a neutral/info color deliberately DISTINCT from the pick-window green
# (0x2ECC71) and red (0xE74C3C) so a lines-locked card never reads as a pick-window
# state change.
LINES_LOCKED_COLOR = 0xF1C40F  # gold — lines/spreads locked (neutral info)


def build_freeze_week_embed(event: dict) -> discord.Embed:
    """Assemble the LIGHT ``freeze.week`` "lines locked" embed card.

    Title = ``Week {week} - Lines Locked``, color = :data:`LINES_LOCKED_COLOR`,
    body = a single deterministic line naming the week. No fields, no footer — this
    is the LIGHT card. Reads ONLY the ``week`` key. Pure: constructs and returns the
    embed, performs NO send.
    """
    week = event.get("week")
    title = f"Week {week} - Lines Locked"
    body = f"Point spreads are locked for Week {week}."
    return discord.Embed(title=title, description=body, color=LINES_LOCKED_COLOR)
