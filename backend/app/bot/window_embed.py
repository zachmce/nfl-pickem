"""Pure, Discord-send-free builder for the pick-window embed cards (260705-j8o).

This module renders a LIGHT :class:`discord.Embed` for the two pick-window state
changes — ``window.opened`` and ``window.closed``. Per the LOCKED design record
(``.planning/notes/discord-chat-message-types.md``) a window opening/locking is a
noteworthy state change worth a card, but a DELIBERATELY LIGHTER one than the
outcome cards (``game.final`` / ``misc.graded``): just a title, a color bar, and a
single deterministic body line — NO fields, NO footer, and NO LLM quip.

Like :mod:`app.bot.misc_graded_embed` it MAY import ``discord`` to construct the
embed, but it takes NO client and performs NO send — mirroring the notifier's
pure/impure split so the whole thing is unit-testable without a live gateway. The
notifier wraps :func:`build_window_embed` in a best-effort try/except and, on ANY
failure, falls back to the existing text send so the notifier loop can never die.

Design decisions (LOCKED in the task PLAN.md):

* **Deterministic body, no LLM** — the window cards carry a static body line, not an
  ``embellish_chat`` quip. A light state-change card does not warrant an LLM
  round-trip; ``window.opened`` is therefore removed from the notifier's embellish
  routing tuple and both events fall to the deterministic ``render_chat`` text as the
  embed's fallback line.
* **Week in the TITLE, not the body** — ``Week {week} - Picks Open`` /
  ``Week {week} - Picks Locked``. The plain title carries NO custom app-emoji token
  (Discord does not render custom emoji in embed titles).
* **Binary color from module constants** — :data:`OPEN_COLOR` (green "go", picks
  open) / :data:`LOCKED_COLOR` (red "locked", picks closed).

The builder consumes ONLY the ``week`` and ``type`` keys of the real
``window_opened_event`` / ``window_closed_event`` payloads; it reads no other key
and invents nothing.
"""

from __future__ import annotations

import discord

# Binary open/locked colors — green "go" when picks open, red "locked" when closed.
OPEN_COLOR = 0x2ECC71  # green — picks open
LOCKED_COLOR = 0xE74C3C  # red — picks locked

# Deterministic body lines (the week lives in the title, not the body).
OPEN_BODY = "Picks are open — get 'em in!"
LOCKED_BODY = "Picks are locked. Good luck, everyone."


def build_window_embed(event: dict) -> discord.Embed:
    """Assemble the LIGHT pick-window embed card.

    * ``window.closed`` -> title ``Week {week} - Picks Locked``, :data:`LOCKED_COLOR`,
      :data:`LOCKED_BODY`;
    * otherwise (``window.opened`` / default) -> title ``Week {week} - Picks Open``,
      :data:`OPEN_COLOR`, :data:`OPEN_BODY`.

    No fields, no footer — this is the LIGHT card. Reads ONLY the ``week`` and
    ``type`` keys. Pure: constructs and returns the embed, performs NO send.
    """
    week = event.get("week")
    etype = event.get("type")
    if etype == "window.closed":
        title = f"Week {week} - Picks Locked"
        color = LOCKED_COLOR
        body = LOCKED_BODY
    else:
        title = f"Week {week} - Picks Open"
        color = OPEN_COLOR
        body = OPEN_BODY
    return discord.Embed(title=title, description=body, color=color)
