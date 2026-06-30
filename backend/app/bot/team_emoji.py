"""Team-logo Discord application-emoji resolver + chat-line decorator (260627-wt5).

Display-only personality glue for the pickem-CHAT feed: it maps a team reference
in an outgoing chat line to that team's custom Discord application emoji and inserts
the posting-form ``<:name:id>`` token after the matched reference.

Design constraints (see PLAN 260627-wt5):

* **No hardcoded emoji ids.** The application emojis are fetched ONCE at startup in
  ``client.py`` via ``fetch_application_emojis()`` and pushed into the module-level
  cache here through :func:`populate_emoji_cache` — mirroring the ``app.bot.llm_client``
  module-singleton pattern. A failed fetch simply leaves the cache empty; the resolver
  then returns ``None`` for everything and lines post undecorated.
* **Two emoji naming forms.** The live application has 33 custom emojis. Most are
  ``<nickname>logo`` (e.g. ``vikingslogo``) but four are the bare nickname
  (``packers``, ``titans``, ``chiefs``, ``bengals``), plus a generic ``nfl``. The
  resolver matches an emoji whose (lowercased) name equals the team's nickname OR the
  nickname with a ``logo`` suffix.
* **Best-effort, never raises.** :func:`decorate_team_logos` returns the original
  text unchanged on ANY internal error — it runs inside the resilient notifier send
  path and must never crash the loop.

This module is Discord-import-light: it imports no ``discord`` symbols. The populate
function accepts any object exposing ``.name`` and ``str()`` (the ``<:name:id>``
form), so it stays trivially unit-testable with a fake emoji.
"""

from __future__ import annotations

import re

# Real Team.abbreviation -> lowercase nickname (last word of display_name).
# Copied from the verified app/seeds/teams.py NFL_TEAMS table — note WSH (not WAS),
# LAR (not LA), LV (not LVR), JAX (not JAC). Do NOT invent abbreviations here.
ABBR_TO_NICKNAME: dict[str, str] = {
    "ATL": "falcons",
    "BUF": "bills",
    "CHI": "bears",
    "CIN": "bengals",
    "CLE": "browns",
    "DAL": "cowboys",
    "DEN": "broncos",
    "DET": "lions",
    "GB": "packers",
    "TEN": "titans",
    "IND": "colts",
    "KC": "chiefs",
    "LV": "raiders",
    "LAR": "rams",
    "MIA": "dolphins",
    "MIN": "vikings",
    "NE": "patriots",
    "NO": "saints",
    "NYG": "giants",
    "NYJ": "jets",
    "PHI": "eagles",
    "ARI": "cardinals",
    "PIT": "steelers",
    "LAC": "chargers",
    "SF": "49ers",
    "SEA": "seahawks",
    "TB": "buccaneers",
    "WSH": "commanders",
    "CAR": "panthers",
    "JAX": "jaguars",
    "BAL": "ravens",
    "HOU": "texans",
}

# The Capitalized whole-word nickname form used for matching in chat text — the
# nickname with its first character uppercased (matching the display_name's last
# word: "Vikings", "Commanders", "49ers"). A digit-leading nickname ("49ers") is
# unchanged by capitalize-first, which is correct.
_NICKNAME_CAPITALIZED: dict[str, str] = {
    abbr: nickname[:1].upper() + nickname[1:] for abbr, nickname in ABBR_TO_NICKNAME.items()
}

# Module-level mutable cache: lowercased emoji-name -> "<:name:id>" string.
# Populated ONCE at startup by client.py; read by resolve_logo. Mirrors the
# app.bot.llm_client module-singleton pattern.
_EMOJI_CACHE: dict[str, str] = {}


def populate_emoji_cache(emojis) -> int:
    """Replace the emoji cache from a list of Discord ``Emoji`` objects.

    Reads ``emoji.name`` (lowercased as the key) and ``str(emoji)`` (the
    ``<:name:id>`` posting form) for each. Called once at startup from
    ``client.py`` after a successful ``fetch_application_emojis()``. Replaces the
    cache wholesale so a re-populate is idempotent. Returns the entry count.
    """
    _EMOJI_CACHE.clear()
    for emoji in emojis:
        _EMOJI_CACHE[emoji.name.lower()] = str(emoji)
    return len(_EMOJI_CACHE)


def reset_emoji_cache() -> None:
    """Clear the emoji cache (test affordance / explicit reset)."""
    _EMOJI_CACHE.clear()


def resolve_logo(abbr: str) -> str | None:
    """Resolve a team abbreviation to its logo emoji string, or ``None``.

    Looks up ``abbr`` in :data:`ABBR_TO_NICKNAME` (unknown -> ``None``), then returns
    the cache entry whose key equals the nickname OR the nickname + ``logo`` (cache
    keys are already lowercased). Returns ``None`` when no such emoji is cached.
    """
    nickname = ABBR_TO_NICKNAME.get(abbr)
    if nickname is None:
        return None
    return _EMOJI_CACHE.get(nickname) or _EMOJI_CACHE.get(nickname + "logo")


def _build_logo_map() -> dict[str, str]:
    """Build the abbr -> resolved-logo map from the live cache (skipping misses)."""
    out: dict[str, str] = {}
    for abbr in ABBR_TO_NICKNAME:
        logo = resolve_logo(abbr)
        if logo is not None:
            out[abbr] = logo
    return out


def decorate_team_logos(text: str, *, logo_map: dict[str, str] | None = None) -> str:
    """Insert team-logo emoji tokens after team references in a chat line.

    For each team with a resolved logo, this tags BOTH its uppercase abbreviation
    (case-sensitive whole word, e.g. ``MIN``) and its Capitalized whole-word nickname
    (e.g. ``Vikings``) by inserting ``" " + logo`` immediately after the matched
    token. A lowercase common word (``bills``/``saints``) is never matched because
    only the Capitalized nickname form is a target. A bare abbreviation inside a
    larger word (``MINISTER``/``GBP``) is never matched (word-boundary anchored).

    The keys of ``logo_map`` are team abbreviations; if omitted the map is built
    from the module emoji cache. A team absent from the map is left undecorated.

    Already-present ``<:name:id>`` tokens are not re-decorated: the match targets are
    uppercase abbreviations and Capitalized nicknames, neither of which can match the
    lowercased emoji name or its numeric id inside an existing token.

    Best-effort: on ANY internal error the original ``text`` is returned unchanged.
    Never raises.
    """
    try:
        if not text:
            return text
        if logo_map is None:
            logo_map = _build_logo_map()
        if not logo_map:
            return text

        # Build the per-token target table: each abbreviation and its Capitalized
        # nickname point at the same logo string. Skip teams with no mapped logo or
        # a non-string logo value (defensive — keeps the function total).
        targets: dict[str, str] = {}
        for abbr, logo in logo_map.items():
            if not isinstance(logo, str):
                continue
            nickname_cap = _NICKNAME_CAPITALIZED.get(abbr)
            targets[abbr] = logo
            if nickname_cap:
                targets[nickname_cap] = logo
        if not targets:
            return text

        # Longest tokens first so an alternation prefers a longer match (e.g. a
        # multi-letter nickname over a shorter abbr that is its prefix).
        tokens = sorted(targets, key=len, reverse=True)
        pattern = re.compile(
            r"(?<![\w:])(" + "|".join(re.escape(tok) for tok in tokens) + r")(?![\w:])"
        )

        def _sub(match: re.Match) -> str:
            tok = match.group(1)
            logo = targets[tok]
            # Skip if this token is ALREADY followed by its logo (don't double
            # decorate a line that was decorated upstream or hand-authored).
            trailing = match.string[match.end() :]
            if trailing[: len(logo) + 1] == f" {logo}":
                return tok
            return f"{tok} {logo}"

        return pattern.sub(_sub, text)
    except Exception:
        return text
