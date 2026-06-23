"""Preordained bot-picks dataset for the season-walkthrough oracle.

This is a **pure static data module** — no I/O, no DB, no network. It imports only
:class:`app.models.PickType` and the standard library. It encodes, per bot, per
week, the picks each demo bot (see ``app/seeds/bots.py``) makes against the 2025
season fixture (``app/seeds/data/nfl_2025_regular_season.json``).

Authoring guarantees
--------------------

* Games are referenced by their **stable ``espn_event_id``**, never the surrogate
  ``Game.id`` PK (which is assigned at seed time and is not stable across runs).
* Every referenced ``espn_event_id`` was hand-verified to exist in the fixture for
  the stated week and to be a FINAL game **carrying odds** — the fixture has 79
  games with ``odds: null``, on which ``FAVORITE_COVER`` / ``UNDERDOG_COVER`` are
  ineligible (``_is_true_pickem``). The test
  ``tests/test_bot_picks_dataset.py`` re-asserts both facts (real game in the
  right week) so the dataset can never silently drift from the fixture.
* Every bot/week roster obeys the conflict rules in
  :mod:`app.services.pick_validation`: no duplicate ``pick_type`` on a game, no
  ``OVER``+``UNDER`` or ``FAVORITE_COVER``+``UNDERDOG_COVER`` on the same game,
  at most one mortal lock, and no spread pick on an odds-less game. The dataset
  test asserts ``validate_roster(...).ok is True`` for **every** roster.
* A **full** roster is the four base types on four distinct games plus one mortal
  lock on a fifth distinct game (5 records). At least one roster is intentionally
  **partial** (fewer than 5 picks) to exercise partial scoring downstream.

Shape
-----

``BOT_PICKS`` maps each bot's ``display_name`` (matching ``BOT_ACCOUNTS`` in
``app/seeds/bots.py``) to a dict of ``week_number -> list[BotPick]``. Each
:class:`BotPick` carries ``(espn_event_id, pick_type, is_mortal_lock)`` — exactly
the ``(game, type, mortal-lock)`` triple ``pick_submission`` builds a ``Pick``
from, so the demo driver (#5) maps the dataset cleanly by resolving
``espn_event_id`` to the seeded ``Game.id``.

> Note: on this machine there is no bare ``python`` on ``PATH``; use the venv
> interpreter ``.venv/bin/python`` for any commands.
"""

from __future__ import annotations

from typing import NamedTuple

from app.models import PickType


class BotPick(NamedTuple):
    """One preordained bot pick, keyed on the stable ESPN event id.

    ``espn_event_id`` resolves to a real seeded ``Game``; ``pick_type`` is one of
    the four base types; ``is_mortal_lock`` marks the (single) per-week mortal
    lock. This mirrors ``pick_submission``'s ``(game_id, pick_type,
    is_mortal_lock)`` shape (game resolved from the event id at persist time).
    """

    espn_event_id: int
    pick_type: PickType
    is_mortal_lock: bool


# Convenience aliases for readability below.
_FAV = PickType.FAVORITE_COVER
_DOG = PickType.UNDERDOG_COVER
_OVER = PickType.OVER
_UNDER = PickType.UNDER

# Per-bot, per-week preordained picks. All event ids are odds-bearing FINAL games
# hand-verified against the fixture for the stated week (see module docstring).
#
# Verified anchor event-ids by week (all carry odds, all FINAL):
#   week 1: 401772510, 401772714, 401772830, 401772829, 401772719
#   week 2: 401772936, 401772725, 401772834, 401772835, 401772724
#   week 3: 401772937, 401772842, 401772733, 401772731, 401772732,
#           401772839, 401772840
BOT_PICKS: dict[str, dict[int, list[BotPick]]] = {
    # ---- bot_alice: full rosters every week --------------------------------
    "bot_alice": {
        1: [
            BotPick(401772510, _DOG, False),
            BotPick(401772714, _FAV, False),
            BotPick(401772830, _OVER, False),
            BotPick(401772829, _UNDER, False),
            BotPick(401772719, _FAV, True),  # mortal lock
        ],
        2: [
            BotPick(401772936, _FAV, False),
            BotPick(401772725, _DOG, False),
            BotPick(401772834, _OVER, False),
            BotPick(401772835, _UNDER, False),
            BotPick(401772724, _FAV, True),  # mortal lock
        ],
        3: [
            BotPick(401772937, _DOG, False),
            BotPick(401772733, _FAV, False),
            BotPick(401772731, _OVER, False),
            BotPick(401772732, _UNDER, False),
            BotPick(401772839, _FAV, True),  # mortal lock
        ],
    },
    # ---- bot_bob: full rosters, different picks ----------------------------
    "bot_bob": {
        1: [
            BotPick(401772510, _OVER, False),
            BotPick(401772714, _DOG, False),
            BotPick(401772830, _UNDER, False),
            BotPick(401772829, _FAV, False),
            BotPick(401772719, _OVER, True),  # mortal lock
        ],
        2: [
            BotPick(401772936, _OVER, False),
            BotPick(401772725, _FAV, False),
            BotPick(401772834, _DOG, False),
            BotPick(401772835, _OVER, False),
            BotPick(401772724, _UNDER, True),  # mortal lock
        ],
        3: [
            BotPick(401772937, _OVER, False),
            BotPick(401772733, _DOG, False),
            BotPick(401772731, _FAV, False),
            BotPick(401772732, _UNDER, False),
            BotPick(401772840, _OVER, True),  # mortal lock
        ],
    },
    # ---- bot_carol: a spread pick + a total pick on the SAME game (allowed),
    #      proving independent types coexist -----------------------------------
    "bot_carol": {
        1: [
            BotPick(401772510, _DOG, False),
            BotPick(401772510, _UNDER, False),  # same game, independent type
            BotPick(401772714, _OVER, False),
            BotPick(401772829, _FAV, False),
            BotPick(401772830, _UNDER, True),  # mortal lock
        ],
        2: [
            BotPick(401772936, _FAV, False),
            BotPick(401772725, _OVER, False),
            BotPick(401772834, _UNDER, False),
            BotPick(401772835, _FAV, True),  # mortal lock
        ],
        3: [
            BotPick(401772733, _FAV, False),
            BotPick(401772731, _UNDER, False),
            BotPick(401772732, _OVER, False),
            BotPick(401772840, _DOG, False),
            BotPick(401772937, _DOG, True),  # mortal lock
        ],
    },
    # ---- bot_dave: includes a PARTIAL week (week 2 has only 3 picks) --------
    "bot_dave": {
        1: [
            BotPick(401772510, _UNDER, False),
            BotPick(401772714, _OVER, False),
            BotPick(401772829, _DOG, False),
            BotPick(401772830, _FAV, False),
            BotPick(401772719, _UNDER, True),  # mortal lock
        ],
        2: [  # PARTIAL roster — only three picks present
            BotPick(401772936, _UNDER, False),
            BotPick(401772725, _OVER, False),
            BotPick(401772724, _FAV, True),  # mortal lock
        ],
        3: [
            BotPick(401772937, _UNDER, False),
            BotPick(401772733, _OVER, False),
            BotPick(401772731, _DOG, False),
            BotPick(401772732, _FAV, False),
            BotPick(401772842, _DOG, True),  # mortal lock
        ],
    },
    # ---- bot_erin: full rosters ---------------------------------------------
    "bot_erin": {
        1: [
            BotPick(401772510, _FAV, False),
            BotPick(401772714, _UNDER, False),
            BotPick(401772830, _DOG, False),
            BotPick(401772829, _OVER, False),
            BotPick(401772719, _DOG, True),  # mortal lock
        ],
        2: [
            BotPick(401772936, _DOG, False),
            BotPick(401772725, _UNDER, False),
            BotPick(401772834, _FAV, False),
            BotPick(401772835, _UNDER, False),
            BotPick(401772724, _OVER, True),  # mortal lock
        ],
        3: [
            BotPick(401772937, _FAV, False),
            BotPick(401772733, _UNDER, False),
            BotPick(401772731, _FAV, False),  # distinct game from the line above
            BotPick(401772732, _DOG, False),
            BotPick(401772839, _OVER, True),  # mortal lock
        ],
    },
}
