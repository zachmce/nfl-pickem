"""Preordained bot-picks dataset for the season-walkthrough oracle.

This is a **pure static data module** — no I/O, no DB, no network. It imports only
:class:`app.models.PickType` and the standard library. It encodes, per bot, per
week, the picks each demo bot (see ``app/seeds/bots.py``) makes against the 2025
season fixture (``app/seeds/data/nfl_2025_regular_season.json``).

Authoring guarantees
--------------------

* Games are referenced by their **stable ``espn_event_id``**, never the surrogate
  ``Game.id`` PK (which is assigned at seed time and is not stable across runs).
* Every referenced ``espn_event_id`` was verified to exist in the fixture for the
  stated week and to be a FINAL game **carrying odds**. Post-jt0 (quick task
  ``260626-jt0``) ALL 272 fixture games carry odds (0 with ``odds: null``), so the
  spread pick types ``FAVORITE_COVER`` / ``UNDERDOG_COVER`` are eligible in every
  week — the dataset now covers the full **weeks 1-18** season (it covered only
  weeks 1-13 before, when late-season games were odds-less and thus unpickable).
  The test ``tests/test_bot_picks_dataset.py`` re-asserts both facts (real game in
  the right week) so the dataset can never silently drift from the fixture.
* Every bot/week roster obeys the conflict rules in
  :mod:`app.services.pick_validation`, including the **slot model**: at most one
  *base* (non-mortal-lock) pick per ``pick_type`` per week — one of each of the
  four bet types (PROJECT.md), even across different games. The mortal lock is
  the only allowed same-type duplicate. Plus: no duplicate ``pick_type`` on a
  game, no ``OVER``+``UNDER`` or ``FAVORITE_COVER``+``UNDERDOG_COVER`` on the same
  game, at most one mortal lock, and no spread pick on an odds-less game. The
  dataset test asserts ``validate_roster(...).ok is True`` for **every** roster.
* A **full** roster is the four base types (one each) on four distinct games plus
  one mortal lock on a fifth distinct game (5 records). At least one roster is
  intentionally **partial** (fewer than 5 picks) to exercise partial scoring
  downstream.

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
# Weeks 4-13 were extended for quick task 260623-t65, and weeks 14-18 for quick
# task 260627-j4s (now possible because jt0 backfilled odds onto every game). Each
# week draws the five lowest-event-id fully-gradeable games (FINAL, nonzero spread,
# total, both sides) as its shared anchor set; every bot composes a full
# one-of-each-type + single-mortal-lock roster over those five games via a distinct
# deterministic strategy (so season standings diverge). Anchor sets by week:
#   week  4: 401772632, 401772716, 401772737, 401772738, 401772739
#   week  5: 401772633, 401772743, 401772744, 401772745, 401772746
#   week  6: 401772634, 401772717, 401772748, 401772749, 401772750
#   week  7: 401772635, 401772753, 401772754, 401772755, 401772756
#   week  8: 401772758, 401772759, 401772760, 401772761, 401772762
#   week  9: 401772763, 401772764, 401772765, 401772766, 401772767
#   week 10: 401772630, 401772636, 401772769, 401772770, 401772771
#   week 11: 401772631, 401772774, 401772775, 401772776, 401772777
#   week 12: 401772779, 401772780, 401772781, 401772782, 401772783
#   week 13: 401772621, 401772694, 401772785, 401772786, 401772787
#   week 14: 401772790, 401772791, 401772792, 401772793, 401772794
#   week 15: 401772795, 401772796, 401772797, 401772798, 401772799
#   week 16: 401772612, 401772613, 401772801, 401772802, 401772803
#   week 17: 401772622, 401772710, 401772711, 401772807, 401772808
#   week 18: 401772955, 401772956, 401772957, 401772958, 401772959
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
        # weeks 4-13: alice favors favorites + the FAVORITE mortal lock.
        4: [
            BotPick(401772632, _FAV, False),
            BotPick(401772716, _DOG, False),
            BotPick(401772737, _OVER, False),
            BotPick(401772738, _UNDER, False),
            BotPick(401772739, _FAV, True),  # mortal lock
        ],
        5: [
            BotPick(401772633, _FAV, False),
            BotPick(401772743, _DOG, False),
            BotPick(401772744, _OVER, False),
            BotPick(401772745, _UNDER, False),
            BotPick(401772746, _FAV, True),  # mortal lock
        ],
        6: [
            BotPick(401772634, _FAV, False),
            BotPick(401772717, _DOG, False),
            BotPick(401772748, _OVER, False),
            BotPick(401772749, _UNDER, False),
            BotPick(401772750, _FAV, True),  # mortal lock
        ],
        7: [
            BotPick(401772635, _FAV, False),
            BotPick(401772753, _DOG, False),
            BotPick(401772754, _OVER, False),
            BotPick(401772755, _UNDER, False),
            BotPick(401772756, _FAV, True),  # mortal lock
        ],
        8: [
            BotPick(401772758, _FAV, False),
            BotPick(401772759, _DOG, False),
            BotPick(401772760, _OVER, False),
            BotPick(401772761, _UNDER, False),
            BotPick(401772762, _FAV, True),  # mortal lock
        ],
        9: [
            BotPick(401772763, _FAV, False),
            BotPick(401772764, _DOG, False),
            BotPick(401772765, _OVER, False),
            BotPick(401772766, _UNDER, False),
            BotPick(401772767, _FAV, True),  # mortal lock
        ],
        10: [
            BotPick(401772630, _FAV, False),
            BotPick(401772636, _DOG, False),
            BotPick(401772769, _OVER, False),
            BotPick(401772770, _UNDER, False),
            BotPick(401772771, _FAV, True),  # mortal lock
        ],
        11: [
            BotPick(401772631, _FAV, False),
            BotPick(401772774, _DOG, False),
            BotPick(401772775, _OVER, False),
            BotPick(401772776, _UNDER, False),
            BotPick(401772777, _FAV, True),  # mortal lock
        ],
        12: [
            BotPick(401772779, _FAV, False),
            BotPick(401772780, _DOG, False),
            BotPick(401772781, _OVER, False),
            BotPick(401772782, _UNDER, False),
            BotPick(401772783, _FAV, True),  # mortal lock
        ],
        13: [
            BotPick(401772621, _FAV, False),
            BotPick(401772694, _DOG, False),
            BotPick(401772785, _OVER, False),
            BotPick(401772786, _UNDER, False),
            BotPick(401772787, _FAV, True),  # mortal lock
        ],
        # weeks 14-18: alice keeps favoring favorites + the FAVORITE mortal lock.
        14: [
            BotPick(401772790, _FAV, False),
            BotPick(401772791, _DOG, False),
            BotPick(401772792, _OVER, False),
            BotPick(401772793, _UNDER, False),
            BotPick(401772794, _FAV, True),  # mortal lock
        ],
        15: [
            BotPick(401772795, _FAV, False),
            BotPick(401772796, _DOG, False),
            BotPick(401772797, _OVER, False),
            BotPick(401772798, _UNDER, False),
            BotPick(401772799, _FAV, True),  # mortal lock
        ],
        16: [
            BotPick(401772612, _FAV, False),
            BotPick(401772613, _DOG, False),
            BotPick(401772801, _OVER, False),
            BotPick(401772802, _UNDER, False),
            BotPick(401772803, _FAV, True),  # mortal lock
        ],
        17: [
            BotPick(401772622, _FAV, False),
            BotPick(401772710, _DOG, False),
            BotPick(401772711, _OVER, False),
            BotPick(401772807, _UNDER, False),
            BotPick(401772808, _FAV, True),  # mortal lock
        ],
        18: [
            BotPick(401772955, _FAV, False),
            BotPick(401772956, _DOG, False),
            BotPick(401772957, _OVER, False),
            BotPick(401772958, _UNDER, False),
            BotPick(401772959, _FAV, True),  # mortal lock
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
            BotPick(401772835, _UNDER, False),  # one-of-each base type
            BotPick(401772724, _UNDER, True),  # mortal lock
        ],
        3: [
            BotPick(401772937, _OVER, False),
            BotPick(401772733, _DOG, False),
            BotPick(401772731, _FAV, False),
            BotPick(401772732, _UNDER, False),
            BotPick(401772840, _OVER, True),  # mortal lock
        ],
        # weeks 4-13: bob favors dogs + unders, with the UNDER mortal lock.
        4: [
            BotPick(401772738, _FAV, False),
            BotPick(401772632, _DOG, False),
            BotPick(401772716, _OVER, False),
            BotPick(401772737, _UNDER, False),
            BotPick(401772739, _UNDER, True),  # mortal lock
        ],
        5: [
            BotPick(401772745, _FAV, False),
            BotPick(401772633, _DOG, False),
            BotPick(401772743, _OVER, False),
            BotPick(401772744, _UNDER, False),
            BotPick(401772746, _UNDER, True),  # mortal lock
        ],
        6: [
            BotPick(401772749, _FAV, False),
            BotPick(401772634, _DOG, False),
            BotPick(401772717, _OVER, False),
            BotPick(401772748, _UNDER, False),
            BotPick(401772750, _UNDER, True),  # mortal lock
        ],
        7: [
            BotPick(401772755, _FAV, False),
            BotPick(401772635, _DOG, False),
            BotPick(401772753, _OVER, False),
            BotPick(401772754, _UNDER, False),
            BotPick(401772756, _UNDER, True),  # mortal lock
        ],
        8: [
            BotPick(401772761, _FAV, False),
            BotPick(401772758, _DOG, False),
            BotPick(401772759, _OVER, False),
            BotPick(401772760, _UNDER, False),
            BotPick(401772762, _UNDER, True),  # mortal lock
        ],
        9: [
            BotPick(401772766, _FAV, False),
            BotPick(401772763, _DOG, False),
            BotPick(401772764, _OVER, False),
            BotPick(401772765, _UNDER, False),
            BotPick(401772767, _UNDER, True),  # mortal lock
        ],
        10: [
            BotPick(401772770, _FAV, False),
            BotPick(401772630, _DOG, False),
            BotPick(401772636, _OVER, False),
            BotPick(401772769, _UNDER, False),
            BotPick(401772771, _UNDER, True),  # mortal lock
        ],
        11: [
            BotPick(401772776, _FAV, False),
            BotPick(401772631, _DOG, False),
            BotPick(401772774, _OVER, False),
            BotPick(401772775, _UNDER, False),
            BotPick(401772777, _UNDER, True),  # mortal lock
        ],
        12: [
            BotPick(401772782, _FAV, False),
            BotPick(401772779, _DOG, False),
            BotPick(401772780, _OVER, False),
            BotPick(401772781, _UNDER, False),
            BotPick(401772783, _UNDER, True),  # mortal lock
        ],
        13: [
            BotPick(401772786, _FAV, False),
            BotPick(401772621, _DOG, False),
            BotPick(401772694, _OVER, False),
            BotPick(401772785, _UNDER, False),
            BotPick(401772787, _UNDER, True),  # mortal lock
        ],
        # weeks 14-18: bob keeps favoring dogs + unders, with the UNDER mortal lock.
        14: [
            BotPick(401772793, _FAV, False),
            BotPick(401772790, _DOG, False),
            BotPick(401772791, _OVER, False),
            BotPick(401772792, _UNDER, False),
            BotPick(401772794, _UNDER, True),  # mortal lock
        ],
        15: [
            BotPick(401772798, _FAV, False),
            BotPick(401772795, _DOG, False),
            BotPick(401772796, _OVER, False),
            BotPick(401772797, _UNDER, False),
            BotPick(401772799, _UNDER, True),  # mortal lock
        ],
        16: [
            BotPick(401772802, _FAV, False),
            BotPick(401772612, _DOG, False),
            BotPick(401772613, _OVER, False),
            BotPick(401772801, _UNDER, False),
            BotPick(401772803, _UNDER, True),  # mortal lock
        ],
        17: [
            BotPick(401772807, _FAV, False),
            BotPick(401772622, _DOG, False),
            BotPick(401772710, _OVER, False),
            BotPick(401772711, _UNDER, False),
            BotPick(401772808, _UNDER, True),  # mortal lock
        ],
        18: [
            BotPick(401772958, _FAV, False),
            BotPick(401772955, _DOG, False),
            BotPick(401772956, _OVER, False),
            BotPick(401772957, _UNDER, False),
            BotPick(401772959, _UNDER, True),  # mortal lock
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
        # weeks 4-13: carol mixes all four types, with the OVER mortal lock.
        4: [
            BotPick(401772716, _FAV, False),
            BotPick(401772737, _DOG, False),
            BotPick(401772738, _OVER, False),
            BotPick(401772632, _UNDER, False),
            BotPick(401772739, _OVER, True),  # mortal lock
        ],
        5: [
            BotPick(401772743, _FAV, False),
            BotPick(401772744, _DOG, False),
            BotPick(401772745, _OVER, False),
            BotPick(401772633, _UNDER, False),
            BotPick(401772746, _OVER, True),  # mortal lock
        ],
        6: [
            BotPick(401772717, _FAV, False),
            BotPick(401772748, _DOG, False),
            BotPick(401772749, _OVER, False),
            BotPick(401772634, _UNDER, False),
            BotPick(401772750, _OVER, True),  # mortal lock
        ],
        7: [
            BotPick(401772753, _FAV, False),
            BotPick(401772754, _DOG, False),
            BotPick(401772755, _OVER, False),
            BotPick(401772635, _UNDER, False),
            BotPick(401772756, _OVER, True),  # mortal lock
        ],
        8: [
            BotPick(401772759, _FAV, False),
            BotPick(401772760, _DOG, False),
            BotPick(401772761, _OVER, False),
            BotPick(401772758, _UNDER, False),
            BotPick(401772762, _OVER, True),  # mortal lock
        ],
        9: [
            BotPick(401772764, _FAV, False),
            BotPick(401772765, _DOG, False),
            BotPick(401772766, _OVER, False),
            BotPick(401772763, _UNDER, False),
            BotPick(401772767, _OVER, True),  # mortal lock
        ],
        10: [
            BotPick(401772636, _FAV, False),
            BotPick(401772769, _DOG, False),
            BotPick(401772770, _OVER, False),
            BotPick(401772630, _UNDER, False),
            BotPick(401772771, _OVER, True),  # mortal lock
        ],
        11: [
            BotPick(401772774, _FAV, False),
            BotPick(401772775, _DOG, False),
            BotPick(401772776, _OVER, False),
            BotPick(401772631, _UNDER, False),
            BotPick(401772777, _OVER, True),  # mortal lock
        ],
        12: [
            BotPick(401772780, _FAV, False),
            BotPick(401772781, _DOG, False),
            BotPick(401772782, _OVER, False),
            BotPick(401772779, _UNDER, False),
            BotPick(401772783, _OVER, True),  # mortal lock
        ],
        13: [
            BotPick(401772694, _FAV, False),
            BotPick(401772785, _DOG, False),
            BotPick(401772786, _OVER, False),
            BotPick(401772621, _UNDER, False),
            BotPick(401772787, _OVER, True),  # mortal lock
        ],
        # weeks 14-18: carol keeps mixing all four types, with the OVER mortal lock.
        14: [
            BotPick(401772791, _FAV, False),
            BotPick(401772792, _DOG, False),
            BotPick(401772793, _OVER, False),
            BotPick(401772790, _UNDER, False),
            BotPick(401772794, _OVER, True),  # mortal lock
        ],
        15: [
            BotPick(401772796, _FAV, False),
            BotPick(401772797, _DOG, False),
            BotPick(401772798, _OVER, False),
            BotPick(401772795, _UNDER, False),
            BotPick(401772799, _OVER, True),  # mortal lock
        ],
        16: [
            BotPick(401772613, _FAV, False),
            BotPick(401772801, _DOG, False),
            BotPick(401772802, _OVER, False),
            BotPick(401772612, _UNDER, False),
            BotPick(401772803, _OVER, True),  # mortal lock
        ],
        17: [
            BotPick(401772710, _FAV, False),
            BotPick(401772711, _DOG, False),
            BotPick(401772807, _OVER, False),
            BotPick(401772622, _UNDER, False),
            BotPick(401772808, _OVER, True),  # mortal lock
        ],
        18: [
            BotPick(401772956, _FAV, False),
            BotPick(401772957, _DOG, False),
            BotPick(401772958, _OVER, False),
            BotPick(401772955, _UNDER, False),
            BotPick(401772959, _OVER, True),  # mortal lock
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
        # weeks 4-13: dave plays full rosters with the DOG mortal lock
        # (week 2 stays the canonical PARTIAL roster — only 3 picks).
        4: [
            BotPick(401772737, _FAV, False),
            BotPick(401772738, _DOG, False),
            BotPick(401772632, _OVER, False),
            BotPick(401772716, _UNDER, False),
            BotPick(401772739, _DOG, True),  # mortal lock
        ],
        5: [
            BotPick(401772744, _FAV, False),
            BotPick(401772745, _DOG, False),
            BotPick(401772633, _OVER, False),
            BotPick(401772743, _UNDER, False),
            BotPick(401772746, _DOG, True),  # mortal lock
        ],
        6: [
            BotPick(401772748, _FAV, False),
            BotPick(401772749, _DOG, False),
            BotPick(401772634, _OVER, False),
            BotPick(401772717, _UNDER, False),
            BotPick(401772750, _DOG, True),  # mortal lock
        ],
        7: [
            BotPick(401772754, _FAV, False),
            BotPick(401772755, _DOG, False),
            BotPick(401772635, _OVER, False),
            BotPick(401772753, _UNDER, False),
            BotPick(401772756, _DOG, True),  # mortal lock
        ],
        8: [
            BotPick(401772760, _FAV, False),
            BotPick(401772761, _DOG, False),
            BotPick(401772758, _OVER, False),
            BotPick(401772759, _UNDER, False),
            BotPick(401772762, _DOG, True),  # mortal lock
        ],
        9: [
            BotPick(401772765, _FAV, False),
            BotPick(401772766, _DOG, False),
            BotPick(401772763, _OVER, False),
            BotPick(401772764, _UNDER, False),
            BotPick(401772767, _DOG, True),  # mortal lock
        ],
        10: [
            BotPick(401772769, _FAV, False),
            BotPick(401772770, _DOG, False),
            BotPick(401772630, _OVER, False),
            BotPick(401772636, _UNDER, False),
            BotPick(401772771, _DOG, True),  # mortal lock
        ],
        11: [
            BotPick(401772775, _FAV, False),
            BotPick(401772776, _DOG, False),
            BotPick(401772631, _OVER, False),
            BotPick(401772774, _UNDER, False),
            BotPick(401772777, _DOG, True),  # mortal lock
        ],
        12: [
            BotPick(401772781, _FAV, False),
            BotPick(401772782, _DOG, False),
            BotPick(401772779, _OVER, False),
            BotPick(401772780, _UNDER, False),
            BotPick(401772783, _DOG, True),  # mortal lock
        ],
        13: [
            BotPick(401772785, _FAV, False),
            BotPick(401772786, _DOG, False),
            BotPick(401772621, _OVER, False),
            BotPick(401772694, _UNDER, False),
            BotPick(401772787, _DOG, True),  # mortal lock
        ],
        # weeks 14-18: dave plays full rosters with the DOG mortal lock, except
        # week 16 — a fresh PARTIAL roster (mortal lock dropped, four base picks)
        # to extend the dataset's human-looking partial-coverage character into
        # the late season alongside dave's canonical week-2 partial.
        14: [
            BotPick(401772792, _FAV, False),
            BotPick(401772793, _DOG, False),
            BotPick(401772790, _OVER, False),
            BotPick(401772791, _UNDER, False),
            BotPick(401772794, _DOG, True),  # mortal lock
        ],
        15: [
            BotPick(401772797, _FAV, False),
            BotPick(401772798, _DOG, False),
            BotPick(401772795, _OVER, False),
            BotPick(401772796, _UNDER, False),
            BotPick(401772799, _DOG, True),  # mortal lock
        ],
        16: [  # PARTIAL roster — four base picks, no mortal lock this week
            BotPick(401772801, _FAV, False),
            BotPick(401772802, _DOG, False),
            BotPick(401772612, _OVER, False),
            BotPick(401772613, _UNDER, False),
        ],
        17: [
            BotPick(401772711, _FAV, False),
            BotPick(401772807, _DOG, False),
            BotPick(401772622, _OVER, False),
            BotPick(401772710, _UNDER, False),
            BotPick(401772808, _DOG, True),  # mortal lock
        ],
        18: [
            BotPick(401772957, _FAV, False),
            BotPick(401772958, _DOG, False),
            BotPick(401772955, _OVER, False),
            BotPick(401772956, _UNDER, False),
            BotPick(401772959, _DOG, True),  # mortal lock
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
            BotPick(401772835, _OVER, False),  # one-of-each base type
            BotPick(401772724, _OVER, True),  # mortal lock
        ],
        3: [
            BotPick(401772937, _FAV, False),
            BotPick(401772733, _UNDER, False),
            BotPick(401772731, _OVER, False),  # one-of-each base type
            BotPick(401772732, _DOG, False),
            BotPick(401772839, _OVER, True),  # mortal lock
        ],
        # weeks 4-13: erin leans underdog/total, with a FAVORITE mortal lock on
        # a different slot than alice (distinct outcomes vs the other bots).
        4: [
            BotPick(401772739, _FAV, False),
            BotPick(401772738, _DOG, False),
            BotPick(401772737, _OVER, False),
            BotPick(401772716, _UNDER, False),
            BotPick(401772632, _FAV, True),  # mortal lock
        ],
        5: [
            BotPick(401772746, _FAV, False),
            BotPick(401772745, _DOG, False),
            BotPick(401772744, _OVER, False),
            BotPick(401772743, _UNDER, False),
            BotPick(401772633, _FAV, True),  # mortal lock
        ],
        6: [
            BotPick(401772750, _FAV, False),
            BotPick(401772749, _DOG, False),
            BotPick(401772748, _OVER, False),
            BotPick(401772717, _UNDER, False),
            BotPick(401772634, _FAV, True),  # mortal lock
        ],
        7: [
            BotPick(401772756, _FAV, False),
            BotPick(401772755, _DOG, False),
            BotPick(401772754, _OVER, False),
            BotPick(401772753, _UNDER, False),
            BotPick(401772635, _FAV, True),  # mortal lock
        ],
        8: [
            BotPick(401772762, _FAV, False),
            BotPick(401772761, _DOG, False),
            BotPick(401772760, _OVER, False),
            BotPick(401772759, _UNDER, False),
            BotPick(401772758, _FAV, True),  # mortal lock
        ],
        9: [
            BotPick(401772767, _FAV, False),
            BotPick(401772766, _DOG, False),
            BotPick(401772765, _OVER, False),
            BotPick(401772764, _UNDER, False),
            BotPick(401772763, _FAV, True),  # mortal lock
        ],
        10: [
            BotPick(401772771, _FAV, False),
            BotPick(401772770, _DOG, False),
            BotPick(401772769, _OVER, False),
            BotPick(401772636, _UNDER, False),
            BotPick(401772630, _FAV, True),  # mortal lock
        ],
        11: [
            BotPick(401772777, _FAV, False),
            BotPick(401772776, _DOG, False),
            BotPick(401772775, _OVER, False),
            BotPick(401772774, _UNDER, False),
            BotPick(401772631, _FAV, True),  # mortal lock
        ],
        12: [
            BotPick(401772783, _FAV, False),
            BotPick(401772782, _DOG, False),
            BotPick(401772781, _OVER, False),
            BotPick(401772780, _UNDER, False),
            BotPick(401772779, _FAV, True),  # mortal lock
        ],
        13: [
            BotPick(401772787, _FAV, False),
            BotPick(401772786, _DOG, False),
            BotPick(401772785, _OVER, False),
            BotPick(401772694, _UNDER, False),
            BotPick(401772621, _FAV, True),  # mortal lock
        ],
        # weeks 14-18: erin keeps leaning underdog/total, with a FAVORITE mortal
        # lock on a different slot (the lowest anchor id) than alice's.
        14: [
            BotPick(401772794, _FAV, False),
            BotPick(401772793, _DOG, False),
            BotPick(401772792, _OVER, False),
            BotPick(401772791, _UNDER, False),
            BotPick(401772790, _FAV, True),  # mortal lock
        ],
        15: [
            BotPick(401772799, _FAV, False),
            BotPick(401772798, _DOG, False),
            BotPick(401772797, _OVER, False),
            BotPick(401772796, _UNDER, False),
            BotPick(401772795, _FAV, True),  # mortal lock
        ],
        16: [
            BotPick(401772803, _FAV, False),
            BotPick(401772802, _DOG, False),
            BotPick(401772801, _OVER, False),
            BotPick(401772613, _UNDER, False),
            BotPick(401772612, _FAV, True),  # mortal lock
        ],
        17: [
            BotPick(401772808, _FAV, False),
            BotPick(401772807, _DOG, False),
            BotPick(401772711, _OVER, False),
            BotPick(401772710, _UNDER, False),
            BotPick(401772622, _FAV, True),  # mortal lock
        ],
        18: [
            BotPick(401772959, _FAV, False),
            BotPick(401772958, _DOG, False),
            BotPick(401772957, _OVER, False),
            BotPick(401772956, _UNDER, False),
            BotPick(401772955, _FAV, True),  # mortal lock
        ],
    },
}
