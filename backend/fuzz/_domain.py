"""Shared fuzz-input builders for the pure backend services.

The pure functions in ``app.services.{scoring,pick_validation,pick_window}`` read
attributes off ``Game`` / ``Pick`` and never ``isinstance``-check them, so inputs
can be lightweight ``types.SimpleNamespace`` duck-types — no DB session or SQLModel
instance is needed. The builders mirror the **real column types** (nullable ints,
``Decimal`` spreads/totals on a half-point grid, tz-aware/naive/absent datetimes)
so the fuzzer exercises genuine domain logic rather than type errors the DB layer
would never allow through. The real enums are imported (the functions do identity
checks like ``pick.pick_type is PickType.MISC``).

Determinism: no clock is ever read — datetimes are derived from a fixed epoch so a
crashing input reproduces exactly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from app.models import GameStatus, PickResult, PickType

_STATUSES = list(GameStatus)
_PICK_TYPES = list(PickType)
_RESULTS = list(PickResult)

# Fixed tz-aware anchor — window math is all relative deltas off this.
_EPOCH = datetime(2025, 9, 1, tzinfo=timezone.utc)


def _opt_int(fdp, lo: int, hi: int):
    """An int in ``[lo, hi]``, or ``None`` ~1/5 of the time (nullable int column)."""
    if fdp.ConsumeIntInRange(0, 4) == 0:
        return None
    return fdp.ConsumeIntInRange(lo, hi)


def _opt_decimal(fdp):
    """A ``Decimal`` on a half-point grid, or ``None`` (nullable spread/total).

    Dividing an int by 2 lands values on ``.0`` and ``.5`` lines so the fuzzer
    reaches the exact-equality (PUSH / cover) boundaries, not just strict in/out.
    """
    if fdp.ConsumeIntInRange(0, 4) == 0:
        return None
    return Decimal(fdp.ConsumeIntInRange(0, 120)) / 2


def opt_datetime(fdp):
    """A tz-aware datetime, a NAIVE one, or ``None``.

    The naive branch deliberately exercises the ``_require_aware`` ``ValueError``
    guards in pick_window; ``None`` mirrors an unscheduled kickoff.
    """
    sel = fdp.ConsumeIntInRange(0, 3)
    if sel == 0:
        return None
    dt = _EPOCH + timedelta(minutes=fdp.ConsumeIntInRange(-100_000, 100_000))
    if sel == 1:
        return dt.replace(tzinfo=None)  # naive → deliberate ValueError path
    return dt


def build_game(fdp, *, game_id: int | None = None):
    """A duck-typed ``Game`` mirroring the columns the pure functions read."""
    gid = game_id if game_id is not None else fdp.ConsumeIntInRange(1, 50)
    return SimpleNamespace(
        id=gid,
        status=_STATUSES[fdp.ConsumeIntInRange(0, len(_STATUSES) - 1)],
        home_score=_opt_int(fdp, 0, 80),
        away_score=_opt_int(fdp, 0, 80),
        spread=_opt_decimal(fdp),
        total=_opt_decimal(fdp),
        home_team_id=fdp.ConsumeIntInRange(1, 32),
        away_team_id=fdp.ConsumeIntInRange(1, 32),
        favorite_team_id=_opt_int(fdp, 1, 32),
        underdog_team_id=_opt_int(fdp, 1, 32),
        kickoff_at=opt_datetime(fdp),
    )


def build_pick(fdp, *, game_id: int | None = None):
    """A duck-typed ``Pick`` mirroring the columns the pure functions read."""
    gid = game_id if game_id is not None else fdp.ConsumeIntInRange(1, 50)
    return SimpleNamespace(
        game_id=gid,
        pick_type=_PICK_TYPES[fdp.ConsumeIntInRange(0, len(_PICK_TYPES) - 1)],
        is_mortal_lock=fdp.ConsumeBool(),
        result=_RESULTS[fdp.ConsumeIntInRange(0, len(_RESULTS) - 1)],
        points=fdp.ConsumeIntInRange(-10, 10),
    )
