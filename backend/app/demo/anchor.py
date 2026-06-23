"""Shared demo anchor/offset derivation — the cross-process determinism crux.

The demo season is positioned by ONE offset that shifts the WHOLE 2025 fixture
rigidly so week-1's earliest kickoff lands a fixed runway ahead of a single
anchor instant. Two distinct processes must agree on that offset:

* the seed process (``app/seeds/demo.py``), which positions the Game/Week rows;
* the worker/beat process, which rebuilds ``Demo2025Source(offset)`` in
  ``app.config.default_scoreboard_source`` to poll the same positioned schedule.

They agree by reading ONE persisted absolute instant (:class:`~app.models.DemoState`,
``demo_started_at``) and feeding it through the SINGLE pure formula
:func:`offset_from_anchor`. Given the same stored anchor and the static fixture,
that formula returns a byte-identical offset in every process — neither process
ever recomputes its own ``now+24h`` (that would desync the two; see
``<threat_model>`` T-sf0-03).

Design — pure core + thin DB shell, mirroring ``app/demo/offset.py``'s purity
discipline: imports only the standard library, ``sqlmodel``,
:class:`app.models.DemoState`, and :func:`app.demo.offset.load_fixture_kickoffs`.
It imports neither ``app.config`` nor ``app.db``; the DB shells operate on a
passed-in session (the caller commits), exactly like ``refresh_games``.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

from sqlmodel import Session, select

from app.demo.offset import load_fixture_kickoffs
from app.models import DemoState

# The runway constant: week-1's earliest positioned kickoff lands this far ahead
# of the anchor instant (~24h gives the operator a pre-season day before the
# season starts unspooling). A single configurable constant so the seed and the
# source seam can never disagree on the buffer.
DEMO_KICKOFF_BUFFER: timedelta = timedelta(hours=24)


def _require_aware(dt: datetime, label: str) -> None:
    """Raise a deliberate, labeled ``ValueError`` if ``dt`` is naive.

    Re-declared locally (mirrors ``offset._require_aware`` / ``demo._require_aware``)
    so a wrong-timezone positioning decision can never be made silently.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware (got a naive datetime)")


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). Re-declared locally (mirrors ``refresh._as_aware``); the normalized copy
    is never persisted back, leaving production-on-Postgres unaffected.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def offset_from_anchor(
    anchor: datetime,
    weeks_kickoffs: Mapping[int, list[datetime]] | None = None,
    *,
    buffer: timedelta = DEMO_KICKOFF_BUFFER,
) -> timedelta:
    """The SINGLE offset formula both processes call (pure, deterministic).

    Returns the ``timedelta`` that, added to every fixture kickoff, shifts the
    whole season rigidly so week-1's EARLIEST kickoff = ``anchor + buffer``::

        offset = (anchor + buffer) - min(weeks_kickoffs[1])

    ``anchor`` must be tz-aware (a naive value raises :class:`ValueError`). The
    ``weeks_kickoffs`` default is the static packaged fixture (via
    :func:`load_fixture_kickoffs`), so given the same stored anchor the result is
    byte-identical across processes — the cross-process determinism guarantee.
    """
    _require_aware(anchor, "anchor")
    if weeks_kickoffs is None:
        weeks_kickoffs = load_fixture_kickoffs()

    week1 = list(weeks_kickoffs.get(1, []))
    if not week1:
        raise ValueError("fixture has no week-1 kickoffs to anchor the demo offset on")
    for ko in week1:
        _require_aware(ko, "week 1 kickoff")

    return (anchor + buffer) - min(week1)


def store_demo_anchor(session: Session, anchor: datetime) -> DemoState:
    """Idempotent single-row upsert of the demo anchor instant (caller commits).

    Looks up the existing :class:`~app.models.DemoState` row and re-stamps its
    ``demo_started_at``; inserts one if absent. So a re-seed re-stamps rather than
    duplicating — there is always at most ONE row. Does NOT commit (the caller
    commits), matching ``refresh_games``' caller-commits contract.
    """
    _require_aware(anchor, "anchor")
    row = session.exec(select(DemoState).order_by(DemoState.id)).first()
    if row is None:
        row = DemoState(demo_started_at=anchor)
    else:
        row.demo_started_at = anchor
    session.add(row)
    return row


def load_demo_anchor(session: Session) -> datetime | None:
    """Read the single persisted demo anchor instant (tz-aware), or ``None``.

    Returns the ``demo_started_at`` of the single :class:`~app.models.DemoState`
    row, normalizing a SQLite-naive read back to UTC (the ``_as_aware`` pattern),
    or ``None`` when the demo has not been seeded.
    """
    row = session.exec(select(DemoState).order_by(DemoState.id)).first()
    if row is None:
        return None
    return _as_aware(row.demo_started_at)
