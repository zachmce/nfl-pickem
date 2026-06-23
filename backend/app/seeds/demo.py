"""Env-gated, idempotent, purgeable demo-season seed (the DEMO data path).

With ``IS_DEMO_DATA=true`` and an empty DB, :func:`seed_demo` positions the real
2025 fixture so week-1's first kickoff lands ``DEMO_KICKOFF_BUFFER`` (~24h) in the
future, persists the SINGLE shared anchor (so the worker/beat process rebuilds the
SAME ``Demo2025Source(offset)``), and seeds the 5 bots' weeks 1-13 picks DIRECTLY
as ``Pick`` rows. The season then unspools in real (shifted) time — the Celery
beat poller flips games SCHEDULED->FINAL on its own as the shifted clock crosses
each kickoff. The human user picks live through the real ``/api/picks`` window.

PROD-LEAK-GUARD (the crux — enforced, not advisory):

* :func:`main` GATES on ``settings.is_demo_data``: when the flag is OFF it prints a
  refusal and returns WITHOUT seeding, so the demo seed can NEVER run in the prod
  path (even though compose calls it unconditionally). When ON it emits a LOUD,
  unmissable startup banner so the demo can never be on silently.
* Demo data stays labeled **season 2025** (the fixture's own season — never
  relabeled); a real season will seed as 2026, physically distinct.
* :func:`seed_demo` is end-to-end IDEMPOTENT (re-seed re-stamps the single anchor,
  the data seeds upsert, the reset+refresh re-positions to the same offset for the
  same ``now``, bot picks skip existing slots — same row counts, no errors).
* :func:`purge_demo` removes the whole demo footprint (FK-safe) so it is cleanly
  removable before go-live.

Design — ``app.config``/``app.db`` are imported LAZILY inside :func:`main` only, so
importing this module never constructs Settings or the Postgres engine.
:func:`seed_demo`/:func:`seed_bot_picks` themselves are config-free and operate on
the passed-in session, so the offline tests drive them against in-memory SQLite.

> Note: on this machine the interpreter is ``python3``; use ``.venv/bin/python``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, delete, select

from app.demo.anchor import (
    DEMO_KICKOFF_BUFFER,
    load_demo_anchor,
    offset_from_anchor,
    store_demo_anchor,
)
from app.models import DemoState, Game, GameStatus, Pick, User, Week
from app.scoreboard.demo import Demo2025Source
from app.seeds.bot_picks import seed_bot_picks
from app.seeds.bots import BOT_ACCOUNTS, seed_bots
from app.seeds.data.bot_picks_2025 import BOT_PICKS
from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.refresh import refresh_games

# The bot display_names that participate in the demo (the keys shared by
# BOT_ACCOUNTS and BOT_PICKS) — used by purge to scope the bot-User deletion.
BOT_NAMES: frozenset[str] = frozenset(name for name, _pw in BOT_ACCOUNTS)


def _season(session: Session) -> int:
    """Resolve the single season present in the seeded Game rows (mirrors driver)."""
    seasons = {g.season for g in session.exec(select(Game)).all()}
    if len(seasons) != 1:
        raise ValueError(
            f"expected exactly one season in the demo DB, found {sorted(seasons)}"
        )
    return next(iter(seasons))


def seed_demo(session: Session, *, now: datetime | None = None) -> dict:
    """Seed the positioned demo season end-to-end on ``session`` (idempotent).

    ``now`` is injected (defaulting to the real UTC clock) so the test can pin it.
    Steps:

    1. Run the EXISTING reference/data seeds in order (idempotent on natural
       keys): :func:`seed_teams` -> :func:`import_fixture_2025` -> :func:`seed_bots`.
       Demo data stays labeled **season 2025** (the fixture's own season).
    2. Capture the anchor = ``now`` and ``store_demo_anchor(session, now)``; compute
       ``offset = offset_from_anchor(now)`` (week-1 first kickoff = now + buffer).
    3. Reset every Game to SCHEDULED with null scores (mirror the driver's setup)
       so ``refresh_games`` has non-FINAL weeks to position, then run
       ``refresh_games(session, Demo2025Source(offset))`` ONCE so the positioned
       ``kickoff_at`` + ``window_closes_at`` land in the Game/Week rows. Commit.
    4. ``seed_bot_picks(session, season=season, weeks=tuple(range(1, 14)))`` to
       persist the bots' weeks 1-13 picks directly. Commit.

    Returns a small summary dict (counts + anchor iso) for the CLI banner.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # (1) Reference/data seeds (idempotent on natural keys).
    seed_teams(session)
    import_fixture_2025(session)
    seed_bots(session)

    # (2) Shared anchor + the single shared offset.
    store_demo_anchor(session, now)
    offset = offset_from_anchor(now)

    season = _season(session)

    # (3) Reset to pre-game state, then position once via the demo source.
    for game in session.exec(select(Game)).all():
        game.status = GameStatus.SCHEDULED
        game.home_score = None
        game.away_score = None
        session.add(game)
    refresh_games(session, Demo2025Source(offset), now=now)
    session.commit()

    # (4) Persist the bots' wk1-13 picks directly.
    bot_picks = seed_bot_picks(session, season=season, weeks=tuple(range(1, 14)))

    weeks = len(session.exec(select(Week).where(Week.season == season)).all())
    games = len(session.exec(select(Game).where(Game.season == season)).all())
    bots = len(BOT_ACCOUNTS)
    week1_first_kickoff = now + DEMO_KICKOFF_BUFFER

    return {
        "season": season,
        "weeks": weeks,
        "games": games,
        "bots": bots,
        "bot_picks": bot_picks,
        "anchor": now.isoformat(),
        "week1_first_kickoff": week1_first_kickoff.isoformat(),
    }


def purge_demo(session: Session) -> None:
    """Delete the whole demo footprint in FK-safe order (purgeable guard).

    Order: Pick -> Game -> Week -> DemoState -> bot Users (by the ``bot_``
    display_name set). Commits once. Leaves the DB free of demo rows so the demo
    can be cleanly removed before go-live.
    """
    session.exec(delete(Pick))
    session.exec(delete(Game))
    session.exec(delete(Week))
    session.exec(delete(DemoState))
    session.exec(delete(User).where(User.display_name.in_(list(BOT_NAMES))))
    session.commit()


def _banner(summary: dict) -> str:
    """Build the loud, unmissable demo-on banner (descriptive prose, not a token)."""
    line = "=" * 72
    return (
        f"\n{line}\n"
        "  WARNING: IS_DEMO_DATA ON — SEEDING FAKE 2025 DEMO SEASON\n"
        "  This is NOT production data. The whole 2025 fixture is time-shifted\n"
        f"  so week-1's first kickoff is ~{int(DEMO_KICKOFF_BUFFER.total_seconds() // 3600)}h out, then unspools in real time.\n"
        f"    anchor (demo_started_at): {summary['anchor']}\n"
        f"    week-1 first kickoff:     {summary['week1_first_kickoff']}\n"
        f"    season={summary['season']} weeks={summary['weeks']} "
        f"games={summary['games']} bots={summary['bots']} "
        f"bot_picks={summary['bot_picks']}\n"
        f"{line}\n"
    )


def main() -> None:
    """CLI entry point — GATED on the demo flag.

    When ``settings.is_demo_data`` is OFF: print a refusal and return WITHOUT
    seeding (the demo seed must never run in the prod path). When ON: run
    ``seed_demo`` and print the loud banner + summary so the demo can never be on
    silently. ``app.config`` / ``app.db`` are imported here (not at module top) so
    importing this module never constructs Settings or the engine.
    """
    from app.config import settings
    from app.db import task_session

    if not settings.is_demo_data:
        print(
            "IS_DEMO_DATA is off — refusing to run the demo seed (prod path). "
            "Set IS_DEMO_DATA=true in .env to enable the demo season."
        )
        return

    with task_session() as session:
        summary = seed_demo(session)
    print(_banner(summary))


if __name__ == "__main__":
    main()
