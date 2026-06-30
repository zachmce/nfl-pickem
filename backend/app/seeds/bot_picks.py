"""Direct ``Pick``-row persistence of the preordained bot picks (DEMO-BOTS).

In the time-shifted demo, the bots' week 2-3 pick windows are FUTURE-dated under
the shift, so their picks CANNOT go through the windowed
:func:`app.services.pick_submission.submit_picks` (it would reject a not-yet-open
window). These are preordained SEED data, not live submissions — the same shape
the demo driver/oracle build — so we persist :class:`~app.models.Pick` rows
DIRECTLY, exactly like the oracle side does.

The human user still picks live through the real ``/api/picks`` window alongside
the bots; only the bots' deterministic picks are seeded directly.

Idempotency (PROD-LEAK-GUARD T-sf0-05): before inserting we check whether the bot
already has a Pick for that ``(week_id, pick_type, is_mortal_lock)`` slot and skip
if present, so a re-seed does not violate the partial-unique indexes
(``uq_pick_user_week_type_base`` / ``uq_pick_user_week_mortal_lock``) and leaves
the same row counts.

Design — config-free, operates on the passed-in session (the caller commits the
seed). Reuses :data:`app.seeds.data.bot_picks_2025.BOT_PICKS` and the
User/Game/Week/Pick models — the dataset is never re-encoded here.

> Note: on this machine the interpreter is ``python3``; use ``.venv/bin/python``.
"""

from __future__ import annotations

from typing import Sequence

from sqlmodel import Session, select

from app.models import Game, Pick, PickResult, User, Week
from app.seeds.data.bot_picks_2025 import BOT_PICKS


def seed_bot_picks(
    session: Session,
    *,
    season: int,
    weeks: Sequence[int] = tuple(range(1, 19)),
) -> int:
    """Persist the bots' preordained ``weeks`` picks directly as ``Pick`` rows.

    For each bot ``display_name`` in :data:`BOT_PICKS` (resolved to its seeded
    :class:`~app.models.User`) and each requested week: resolve the ``Week`` row
    id for ``(season, week)``, build a ``{espn_event_id: Game.id}`` index for the
    season, and for each :class:`~app.seeds.data.bot_picks_2025.BotPick` insert a
    :class:`~app.models.Pick` (``result=PENDING``, ``points=0`` defaults — the
    poller/scoring grade them later, exactly like the live path).

    IDEMPOTENT: a pick already present for the bot's ``(week_id, pick_type,
    is_mortal_lock)`` slot is skipped, so a re-run inserts nothing new and never
    trips the partial-unique indexes. Commits once at the end. Returns the number
    of bot ``Pick`` rows present after the run.
    """
    week_set = set(weeks)

    # Resolve bots by display_name -> User (only those present in BOT_PICKS).
    users_by_name = {
        u.display_name: u for u in session.exec(select(User)).all() if u.display_name in BOT_PICKS
    }
    bot_user_ids = [u.id for u in users_by_name.values() if u.id is not None]

    # (season, week) -> week_id.
    week_ids = {w.week: w.id for w in session.exec(select(Week).where(Week.season == season)).all()}

    # espn_event_id -> Game.id for the season.
    event_to_game = {
        g.espn_event_id: g.id for g in session.exec(select(Game).where(Game.season == season)).all()
    }

    for name, user in users_by_name.items():
        assert user.id is not None  # a persisted bot always has an id
        by_week = BOT_PICKS.get(name, {})
        for week, records in by_week.items():
            if week not in week_set:
                continue
            week_id = week_ids.get(week)
            if week_id is None:
                continue

            # Slots this bot already has for the week (idempotency guard).
            existing_slots = {
                (p.pick_type, p.is_mortal_lock)
                for p in session.exec(
                    select(Pick).where(Pick.user_id == user.id, Pick.week_id == week_id)
                ).all()
            }

            for bp in records:
                slot = (bp.pick_type, bp.is_mortal_lock)
                if slot in existing_slots:
                    continue  # already persisted — skip (idempotent re-seed)
                game_id = event_to_game.get(bp.espn_event_id)
                if game_id is None:
                    continue
                session.add(
                    Pick(
                        user_id=user.id,
                        game_id=game_id,
                        week_id=week_id,
                        pick_type=bp.pick_type,
                        is_mortal_lock=bp.is_mortal_lock,
                        result=PickResult.PENDING,
                        points=0,
                    )
                )
                existing_slots.add(slot)

    session.commit()

    if not bot_user_ids:
        return 0
    return len(list(session.exec(select(Pick).where(Pick.user_id.in_(bot_user_ids))).all()))
