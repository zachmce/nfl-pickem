"""Admin pick-override service — pick_submission MINUS window/lock, PLUS audit.

This is the admin write path behind ``GET/PUT/DELETE
/api/admin/users/{user_id}/picks`` (QT-1). It lets an admin set / add / change /
clear ANY user's pick for ANY game — past or upcoming — for the small-group
convenience case (someone couldn't submit/fix picks in time). It is deliberately
:mod:`app.services.pick_submission`'s logic with the window/lock gates REMOVED,
plus one :class:`~app.models.PickEditAudit` row per override.

What it KEEPS (locked decision 1 — roster integrity is NOT bypassed):
* the pure :func:`app.services.pick_validation.check_new_pick` predicate, so
  duplicate base type / same-game contradiction / >1 mortal lock / pick'em-spread
  ineligibility are STILL rejected (T-m66-03);
* the same :func:`app.services.pick_submission.violation_to_exception` mapping, so
  rejections surface as the SAME typed 4xx envelope as the user-facing path.

What it SKIPS (locked decision 1 — the only thing bypassed): the pick-window and
per-game-lock gates. This module does NOT import or call the pick_window
predicates at all — an admin write succeeds on a closed-window week and on a
locked/FINAL game.

Caller-owns-commit (mirrors :mod:`app.services.pick_submission`): the service
``session.add(...)`` / ``session.delete(...)`` the pick mutation AND the audit row
in the SAME transaction, but does NOT commit — the router owns the single commit,
so the pick change and its audit row persist atomically (T-m66-05).

``game_was_final`` is recorded as ``game.status == GameStatus.FINAL`` at edit time.
This module does NOT touch scoring (recompute-on-read, so editing a past pick
needs no standings recompute/backfill) and does NOT touch the user-facing
submit/clear paths or the Discord cog. It never writes the vestigial
``Pick.result`` / ``Pick.points`` columns.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python`` for any
> commands.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, select

from app.exceptions import NotFoundError, ValidationError
from app.models import (
    Game,
    GameStatus,
    Pick,
    PickEditAudit,
    PickResult,
    PickType,
)
from app.services.pick_submission import (
    _load_week_games,
    _normalized_game,
    _resolve_week,
    violation_to_exception,
)
from app.services.pick_validation import check_new_pick


def admin_set_pick(
    session: Session,
    *,
    caller_id: int,
    target_user_id: int,
    season: int,
    week: int,
    game_id: int,
    pick_type: PickType,
    is_mortal_lock: bool,
    misc_text: str | None = None,
    now: datetime | None = None,
) -> Pick:
    """Upsert ``target_user_id``'s ``{week, pick_type, lock}`` slot to ``game_id``.

    The admin set/add/change path: bypasses the window/lock gates (there is no
    window/lock decision here) but KEEPS roster integrity via
    :func:`check_new_pick`. Add-missing is first-class — if the slot is empty a new
    row is inserted; otherwise the existing slot row's ``game_id`` is updated
    (mirroring :func:`submit_picks`'s base-slot upsert). Writes ONE
    :class:`PickEditAudit` row (``action="set"``) in the same txn. Caller commits.

    Raises :class:`NotFoundError` for an unknown week (``reason="week_not_found"``)
    or a game not in the week (``reason="game_not_in_week"``), and a typed
    conflict/validation exception (via :func:`violation_to_exception`) for any
    roster-integrity violation — nothing is added in that case.

    ``now`` is accepted for signature parity with the user-facing path but the
    admin path makes NO window/lock decision, so it is unused for gating.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    week_row = _resolve_week(session, season, week)
    assert week_row.id is not None  # a persisted week always has an id

    week_games = _load_week_games(session, season, week)
    games_by_id = {g.id: g for g in week_games}
    norm_by_id = {g.id: _normalized_game(g) for g in week_games}

    if game_id not in games_by_id:
        raise NotFoundError(
            f"Game {game_id} is not part of season {season} week {week}.",
            reason="game_not_in_week",
        )

    # Before-state: the existing row in this EXACT slot (if any).
    before_row = session.exec(
        select(Pick).where(
            Pick.user_id == target_user_id,
            Pick.week_id == week_row.id,
            Pick.pick_type == pick_type,
            Pick.is_mortal_lock == is_mortal_lock,
        )
    ).first()

    # The user's OTHER picks for the week (everything except this exact slot row)
    # form the existing set the candidate is validated against. Excluding the slot
    # row itself lets a same-slot change to a different game pass (it is not a
    # self-conflict — it is a replacement).
    existing = list(
        session.exec(
            select(Pick).where(
                Pick.user_id == target_user_id, Pick.week_id == week_row.id
            )
        ).all()
    )
    if before_row is not None:
        existing = [p for p in existing if p.id != before_row.id]

    # Roster integrity (KEEP) — window/lock are NOT consulted here.
    candidate = Pick(
        user_id=target_user_id,
        game_id=game_id,
        week_id=week_row.id,
        pick_type=pick_type,
        is_mortal_lock=is_mortal_lock,
        misc_text=misc_text,
    )
    decision = check_new_pick(candidate, existing, norm_by_id)
    if not decision.ok:
        v = decision.violations[0]
        raise violation_to_exception(v.code, v.message)

    # Upsert the slot.
    if before_row is not None:
        before_existed = True
        before_pick_type: PickType | None = before_row.pick_type
        before_is_mortal_lock: bool | None = before_row.is_mortal_lock
        before_row.game_id = game_id
        # Carry the (possibly updated) MISC text on a retroactive change.
        before_row.misc_text = misc_text
        before_row.updated_at = now
        session.add(before_row)
        persisted = before_row
    else:
        before_existed = False
        before_pick_type = None
        before_is_mortal_lock = None
        session.add(candidate)
        persisted = candidate

    game_was_final = games_by_id[game_id].status == GameStatus.FINAL

    audit = PickEditAudit(
        admin_user_id=caller_id,
        target_user_id=target_user_id,
        game_id=game_id,
        week_id=week_row.id,
        action="set",
        before_existed=before_existed,
        before_pick_type=before_pick_type,
        before_is_mortal_lock=before_is_mortal_lock,
        after_pick_type=pick_type,
        after_is_mortal_lock=is_mortal_lock,
        game_was_final=game_was_final,
    )
    session.add(audit)

    return persisted


def admin_clear_pick(
    session: Session,
    *,
    caller_id: int,
    target_user_id: int,
    season: int,
    week: int,
    pick_type: PickType,
    is_mortal_lock: bool,
    now: datetime | None = None,
) -> None:
    """Delete ``target_user_id``'s ``{week, pick_type, lock}`` slot (admin path).

    The admin clear path: bypasses the window/lock gates entirely (no window/lock
    decision). Writes ONE :class:`PickEditAudit` row (``action="clear"``) in the
    same txn, then ``session.delete(...)`` the matched row. Caller commits.

    Raises :class:`NotFoundError` for an unknown week (``reason="week_not_found"``)
    or a missing slot (``reason="pick_not_found"``, mirroring
    :func:`clear_pick`'s message shape) — nothing is deleted in that case.

    ``now`` is accepted for signature parity but the admin path makes NO
    window/lock decision, so it is unused for gating.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    week_row = _resolve_week(session, season, week)
    assert week_row.id is not None  # a persisted week always has an id

    matched = session.exec(
        select(Pick).where(
            Pick.user_id == target_user_id,
            Pick.week_id == week_row.id,
            Pick.pick_type == pick_type,
            Pick.is_mortal_lock == is_mortal_lock,
        )
    ).first()
    if matched is None:
        lock_label = " (mortal lock)" if is_mortal_lock else ""
        raise NotFoundError(
            f"No {pick_type.value}{lock_label} pick to clear for season "
            f"{season} week {week}.",
            reason="pick_not_found",
        )

    game = session.get(Game, matched.game_id)
    game_was_final = game is not None and game.status == GameStatus.FINAL

    audit = PickEditAudit(
        admin_user_id=caller_id,
        target_user_id=target_user_id,
        game_id=matched.game_id,
        week_id=week_row.id,
        action="clear",
        before_existed=True,
        before_pick_type=matched.pick_type,
        before_is_mortal_lock=matched.is_mortal_lock,
        after_pick_type=None,
        after_is_mortal_lock=None,
        game_was_final=game_was_final,
    )
    session.add(audit)

    session.delete(matched)
    return None


def admin_grade_misc(
    session: Session,
    *,
    caller_id: int,
    target_user_id: int,
    season: int,
    week: int,
    result: PickResult,
    points: int,
    now: datetime | None = None,
) -> Pick:
    """Grade ``target_user_id``'s MISC pick for ``{season, week}`` — set result/points.

    The admin grading path for the ONE manually-graded type: it locates the
    user's MISC pick for the week and writes the admin-decided ``result`` /
    ``points`` onto it, plus ONE :class:`PickEditAudit` row in the same txn.
    Caller commits.

    DELIBERATE EXCEPTION to this module's "never writes the vestigial
    ``Pick.result`` / ``Pick.points`` columns" rule: for a MISC pick those columns
    are AUTHORITATIVE (``app.services.scoring.grade_pick`` passes them through), so
    this is the single place app code legitimately writes them. The audit row
    records WHO graded WHOSE pick — it reuses the EXISTING PickEditAudit shape (no
    new columns; the grade itself lives on the Pick).

    Grading must DECIDE the pick, so ``result`` must be WIN or LOSS — ``PENDING``
    is rejected with :class:`ValidationError` (``reason="misc_grade_must_decide"``).
    Window/lock are NOT consulted (grading is post-hoc by design).

    Raises :class:`NotFoundError` for an unknown week (``reason="week_not_found"``)
    or an absent MISC pick (``reason="pick_not_found"``).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if result is PickResult.PENDING:
        raise ValidationError(
            "Grading a MISC pick must decide it (WIN or LOSS), not PENDING.",
            reason="misc_grade_must_decide",
        )

    week_row = _resolve_week(session, season, week)
    assert week_row.id is not None  # a persisted week always has an id

    pick = session.exec(
        select(Pick).where(
            Pick.user_id == target_user_id,
            Pick.week_id == week_row.id,
            Pick.pick_type == PickType.MISC,
        )
    ).first()
    if pick is None:
        raise NotFoundError(
            f"No MISC pick to grade for season {season} week {week}.",
            reason="pick_not_found",
        )

    # The ONE place app code legitimately writes Pick.result / Pick.points: for
    # MISC those columns are authoritative (grade_pick passes them through).
    pick.result = result
    pick.points = points
    pick.updated_at = now
    session.add(pick)

    game = session.get(Game, pick.game_id)
    game_was_final = game is not None and game.status == GameStatus.FINAL

    audit = PickEditAudit(
        admin_user_id=caller_id,
        target_user_id=target_user_id,
        game_id=pick.game_id,
        week_id=week_row.id,
        action="set",
        before_existed=True,
        before_pick_type=PickType.MISC,
        before_is_mortal_lock=pick.is_mortal_lock,
        after_pick_type=PickType.MISC,
        after_is_mortal_lock=pick.is_mortal_lock,
        game_was_final=game_was_final,
    )
    session.add(audit)

    return pick
