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

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlmodel import Session, col, func, select

from app.exceptions import NotFoundError, ValidationError
from app.models import (
    Game,
    GameStatus,
    Pick,
    PickEditAudit,
    PickResult,
    PickType,
    Team,
    User,
    Week,
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
        now = datetime.now(UTC)

    # Guard a missing/typo'd target user up front: a non-existent target_user_id
    # would otherwise become an FK IntegrityError 500 at commit (the pick/audit
    # add references it). Raise a clean 404 BEFORE any session.add (T-nus-01).
    if session.get(User, target_user_id) is None:
        raise NotFoundError(
            f"No user {target_user_id} to set a pick for.",
            reason="user_not_found",
        )

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
            select(Pick).where(Pick.user_id == target_user_id, Pick.week_id == week_row.id)
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
        now = datetime.now(UTC)

    # Guard a missing/typo'd target user up front: the audit row references it,
    # so a non-existent target_user_id would be an FK IntegrityError 500 at
    # commit. Raise a clean 404 BEFORE any session.add (T-nus-01).
    if session.get(User, target_user_id) is None:
        raise NotFoundError(
            f"No user {target_user_id} to clear a pick for.",
            reason="user_not_found",
        )

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
            f"No {pick_type.value}{lock_label} pick to clear for season {season} week {week}.",
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
        now = datetime.now(UTC)

    if result is PickResult.PENDING:
        raise ValidationError(
            "Grading a MISC pick must decide it (WIN or LOSS), not PENDING.",
            reason="misc_grade_must_decide",
        )

    # Guard a missing/typo'd target user up front: the audit row references it,
    # so a non-existent target_user_id would be an FK IntegrityError 500 at
    # commit. Raise a clean 404 BEFORE any session.add (T-nus-01).
    if session.get(User, target_user_id) is None:
        raise NotFoundError(
            f"No user {target_user_id} to grade a MISC pick for.",
            reason="user_not_found",
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


@dataclass(frozen=True)
class MiscPickRow:
    """One MISC pick for the admin grading panel, with its game for context."""

    pick: Pick
    display_name: str
    game: Game
    away_abbr: str
    home_abbr: str


def default_misc_grading_week(session: Session) -> tuple[int, int] | None:
    """The latest season's most recent week whose games are ALL FINAL.

    Only the latest season is considered, so a new season does not fall back to
    last season's week 18. If no week of that season is complete yet, return its
    first week. ``None`` when there are no weeks at all.
    """
    latest_season = session.exec(select(func.max(Week.season))).one()
    if latest_season is None:
        return None
    games = session.exec(select(Game.week, Game.status).where(Game.season == latest_season)).all()
    weeks: dict[int, bool] = {}
    for week, status in games:
        weeks[week] = weeks.get(week, True) and status is GameStatus.FINAL
    complete = [w for w, done in weeks.items() if done]
    if complete:
        return latest_season, max(complete)
    first = session.exec(select(func.min(Week.week)).where(Week.season == latest_season)).one()
    return latest_season, first if first is not None else 1


def list_week_misc_picks(session: Session, *, season: int, week: int) -> list[MiscPickRow]:
    """Every user's MISC pick for ``{season, week}``, ordered by display name.

    Raises :class:`NotFoundError` (``reason="week_not_found"``) for an unknown week.
    """
    week_row = _resolve_week(session, season, week)
    rows = session.exec(
        select(Pick, User, Game)
        .join(User, col(Pick.user_id) == col(User.id))
        .join(Game, col(Pick.game_id) == col(Game.id))
        .where(Pick.week_id == week_row.id, Pick.pick_type == PickType.MISC)
    ).all()
    team_ids = {t for _, _, g in rows for t in (g.home_team_id, g.away_team_id)}
    abbr = {
        t.id: t.abbreviation
        for t in session.exec(select(Team).where(col(Team.id).in_(team_ids))).all()
    }
    out = [
        MiscPickRow(
            pick=pick,
            display_name=user.display_name,
            game=game,
            away_abbr=abbr.get(game.away_team_id, "?"),
            home_abbr=abbr.get(game.home_team_id, "?"),
        )
        for pick, user, game in rows
    ]
    return sorted(out, key=lambda r: r.display_name.casefold())
