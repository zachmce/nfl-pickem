"""User-agnostic, DB-sourced season standings + per-week graded results.

This is the read-side service behind the season-scoreboard HTTP view
(build-order #6): the human-facing counterpart to the pick-submission API. It
computes, from PERSISTED ``Pick`` rows, (1) cumulative season standings over
**all** users and (2) per-week graded results for a ``{season, week}``.

Why a new service instead of reusing :func:`app.demo.driver.compute_db_standings`
-----------------------------------------------------------------------------

``driver.compute_db_standings`` is deliberately **bot-scoped** — it loads only
the walkthrough bots (``_bot_users``) — because it is the *actual* side of the
demo capstone's ``actual == expected`` proof against
:func:`app.demo.oracle.compute_standings`. The real scoreboard, by contrast,
must show ALL users. Rather than mutate the demo path (which would risk the
integration proof), this module is an **additive extraction**: it computes
all-users standings while reusing the exact same building blocks —
:func:`app.services.scoring.score_week` / :func:`~app.services.scoring.grade_pick`
for the math and :class:`app.demo.oracle.Standings` /
:class:`~app.demo.oracle.BotSeasonResult` for the shape and the
``(-season_total, display_name)`` ordering. The import is one-directional (this
service imports the demo module; the demo never imports this service), so there
is no cycle and the demo proof is structurally unaffected.

Purity / side effects
----------------------

This is a **read** service: it reads the passed-in session and writes nothing —
no ``add``, no ``commit``. As in the sibling services it re-attaches UTC to the
naive ``kickoff_at`` that ``DateTime(timezone=True)`` round-trips on SQLite, but
only on in-memory ``Game`` copies that are never persisted (mirrors
``driver._as_aware`` / ``test_picks_api._aware``); production-on-Postgres is
unaffected.

> Note: on this machine there is no bare ``python`` on ``PATH``; use the venv
> interpreter ``.venv/bin/python`` for any commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.demo.oracle import (
    BotSeasonResult,
    Standings,
    games_by_pk_index,
)
from app.models import Game, GameStatus, Pick, PickType, User, Week
from app.services.pick_window import compute_window
from app.services.scoring import GradeOutcome, grade_pick, score_week


@dataclass(frozen=True)
class WeekResultPick:
    """One graded pick within a user's week, ready for HTTP shaping.

    ``outcome`` is the :class:`~app.services.scoring.GradeOutcome` string value
    and ``points`` is the points it earned (per the scoring table). These come
    straight from :func:`app.services.scoring.grade_pick` — never re-derived.
    """

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool
    outcome: str
    points: int
    # The free-text MISC prediction. Carried ONLY on a REVEALED entry — the
    # week-level privacy gate below omits ALL of an OTHER user's picks while the
    # week's pick window is OPEN, so misc_text is only ever set on the caller's
    # own pick or on any pick once the week's window has CLOSED (the week's first
    # kickoff, after which everyone's picks for the week are revealed). NULL for
    # every non-MISC pick.
    misc_text: str | None = None


@dataclass(frozen=True)
class UserWeekResult:
    """One user's graded picks + weekly score for a single ``{season, week}``.

    ``weekly_score`` equals the sum of ``picks``' points and equals what
    :func:`app.services.scoring.score_week` returns for the same picks — the
    score is taken from ``score_week`` directly, the per-pick points from
    ``grade_pick``, so they are consistent by construction. ``user_id`` is
    deliberately ABSENT (display_name only) matching the read-side privacy
    posture of ``PickRead``. ``discord_id`` + ``discord_avatar_hash`` carry the
    avatar identity (both ``None`` when the user has no Discord avatar) so the
    frontend can build a CDN avatar URL — ``user_id`` stays omitted.
    """

    display_name: str
    weekly_score: int
    picks: tuple[WeekResultPick, ...]
    discord_id: int | None = None
    discord_avatar_hash: str | None = None


def active_season(session: Session) -> int | None:
    """Resolve the active season as ``max(Game.season)`` over the persisted games.

    The newest persisted season is the active one — deterministic, with NO clock
    or game-status dependency. Returns:

    - ``max(Game.season)`` over the distinct persisted seasons (the lone season on
      a single-season DB; the larger on a multi-season DB, e.g. ``2025`` for
      ``{2024, 2025}``);
    - ``None`` only when there are ZERO ``Game`` rows (the empty-DB guard).

    This is the ONE shared active-season selector (spec:
    ``.planning/notes/active-season-model.md``). The three call sites that used
    to derive the active season independently (and disagree) all delegate here.

    ``session.exec(select(<single column>))`` yields scalar ints here (not Row
    tuples), so iterate the scalars directly (do NOT ``for (s,) in ...``, which
    raises "cannot unpack non-iterable int object").
    """
    seasons = set(session.exec(select(Game.season).distinct()).all())
    return max(seasons) if seasons else None


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). Re-declared locally (mirrors ``driver._as_aware`` /
    ``pick_submission._as_aware``) rather than importing a private helper. The
    normalized copy is never persisted, leaving production-on-Postgres
    unaffected.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _season_games_by_pk(session: Session, *, season: int) -> dict[int, Game]:
    """Load the season's games once and build the PK index ``score_week`` wants.

    Normalizes each in-memory ``Game``'s ``kickoff_at`` to UTC (sqlite tz
    round-trip); the copies are never committed.
    """
    games = list(session.exec(select(Game).where(Game.season == season)).all())
    for g in games:
        g.kickoff_at = _as_aware(g.kickoff_at)  # in-memory copy only; not committed
    return games_by_pk_index(games)


def _season_week_ids(session: Session, *, season: int) -> dict[int, int]:
    """Map ``week number -> Week.id`` for the season (one query, one place)."""
    return {
        w.week: w.id
        for w in session.exec(select(Week).where(Week.season == season)).all()
        if w.id is not None
    }


@dataclass(frozen=True)
class UserIdentity:
    """Avatar-identity fields for one user, looked up alongside display_name.

    Used to thread ``discord_id`` + ``discord_avatar_hash`` through the read
    paths without mutating the demo oracle ``BotSeasonResult``. ``display_name``
    is unique on the ``User`` model, so standings can join this map by name.
    """

    display_name: str
    discord_id: int | None
    discord_avatar_hash: str | None


def season_standings(session: Session, *, season: int) -> tuple[Standings, dict[str, UserIdentity]]:
    """Cumulative season standings over ALL users with picks in ``season``.

    Derives the user set from the distinct ``user_id`` of ``Pick`` rows in the
    season (same convention as ``driver.compute_db_standings`` derives its set
    from the bot users — only users who actually picked appear). For each such
    user, scores every week they picked via
    :func:`app.services.scoring.score_week` (a week with no picks contributes no
    entry, matching the oracle's "only weeks present" convention) and builds a
    :class:`~app.demo.oracle.BotSeasonResult`. Returns a
    :class:`~app.demo.oracle.Standings` ordered by ``(-season_total,
    display_name)``.

    Returns a ``(Standings, identities_by_display_name)`` 2-tuple: the second
    element maps each row's unique ``display_name`` to its
    :class:`UserIdentity` (``discord_id`` + ``discord_avatar_hash``) so the
    response schema can attach avatar identity per row WITHOUT adding fields to
    the demo oracle ``BotSeasonResult`` (the ``actual == expected`` proof
    dataclass must stay byte-identical). There is exactly one caller
    (:func:`app.api.results.read_standings`).

    Pure read: opens no transaction state of its own, writes nothing.
    """
    games_by_pk = _season_games_by_pk(session, season=season)
    week_id_to_number = {
        week_id: number for number, week_id in _season_week_ids(session, season=season).items()
    }
    season_week_ids = set(week_id_to_number)

    # Every persisted pick in the season, in one query, grouped per user/week.
    # picks_by_user[user_id][week_number] -> list[Pick]
    picks_by_user: dict[int, dict[int, list[Pick]]] = {}
    all_picks = session.exec(select(Pick).where(Pick.week_id.in_(season_week_ids))).all()
    for pick in all_picks:
        week_number = week_id_to_number.get(pick.week_id)
        if week_number is None:  # pick outside this season's weeks
            continue
        picks_by_user.setdefault(pick.user_id, {}).setdefault(week_number, []).append(pick)

    identities = _identities_for(session, user_ids=picks_by_user.keys())

    results: list[BotSeasonResult] = []
    for user_id, weeks in picks_by_user.items():
        weekly_scores: dict[int, int] = {
            week_number: score_week(games_by_pk, week_picks)
            for week_number, week_picks in weeks.items()
        }
        results.append(
            BotSeasonResult(
                display_name=identities[user_id].display_name,
                weekly_scores=weekly_scores,
                season_total=sum(weekly_scores.values()),
            )
        )

    results.sort(key=lambda r: (-r.season_total, r.display_name))
    identities_by_display_name = {
        identity.display_name: identity for identity in identities.values()
    }
    return Standings(results=tuple(results)), identities_by_display_name


def season_is_complete(session: Session, *, season: int) -> bool:
    """Whether every game in ``season`` is FINAL (the season-end state).

    Returns:
    - ``True`` iff the season has at least one ``Game`` row AND no ``Game`` row
      for the season has a status other than :class:`~app.models.GameStatus.FINAL`.
    - ``False`` if ANY game for the season is non-FINAL (SCHEDULED / IN_PROGRESS).
    - ``False`` if the season has ZERO ``Game`` rows (an empty season is not
      "complete").

    Pure read: an efficient existence-based check that never loads full ``Game``
    objects — it reads only the bare existence of (a) any game and (b) any
    non-FINAL game for the season.
    """
    any_game = session.exec(select(Game.id).where(Game.season == season).limit(1)).first()
    if any_game is None:
        return False
    any_non_final = session.exec(
        select(Game.id).where(Game.season == season, Game.status != GameStatus.FINAL).limit(1)
    ).first()
    return any_non_final is None


def week_results(
    session: Session, *, season: int, week: int, caller_user_id: int | None = None
) -> list[UserWeekResult]:
    """Per-user graded results for a single ``{season, week}`` over ALL users.

    Resolves the ``Week`` row, loads that week's persisted picks grouped by
    user, grades each pick via :func:`app.services.scoring.grade_pick` (game
    resolved from the season's normalized PK index), and reports the user's
    weekly score via :func:`app.services.scoring.score_week` (so the score is
    never re-derived from the per-pick points). Users with no picks that week
    are omitted. Ordered by ``(-weekly_score, display_name)``.

    Pick-privacy gate (information-disclosure mitigation)
    -----------------------------------------------------

    A single WEEK-LEVEL boundary decides visibility: the week's pick window
    closes at the week's earliest kickoff
    (``compute_window(week_games).close_at``). While the window is OPEN
    (``now < close_at``), revealing OTHER users' picks would leak what everyone
    picked before picking is frozen (a copy/counter leak); once it has CLOSED
    (``now >= close_at``) all picking for the week is frozen for everyone, so
    every user's picks for the week are revealed — including picks on games
    later in the week that have NOT yet kicked off. This gate omits the redacted
    entries SERVER-SIDE (frontend hiding alone still ships the data over the
    wire). The single ``picks_locked`` boolean is computed once for the week,
    not per game:

    * the caller (``caller_user_id``) always sees ALL of their OWN picks,
      whether the window is open or closed;
    * once the week's window has CLOSED, ALL users' picks for the week are
      included (a later not-yet-kicked-off game is no exception);
    * while the window is OPEN, ALL of every OTHER user's picks are OMITTED.

    ``now`` is read once from the real clock (``datetime.now(timezone.utc)``),
    mirroring the real-clock-vs-persisted-kickoffs posture of
    :func:`app.api.current_week.read_current_week` and
    :mod:`app.services.pick_submission` — there is no IS_DEMO_DATA branch; the
    demo time-shift lives in the persisted kickoffs. ``caller_user_id=None``
    (the default) gates ALL users with no caller bypass — for callers that have
    no identity.

    Crucially, ``weekly_score`` is ALWAYS computed over the user's FULL
    persisted picks (not the gated subset), so redaction never changes a score:
    not-yet-locked games grade to ``UNGRADEABLE``/0 and reveal nothing. Users
    whose every pick entry is redacted still appear (with their whole
    ``weekly_score`` and an empty ``picks`` tuple), preserving the
    ``(-weekly_score, display_name)`` ordering.

    An empty season/week simply yields an empty list — a 404-style miss is NOT
    raised here (this is a pure read).
    """
    week_id = _season_week_ids(session, season=season).get(week)
    if week_id is None:
        return []

    now = datetime.now(timezone.utc)
    games_by_pk = _season_games_by_pk(session, season=season)

    # Single WEEK-LEVEL visibility boundary: the week's pick window closes at the
    # week's earliest kickoff. Reuse compute_window (do NOT re-implement the
    # min-kickoff math); only close_at matters for visibility, so no prev-week
    # arg is passed (open_at is irrelevant here). week_games is derived from the
    # already tz-normalized PK index so compute_window's tz-awareness guard is
    # satisfied.
    week_games = [g for g in games_by_pk.values() if g.week_id == week_id]
    picks_locked = now >= compute_window(week_games).close_at

    picks_by_user: dict[int, list[Pick]] = {}
    for pick in session.exec(select(Pick).where(Pick.week_id == week_id)).all():
        picks_by_user.setdefault(pick.user_id, []).append(pick)

    identities = _identities_for(session, user_ids=picks_by_user.keys())

    results: list[UserWeekResult] = []
    for user_id, picks in picks_by_user.items():
        is_caller = user_id == caller_user_id
        graded: list[WeekResultPick] = []
        for pick in picks:
            game = games_by_pk[pick.game_id]
            # Week-level privacy gate: while the week's window is OPEN, omit ALL
            # of an OTHER user's picks; once it has CLOSED, NO entries are skipped
            # for any user. The caller always sees their own.
            if not is_caller and not picks_locked:
                continue
            decision = grade_pick(game, pick)
            graded.append(
                WeekResultPick(
                    game_id=pick.game_id,
                    pick_type=pick.pick_type,
                    is_mortal_lock=pick.is_mortal_lock,
                    outcome=GradeOutcome(decision.outcome).value,
                    points=decision.points,
                    # Only reached for a revealed entry (the gate above already
                    # omitted every other-user pick while the window was open),
                    # so this is safe to carry.
                    misc_text=pick.misc_text,
                )
            )
        identity = identities[user_id]
        results.append(
            UserWeekResult(
                display_name=identity.display_name,
                # Score over the user's FULL persisted picks (NOT the gated
                # subset) so redaction never changes the score — taken from the
                # scorer directly so nothing is re-derived.
                weekly_score=score_week(games_by_pk, picks),
                picks=tuple(graded),
                discord_id=identity.discord_id,
                discord_avatar_hash=identity.discord_avatar_hash,
            )
        )

    results.sort(key=lambda r: (-r.weekly_score, r.display_name))
    return results


def _identities_for(session: Session, *, user_ids) -> dict[int, UserIdentity]:
    """Look up ``{User.id: UserIdentity}`` for the given user ids (one query).

    Reads ``display_name`` plus the avatar-identity attributes (``discord_id``,
    ``discord_avatar_hash``) off the already-loaded ``User`` rows — the same
    single ``select(User).where(User.id.in_(ids))`` as before, just reading the
    extra columns. Feeds both ``season_standings`` and ``week_results``.
    """
    ids = set(user_ids)
    if not ids:
        return {}
    return {
        u.id: UserIdentity(
            display_name=u.display_name,
            discord_id=u.discord_id,
            discord_avatar_hash=u.discord_avatar_hash,
        )
        for u in session.exec(select(User).where(User.id.in_(ids))).all()
        if u.id is not None
    }
