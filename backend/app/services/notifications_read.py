"""Read-only history service feeding the pickem-chat pattern scanner (260627-nef).

This is the READ side that turns persisted ``Pick`` rows into the plain pick-key
dicts that :func:`app.services.pick_patterns.scan_streak` consumes. It mirrors
the posture of :mod:`app.services.standings`: it takes an EXISTING ``Session``,
reads only, and writes NOTHING (no ``add``, no ``commit`` — the caller owns the
session lifecycle). It imports no ``discord`` and no ``httpx``.

Information-disclosure boundary (T-nef-01)
------------------------------------------
Results are keyed by ``display_name`` ONLY — never ``user_id`` — and each key
carries only ``{week, team_abbr, side}``. No ORM instance escapes any function;
only plain dicts/lists cross the boundary. Combined with the caller firing this
ONLY on ``window.closed`` (T-nef-02), no open-window pick is ever read.

Pick-key derivation (the team-keyed expansion)
----------------------------------------------
Each base, auto-gradeable pick becomes one or more ``(team_abbr, side)`` keys,
where ``team_abbr`` is ALWAYS a real team's abbreviation:

* ``FAVORITE_COVER`` -> ONE key ``("FAVORITE", favorite_team.abbreviation)``
* ``UNDERDOG_COVER`` -> ONE key ``("UNDERDOG", underdog_team.abbreviation)``
* ``OVER``           -> TWO keys ``("OVER", favorite_abbr)`` AND ``("OVER", underdog_abbr)``
* ``UNDER``          -> TWO keys ``("UNDER", favorite_abbr)`` AND ``("UNDER", underdog_abbr)``

``MISC`` picks and mortal-lock picks are SKIPPED (only base, auto-gradeable
picks streak). A totals pick keys on both teams in its game so a totals streak
survives a changing opponent.
"""

from __future__ import annotations

import structlog
from sqlmodel import Session, select

from app.models import Game, Pick, PickType, Team, User, Week
from app.services.standings import season_standings, week_results

logger = structlog.get_logger(__name__)


def current_season(session: Session) -> int | None:
    """Resolve the single active season from the distinct ``Game.season`` values.

    Mirrors :func:`app.tasks._active_refresh_season` exactly: if exactly one
    season is present, that is the active season; an ambiguous (multi-season) or
    empty db yields ``None`` and the caller skips the social ping (lossy is
    acceptable per the QT-3 design decision).

    ``session.exec(select(<single column>))`` yields scalar ints here (not Row
    tuples), so iterate the scalars directly.
    """
    seasons = set(session.exec(select(Game.season).distinct()).all())
    return next(iter(seasons)) if len(seasons) == 1 else None


def _pick_keys_for_weeks(
    session: Session, *, season: int, weeks: list[int]
) -> dict[str, list[dict]]:
    """Build ``{display_name: [pick-key dict, ...]}`` for the given season weeks.

    The single shared query path behind both public builders. Joins ``Pick`` to
    its ``Game`` (for the favorite/underdog team ids) and to ``User`` (for the
    display name), resolves team abbreviations once, and expands each base pick
    into its one-or-two ``(team_abbr, side)`` keys. Skips MISC + mortal-lock and
    any pick whose game is missing favorite/underdog (no line — not streakable).
    Keys results by ``display_name`` (NEVER ``user_id``).
    """
    if not weeks:
        return {}

    # Resolve the week-number -> week.id map for this season's requested weeks.
    week_rows = session.exec(
        select(Week).where(Week.season == season, Week.week.in_(weeks))
    ).all()
    week_id_to_number = {w.id: w.week for w in week_rows if w.id is not None}
    if not week_id_to_number:
        return {}

    # Team abbreviation lookup (one query for the whole season's teams in play).
    abbr_by_team_id = {
        t.id: t.abbreviation
        for t in session.exec(select(Team)).all()
        if t.id is not None
    }

    games_by_id = {
        g.id: g
        for g in session.exec(select(Game).where(Game.season == season)).all()
        if g.id is not None
    }

    picks = session.exec(
        select(Pick).where(Pick.week_id.in_(week_id_to_number.keys()))
    ).all()

    # Resolve display names for the users who actually picked (one query).
    user_ids = {p.user_id for p in picks}
    display_name_by_user_id = {
        u.id: u.display_name
        for u in session.exec(select(User).where(User.id.in_(user_ids))).all()
        if u.id is not None
    }

    result: dict[str, list[dict]] = {}
    for pick in picks:
        # Skip the non-base / non-auto-gradeable picks: MISC + any mortal lock.
        if pick.pick_type is PickType.MISC or pick.is_mortal_lock:
            continue
        week_number = week_id_to_number.get(pick.week_id)
        if week_number is None:
            continue
        game = games_by_id.get(pick.game_id)
        if game is None:
            continue
        keys = _expand_pick_keys(week_number, pick.pick_type, game, abbr_by_team_id)
        if not keys:
            continue
        display_name = display_name_by_user_id.get(pick.user_id)
        if display_name is None:
            continue
        result.setdefault(display_name, []).extend(keys)

    return result


def _expand_pick_keys(
    week: int, pick_type: PickType, game: Game, abbr_by_team_id: dict[int, str]
) -> list[dict]:
    """Expand one base pick into its one-or-two ``{week, team_abbr, side}`` keys.

    A spread pick yields ONE key (the favorite or underdog team); a totals pick
    yields TWO (one per team in the game). Returns ``[]`` when the game lacks the
    favorite/underdog team needed to name a real team (no line snapshotted).
    """
    favorite_abbr = abbr_by_team_id.get(game.favorite_team_id)
    underdog_abbr = abbr_by_team_id.get(game.underdog_team_id)

    if pick_type is PickType.FAVORITE_COVER:
        if favorite_abbr is None:
            return []
        return [{"week": week, "team_abbr": favorite_abbr, "side": "FAVORITE"}]
    if pick_type is PickType.UNDERDOG_COVER:
        if underdog_abbr is None:
            return []
        return [{"week": week, "team_abbr": underdog_abbr, "side": "UNDERDOG"}]
    if pick_type in (PickType.OVER, PickType.UNDER):
        side = pick_type.value  # "OVER" | "UNDER"
        if favorite_abbr is None or underdog_abbr is None:
            return []
        return [
            {"week": week, "team_abbr": favorite_abbr, "side": side},
            {"week": week, "team_abbr": underdog_abbr, "side": side},
        ]
    return []


def get_week_pick_keys(session: Session, season: int, week: int) -> dict[str, list[dict]]:
    """``{display_name: [pick-key dict, ...]}`` for the target week's locked picks.

    Pure read. Only display-only data leaves the function (keyed by display_name;
    keys carry ``{week, team_abbr, side}`` only). Skips MISC + mortal-lock picks.
    """
    return _pick_keys_for_weeks(session, season=season, weeks=[week])


def get_history_pick_keys(
    session: Session, season: int, week: int, weeks_back: int = 6
) -> dict[str, list[dict]]:
    """``{display_name: [pick-key dict across recent weeks, ...]}`` ending at ``week``.

    Covers ``week`` and the prior ``weeks_back - 1`` weeks (clamped at week 1), so
    the scanner can count a consecutive run ending at the target week. Pure read,
    display-only output, skips MISC + mortal-lock.
    """
    lowest = max(1, week - weeks_back + 1)
    weeks = list(range(lowest, week + 1))
    return _pick_keys_for_weeks(session, season=season, weeks=weeks)


def get_recap_context(session: Session, season: int, week: int) -> dict:
    """Display-only ``{week, weekly_scores, season_standings}`` for the recap column.

    The READ side behind the Tier-2 LLM weekly recap (260627-tfb). It reuses the
    EXISTING standings services and re-implements NO scoring/standings math:

    * ``weekly_scores`` = ``[{display_name, weekly_score}, ...]`` from
      :func:`app.services.standings.week_results` (already ordered high->low).
      ``caller_user_id`` is left ``None``: the recap fires only after the week is
      fully FINAL, so the public/post-close shape is what we want and we read only
      the per-user ``weekly_score`` (never an individual pick).
    * ``season_standings`` = ``[{display_name, season_total, rank, gap_to_leader},
      ...]`` from :func:`app.services.standings.season_standings` (already ordered
      by ``(-season_total, display_name)``), with a 1-based dense ``rank`` and
      ``gap_to_leader`` = the leader's ``season_total`` minus this row's total
      (the leader's gap is 0).

    Information-disclosure boundary (T-tfb-01): carries ``display_name`` + integer
    fields ONLY — NEVER ``user_id`` (mirrors the T-nef-01 boundary at the top of
    this module). Pure read: no ``add``/``commit`` — the caller owns the session.
    Discord-free and httpx-free. An empty week/season yields empty lists.
    """
    weekly_scores = [
        {"display_name": r.display_name, "weekly_score": r.weekly_score}
        for r in week_results(session, season=season, week=week, caller_user_id=None)
    ]

    standings_results = season_standings(session, season=season).results
    leader_total = standings_results[0].season_total if standings_results else 0
    standings = [
        {
            "display_name": r.display_name,
            "season_total": r.season_total,
            "rank": idx,
            "gap_to_leader": leader_total - r.season_total,
        }
        for idx, r in enumerate(standings_results, start=1)
    ]

    return {
        "week": week,
        "weekly_scores": weekly_scores,
        "season_standings": standings,
    }
