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

from app.models import Game, GameStatus, Pick, PickType, Team, User, Week
from app.services.pick_submission import main_picks_complete
from app.services.scoring import GradeOutcome, grade_pick
from app.services.standings import active_season, season_standings, week_results

logger = structlog.get_logger(__name__)


def current_season(session: Session) -> int | None:
    """Resolve the active season as ``max(Game.season)`` (None only on empty DB).

    Delegates to the shared :func:`app.services.standings.active_season` selector
    (the one place the active season is derived): the newest persisted season is
    active, so on a multi-season DB this resolves the larger season; ``None`` only
    when there are ZERO games, in which case the caller skips the social ping
    (lossy is acceptable per the QT-3 design decision). The public name is KEPT —
    ``app/bot/db_bridge.py`` imports and calls it in 8 places.
    """
    return active_season(session)


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
    week_rows = session.exec(select(Week).where(Week.season == season, Week.week.in_(weeks))).all()
    week_id_to_number = {w.id: w.week for w in week_rows if w.id is not None}
    if not week_id_to_number:
        return {}

    # Team abbreviation lookup (one query for the whole season's teams in play).
    abbr_by_team_id = {
        t.id: t.abbreviation for t in session.exec(select(Team)).all() if t.id is not None
    }

    games_by_id = {
        g.id: g
        for g in session.exec(select(Game).where(Game.season == season)).all()
        if g.id is not None
    }

    picks = session.exec(select(Pick).where(Pick.week_id.in_(week_id_to_number.keys()))).all()

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

    standings_results = season_standings(session, season=season)[0].results
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


# --------------------------------------------------------------------------- #
# 260627-vpc — embellished-chat context readers (scope items 1, 2, 3, 5, 6).
#
# The READ side behind the three LLM-embellished pickem-chat events. Each is a
# pure read (no add/commit) returning a plain, DISPLAY-ONLY dict (display_name +
# integers + team abbrs + line numbers + counts) — NEVER user_id, and for the
# OPEN-window roster.complete event NEVER any outstanding name or pick content.
# They REUSE the existing scoring/standings/completion services — no cover /
# over-under / standings / completion math is re-implemented here.
# --------------------------------------------------------------------------- #


def _game_final_side_label(
    pick_type: PickType,
    *,
    favorite_abbr: str | None,
    underdog_abbr: str | None,
    home_abbr: str | None,
    away_abbr: str | None,
) -> str:
    """A concise public side label for a pick on THIS game.

    Reuses the favorite/underdog/over-under convention of
    :func:`app.services.notifications.pick_log_detail` (favorite/underdog name the
    covered team; totals name the matchup) so the embellished line speaks the same
    language as the logger feed. The game is FINAL/public, so naming the team is
    fine. Never returns a raw enum name.
    """
    if pick_type is PickType.FAVORITE_COVER:
        return f"Favorite ({favorite_abbr})" if favorite_abbr else "Favorite"
    if pick_type is PickType.UNDERDOG_COVER:
        return f"Underdog ({underdog_abbr})" if underdog_abbr else "Underdog"
    if pick_type in (PickType.OVER, PickType.UNDER):
        side = "OVER" if pick_type is PickType.OVER else "UNDER"
        if away_abbr and home_abbr:
            return f"{side} {away_abbr}@{home_abbr}"
        return side
    # MISC has no auto-graded side; label it plainly (rarely an impact here).
    return "Misc"


def _not_found_game_final() -> dict:
    """The not-found / unresolvable shape for :func:`get_game_final_context`.

    Lets the caller fall back to a basic event-field fact without branching on a
    missing key. Display-only and inert: no game, no line results, no impacts.
    """
    return {
        "found": False,
        "away": None,
        "home": None,
        "away_score": None,
        "home_score": None,
        "spread_result": None,
        "total_result": None,
        "pick_impacts": [],
    }


def get_game_final_context(
    session: Session,
    season: int,
    week: int,
    *,
    away_abbr: str,
    home_abbr: str,
) -> dict:
    """Display-only context for a ``game.final`` chat line — resolves THE game.

    Resolves the season's :class:`~app.models.Team` rows to an
    ``abbreviation -> team_id`` map, finds the FINAL :class:`~app.models.Game` in
    ``(season, week)`` whose away/home team ids match ``away_abbr``/``home_abbr``,
    and returns a plain dict carrying:

    * ``found`` — ``True`` only when exactly one game resolved with both scores;
    * ``away`` / ``home`` / ``away_score`` / ``home_score`` — echoed from the game;
    * ``spread_result`` — ``{favorite_abbr, spread, did_cover}`` or ``None`` on a
      true pick'em (no gradeable spread side);
    * ``total_result`` — ``{total, went_over}`` or ``None`` when no total posted;
    * ``pick_impacts`` — ``[{display_name, side_label, is_mortal_lock, outcome}]``
      for picks ON THIS game, ``outcome`` taken verbatim from
      :func:`app.services.scoring.grade_pick` (mortal-lock hits/busts surfaced
      first, then base winners/losers, bounded).

    The line results are DERIVED from the SAME comparison the scoring engine uses:
    ``did_cover`` is the favorite-cover result of a reference ``FAVORITE_COVER``
    grade against this game; ``went_over`` is the over result of a reference
    ``OVER`` grade — so no cover/over-under math is hand-rolled here (it reuses
    :func:`app.services.scoring.grade_pick`). The game is FINAL and public, so
    naming pick winners/losers by ``display_name`` is fine. Returns the not-found
    shape (``found`` False, results ``None``, impacts ``[]``) when the game cannot
    be resolved or is ambiguous. Pure read; never raises on unknown inputs.
    """
    # Resolve abbreviation -> team_id (and the inverse) once for this season.
    teams = session.exec(select(Team)).all()
    team_id_by_abbr = {t.abbreviation: t.id for t in teams if t.id is not None}
    abbr_by_team_id = {t.id: t.abbreviation for t in teams if t.id is not None}

    away_id = team_id_by_abbr.get(away_abbr)
    home_id = team_id_by_abbr.get(home_abbr)
    if away_id is None or home_id is None:
        return _not_found_game_final()

    games = session.exec(
        select(Game).where(
            Game.season == season,
            Game.week == week,
            Game.away_team_id == away_id,
            Game.home_team_id == home_id,
        )
    ).all()
    if len(games) != 1:  # unresolved or ambiguous -> caller falls back
        return _not_found_game_final()
    game = games[0]
    if game.status is not GameStatus.FINAL or game.home_score is None or game.away_score is None:
        return _not_found_game_final()

    favorite_abbr = abbr_by_team_id.get(game.favorite_team_id)
    underdog_abbr = abbr_by_team_id.get(game.underdog_team_id)

    # Derive the line results by GRADING reference picks against THIS game — same
    # engine standings/scoring use, so no cover/over-under math is re-implemented.
    spread_result: dict | None = None
    if favorite_abbr is not None and game.spread is not None and game.spread != 0:
        ref_fav = Pick(
            user_id=0,
            game_id=game.id,
            week_id=game.week_id,
            pick_type=PickType.FAVORITE_COVER,
        )
        fav_outcome = grade_pick(game, ref_fav).outcome
        # WIN = favorite covered; LOSS = it did not; PUSH = landed on the number.
        if fav_outcome in (GradeOutcome.WIN, GradeOutcome.LOSS):
            spread_result = {
                "favorite_abbr": favorite_abbr,
                "spread": str(game.spread),
                "did_cover": fav_outcome is GradeOutcome.WIN,
            }

    total_result: dict | None = None
    if game.total is not None:
        ref_over = Pick(
            user_id=0,
            game_id=game.id,
            week_id=game.week_id,
            pick_type=PickType.OVER,
        )
        over_outcome = grade_pick(game, ref_over).outcome
        if over_outcome in (GradeOutcome.WIN, GradeOutcome.LOSS):
            total_result = {
                "total": str(game.total),
                "went_over": over_outcome is GradeOutcome.WIN,
            }

    # Pick impacts: grade THIS game's picks via the scoring engine. Mortal-lock
    # rows first (the dramatic hits/busts), then base rows, bounded.
    picks = session.exec(select(Pick).where(Pick.game_id == game.id)).all()
    user_ids = {p.user_id for p in picks}
    display_name_by_user_id = {
        u.id: u.display_name
        for u in session.exec(select(User).where(User.id.in_(user_ids))).all()
        if u.id is not None
    }
    impacts: list[dict] = []
    for pick in picks:
        display_name = display_name_by_user_id.get(pick.user_id)
        if display_name is None:
            continue
        outcome = grade_pick(game, pick).outcome
        impacts.append(
            {
                "display_name": display_name,
                "side_label": _game_final_side_label(
                    pick.pick_type,
                    favorite_abbr=favorite_abbr,
                    underdog_abbr=underdog_abbr,
                    home_abbr=home_abbr,
                    away_abbr=away_abbr,
                ),
                "is_mortal_lock": pick.is_mortal_lock,
                "outcome": outcome.value,
            }
        )
    # Mortal locks first, then by display_name for a stable, bounded ordering.
    impacts.sort(key=lambda i: (not i["is_mortal_lock"], i["display_name"]))

    return {
        "found": True,
        "away": away_abbr,
        "home": home_abbr,
        "away_score": game.away_score,
        "home_score": game.home_score,
        "spread_result": spread_result,
        "total_result": total_result,
        "pick_impacts": impacts,
    }


def get_roster_complete_context(session: Session, season: int, week: int, *, actor: str) -> dict:
    """Display-only context for a ``roster.complete`` chat line — COUNTS only.

    The roster.complete event fires while the week's pick window is OPEN, so the
    HARD LEAK RULE applies: only the COUNT of outstanding players may cross the
    boundary — NEVER their names, NEVER anyone's pick content. Returns:

    * ``actor`` — the submitting player's display name (echoed in);
    * ``rank`` / ``season_total`` — the actor's public standing from
      :func:`app.services.standings.season_standings` (matched by ``display_name``;
      ``rank`` ``None`` and ``season_total`` ``0`` when the actor is absent from
      standings, e.g. has no graded picks yet);
    * ``completed_count`` — how many players in the pool hold a full standard card
      (four base bet types plus a mortal lock) for this week, via
      :func:`app.services.pick_submission.main_picks_complete`;
    * ``total_players`` — the player pool size;
    * ``outstanding_count`` — ``total_players - completed_count``;
    * ``standings_meaningful`` — ``True`` once ANY game in the season is
      :class:`~app.models.GameStatus` ``FINAL`` (i.e. at least one game is graded);
      gates the downstream season-rank clause so a meaningless 0-point standing is
      not surfaced before any game is graded.

    Player pool choice (decision D-1, user-locked): the pool is now ALL active
    accounts EXCEPT the single protected break-glass admin — it intentionally NO
    LONGER matches the :func:`season_standings` user set. Bots are ``is_active`` and
    SHOULD count; a playing admin is ``is_active`` and is NOT ``is_protected`` so is
    counted (we do NOT filter on ``is_admin``); only the lone protected account
    (``is_protected``, ``discord_id`` NULL — the non-playing break-glass login) is
    excluded. This makes zero-pick active players count toward ``total_players`` so
    ``outstanding_count`` reflects the real league. Pure read; reuses the existing
    standings + completion services (no completion math re-implemented). Never
    returns outstanding names or pick content.
    """
    standings_results = season_standings(session, season=season)[0].results
    rank: int | None = None
    season_total = 0
    for idx, r in enumerate(standings_results, start=1):
        if r.display_name == actor:
            rank = idx
            season_total = r.season_total
            break

    # Player pool = all active, non-protected accounts (decision D-1). Bots and
    # playing admins are is_active and count; the protected break-glass admin is
    # the only exclusion. Identity comparisons (is_/is_not) avoid the E712 lint.
    pool_user_ids = {
        uid
        for uid in session.exec(
            select(User.id).where(User.is_active.is_(True), User.is_protected.is_(False))
        ).all()
        if uid is not None
    }

    total_players = len(pool_user_ids)
    completed_count = sum(
        1
        for uid in pool_user_ids
        if main_picks_complete(session, user_id=uid, season=season, week=week)
    )
    outstanding_count = total_players - completed_count

    # standings_meaningful gates the season-rank clause downstream: True once any
    # season game is FINAL (mirrors the existence-probe in standings.season_is_complete).
    standings_meaningful = (
        session.exec(
            select(Game.id).where(Game.season == season, Game.status == GameStatus.FINAL).limit(1)
        ).first()
        is not None
    )

    return {
        "actor": actor,
        "rank": rank,
        "season_total": season_total,
        "completed_count": completed_count,
        "total_players": total_players,
        "outstanding_count": outstanding_count,
        "standings_meaningful": standings_meaningful,
    }


def get_leaders_context(session: Session, season: int) -> dict:
    """Display-only context for a ``window.opened`` hype line — the season leaders.

    Reuses :func:`app.services.standings.season_standings` (already ordered by
    ``(-season_total, display_name)``) and reports the top one/two rows:

    * ``leader`` / ``leader_total`` — the top row, or ``None`` / ``0`` for an empty
      season;
    * ``runner_up`` / ``runner_up_total`` — the second row, or ``None`` when only
      one player has picked;
    * ``gap`` — ``leader_total - runner_up_total`` (``None`` when there is no
      runner-up).

    Display_name + integers only. Pure read; never raises on an empty season.
    """
    results = season_standings(session, season=season)[0].results
    if not results:
        return {
            "leader": None,
            "leader_total": 0,
            "runner_up": None,
            "runner_up_total": None,
            "gap": None,
        }

    leader = results[0]
    if len(results) >= 2:
        runner_up = results[1]
        return {
            "leader": leader.display_name,
            "leader_total": leader.season_total,
            "runner_up": runner_up.display_name,
            "runner_up_total": runner_up.season_total,
            "gap": leader.season_total - runner_up.season_total,
        }
    return {
        "leader": leader.display_name,
        "leader_total": leader.season_total,
        "runner_up": None,
        "runner_up_total": None,
        "gap": None,
    }
