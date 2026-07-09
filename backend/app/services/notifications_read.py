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

from datetime import datetime, timezone
from decimal import Decimal

import structlog
from sqlmodel import Session, select

from app.models import Game, GameStatus, Pick, PickType, Team, User, Week
from app.services.pick_submission import main_picks_complete
from app.services.pick_window import compute_window
from app.services.scoring import GradeOutcome, grade_pick
from app.services.standings import active_season, season_standings, week_results

logger = structlog.get_logger(__name__)


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). Re-declared locally (mirrors ``standings._as_aware`` /
    ``current_week._as_aware``) rather than importing a private helper. The
    normalized copy is never persisted, leaving production-on-Postgres unaffected.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


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

    Information-disclosure boundary (T-tfb-01 / T-jun-01): carries ``display_name`` +
    integer fields ONLY — NEVER ``user_id`` (mirrors the T-nef-01 boundary at the top
    of this module). The ``storylines`` key (260703-jun) is likewise display-only:
    curated, deterministic season-storyline tags (``{kind, text, fresh}``) computed
    read-time over FINAL games about the recap's own featured players (week winner +
    season leader) plus at most one league superlative. Pure read: no ``add``/``commit``
    — the caller owns the session. Discord-free and httpx-free. An empty week/season
    yields empty lists (and an empty ``storylines`` list).
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

    # Featured players are the ones the recap already surfaces (week winner = top weekly
    # score, season leader = top standings). Storylines attach around THEM (+ <=1 league
    # superlative). Best-effort + display-only: a slip yields []. Local import keeps the
    # storyline service's Session/scoring imports off this module's hot import path.
    from app.services.storylines import get_season_storylines

    winner = weekly_scores[0]["display_name"] if weekly_scores else None
    leader = standings[0]["display_name"] if standings else None
    featured_players = [p for p in (winner, leader) if p is not None]
    storylines = get_season_storylines(
        session, season=season, week=week, featured_players=featured_players
    )

    return {
        "week": week,
        "weekly_scores": weekly_scores,
        "season_standings": standings,
        "storylines": storylines,
    }


def _recap_upset_key(candidate: dict):
    """Sort key for the best-call / biggest-bust ranking (upset magnitude).

    Ranks by ``Game.spread`` DESCENDING (the biggest line first — the gutsiest
    underdog win / the worst favorite bust), with a mortal-lock breaking a spread
    tie (a mortal-lock call is amplified), then ``display_name`` for a stable final
    tie-break. Used with :func:`min` (smallest key wins): a larger spread yields a
    more-negative first element, ``not is_mortal_lock`` puts locks (``False``)
    ahead, and ``display_name`` orders the rest alphabetically.
    """
    return (-candidate["_spread"], not candidate["is_mortal_lock"], candidate["display_name"])


def _recap_top_impact(candidates: list[dict]) -> dict | None:
    """Pick the single top upset impact from ``candidates`` (or ``None``).

    Applies :func:`_recap_upset_key` and strips the private ``_spread`` sort helper
    so only the display-only keys (``display_name``, ``team_abbr``, ``side_label``,
    ``spread`` STRING, ``is_mortal_lock``) cross the boundary.
    """
    if not candidates:
        return None
    top = min(candidates, key=_recap_upset_key)
    return {k: v for k, v in top.items() if k != "_spread"}


def get_week_recap_context(session: Session, season: int, week: int) -> dict:
    """Display-only ``{standings, best_call, biggest_bust, mortal_locks}`` for the recap card.

    The READ side behind the marquee ``week.recap`` "closing ceremony" Discord embed
    (260705-kuv). It REUSES the existing scoring/standings services and
    re-implements NO scoring/standings math:

    * ``standings`` — ``[{rank, display_name, season_total, week_delta}, ...]`` built
      by joining :func:`get_recap_context`'s ``season_standings`` rows to its
      ``weekly_scores`` by ``display_name`` (``week_delta`` = that player's
      ``weekly_score`` this week, ``0`` when the player has no weekly entry).
    * ``best_call`` — the UNDERDOG_COVER pick that WON on the FINAL game with the
      largest :attr:`~app.models.Game.spread` (upset magnitude), or ``None``. Ranked
      by spread DESC, then mortal-lock, then ``display_name``.
    * ``biggest_bust`` — the FAVORITE_COVER pick that LOST on the FINAL game with the
      largest ``Game.spread``, or ``None`` (a busted mortal lock is amplified — it
      breaks spread ties). Ranked by spread DESC (mortal-lock tie-break), then
      ``display_name``.
    * ``mortal_locks`` — one ``{display_name, hit, points, side_label}`` row per
      ``is_mortal_lock`` pick on a FINAL game, graded via
      :func:`app.services.scoring.grade_pick` (``hit`` = outcome is ``WIN``); empty
      when nobody used a mortal lock this week.

    Grading is done ONLY through :func:`app.services.scoring.grade_pick` over FINAL
    games (``Pick.result`` is vestigial for non-MISC types), and ``Game.spread`` (the
    frozen positive-magnitude line — NOT a non-existent ``Game.n``) is the upset rank
    key, carried across the boundary as ``str(game.spread)`` exactly like
    :func:`get_game_final_context` does for ``spread_result``.

    Information-disclosure boundary (T-kuv-01 / T-tfb-01 / T-nef-01): carries
    ``display_name`` + integer/boolean fields + team abbreviations + a spread STRING
    ONLY — NEVER a ``user_id``. ``side_label`` reuses :func:`_game_final_side_label`.
    Pure read: no ``add``/``commit`` — the caller owns the session. Discord-free and
    httpx-free. An empty/ambiguous week or season yields the all-empty shape
    ``{standings: [], best_call: None, biggest_bust: None, mortal_locks: []}`` and
    never raises on well-typed inputs.
    """
    # Standings rows: reuse the recap context (do NOT re-query standings) and join
    # season_standings -> weekly_scores by display_name for the per-week delta.
    recap = get_recap_context(session, season, week)
    weekly_by_name = {row["display_name"]: row["weekly_score"] for row in recap["weekly_scores"]}
    standings = [
        {
            "rank": row["rank"],
            "display_name": row["display_name"],
            "season_total": row["season_total"],
            "week_delta": weekly_by_name.get(row["display_name"], 0),
        }
        for row in recap["season_standings"]
    ]

    empty_blocks = {"best_call": None, "biggest_bust": None, "mortal_locks": []}

    # Resolve the week's FINAL games; an ambiguous/empty week means no upset blocks.
    week_row = session.exec(select(Week).where(Week.season == season, Week.week == week)).first()
    if week_row is None or week_row.id is None:
        return {"standings": standings, **empty_blocks}

    final_games = session.exec(
        select(Game).where(
            Game.season == season,
            Game.week == week,
            Game.status == GameStatus.FINAL,
        )
    ).all()
    games_by_id = {g.id: g for g in final_games if g.id is not None}
    if not games_by_id:
        return {"standings": standings, **empty_blocks}

    # Season Team abbreviation map (mirror get_game_final_context's abbr_by_team_id).
    abbr_by_team_id = {
        t.id: t.abbreviation for t in session.exec(select(Team)).all() if t.id is not None
    }

    picks = session.exec(select(Pick).where(Pick.game_id.in_(games_by_id.keys()))).all()
    user_ids = {p.user_id for p in picks}
    name_by_user_id = {
        u.id: u.display_name
        for u in session.exec(select(User).where(User.id.in_(user_ids))).all()
        if u.id is not None
    }

    best_call_candidates: list[dict] = []
    bust_candidates: list[dict] = []
    mortal_locks: list[dict] = []
    for pick in picks:
        game = games_by_id.get(pick.game_id)
        if game is None:
            continue
        display_name = name_by_user_id.get(pick.user_id)
        if display_name is None:
            continue

        favorite_abbr = abbr_by_team_id.get(game.favorite_team_id)
        underdog_abbr = abbr_by_team_id.get(game.underdog_team_id)
        home_abbr = abbr_by_team_id.get(game.home_team_id)
        away_abbr = abbr_by_team_id.get(game.away_team_id)
        side_label = _game_final_side_label(
            pick.pick_type,
            favorite_abbr=favorite_abbr,
            underdog_abbr=underdog_abbr,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
        )
        grade = grade_pick(game, pick)

        if pick.is_mortal_lock:
            mortal_locks.append(
                {
                    "display_name": display_name,
                    "hit": grade.outcome is GradeOutcome.WIN,
                    "points": grade.points,
                    "side_label": side_label,
                }
            )

        # best_call: the gutsiest RIGHT call — an UNDERDOG_COVER win on the biggest
        # line. grade_pick already voids a true pick'em to INELIGIBLE, so a candidate
        # here always has a real positive spread.
        if (
            pick.pick_type is PickType.UNDERDOG_COVER
            and grade.outcome is GradeOutcome.WIN
            and game.spread is not None
        ):
            best_call_candidates.append(
                {
                    "display_name": display_name,
                    "team_abbr": underdog_abbr,
                    "side_label": side_label,
                    "spread": str(game.spread),
                    "is_mortal_lock": pick.is_mortal_lock,
                    "_spread": game.spread,
                }
            )

        # biggest_bust: the worst miss — a FAVORITE_COVER loss on the biggest line.
        if (
            pick.pick_type is PickType.FAVORITE_COVER
            and grade.outcome is GradeOutcome.LOSS
            and game.spread is not None
        ):
            bust_candidates.append(
                {
                    "display_name": display_name,
                    "team_abbr": favorite_abbr,
                    "side_label": side_label,
                    "spread": str(game.spread),
                    "is_mortal_lock": pick.is_mortal_lock,
                    "_spread": game.spread,
                }
            )

    # Stable, display-only board ordering (mortal-lock rows carry no user_id).
    mortal_locks.sort(key=lambda m: m["display_name"])

    return {
        "standings": standings,
        "best_call": _recap_top_impact(best_call_candidates),
        "biggest_bust": _recap_top_impact(bust_candidates),
        "mortal_locks": mortal_locks,
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


# A large gap (points) between the actual favorite margin and the posted spread is
# an expectation swing — the result blew past what the line implied.
_SWING_POINTS = 10


def _empty_narrative() -> dict:
    """The all-False narrative shape (upset / shutout / expectation_swing)."""
    return {"upset": False, "shutout": False, "expectation_swing": False}


def _game_narrative(
    *,
    favorite_abbr: str | None,
    away_score: int | None,
    home_score: int | None,
    spread: Decimal | None,
    favorite_is_home: bool,
) -> dict:
    """Compute the DETERMINISTIC game.final narrative tags from primitives only.

    Pure and db-free (no ORM, no session) so it is unit-testable without a db. All
    tags are DERIVED entirely from the stored final score + posted spread + which
    side was favored — nothing is invented:

    * ``shutout`` — one side was held to 0 while the other scored.
    * ``upset`` — the favorite (``favorite_abbr`` known) lost outright: its final
      score is strictly LESS than the underdog's.
    * ``expectation_swing`` — the spread is a positive magnitude and the actual
      favorite margin (``favorite_score - underdog_score``, negative when the
      favorite lost) differs from the posted spread by at least ``_SWING_POINTS``.

    Guards against missing scores/spread -> the corresponding tag stays False. Best
    effort: returns a plain dict of booleans, never raises on well-typed inputs.
    """
    narrative = _empty_narrative()
    if away_score is None or home_score is None:
        return narrative

    # shutout: one team held scoreless while the other scored.
    if min(away_score, home_score) == 0 and max(away_score, home_score) > 0:
        narrative["shutout"] = True

    if favorite_abbr is not None:
        favorite_score = home_score if favorite_is_home else away_score
        underdog_score = away_score if favorite_is_home else home_score
        # upset: the favorite lost outright.
        if favorite_score < underdog_score:
            narrative["upset"] = True
        # expectation_swing: the actual favorite margin blew past the posted line.
        if spread is not None and spread > 0:
            actual_favorite_margin = favorite_score - underdog_score
            if abs(actual_favorite_margin - float(spread)) >= _SWING_POINTS:
                narrative["expectation_swing"] = True

    return narrative


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
        "narrative": _empty_narrative(),
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

    # Deterministic narrative tags derived ONLY from the stored score + spread +
    # which side was favored. Best-effort: a bad value degrades to all-False so a
    # narrative slip never breaks the (pure, never-raising) builder.
    try:
        narrative = _game_narrative(
            favorite_abbr=favorite_abbr,
            away_score=game.away_score,
            home_score=game.home_score,
            spread=game.spread,
            favorite_is_home=game.favorite_team_id == home_id,
        )
    except Exception:  # pragma: no cover - defensive; helper is pure
        narrative = _empty_narrative()

    return {
        "found": True,
        "away": away_abbr,
        "home": home_abbr,
        "away_score": game.away_score,
        "home_score": game.home_score,
        "spread_result": spread_result,
        "total_result": total_result,
        "pick_impacts": impacts,
        "narrative": narrative,
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


# --------------------------------------------------------------------------- #
# 260709-k5w — inbound @mention Q&A read seams (Path A v1).
#
# The READ side behind the four supported Q&A intents (pick status, standings,
# lines/slate, scores). Each is a pure read (no add/commit) returning a plain,
# DISPLAY-ONLY dict. They REUSE the existing scoring/standings/window/completion
# services — no scoring / standings / window / completion math is re-implemented
# here. STANDINGS reuses get_leaders_context (no new standings reader).
#
# Information-disclosure boundary: the ONLY asker-scoped reader,
# get_pick_status_for_user, resolves the caller's own row by ``discord_id`` and
# NEVER reads another user; every other reader is already-public data (final /
# in-progress scores, posted lines, the leaderboard). None of them carries a
# ``user_id`` across the boundary.
# --------------------------------------------------------------------------- #

# The four base bet slots, named in a stable display order for the pick-status
# "what's left" list (the caller's OWN card, so revealing is fine). Mortal lock is
# the fifth requirement for a complete standard card.
_PICK_SLOT_LABELS: tuple[tuple[PickType, str], ...] = (
    (PickType.UNDERDOG_COVER, "underdog cover"),
    (PickType.FAVORITE_COVER, "favorite cover"),
    (PickType.OVER, "over"),
    (PickType.UNDER, "under"),
)


def resolve_current_week(session: Session, season: int) -> int | None:
    """The current week number for ``season`` — earliest not-yet-final week.

    Mirrors the WEEK-NUMBER selection in
    :func:`app.api.current_week.read_current_week` (only the number, never the
    four-state): the earliest week that still has a non-FINAL game, else the latest
    week when every week is complete. Returns ``None`` when the season has no games.
    The slate / scores / pick-status readers call this when the classifier gave no
    explicit ``week``.
    """
    games = list(session.exec(select(Game).where(Game.season == season)).all())
    if not games:
        return None
    by_week: dict[int, list[Game]] = {}
    for g in games:
        by_week.setdefault(g.week, []).append(g)
    weeks = sorted(by_week)
    incomplete = [wk for wk in weeks if not all(g.status is GameStatus.FINAL for g in by_week[wk])]
    return incomplete[0] if incomplete else weeks[-1]


def get_pick_status_for_user(session: Session, season: int, week: int, *, discord_id: int) -> dict:
    """ASKER-ONLY pick status for the caller identified by ``discord_id``.

    Resolves the caller's :class:`~app.models.User` by ``discord_id`` and returns a
    display-only dict ``{registered, display_name, complete, remaining_labels}``:
    ``complete`` reuses :func:`app.services.pick_submission.main_picks_complete`
    (all four base bet types + a mortal lock), and ``remaining_labels`` names the
    still-unfilled slots of the caller's OWN standard card. Returns
    ``{registered: False}`` when the discord_id has no account. NEVER reads another
    user — there is no parameter to ask for anyone else's picks (leak-safe by
    construction, T-k5w-01).
    """
    user = session.exec(select(User).where(User.discord_id == discord_id)).one_or_none()
    if user is None or user.id is None:
        return {"registered": False}

    week_row = session.exec(
        select(Week).where(Week.season == season, Week.week == week)
    ).one_or_none()

    # Read ONLY this caller's picks for the week (asker-scoped by user_id).
    picks: list[Pick] = []
    if week_row is not None and week_row.id is not None:
        picks = list(
            session.exec(
                select(Pick).where(Pick.user_id == user.id, Pick.week_id == week_row.id)
            ).all()
        )

    complete = False
    if week_row is not None:
        complete = main_picks_complete(session, user_id=user.id, season=season, week=week)

    base_slot_types = {slot for slot, _ in _PICK_SLOT_LABELS}
    present_base = {
        p.pick_type for p in picks if not p.is_mortal_lock and p.pick_type in base_slot_types
    }
    has_mortal_lock = any(p.is_mortal_lock for p in picks)
    remaining_labels = [label for slot, label in _PICK_SLOT_LABELS if slot not in present_base]
    if not has_mortal_lock:
        remaining_labels.append("mortal lock")

    # Whether the week's pick window is still open — so an incomplete card after the
    # window closes is reported as locked-with-gaps, not as a still-actionable to-do.
    # Same real-clock-vs-persisted-kickoffs comparison the rest of the app uses
    # (demo-correct with no demo branch — see app.api.current_week).
    week_games = list(
        session.exec(select(Game).where(Game.season == season, Game.week == week)).all()
    )
    close_at = _slate_close_at(week_games)
    pick_open = close_at is not None and datetime.now(timezone.utc) < close_at

    return {
        "registered": True,
        "display_name": user.display_name,
        "complete": complete,
        "remaining_labels": remaining_labels,
        "pick_open": pick_open,
    }


def _team_ids_for_token(teams: list[Team], token: str) -> set[int]:
    """Resolve a real-team ``token`` (abbreviation or display-name word) to team ids.

    Pure and case-insensitive: matches a team when the upper-cased ``token`` equals
    its abbreviation, equals its full display name, or is one of the display name's
    words (so "CHIEFS" resolves "Kansas City Chiefs"). ``token`` is already a real
    32-team token from the validator, so this only maps it back to id(s).
    """
    needle = token.strip().upper()
    ids: set[int] = set()
    for t in teams:
        if t.id is None:
            continue
        name = t.display_name.upper()
        if needle == t.abbreviation.upper() or needle == name or needle in name.split():
            ids.add(t.id)
    return ids


def _slate_close_at(games: list[Game]) -> datetime | None:
    """The week's pick-window close time (its first kickoff), or ``None``.

    Reuses :func:`app.services.pick_window.compute_window` over tz-normalized
    kickoff copies (never mutating the store rows) so no window math is
    re-implemented; returns ``None`` when no game has a kickoff to close on.
    """
    if not games:
        return None
    # Shallow copies with tz-aware kickoffs so compute_window can run without
    # mutating the store rows (mirrors app.api.current_week._normalized). Only
    # kickoff_at is read by the window math, but the other required fields are
    # copied so the model constructs cleanly.
    aware = [
        Game(
            espn_event_id=g.espn_event_id,
            week_id=g.week_id,
            season=g.season,
            week=g.week,
            home_team_id=g.home_team_id,
            away_team_id=g.away_team_id,
            kickoff_at=_as_aware(g.kickoff_at),
            status=g.status,
        )
        for g in games
    ]
    try:
        return compute_window(aware).close_at
    except ValueError:
        return None


def get_lines_slate(
    session: Session, season: int, week: int, *, team_abbr: str | None = None
) -> dict:
    """Display-only lines/slate for ``{season, week}`` — optionally one team's game.

    Returns ``{week, close_at, games: [{away, home, favorite, spread, total}, ...]}``
    where ``spread``/``total`` are stringified frozen line values (or ``None`` when
    unposted) and ``close_at`` is the week's pick-window close (its first kickoff)
    via :func:`compute_window`. When ``team_abbr`` (a real validator token) is
    given, ``games`` is narrowed to that team's game. Display-only; pure read.
    """
    games = list(session.exec(select(Game).where(Game.season == season, Game.week == week)).all())
    teams = list(session.exec(select(Team)).all())
    abbr_by_team_id = {t.id: t.abbreviation for t in teams if t.id is not None}

    close_at = _slate_close_at(games)
    # Window open/closed for tense-correct "picks close/closed <when>" phrasing.
    pick_open = close_at is not None and datetime.now(timezone.utc) < close_at

    if team_abbr is not None:
        team_ids = _team_ids_for_token(teams, team_abbr)
        games = [g for g in games if g.home_team_id in team_ids or g.away_team_id in team_ids]

    game_dicts = [
        {
            "away": abbr_by_team_id.get(g.away_team_id),
            "home": abbr_by_team_id.get(g.home_team_id),
            "favorite": (
                abbr_by_team_id.get(g.favorite_team_id) if g.favorite_team_id is not None else None
            ),
            "spread": str(g.spread) if g.spread is not None else None,
            "total": str(g.total) if g.total is not None else None,
        }
        for g in games
    ]

    return {"week": week, "close_at": close_at, "pick_open": pick_open, "games": game_dicts}


def get_week_scores(session: Session, season: int, week: int) -> dict:
    """Display-only final + in-progress scores for ``{season, week}``.

    Returns ``{week, games: [{away, home, away_score, home_score, status}, ...]}``
    for the week's games that are FINAL or IN_PROGRESS (SCHEDULED games have no
    score yet and are omitted). Scores are integers; ``status`` is the plain
    :class:`~app.models.GameStatus` value. Display-only (public); pure read.
    """
    games = list(session.exec(select(Game).where(Game.season == season, Game.week == week)).all())
    teams = list(session.exec(select(Team)).all())
    abbr_by_team_id = {t.id: t.abbreviation for t in teams if t.id is not None}

    scored = [
        {
            "away": abbr_by_team_id.get(g.away_team_id),
            "home": abbr_by_team_id.get(g.home_team_id),
            "away_score": g.away_score,
            "home_score": g.home_score,
            "status": g.status.value,
        }
        for g in games
        if g.status in (GameStatus.FINAL, GameStatus.IN_PROGRESS)
    ]

    return {"week": week, "games": scored}


def get_real_team_tokens(session: Session) -> set[str]:
    """The real-team token set for the validator — abbreviations + name tokens.

    For each seeded :class:`~app.models.Team`: the upper-cased abbreviation, the
    upper-cased full display name, and each word of the display name (so "Chiefs"
    resolves as a display-name token). This is the ``known_team_tokens`` the pure
    :func:`app.bot.qa.validate_classification` coerces against — anything not in it
    is a non-real team and becomes ``unknown``. Returns an empty set on an unseeded DB.
    """
    tokens: set[str] = set()
    for t in session.exec(select(Team)).all():
        tokens.add(t.abbreviation.upper())
        name = t.display_name.upper()
        tokens.add(name)
        tokens.update(name.split())
    return tokens
