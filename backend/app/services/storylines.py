"""Deterministic season-storyline computations for the ``week.recap`` chat column
(260703-jun).

This is the "precompute a compact digest, never dump the season" layer sketched in
``.planning/seeds/season-storyline-fact-bundle.md``: the BOT owns every fact and the
LLM only phrases a small, curated bundle. It lets ``week.recap`` chat lines make
cross-week callbacks ("missed her mortal lock three weeks running", "Bob has led
since week 4", "biggest upset of the season so far") WITHOUT sending the model a raw
season history.

Purity (this half of the module)
--------------------------------
Everything ABOVE ``get_season_storylines`` is **pure**: no ``Session``, no
``discord``, no ``httpx`` — only the standard library and typing (mirrors the
``_game_narrative`` purity in :mod:`app.services.notifications_read` and the whole of
:mod:`app.services.scoring`). Every function takes ALREADY-NORMALIZED plain inputs
(ints / strings / small tuples) and returns a display-only :class:`Storyline` (or
``None``). The DB-touching shell that gathers those inputs lives at the bottom of the
file (:func:`get_season_storylines`) and is the ONLY function here that imports a
``Session``.

Display-only boundary (T-jun-01)
--------------------------------
A :class:`Storyline` carries only a ``display_name``-derived text plus ints/booleans —
NEVER a ``user_id`` (mirrors the T-tfb-01 / T-nef-01 posture). Only plain values cross
the boundary.

Freshness (stateless, T-jun freshness rule)
-------------------------------------------
"Fresh" = the storyline changed state at week ``W`` — i.e. its most recent constituent
event is week ``W`` (a lock streak that extended this week, a lead that just flipped, a
superlative set this week, a form window ending this week). Because a "no lock result"
week is transparent (see below), a storyline whose defining event is week ``W`` is
exactly the storyline whose ``W``-vs-``W-1`` recompute differs — so this encodes the
stateless W/W-1 diff without a tracking table. Freshness is PREFERRED, not required.

Thresholds (Claude's discretion per CONTEXT.md — small, documented defaults)
----------------------------------------------------------------------------
* mortal-lock streak: a run of >= ``_MIN_STREAK`` (2) consecutive missed-or-hit locks.
* leader tenure: reported only when the leader has held for >= 2 weeks OR the lead
  just flipped this week (otherwise a trivial "leads since this week" is noise).
* hot/cold form: a window of ``_FORM_WINDOW`` (3) weeks, reported only at the extremes
  (0 base-pick wins = cold, or a perfect window = hot) so it stays DISTINCT from the
  mortal-lock streak (it reads the overall base slate, not the lock slot).
* superlative: the single highest-magnitude candidate (biggest upset / highest weekly
  score), ties broken toward the more recent week then the label for determinism.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import structlog
from sqlmodel import Session, select

from app.models import Game, GameStatus, PickType, Team
from app.services.scoring import GradeOutcome
from app.services.standings import week_results

logger = structlog.get_logger(__name__)

# ---- thresholds (documented small defaults; Claude's discretion per CONTEXT.md) ---- #
_MIN_STREAK = 2
_FORM_WINDOW = 3

# Normalized mortal-lock results. A base-pick WIN -> "hit", a LOSS -> "miss"; every
# other grade (PUSH / INELIGIBLE / UNGRADEABLE / no lock that week) -> "none", which is
# TRANSPARENT: it neither extends nor breaks a streak (key_facts: "treat as no lock
# result that week").
LOCK_HIT = "hit"
LOCK_MISS = "miss"
LOCK_NONE = "none"


@dataclass(frozen=True)
class Storyline:
    """One display-only storyline tag ready to hand the recap LLM as DATA.

    * ``kind`` — a stable label ("mortal_lock_streak" / "leader_tenure" / "form" /
      "superlative") for selection + tests.
    * ``text`` — a COMPLETE display-only sentence (built from ``display_name`` +
      ints only; never a ``user_id``).
    * ``fresh`` — whether the storyline changed state at week ``W`` (see module doc).
    * ``subject`` — the ``display_name`` the storyline is ABOUT, or ``None`` for a
      league-wide superlative.
    * ``league_wide`` — ``True`` for a league superlative (selection caps these at 1).
    """

    kind: str
    text: str
    fresh: bool
    subject: str | None
    league_wide: bool = False


@dataclass(frozen=True)
class SuperlativeCandidate:
    """A league-wide superlative candidate the shell collects for ranking.

    ``magnitude`` orders notability (bigger = more notable); ``week_set`` is the week
    the superlative was established (drives freshness); ``text`` is the finished
    display-only sentence; ``label`` disambiguates ties deterministically.
    """

    label: str
    text: str
    magnitude: float
    week_set: int


def mortal_lock_streak(
    subject: str, lock_results: Sequence[tuple[int, str]], *, week: int
) -> Storyline | None:
    """A player's trailing run of consecutive missed (or hit) mortal locks.

    ``lock_results`` is that player's ``(week_number, result)`` sequence, ``result`` in
    ``{"hit", "miss", "none"}``. "none" weeks are TRANSPARENT (dropped) — they neither
    extend nor break a streak. Over the remaining graded weeks the trailing run of the
    SAME kind ending at the latest graded week is the streak. Returns ``None`` when the
    run is shorter than :data:`_MIN_STREAK`. ``fresh`` is ``True`` when that latest
    graded week is ``week`` (the streak extended this week). Distinct from hot/cold
    form: this reads ONLY the mortal-lock slot.
    """
    graded = [(wk, r) for wk, r in sorted(lock_results) if r in (LOCK_HIT, LOCK_MISS)]
    if not graded:
        return None

    latest_kind = graded[-1][1]
    run_weeks = []
    for wk, result in reversed(graded):
        if result == latest_kind:
            run_weeks.append(wk)
        else:
            break

    if len(run_weeks) < _MIN_STREAK:
        return None

    length = len(run_weeks)
    latest_week = max(run_weeks)
    fresh = latest_week == week
    verb = "missed" if latest_kind == LOCK_MISS else "hit"
    text = f"{subject} has {verb} their mortal lock {length} weeks running"
    return Storyline(kind="mortal_lock_streak", text=text, fresh=fresh, subject=subject)


def leader_tenure(leader_by_week: Sequence[tuple[int, str]], *, week: int) -> Storyline | None:
    """Season-leader tenure / lead change for the CURRENT leader.

    ``leader_by_week`` is the ordered ``(week_number, leader_display_name)`` sequence of
    the cumulative-through-that-week leader. The current leader is the last entry; their
    tenure is the earliest week of the trailing run in which they were continuously the
    leader ("led since week N"). ``fresh`` is ``True`` when the lead FLIPPED at the most
    recent week (a new leader this week). Returns ``None`` for a trivial tenure (leader
    has held only the latest week and the lead did not just flip) so a one-week "lead"
    is not reported as a storyline.
    """
    seq = sorted(leader_by_week)
    if not seq:
        return None

    latest_week, current = seq[-1]
    since = latest_week
    for wk, name in reversed(seq):
        if name == current:
            since = wk
        else:
            break

    flipped = len(seq) >= 2 and seq[-2][1] != current
    if not flipped and since == latest_week:
        return None  # trivial single-week "lead" — not a storyline

    fresh = flipped
    if flipped:
        text = f"{current} took over the season lead in week {week}"
    else:
        text = f"{current} has led since week {since}"
    return Storyline(kind="leader_tenure", text=text, fresh=fresh, subject=current)


def form_streak(
    subject: str,
    weekly_records: Sequence[tuple[int, int, int]],
    *,
    week: int,
    window: int = _FORM_WINDOW,
) -> Storyline | None:
    """A player's hot/cold OVERALL base-slate form over their last ``window`` weeks.

    ``weekly_records`` is that player's ordered ``(week_number, base_wins, base_total)``
    sequence — ``base_total`` counts only the auto-graded (WIN/LOSS) base picks that
    week, ``base_wins`` the WINs among them. Over the last ``window`` weeks it reports
    only the extremes: 0 base wins (cold) or a clean sweep (hot). Returns ``None``
    otherwise (a middling record is not a storyline) — keeping this CONCEPTUALLY
    DISTINCT from :func:`mortal_lock_streak` (the lock slot). ``fresh`` is ``True`` when
    the window ends at ``week``.
    """
    seq = sorted(weekly_records)
    if len(seq) < window:
        return None

    tail = seq[-window:]
    wins = sum(w for _, w, _ in tail)
    total = sum(t for _, _, t in tail)
    if total == 0:
        return None

    fresh = tail[-1][0] == week
    if wins == 0:
        text = f"{subject} has gone {wins}-for-{total} on base picks over their last {window} weeks (ice cold)"
        return Storyline(kind="form", text=text, fresh=fresh, subject=subject)
    if wins == total:
        text = f"{subject} has gone a perfect {wins}-for-{total} on base picks over their last {window} weeks (red hot)"
        return Storyline(kind="form", text=text, fresh=fresh, subject=subject)
    return None


def season_superlative(
    candidates: Sequence[SuperlativeCandidate], *, week: int
) -> Storyline | None:
    """Pick the single most notable league-wide superlative from ``candidates``.

    Chooses the highest ``magnitude`` candidate (ties broken toward the more recent
    ``week_set`` then ``label`` for determinism). ``fresh`` is ``True`` when that
    superlative was set at ``week``. Returns ``None`` when there are no candidates.
    """
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c.magnitude, c.week_set, c.label))
    fresh = best.week_set == week
    return Storyline(
        kind="superlative", text=best.text, fresh=fresh, subject=None, league_wide=True
    )


def select_storylines(
    storylines: Sequence[Storyline], *, featured_players: Sequence[str], max_total: int = 3
) -> list[Storyline]:
    """Select the recap bundle: featured players' storylines + <=1 league superlative.

    Keeps only storylines ABOUT a featured player (``subject in featured_players``)
    plus AT MOST ONE league-wide superlative, capped at ``max_total`` (~2-3) tags total.
    Freshness-PREFERRED: within each group fresh storylines sort first (a stable sort,
    so ties keep the deterministic input order), but a non-fresh storyline is still
    selected rather than emitting nothing. One superlative slot is reserved when any
    league superlative exists. Pure + deterministic: same input -> same output; no
    ``user_id`` is ever present (a :class:`Storyline` has none).
    """
    featured_set = set(featured_players)
    featured = [s for s in storylines if not s.league_wide and s.subject in featured_set]
    league = [s for s in storylines if s.league_wide]

    def fresh_first(items: list[Storyline]) -> list[Storyline]:
        return sorted(items, key=lambda s: not s.fresh)  # stable: fresh (False) first

    league_pick = fresh_first(league)[0] if league else None
    slots_for_featured = max_total - (1 if league_pick is not None else 0)

    selected = fresh_first(featured)[: max(slots_for_featured, 0)]
    if league_pick is not None:
        selected = [*selected, league_pick]
    return selected[:max_total]


# --------------------------------------------------------------------------- #
# DB shell — the ONE Session-touching function. Gathers the normalized inputs the
# pure core above needs (read-time over FINAL games, reusing week_results / the
# scoring engine — no scoring/standings math re-implemented) and returns the
# selected, DISPLAY-ONLY storyline bundle. Best-effort: any slip returns [] and
# NEVER raises into the recap/notifier loop (T-jun-02).
# --------------------------------------------------------------------------- #


def _biggest_upset(
    session: Session, *, season: int, through_week: int
) -> SuperlativeCandidate | None:
    """The biggest outright upset over the season's FINAL games up to ``through_week``.

    An "upset" is the favorite losing outright (favorite final score < underdog's) —
    the SAME pure score comparison :func:`app.services.notifications_read._game_narrative`
    uses (no cover math is hand-rolled here). Magnitude = the underdog's winning margin;
    ties broken toward the more recent week. Returns ``None`` when no upset occurred.
    """
    abbr_by_team_id = {
        t.id: t.abbreviation for t in session.exec(select(Team)).all() if t.id is not None
    }
    games = session.exec(
        select(Game).where(
            Game.season == season,
            Game.week <= through_week,
            Game.status == GameStatus.FINAL,
        )
    ).all()

    best: SuperlativeCandidate | None = None
    for game in games:
        if game.home_score is None or game.away_score is None or game.favorite_team_id is None:
            continue
        favorite_is_home = game.favorite_team_id == game.home_team_id
        favorite_score = game.home_score if favorite_is_home else game.away_score
        underdog_score = game.away_score if favorite_is_home else game.home_score
        if favorite_score >= underdog_score:
            continue  # favorite did not lose outright -> not an upset
        margin = underdog_score - favorite_score
        favorite_abbr = abbr_by_team_id.get(game.favorite_team_id)
        underdog_abbr = (
            abbr_by_team_id.get(game.underdog_team_id)
            if game.underdog_team_id is not None
            else None
        )
        candidate = SuperlativeCandidate(
            label="biggest upset",
            text=(
                f"the biggest upset so far: {underdog_abbr} stunned "
                f"{favorite_abbr} in Week {game.week}"
            ),
            magnitude=float(margin),
            week_set=game.week,
        )
        if best is None or (candidate.magnitude, candidate.week_set) > (
            best.magnitude,
            best.week_set,
        ):
            best = candidate
    return best


def get_season_storylines(
    session: Session, *, season: int, week: int, featured_players: Sequence[str]
) -> list[dict]:
    """Read-time, best-effort, display-only storyline bundle for ``(season, week)``.

    Gathers the normalized inputs the pure core needs by REUSING
    :func:`app.services.standings.week_results` (which grades every pick via the scoring
    engine) for weeks ``1..week`` — so NO scoring/standings math is re-implemented:

    * per-player mortal-lock result sequences (WIN -> hit, LOSS -> miss, else -> none);
    * per-player weekly base-slate ``(wins, total)`` records for hot/cold form;
    * the cumulative-through-week leader sequence (leader tenure / lead flip);
    * league superlative candidates (highest weekly score + biggest upset).

    Freshness is stateless (encoded as "changed state at ``week``"; see module doc), then
    the pure :func:`select_storylines` picks the featured-player storylines + <=1 league
    superlative, capped ~2-3. Returns a ``list[dict]`` of display-only tags
    (``{kind, text, fresh}``) — never a ``user_id``. The whole body is wrapped so ANY
    error returns ``[]`` (best-effort; never raises into the recap/notifier loop).
    """
    try:
        cumulative: dict[str, int] = {}
        leader_by_week: list[tuple[int, str]] = []
        lock_seqs: dict[str, list[tuple[int, str]]] = {}
        form_recs: dict[str, list[tuple[int, int, int]]] = {}
        weekly_high: list[tuple[int, int, str]] = []  # (score, week, display_name)

        for wk in range(1, week + 1):
            results = week_results(session, season=season, week=wk, caller_user_id=None)
            if not results:
                continue
            for row in results:
                cumulative[row.display_name] = (
                    cumulative.get(row.display_name, 0) + row.weekly_score
                )
                weekly_high.append((row.weekly_score, wk, row.display_name))

                lock_result = LOCK_NONE
                base_wins = 0
                base_total = 0
                for pick in row.picks:
                    if pick.is_mortal_lock:
                        if pick.outcome == GradeOutcome.WIN.value:
                            lock_result = LOCK_HIT
                        elif pick.outcome == GradeOutcome.LOSS.value:
                            lock_result = LOCK_MISS
                    elif pick.pick_type is not PickType.MISC:
                        if pick.outcome == GradeOutcome.WIN.value:
                            base_wins += 1
                            base_total += 1
                        elif pick.outcome == GradeOutcome.LOSS.value:
                            base_total += 1
                lock_seqs.setdefault(row.display_name, []).append((wk, lock_result))
                form_recs.setdefault(row.display_name, []).append((wk, base_wins, base_total))

            leader = min(cumulative.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            leader_by_week.append((wk, leader))

        storylines: list[Storyline] = []
        for name in sorted(lock_seqs):
            streak = mortal_lock_streak(name, lock_seqs[name], week=week)
            if streak is not None:
                storylines.append(streak)
        for name in sorted(form_recs):
            form = form_streak(name, form_recs[name], week=week)
            if form is not None:
                storylines.append(form)
        tenure = leader_tenure(leader_by_week, week=week)
        if tenure is not None:
            storylines.append(tenure)

        candidates: list[SuperlativeCandidate] = []
        if weekly_high:
            top_score, top_week, top_name = max(weekly_high, key=lambda x: (x[0], x[1], x[2]))
            candidates.append(
                SuperlativeCandidate(
                    label="highest weekly score",
                    text=f"the season's highest weekly haul so far: {top_name} scored {top_score} in Week {top_week}",
                    magnitude=float(top_score),
                    week_set=top_week,
                )
            )
        upset = _biggest_upset(session, season=season, through_week=week)
        if upset is not None:
            candidates.append(upset)
        superlative = season_superlative(candidates, week=week)
        if superlative is not None:
            storylines.append(superlative)

        selected = select_storylines(storylines, featured_players=featured_players)
        return [{"kind": s.kind, "text": s.text, "fresh": s.fresh} for s in selected]
    except Exception:  # best-effort: a computation slip degrades to no storylines
        logger.warning("season_storylines_failed", season=season, week=week, exc_info=True)
        return []
