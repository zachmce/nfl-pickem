from sqlmodel import Session

from app.celery_app import celery_app
from app.config import default_scoreboard_source
from app.db import task_session
from app.services.freeze import freeze_week
from app.services.ingest import ingest_season
from app.services.notifications import (
    freeze_week_event,
    game_final_event,
    ingest_season_event,
    publish_event,
    week_recap_event,
    window_closed_event,
    window_opened_event,
)
from app.services.scheduler import ODDS_JOB, SCORES_JOB
from app.services.standings import active_season, season_standings, week_results


@celery_app.task(name="app.tasks.refresh_games")
def refresh_games_task() -> dict:
    """Beat-driven poller: reconcile non-final games + stamp pick windows.

    Thin wrapper that DISPATCHES the scores
    :class:`~app.services.scheduler.PollingJob`: open a non-HTTP
    ``task_session()``, resolve the gated default scoreboard source (the real
    ESPN adapter in prod, the time-shifted Demo2025Source under the demo gate) via
    :func:`app.config.default_scoreboard_source` — the gated resolution stays HERE
    in the thin wrapper, NOT in the source-agnostic scheduler — run the scores
    job's reconcile (which delegates to the unchanged
    :func:`app.services.refresh.refresh_games` core), commit, and return a
    JSON-serializable summary so Celery's json result serializer accepts it.

    Cadence trigger: registered in ``app.celery_app``'s ``beat_schedule`` (built
    from the polling-job registry, so the scores job's ``REFRESH_GAMES_INTERVAL_SECONDS``
    cadence) so beat drives this poller on a timer.
    """
    with task_session() as session:
        # Pass the open session so the demo branch reuses it (reading the shared
        # anchor on this same session) instead of opening a second one; the ESPN
        # (prod) branch ignores the arg, keeping prod behavior identical.
        source = default_scoreboard_source(session)
        result = SCORES_JOB.reconcile(session, source)
        session.commit()

        # QT-3 pickem-CHAT: publish the in-cycle edges the reconcile collected,
        # POST-COMMIT and best-effort (publish_event swallows Redis errors, so a
        # chat hiccup can never break the poll). The edges are display-only data
        # returned on the RefreshResult — refresh_games itself never publishes.
        _publish_refresh_chat_edges(session, result)

        return {
            "weeks_polled": result.weeks_polled,
            "games_updated": result.games_updated,
            "windows_stamped": result.windows_stamped,
            "failed_weeks": [list(w) for w in result.failed_weeks],
        }


def _active_refresh_season(session: Session) -> int | None:
    """Resolve the season the poller is reconciling, for recap standings.

    The in-cycle edges carry a ``week`` number only; the standings services need
    the ``season`` too. Delegates to the shared
    :func:`app.services.standings.active_season` selector: the newest persisted
    season (``max(Game.season)``) is active, so on a multi-season DB the newest
    season's recap is published (not skipped); ``None`` only when there are ZERO
    games, in which case the caller skips the season-scoped recap (a non-essential
    social ping — lossy is acceptable per the QT-3 design decision). The name and
    ``int | None`` signature are preserved so ``_publish_refresh_chat_edges`` and
    its tests are unaffected.
    """
    return active_season(session)


def _publish_refresh_chat_edges(session: Session, result) -> None:
    """Publish the collected refresh edges to the pickem-CHAT feed, best-effort.

    One ``game.final`` per finalized game; one ``window.opened`` / ``window.closed``
    per crossing week; one ``week.recap`` per recap week (the recap payload — the
    week winner + season leader display names and scores — is pulled from the
    existing standings services, NO re-implemented scoring). Every publish is
    post-commit + best-effort (publish_event swallows), so a chat outage never
    breaks the poll cycle.
    """
    for week, away, home, away_score, home_score in result.finalized_games:
        publish_event(
            game_final_event(
                week=week,
                away_abbr=away,
                home_abbr=home,
                away_score=away_score,
                home_score=home_score,
            )
        )

    for week in result.windows_opened:
        publish_event(window_opened_event(week))
    for week in result.windows_closed:
        publish_event(window_closed_event(week))

    if not result.recap_weeks:
        return

    season = _active_refresh_season(session)
    if season is None:
        return  # ambiguous season — skip the season-scoped recap (lossy is fine)

    season_leaders, _ = season_standings(session, season=season)
    leader = season_leaders.results[0] if season_leaders.results else None
    for week in result.recap_weeks:
        week_winners = week_results(session, season=season, week=week)
        winner = week_winners[0] if week_winners else None
        if winner is None or leader is None:
            continue  # no graded picks yet — nothing to recap
        publish_event(
            week_recap_event(
                week=week,
                winner=winner.display_name,
                winner_score=winner.weekly_score,
                leader=leader.display_name,
                leader_score=leader.season_total,
            )
        )


@celery_app.task(name="app.tasks.refresh_odds")
def refresh_odds_task() -> dict:
    """Beat-driven poller: reconcile betting lines until the week freezes.

    Thin wrapper that DISPATCHES the odds
    :class:`~app.services.scheduler.PollingJob` (the sibling of the scores job):
    open a non-HTTP ``task_session()``, resolve the gated default scoreboard
    source (the real ESPN adapter in prod, the time-shifted Demo2025Source under
    the demo gate) via :func:`app.config.default_scoreboard_source` — the gated
    resolution stays HERE in the thin wrapper, NOT in the source-agnostic odds
    service — run the odds job's reconcile (which delegates to
    :func:`app.services.odds.reconcile_odds_games`), commit, and return a
    JSON-serializable summary so Celery's json result serializer accepts it.

    Cadence trigger: registered in ``app.celery_app``'s ``beat_schedule`` (built
    from the polling-job registry, so the odds job's
    ``REFRESH_ODDS_INTERVAL_SECONDS`` slower cadence) so beat drives this poller
    on its own timer, independent of the scores poller.
    """
    with task_session() as session:
        # Pass the open session so the demo branch reuses it (reading the shared
        # anchor on this same session); the ESPN (prod) branch ignores the arg.
        source = default_scoreboard_source(session)
        result = ODDS_JOB.reconcile(session, source)
        session.commit()
        return {
            "weeks_polled": result.weeks_polled,
            "games_updated": result.games_updated,
            "frozen_weeks": [list(w) for w in result.frozen_weeks],
            "failed_weeks": [list(w) for w in result.failed_weeks],
        }


@celery_app.task(name="app.tasks.ingest_season")
def ingest_season_task(season: int) -> dict:
    """Manual worker-callable trigger: CREATE the Week+Game skeleton for a season.

    Thin wrapper mirroring :func:`refresh_games_task` / :func:`refresh_odds_task`
    EXACTLY — the schedule-CREATE path the pollers do NOT cover (they only UPDATE
    existing rows). Opens a non-HTTP ``task_session()``, resolves the GATED default
    scoreboard source (the real ESPN adapter in prod, the time-shifted
    Demo2025Source under the demo gate) via
    :func:`app.config.default_scoreboard_source` — the gated resolution stays HERE
    in the thin wrapper, NOT in the source-agnostic
    :func:`app.services.ingest.ingest_season` service — runs the ingest, and
    returns a JSON-serializable summary so Celery's json result serializer accepts
    it (``failed_weeks`` flattened to lists, matching the refresh/odds wrappers).

    This is a MANUAL trigger only (callable like the other tasks). It is
    deliberately NOT registered in the polling-job registry / ``beat_schedule`` and
    has NO HTTP route — an automated cadence + admin trigger UI is out of scope
    (QT-1) and deferred to QT-2.

    :param season: the NFL season year to ingest (e.g. ``2026``).
    """
    with task_session() as session:
        # Pass the open session so the demo branch reuses it (reading the shared
        # anchor on this same session); the ESPN (prod) branch ignores the arg.
        source = default_scoreboard_source(session)
        result = ingest_season(session, source, season)
        # ingest_season commits once at the end; this wrapper-level commit is a
        # harmless no-op that keeps the wrapper shape identical to its siblings.
        session.commit()
        # Post-commit, best-effort pickem-logger ops line — reuse the SAME
        # non-sensitive summary fields the task already returns. publish_event
        # swallows Redis errors, so logging can never break the ingest.
        publish_event(
            ingest_season_event(
                season=season,
                weeks=result.weeks_present,
                games=result.games_present,
                failed=len(result.failed_weeks),
            )
        )
        return {
            "weeks_present": result.weeks_present,
            "games_present": result.games_present,
            "weeks_created": result.weeks_created,
            "games_created": result.games_created,
            "failed_weeks": [list(w) for w in result.failed_weeks],
        }


@celery_app.task(name="app.tasks.freeze_week")
def freeze_week_task(season: int, week: int) -> dict:
    """Manual worker-callable trigger: re-snapshot + LOCK one week's lines NOW.

    Thin gated wrapper mirroring :func:`ingest_season_task` EXACTLY — the
    on-demand line-freeze the computed odds clock does NOT cover (it freezes on a
    cadence; this freezes immediately, before the ephemeral DraftKings line
    vanishes). Opens a non-HTTP ``task_session()``, resolves the GATED default
    scoreboard source (the real ESPN adapter in prod, the time-shifted
    Demo2025Source under the demo gate) via
    :func:`app.config.default_scoreboard_source` — the gated resolution stays HERE
    in the thin wrapper, NOT in the source-agnostic
    :func:`app.services.freeze.freeze_week` service — runs the freeze, and returns
    a JSON-serializable summary so Celery's json result serializer accepts it.

    This is a MANUAL trigger only (dispatched from the admin
    ``POST /api/admin/freeze-week`` route). It is deliberately NOT registered in
    the polling-job registry / ``beat_schedule``.

    :param season: the NFL season year (e.g. ``2026``).
    :param week: the regular-season week to freeze (``1``..``18``).
    """
    with task_session() as session:
        # Pass the open session so the demo branch reuses it (reading the shared
        # anchor on this same session); the ESPN (prod) branch ignores the arg.
        source = default_scoreboard_source(session)
        result = freeze_week(session, source, season, week)
        # freeze_week commits once at the end; this wrapper-level commit is a
        # harmless no-op that keeps the wrapper shape identical to its siblings.
        session.commit()
        # Post-commit, best-effort pickem-logger ops line. publish_event swallows
        # Redis errors, so logging can never break the freeze.
        publish_event(freeze_week_event(week=result.week))
        return {
            "season": result.season,
            "week": result.week,
            "games_updated": result.games_updated,
            "already_frozen": result.already_frozen,
            "failed": result.failed,
        }
