from sqlmodel import Session

from app.celery_app import celery_app
from app.config import default_scoreboard_source
from app.db import engine, task_session
from app.models import TaskRun
from app.services.scheduler import ODDS_JOB, SCORES_JOB


@celery_app.task(name="app.tasks.ping")
def ping(message: str = "pong") -> dict:
    """Fake task that proves the worker can reach Postgres.

    It writes a ``TaskRun`` row through the shared engine and returns its id.
    """
    with Session(engine) as session:
        run = TaskRun(message=message)
        session.add(run)
        session.commit()
        session.refresh(run)
        return {"id": run.id, "message": run.message}


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
        return {
            "weeks_polled": result.weeks_polled,
            "games_updated": result.games_updated,
            "windows_stamped": result.windows_stamped,
            "failed_weeks": [list(w) for w in result.failed_weeks],
        }


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
