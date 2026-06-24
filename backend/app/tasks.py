from sqlmodel import Session

from app.celery_app import celery_app
from app.config import default_scoreboard_source
from app.db import engine, task_session
from app.models import TaskRun
from app.services.scheduler import SCORES_JOB


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
