from sqlmodel import Session

from app.celery_app import celery_app
from app.config import default_scoreboard_source
from app.db import engine, task_session
from app.models import TaskRun
from app.services.refresh import refresh_games


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

    Thin wrapper around the source-agnostic
    :func:`app.services.refresh.refresh_games` service: open a non-HTTP
    ``task_session()``, resolve the production default scoreboard source (the
    real ESPN adapter) via :func:`app.config.default_scoreboard_source`, run the
    reconciliation, commit, and return a JSON-serializable summary so Celery's
    json result serializer accepts it.

    Beat schedule wiring (the cadence trigger) is intentionally OUT OF SCOPE.
    """
    with task_session() as session:
        source = default_scoreboard_source()
        result = refresh_games(session, source)
        session.commit()
        return {
            "weeks_polled": result.weeks_polled,
            "games_updated": result.games_updated,
            "windows_stamped": result.windows_stamped,
            "failed_weeks": [list(w) for w in result.failed_weeks],
        }
