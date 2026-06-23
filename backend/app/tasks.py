from sqlmodel import Session

from app.celery_app import celery_app
from app.db import engine
from app.models import TaskRun


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
