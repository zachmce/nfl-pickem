from celery import Celery

from app.config import settings
from app.services.scheduler import POLLING_JOBS

celery_app = Celery(
    "nfl_pickem",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)

# Beat cadence for the game-reconciliation poller (DEMO-BEAT). 60s is frequent
# enough to flip games SCHEDULED->FINAL promptly as the shifted demo clock crosses
# each kickoff, and is a reasonable cadence for real ESPN polling in prod too. The
# poller is source-agnostic — it polls whatever default_scoreboard_source() resolves
# (the gated ESPN-or-demo source), so this beat wiring is general prod work, just
# driven by the gated source. This is the ONE home of the scores cadence: the
# scheduler's scores PollingJob carries the same 60.0s value, and the beat below
# is built FROM the registry so the two cannot drift.
REFRESH_GAMES_INTERVAL_SECONDS: float = 60.0


def _beat_schedule_from_registry() -> dict[str, dict[str, object]]:
    """Derive the Celery ``beat_schedule`` from the polling-job registry.

    Each :class:`~app.services.scheduler.PollingJob` emits one beat entry keyed by
    its ``beat_name``, dispatching its registered ``task_name`` (a string — never
    an import — so the beat entry can't create a circular import; the task is
    registered via ``include=["app.tasks"]``) on its ``schedule_seconds`` cadence.
    For the scores job this produces exactly the historical entry:
    ``{"refresh-games-poller": {"task": "app.tasks.refresh_games", "schedule":
    60.0}}``. A sibling odds job appended to the registry adds its own entry here
    automatically, without touching the scores entry.
    """
    return {
        job.beat_name: {
            "task": job.task_name,
            "schedule": job.schedule_seconds,
        }
        for job in POLLING_JOBS
    }


celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule=_beat_schedule_from_registry(),
)
