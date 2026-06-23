from celery import Celery

from app.config import settings

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
# driven by the gated source.
REFRESH_GAMES_INTERVAL_SECONDS: float = 60.0

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Reference the task by its registered string name (not an import) so the beat
    # entry never creates a circular import — include=["app.tasks"] registers it.
    beat_schedule={
        "refresh-games-poller": {
            "task": "app.tasks.refresh_games",
            "schedule": REFRESH_GAMES_INTERVAL_SECONDS,
        },
    },
)
