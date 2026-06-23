from collections.abc import Generator
from contextlib import contextmanager

from sqlmodel import Session, create_engine

from app.config import settings

# Single shared synchronous engine. Used by the FastAPI app, the Celery
# workers/beat, AND the Discord bot (via app.bot.db_bridge, which offloads
# these sync calls to a thread so it never blocks the discord.py event loop).
# Keeping one sync engine means the same Session code runs everywhere.
engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session (one per request)."""
    with Session(engine) as session:
        yield session


@contextmanager
def task_session() -> Generator[Session, None, None]:
    """Session context manager for non-HTTP callers (Celery tasks, the bot).

    Do NOT share a request-scoped session into a task or the bot thread — open
    a fresh one here.
    """
    with Session(engine) as session:
        yield session
