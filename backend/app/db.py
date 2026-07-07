from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine

from app.config import settings
from app.exceptions import ConflictError

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


def commit_or_conflict(session: Session, *, reason: str = "concurrent_pick_conflict") -> None:
    """Commit ``session``, mapping a unique-index race to a 409 ConflictError.

    Reuses the EXACT recovery shape proven in ``provision_user``
    (app/services/auth.py:254): commit inside a ``try``; on an ``IntegrityError``
    roll back, and if the driver reports Postgres SQLSTATE ``23505`` (unique
    violation) raise a typed :class:`~app.exceptions.ConflictError` (the global
    handler envelopes it as 409). Any OTHER IntegrityError is re-raised unchanged
    so a genuine data-integrity fault is never silently masked behind a 409.

    Note: ``orig.sqlstate == "23505"`` is Postgres-only. SQLite does NOT populate
    ``orig.sqlstate``, so a plain duplicate-insert on the offline SQLite suite
    would NOT hit this branch — the offline tests simulate the Postgres error by
    monkeypatching the session's ``commit`` to raise a crafted IntegrityError.
    """
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        if getattr(e.orig, "sqlstate", None) == "23505":
            raise ConflictError(
                "That pick was just modified in another request — please retry.",
                reason=reason,
            ) from e
        raise


@contextmanager
def task_session() -> Generator[Session, None, None]:
    """Session context manager for non-HTTP callers (Celery tasks, the bot).

    Do NOT share a request-scoped session into a task or the bot thread — open
    a fresh one here.
    """
    with Session(engine) as session:
        yield session
