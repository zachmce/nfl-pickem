"""Idempotent admin-bootstrap seed (QT-B, locked decision 1).

This replaces the legacy "first-user-is-admin" rule with a deterministic,
create-if-absent admin bootstrap, mirroring ``app/seeds/bots.py`` /
``app/seeds/teams.py``. It mints a single privileged (``is_admin=True``) account
from the ``DEFAULT_ADMIN_USERNAME`` / ``DEFAULT_ADMIN_PASSWORD`` environment
variables so the system can never be locked out of admin (pairs with QT-C's
last-admin guard) and so demo/UI testing has a reproducible admin login —
without auto-granting admin on registration.

Behavior:

* **Create-if-absent, keyed on the unique** ``display_name`` (the natural key the
  seeder owns; never the surrogate PK). If a user with that ``display_name``
  already exists, the seed **skips** it and never mutates the existing row's
  ``is_admin`` / ``password_hash`` (T-39u-01 EoP, T-39u-02 tampering): the
  password is set exactly once, at creation.
* **Self-gating**: both credentials are required. If either is blank/None the
  seed is a no-op (returns ``False``), so leaving it unconditionally in the
  migrate chain is safe even on the prod path with the vars unset (T-39u-05).
* Credentials default to ``settings.default_admin_username`` /
  ``settings.default_admin_password`` (env, blank->None via the shared Settings
  validator); tests drive it directly via the ``username=`` / ``password=``
  keyword seam.
* Hashing is via :func:`app.services.auth.hash_password` (argon2id) — **never a
  hand-rolled hash** (T-39u-04). The seed logs only the ``display_name`` and a
  created/skipped reason — never the plaintext password or the hash
  (T-39u-03).
* Importing this module is **side-effect-free**: ``app.config`` / ``app.db`` are
  imported only inside the functions (mirroring ``bots.py``), so importing builds
  no Settings singleton and no Postgres engine at module scope.

Run it from the ``backend/`` directory::

    cd backend
    .venv/bin/python -m app.seeds.admins
"""

from __future__ import annotations

import structlog
from sqlmodel import Session, select

from app.models import User
from app.services.auth import hash_password

logger = structlog.get_logger(__name__)


def seed_admin(
    session: Session,
    *,
    username: str | None = None,
    password: str | None = None,
) -> bool:
    """Idempotently bootstrap the deterministic admin user.

    Resolves credentials from the ``username`` / ``password`` keyword args when
    given; otherwise falls back to ``settings.default_admin_username`` /
    ``settings.default_admin_password`` (``app.config`` imported lazily so this
    module stays import-side-effect-free).

    Returns ``True`` only when it creates a new admin; ``False`` on every skip:

    * either resolved credential is falsy (env unset) — no-op,
    * a user with that ``display_name`` already exists — no-op, and the existing
      row's ``password_hash`` is left untouched.

    The created user is canonical: ``is_admin=True``, ``is_active=True``,
    ``discord_id=None``, ``is_protected=True`` (the break-glass marker so the web
    admin service can never delete / demote / deactivate this account), and
    ``password_hash`` produced by :func:`app.services.auth.hash_password`
    (argon2id). NOTE: an existing bootstrap-admin row that predates the
    ``is_protected`` column is handled by migration 0012's backfill, NOT here —
    this seed never mutates an existing row (T-39u-01/02).
    """
    if username is None or password is None:
        # Lazy import keeps the module import side-effect-free (no Settings build
        # at import). Only fill in unset args from settings.
        from app.config import settings

        if username is None:
            username = settings.default_admin_username
        if password is None:
            password = settings.default_admin_password

    if not username or not password:
        logger.info("admin_seed_skipped", reason="credentials unset")
        return False

    existing = session.exec(select(User).where(User.display_name == username)).first()
    if existing is not None:
        # Never touch an existing row's password_hash / is_admin (T-39u-01/02).
        logger.info(
            "admin_seed_skipped",
            reason="username exists",
            user_id=existing.id,
        )
        return False

    session.add(
        User(
            display_name=username,
            password_hash=hash_password(password),
            is_admin=True,
            is_active=True,
            discord_id=None,
            is_protected=True,
        )
    )
    session.commit()
    logger.info("admin_seed_created", display_name=username)
    return True


def main() -> None:
    """CLI entry point: open a task session, seed the admin, print a summary."""
    # Imported here (not at module top) so importing this module never builds the
    # Postgres engine in app.db — keeps the module import side-effect-free.
    from app.db import task_session

    with task_session() as session:
        created = seed_admin(session)
    if created:
        print("Seeded 1 admin user.")
    else:
        print("Admin seed skipped (credentials unset or admin already exists).")


if __name__ == "__main__":
    main()
