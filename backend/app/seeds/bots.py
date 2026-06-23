"""Idempotent demo/bot-user seed for the season-walkthrough oracle.

These are **demo accounts**, not real users. They are the deterministic
"contestants" of the season-walkthrough-as-proof strategy (PROJECT.md Key
Decision: *the stakeholder walkthrough doubles as the integration proof*): the
same bots making the same preordained picks against the same 2025 season must
produce the same standings forever. The results oracle (``app/demo/oracle.py``)
computes the precomputed-expected side of that equation; this seed materializes
the bot User rows the demo driver (#5) persists picks for.

This seed mirrors ``app/seeds/teams.py`` exactly:

* It is **idempotent** — it upserts each bot keyed on the unique ``display_name``
  (the natural key the seeder owns; never the surrogate PK). Re-running leaves
  exactly N bot rows with no duplicates and no error, and re-asserts the canonical
  ``is_active`` / ``is_admin`` flags. It deliberately does **not** re-hash an
  existing row's password: argon2 hashing is nondeterministic, so re-hashing on
  rerun would silently dirty idempotency.
* Bot credentials are created through the **real auth path**
  (:func:`app.services.auth.hash_password`, argon2id) — never a hand-rolled hash.
* Importing this module is **side-effect-free**: it opens no DB connection and
  performs no argon2 work at import time. The bot passwords are static literals;
  hashing happens only inside :func:`seed_bots`, and ``app.db`` is imported only
  inside :func:`main`.

Bot accounts are clearly labeled with a ``bot_`` ``display_name`` prefix and are
ordinary non-admin users (``is_admin=False``) — they carry no elevated rights and
are intended for the disposable demo DB, not production. The passwords are static,
low-value, demo-only credentials (no real-user data).

Run it from the ``backend/`` directory::

    cd backend
    .venv/bin/python -m app.seeds.bots

> Note: on this machine there is no bare ``python`` on ``PATH`` (the interpreter
> is ``python3`` / the venv interpreter ``.venv/bin/python``); use
> ``.venv/bin/python -m app.seeds.bots``.
"""

from __future__ import annotations

from sqlmodel import Session, select

from app.models import User
from app.services.auth import hash_password

# Static demo/bot roster: (display_name, password). Five bots for richer
# standings. display_name is the unique natural key the seed upserts on and is
# clearly demo-labeled (``bot_`` prefix). Passwords are static, low-value,
# demo-only credentials (documented in the module docstring; never real-user
# data). These names MUST match the keys in
# ``app.seeds.data.bot_picks_2025.BOT_PICKS``.
BOT_ACCOUNTS: tuple[tuple[str, str], ...] = (
    ("bot_alice", "demo-alice-2025"),
    ("bot_bob", "demo-bob-2025"),
    ("bot_carol", "demo-carol-2025"),
    ("bot_dave", "demo-dave-2025"),
    ("bot_erin", "demo-erin-2025"),
)


def seed_bots(session: Session) -> int:
    """Idempotently upsert all demo bot users, keyed on ``display_name``.

    For each account, look up the existing :class:`~app.models.User` by its unique
    ``display_name`` (the stable natural key the seeder owns, never the surrogate
    PK). If absent, insert a new row whose ``password_hash`` is produced by
    :func:`app.services.auth.hash_password` (argon2id — never a hand-rolled hash),
    with ``is_active=True``, ``is_admin=False`` and ``discord_id=None``. If
    present, re-assert the canonical ``is_active`` / ``is_admin`` flags but do
    **not** re-hash the password (hashing is nondeterministic; re-hashing on rerun
    would dirty idempotency). Commits once at the end.

    Returns the number of bot accounts processed (N).
    """
    for display_name, password in BOT_ACCOUNTS:
        existing = session.exec(
            select(User).where(User.display_name == display_name)
        ).first()
        if existing is None:
            session.add(
                User(
                    display_name=display_name,
                    password_hash=hash_password(password),
                    is_active=True,
                    is_admin=False,
                    discord_id=None,
                )
            )
        else:
            # Correct-on-rerun (mirrors teams.py), but never re-hash the password.
            existing.is_active = True
            existing.is_admin = False
            session.add(existing)

    session.commit()
    return len(BOT_ACCOUNTS)


def main() -> None:
    """CLI entry point: open a task session, seed bots, print a summary."""
    # Imported here (not at module top) so importing this module never builds the
    # Postgres engine in app.db — keeps the module import side-effect-free.
    from app.db import task_session

    with task_session() as session:
        count = seed_bots(session)
    print(f"Seeded {count} demo bot users.")


if __name__ == "__main__":
    main()
