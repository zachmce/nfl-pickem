"""admin authorization floor — is_protected column + one-null discord_id

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-29 00:00:00.000000

Hand-written (mirrors 0006/0011's style). Establishes the admin authorization
floor before stakeholder testing: it adds an ``is_protected`` break-glass marker
to ``users``, collapses every existing NULL ``discord_id`` row down to the single
bootstrap admin (by assigning deterministic small-int fake ids to the rest), and
finally adds a PARTIAL UNIQUE index enforcing AT MOST ONE NULL ``discord_id`` (the
lone protected admin).

This is a PLAIN structural migration: there is NO enum involved, so the Postgres
enum-reuse caveat (``postgresql.ENUM(create_type=False)``, see 0007/0008) does
NOT apply here.

ORDER IS LOAD-BEARING (the index-first ordering fails on today's 9 NULL rows):
  1. add ``is_protected`` BOOLEAN NOT NULL server_default false (so existing rows
     backfill to false without a NULL-violation).
  2. backfill is_protected=true for the bootstrap admin (matched on
     DEFAULT_ADMIN_USERNAME from settings — never hardcoded). Harmless no-op on a
     fresh DB with no admin yet.
  3. backfill deterministic small-int fake discord_ids for EVERY remaining NULL
     row that is NOT the protected admin (today's 5 bots + demo_admin + any other
     non-protected null), starting at 1 ordered by id so values are stable. This
     collapses the 9 current nulls down to the single protected admin.
  4. create the partial unique index ``uq_users_one_null_discord_id``
     (UNIQUE WHERE discord_id IS NULL) — at-most-one-null from here on.

NOTE on demo_admin: there is NO app-level ``demo_admin`` seed (it lives only in
the untracked ``backend/_demo_tools/dw.py``). So THIS migration's step 3 is what
assigns ``demo_admin`` its fake id on the live DB — no app seed change is needed
for it.

The SQLite-only OFFLINE tests do not run migrations — they build the schema from
the SQLModel metadata (``SQLModel.metadata.create_all``), so the ``is_protected``
column and the partial unique index are exercised in tests via the new
``User.is_protected`` field and the model-level ``uq_users_one_null_discord_id``
index, and by this migration only on the live Postgres path.

``downgrade()`` drops the index then the column. It deliberately does NOT attempt
to restore the NULL discord_ids: the fake ids are intentional and
irreversible-by-design.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401  (kept for parity with 0004..0011)

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Lazy import (matching the seeds' discipline) so the bootstrap admin's
    # display_name is resolved from settings, never hardcoded.
    from app.config import settings

    # 1. Add the break-glass marker. server_default false backfills existing rows
    #    so the NOT NULL holds without a separate UPDATE.
    op.add_column(
        "users",
        sa.Column(
            "is_protected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("discord_id", sa.BigInteger),
        sa.column("display_name", sa.String),
        sa.column("is_protected", sa.Boolean),
    )

    # 2. Mark the bootstrap admin protected (matched on DEFAULT_ADMIN_USERNAME).
    #    Harmless no-op if no such row exists yet (fresh DB).
    admin_username = settings.default_admin_username
    if admin_username:
        op.execute(
            users.update().where(users.c.display_name == admin_username).values(is_protected=True)
        )

    # 3. Backfill deterministic small-int fake discord_ids for every remaining
    #    NULL row that is NOT the protected admin. Ordered by id so the values are
    #    stable; start at 1. Single-digit/small ids are collision-proof against
    #    real ~10^17 Discord snowflakes.
    conn = op.get_bind()
    null_ids = conn.execute(
        sa.select(users.c.id)
        .where(users.c.discord_id.is_(None))
        .where(users.c.is_protected.is_(False))
        .order_by(users.c.id)
    ).fetchall()
    for fake_id, (user_id,) in enumerate(null_ids, start=1):
        conn.execute(users.update().where(users.c.id == user_id).values(discord_id=fake_id))

    # 4. Now that at most one NULL remains (the protected admin), enforce it.
    #    The indexed EXPRESSION is the constant ``1`` (not the ``discord_id``
    #    column): a UNIQUE index over the column would still treat each NULL as
    #    distinct on Postgres (NULLs are never equal), so it would not cap them.
    #    Indexing a constant makes every ``discord_id IS NULL`` row collide on the
    #    same key, enforcing at-most-one. This mirrors the model-level
    #    ``uq_users_one_null_discord_id`` index so the schemas stay in agreement.
    op.create_index(
        "uq_users_one_null_discord_id",
        "users",
        [sa.text("(1)")],
        unique=True,
        postgresql_where=sa.text("discord_id IS NULL"),
    )


def downgrade() -> None:
    # Drop the index then the column. The fake discord_ids are intentional and
    # irreversible-by-design — no attempt is made to restore the prior NULLs.
    op.drop_index("uq_users_one_null_discord_id", table_name="users")
    op.drop_column("users", "is_protected")
