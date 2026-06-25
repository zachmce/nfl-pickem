"""pick_edit_audit — permanent record of admin pick overrides

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-25 00:00:00.000000

Hand-written (mirrors 0004/0005/0006's style) so the new ``pick_edit_audit``
table is created deterministically and — critically — reuses the EXISTING
``picktype`` native enum (created by 0004) instead of trying to recreate it.

The two user FKs (``admin_user_id`` / ``target_user_id``) deliberately carry NO
``ondelete`` cascade: this is the OPPOSITE of ``pick.user_id`` (0006). The audit
is a permanent record, so deleting a user must NOT delete the audit rows that
reference them (locked decision 6 in .planning/notes/admin-pick-override-design.md).

``downgrade()`` drops only the table — it does NOT drop the shared ``picktype``
enum, which predates this migration (0004 owns it).
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel  # noqa: F401  (kept for parity with 0004/0005/0006)

from app.models import PickType

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pick_edit_audit",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("admin_user_id", sa.Integer(), nullable=False),
        sa.Column("target_user_id", sa.Integer(), nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("week_id", sa.Integer(), nullable=False),
        sa.Column(
            "action",
            sqlmodel.sql.sqltypes.AutoString(length=10),
            nullable=False,
        ),
        sa.Column("before_existed", sa.Boolean(), nullable=False),
        # Reuse the EXISTING picktype enum — 0004 created it. NOTE: ``create_type``
        # is a PostgreSQL-dialect parameter; the generic ``sa.Enum`` silently
        # ignores it and would re-emit ``CREATE TYPE picktype`` (DuplicateObject
        # against 0004's type). Use ``postgresql.ENUM`` so create_type=False is
        # honored and the existing type is referenced, not recreated.
        sa.Column(
            "before_pick_type",
            postgresql.ENUM(PickType, name="picktype", create_type=False),
            nullable=True,
        ),
        sa.Column("before_is_mortal_lock", sa.Boolean(), nullable=True),
        sa.Column(
            "after_pick_type",
            postgresql.ENUM(PickType, name="picktype", create_type=False),
            nullable=True,
        ),
        sa.Column("after_is_mortal_lock", sa.Boolean(), nullable=True),
        sa.Column("game_was_final", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        # User FKs: NO ondelete — the audit survives a user delete (opposite of
        # pick.user_id's ON DELETE CASCADE).
        sa.ForeignKeyConstraint(
            ["admin_user_id"], ["users.id"],
            name="fk_pick_edit_audit_admin_user_id_users",
        ),
        sa.ForeignKeyConstraint(
            ["target_user_id"], ["users.id"],
            name="fk_pick_edit_audit_target_user_id_users",
        ),
        sa.ForeignKeyConstraint(
            ["game_id"], ["game.id"],
            name="fk_pick_edit_audit_game_id_game",
        ),
        sa.ForeignKeyConstraint(
            ["week_id"], ["week.id"],
            name="fk_pick_edit_audit_week_id_week",
        ),
    )


def downgrade() -> None:
    # Drop only the table. The shared picktype enum predates this migration
    # (0004 owns it), so it is intentionally left in place.
    op.drop_table("pick_edit_audit")
