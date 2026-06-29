"""pick_edit_audit user FKs — recreate with ON DELETE CASCADE

Revision ID: 0013
Revises: 0012
Create Date: 2026-06-29 00:00:00.000000

Hand-written (mirrors 0006's FK drop/recreate style) so BOTH user FKs on
``pick_edit_audit`` — ``admin_user_id`` and ``target_user_id`` — carry
``ON DELETE CASCADE`` at the database level. With this in place, hard-deleting a
user who appears in audit rows (as the acting admin OR the target) removes those
audit rows instead of raising a Postgres FK violation in
``app.services.admin.delete_user``.

REVERSES the prior decision: migration 0007 created these FKs with NO ``ondelete``
on the explicit "the audit is a PERMANENT record / must survive a user delete"
locked decision. That decision was reversed in
.planning/notes/admin-hardening-pre-stakeholder.md decision 6 — Zach explicitly
accepts that deleting a user also deletes their audit rows. This migration is the
Postgres-path companion to the model-level ``ondelete="CASCADE"`` on
``PickEditAudit.admin_user_id`` / ``target_user_id``.

Targets Postgres (the production DB): drop each existing plain FK and recreate it
WITH cascade. The EXACT existing constraint names from 0007
(``fk_pick_edit_audit_admin_user_id_users`` /
``fk_pick_edit_audit_target_user_id_users``) are reused so the SQLModel model
metadata and the migrated DB schema stay in agreement. No table/enum/index
changes in either direction — only these two constraints are swapped.

Downgrade reverses it: drop the cascade FKs and recreate the plain NO ACTION FKs
under the same names.
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa  # noqa: F401  (kept for parity with 0006/0007)
import sqlmodel  # noqa: F401  (kept for parity with 0006/0007)

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Recreate BOTH user FKs at users with cascade. Reuse the exact names from 0007.
    op.drop_constraint(
        "fk_pick_edit_audit_admin_user_id_users",
        "pick_edit_audit",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_pick_edit_audit_admin_user_id_users",
        "pick_edit_audit",
        "users",
        ["admin_user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(
        "fk_pick_edit_audit_target_user_id_users",
        "pick_edit_audit",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_pick_edit_audit_target_user_id_users",
        "pick_edit_audit",
        "users",
        ["target_user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # Restore the plain NO ACTION FKs (no ondelete) under the same names.
    op.drop_constraint(
        "fk_pick_edit_audit_admin_user_id_users",
        "pick_edit_audit",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_pick_edit_audit_admin_user_id_users",
        "pick_edit_audit",
        "users",
        ["admin_user_id"],
        ["id"],
    )
    op.drop_constraint(
        "fk_pick_edit_audit_target_user_id_users",
        "pick_edit_audit",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_pick_edit_audit_target_user_id_users",
        "pick_edit_audit",
        "users",
        ["target_user_id"],
        ["id"],
    )
