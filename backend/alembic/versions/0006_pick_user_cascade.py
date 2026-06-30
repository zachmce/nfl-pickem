"""pick.user_id FK — recreate with ON DELETE CASCADE

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-25 00:00:00.000000

Hand-written (mirrors 0004/0005's style) so the only FK pointing at ``users`` —
``pick.user_id`` — carries ``ON DELETE CASCADE`` at the database level. With this
in place, deleting a user automatically removes that user's picks instead of
raising a Postgres FK violation (foundation for the admin "delete user" action;
locked decision 4 in .planning/notes/admin-area-design.md).

Targets Postgres (the production DB), where altering a FK constraint is the
standard path: drop the existing plain ``fk_pick_user_id_users`` and recreate it
WITH cascade. The EXACT existing constraint name (from 0004) is reused so the
SQLModel model metadata and the migrated DB schema stay in agreement.

Downgrade reverses it: drop the cascade FK and recreate the plain NO ACTION FK
under the same name. No table/enum/index changes in either direction — only this
one constraint is swapped.
"""

from collections.abc import Sequence

from alembic import op
import sqlmodel  # noqa: F401  (kept for parity with 0004/0005)

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Recreate the only FK at users with cascade. Reuse the exact name from 0004.
    op.drop_constraint("fk_pick_user_id_users", "pick", type_="foreignkey")
    op.create_foreign_key(
        "fk_pick_user_id_users",
        "pick",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # Restore the plain NO ACTION FK (no ondelete) under the same name.
    op.drop_constraint("fk_pick_user_id_users", "pick", type_="foreignkey")
    op.create_foreign_key(
        "fk_pick_user_id_users",
        "pick",
        "users",
        ["user_id"],
        ["id"],
    )
