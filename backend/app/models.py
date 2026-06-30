from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import ClassVar

import sqlalchemy as sa
from sqlalchemy import BigInteger, Column
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
#
# No enums existed in the codebase before the pick'em domain. Convention
# established here: a Python ``str, Enum`` for type-safety in app code, mapped
# to a *native* Postgres enum type via ``sa.Enum(EnumClass, name=...)`` so the
# database enforces the allowed values too. Members are UPPER_SNAKE.
# ---------------------------------------------------------------------------
class GameStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    FINAL = "FINAL"


class PickType(str, Enum):
    UNDERDOG_COVER = "UNDERDOG_COVER"
    FAVORITE_COVER = "FAVORITE_COVER"
    OVER = "OVER"
    UNDER = "UNDER"
    # The one manually-graded type: a weekly free-text prediction tied to any
    # real game. Its outcome cannot be auto-derived from the game, so an admin
    # sets ``Pick.result`` / ``Pick.points`` and the scoring engine passes those
    # stored values through (see ``app.services.scoring.grade_pick``). MISC is
    # never a mortal lock, so the existing ``uq_pick_user_week_type_base`` partial
    # unique index already enforces one MISC base pick per user/week.
    MISC = "MISC"


class PickResult(str, Enum):
    PENDING = "PENDING"
    WIN = "WIN"
    LOSS = "LOSS"


class User(SQLModel, table=True):
    __tablename__: ClassVar[str] = "users"

    # The plain ``unique=True`` on the ``discord_id`` Column keeps every NON-null
    # discord_id distinct, but SQL UNIQUE treats NULLs as distinct so it does NOT
    # cap the number of null rows. This partial unique index adds the
    # at-most-one-NULL guarantee (the lone null is the web break-glass admin). It
    # is declared at the MODEL level (not only in the migration) so the SQLite
    # ``create_all`` test DB enforces "one null" too — mirror of the dialect-kwarg
    # pattern already used by ``Pick.uq_pick_user_week_type_base`` below.
    #
    # The indexed EXPRESSION is the constant ``1`` (not the ``discord_id`` column):
    # within the filtered set ``discord_id IS NULL`` is always true, and a UNIQUE
    # index over the *column* would still treat each NULL as distinct (NULLs are
    # never equal) on BOTH SQLite and Postgres, so a column index would not cap
    # them. Indexing a constant makes every qualifying row collide on the same
    # key, enforcing at-most-one.
    __table_args__ = (
        sa.Index(
            "uq_users_one_null_discord_id",
            sa.text("1"),
            unique=True,
            postgresql_where=sa.text("discord_id IS NULL"),
            sqlite_where=sa.text("discord_id IS NULL"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    discord_id: int | None = Field(
        sa_column=Column(BigInteger, nullable=True, unique=True, index=True),
        ge=0,
        le=18446744073709551615,
        description="The unique Discord User Snowflake ID (NULL for web-bootstrap admin)",
    )
    password_hash: str | None = None
    # The Discord avatar hash (32-char hex, or prefixed ``a_`` for an animated
    # avatar). None when the member has no custom avatar (default avatar) or for
    # web-only / seeded-admin accounts. The site builds the CDN URL from this hash
    # downstream; storing only the hash keeps this layer Discord-free of URLs.
    discord_avatar_hash: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(64), nullable=True)
    )
    display_name: str = Field(max_length=100, unique=True)
    is_admin: bool = Field(default=False)
    is_active: bool = Field(default=False)
    # The break-glass marker: True ONLY for the bootstrap seed admin (the lone
    # NULL-discord_id account). A protected row can never be deleted / demoted /
    # deactivated by the web admin service, so the system can never be locked out
    # of admin. Set at seed-time create (admins.py) AND at migrate-time backfill
    # (0012); read by the admin-service guards and surfaced through
    # AdminUserRow -> AdminUserRead -> the AdminPage row disable.
    is_protected: bool = Field(default=False)
    # Monotonic session-invalidation counter. Carried in the signed session cookie
    # as ``sv`` and compared on every authenticated request (deps.get_current_user):
    # a cookie whose ``sv`` differs from this column is treated as logged-out. Every
    # password-write site (change_password, reset_password_for_discord) bumps it so a
    # password change invalidates all previously-issued cookies. Default 0 +
    # server_default "0": a Postgres ADD COLUMN backfills existing rows to 0, and a
    # legacy cookie that carries no ``sv`` key decodes to 0 — so existing valid
    # sessions keep working and a deploy does NOT force-log-out the current user base.
    session_version: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )
    created_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False), default_factory=sa.func.now
    )


class DemoState(SQLModel, table=True):
    """The single demo anchor instant — the determinism crux of the demo mode.

    Persists ONE absolute real-clock instant (``demo_started_at``), captured at
    seed time, that BOTH the seed process and the worker/beat process read to
    rebuild the SAME ``Demo2025Source(offset)``. Persisting an absolute instant
    (not a recomputed "now+24h" and not a timedelta) is what keeps the two
    processes in sync — neither process recomputes its own positioning clock.

    Single-row table (upserted), deliberately minimal.
    """

    __tablename__ = "demo_state"

    id: int | None = Field(default=None, primary_key=True)
    demo_started_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False)
    )


class AppSetting(SQLModel, table=True):
    """A keyed app-wide settings row (the generic key/value store).

    Mirrors :class:`DemoState`'s single-row minimalism but is KEYED rather than
    a singleton: one row per ``setting_key`` (unique + indexed), carrying an
    opaque string ``setting_value`` and a tz-aware ``updated_at`` bumped on every
    write. The first consumer is the admin-selectable bot personality (stored
    under key ``bot_personality``); the table is deliberately generic so future
    app-wide toggles can reuse it without a new migration per setting.

    No enum is involved (the personality id is a free string validated in the
    service against the registry), so there is no PG-enum-reuse concern here.
    """

    __tablename__ = "app_setting"

    id: int | None = Field(default=None, primary_key=True)
    setting_key: str = Field(
        sa_column=sa.Column(sa.String(100), nullable=False, unique=True, index=True),
    )
    setting_value: str = Field(sa_column=sa.Column(sa.String, nullable=False))
    updated_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
        default_factory=_utcnow,
    )


# ---------------------------------------------------------------------------
# Pick'em domain
#
# Schema authority: .planning/notes/data-model.md. Surrogate int PKs; ESPN ids
# stored as unique columns (never PKs); odds embedded on Game (single frozen
# line, all nullable); Pick stores only (game, type) — the side is derived.
# ---------------------------------------------------------------------------
class Team(SQLModel, table=True):
    """NFL team reference data (~32 rows, seeded once)."""

    __tablename__ = "team"

    id: int | None = Field(default=None, primary_key=True)
    espn_team_id: int = Field(
        sa_column=sa.Column(sa.Integer, nullable=False, unique=True),
        description="ESPN team id — stored as a unique column, never the PK.",
    )
    abbreviation: str = Field(max_length=10, nullable=False)
    display_name: str = Field(max_length=100, nullable=False)


class Week(SQLModel, table=True):
    """A scoring week — owns the pick window and the line-freeze state."""

    __tablename__ = "week"
    __table_args__ = (sa.UniqueConstraint("season", "week", name="uq_week_season_week"),)

    id: int | None = Field(default=None, primary_key=True)
    season: int = Field(nullable=False)
    week: int = Field(nullable=False)
    # Window stored (not derived) so it is explicit: closes at the first
    # kickoff of the week, opens after the previous week's last game.
    window_opens_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    window_closes_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    lines_frozen: bool = Field(default=False, nullable=False)


class Game(SQLModel, table=True):
    """A single NFL game with its embedded (single, frozen) betting line."""

    __tablename__ = "game"

    id: int | None = Field(default=None, primary_key=True)
    espn_event_id: int = Field(
        sa_column=sa.Column(sa.Integer, nullable=False, unique=True),
        description="ESPN event id — unique column, never the PK.",
    )
    # Nullable to be safe for ingest ordering; the data-model lists it without a
    # uniqueness/null note.
    espn_competition_id: int | None = Field(default=None, nullable=True)
    week_id: int = Field(foreign_key="week.id", nullable=False)
    # Convenience duplicates of the owning week's (season, week) per data-model.
    season: int = Field(nullable=False)
    week: int = Field(nullable=False)
    home_team_id: int = Field(foreign_key="team.id", nullable=False)
    away_team_id: int = Field(foreign_key="team.id", nullable=False)
    kickoff_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    status: GameStatus = Field(
        default=GameStatus.SCHEDULED,
        sa_column=sa.Column(sa.Enum(GameStatus, name="gamestatus"), nullable=False),
    )
    home_score: int | None = Field(default=None, nullable=True)
    away_score: int | None = Field(default=None, nullable=True)

    # --- Embedded odds (all nullable: a game may exist before lines are posted) ---
    # ``spread`` is the positive half-point magnitude the favorite must cover.
    spread: Decimal | None = Field(
        default=None, sa_column=sa.Column(sa.Numeric(4, 1), nullable=True)
    )
    total: Decimal | None = Field(
        default=None, sa_column=sa.Column(sa.Numeric(5, 1), nullable=True)
    )
    favorite_team_id: int | None = Field(default=None, foreign_key="team.id")
    underdog_team_id: int | None = Field(default=None, foreign_key="team.id")
    odds_provider: str | None = Field(default=None, max_length=50, nullable=True)
    # The persisted chosen-provider id — a drift-proof audit companion to
    # ``odds_provider`` (the name). Both are captured from the SAME selected odds
    # item at ingest time so a line can always be traced back to its exact source
    # (provider ids drift across ESPN endpoints, so the name alone is ambiguous).
    odds_provider_id: str | None = Field(default=None, max_length=50, nullable=True)
    odds_frozen: bool = Field(default=False, nullable=False)
    odds_captured_at: datetime | None = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )


class Pick(SQLModel, table=True):
    """A user's pick on a game for a week. Side is derived from game + type.

    Two Postgres partial unique indexes enforce, at the DB level:
      * one of each base pick type per user/week (``is_mortal_lock = false``)
      * one mortal lock per user/week (``is_mortal_lock = true``)
    """

    __tablename__ = "pick"
    __table_args__ = (
        sa.Index(
            "uq_pick_user_week_type_base",
            "user_id",
            "week_id",
            "pick_type",
            unique=True,
            postgresql_where=sa.text("is_mortal_lock = false"),
            sqlite_where=sa.text("is_mortal_lock = false"),
        ),
        sa.Index(
            "uq_pick_user_week_mortal_lock",
            "user_id",
            "week_id",
            unique=True,
            postgresql_where=sa.text("is_mortal_lock = true"),
            sqlite_where=sa.text("is_mortal_lock = true"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", ondelete="CASCADE", nullable=False)
    game_id: int = Field(foreign_key="game.id", nullable=False)
    week_id: int = Field(foreign_key="week.id", nullable=False)
    pick_type: PickType = Field(
        sa_column=sa.Column(sa.Enum(PickType, name="picktype"), nullable=False),
    )
    is_mortal_lock: bool = Field(default=False, nullable=False)
    # Free-text prediction, populated ONLY for a MISC pick (NULL for every other
    # type). Rendered as a nullable VARCHAR(280); no new index is needed — the
    # existing partial unique index ``uq_pick_user_week_type_base`` already caps a
    # user at one MISC base pick per week (MISC is never a mortal lock).
    misc_text: str | None = Field(default=None, max_length=280, nullable=True)
    result: PickResult = Field(
        default=PickResult.PENDING,
        sa_column=sa.Column(sa.Enum(PickResult, name="pickresult"), nullable=False),
    )
    points: int = Field(default=0, nullable=False)
    created_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
        default_factory=_utcnow,
    )
    updated_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
        default_factory=_utcnow,
    )


class PickEditAudit(SQLModel, table=True):
    """One record of a single admin pick override.

    The admin override path (:mod:`app.services.admin_picks`) writes one of these
    rows for every set/clear it performs, recording WHO (the acting admin) edited
    WHOSE (the target user) pick on which game/week, the before->after slot state,
    and whether the game was already FINAL at edit time. It is a Repudiation
    mitigation (T-m66-04): the audit answers "who changed whose pick".

    CASCADE (reverses the prior pick-edit-audit decision 6, per
    ``.planning/notes/admin-hardening-pre-stakeholder.md`` decision 6): both user
    FKs (``admin_user_id`` and ``target_user_id``) now carry ``ondelete="CASCADE"``,
    MIRRORING :attr:`Pick.user_id`. Deleting a user therefore removes their audit
    rows (whether they were the acting admin or the target) instead of raising a
    Postgres FK violation when an admin hard-deletes a user. This intentionally
    drops the audit's former "survive the user / permanent record" guarantee —
    Zach explicitly accepted that a user delete also deletes their audit rows. The
    audit's PURPOSE (who-changed-whose-pick repudiation mitigation) is unchanged;
    only its permanence is. The Postgres-path companion is migration 0013.

    Reuses the EXISTING ``picktype`` Postgres enum for the before/after pick-type
    columns (no new enum type is introduced). Does NOT model the deferred
    standings-shift surface — only the ``game_was_final`` bool is recorded
    (scoring is recompute-on-read, so editing a past pick needs no backfill).
    """

    __tablename__: ClassVar[str] = "pick_edit_audit"

    id: int | None = Field(default=None, primary_key=True)
    # Both user FKs cascade on user delete (mirrors Pick.user_id) — deleting a
    # user removes their audit rows. Reverses the prior no-ondelete "audit
    # survives the user" decision (hardening note decision 6). No relationship is
    # declared on this table, so the column-level ondelete is what drives the
    # cascade (exactly as Pick.user_id does); no passive_deletes is needed.
    admin_user_id: int = Field(foreign_key="users.id", ondelete="CASCADE", nullable=False)
    target_user_id: int = Field(foreign_key="users.id", ondelete="CASCADE", nullable=False)
    game_id: int = Field(foreign_key="game.id", nullable=False)
    week_id: int = Field(foreign_key="week.id", nullable=False)
    # "set" | "clear" — the override action that produced this row.
    action: str = Field(max_length=10, nullable=False)
    before_existed: bool = Field(nullable=False)
    # Reuse the EXISTING picktype enum (do NOT create a new enum type).
    before_pick_type: PickType | None = Field(
        default=None,
        sa_column=sa.Column(sa.Enum(PickType, name="picktype"), nullable=True),
    )
    before_is_mortal_lock: bool | None = Field(default=None, nullable=True)
    after_pick_type: PickType | None = Field(
        default=None,
        sa_column=sa.Column(sa.Enum(PickType, name="picktype"), nullable=True),
    )
    after_is_mortal_lock: bool | None = Field(default=None, nullable=True)
    game_was_final: bool = Field(nullable=False)
    created_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
        default_factory=_utcnow,
    )
