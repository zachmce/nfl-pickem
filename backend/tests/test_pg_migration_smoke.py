"""OPT-IN Postgres + Alembic migration smoke test (Codex Theme 6).

The default offline ``python -m unittest`` suite builds the schema with
``SQLModel.metadata.create_all()`` on SQLite and NEVER runs Alembic, so
Postgres-only behaviors (native enum reuse, partial indexes, FK ``ON DELETE``
actions, migration ordering) are never exercised end-to-end. Migrations
0012/0013 had to be hand-verified against Postgres for exactly this reason.

This test closes that gap. When pointed at a REAL, throwaway Postgres it runs
the SAME entrypoint the compose ``migrate`` service uses — ``alembic upgrade
head`` shelled out as a subprocess — and asserts the four load-bearing schema
invariants plus a scoped 0013 down/up round-trip. It does NOT call
``create_all`` and it does NOT import ``app.config`` / ``app.db`` (those point
at the dev/demo DB); the only database it touches is ``TEST_DATABASE_URL``.

It is SKIPPED unless ``TEST_DATABASE_URL`` is set (mirrors the ``RUN_ESPN_LIVE``
``@unittest.skipUnless`` idiom in ``tests/test_scoreboard_espn.py``), so the
default offline suite stays green and Postgres-free.

Run it via the throwaway-Postgres runner (stands up an isolated container on a
non-default port, runs this test, tears it down — NEVER touches the dev/demo
DB)::

    bash backend/scripts/run_pg_smoke.sh

Or point it at any disposable Postgres yourself::

    TEST_DATABASE_URL=postgresql+psycopg://u:p@localhost:5433/db \\
        .venv/bin/python -m unittest tests.test_pg_migration_smoke -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.

No pytest dependency is required (none is configured for this project).

WHY A SUBPROCESS: ``app.config.settings`` is ``lru_cache``d, so the in-process
settings object cannot be repointed at a different DB. A fresh ``python -m
alembic`` subprocess re-imports ``app.config``, which rebuilds
``settings.database_url`` from the ``POSTGRES_*`` env vars we set in the child
environment (derived by parsing ``TEST_DATABASE_URL``) — cleanly repointing the
migration at the scratch DB.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

# Postgres driver (psycopg 3) — already an installed dependency. Only imported /
# used inside the opt-in test body; the module import itself is cheap.
import psycopg
from sqlalchemy.engine import make_url

# The backend/ directory (this file lives in backend/tests/).
BACKEND_DIR = Path(__file__).resolve().parent.parent

# The single expected head after `alembic upgrade head`.
EXPECTED_HEAD = "0017"

# The exact picktype enum labels the schema must carry — exactly once, not
# duplicated. Mirrors app.models.PickType (kept literal here so the test does NOT
# import app.* — see module docstring).
EXPECTED_PICKTYPE_LABELS = {
    "UNDERDOG_COVER",
    "FAVORITE_COVER",
    "OVER",
    "UNDER",
    "MISC",
}

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def _child_env_from_url(url: str) -> dict[str, str]:
    """Build a child-process env that repoints app.config at ``url``.

    Parses the SQLAlchemy URL into parts and overlays POSTGRES_* on a copy of the
    current environment. The fresh subprocess re-imports app.config, which
    rebuilds settings.database_url from these (the in-process settings object is
    lru_cached and cannot be repointed — that is why we shell out).
    """
    parsed = make_url(url)
    env = os.environ.copy()
    env["POSTGRES_USER"] = parsed.username or ""
    env["POSTGRES_PASSWORD"] = parsed.password or ""
    env["POSTGRES_HOST"] = parsed.host or "localhost"
    env["POSTGRES_PORT"] = str(parsed.port or 5432)
    env["POSTGRES_DB"] = parsed.database or ""
    return env


def _psycopg_dsn(url: str) -> str:
    """Strip any SQLAlchemy ``+driver`` suffix so psycopg.connect accepts the URL."""
    parsed = make_url(url)
    # render_as_string keeps credentials; set_backend_name drops "+psycopg".
    return parsed.set(drivername="postgresql").render_as_string(hide_password=False)


def _alembic(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m alembic <args>`` from backend/ with the child env."""
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "TEST_DATABASE_URL is unset — opt-in PG/Alembic migration smoke test skipped. "
    "Run it via backend/scripts/run_pg_smoke.sh (stands up an isolated throwaway "
    "Postgres and runs the real `alembic upgrade head`).",
)
class PgMigrationSmokeTest(unittest.TestCase):
    """Runs real Alembic migrations against a scratch Postgres + asserts invariants."""

    child_env: dict[str, str]
    dsn: str

    @classmethod
    def setUpClass(cls) -> None:
        assert TEST_DATABASE_URL is not None  # guarded by skipUnless
        cls.child_env = _child_env_from_url(TEST_DATABASE_URL)
        cls.dsn = _psycopg_dsn(TEST_DATABASE_URL)

        # Run the REAL migration entrypoint (the same one the compose migrate
        # service runs) against the scratch DB. NOT create_all.
        result = _alembic(["upgrade", "head"], cls.child_env)
        if result.returncode != 0:
            raise AssertionError(
                "`alembic upgrade head` failed (rc="
                f"{result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    def setUp(self) -> None:
        # Each invariant gets a fresh connection; cleanup truncates the touched
        # tables so invariants do not interfere with one another.
        self.conn = psycopg.connect(self.dsn)
        # Cleanups run LIFO: register _truncate_touched FIRST so it runs LAST —
        # i.e. AFTER self.conn is closed. Otherwise a test that ends with an open
        # transaction on self.conn (e.g. a trailing SELECT) holds a lock that
        # blocks the TRUNCATE's ACCESS EXCLUSIVE lock forever (no lock_timeout) —
        # the cause of the original indefinite hang in invariant_c.
        self.addCleanup(self._truncate_touched)
        self.addCleanup(self.conn.close)

    def _truncate_touched(self) -> None:
        with psycopg.connect(self.dsn) as c:
            # Fail loudly instead of hanging if a prior test ever leaves a lock.
            c.execute("SET lock_timeout = '10s'")
            c.execute(
                "TRUNCATE pick_edit_audit, pick, game, week, team, users RESTART IDENTITY CASCADE"
            )
            c.commit()

    # --- helpers to insert the minimal NOT NULL columns each row requires ------

    def _insert_user(self, cur, *, display_name: str, discord_id: int | None) -> int:
        cur.execute(
            "INSERT INTO users (discord_id, display_name, is_admin, is_active, "
            "is_protected, created_at) VALUES (%s, %s, false, true, false, now()) "
            "RETURNING id",
            (discord_id, display_name),
        )
        return cur.fetchone()[0]

    def _insert_team(self, cur, *, espn_team_id: int, abbr: str) -> int:
        cur.execute(
            "INSERT INTO team (espn_team_id, abbreviation, display_name) "
            "VALUES (%s, %s, %s) RETURNING id",
            (espn_team_id, abbr, abbr),
        )
        return cur.fetchone()[0]

    def _insert_week(self, cur, *, season: int, week: int) -> int:
        cur.execute(
            "INSERT INTO week (season, week, lines_frozen) VALUES (%s, %s, false) RETURNING id",
            (season, week),
        )
        return cur.fetchone()[0]

    def _insert_game(self, cur, *, week_id: int, home: int, away: int, espn: int) -> int:
        cur.execute(
            "INSERT INTO game (espn_event_id, week_id, season, week, home_team_id, "
            "away_team_id, status, odds_frozen) "
            "VALUES (%s, %s, 2025, 1, %s, %s, 'SCHEDULED', false) RETURNING id",
            (espn, week_id, home, away),
        )
        return cur.fetchone()[0]

    def _insert_pick(self, cur, *, user_id: int, game_id: int, week_id: int) -> int:
        cur.execute(
            "INSERT INTO pick (user_id, game_id, week_id, pick_type, is_mortal_lock, "
            "result, points, created_at, updated_at) "
            "VALUES (%s, %s, %s, 'OVER', false, 'PENDING', 0, now(), now()) "
            "RETURNING id",
            (user_id, game_id, week_id),
        )
        return cur.fetchone()[0]

    # --- Invariant A: single head 0017 ----------------------------------------

    def test_invariant_a_single_head_0017(self) -> None:
        """`alembic heads`/`current` reports exactly one head and it is 0017."""
        heads = _alembic(["heads"], self.child_env)
        self.assertEqual(heads.returncode, 0, heads.stderr)
        head_lines = [ln for ln in heads.stdout.splitlines() if ln.strip()]
        self.assertEqual(len(head_lines), 1, f"expected exactly one head, got: {heads.stdout!r}")
        self.assertIn(EXPECTED_HEAD, heads.stdout, heads.stdout)

        current = _alembic(["current"], self.child_env)
        self.assertEqual(current.returncode, 0, current.stderr)
        self.assertIn(EXPECTED_HEAD, current.stdout, current.stdout)

    # --- Invariant B: one-null partial unique index ----------------------------

    def test_invariant_b_one_null_discord_id(self) -> None:
        """uq_users_one_null_discord_id allows ONE null discord_id, rejects a 2nd."""
        with self.conn.cursor() as cur:
            self._insert_user(cur, display_name="null-admin-1", discord_id=None)
            self.conn.commit()

        # A second NULL-discord_id row must violate the partial unique index.
        with self.assertRaises(psycopg.errors.UniqueViolation):
            with self.conn.cursor() as cur:
                self._insert_user(cur, display_name="null-admin-2", discord_id=None)
        self.conn.rollback()

        # Sanity: distinct NON-null discord_ids are fine.
        with self.conn.cursor() as cur:
            self._insert_user(cur, display_name="discord-user", discord_id=12345)
            self.conn.commit()

    # --- Invariant C: PickEditAudit FK cascades (migration 0013) ----------------

    def test_invariant_c_pick_edit_audit_cascades(self) -> None:
        """Deleting a referenced user cascades the audit row away (0013)."""
        with self.conn.cursor() as cur:
            admin = self._insert_user(cur, display_name="audit-admin", discord_id=111)
            target = self._insert_user(cur, display_name="audit-target", discord_id=222)
            home = self._insert_team(cur, espn_team_id=1, abbr="AAA")
            away = self._insert_team(cur, espn_team_id=2, abbr="BBB")
            week_id = self._insert_week(cur, season=2025, week=1)
            game_id = self._insert_game(cur, week_id=week_id, home=home, away=away, espn=900001)
            cur.execute(
                "INSERT INTO pick_edit_audit (admin_user_id, target_user_id, game_id, "
                "week_id, action, before_existed, game_was_final, created_at) "
                "VALUES (%s, %s, %s, %s, 'set', false, false, now())",
                (admin, target, game_id, week_id),
            )
            self.conn.commit()

        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pick_edit_audit")
            self.assertEqual(cur.fetchone()[0], 1)
            # Deleting the acting admin must cascade the audit row away (no FK error).
            cur.execute("DELETE FROM users WHERE id = %s", (admin,))
            self.conn.commit()
            cur.execute("SELECT count(*) FROM pick_edit_audit")
            self.assertEqual(cur.fetchone()[0], 0)

    # --- Invariant D: Pick.user_id cascade (QT-A) -------------------------------

    def test_invariant_d_pick_user_cascade(self) -> None:
        """Deleting a user with picks removes those picks (no FK error)."""
        with self.conn.cursor() as cur:
            user = self._insert_user(cur, display_name="picker", discord_id=333)
            home = self._insert_team(cur, espn_team_id=3, abbr="CCC")
            away = self._insert_team(cur, espn_team_id=4, abbr="DDD")
            week_id = self._insert_week(cur, season=2025, week=2)
            game_id = self._insert_game(cur, week_id=week_id, home=home, away=away, espn=900002)
            self._insert_pick(cur, user_id=user, game_id=game_id, week_id=week_id)
            self.conn.commit()

        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pick WHERE user_id = %s", (user,))
            self.assertEqual(cur.fetchone()[0], 1)
            cur.execute("DELETE FROM users WHERE id = %s", (user,))
            self.conn.commit()
            cur.execute("SELECT count(*) FROM pick WHERE user_id = %s", (user,))
            self.assertEqual(cur.fetchone()[0], 0)

    # --- Invariant E: single picktype enum with expected labels -----------------

    def test_invariant_e_picktype_enum_single_with_labels(self) -> None:
        """`picktype` enum exists exactly once with the five expected labels."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_type WHERE typname = 'picktype'")
            self.assertEqual(cur.fetchone()[0], 1, "picktype enum must exist exactly once")
            cur.execute(
                "SELECT enumlabel FROM pg_enum e "
                "JOIN pg_type t ON t.oid = e.enumtypid "
                "WHERE t.typname = 'picktype'"
            )
            labels = {row[0] for row in cur.fetchall()}
        self.assertEqual(labels, EXPECTED_PICKTYPE_LABELS)

    # --- Scoped 0017 down/up round-trip ----------------------------------------

    def test_invariant_f_scoped_0017_round_trip(self) -> None:
        """downgrade -1 (0017 -> 0016) then upgrade head both succeed.

        Proves 0017's additive CREATE TABLE (historical_game) reverses cleanly —
        downgrade drops the table and lands on 0016. Scoped to the single 0017 step
        ONLY — 0012's backfill is irreversible-by-design (see its docstring), so we
        do NOT downgrade further.
        """
        down = _alembic(["downgrade", "-1"], self.child_env)
        self.assertEqual(down.returncode, 0, f"downgrade -1 failed:\n{down.stderr}")

        current = _alembic(["current"], self.child_env)
        self.assertIn("0016", current.stdout, current.stdout)

        up = _alembic(["upgrade", "head"], self.child_env)
        self.assertEqual(up.returncode, 0, f"upgrade head failed:\n{up.stderr}")

        current2 = _alembic(["current"], self.child_env)
        self.assertIn(EXPECTED_HEAD, current2.stdout, current2.stdout)


if __name__ == "__main__":
    unittest.main()
