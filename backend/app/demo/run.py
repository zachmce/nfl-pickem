"""Non-prod-gated CLI entrypoint for the demo season walkthrough (#5).

Thin shell over :mod:`app.demo.driver` — it wires a throwaway demo database +
the default :class:`~app.scoreboard.demo.Demo2025Source` factory and runs the
weeks-1-3 walkthrough in one of two modes:

* ``--assert`` — the runnable INTEGRATION PROOF. Runs the full walkthrough with
  ``assert_oracle=True``; exits 0 when the DB-sourced standings equal the oracle
  for every week, and exits NON-ZERO (the
  :class:`~app.demo.driver.WalkthroughAssertionError`) on any mismatch. This is
  the "tests pass but prod breaks" hedge made executable in CI/by hand.
* narrated (default) — runs the same core and prints each week's cumulative
  standings (display_name + per-week + season totals) human-readably for a
  stakeholder, then exits 0.

Non-prod gate (the security boundary — see ``<threat_model>`` T-qqm-01)
----------------------------------------------------------------------

The demo writes a whole fake 2025 season; pointing it at the production database
would corrupt real data. The gate is therefore strict and explicit:

* An explicit demo DB URL is **mandatory** — passed via ``--demo-db`` or the
  ``DEMO_DATABASE_URL`` env var. Omitting it is a hard ``SystemExit`` (non-zero).
* A demo URL equal to the resolved production URL (``settings.database_url``) is
  **rejected** with a non-zero ``SystemExit`` — you cannot launder the prod DB in
  through the demo flag.
* This entrypoint builds its OWN engine/``Session`` from the explicit demo URL
  and **never** imports/uses ``app.db.engine`` or ``app.db.task_session`` (the
  shared production engine). It reads ``settings.database_url`` ONLY to compare
  against (to reject it), never to connect.

``app.config`` is imported lazily inside the gate so merely importing this module
never constructs Settings; the offline gate tests pass a sentinel prod URL
directly and assert on the exception without opening any real connection.

> Note: on this machine there is no bare ``python`` on ``PATH``; use the venv
> interpreter ``.venv/bin/python`` for any commands.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from sqlmodel import Session, SQLModel, create_engine

from app.demo.driver import (
    DEFAULT_WEEKS,
    WalkthroughAssertionError,
    WalkthroughResult,
    run_walkthrough,
    setup,
)
from app.scoreboard.demo import Demo2025Source

# The env var the entrypoint reads directly for the demo DB target (NOT
# app.config / settings.database_url — that is the prod URL we reject against).
DEMO_DB_ENV = "DEMO_DATABASE_URL"


def _resolved_prod_url() -> str:
    """The resolved production database URL — imported lazily.

    Read ONLY so the gate can REJECT a demo target equal to it; this entrypoint
    never connects to it. Imported inside the function so importing this module
    has no side effect of constructing Settings / the prod engine.
    """
    from app.config import settings

    return settings.database_url


def require_demo_db(url: str | None, *, prod_url: str | None = None) -> str:
    """Validate + return the explicit demo DB URL, or raise ``SystemExit``.

    The non-prod gate. Rejects with a non-zero :class:`SystemExit` when:

    * ``url`` is missing/blank — an explicit demo DB target is mandatory; or
    * ``url`` equals the resolved production URL (``prod_url``, defaulting to
      ``settings.database_url``) — the prod DB may never be the demo target.

    Returns the validated demo URL on success. ``prod_url`` is injectable so the
    offline gate tests pass a sentinel without importing real Settings.
    """
    if not url or not url.strip():
        raise SystemExit(
            "refusing to run: an explicit demo DB URL is required "
            f"(--demo-db or {DEMO_DB_ENV}=...). The demo never uses the default "
            "app engine / production database."
        )
    target = url.strip()
    resolved_prod = prod_url if prod_url is not None else _resolved_prod_url()
    if target == resolved_prod:
        raise SystemExit(
            "refusing to run against the production database: the demo DB URL "
            "equals the resolved production database_url. Point --demo-db at a "
            "throwaway demo database instead."
        )
    return target


def _print_narrated(result: WalkthroughResult) -> None:
    """Print each completed week's cumulative standings human-readably."""
    for week in sorted(result.snapshots):
        standings = result.snapshots[week]
        print(f"\n=== Standings after week {week} ===")
        for rank, r in enumerate(standings.results, start=1):
            weekly = ", ".join(
                f"wk{w}={r.weekly_scores[w]}"
                for w in sorted(r.weekly_scores)
            )
            print(
                f"  {rank}. {r.display_name:<12} "
                f"season={r.season_total:<4} [{weekly}]"
            )


def _build_engine(demo_url: str):
    """Build a fresh engine for the explicit demo DB target (never app.db.engine)."""
    return create_engine(demo_url)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.demo.run",
        description=(
            "Run the demo 2025 season walkthrough against a throwaway demo "
            "database. Requires an explicit demo DB URL; refuses the prod DB."
        ),
    )
    parser.add_argument(
        "--demo-db",
        default=os.environ.get(DEMO_DB_ENV),
        help=(
            "explicit demo database URL (or set "
            f"{DEMO_DB_ENV}). The prod database_url is rejected."
        ),
    )
    parser.add_argument(
        "--assert",
        dest="assert_mode",
        action="store_true",
        help="integration-proof mode: exit non-zero on actual!=oracle mismatch",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        default=list(DEFAULT_WEEKS),
        help="weeks to walk through (default: 1 2 3)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: gate the demo DB, run the walkthrough, return an exit code.

    In ``--assert`` mode a :class:`WalkthroughAssertionError` (actual != oracle)
    is caught and turned into a non-zero exit code — the runnable proof's failure
    signal. In narrated mode the per-week standings are printed and 0 returned.
    """
    args = _parse_args(argv)
    demo_url = require_demo_db(args.demo_db)

    engine = _build_engine(demo_url)
    SQLModel.metadata.create_all(engine)

    weeks = tuple(args.weeks)
    try:
        with Session(engine) as session:
            setup(session)
            result = run_walkthrough(
                session,
                weeks=weeks,
                source_factory=lambda offset: Demo2025Source(offset),
                assert_oracle=args.assert_mode,
            )
    except WalkthroughAssertionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    if args.assert_mode:
        print(f"PASS: weeks {list(weeks)} actual == oracle.")
        return 0

    _print_narrated(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
