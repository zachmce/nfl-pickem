"""Shell-free migrate/seed one-shot entrypoint (the retired ``sh -c`` chain).

This module replaces the compose ``migrate`` command that used to be::

    sh -c "alembic upgrade head \\
        && python -m app.seeds.teams \\
        && python -m app.seeds.demo \\
        && python -m app.seeds.admins"

Running it as ``python -m app.bootstrap`` removes the ONLY shell dependency in
any backend container command, which is Step 1 of the Chainguard/Wolfi base
migration: a distroless-style runtime image ships no ``sh``, so the ``&&``
chaining has to move into Python (see
``.planning/notes/backend-base-chainguard-decision.md``). This ships
independently and stays fully working on the current Debian ``python:3.14-slim``
image — no base-image change here.

Fail-fast contract (preserves the old ``A && B && C && D`` short-circuit): the
steps run in strict order — migrations, then ``teams`` -> ``demo`` -> ``admins``
— and NOTHING catches, so the first step that raises propagates out of
:func:`main` (non-zero exit) and no later step runs. The seed behaviors
themselves are unchanged: this module never reimplements seed logic, it calls
each existing ``main()`` entrypoint, so the teams upsert, the demo env-gate +
anchor-idempotency, and the admin self-gate are byte-identical to before.

Run it from the ``backend/`` directory::

    cd backend
    .venv/bin/python -m app.bootstrap
"""

from __future__ import annotations

from pathlib import Path

import structlog
from alembic import command
from alembic.config import Config

from app.config import settings
from app.logging_config import configure_logging
from app.seeds import admins, demo, teams

logger = structlog.get_logger(__name__)

# Backend root = parent of the ``app`` package dir (this file is app/bootstrap.py,
# so parent.parent is backend/). ``alembic.ini`` and the ``alembic/`` script dir
# both live there. The bare ``alembic`` CLI finds them by relying on WORKDIR /app
# as the CWD; resolving from ``__file__`` replicates that WITHOUT assuming any CWD,
# so ``command.upgrade`` works no matter where the process was launched from.
_ROOT = Path(__file__).resolve().parent.parent


def _alembic_config() -> Config:
    """Build an Alembic ``Config`` whose paths resolve by package location.

    Loads ``backend/alembic.ini`` and then explicitly pins ``script_location`` to
    the absolute ``backend/alembic`` dir, so migrations resolve the same versions
    directory the bare CLI does under WORKDIR /app — independent of the process
    CWD. ``sqlalchemy.url`` is left untouched: ``alembic/env.py`` already injects
    it at runtime from ``settings.database_url``.
    """
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    return cfg


def run_migrations() -> None:
    """Upgrade the schema to head via Alembic's Python API (no subprocess)."""
    logger.info("bootstrap_migrations_start")
    command.upgrade(_alembic_config(), "head")
    logger.info("bootstrap_migrations_done")


def main() -> None:
    """Migrate then seed, fail-fast — the shell-free ``migrate`` one-shot.

    Configures logging ONCE (the repo's process-entrypoint convention, mirroring
    ``app.bot.client.main``), then runs, in strict order with NO exception
    swallowing: migrations -> ``teams.main`` -> ``demo.main`` -> ``admins.main``.
    Because nothing catches, the first failing step propagates and later steps
    never run — exactly mirroring the retired ``A && B && C && D`` chain.
    """
    configure_logging(settings.log_level)

    run_migrations()
    teams.main()
    demo.main()
    admins.main()


if __name__ == "__main__":
    main()
