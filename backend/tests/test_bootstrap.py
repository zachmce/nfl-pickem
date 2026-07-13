"""Offline unit tests for the shell-free migrate/seed entrypoint.

These tests exercise :mod:`app.bootstrap` with EVERY side-effecting call patched
out — no Postgres, no live Alembic migrations, no network. They prove the
Python entrypoint preserves the retired ``sh -c "alembic upgrade head && ... "``
chain's two guarantees:

* **strict step order** — migrations, then teams -> historical_games -> demo
  -> admins, and
* **fail-fast short-circuit** — the first step that raises propagates and no
  later step runs (mirroring ``A && B && C && D``).

plus that :func:`app.bootstrap._alembic_config` resolves ``script_location`` by
package location (an existing dir with ``alembic.ini`` alongside), proving the
CWD-independent path resolution without touching a DB.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && .venv/bin/python -m unittest tests.test_bootstrap -v
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from app import bootstrap


class BootstrapOrderTests(unittest.TestCase):
    """Step order + fail-fast short-circuit, all side effects patched out."""

    def test_runs_all_steps_in_order(self) -> None:
        # A single parent Mock records the ABSOLUTE call order across all steps.
        parent = mock.Mock()
        with (
            mock.patch.object(bootstrap, "configure_logging"),
            mock.patch.object(bootstrap, "run_migrations") as m_migrate,
            mock.patch("app.seeds.teams.main") as m_teams,
            mock.patch("app.seeds.historical_games.main") as m_historical,
            mock.patch("app.seeds.demo.main") as m_demo,
            mock.patch("app.seeds.admins.main") as m_admins,
        ):
            parent.attach_mock(m_migrate, "migrate")
            parent.attach_mock(m_teams, "teams")
            parent.attach_mock(m_historical, "historical_games")
            parent.attach_mock(m_demo, "demo")
            parent.attach_mock(m_admins, "admins")

            bootstrap.main()

        self.assertEqual(
            parent.mock_calls,
            [
                mock.call.migrate(),
                mock.call.teams(),
                mock.call.historical_games(),
                mock.call.demo(),
                mock.call.admins(),
            ],
        )

    def test_migration_failure_skips_all_seeds(self) -> None:
        with (
            mock.patch.object(bootstrap, "configure_logging"),
            mock.patch.object(bootstrap, "run_migrations", side_effect=RuntimeError("boom")),
            mock.patch("app.seeds.teams.main") as m_teams,
            mock.patch("app.seeds.historical_games.main") as m_historical,
            mock.patch("app.seeds.demo.main") as m_demo,
            mock.patch("app.seeds.admins.main") as m_admins,
        ):
            with self.assertRaises(RuntimeError):
                bootstrap.main()

            m_teams.assert_not_called()
            m_historical.assert_not_called()
            m_demo.assert_not_called()
            m_admins.assert_not_called()

    def test_seed_failure_short_circuits(self) -> None:
        # Migrations succeed; teams then raises -> historical_games/demo/admins
        # never run.
        with (
            mock.patch.object(bootstrap, "configure_logging"),
            mock.patch.object(bootstrap, "run_migrations"),
            mock.patch("app.seeds.teams.main", side_effect=RuntimeError("seed boom")),
            mock.patch("app.seeds.historical_games.main") as m_historical,
            mock.patch("app.seeds.demo.main") as m_demo,
            mock.patch("app.seeds.admins.main") as m_admins,
        ):
            with self.assertRaises(RuntimeError):
                bootstrap.main()

            m_historical.assert_not_called()
            m_demo.assert_not_called()
            m_admins.assert_not_called()


class BootstrapConfigTests(unittest.TestCase):
    """Config path resolution — no DB, no migration run."""

    def test_alembic_config_resolves_by_package_location(self) -> None:
        cfg = bootstrap._alembic_config()
        script_location = cfg.get_main_option("script_location")

        self.assertIsNotNone(script_location)
        assert script_location is not None  # narrow for the type checker
        self.assertTrue(os.path.isdir(script_location), script_location)
        ini = os.path.join(os.path.dirname(script_location), "alembic.ini")
        self.assertTrue(os.path.exists(ini), ini)


if __name__ == "__main__":
    unittest.main()
