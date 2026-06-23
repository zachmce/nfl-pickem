"""Offline unit tests for the normalized scoreboard data contract.

These exercise the frozen dataclasses in :mod:`app.scoreboard.types` and the
:class:`~app.scoreboard.port.ScoreboardSource` port shape. They touch no
database and no network — they only construct value objects and inspect them.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_scoreboard_types -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m unittest ...`` or the venv
> interpreter ``.venv/bin/python -m unittest ...``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import dataclasses
import unittest

from app.models import GameStatus
from app.scoreboard.port import ScoreboardFetchError, ScoreboardSource
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds, ScoreboardTeam


class ScoreboardTypesTest(unittest.TestCase):
    def test_team_is_frozen(self) -> None:
        team = ScoreboardTeam(espn_team_id="21", abbreviation="PHI", score=24)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            team.score = 0  # type: ignore[misc]

    def test_team_score_defaults_to_none(self) -> None:
        team = ScoreboardTeam(espn_team_id="6", abbreviation="DAL")
        self.assertIsNone(team.score)

    def test_odds_is_frozen(self) -> None:
        odds = ScoreboardOdds(provider="DraftKings", spread=-3.5, total=44.5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            odds.spread = 0.0  # type: ignore[misc]

    def test_odds_carries_raw_signed_spread(self) -> None:
        # The port type stores the raw home-relative signed value as-is; it does
        # not take abs() — that is the importer's/poller's job (out of scope).
        odds = ScoreboardOdds(
            provider="ESPN BET",
            spread=-7.5,
            total=47.5,
            favorite_team_id="21",
            underdog_team_id="6",
        )
        self.assertEqual(odds.spread, -7.5)
        self.assertEqual(odds.favorite_team_id, "21")
        self.assertEqual(odds.underdog_team_id, "6")

    def test_game_is_frozen(self) -> None:
        game = ScoreboardGame(
            espn_event_id="401772510",
            season=2025,
            week=1,
            kickoff_at=None,
            status=GameStatus.SCHEDULED,
            home=ScoreboardTeam(espn_team_id="21", abbreviation="PHI"),
            away=ScoreboardTeam(espn_team_id="6", abbreviation="DAL"),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            game.status = GameStatus.FINAL  # type: ignore[misc]

    def test_game_odds_defaults_to_none(self) -> None:
        game = ScoreboardGame(
            espn_event_id="1",
            season=2025,
            week=1,
            kickoff_at=None,
            status=GameStatus.SCHEDULED,
            home=ScoreboardTeam(espn_team_id="1", abbreviation="AAA"),
            away=ScoreboardTeam(espn_team_id="2", abbreviation="BBB"),
        )
        self.assertIsNone(game.odds)

    def test_game_can_carry_odds_block(self) -> None:
        odds = ScoreboardOdds(provider="DraftKings", spread=-3.5, total=44.5)
        game = ScoreboardGame(
            espn_event_id="2",
            season=2025,
            week=2,
            kickoff_at=None,
            status=GameStatus.SCHEDULED,
            home=ScoreboardTeam(espn_team_id="26", abbreviation="SEA"),
            away=ScoreboardTeam(espn_team_id="17", abbreviation="NE"),
            odds=odds,
        )
        self.assertIs(game.odds, odds)
        self.assertEqual(game.status, GameStatus.SCHEDULED)

    def test_fetch_error_is_runtimeerror(self) -> None:
        self.assertTrue(issubclass(ScoreboardFetchError, RuntimeError))

    def test_port_is_runtime_checkable(self) -> None:
        # A trivial conforming object should satisfy the runtime_checkable
        # Protocol structurally (no inheritance required).
        class _Fake:
            def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
                return []

        self.assertIsInstance(_Fake(), ScoreboardSource)

        class _NotConforming:
            pass

        self.assertNotIsInstance(_NotConforming(), ScoreboardSource)


if __name__ == "__main__":
    unittest.main()
