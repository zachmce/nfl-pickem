"""Offline tests for the pure results oracle.

These tests exercise :mod:`app.demo.oracle` against the real 2025 fixture, fully
offline (in-memory SQLite, no Postgres, no network, no ``app.db`` import). They
seed teams + import the fixture to get real FINAL ``Game`` rows, then assert:

* the oracle produces per-bot weekly scores + season totals for every bot, with
  standings ordered by season total descending and a deterministic tie-break;
* the oracle delegates scoring to ``scoring.score_week`` (cross-checked directly);
* a partial-roster bot scores only its present picks;
* two **hand-anchored** (bot, week) totals — computed by hand below from the real
  2025 outcomes — match the oracle exactly.

Run from the ``backend/`` directory::

    cd backend && .venv/bin/python -m unittest tests.test_results_oracle -v
"""

from __future__ import annotations

import unittest

from sqlmodel import Session, SQLModel, create_engine, select

from app.demo.oracle import compute_standings, games_by_pk_index
from app.models import Game, Pick
from app.seeds.data.bot_picks_2025 import BOT_PICKS
from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.scoring import score_week


class ResultsOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _seed(self, session: Session) -> list[Game]:
        seed_teams(session)
        import_fixture_2025(session)
        return list(session.exec(select(Game)).all())

    def test_produces_results_for_every_bot_ordered(self) -> None:
        with Session(self.engine) as session:
            games = self._seed(session)
            standings = compute_standings(BOT_PICKS, games)

            names = {r.display_name for r in standings.results}
            self.assertEqual(names, set(BOT_PICKS.keys()))

            for r in standings.results:
                # Weekly scores present for every week the bot has picks for.
                self.assertEqual(
                    set(r.weekly_scores.keys()),
                    set(BOT_PICKS[r.display_name].keys()),
                )
                self.assertEqual(r.season_total, sum(r.weekly_scores.values()))

            # Ordered by (-season_total, display_name).
            ordered = sorted(standings.results, key=lambda r: (-r.season_total, r.display_name))
            self.assertEqual(list(standings.results), ordered)

    def test_delegates_to_score_week(self) -> None:
        # The oracle's weekly score must equal a direct score_week call built
        # the same way — proving it delegates rather than re-implements.
        with Session(self.engine) as session:
            games = self._seed(session)
            by_event = {g.espn_event_id: g for g in games}
            games_by_pk = games_by_pk_index(games)
            standings = compute_standings(BOT_PICKS, games)
            by_name = {r.display_name: r for r in standings.results}

            for name, weeks in BOT_PICKS.items():
                for wk, bot_picks in weeks.items():
                    picks = [
                        Pick(
                            user_id=1,
                            game_id=by_event[bp.espn_event_id].id,
                            week_id=by_event[bp.espn_event_id].week_id,
                            pick_type=bp.pick_type,
                            is_mortal_lock=bp.is_mortal_lock,
                        )
                        for bp in bot_picks
                    ]
                    expected = score_week(games_by_pk, picks)
                    self.assertEqual(by_name[name].weekly_scores[wk], expected)

    def test_partial_roster_scores_only_present_picks(self) -> None:
        # bot_dave week 2 is the partial roster (3 picks). Its oracle score must
        # equal a direct score_week over exactly those 3 picks.
        with Session(self.engine) as session:
            games = self._seed(session)
            by_event = {g.espn_event_id: g for g in games}
            games_by_pk = games_by_pk_index(games)

            partial = BOT_PICKS["bot_dave"][2]
            self.assertEqual(len(partial), 3)
            picks = [
                Pick(
                    user_id=1,
                    game_id=by_event[bp.espn_event_id].id,
                    week_id=by_event[bp.espn_event_id].week_id,
                    pick_type=bp.pick_type,
                    is_mortal_lock=bp.is_mortal_lock,
                )
                for bp in partial
            ]
            expected = score_week(games_by_pk, picks)

            standings = compute_standings(BOT_PICKS, games)
            dave = next(r for r in standings.results if r.display_name == "bot_dave")
            self.assertEqual(dave.weekly_scores[2], expected)

    def test_hand_anchored_spot_checks(self) -> None:
        # Hand-computed expected weekly totals from the real 2025 outcomes.
        #
        # bot_alice, week 1 (all lines fractional -> no PUSH possible):
        #   401772510 UNDERDOG_COVER  PHI(-7.5) won by 4 -> dog covers   WIN  +1
        #   401772714 FAVORITE_COVER  KC(-3.5)  lost by 6 -> no cover     LOSS  0
        #   401772830 OVER (47.5)     combined 43 < 47.5                  LOSS  0
        #   401772829 UNDER (47.5)    combined 33 < 47.5                  WIN  +1
        #   401772719 FAVORITE_COVER lock IND(-1.5) won by 25 -> cover    WIN  +2
        #   ------------------------------------------------------------- = 4
        #
        # bot_dave, week 2 (partial — 3 picks):
        #   401772936 UNDER (47.5)    combined 45 < 47.5                  WIN  +1
        #   401772725 OVER (50.5)     combined 58 > 50.5                  WIN  +1
        #   401772724 FAVORITE_COVER lock LAR(-5.5) won by 14 -> cover    WIN  +2
        #   ------------------------------------------------------------- = 4
        with Session(self.engine) as session:
            games = self._seed(session)
            standings = compute_standings(BOT_PICKS, games)
            by_name = {r.display_name: r for r in standings.results}

            self.assertEqual(by_name["bot_alice"].weekly_scores[1], 4)
            self.assertEqual(by_name["bot_dave"].weekly_scores[2], 4)

    def test_oracle_is_deterministic_and_pure(self) -> None:
        with Session(self.engine) as session:
            games = self._seed(session)
            first = compute_standings(BOT_PICKS, games)
            second = compute_standings(BOT_PICKS, games)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
