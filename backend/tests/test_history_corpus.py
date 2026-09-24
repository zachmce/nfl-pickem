"""Offline tests for the static history corpus (issue #248 item 11): the committed CSV
files, their readers and the championships / Hall of Fame / awards tools over them.

Run with: ``backend/.venv/bin/python -m unittest tests.test_history_corpus -v``
"""

from __future__ import annotations

import asyncio
import unittest

from app.bot import qa_open
from app.services import history_corpus


def _run(coro):
    return asyncio.run(coro)


class CorpusFileTests(unittest.TestCase):
    """The committed files hold what the generator promised, with no parse debris."""

    def test_every_super_bowl_and_the_early_champions_are_present(self) -> None:
        rows = history_corpus.championships()
        bowls = [r for r in rows if r["game"].startswith("Super Bowl")]
        self.assertGreaterEqual(len(bowls), 60)
        self.assertEqual(bowls[0]["winner"], "Green Bay Packers")  # no AFL/NFL marker
        self.assertEqual(rows[0]["season"], "1920")
        for r in rows:
            self.assertNotRegex(r["winner"] + r["loser"], r"[\[\]*^†‡~]|\(\d")

    def test_co_winners_are_one_row_each(self) -> None:
        self.assertEqual(
            sorted(r["winner"] for r in history_corpus.awards(season=1997, award="mvp")),
            ["Barry Sanders", "Brett Favre"],
        )

    def test_a_super_bowl_mvp_belongs_to_the_season_before_the_game(self) -> None:
        self.assertEqual(
            history_corpus.awards(season=2012, award="super bowl mvp")[0]["winner"], "Joe Flacco"
        )


class CorpusReaderTests(unittest.TestCase):
    def test_franchise_follows_old_names_and_aliases(self) -> None:
        self.assertEqual(history_corpus.franchise("Baltimore Colts"), "IND")
        self.assertEqual(history_corpus.franchise("oak"), "LV")
        self.assertEqual(history_corpus.franchise("was"), "WSH")
        self.assertIsNone(history_corpus.franchise("Akron Pros"))
        self.assertIsNone(history_corpus.franchise(""))

    def test_title_counts_span_every_name_of_a_franchise(self) -> None:
        self.assertEqual(
            history_corpus.championship_record("IND"),
            {"super_bowls_won": 2, "super_bowls_lost": 2, "nfl_titles_before_1966": 3},
        )
        self.assertEqual(history_corpus.championship_record("CHI")["nfl_titles_before_1966"], 8)
        self.assertEqual(history_corpus.championship_record("TEN")["super_bowls_lost"], 1)

    def test_hall_of_fame_matches_every_word_of_a_name(self) -> None:
        white = history_corpus.hall_of_fame(player="reggie white")
        self.assertEqual([r["class"] for r in white], ["2006"])
        self.assertEqual(white[0]["first_year_of_eligibility"], "yes")
        self.assertEqual(history_corpus.hall_of_fame(player="white reggie"), white)
        self.assertEqual(history_corpus.hall_of_fame(player="Patrick Mahomes"), [])


class ChampionshipsToolTests(unittest.TestCase):
    def test_a_team_gets_its_count_and_every_title_game(self) -> None:
        body = _run(qa_open._lookup_championships(team="pit"))
        assert isinstance(body, dict)
        self.assertEqual(body["record"]["super_bowls_won"], 6)
        self.assertIn("won 6 Super Bowls and lost 2", body["championships_statement"])
        self.assertTrue(any("Super Bowl XLIII" in g for g in body["games"]))
        self.assertIn("no AFL championship", body["caveat"])

    def test_a_season_and_no_argument(self) -> None:
        body = _run(qa_open._lookup_championships(season=1958))
        assert isinstance(body, dict)
        self.assertEqual(len(body["games"]), 1)
        self.assertIn("Baltimore Colts beat the New York Giants 23-17", body["games"][0])
        every = _run(qa_open._lookup_championships())
        assert isinstance(every, dict)
        self.assertTrue(all("Super Bowl" in g for g in every["games"]))

    def test_misses_are_notes(self) -> None:
        body = _run(qa_open._lookup_championships(team="Akron"))
        self.assertEqual(body, {"note": qa_open._UNKNOWN_ATS_TEAM_NOTE.format(team="AKRON")})
        none = _run(qa_open._lookup_championships(season=1800))
        assert isinstance(none, dict)
        self.assertIn("never name a title game from your own memory", none["note"])


class HallOfFameToolTests(unittest.TestCase):
    def test_a_player_is_found_with_his_class_and_teams(self) -> None:
        body = _run(qa_open._lookup_hall_of_fame(player="Peyton Manning"))
        assert isinstance(body, dict)
        self.assertEqual(body["inductees"][0]["class"], "2021")
        self.assertIn("Denver Broncos", body["inductees"][0]["teams"])

    def test_a_long_team_list_drops_the_detail(self) -> None:
        body = _run(qa_open._lookup_hall_of_fame(team="CHI"))
        assert isinstance(body, dict)
        self.assertGreater(len(body["inductees"]), qa_open._HOF_DETAIL_LIMIT)
        self.assertNotIn("teams", body["inductees"][0])

    def test_misses_are_notes(self) -> None:
        self.assertEqual(_run(qa_open._lookup_hall_of_fame()), {"note": qa_open._NO_HOF_QUERY_NOTE})
        body = _run(qa_open._lookup_hall_of_fame(player="Patrick Mahomes"))
        assert isinstance(body, dict)
        self.assertIn("is not in the Pro Football Hall of Fame", body["note"])


class AwardsCorpusToolTests(unittest.TestCase):
    def test_a_player_without_a_season_counts_his_wins(self) -> None:
        body = _run(qa_open._lookup_season_awards(player="Peyton Manning", award="mvp"))
        assert isinstance(body, dict)
        self.assertEqual(
            [w["season"] for w in body["wins"]], ["2003", "2004", "2008", "2009", "2013"]
        )

    def test_a_season_before_espn_reads_only_the_tables(self) -> None:
        body = _run(qa_open._lookup_season_awards(season=1957, award="mvp"))
        assert isinstance(body, dict)
        self.assertEqual(body["awards"][0]["source"], "AP award tables")
        self.assertEqual(body["awards"][0]["winners"][0]["winner"], "Jim Brown")

    def test_an_unknown_player_is_a_note_that_names_the_covered_awards(self) -> None:
        body = _run(qa_open._lookup_season_awards(player="Nobody Atall"))
        assert isinstance(body, dict)
        self.assertIn("Super Bowl MVP", body["note"])
        self.assertIn("never add an award", body["note"])


if __name__ == "__main__":
    unittest.main()
