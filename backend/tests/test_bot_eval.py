"""Offline tests for ``scripts/bot_eval.py``: the case loader and the scoring. No LLM.

Run with: ``backend/.venv/bin/python -m unittest tests.test_bot_eval -v``
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import bot_eval


def _case(**fields) -> bot_eval.Case:
    base = {"id": "g.x", "question": "q", "samples": 1, "discord_id": 1, "asker_name": "a"}
    return bot_eval.Case(**{**base, **fields})


def _score(case: bot_eval.Case, answer: str | None = "The Bills lead 35-19.", **kwargs):
    args = {"intent": "open_nfl", "open_path": True, "tools": [], "degrade_lines": {"gave up"}}
    return bot_eval.score(case, answer=answer, **{**args, **kwargs})


class LoaderTests(unittest.TestCase):
    def test_the_shipped_cases_load_and_every_id_is_grouped(self) -> None:
        cases = bot_eval.load_cases()
        self.assertGreaterEqual(len(cases), 60)
        self.assertTrue(all("." in c.id for c in cases))

    def test_defaults_merge_and_single_values_become_lists(self) -> None:
        data = {
            "defaults": {"samples": 3, "discord_id": 7, "asker_name": "zed"},
            "cases": [{"id": "a.b", "question": "q", "tools_any": "t1", "samples": 1}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(data))
            (case,) = bot_eval.load_cases(path)
        self.assertEqual((case.samples, case.discord_id, case.asker_name), (1, 7, "zed"))
        self.assertEqual(case.tools_any, ["t1"])
        self.assertEqual(case.group, "a")

    def test_an_unknown_key_or_a_duplicate_id_is_refused(self) -> None:
        for cases in (
            [{"id": "a.b", "question": "q", "tool_any": "t"}],
            [{"id": "a.b", "question": "q"}, {"id": "a.b", "question": "q"}],
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "cases.json"
                path.write_text(json.dumps({"cases": cases}))
                with self.assertRaises(ValueError):
                    bot_eval.load_cases(path)

    def test_a_placeholder_without_a_value_is_none(self) -> None:
        self.assertEqual(bot_eval.fill("week {open_week}", {"open_week": 7}), "week 7")
        self.assertIsNone(bot_eval.fill("week {open_week}", {"open_week": None}))
        self.assertEqual(bot_eval.fill("no placeholder", {}), "no placeholder")


class ScoreTests(unittest.TestCase):
    def test_a_clean_answer_passes(self) -> None:
        case = _case(expect_intent=["open_nfl"], tools_any=["t1"], must_match=["35"])
        self.assertEqual(_score(case, tools=["t1"]), [])

    def test_no_answer_and_a_degrade_line_fail(self) -> None:
        self.assertEqual(_score(_case(), answer=None), ["no answer"])
        self.assertEqual(_score(_case(), answer="gave up"), ["degrade line"])
        self.assertEqual(_score(_case(allow_degrade=True), answer="gave up"), [])

    def test_each_check_names_its_own_reason(self) -> None:
        case = _case(
            expect_intent=["weather"],
            open_path=False,
            tools_any=["t1"],
            tools_all=["t2"],
            tools_none=["t3"],
            must_match=["Dolphins"],
            must_not_match=["35"],
        )
        reasons = _score(case, tools=["t3"])
        self.assertEqual(len(reasons), 7)
        self.assertIn("intent open_nfl not in ['weather']", reasons)
        self.assertIn("forbidden tools called ['t3']", reasons)

    def test_no_tools_fails_on_any_call(self) -> None:
        self.assertEqual(_score(_case(no_tools=True), tools=["t1"]), ["tools called ['t1']"])

    def test_the_summary_rates_every_sample(self) -> None:
        report = {
            "seconds": 1.0,
            "skipped": {"k.web": "needs SEARXNG_URL"},
            "results": [
                {"case": "g.a", "sample": 0, "passed": True, "reasons": []},
                {"case": "g.a", "sample": 1, "passed": False, "reasons": ["no answer"]},
                {"case": "h.b", "sample": 0, "passed": True, "reasons": []},
            ],
        }
        table, rate = bot_eval.summarize(report)
        self.assertAlmostEqual(rate, 2 / 3)
        self.assertIn("FAIL 1/2  g.a", table)
        self.assertIn("skipped k.web: needs SEARXNG_URL", table)


if __name__ == "__main__":
    unittest.main()
