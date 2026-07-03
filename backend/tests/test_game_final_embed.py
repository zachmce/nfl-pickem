"""Offline unit tests for the pure game.final embed builder (260703-piv).

The builder is Discord-send-free: it constructs a ``discord.Embed`` but takes no
client and performs no send, so it is fully unit-testable without a live gateway.
The application-emoji cache is populated directly via ``populate_emoji_cache`` /
``reset_emoji_cache`` with fake ``<:name:id>`` strings for the logo assertions.

Run with: ``backend/.venv/bin/python -m unittest tests.test_game_final_embed -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, cast

from app.bot.game_final_embed import (
    build_game_final_embed,
    build_impact_fields,
    build_score_line,
    select_winner_color,
)
from app.bot.team_emoji import (
    NEUTRAL_EMBED_COLOR,
    populate_emoji_cache,
    reset_emoji_cache,
    resolve_team_color,
)
from app.services.notifications import GameFinalImpact, game_final_event


@dataclass
class _FakeEmoji:
    name: str
    id: int

    def __str__(self) -> str:
        return f"<:{self.name}:{self.id}>"


def _event(
    *,
    week: int = 3,
    away_abbr: str = "LAC",
    home_abbr: str = "KC",
    away_score: int = 20,
    home_score: int = 27,
    impacts: list[dict[str, Any]] | None = None,
) -> dict:
    return game_final_event(
        week=week,
        away_abbr=away_abbr,
        home_abbr=home_abbr,
        away_score=away_score,
        home_score=home_score,
        impacts=cast("list[GameFinalImpact]", impacts or []),
    )


class BuildScoreLineTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_emoji_cache()

    def tearDown(self) -> None:
        reset_emoji_cache()

    def test_away_before_home_with_logos(self) -> None:
        populate_emoji_cache([_FakeEmoji("chiefs", 7), _FakeEmoji("chargerslogo", 8)])
        line = build_score_line(_event())
        # Away (LAC) appears before home (KC) — order fixed by us, not the LLM.
        self.assertLess(line.index("LAC"), line.index("KC"))
        self.assertIn("<:chargerslogo:8>", line)
        self.assertIn("<:chiefs:7>", line)
        self.assertIn("20", line)
        self.assertIn("27", line)

    def test_uncached_logo_falls_back_to_bare_abbr(self) -> None:
        # KC cached, LAC not — LAC renders with no logo, never raises.
        populate_emoji_cache([_FakeEmoji("chiefs", 7)])
        line = build_score_line(_event())
        self.assertIn("<:chiefs:7>", line)
        self.assertIn("LAC", line)
        # No stray emoji token for the uncached away side.
        self.assertEqual(line.count("<:"), 1)

    def test_empty_cache_renders_bare_abbrs(self) -> None:
        line = build_score_line(_event())
        self.assertNotIn("<:", line)
        self.assertIn("LAC", line)
        self.assertIn("KC", line)


class SelectWinnerColorTests(unittest.TestCase):
    def test_home_win_uses_home_color(self) -> None:
        color = select_winner_color(_event(away_score=20, home_score=27))
        self.assertEqual(color, resolve_team_color("KC"))

    def test_away_win_uses_away_color(self) -> None:
        color = select_winner_color(_event(away_score=30, home_score=27))
        self.assertEqual(color, resolve_team_color("LAC"))

    def test_tie_uses_neutral(self) -> None:
        color = select_winner_color(_event(away_score=21, home_score=21))
        self.assertEqual(color, NEUTRAL_EMBED_COLOR)

    def test_unknown_winner_abbr_uses_neutral(self) -> None:
        color = select_winner_color(
            _event(away_abbr="ZZZ", home_abbr="YYY", away_score=30, home_score=27)
        )
        self.assertEqual(color, NEUTRAL_EMBED_COLOR)


class BuildImpactFieldsTests(unittest.TestCase):
    @staticmethod
    def _impact(username, outcome, lock=False):
        return {"username": username, "outcome": outcome, "was_mortal_lock": lock}

    def test_no_impacts_yields_zero_fields(self) -> None:
        self.assertEqual(build_impact_fields([]), [])

    def test_only_busts_yields_single_busted_field(self) -> None:
        fields = build_impact_fields([self._impact("bob", "busted")])
        self.assertEqual(len(fields), 1)
        name, value = fields[0]
        self.assertEqual(name, "Busted")
        self.assertIn("bob", value)

    def test_only_cashes_yields_single_cashed_field(self) -> None:
        fields = build_impact_fields([self._impact("amy", "cashed")])
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0][0], "Cashed")
        self.assertIn("amy", fields[0][1])

    def test_both_sides_yield_two_fields(self) -> None:
        fields = build_impact_fields([self._impact("bob", "busted"), self._impact("amy", "cashed")])
        names = [n for n, _ in fields]
        self.assertIn("Busted", names)
        self.assertIn("Cashed", names)
        self.assertEqual(len(fields), 2)

    def test_mortal_lock_marked_in_value(self) -> None:
        fields = build_impact_fields([self._impact("bob", "busted", lock=True)])
        self.assertIn("\U0001f512", fields[0][1])  # 🔒

    def test_long_list_capped_with_more_suffix(self) -> None:
        impacts = [self._impact(f"user{i}", "cashed") for i in range(9)]
        fields = build_impact_fields(impacts)
        value = fields[0][1]
        self.assertIn("more", value)
        # First few names shown, the tail collapsed into "+k more".
        self.assertIn("user0", value)
        self.assertNotIn("user8", value)
        self.assertIn("+3 more", value)


class BuildGameFinalEmbedTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_emoji_cache()

    def tearDown(self) -> None:
        reset_emoji_cache()

    def test_plain_title_no_custom_emoji(self) -> None:
        populate_emoji_cache([_FakeEmoji("chiefs", 7)])
        embed = build_game_final_embed(_event(), "KC takes it")
        self.assertIsNotNone(embed.title)
        self.assertNotIn("<:", embed.title or "")
        self.assertIn("Final", embed.title or "")

    def test_description_is_score_line_then_quip_verbatim(self) -> None:
        populate_emoji_cache([_FakeEmoji("chiefs", 7)])
        quip = "The Chiefs slammed the door shut. 🔥"
        embed = build_game_final_embed(_event(), quip)
        desc = embed.description or ""
        lines = desc.split("\n")
        # Line 1 = deterministic score; the quip appears verbatim after it.
        self.assertIn("KC", lines[0])
        self.assertIn("<:chiefs:7>", lines[0])
        self.assertIn(quip, desc)
        self.assertGreater(desc.index(quip), desc.index(lines[0]))

    def test_winner_color_on_bar(self) -> None:
        embed = build_game_final_embed(_event(away_score=20, home_score=27), "q")
        assert embed.color is not None
        self.assertEqual(embed.color.value, resolve_team_color("KC"))

    def test_impacts_populate_fields(self) -> None:
        event = _event(
            impacts=[
                {"username": "bob", "outcome": "busted", "was_mortal_lock": True},
                {"username": "amy", "outcome": "cashed", "was_mortal_lock": False},
            ]
        )
        embed = build_game_final_embed(event, "q")
        field_names = [f.name for f in embed.fields]
        self.assertIn("Busted", field_names)
        self.assertIn("Cashed", field_names)

    def test_no_impacts_means_no_fields(self) -> None:
        embed = build_game_final_embed(_event(impacts=[]), "q")
        self.assertEqual(len(embed.fields), 0)

    def test_field_names_have_no_custom_emoji(self) -> None:
        event = _event(impacts=[{"username": "bob", "outcome": "busted", "was_mortal_lock": False}])
        embed = build_game_final_embed(event, "q")
        for field in embed.fields:
            self.assertNotIn("<:", field.name or "")


if __name__ == "__main__":
    unittest.main()
