"""Offline unit tests for the team-logo emoji resolver + decorator (260627-wt5).

These never touch a live Discord gateway: the application-emoji cache is populated
directly via the module's ``populate_emoji_cache`` / ``reset_emoji_cache`` affordance
with fake ``<:name:id>`` strings, and ``decorate_team_logos`` is exercised both
against the real (populated) cache and against an injected fake abbr->logo map.

Two naming forms are covered deliberately: the ``<nickname>logo`` form (e.g.
``vikingslogo``) and the bare-nickname form (e.g. ``chiefs``) — the resolver must
match either.

Run with: ``backend/.venv/bin/python -m unittest tests.test_team_emoji -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.bot.team_emoji import (
    ABBR_TO_NICKNAME,
    NEUTRAL_EMBED_COLOR,
    decorate_team_logos,
    populate_emoji_cache,
    reset_emoji_cache,
    resolve_logo,
    resolve_team_color,
)


@dataclass
class _FakeEmoji:
    """Minimal stand-in for discord.Emoji: ``populate_emoji_cache`` reads ``.name``
    and ``str(emoji)`` (the ``<:name:id>`` posting form)."""

    name: str
    id: int

    def __str__(self) -> str:
        return f"<:{self.name}:{self.id}>"


class AbbrToNicknameMapTests(unittest.TestCase):
    def test_covers_all_32_teams(self) -> None:
        self.assertEqual(len(ABBR_TO_NICKNAME), 32)

    def test_uses_real_abbreviations(self) -> None:
        # The four that are commonly mis-spelled must be the REAL strings.
        self.assertIn("WSH", ABBR_TO_NICKNAME)
        self.assertIn("LAR", ABBR_TO_NICKNAME)
        self.assertIn("LV", ABBR_TO_NICKNAME)
        self.assertIn("JAX", ABBR_TO_NICKNAME)
        self.assertNotIn("WAS", ABBR_TO_NICKNAME)
        self.assertNotIn("LA", ABBR_TO_NICKNAME)
        self.assertNotIn("LVR", ABBR_TO_NICKNAME)
        self.assertNotIn("JAC", ABBR_TO_NICKNAME)

    def test_sample_nicknames(self) -> None:
        self.assertEqual(ABBR_TO_NICKNAME["MIN"], "vikings")
        self.assertEqual(ABBR_TO_NICKNAME["KC"], "chiefs")
        self.assertEqual(ABBR_TO_NICKNAME["SF"], "49ers")
        self.assertEqual(ABBR_TO_NICKNAME["WSH"], "commanders")
        self.assertEqual(ABBR_TO_NICKNAME["GB"], "packers")


class ResolveLogoTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_emoji_cache()

    def tearDown(self) -> None:
        reset_emoji_cache()

    def test_resolves_via_nickname_logo_form(self) -> None:
        populate_emoji_cache([_FakeEmoji("vikingslogo", 111)])
        self.assertEqual(resolve_logo("MIN"), "<:vikingslogo:111>")

    def test_resolves_via_bare_nickname_form(self) -> None:
        populate_emoji_cache([_FakeEmoji("chiefs", 222)])
        self.assertEqual(resolve_logo("KC"), "<:chiefs:222>")

    def test_match_is_case_insensitive_on_emoji_name(self) -> None:
        populate_emoji_cache([_FakeEmoji("VikingsLogo", 333)])
        self.assertEqual(resolve_logo("MIN"), "<:VikingsLogo:333>")

    def test_known_abbr_with_no_cached_emoji_returns_none(self) -> None:
        populate_emoji_cache([_FakeEmoji("chiefs", 222)])
        self.assertIsNone(resolve_logo("MIN"))

    def test_unknown_abbr_returns_none(self) -> None:
        populate_emoji_cache([_FakeEmoji("vikingslogo", 111)])
        self.assertIsNone(resolve_logo("ZZZ"))

    def test_empty_cache_returns_none(self) -> None:
        self.assertIsNone(resolve_logo("MIN"))


class DecorateTeamLogosTests(unittest.TestCase):
    """``decorate_team_logos`` against an INJECTED fake abbr->logo map so the
    pure decoration behavior is tested without the cache."""

    FAKE_MAP = {
        "MIN": "<:vikingslogo:1>",
        "KC": "<:chiefs:2>",
        "BUF": "<:billslogo:3>",
        "NO": "<:saintslogo:4>",
        "GB": "<:packers:5>",
        "SF": "<:49erslogo:6>",
        # CIN deliberately omitted to test "no resolved emoji -> token unchanged".
    }

    def _dec(self, text: str) -> str:
        return decorate_team_logos(text, logo_map=self.FAKE_MAP)

    def test_uppercase_abbr_decorated_after_token(self) -> None:
        self.assertEqual(self._dec("MIN wins"), "MIN <:vikingslogo:1> wins")

    def test_capitalized_nickname_decorated(self) -> None:
        self.assertEqual(self._dec("Vikings roll"), "Vikings <:vikingslogo:1> roll")

    def test_lowercase_common_word_not_decorated(self) -> None:
        self.assertEqual(self._dec("bills are due"), "bills are due")
        self.assertEqual(self._dec("the saints go"), "the saints go")

    def test_abbr_substring_inside_word_not_decorated(self) -> None:
        self.assertEqual(self._dec("MINISTER speaks"), "MINISTER speaks")
        self.assertEqual(self._dec("GBP rate"), "GBP rate")

    def test_team_with_no_mapped_logo_left_unchanged(self) -> None:
        # CIN is a real abbr but absent from FAKE_MAP -> not decorated.
        self.assertEqual(self._dec("CIN are hot"), "CIN are hot")

    def test_no_double_decorate_inside_existing_emoji_token(self) -> None:
        # The numeric id / lowercased name inside an emoji token must not be matched.
        text = "score <:vikingslogo:1> here"
        self.assertEqual(self._dec(text), "score <:vikingslogo:1> here")

    def test_already_decorated_abbr_not_double_decorated(self) -> None:
        # An abbr already followed by its emoji should gain at most one emoji.
        text = "MIN <:vikingslogo:1> wins"
        out = self._dec(text)
        self.assertEqual(out.count("<:vikingslogo:1>"), 1)

    def test_empty_string_unchanged(self) -> None:
        self.assertEqual(self._dec(""), "")

    def test_no_team_token_unchanged(self) -> None:
        self.assertEqual(self._dec("good luck everyone"), "good luck everyone")

    def test_multiple_tokens_each_decorated(self) -> None:
        self.assertEqual(
            self._dec("KC beat MIN"),
            "KC <:chiefs:2> beat MIN <:vikingslogo:1>",
        )

    def test_digit_nickname_capitalized_form(self) -> None:
        # SF -> "49ers"; the Capitalized form is "49ers" (starts with a digit).
        self.assertEqual(self._dec("49ers win"), "49ers <:49erslogo:6> win")

    def test_never_raises_returns_input_on_bad_map(self) -> None:
        # A map whose value is not a string could blow up an insertion; the
        # function must swallow it and return the original text.
        bad = {"MIN": None}
        self.assertEqual(decorate_team_logos("MIN wins", logo_map=bad), "MIN wins")

    def test_default_map_uses_live_cache(self) -> None:
        # With no logo_map passed it builds from the module cache; an empty cache
        # leaves the line unchanged.
        reset_emoji_cache()
        self.assertEqual(decorate_team_logos("MIN wins"), "MIN wins")
        populate_emoji_cache([_FakeEmoji("vikingslogo", 9)])
        self.assertEqual(decorate_team_logos("MIN wins"), "MIN <:vikingslogo:9> wins")
        reset_emoji_cache()


class DoubleLogoDedupeTests(unittest.TestCase):
    """A line naming ONE team by BOTH its abbr and its Capitalized nickname (which
    resolve to the SAME logo string) must gain that logo exactly ONCE — the
    2026-07-03 live-fire "Vikings (MIN)" double-logo bug."""

    FAKE_MAP = {
        "MIN": "<:vikingslogo:1>",
        "KC": "<:chiefs:2>",
    }

    def _dec(self, text: str) -> str:
        return decorate_team_logos(text, logo_map=self.FAKE_MAP)

    def test_same_team_by_abbr_and_nickname_gets_one_logo(self) -> None:
        out = self._dec("Vikings (MIN) rolled")
        self.assertEqual(out.count("<:vikingslogo:1>"), 1)
        # The nickname (first mention) carries the logo; the abbr stays bare.
        self.assertEqual(out, "Vikings <:vikingslogo:1> (MIN) rolled")

    def test_same_team_abbr_then_nickname_gets_one_logo(self) -> None:
        out = self._dec("MIN — the Vikings — won")
        self.assertEqual(out.count("<:vikingslogo:1>"), 1)

    def test_two_different_teams_each_get_their_logo(self) -> None:
        out = self._dec("KC over MIN and the Vikings")
        self.assertEqual(out.count("<:chiefs:2>"), 1)
        self.assertEqual(out.count("<:vikingslogo:1>"), 1)


class MarketCoverageTests(unittest.TestCase):
    """Paraphrased city/market names decorate with the right logo (once each);
    ambiguous shared-city forms ("New York", "Los Angeles") are NOT decorated."""

    MARKET_MAP = {
        "LV": "<:raiderslogo:10>",
        "NE": "<:patriotslogo:11>",
        "NO": "<:saintslogo:12>",
        "TB": "<:buccaneerslogo:13>",
        "ARI": "<:cardinalslogo:14>",
        "WSH": "<:commanderslogo:15>",
        "JAX": "<:jaguarslogo:16>",
        "ATL": "<:falconslogo:17>",
        "CAR": "<:pantherslogo:18>",
        "NYG": "<:giantslogo:19>",
        "NYJ": "<:jetslogo:20>",
        "LAR": "<:ramslogo:21>",
        "LAC": "<:chargerslogo:22>",
    }

    def _dec(self, text: str) -> str:
        return decorate_team_logos(text, logo_map=self.MARKET_MAP)

    def test_multiword_market_phrases_decorate(self) -> None:
        self.assertEqual(self._dec("Las Vegas rolled"), "Las Vegas <:raiderslogo:10> rolled")
        self.assertEqual(self._dec("New England won"), "New England <:patriotslogo:11> won")
        self.assertEqual(self._dec("New Orleans fell"), "New Orleans <:saintslogo:12> fell")
        self.assertEqual(self._dec("Tampa Bay cruised"), "Tampa Bay <:buccaneerslogo:13> cruised")

    def test_single_city_forms_decorate(self) -> None:
        self.assertEqual(self._dec("Arizona surged"), "Arizona <:cardinalslogo:14> surged")
        self.assertEqual(self._dec("Washington held on"), "Washington <:commanderslogo:15> held on")
        self.assertEqual(self._dec("Jacksonville lost"), "Jacksonville <:jaguarslogo:16> lost")
        self.assertEqual(self._dec("Atlanta blew it"), "Atlanta <:falconslogo:17> blew it")
        self.assertEqual(self._dec("Carolina rebuilt"), "Carolina <:pantherslogo:18> rebuilt")

    def test_ambiguous_shared_city_forms_not_decorated(self) -> None:
        # "New York" = NYG/NYJ and "Los Angeles" = LAR/LAC — no arbitrary pick.
        self.assertEqual(self._dec("New York is loud"), "New York is loud")
        self.assertEqual(self._dec("Los Angeles is sunny"), "Los Angeles is sunny")
        self.assertNotIn("<:", self._dec("New York and Los Angeles"))

    def test_market_phrase_dedupes_with_abbr(self) -> None:
        # "Las Vegas (LV)" resolves the SAME logo twice -> one insertion.
        out = self._dec("Las Vegas (LV) won")
        self.assertEqual(out.count("<:raiderslogo:10>"), 1)


class TeamColorTests(unittest.TestCase):
    """The 32-team primary-color map + neutral fallback (D-02)."""

    def test_known_abbr_returns_int(self) -> None:
        self.assertIsInstance(resolve_team_color("KC"), int)

    def test_unknown_abbr_returns_none(self) -> None:
        self.assertIsNone(resolve_team_color("ZZZ"))

    def test_all_32_abbrs_have_a_color(self) -> None:
        for abbr in ABBR_TO_NICKNAME:
            self.assertIsInstance(resolve_team_color(abbr), int, msg=abbr)

    def test_neutral_fallback_constant_is_int(self) -> None:
        self.assertIsInstance(NEUTRAL_EMBED_COLOR, int)

    def test_colors_are_in_rgb_range(self) -> None:
        for abbr in ABBR_TO_NICKNAME:
            color = resolve_team_color(abbr)
            assert color is not None
            self.assertGreaterEqual(color, 0)
            self.assertLessEqual(color, 0xFFFFFF)


if __name__ == "__main__":
    unittest.main()
