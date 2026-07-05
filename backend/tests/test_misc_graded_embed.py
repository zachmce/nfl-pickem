"""Offline unit tests for the pure misc.graded embed builder (260705-if1).

The builder is Discord-send-free: it constructs a ``discord.Embed`` but takes no
client and performs no send, so it is fully unit-testable without a live gateway.
Events are built via the real :func:`app.services.notifications.misc_graded_event`
so the tests consume the real payload shape (never hand-rolled dict literals),
exactly as ``test_game_final_embed`` uses ``game_final_event``.

Layout under test (post-live-fire redesign):
* title = ``Week {week} - MISC Graded · {actor}`` (player in the title);
* two inline fields — ``Result`` (✅ Cashed / ❌ Busted) and ``Verdict``
  (verdict word + points, zero rendered as a plain ``0``);
* a final full-width quip field (zero-width name), omitted when blank;
* a ``Graded by {grader}`` footer when the event carries a grader; no Player field,
  no Prediction field, no description.

Run with: ``backend/.venv/bin/python -m unittest tests.test_misc_graded_embed -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

from app.bot.misc_graded_embed import (
    HIT_COLOR,
    MISS_COLOR,
    build_misc_graded_embed,
    build_result_value,
    build_verdict_value,
    is_hit,
    select_color,
)
from app.services.notifications import misc_graded_event


def _event(
    *,
    week: int = 3,
    actor: str = "alice",
    prediction: str = "over 45.5 total points",
    verdict: str = "correct",
    points: int = 3,
    grader: str | None = None,
) -> dict:
    return misc_graded_event(
        actor=actor,
        week=week,
        prediction=prediction,
        verdict=verdict,
        points=points,
        grader=grader,
    )


class ColorAndResultTests(unittest.TestCase):
    # The SIGN of points — not the verdict word — drives color/marker per the design.
    def test_hit_uses_hit_color_and_cashed_result(self) -> None:
        event = _event(points=3)
        self.assertTrue(is_hit(event))
        self.assertEqual(select_color(event), HIT_COLOR)
        self.assertIn("Cashed", build_result_value(event))

    def test_miss_uses_miss_color_and_busted_result(self) -> None:
        event = _event(points=-2)
        self.assertFalse(is_hit(event))
        self.assertEqual(select_color(event), MISS_COLOR)
        self.assertIn("Busted", build_result_value(event))

    def test_zero_points_is_a_miss(self) -> None:
        # Boundary: points == 0 is a MISS (red / Busted).
        event = _event(points=0)
        self.assertFalse(is_hit(event))
        self.assertEqual(select_color(event), MISS_COLOR)
        self.assertIn("Busted", build_result_value(event))


class BuildVerdictValueTests(unittest.TestCase):
    def test_positive_points_show_signed_plus(self) -> None:
        value = build_verdict_value(_event(verdict="correct", points=3))
        self.assertEqual(value, "correct (+3)")

    def test_negative_points_show_signed_minus(self) -> None:
        value = build_verdict_value(_event(verdict="incorrect", points=-2))
        self.assertEqual(value, "incorrect (-2)")

    def test_zero_points_unsigned(self) -> None:
        # Zero renders as a plain "0", NOT "+0".
        value = build_verdict_value(_event(verdict="incorrect", points=0))
        self.assertEqual(value, "incorrect (0)")
        self.assertNotIn("+0", value)


class BuildMiscGradedEmbedTests(unittest.TestCase):
    def test_title_has_actor_and_no_custom_emoji(self) -> None:
        embed = build_misc_graded_embed(_event(actor="bot_carol", week=1), "nice call")
        self.assertEqual(embed.title, "Week 1 - MISC Graded · bot_carol")
        self.assertNotIn("<:", embed.title or "")

    def test_no_description(self) -> None:
        embed = build_misc_graded_embed(_event(), "a quip")
        self.assertIn(embed.description, (None, ""))

    def test_hit_embed_color(self) -> None:
        embed = build_misc_graded_embed(_event(points=3), "q")
        assert embed.color is not None
        self.assertEqual(embed.color.value, HIT_COLOR)

    def test_miss_embed_color(self) -> None:
        embed = build_misc_graded_embed(_event(points=-1), "q")
        assert embed.color is not None
        self.assertEqual(embed.color.value, MISS_COLOR)

    def test_result_and_verdict_inline_fields(self) -> None:
        embed = build_misc_graded_embed(_event(verdict="correct", points=3), "q")
        results = [f for f in embed.fields if f.name == "Result"]
        verdicts = [f for f in embed.fields if f.name == "Verdict"]
        self.assertEqual(len(results), 1)
        self.assertEqual(len(verdicts), 1)
        self.assertIn("Cashed", results[0].value or "")
        self.assertTrue(results[0].inline)
        self.assertEqual(verdicts[0].value, "correct (+3)")
        self.assertTrue(verdicts[0].inline)

    def test_verdict_field_zero_boundary(self) -> None:
        embed = build_misc_graded_embed(_event(verdict="incorrect", points=0), "q")
        verdicts = [f.value or "" for f in embed.fields if f.name == "Verdict"]
        self.assertEqual(verdicts, ["incorrect (0)"])

    def test_no_player_or_prediction_field(self) -> None:
        embed = build_misc_graded_embed(_event(actor="alice", prediction="KC covers"), "q")
        names = [f.name for f in embed.fields]
        self.assertNotIn("Player", names)
        self.assertNotIn("Prediction", names)


class QuipFieldTests(unittest.TestCase):
    def test_quip_is_last_field_verbatim(self) -> None:
        quip = "alice nailed it. 🔥"
        embed = build_misc_graded_embed(_event(points=3), quip)
        last = embed.fields[-1]
        # Quip sits at the bottom (after the Result/Verdict columns) as its own field.
        self.assertEqual(last.value, quip)
        self.assertGreater(len(embed.fields), 2)
        # Zero-width-space name so only the quip text shows.
        self.assertNotIn(last.name, ("Result", "Verdict"))

    def test_blank_quip_omits_field(self) -> None:
        embed = build_misc_graded_embed(_event(), "   ")
        # Only the two inline columns remain — no trailing quip field.
        self.assertEqual([f.name for f in embed.fields], ["Result", "Verdict"])

    def test_empty_quip_omits_field(self) -> None:
        embed = build_misc_graded_embed(_event(), "")
        self.assertEqual([f.name for f in embed.fields], ["Result", "Verdict"])


class GraderFooterTests(unittest.TestCase):
    def test_footer_present_when_grader_set(self) -> None:
        embed = build_misc_graded_embed(_event(grader="admin_zach"), "q")
        self.assertEqual(embed.footer.text, "Graded by admin_zach")

    def test_no_footer_when_grader_none(self) -> None:
        embed = build_misc_graded_embed(_event(grader=None), "q")
        self.assertIn(embed.footer.text, (None, ""))

    def test_no_footer_when_grader_key_absent(self) -> None:
        # Older / synthetic events with NO grader key must still render.
        event = _event()
        del event["grader"]
        embed = build_misc_graded_embed(event, "q")
        self.assertIn(embed.footer.text, (None, ""))


class MinimalEventTests(unittest.TestCase):
    def test_minimal_event_does_not_raise(self) -> None:
        import discord

        embed = build_misc_graded_embed(_event(), "q")
        self.assertIsInstance(embed, discord.Embed)


if __name__ == "__main__":
    unittest.main()
