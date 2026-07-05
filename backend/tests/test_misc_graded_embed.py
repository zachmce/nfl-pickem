"""Offline unit tests for the pure misc.graded embed builder (260705-if1).

The builder is Discord-send-free: it constructs a ``discord.Embed`` but takes no
client and performs no send, so it is fully unit-testable without a live gateway.
Events are built via the real :func:`app.services.notifications.misc_graded_event`
so the tests consume the real payload shape (never hand-rolled dict literals),
exactly as ``test_game_final_embed`` uses ``game_final_event``.

Run with: ``backend/.venv/bin/python -m unittest tests.test_misc_graded_embed -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

from app.bot.misc_graded_embed import (
    HIT_COLOR,
    MISS_COLOR,
    build_marker_line,
    build_misc_graded_embed,
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
) -> dict:
    return misc_graded_event(
        actor=actor,
        week=week,
        prediction=prediction,
        verdict=verdict,
        points=points,
    )


class ColorAndMarkerTests(unittest.TestCase):
    # The SIGN of points — not the verdict word — drives color/marker per the design.
    def test_hit_uses_hit_color_and_cashed_marker(self) -> None:
        event = _event(points=3)
        self.assertTrue(is_hit(event))
        self.assertEqual(select_color(event), HIT_COLOR)
        self.assertIn("Cashed", build_marker_line(event))

    def test_miss_uses_miss_color_and_busted_marker(self) -> None:
        event = _event(points=-2)
        self.assertFalse(is_hit(event))
        self.assertEqual(select_color(event), MISS_COLOR)
        self.assertIn("Busted", build_marker_line(event))

    def test_zero_points_is_a_miss(self) -> None:
        # Boundary: points == 0 is a MISS (red / Busted).
        event = _event(points=0)
        self.assertFalse(is_hit(event))
        self.assertEqual(select_color(event), MISS_COLOR)
        self.assertIn("Busted", build_marker_line(event))


class BuildVerdictValueTests(unittest.TestCase):
    def test_positive_points_show_signed_plus(self) -> None:
        value = build_verdict_value(_event(verdict="correct", points=3))
        self.assertIn("correct", value)
        self.assertIn("+3", value)

    def test_negative_points_show_signed_minus(self) -> None:
        value = build_verdict_value(_event(verdict="incorrect", points=-2))
        self.assertIn("incorrect", value)
        self.assertIn("-2", value)


class BuildMiscGradedEmbedTests(unittest.TestCase):
    def test_plain_title_no_custom_emoji(self) -> None:
        embed = build_misc_graded_embed(_event(), "nice call")
        self.assertEqual(embed.title, "Week 3 - MISC Graded")
        self.assertNotIn("<:", embed.title or "")

    def test_description_is_marker_then_quip_verbatim(self) -> None:
        quip = "alice nailed it. 🔥"
        embed = build_misc_graded_embed(_event(points=3), quip)
        desc = embed.description or ""
        lines = desc.split("\n")
        # Line 1 = the marker; the quip appears verbatim after it.
        self.assertIn("Cashed", lines[0])
        self.assertIn(quip, desc)
        self.assertGreater(desc.index(quip), desc.index(lines[0]))

    def test_hit_embed_color(self) -> None:
        embed = build_misc_graded_embed(_event(points=3), "q")
        assert embed.color is not None
        self.assertEqual(embed.color.value, HIT_COLOR)

    def test_miss_embed_color(self) -> None:
        embed = build_misc_graded_embed(_event(points=-1), "q")
        assert embed.color is not None
        self.assertEqual(embed.color.value, MISS_COLOR)

    def test_player_field_equals_actor(self) -> None:
        embed = build_misc_graded_embed(_event(actor="bob"), "q")
        players = [f.value for f in embed.fields if f.name == "Player"]
        self.assertEqual(players, ["bob"])

    def test_verdict_field_has_word_and_signed_points(self) -> None:
        embed = build_misc_graded_embed(_event(verdict="correct", points=3), "q")
        verdicts = [f.value or "" for f in embed.fields if f.name == "Verdict"]
        self.assertEqual(len(verdicts), 1)
        self.assertIn("correct", verdicts[0])
        self.assertIn("+3", verdicts[0])


class PredictionOmitEmptyTests(unittest.TestCase):
    def test_populated_prediction_yields_field(self) -> None:
        embed = build_misc_graded_embed(_event(prediction="KC covers -3"), "q")
        preds = [f.value for f in embed.fields if f.name == "Prediction"]
        self.assertEqual(preds, ["KC covers -3"])

    def test_empty_prediction_omits_field(self) -> None:
        embed = build_misc_graded_embed(_event(prediction=""), "q")
        self.assertNotIn("Prediction", [f.name for f in embed.fields])

    def test_whitespace_prediction_omits_field(self) -> None:
        embed = build_misc_graded_embed(_event(prediction="   "), "q")
        self.assertNotIn("Prediction", [f.name for f in embed.fields])


class MinimalEventTests(unittest.TestCase):
    def test_minimal_event_does_not_raise(self) -> None:
        import discord

        embed = build_misc_graded_embed(_event(), "q")
        self.assertIsInstance(embed, discord.Embed)


if __name__ == "__main__":
    unittest.main()
