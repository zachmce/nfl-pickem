"""Offline unit tests for the pure week.recap embed builder (260705-kuv).

The builder is Discord-send-free: it constructs a ``discord.Embed`` but takes no
client and performs no send, so it is fully unit-testable without a live gateway.
Events are built via the REAL :func:`app.services.notifications.week_recap_event`
(never hand-rolled dicts) so the tests consume the real payload shape.

Run with: ``backend/.venv/bin/python -m unittest tests.test_week_recap_embed -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

from app.bot.game_final_embed import MORTAL_LOCK_MARKER
from app.bot.misc_graded_embed import HIT_MARKER, MISS_MARKER
from app.bot.week_recap_embed import (
    RECAP_EMBED_COLOR,
    RECAP_ROW_CAP,
    build_week_recap_embed,
)
from app.services.notifications import week_recap_event


def _standings(n: int) -> list[dict]:
    return [
        {
            "rank": i,
            "display_name": f"player{i}",
            "season_total": 100 - i,
            "week_delta": 0 if i % 2 else i,
        }
        for i in range(1, n + 1)
    ]


def _event(
    *,
    week: int = 3,
    winner: str = "alice",
    winner_score: int = 9,
    leader: str = "bob",
    leader_score: int = 30,
    standings: list[dict] | None = None,
    best_call: dict | None = None,
    biggest_bust: dict | None = None,
    mortal_locks: list[dict] | None = None,
) -> dict:
    return week_recap_event(
        week=week,
        winner=winner,
        winner_score=winner_score,
        leader=leader,
        leader_score=leader_score,
        standings=standings,  # type: ignore[arg-type]
        best_call=best_call,  # type: ignore[arg-type]
        biggest_bust=biggest_bust,  # type: ignore[arg-type]
        mortal_locks=mortal_locks,  # type: ignore[arg-type]
    )


class BuildWeekRecapEmbedTests(unittest.TestCase):
    def test_plain_title_no_custom_emoji(self) -> None:
        embed = build_week_recap_embed(_event(week=7), "narration")
        self.assertEqual(embed.title, "Week 7 - Recap")
        self.assertNotIn("<:", embed.title or "")

    def test_amethyst_color(self) -> None:
        embed = build_week_recap_embed(_event(), "n")
        assert embed.color is not None
        self.assertEqual(embed.color.value, RECAP_EMBED_COLOR)

    def test_description_is_narration_verbatim(self) -> None:
        narration = "alice ran the table while bob coughed up the lead. 🏈"
        embed = build_week_recap_embed(_event(), narration)
        self.assertEqual(embed.description, narration)

    def test_minimal_payload_renders_only_winner_headline(self) -> None:
        # week_recap_event with only the 4 original kwargs -> new blocks default
        # []/None, so only the Week Winner field renders (never raises).
        embed = build_week_recap_embed(_event(), "n")
        names = [f.name for f in embed.fields]
        self.assertEqual(names, ["Week Winner"])
        self.assertIn("alice", embed.fields[0].value or "")
        self.assertIn("9", embed.fields[0].value or "")

    def test_no_winner_renders_zero_fields(self) -> None:
        embed = build_week_recap_embed(_event(winner=""), "n")
        self.assertEqual(len(embed.fields), 0)

    def test_all_blocks_populate_their_fields(self) -> None:
        event = _event(
            standings=_standings(3),
            best_call={
                "display_name": "alice",
                "team_abbr": "T2",
                "side_label": "Underdog (T2)",
                "spread": "10.0",
                "is_mortal_lock": False,
            },
            biggest_bust={
                "display_name": "bob",
                "team_abbr": "T7",
                "side_label": "Favorite (T7)",
                "spread": "7.5",
                "is_mortal_lock": True,
            },
            mortal_locks=[{"display_name": "bob", "hit": True, "points": 2, "side_label": "U"}],
        )
        embed = build_week_recap_embed(event, "n")
        names = [f.name for f in embed.fields]
        self.assertEqual(
            names, ["Week Winner", "Best Call", "Biggest Bust", "Mortal Locks", "Standings"]
        )

    def test_best_call_names_player_spread_and_no_lock_marker(self) -> None:
        event = _event(
            best_call={
                "display_name": "alice",
                "team_abbr": "T2",
                "side_label": "Underdog (T2)",
                "spread": "10.0",
                "is_mortal_lock": False,
            }
        )
        embed = build_week_recap_embed(event, "n")
        best = next(f for f in embed.fields if f.name == "Best Call")
        value = best.value or ""
        self.assertIn("alice", value)
        self.assertIn("Underdog (T2)", value)
        self.assertIn("+10.0", value)
        self.assertNotIn(MORTAL_LOCK_MARKER, value)  # not a mortal lock

    def test_biggest_bust_marks_mortal_lock(self) -> None:
        event = _event(
            biggest_bust={
                "display_name": "bob",
                "team_abbr": "T7",
                "side_label": "Favorite (T7)",
                "spread": "7.5",
                "is_mortal_lock": True,
            }
        )
        embed = build_week_recap_embed(event, "n")
        bust = next(f for f in embed.fields if f.name == "Biggest Bust")
        value = bust.value or ""
        self.assertIn("bob", value)
        self.assertIn("+7.5", value)
        self.assertIn(MORTAL_LOCK_MARKER, value)  # amplified mortal-lock bust

    def test_mortal_lock_board_marks_hit_and_miss(self) -> None:
        event = _event(
            mortal_locks=[
                {"display_name": "bob", "hit": True, "points": 2, "side_label": "Underdog (T2)"},
                {
                    "display_name": "alice",
                    "hit": False,
                    "points": -1,
                    "side_label": "Favorite (T7)",
                },
            ]
        )
        embed = build_week_recap_embed(event, "n")
        board = next(f for f in embed.fields if f.name == "Mortal Locks")
        value = board.value or ""
        self.assertIn(f"{HIT_MARKER} bob (+2)", value)
        self.assertIn(f"{MISS_MARKER} alice (-1)", value)

    def test_standings_delta_formatting_signed_and_plain_zero(self) -> None:
        event = _event(
            standings=[
                {"rank": 1, "display_name": "bob", "season_total": 30, "week_delta": 6},
                {"rank": 2, "display_name": "alice", "season_total": 20, "week_delta": 0},
            ]
        )
        embed = build_week_recap_embed(event, "n")
        standings = next(f for f in embed.fields if f.name == "Standings")
        value = standings.value or ""
        self.assertIn("1. bob — 30 (+6 this wk)", value)  # signed non-zero
        self.assertIn("2. alice — 20 (0 this wk)", value)  # plain zero, no +0

    def test_long_standings_list_is_capped_with_more_tail(self) -> None:
        n = RECAP_ROW_CAP + 3
        embed = build_week_recap_embed(_event(standings=_standings(n)), "n")
        standings = next(f for f in embed.fields if f.name == "Standings")
        value = standings.value or ""
        self.assertIn("+3 more", value)
        self.assertIn("player1", value)
        self.assertNotIn(f"player{n}.", value)  # the tail row is collapsed
        # Value stays within Discord's 1024-char-per-field limit.
        self.assertLessEqual(len(value), 1024)

    def test_no_custom_emoji_in_any_field_name(self) -> None:
        event = _event(
            standings=_standings(2),
            best_call={
                "display_name": "a",
                "team_abbr": "T2",
                "side_label": "Underdog (T2)",
                "spread": "10.0",
                "is_mortal_lock": False,
            },
            mortal_locks=[{"display_name": "b", "hit": True, "points": 2, "side_label": "U"}],
        )
        embed = build_week_recap_embed(event, "n")
        for field in embed.fields:
            self.assertNotIn("<:", field.name or "")


if __name__ == "__main__":
    unittest.main()
