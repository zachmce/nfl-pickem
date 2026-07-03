"""Offline tests pinning the personality-swap invariants (260627-xbb).

The whole point of the swap: a personality changes ONLY the leading voice
preamble — never the byte-identical anti-hallucination GUARD, never the per-event
ROLE clauses (the roster count-only leak clause, the misc verdict-preservation
clause, the recap supplied-storyline clause). These tests compose every event's
system prompt for EVERY personality and assert:

* the ``_FACTS_FIRST_GUARD`` (and the recap/repeated-pick guard tails) appear
  verbatim in every composed prompt for every voice;
* the roster.complete composed prompt always carries the count-only leak clause;
* swapping the active id changes the voice preamble but NOT the invariant tail;
* ``get_bot_personality`` falls back to sarcastic when the setting is unset.

Fully OFFLINE (in-memory SQLite for the default-fallback check; pure string
composition for the rest). Run from backend/ with
``.venv/bin/python -m unittest`` (unittest, NOT pytest).
"""

from __future__ import annotations

import unittest

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.bot import chat_personality, llm_client, recap
from app.bot.personality import (
    DEFAULT_PERSONALITY_ID,
    PERSONALITIES,
    available_personality_ids,
    compose_prompt,
    voice_for,
)
from app.services import app_settings


# The per-event (ROLE, GUARD) pairs across the three bot files. Each composed
# prompt = voice + ROLE + GUARD. The GUARD (and the invariant ROLE clauses) must be
# byte-identical for every voice.
_EVENT_PARTS = {
    "window.opened": (chat_personality._WINDOW_OPENED_ROLE, chat_personality._FACTS_FIRST_GUARD),
    "game.final": (chat_personality._GAME_FINAL_ROLE, chat_personality._FACTS_FIRST_GUARD),
    "roster.complete": (
        chat_personality._ROSTER_COMPLETE_ROLE,
        chat_personality._FACTS_FIRST_GUARD,
    ),
    "misc.graded": (chat_personality._MISC_GRADED_ROLE, chat_personality._FACTS_FIRST_GUARD),
    "week.recap": (recap.RECAP_ROLE, recap.RECAP_GUARD),
    "repeated.pick": (llm_client.REPEATED_PICK_ROLE, llm_client.REPEATED_PICK_GUARD),
}

# The exact roster.complete count-only leak clause that must survive every voice.
_LEAK_CLAUSE = (
    "you do NOT know who they are or what anyone picked, so NEVER name "
    "another player and NEVER guess any pick"
)

# The misc.graded verdict-preservation clause that must survive every voice.
_VERDICT_CLAUSE = "do NOT alter the verdict or the points"

# The recap supplied-storyline clause that must survive every voice (260703-jun: the
# guard now PERMITS supplied storylines while forbidding invented ones).
_RECAP_STORYLINE_CLAUSE = "never claim anyone rose or fell except as those notes state"


def _composed(personality_id: str) -> dict[str, str]:
    """Compose every event's system prompt under one personality's voice."""
    voice = voice_for(personality_id)
    return {
        event: compose_prompt(voice, role, guard) for event, (role, guard) in _EVENT_PARTS.items()
    }


class GuardInvariantAcrossPersonalitiesTests(unittest.TestCase):
    """The invariant guard tail is present and byte-identical for every voice."""

    def test_facts_first_guard_in_every_event_for_every_personality(self) -> None:
        for pid in available_personality_ids():
            composed = _composed(pid)
            for event in ("window.opened", "game.final", "roster.complete", "misc.graded"):
                self.assertIn(
                    chat_personality._FACTS_FIRST_GUARD,
                    composed[event],
                    f"{event} under {pid} dropped the facts-first guard",
                )

    def test_recap_and_repeated_guards_present_for_every_personality(self) -> None:
        for pid in available_personality_ids():
            composed = _composed(pid)
            self.assertIn(recap.RECAP_GUARD, composed["week.recap"])
            self.assertIn(llm_client.REPEATED_PICK_GUARD, composed["repeated.pick"])

    def test_guard_tail_byte_identical_across_personalities(self) -> None:
        # The substring AFTER the voice (role + guard) must be identical for every
        # personality — only the leading voice differs.
        for event, (role, guard) in _EVENT_PARTS.items():
            tails = {
                _composed(pid)[event][len(voice_for(pid)) :] for pid in available_personality_ids()
            }
            self.assertEqual(len(tails), 1, f"{event} invariant tail differs across personalities")


class LeakAndVerdictClauseSurviveTests(unittest.TestCase):
    """The roster leak clause, misc verdict clause, and recap no-prior-standings
    clause survive every personality verbatim."""

    def test_roster_leak_clause_present_for_every_personality(self) -> None:
        for pid in available_personality_ids():
            self.assertIn(
                _LEAK_CLAUSE,
                _composed(pid)["roster.complete"],
                f"roster.complete under {pid} dropped the count-only leak clause",
            )

    def test_misc_verdict_clause_present_for_every_personality(self) -> None:
        for pid in available_personality_ids():
            self.assertIn(_VERDICT_CLAUSE, _composed(pid)["misc.graded"])

    def test_recap_storyline_clause_present_for_every_personality(self) -> None:
        for pid in available_personality_ids():
            self.assertIn(_RECAP_STORYLINE_CLAUSE, _composed(pid)["week.recap"])


class SwapChangesVoiceNotGuardTests(unittest.TestCase):
    """Swapping the active id changes the voice preamble but not the invariant
    tail."""

    def test_voice_differs_but_tail_identical_between_two_personalities(self) -> None:
        a, b = "sarcastic", "collinsworth"
        self.assertNotEqual(voice_for(a), voice_for(b))
        for event in _EVENT_PARTS:
            prompt_a = _composed(a)[event]
            prompt_b = _composed(b)[event]
            # The composed prompts differ (different voice)...
            self.assertNotEqual(prompt_a, prompt_b)
            # ...but the tail after each voice is identical (same role + guard).
            self.assertEqual(prompt_a[len(voice_for(a)) :], prompt_b[len(voice_for(b)) :])

    def test_guard_text_is_never_inside_any_voice_preamble(self) -> None:
        # No voice preamble may carry guard/clause text — the guard lives only in
        # the composed tail.
        for pid, voice in PERSONALITIES.items():
            self.assertNotIn(chat_personality._FACTS_FIRST_GUARD, voice)
            self.assertNotIn(_LEAK_CLAUSE, voice)
            self.assertNotIn(_VERDICT_CLAUSE, voice)
            self.assertNotIn(_RECAP_STORYLINE_CLAUSE, voice)


class DefaultFallbackTests(unittest.TestCase):
    """The unset setting resolves to the sarcastic default voice/behavior."""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_get_bot_personality_unset_is_sarcastic(self) -> None:
        with Session(self.engine) as session:
            self.assertEqual(app_settings.get_bot_personality(session), DEFAULT_PERSONALITY_ID)

    def test_voice_for_unset_resolves_to_sarcastic_voice(self) -> None:
        self.assertEqual(voice_for(None), PERSONALITIES["sarcastic"])
        self.assertEqual(voice_for("unknown_id"), PERSONALITIES["sarcastic"])


if __name__ == "__main__":
    unittest.main()
