"""Offline unit tests for the @mention Q&A classifier + validator (260709-k5w Task 1).

These tests NEVER touch a live LLM endpoint. Two seams are exercised:

* ``qa.llm_client.classify`` is monkeypatched with an async fake returning canned
  JSON / ``None`` / a raise, so :func:`app.bot.qa.classify_question` can be driven
  offline (mirrors the monkeypatch style of ``tests/test_chat_personality.py``).
* A REGRESSION test exercises the REAL ``llm_client.classify`` with ``httpx``
  monkeypatched (as in ``tests/test_llm_commentary.py``) to capture the request body
  and PROVE the classifier is NOT fed the closer-variety chat directive and decodes
  deterministically — so routing the classifier back through ``phrase`` cannot
  silently regress.

The pure, DB-free :func:`app.bot.qa.validate_classification` is called directly with
a hand-built ``known_team_tokens`` set (no db, no network).

Run with: ``backend/.venv/bin/python -m unittest tests.test_qa_classifier -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import httpx

from app.bot import chat_personality, llm_client, qa
from app.bot.qa import QaIntent, QaResult, _normalize_team, validate_classification
from app.config import settings


def _run(coro):
    return asyncio.run(coro)


# A hand-built real-team token set (abbreviations + display-name tokens). The
# validator only checks membership — it never touches a db.
_KNOWN_TEAMS = {"KC", "CHIEFS", "SF", "49ERS", "PHI", "EAGLES", "DAL", "COWBOYS"}


def _classify_returns(value):
    """Patch qa.llm_client.classify with an async fake returning ``value``,
    recording the (user_content, system_prompt) it was called with."""
    calls: list[dict] = []

    async def _fake(user_content, *, system_prompt):
        calls.append({"user_content": user_content, "system_prompt": system_prompt})
        return value

    return mock.patch.object(qa.llm_client, "classify", _fake), calls


def _classify_raises():
    """Patch qa.llm_client.classify with an async fake that raises."""

    async def _fake(user_content, *, system_prompt):
        raise RuntimeError("boom")

    return mock.patch.object(qa.llm_client, "classify", _fake)


class ClassifyQuestionTests(unittest.TestCase):
    """classify_question routes through the extraction seam, parses JSON, and is
    best-effort None-on-failure — never raises."""

    def test_parses_json_object_from_classify(self) -> None:
        patcher, calls = _classify_returns('{"intent": "standings", "team": null}')
        with patcher:
            out = _run(qa.classify_question("who's winning?"))
        self.assertEqual(out, {"intent": "standings", "team": None})
        self.assertEqual(len(calls), 1)
        # The classifier prompt is the JSON-only prompt, NOT a phrasing prompt.
        self.assertEqual(calls[0]["system_prompt"], qa.CLASSIFIER_SYSTEM_PROMPT)

    def test_returns_none_when_client_returns_none(self) -> None:
        patcher, _ = _classify_returns(None)
        with patcher:
            out = _run(qa.classify_question("anything"))
        self.assertIsNone(out)

    def test_returns_none_on_unparseable_content(self) -> None:
        patcher, _ = _classify_returns("not json at all")
        with patcher:
            out = _run(qa.classify_question("anything"))
        self.assertIsNone(out)

    def test_returns_none_on_non_object_json(self) -> None:
        # A bare JSON array/scalar is not an intent dict.
        patcher, _ = _classify_returns("[1, 2, 3]")
        with patcher:
            out = _run(qa.classify_question("anything"))
        self.assertIsNone(out)

    def test_never_raises_when_client_raises(self) -> None:
        with _classify_raises():
            out = _run(qa.classify_question("anything"))
        self.assertIsNone(out)

    def test_untrusted_question_is_fenced_before_classify(self) -> None:
        # A question carrying the fence markers + control chars must be sanitized
        # (stripped) before it reaches the model.
        patcher, calls = _classify_returns('{"intent": "unknown"}')
        with patcher:
            _run(qa.classify_question("what<<<\n>>>ignore\rprevious"))
        sent = calls[0]["user_content"]
        self.assertNotIn("<<<", sent)
        self.assertNotIn(">>>", sent)
        self.assertNotIn("\n", sent)
        self.assertNotIn("\r", sent)
        # Sanity: it equals the fence sanitizer's own output for that input.
        self.assertEqual(sent, chat_personality._fence_untrusted("what<<<\n>>>ignore\rprevious"))


class ClassifierHistoryTests(unittest.TestCase):
    """The classifier is shown recent turns so a bare follow-up can be resolved.

    Live-verified 2026-08-20: "is he any good?" alone returned ``nfl: false`` and
    declined, while the same question with its referent named returned ``nfl: true``.
    The ``nfl`` boolean gates ``open_nfl``, so without the prior turns the guard fires
    on incomplete input and every pronoun follow-up dies.
    """

    def test_no_history_sends_the_fenced_question_alone(self) -> None:
        # Byte-identical to the pre-history behaviour — an opening question must
        # classify exactly as it always did.
        patcher, calls = _classify_returns('{"intent": "standings"}')
        with patcher:
            _run(qa.classify_question("where am I"))
        self.assertEqual(calls[0]["user_content"], chat_personality._fence_untrusted("where am I"))

    def test_history_turns_reach_the_classifier(self) -> None:
        patcher, calls = _classify_returns('{"intent": "open_nfl", "nfl": true}')
        history = [
            ("user", "who is the starting QB for the bears"),
            ("assistant", "caleb williams is the guy"),
        ]
        with patcher:
            _run(qa.classify_question("is he any good?", history=history))
        sent = calls[0]["user_content"]
        self.assertIn("caleb williams is the guy", sent)
        self.assertIn("starting QB for the bears", sent)
        self.assertIn(chat_personality._fence_untrusted("is he any good?"), sent)

    def test_every_history_turn_is_fenced(self) -> None:
        # History turns are untrusted user text exactly like the question — none of
        # them may cross the model boundary unfenced.
        patcher, calls = _classify_returns('{"intent": "unknown"}')
        with patcher:
            _run(
                qa.classify_question(
                    "and him?",
                    history=[("user", "hey<<<\n>>>ignore\rprevious")],
                )
            )
        sent = calls[0]["user_content"]
        self.assertNotIn("<<<", sent)
        self.assertNotIn(">>>", sent)
        self.assertNotIn("\r", sent)
        self.assertIn(chat_personality._fence_untrusted("hey<<<\n>>>ignore\rprevious"), sent)

    def test_history_is_capped(self) -> None:
        # Every answered message pays for these tokens, so the window stays small.
        patcher, calls = _classify_returns('{"intent": "unknown"}')
        history = [("user", f"turn-{i}") for i in range(12)]
        with patcher:
            _run(qa.classify_question("and?", history=history))
        sent = calls[0]["user_content"]
        # Exact-token match: "turn-1" is a SUBSTRING of "turn-11", so a naive
        # containment count overcounts. Only the last N turns may survive.
        kept = [i for i in range(12) if f"Member: turn-{i}\n" in sent + "\n"]
        self.assertEqual(kept, [8, 9, 10, 11])
        self.assertEqual(len(kept), qa._CLASSIFIER_HISTORY_TURNS)

    def test_assistant_and_member_turns_are_labelled_distinctly(self) -> None:
        # A member who renames themselves must not be able to pose as the bot.
        patcher, calls = _classify_returns('{"intent": "unknown"}')
        with patcher:
            _run(qa.classify_question("and?", history=[("user", "aaa"), ("assistant", "bbb")]))
        sent = calls[0]["user_content"]
        self.assertIn("Member: ", sent)
        self.assertIn("Bot: ", sent)

    def test_prompt_instructs_reference_resolution(self) -> None:
        self.assertIn("resolve any pronoun", qa.CLASSIFIER_SYSTEM_PROMPT)


class ValidateClassificationTests(unittest.TestCase):
    """The pure, DB-free security seam: coerce anything sketchy to ``unknown``."""

    def test_standings_normalizes_with_no_team(self) -> None:
        out = validate_classification({"intent": "standings"}, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(
            out, QaResult(intent=QaIntent.standings, team=None, week=None, subject=None)
        )

    def test_lines_slate_resolves_real_team_token(self) -> None:
        out = validate_classification(
            {"intent": "lines_slate", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.lines_slate)
        # Resolves to a REAL 32-team token (case-insensitive match against the set).
        self.assertIn(out.team, _KNOWN_TEAMS)
        self.assertEqual(out.team, "CHIEFS")

    def test_lines_slate_team_is_case_insensitive(self) -> None:
        out = validate_classification(
            {"intent": "lines_slate", "team": "  kc "}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.lines_slate)
        self.assertEqual(out.team, "KC")

    def test_non_real_team_coerces_to_unknown(self) -> None:
        out = validate_classification(
            {"intent": "lines_slate", "team": "Narnia"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_off_enum_intent_coerces_to_unknown(self) -> None:
        out = validate_classification({"intent": "teleport"}, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_model_emitted_unknown_stays_unknown(self) -> None:
        # The genuine-nonsense path is untouched by the news team-OPTIONAL fix: an
        # intent the MODEL itself classified as unknown stays unknown (unknown is not a
        # team-bearing intent, so a stray team is simply dropped, never resurrected).
        out = validate_classification(
            {"intent": "unknown", "team": "moon"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.unknown)
        self.assertIsNone(out.team)

    def test_none_input_coerces_to_unknown(self) -> None:
        out = validate_classification(None, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_empty_dict_coerces_to_unknown(self) -> None:
        out = validate_classification({}, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_non_dict_input_coerces_to_unknown(self) -> None:
        for bad in ("a string", 42, ["intent"], object()):
            out = validate_classification(bad, known_team_tokens=_KNOWN_TEAMS)
            self.assertEqual(out.intent, QaIntent.unknown)

    def test_coming_soon_is_a_legal_value_not_coerced(self) -> None:
        out = validate_classification({"intent": "coming_soon"}, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(out.intent, QaIntent.coming_soon)

    def test_bot_help_is_a_legal_value_not_coerced(self) -> None:
        out = validate_classification({"intent": "bot_help"}, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(out.intent, QaIntent.bot_help)

    def test_bot_help_takes_no_params(self) -> None:
        # bot_help is in none of the team/week/subject intent sets, so every param
        # scrubs to None for free — "how do I register for the Chiefs in week 3"
        # still answers with the plain command list.
        out = validate_classification(
            {"intent": "bot_help", "team": "Chiefs", "week": 3, "subject": "registering"},
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(out.intent, QaIntent.bot_help)
        self.assertIsNone(out.team)
        self.assertIsNone(out.week)
        self.assertIsNone(out.subject)

    def test_bot_help_is_offered_in_the_classifier_prompt(self) -> None:
        # The intent is unreachable unless the prompt tells the model it exists.
        self.assertIn("bot_help", qa.CLASSIFIER_SYSTEM_PROMPT)

    def test_team_dropped_for_intent_without_team(self) -> None:
        # A team field on standings (which takes no team) is dropped, not an error.
        out = validate_classification(
            {"intent": "standings", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.standings)
        self.assertIsNone(out.team)

    def test_week_coerced_into_range(self) -> None:
        good = validate_classification(
            {"intent": "scores", "week": 5}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(good.week, 5)
        out_of_range = validate_classification(
            {"intent": "scores", "week": 99}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertIsNone(out_of_range.week)

    def test_validator_is_pure_no_db_no_network(self) -> None:
        # Calling it with a hand-built set and no patched db/network proves purity:
        # if it touched a db or the network this call would fail in the offline suite.
        out = validate_classification(
            {"intent": "lines_slate", "team": "SF", "week": 3, "subject": "spread"},
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(
            out, QaResult(intent=QaIntent.lines_slate, team="SF", week=3, subject="spread")
        )


class InjuriesClassificationTests(unittest.TestCase):
    """The injuries intent is a REAL team-bearing intent (graduated out of coming_soon)."""

    def test_injuries_validates_as_real_intent_with_resolved_team(self) -> None:
        out = validate_classification(
            {"intent": "injuries", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.injuries)
        # A team-bearing intent: the real token is resolved + carried through.
        self.assertEqual(out.team, "CHIEFS")

    def test_injuries_non_real_team_coerces_to_unknown(self) -> None:
        out = validate_classification(
            {"intent": "injuries", "team": "Narnia"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_teamless_injuries_stays_injuries_with_no_team(self) -> None:
        # A teamless injuries question is a VALID injuries result (team None) — the
        # soft-decline is decided downstream in _build_fact, not coerced to unknown.
        out = validate_classification(
            {"intent": "injuries", "team": None}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.injuries)
        self.assertIsNone(out.team)

    def test_injuries_left_the_coming_soon_lane_in_the_prompt(self) -> None:
        # Regression: injuries must be a first-class intent in the prompt, and must no
        # longer be listed among the coming_soon recognized-but-unsupported topics.
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("injuries (a team's injury report", prompt)
        self.assertNotIn("injuries, weather, news", prompt)


class WeatherClassificationTests(unittest.TestCase):
    """The weather intent is a REAL team-bearing intent (graduated out of coming_soon)."""

    def test_weather_validates_as_real_intent_with_resolved_team(self) -> None:
        out = validate_classification(
            {"intent": "weather", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.weather)
        # A team-bearing intent: the real token is resolved + carried through.
        self.assertEqual(out.team, "CHIEFS")

    def test_weather_non_real_team_coerces_to_unknown(self) -> None:
        out = validate_classification(
            {"intent": "weather", "team": "Narnia"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_teamless_weather_stays_weather_with_no_team(self) -> None:
        # A teamless weather question is a VALID weather result (team None) — the
        # soft-decline is decided downstream in _build_fact, not coerced to unknown.
        out = validate_classification(
            {"intent": "weather", "team": None}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.weather)
        self.assertIsNone(out.team)

    def test_weather_left_the_coming_soon_lane_in_the_prompt(self) -> None:
        # Regression: weather must be a first-class intent in the prompt, and must no
        # longer be listed among the coming_soon recognized-but-unsupported topics.
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("weather (", prompt)
        self.assertIn("weather (the game-time forecast", prompt)
        self.assertNotIn("topic: weather", prompt)
        self.assertNotIn("weather, news", prompt)


class NewsClassificationTests(unittest.TestCase):
    """The news intent is a REAL team-OPTIONAL intent (graduated out of coming_soon)."""

    def test_news_validates_as_real_intent_with_resolved_team(self) -> None:
        out = validate_classification(
            {"intent": "news", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.news)
        # A team-bearing intent: the real token is resolved + carried through.
        self.assertEqual(out.team, "CHIEFS")

    def test_news_non_real_team_falls_back_to_league_team_none(self) -> None:
        # news is team-OPTIONAL: a NON-REAL team (a division like "AFC West", "NFC", or
        # a misspelled team) is NOT a coercion trigger — it falls through to team=None so
        # the LEAGUE answer is delivered downstream, NOT the unknown capability menu. This
        # is the flipped issue-#114 contract (the old behavior coerced this to unknown).
        out = validate_classification(
            {"intent": "news", "team": "Narnia"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.news)
        self.assertIsNone(out.team)

    def test_news_non_real_team_keeps_subject_for_league_narrowing(self) -> None:
        # The exact issue-#114 shape at the validator level: the classifier put the
        # division into the team slot AND the subject. The non-real team falls through to
        # None, but the subject SURVIVES (news is in _SUBJECT_INTENTS) so it can narrow the
        # league feed via filter_news_by_subject downstream.
        out = validate_classification(
            {"intent": "news", "team": "AFC West", "subject": "AFC West"},
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(
            out, QaResult(intent=QaIntent.news, team=None, week=None, subject="AFC West")
        )

    def test_teamless_news_stays_news_with_no_team(self) -> None:
        # A teamless news question is a VALID news result (team None) — the LEAGUE
        # answer is decided downstream in _build_fact, NOT coerced to unknown (this is
        # the key news-specific difference from injuries/weather).
        out = validate_classification(
            {"intent": "news", "team": None}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.news)
        self.assertIsNone(out.team)

    def test_news_left_the_coming_soon_lane_in_the_prompt(self) -> None:
        # Regression: news must be a first-class intent in the prompt, and must no
        # longer be listed among the coming_soon recognized-but-unsupported topics.
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("news (", prompt)
        self.assertIn("news (recent ESPN headlines", prompt)
        self.assertNotIn("topic: news", prompt)


class PredictionClassificationTests(unittest.TestCase):
    """The prediction intent is a REAL team-bearing, team-REQUIRED intent (260710-mpw)."""

    def test_prediction_validates_as_real_intent_with_resolved_team(self) -> None:
        out = validate_classification(
            {"intent": "prediction", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.prediction)
        # Team-bearing: the real token is resolved + carried through.
        self.assertEqual(out.team, "CHIEFS")

    def test_prediction_non_real_team_coerces_to_unknown(self) -> None:
        # Team-REQUIRED: a non-real team named on a prediction is untrustworthy -> unknown.
        out = validate_classification(
            {"intent": "prediction", "team": "Narnia"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.unknown)

    def test_teamless_prediction_stays_prediction_with_no_team(self) -> None:
        # A null team stays a VALID prediction (team None) — the downstream soft-decline
        # ("name a team") is decided in _build_fact, NOT coerced to unknown here.
        out = validate_classification(
            {"intent": "prediction", "team": None}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.prediction)
        self.assertIsNone(out.team)

    def test_prediction_is_first_class_in_prompt_and_who_wins_left_coming_soon(self) -> None:
        # Regression: prediction is a first-class prompt intent, and the who-will-win
        # topic no longer rides in the coming_soon wink (only line movement remains there).
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("prediction (who will win a specific team's game", prompt)
        self.assertIn("coming_soon (a recognized but unsupported topic: line movement)", prompt)
        self.assertNotIn("who-will-win prediction", prompt)
        self.assertNotIn("or a who-will-win", prompt)

    def test_coming_soon_still_validates(self) -> None:
        out = validate_classification({"intent": "coming_soon"}, known_team_tokens=_KNOWN_TEAMS)
        self.assertEqual(out.intent, QaIntent.coming_soon)


class SlatePredictionsClassificationTests(unittest.TestCase):
    """slate_predictions is the whole-slate model-opinion intent (260713-k6z), first-class
    and disambiguated from single-team prediction and factual lines_slate."""

    def test_slate_predictions_validates_as_real_intent_no_team(self) -> None:
        # A "what are your picks this week?" style ask -> the whole-slate intent. It is NOT
        # team-bearing, so a stray team is simply dropped (never resurrected).
        out = validate_classification(
            {"intent": "slate_predictions"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.slate_predictions)
        self.assertIsNone(out.team)

    def test_slate_predictions_drops_a_stray_team(self) -> None:
        out = validate_classification(
            {"intent": "slate_predictions", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.slate_predictions)
        self.assertIsNone(out.team)  # not team-bearing -> dropped, not an error

    def test_single_team_prediction_still_validates_prediction(self) -> None:
        # Disambiguation: a single-team "will KC cover?" stays the single-game prediction.
        out = validate_classification(
            {"intent": "prediction", "team": "Chiefs"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out.intent, QaIntent.prediction)
        self.assertEqual(out.team, "CHIEFS")

    def test_factual_spread_ask_stays_lines_slate(self) -> None:
        # Disambiguation: a factual "what's the spread this week?" stays lines_slate.
        out = validate_classification(
            {"intent": "lines_slate", "team": None, "subject": "the spread"},
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(out.intent, QaIntent.lines_slate)

    def test_slate_predictions_is_first_class_in_prompt(self) -> None:
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("slate_predictions (", prompt)
        # The disambiguation guidance must NOT reintroduce the negative-asserted phrase.
        self.assertNotIn("who-will-win prediction", prompt)


# A token set that CONTAINS the alias targets under test (real abbreviations only —
# _normalize_team's real-set guard means an alias only resolves when its canonical
# abbreviation is itself a member of this set).
_ALIAS_TOKENS = {"SF", "PHI", "DEN", "NYG", "SEA", "DAL", "KC", "CHIEFS"}


class NormalizeTeamAliasTests(unittest.TestCase):
    """Deterministic slang nickname aliases folded into the shared _normalize_team
    chokepoint — a pure FALLBACK that never shadows real resolution and never emits
    a non-real team."""

    def test_alias_resolves_to_canonical_abbr(self) -> None:
        # A curated slang nickname resolves to its canonical team abbreviation,
        # independent of the classifier.
        self.assertEqual(_normalize_team("niners", _ALIAS_TOKENS), "SF")
        self.assertEqual(_normalize_team("iggles", _ALIAS_TOKENS), "PHI")

    def test_alias_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(_normalize_team("  Donkeys ", _ALIAS_TOKENS), "DEN")
        self.assertEqual(_normalize_team("9ERS", _ALIAS_TOKENS), "SF")

    def test_real_tokens_unchanged_alias_is_pure_fallback(self) -> None:
        # The alias path NEVER shadows real resolution: a real abbreviation / display
        # token resolves exactly as before (the alias branch is only reached on a miss).
        self.assertEqual(_normalize_team("KC", _ALIAS_TOKENS), "KC")
        self.assertEqual(_normalize_team("Chiefs", _ALIAS_TOKENS), "CHIEFS")

    def test_unknown_garbage_still_none(self) -> None:
        self.assertIsNone(_normalize_team("narnia", _ALIAS_TOKENS))

    def test_defensive_guard_alias_target_absent_from_set(self) -> None:
        # Even though "donkeys"->DEN is in the map, DEN is NOT in this set (an unseeded
        # / partial DB or a map typo): never emit a non-real team.
        tokens_without_den = {"SF", "PHI", "KC", "CHIEFS"}
        self.assertIsNone(_normalize_team("donkeys", tokens_without_den))

    def test_ambiguous_slang_excluded_from_map(self) -> None:
        # "birds" maps to >1 team (Eagles/Cardinals/Ravens/Seahawks) so it is NOT in
        # the map and does not resolve.
        self.assertIsNone(_normalize_team("birds", _ALIAS_TOKENS))

    def test_alias_resolves_end_to_end_through_injuries(self) -> None:
        out = validate_classification(
            {"intent": "injuries", "team": "donkeys"}, known_team_tokens=_ALIAS_TOKENS
        )
        self.assertEqual(out, QaResult(intent=QaIntent.injuries, team="DEN"))

    def test_alias_resolves_end_to_end_through_news(self) -> None:
        out = validate_classification(
            {"intent": "news", "team": "niners"}, known_team_tokens=_ALIAS_TOKENS
        )
        self.assertEqual(out, QaResult(intent=QaIntent.news, team="SF"))

    def test_alias_key_normalizes_spaces_hyphens_apostrophes(self) -> None:
        # Multi-word / punctuated slang all hits the same normalized alias entry, so
        # "big blue", "Big-Blue", "'boys" resolve like their compact keys.
        self.assertEqual(_normalize_team("big blue", _ALIAS_TOKENS), "NYG")
        self.assertEqual(_normalize_team("Big-Blue", _ALIAS_TOKENS), "NYG")
        self.assertEqual(_normalize_team("'boys", _ALIAS_TOKENS), "DAL")
        self.assertEqual(_normalize_team("go birds", _ALIAS_TOKENS), "PHI")

    def test_crude_and_user_added_aliases_resolve(self) -> None:
        # Crude fan slang is intentional (routes a fan's input to the team); "chefs" is
        # the hand-added KC alias.
        self.assertEqual(_normalize_team("qweefs", _ALIAS_TOKENS), "KC")
        self.assertEqual(_normalize_team("chefs", _ALIAS_TOKENS), "KC")
        self.assertEqual(_normalize_team("cowgirls", _ALIAS_TOKENS), "DAL")

    def test_all_alias_targets_are_real_abbreviations_and_unambiguous(self) -> None:
        # Every alias must target one of the 32 real abbreviations, and no key may be a
        # bare display-word that would shadow (each maps to EXACTLY one team).
        from app.bot.qa import _TEAM_ALIASES

        real_abbrs = {
            "ARI",
            "ATL",
            "BAL",
            "BUF",
            "CAR",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "HOU",
            "IND",
            "JAX",
            "KC",
            "LV",
            "LAC",
            "LAR",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "PIT",
            "SF",
            "SEA",
            "TB",
            "TEN",
            "WSH",
        }
        for key, abbr in _TEAM_ALIASES.items():
            self.assertIn(abbr, real_abbrs, msg=f"{key!r} -> {abbr!r} is not a real abbr")
            # keys are already normalized (lowercase alphanumerics only)
            self.assertEqual(key, "".join(c for c in key.lower() if c.isalnum()), msg=key)


# --------------------------------------------------------------------------- #
# 260820-lw6 — the open_nfl intent + the DETERMINISTIC NFL topic guard.
#
# The greedy-route defect this closes: with no legal home for an off-menu football
# question, "who is the starting QB for the Bears" routed to ``injuries`` (the only
# intent described as "a team plus a player") and the validator passed it.
# --------------------------------------------------------------------------- #


class OpenNflValidationTests(unittest.TestCase):
    def test_open_nfl_with_boolean_true_nfl_key_survives(self) -> None:
        out = validate_classification(
            {"intent": "open_nfl", "nfl": True}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(out, QaResult(intent=QaIntent.open_nfl))

    def test_open_nfl_drops_team_week_and_subject(self) -> None:
        # open_nfl is in NONE of the three param frozensets — the open path reads the
        # RAW question, so a classifier-supplied team/week/subject is scrubbed away.
        out = validate_classification(
            {
                "intent": "open_nfl",
                "nfl": True,
                "team": "KC",
                "week": 4,
                "subject": "starting quarterback",
            },
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(
            out, QaResult(intent=QaIntent.open_nfl, team=None, week=None, subject=None)
        )

    def test_topic_guard_fails_closed_on_every_non_boolean_true_nfl_value(self) -> None:
        # Identity against True, NOT truthiness: a missing key, a string, a 1, and a
        # literal false must ALL decline (the measured lasagna-recipe case).
        for raw in (
            {"intent": "open_nfl"},
            {"intent": "open_nfl", "nfl": False},
            {"intent": "open_nfl", "nfl": None},
            {"intent": "open_nfl", "nfl": "true"},
            {"intent": "open_nfl", "nfl": 1},
            {"intent": "open_nfl", "nfl": "yes"},
        ):
            with self.subTest(raw=raw):
                out = validate_classification(raw, known_team_tokens=_KNOWN_TEAMS)
                self.assertEqual(out, QaResult(intent=QaIntent.unknown))

    def test_nfl_key_does_not_affect_the_ten_grounded_intents(self) -> None:
        # The topic guard is scoped to open_nfl ONLY — a grounded intent is unchanged
        # whether or not the classifier bothered to emit the new key.
        with_key = validate_classification(
            {"intent": "injuries", "team": "KC", "nfl": False}, known_team_tokens=_KNOWN_TEAMS
        )
        without_key = validate_classification(
            {"intent": "injuries", "team": "KC"}, known_team_tokens=_KNOWN_TEAMS
        )
        self.assertEqual(with_key, QaResult(intent=QaIntent.injuries, team="KC"))
        self.assertEqual(with_key, without_key)

    def test_open_nfl_is_absent_from_all_three_param_frozensets(self) -> None:
        self.assertNotIn(QaIntent.open_nfl, qa._TEAM_INTENTS)
        self.assertNotIn(QaIntent.open_nfl, qa._WEEK_INTENTS)
        self.assertNotIn(QaIntent.open_nfl, qa._SUBJECT_INTENTS)


class OpenNflClassifierPromptTests(unittest.TestCase):
    """An enum member the prompt never names is dead; a prompt intent the validator
    rejects silently coerces back to unknown. Both halves must be wired."""

    def test_prompt_names_the_open_intent_and_the_new_key(self) -> None:
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("open_nfl", prompt)
        self.assertIn('"nfl"', prompt)

    def test_prompt_still_names_every_grounded_intent(self) -> None:
        # ADD-ONLY: none of the ten grounded intents lost its description.
        for intent in (
            "pick_status",
            "standings",
            "lines_slate",
            "scores",
            "injuries",
            "weather",
            "news",
            "prediction",
            "slate_predictions",
            "bot_help",
            "coming_soon",
            "unknown",
        ):
            with self.subTest(intent=intent):
                self.assertIn(intent, qa.CLASSIFIER_SYSTEM_PROMPT)

    def test_every_enum_member_is_named_in_the_prompt(self) -> None:
        for member in QaIntent:
            with self.subTest(member=member.value):
                self.assertIn(member.value, qa.CLASSIFIER_SYSTEM_PROMPT)


# --------------------------------------------------------------------------- #
# REGRESSION: the classifier must NOT be fed the closer-variety directive and MUST
# decode deterministically. Exercise the REAL llm_client.classify with httpx
# monkeypatched to capture the request body.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _CapturingAsyncClient:
    last_json: dict | None = None
    _response: object = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url, *, json=None, headers=None):  # noqa: A002
        type(self).last_json = json
        return self._response


def _configured():
    return mock.patch.multiple(
        settings,
        llm_api_server="http://llm:8000/v1",
        llm_api_model="gemma",
        llm_api_key="secret-key",
    )


class ClassifyWireFormatRegressionTests(unittest.TestCase):
    """The security blocker: classify must send the caller's prompt VERBATIM (no
    closer-variety), decode near-deterministically, and use the JSON max_tokens."""

    def test_classify_omits_closer_variety_and_is_deterministic(self) -> None:
        _CapturingAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": '{"intent": "unknown"}'}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(llm_client.classify("q", system_prompt=qa.CLASSIFIER_SYSTEM_PROMPT))
        self.assertEqual(out, '{"intent": "unknown"}')

        body = _CapturingAsyncClient.last_json
        assert body is not None
        system_msg = body["messages"][0]["content"]
        # (a) NOT fed the closer-variety directive, and equals the caller's prompt verbatim.
        self.assertNotIn("Vary your closing line", system_msg)
        self.assertEqual(system_msg, qa.CLASSIFIER_SYSTEM_PROMPT)
        # (b) deterministic sampling + JSON-sized max_tokens (not the 80-token chat cap).
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], llm_client._CLASSIFY_MAX_TOKENS)
        self.assertGreater(body["max_tokens"], llm_client._MAX_TOKENS)
        # HARD wire rule still holds — enable_thinking False or content comes back empty.
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)

    def test_phrase_still_appends_closer_variety(self) -> None:
        # Guard the other direction: phrase's behavior is unchanged.
        _CapturingAsyncClient._response = _FakeResponse(
            200, {"choices": [{"message": {"content": "hi"}}]}
        )
        with _configured(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            _run(llm_client.phrase("fact", system_prompt="VOICE ROLE GUARD"))
        body = _CapturingAsyncClient.last_json
        assert body is not None
        self.assertIn("Vary your closing line", body["messages"][0]["content"])


class ScoresClassifierPromptTests(unittest.TestCase):
    """The measured cause of #187. Each fragment is chosen to BREAK if the clause is
    softened back into a bare limitation (the #179 failure: a route described only by
    what it does not cover was skipped 5 times out of 5)."""

    def test_scores_is_scoped_to_the_current_week_score(self) -> None:
        self.assertIn(
            "the final or in-progress SCORE of a game in the CURRENT week",
            qa.CLASSIFIER_SYSTEM_PROMPT,
        )

    def test_statistics_are_routed_to_open_nfl_as_an_instruction(self) -> None:
        prompt = qa.CLASSIFIER_SYSTEM_PROMPT
        self.assertIn("a player's or a team's STATISTICS in a game", prompt)
        self.assertIn("is open_nfl, NOT scores", prompt)

    def test_the_pronoun_case_is_named(self) -> None:
        self.assertIn(
            'even when the question calls the game "that game"', qa.CLASSIFIER_SYSTEM_PROMPT
        )

    def test_an_earlier_season_is_open_nfl(self) -> None:
        self.assertIn(
            "a game in an EARLIER season is open_nfl, not scores", qa.CLASSIFIER_SYSTEM_PROMPT
        )


class GameStatisticsPredicateTests(unittest.TestCase):
    """The pure subject predicate behind the stats-phrased ``scores`` guard (#187)."""

    def test_statistics_phrases_hit(self) -> None:
        for subject in (
            "passing yards",
            "yards passing",
            "yardage",
            "rushing",
            "receiving",
            "touchdowns",
            "sacks",
            "tackles",
            "interceptions",
            "completions",
            "carries",
            "receptions",
            "catches",
            "stats",
            "statistics",
            "who led the game",
            "leader in rushing",
            "passer rating",
            "attempts",
            "fumbles",
            "targets",
        ):
            with self.subTest(subject=subject):
                self.assertTrue(qa._wants_game_statistics(subject))

    def test_score_phrases_miss(self) -> None:
        # The regression pin: "points" and "score" are legitimately the scores intent.
        # "status" and "called" are the measured bare-substring traps ("stat", "led").
        for subject in (
            "points",
            "how many points did the vikings score",
            "score",
            "final score",
            "who won",
            "status",
            "the game they called",
            "standings",
            "",
            None,
        ):
            with self.subTest(subject=subject):
                self.assertFalse(qa._wants_game_statistics(subject))


class StatsPhrasedScoresGuardTests(unittest.TestCase):
    """Issue #187: a statistics question the model labelled ``scores`` was answered
    with the CURRENT week's scoreboard, because the scores handler ignores both the
    team and the week. The guard reroutes it to the open path."""

    def test_stats_phrased_scores_becomes_open_nfl_with_no_params(self) -> None:
        out = validate_classification(
            {
                "intent": "scores",
                "subject": "passing yards in that game",
                "nfl": True,
                "team": "SEA",
                "week": 5,
            },
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(out, QaResult(intent=QaIntent.open_nfl))

    def test_guard_fails_closed_on_every_non_boolean_true_nfl_value(self) -> None:
        # Same identity-against-True rule as the topic guard above it.
        for raw in (
            {"intent": "scores", "subject": "passing yards in that game"},
            {"intent": "scores", "subject": "passing yards in that game", "nfl": False},
            {"intent": "scores", "subject": "passing yards in that game", "nfl": None},
            {"intent": "scores", "subject": "passing yards in that game", "nfl": "true"},
            {"intent": "scores", "subject": "passing yards in that game", "nfl": 1},
        ):
            with self.subTest(raw=raw):
                out = validate_classification(raw, known_team_tokens=_KNOWN_TEAMS)
                self.assertEqual(out, QaResult(intent=QaIntent.unknown))

    def test_a_real_score_question_is_untouched(self) -> None:
        out = validate_classification(
            {"intent": "scores", "subject": "how many points did the vikings score", "week": 5},
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(out, QaResult(intent=QaIntent.scores, week=5))

    def test_guard_is_scoped_to_the_scores_intent(self) -> None:
        lines = validate_classification(
            {"intent": "lines_slate", "subject": "passing yards", "nfl": True},
            known_team_tokens=_KNOWN_TEAMS,
        )
        news = validate_classification(
            {"intent": "news", "subject": "passing yards", "nfl": True},
            known_team_tokens=_KNOWN_TEAMS,
        )
        self.assertEqual(lines, QaResult(intent=QaIntent.lines_slate, subject="passing yards"))
        self.assertEqual(news, QaResult(intent=QaIntent.news, subject="passing yards"))

    def test_scores_is_absent_from_subject_intents(self) -> None:
        # WHY the guard reads raw["subject"] before the scrub. If scores is ever added
        # to this frozenset, the pre-scrub read is still correct — but deliberate.
        self.assertNotIn(QaIntent.scores, qa._SUBJECT_INTENTS)


if __name__ == "__main__":
    unittest.main()
