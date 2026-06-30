"""The admin-selectable bot-personality registry (260627-xbb).

The whole pickem-chat layer keeps ONE hard invariant: the bot owns the FACTS and
a byte-identical anti-hallucination / leak-safety GUARD, and the LLM only phrases
them. A "personality" swaps ONLY the leading *voice preamble* of each system
prompt — never the guard, never the per-event ROLE context, never the leak /
verdict / no-prior-standings clauses (those were hard-won across nef -> vpc and
the facts-first fixes be32f8c / edd3b5b and must survive every voice).

This module is the open ``id -> voice preamble`` registry plus the pure
:func:`compose_prompt` helper that concatenates ``<voice> <role line> <guard>``.
The voice preambles are short, tone-only sentences carrying NO facts and NO
guard text. ``sarcastic`` is the DEFAULT and is byte-identical to the current
leading house-bot sentence so an unset / unreadable setting reproduces today's
behavior exactly.

Resolution of the ACTIVE voice happens inside the db_bridge async/thread seam
(:func:`app.bot.db_bridge.resolve_active_voice_async`), never from the pure
``llm_client.phrase`` string layer.
"""

from __future__ import annotations

# The default id. Unset / unreadable / unknown all resolve to this so behavior is
# unchanged from before the swap feature existed.
DEFAULT_PERSONALITY_ID = "sarcastic"

# id -> voice preamble. A voice is a short, tone-only leading sentence: it sets
# the persona and NOTHING else (no facts, no guard, no per-event role). The
# sarcastic voice is byte-identical to the prior hardcoded house-bot lead so the
# default reproduces today's prompts exactly.
PERSONALITIES: dict[str, str] = {
    # The CURRENT snarky/sardonic house bot, kept verbatim as the default voice.
    "sarcastic": "You are the snarky house bot for an NFL pick'em league.",
    # Analytical, numbers-forward, dry — leans into the data without the snark.
    "stats_nerd": (
        "You are the resident stats-nerd bot for an NFL pick'em league: "
        "analytical, numbers-forward, and dry, you let the figures do the talking."
    ),
    # An enthusiastic NFL booth-announcer impression (Cris Collinsworth flavor).
    # Leans into the signature booth cadence and lead-ins on purpose (Zach's call).
    "collinsworth": (
        "You are an excitable prime-time NFL color commentator calling an NFL "
        "pick'em league like a marquee broadcast booth: breathless, enthusiastic, "
        "and quick with breakdowns. Lean into the signature booth tics — open a "
        'breakdown with a lead-in like "Now, here\'s a guy..." and gush over the '
        "standout player or pick."
    ),
}


def available_personality_ids() -> list[str]:
    """Return the registry's known personality ids (the DEFAULT listed first).

    Order is stable and deterministic: the default id leads, then the remaining
    ids in registry insertion order — so the admin selector renders the current /
    safe option first.
    """
    rest = [pid for pid in PERSONALITIES if pid != DEFAULT_PERSONALITY_ID]
    return [DEFAULT_PERSONALITY_ID, *rest]


def voice_for(personality_id: str | None) -> str:
    """Resolve a personality id to its voice preamble, defaulting to sarcastic.

    Best-effort: an unset (``None`` / blank) OR unknown id resolves to the
    DEFAULT sarcastic voice so a missing / stale setting can never break or blank
    the prompt — the bot keeps its current voice.
    """
    if not personality_id:
        return PERSONALITIES[DEFAULT_PERSONALITY_ID]
    return PERSONALITIES.get(personality_id, PERSONALITIES[DEFAULT_PERSONALITY_ID])


def compose_prompt(voice: str, role_line: str, guard: str) -> str:
    """Compose a full system prompt as ``<voice> <role line> <guard>``.

    Pure and deterministic: a single space joins the three already-built parts.
    The ``guard`` (and any invariant leak / verdict / no-prior-standings clause it
    or ``role_line`` carries) is passed through verbatim — this helper NEVER edits
    or relocates it, so the invariant tail stays byte-identical for every voice.
    """
    return f"{voice} {role_line} {guard}"
