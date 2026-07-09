"""Best-effort local-LLM client for the pickem-chat personality layer.

The BOT owns the FACTS (a deterministic scanner for repeated picks; the notifier
for the reactive chat events); this client ONLY phrases a supplied fact into one
short line. :func:`phrase` is the general core — it takes the system prompt as a
parameter so any event can supply its own persona (260627-t5u). :func:`phrase_pattern`
is the back-compat thin wrapper for the 260627-nef repeated-pick path: it delegates
to :func:`phrase` with the repeated-pick prompt and is unchanged for its callers.

Both are best-effort by contract: they NEVER raise and return ``None`` on ANY
failure (timeout, non-200, empty content, or the feature being unconfigured), so
the caller can fall back to its deterministic line and the notifier loop survives.

HARD wire-format rule (T-nef-04 + the served-model quirk): the request body MUST
carry ``chat_template_kwargs: {"enable_thinking": False}`` — without it the served
gemma reasoning model emits its thinking trace and returns EMPTY visible content.

No ``discord`` import (this module stays on the Discord-free side). Uses
``httpx`` (already a dependency).
"""

from __future__ import annotations

import httpx
import structlog

from app.bot.personality import DEFAULT_PERSONALITY_ID, PERSONALITIES, compose_prompt
from app.config import settings

logger = structlog.get_logger(__name__)

# Keep the line short — this is one chat quip, not an essay.
_MAX_TOKENS = 80
# Modestly raised (0.9 → 1.0) for lexical diversity; bounded by _TOP_P nucleus
# sampling so the low-probability tail (where invented facts live) stays capped.
_TEMPERATURE = 1.0
# Nucleus sampling: keep the top 95% probability mass, cutting the long tail. This
# is the higher-risk knob — committed separately from the Task 1 prompt fix so it can
# be reverted alone if a live capture ever shows fact drift.
_TOP_P = 0.95
_TIMEOUT_SECONDS = 10.0

# --------------------------------------------------------------------------- #
# Deterministic JSON EXTRACTION seam (260709-k5w). The @mention Q&A classifier
# needs the model to emit ONLY a compact JSON object — the exact opposite of the
# chat-quip decode above. So :func:`classify` decodes near-deterministically (no
# nucleus wandering into off-JSON prose), gives the object room to close (a larger
# max_tokens than the 80-token chat cap), and — critically — does NOT append the
# ``_CLOSER_VARIETY`` chat-styling directive, whose ~130 words of "vary your kicker"
# sabotage an "emit ONLY JSON" instruction on the small local Gemma. It keeps the
# SAME best-effort contract as :func:`phrase` (None on any failure, never raises)
# and the SAME mandatory ``chat_template_kwargs.enable_thinking = False``.
# --------------------------------------------------------------------------- #
# Near-deterministic decode: greedy so the same question maps to the same JSON.
_CLASSIFY_TEMPERATURE = 0.0
_CLASSIFY_TOP_P = 1.0
# JSON-appropriate cap: a compact ``{intent, team, week, subject}`` object needs
# far more than the 80-token chat cap (which can truncate JSON to unparseable), but
# nothing essay-sized.
_CLASSIFY_MAX_TOKENS = 256

# Style-only anti-repetition directive appended to EVERY phrasing call, AFTER the
# caller's facts-first guard (so facts-first still leads). It fights the stock-closer
# collapse — the model anchoring on one metaphor (e.g. reusing "maybe try a crystal
# ball next week? 📉" across unrelated failed picks). It licenses NO new fact: it only
# changes the SHAPE of the closer. Lives OUTSIDE every guard/ROLE constant so the
# byte-identical guard invariants (test_personality.py) stay green. Lead phrase
# ("Vary your closing line") is stable so wire-format tests can grep for it.
_CLOSER_VARIETY = (
    "Vary your closing line every single time — never lean on a stock kicker or reuse "
    "the same closing metaphor from one message to the next, and steer clear of the "
    'canned "maybe try a crystal ball next week" / "better luck next week" trap. '
    'Do NOT open your kicker with "Imagine ..." ("Imagine betting on...", "Imagine '
    'being the one who...", "Imagine actually...") — that opener is its own stock '
    "scaffold; reach for a different construction. "
    "Rotate the SHAPE of your closer: sometimes a deadpan stat, sometimes a backhanded "
    "compliment, sometimes mock sympathy, sometimes a rhetorical question, and sometimes "
    "just stop after the facts with no kicker at all. Never attach a name, handle, "
    'byline, attribution, or signature (no "— Name", no @handle, no 🤖 sign-off) — the '
    "Discord username already shows who is speaking. This is a STYLE instruction ONLY — "
    "it never licenses adding any fact, stat, line value, or detail beyond the ones you "
    "are given."
)

# The repeated-pick ROLE line (the event-specific context) + the INVARIANT guard
# tail, split out from the swappable voice (260627-xbb). The leading voice sentence
# is supplied by the active personality at compose time; the ROLE + guard below are
# byte-identical for every voice (the facts-first / anti-hallucination guarantee).
REPEATED_PICK_ROLE = (
    "You are given ONE fact about a player's repeated pick: who they are, the team + "
    "side they keep taking, and for how many weeks running."
)

REPEATED_PICK_GUARD = (
    "STATE THAT FACT FIRST — name the player, the team and side, and the streak "
    "length — THEN add a short playful roast. Flavor must NEVER replace the fact; a "
    "reader who sees only your line must still know who did what. Reply with ONE "
    "short line and at most one emoji. Use ONLY the fact you are given: NEVER invent "
    "any stat or detail beyond it."
)

# Back-compat: the composed default (sarcastic) repeated-pick prompt. The pure
# ``phrase_pattern`` accepts an optional resolved voice and defaults to this voice
# when none is supplied — phrase()/phrase_pattern() NEVER read the DB (the active
# voice is resolved upstream in the db_bridge seam by the caller).
REPEATED_PICK_SYSTEM_PROMPT = compose_prompt(
    PERSONALITIES[DEFAULT_PERSONALITY_ID], REPEATED_PICK_ROLE, REPEATED_PICK_GUARD
)


async def _chat_completion(
    *,
    system_content: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    log_prefix: str,
) -> str | None:
    """Shared best-effort POST/parse for the chat-completions endpoint.

    POSTs an OpenAI-compatible chat-completions request to
    ``{llm_api_server}/chat/completions`` with bearer auth, ``system_content`` as
    the system message (already fully built by the caller — this helper NEVER
    appends any directive of its own) and ``user_content`` as the user message, the
    mandatory ``chat_template_kwargs.enable_thinking = False``, the supplied
    sampling knobs, and a ~10s timeout. Returns the stripped assistant content on a
    clean 200 with non-empty content; returns ``None`` (logging a structlog warning
    under ``log_prefix``) on missing config, any exception/timeout, a non-200, or
    empty/whitespace content. NEVER raises. Shared by :func:`phrase` (chat decode +
    closer-variety) and :func:`classify` (deterministic JSON decode, no directive).
    """
    server = settings.llm_api_server
    model = settings.llm_api_model
    key = settings.llm_api_key
    if not server or not model or not key:
        return None  # feature disabled / not configured

    url = f"{server}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        # HARD RULE — without this the served gemma model returns empty content.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=body, headers=headers)
        if response.status_code != 200:
            logger.warning(f"{log_prefix}_non_200", status_code=response.status_code)
            return None
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        content = (content or "").strip()
        if not content:
            logger.warning(f"{log_prefix}_empty_content")
            return None
        return content
    except Exception:
        # Best-effort: a timeout / connection error / malformed body must never
        # raise out of here (the caller falls back to the deterministic line).
        logger.warning(f"{log_prefix}_failed", exc_info=True)
        return None


async def phrase(fact_text: str, *, system_prompt: str) -> str | None:
    """Phrase ``fact_text`` into one chat line under ``system_prompt``, or ``None``.

    The general best-effort core: appends the style-only ``_CLOSER_VARIETY``
    directive AFTER the caller's guard-bearing prompt (facts-first still leads; the
    caller's argument is never mutated), then delegates to :func:`_chat_completion`
    with the chat-quip sampling knobs (small ``max_tokens``, higher temperature +
    nucleus ``top_p``). Returns the stripped assistant content on a clean 200 with
    non-empty content; returns ``None`` on a missing config, any exception/timeout,
    a non-200, or empty/whitespace content. NEVER raises.
    """
    system_content = f"{system_prompt} {_CLOSER_VARIETY}"
    return await _chat_completion(
        system_content=system_content,
        user_content=fact_text,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
        log_prefix="llm_phrase",
    )


async def classify(user_content: str, *, system_prompt: str) -> str | None:
    """Extract a compact JSON object from ``user_content`` under ``system_prompt``.

    The deterministic EXTRACTION seam for the @mention Q&A classifier (260709-k5w).
    Unlike :func:`phrase` it does NOT append ``_CLOSER_VARIETY`` — the system
    message is the caller's ``system_prompt`` VERBATIM — decodes
    near-deterministically (``temperature`` 0.0 / ``top_p`` 1.0) so the same
    question maps to the same JSON, and uses a JSON-appropriate ``max_tokens`` so a
    compact object is not truncated to unparseable. Keeps the SAME best-effort
    contract as :func:`phrase` (returns ``None`` on missing config / any exception /
    non-200 / empty content, NEVER raises) and the SAME mandatory
    ``chat_template_kwargs.enable_thinking = False``. Returns the raw stripped
    content string (the caller parses the JSON), or ``None``.
    """
    return await _chat_completion(
        system_content=system_prompt,
        user_content=user_content,
        max_tokens=_CLASSIFY_MAX_TOKENS,
        temperature=_CLASSIFY_TEMPERATURE,
        top_p=_CLASSIFY_TOP_P,
        log_prefix="llm_classify",
    )


async def phrase_pattern(fact_text: str, *, voice: str | None = None) -> str | None:
    """Phrase a repeated-pick fact into one roast line, or ``None`` on any failure.

    Thin wrapper (260627-nef): composes the system prompt from the active ``voice``
    preamble + the repeated-pick ROLE + the invariant guard, then delegates to
    :func:`phrase`. This function is PURE — it never reads the DB; the active voice
    must be resolved upstream in the db_bridge seam and passed in by the caller
    (``commentary.build_lock_commentary``). When ``voice`` is omitted it defaults to
    the sarcastic voice, so ``phrase_pattern(fact)`` reproduces the prior behavior
    and the existing callers/tests are unchanged.
    """
    active_voice = voice if voice is not None else PERSONALITIES[DEFAULT_PERSONALITY_ID]
    system_prompt = compose_prompt(active_voice, REPEATED_PICK_ROLE, REPEATED_PICK_GUARD)
    return await phrase(fact_text, system_prompt=system_prompt)
