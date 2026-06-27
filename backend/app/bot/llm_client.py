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

from app.config import settings

logger = structlog.get_logger(__name__)

# Keep the line short — this is one chat quip, not an essay.
_MAX_TOKENS = 80
_TEMPERATURE = 0.9
_TIMEOUT_SECONDS = 10.0

REPEATED_PICK_SYSTEM_PROMPT = (
    "You are the snarky house bot for an NFL pick'em league. Given ONE fact about "
    "a player's repeated pick, reply with ONE short, playful roast line. Use at "
    "most one emoji. NEVER invent any stat or detail beyond the fact you are given."
)


async def phrase(fact_text: str, *, system_prompt: str) -> str | None:
    """Phrase ``fact_text`` into one chat line under ``system_prompt``, or ``None``.

    The general best-effort core: POSTs an OpenAI-compatible chat-completions
    request to ``{llm_api_server}/chat/completions`` with bearer auth, the
    SUPPLIED ``system_prompt`` as the system message and ``fact_text`` as the user
    message, the mandatory ``chat_template_kwargs.enable_thinking = False``, a small
    ``max_tokens`` and a ~10s timeout. Returns the stripped assistant content on a
    clean 200 with non-empty content; returns ``None`` (logging a structlog
    warning) on a missing config, any exception/timeout, a non-200, or
    empty/whitespace content. NEVER raises.
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fact_text},
        ],
        # HARD RULE — without this the served gemma model returns empty content.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": _MAX_TOKENS,
        "temperature": _TEMPERATURE,
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=body, headers=headers)
        if response.status_code != 200:
            logger.warning("llm_phrase_non_200", status_code=response.status_code)
            return None
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        content = (content or "").strip()
        if not content:
            logger.warning("llm_phrase_empty_content")
            return None
        return content
    except Exception:
        # Best-effort: a timeout / connection error / malformed body must never
        # raise out of here (the caller falls back to the deterministic line).
        logger.warning("llm_phrase_failed", exc_info=True)
        return None


async def phrase_pattern(fact_text: str) -> str | None:
    """Phrase a repeated-pick fact into one roast line, or ``None`` on any failure.

    Back-compat thin wrapper (260627-nef): delegates to :func:`phrase` with
    :data:`REPEATED_PICK_SYSTEM_PROMPT`. Signature and observable behavior are
    unchanged — ``commentary.build_lock_commentary`` and the existing tests call
    it as before.
    """
    return await phrase(fact_text, system_prompt=REPEATED_PICK_SYSTEM_PROMPT)
