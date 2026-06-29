"""Shared annotated field types for the response schemas.

Centralizes wire-serialization quirks that must be identical across every schema
that exposes the field, so the rule lives in exactly one place.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import PlainSerializer

# Discord snowflake ids are 64-bit integers (e.g. 302924379799683073) that exceed
# JavaScript's Number.MAX_SAFE_INTEGER (2**53). If emitted as a JSON number, the
# browser's JSON.parse rounds them to the nearest double — e.g. ...683073 arrives
# as ...683100 — which corrupts the Discord CDN avatar URL built from the id and
# 404s every real user's avatar. So the id MUST cross the wire as a STRING (the
# same reason the Discord/Twitter APIs return id fields as strings). The value is
# only ever interpolated into a URL or null-checked on the client; no arithmetic
# is done on it, so a string is strictly safe. See quick task 260629-kor.
#
# The field is still CONSTRUCTED from an int | None (the source is a BigInteger
# column), and only serialized to a string on the JSON output path.
DiscordId = Annotated[
    int | None,
    PlainSerializer(
        lambda v: None if v is None else str(v),
        return_type=(str | None),
    ),
]
