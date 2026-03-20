"""Generic auth utilities — JWT decode and expiry checks (stdlib only)."""

from __future__ import annotations

import base64
import json
import time


def decode_jwt_exp(token: str) -> float:
    """Decode a JWT payload and extract the ``exp`` claim.

    No cryptographic verification — only used to read the expiry timestamp
    from an already-trusted token.
    """
    payload_b64 = token.split(".")[1]
    # Add padding so urlsafe_b64decode doesn't choke on missing '='
    payload_bytes = base64.urlsafe_b64decode(payload_b64 + "==")
    payload = json.loads(payload_bytes)
    return payload["exp"]


def is_token_expired(token: str, margin_seconds: int = 30) -> bool:
    """Return True if *token* expires within *margin_seconds* from now."""
    exp = decode_jwt_exp(token)
    return time.time() + margin_seconds >= exp
