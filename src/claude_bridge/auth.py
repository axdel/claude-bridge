"""Generic auth utilities — JWT decode and expiry checks (stdlib only)."""

from __future__ import annotations

import base64
import binascii
import json
import time


def decode_jwt_exp(token: str) -> float:
    """Decode a JWT payload and extract the ``exp`` claim.

    No cryptographic verification — only used to read the expiry timestamp
    from an already-trusted token.

    Raises:
        ValueError: If the token is malformed (missing segments, bad base64,
            invalid JSON, or missing ``exp`` claim).
    """
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("Invalid JWT: missing payload segment")
    try:
        payload_bytes = base64.urlsafe_b64decode(parts[1] + "==")
    except binascii.Error as exc:
        raise ValueError("Invalid JWT: payload is not valid base64") from exc
    try:
        payload = json.loads(payload_bytes)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Invalid JWT: payload is not valid JSON") from exc
    if "exp" not in payload:
        raise ValueError("Invalid JWT: missing 'exp' claim")
    return payload["exp"]


def is_token_expired(token: str, margin_seconds: int = 30) -> bool:
    """Return True if *token* expires within *margin_seconds* from now.

    Raises:
        ValueError: If the token cannot be decoded (wraps decode errors with context).
    """
    try:
        exp = decode_jwt_exp(token)
    except ValueError as exc:
        raise ValueError(f"Cannot check token expiry: {exc}") from exc
    return time.time() + margin_seconds >= exp
