"""Client-facing HTTP/SSE wire for the proxy.

Owns the bytes written back to the Claude Code client: minimal HTTP/1.1 and SSE
response framing, the ``StreamOutcome`` result type shared by the sync and streaming
paths, and the translation of upstream/provider errors into Anthropic-format error
envelopes and operator-safe log summaries. A pure leaf — imports no other proxy module.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")

# HTTP status → Anthropic error type (docs.anthropic.com/en/api/errors). Anything
# unmapped falls back to ``api_error`` so a novel upstream status never crashes.
_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


class ClientDisconnected(Exception):
    """Raised when a write/drain fails because the client closed the connection."""


@dataclass(frozen=True)
class StreamOutcome:
    """Result of a streaming response.

    ``status`` is the client-visible HTTP status; ``tokens_in``/``tokens_out`` are the
    usage parsed from the terminal stream event; ``error`` flags a semantic failure
    (truncated stream, provider error event) that the outer status cannot express —
    it is set even when the client already received a 200.
    """

    status: int
    tokens_in: int = 0
    tokens_out: int = 0
    error: bool = False


async def safe_write(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write data and drain, raising ClientDisconnected on broken pipe."""
    try:
        writer.write(data)
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
        raise ClientDisconnected from exc


def write_sse_headers(writer: asyncio.StreamWriter) -> None:
    """Write HTTP/1.1 200 headers for an SSE stream."""
    writer.write(b"HTTP/1.1 200 OK\r\n")
    writer.write(b"Content-Type: text/event-stream\r\n")
    writer.write(b"Cache-Control: no-cache\r\n")
    writer.write(b"Connection: keep-alive\r\n")
    writer.write(b"\r\n")


def write_response(
    writer: asyncio.StreamWriter,
    status: int,
    body: bytes,
    extra_headers: list[tuple[str, str]] | None = None,
) -> None:
    """Write a minimal HTTP/1.1 response."""
    reasons = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        413: "Payload Too Large",
        502: "Bad Gateway",
    }
    reason = reasons.get(status, "Error")
    writer.write(f"HTTP/1.1 {status} {reason}\r\n".encode())
    writer.write(b"Content-Type: application/json\r\n")
    writer.write(f"Content-Length: {len(body)}\r\n".encode())
    for key, value in extra_headers or []:
        writer.write(f"{key}: {value}\r\n".encode())
    writer.write(b"Connection: close\r\n")
    writer.write(b"\r\n")
    writer.write(body)


def anthropic_error_body(status_code: int, message: str) -> bytes:
    """Build an Anthropic-shaped error envelope for a status code and message."""
    error_type = _ANTHROPIC_ERROR_TYPES.get(status_code, "api_error")
    return json.dumps(
        {"type": "error", "error": {"type": error_type, "message": message}}
    ).encode()


def provider_error_message(raw_body: bytes) -> str:
    """Extract a human-readable message from an upstream provider error body.

    Understands the OpenAI ``{"error": {"message": ...}}`` shape and degrades to the
    decoded body (truncated) when the payload is not the expected JSON.
    """
    try:
        parsed = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        return raw_body.decode("utf-8", errors="replace")[:500]
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return error.get("message") or error.get("type") or json.dumps(error)[:500]
    if isinstance(error, str):
        return error
    if isinstance(parsed, dict) and parsed.get("message"):
        return parsed["message"]
    return raw_body.decode("utf-8", errors="replace")[:500]


def safe_log_excerpt(value: object, *, limit: int = 200) -> str:
    """Return a bounded single-line excerpt for provider-controlled diagnostics."""
    cleaned = _CONTROL_CHARS.sub(" ", str(value)).strip()
    if len(cleaned) > limit:
        return cleaned[:limit] + "..."
    return cleaned


def provider_error_log_summary(raw_body: bytes) -> str:
    """Return an operator-safe provider error summary without raw body fallback."""
    try:
        parsed = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        return f"unparseable provider error body ({len(raw_body)}B)"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        details = []
        for key in ("type", "code", "message"):
            value = error.get(key)
            if value:
                details.append(f"{key}={safe_log_excerpt(value)}")
        if details:
            return f"provider error ({', '.join(details)}, body={len(raw_body)}B)"
    if isinstance(error, str):
        return f"provider error (error={safe_log_excerpt(error)}, body={len(raw_body)}B)"
    if isinstance(parsed, dict) and parsed.get("message"):
        message = safe_log_excerpt(parsed["message"])
        return f"provider error (message={message}, body={len(raw_body)}B)"
    return f"provider error body without message ({len(raw_body)}B)"
