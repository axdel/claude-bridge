#!/usr/bin/env python3
"""One-off diagnostic: does the Codex (chatgpt.com) backend accept non-text content?

Gating probe for the nontext-content-translation branch (T-001). Sends a minimal
Responses request carrying an ``input_image`` (and best-effort ``input_file``) to the
codex_oauth endpoint, reusing the bridge's own auth and the exact request shape the
proxy forwards (Authorization bearer + Content-Type: application/json).

NOT part of the package or the test suite — a standalone diagnostic. Never prints
base64 payloads or bearer tokens; error bodies are truncated and base64-stripped.

Usage:
    uv run python scripts/probe_codex_modality.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import urllib.error
import urllib.request

# Reuse the bridge's auth + endpoint — the probe must mirror the real request path.
from claude_bridge.providers.openai import (
    _DEFAULT_AUTH_PATH,
    DEFAULT_MODEL,
    OpenAIProvider,
)

# 1x1 transparent PNG — the smallest valid image input.
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)

_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def _redact(text: str, limit: int = 300) -> str:
    """Truncate an upstream message and strip any long base64 run before display."""
    stripped = _BASE64_RUN.sub("<base64-redacted>", text)
    return stripped[:limit]


def _minimal_pdf_b64() -> str:
    """Build a tiny structurally-valid PDF and return its base64 (no data: prefix)."""
    import base64

    body = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    return base64.b64encode(body).decode()


def _request_body(content_parts: list[dict]) -> dict:
    """Mirror the shape anthropic_to_openai emits (model/reasoning/store/stream/input)."""
    return {
        "model": DEFAULT_MODEL,
        "reasoning": {"effort": "low"},
        "store": False,
        "stream": True,
        "instructions": "You are a helpful assistant.",
        "input": [{"role": "user", "content": content_parts}],
    }


def _classify(status: int, message: str) -> str:
    """Map an upstream (status, message) to a modality verdict."""
    if status == 200:
        return "ACCEPTED — backend processed the request"
    if status in (401, 403):
        return "AUTH — credential/scope problem, NOT a modality signal"
    if status == 404:
        return "ENDPOINT — codex /responses path narrower than public API"
    low = message.lower()
    if status == 400 and any(
        token in low
        for token in ("image", "input_image", "input_file", "vision", "multimodal", "content")
    ):
        return "REJECTED — backend does not accept this content type"
    return f"INCONCLUSIVE — HTTP {status}, message did not name a modality cause"


def _post(endpoint: str, auth_headers: dict[str, str], payload: dict) -> tuple[int, str]:
    """POST the payload exactly as proxy._do_provider_request does. Returns (status, message)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint, data=data, method="POST")  # noqa: S310
    req.add_header("Content-Type", "application/json")
    for key, value in auth_headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            resp.read(256)  # drain a little of the stream; we only need the status
            return resp.status, "ok"
    except urllib.error.HTTPError as exc:  # non-2xx
        return exc.code, _redact(exc.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        return 0, f"network error: {exc.reason}"


async def main() -> int:
    if not _DEFAULT_AUTH_PATH.exists():
        print(f"NO-CREDENTIAL: {_DEFAULT_AUTH_PATH} absent.")
        print(
            "Verdict: default codex_oauth input_modalities={'text'} (conservative); unblock T-006."
        )
        return 2

    provider = OpenAIProvider(auth_mode="codex_oauth")
    auth_headers = await provider.authenticate()
    endpoint = provider.endpoint
    print(f"endpoint: {endpoint}")
    print(f"model: {DEFAULT_MODEL}\n")

    probes = {
        "image (input_image, 1x1 png)": [
            {"type": "input_text", "text": "Reply with the single word OK."},
            {"type": "input_image", "image_url": f"data:image/png;base64,{_PNG_1X1_B64}"},
        ],
        "document (input_file, minimal pdf)": [
            {"type": "input_text", "text": "Reply with the single word OK."},
            {
                "type": "input_file",
                "filename": "probe.pdf",
                "file_data": f"data:application/pdf;base64,{_minimal_pdf_b64()}",
            },
        ],
        "baseline (text only)": [
            {"type": "input_text", "text": "Reply with the single word OK."},
        ],
    }

    for label, parts in probes.items():
        status, message = _post(endpoint, auth_headers, _request_body(parts))
        print(f"[{label}] HTTP {status} -> {_classify(status, message)}")
        if status not in (200,):
            print(f"    upstream: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
