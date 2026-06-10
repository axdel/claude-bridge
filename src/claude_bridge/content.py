"""Policy-free parser for Anthropic media content blocks.

Normalizes ``image`` and ``document`` blocks into a single ``MediaSource``
value object so every provider DERIVES the source shape from one place instead
of re-encoding it. This module is a leaf: it parses and normalizes only — it
emits no warnings, performs no provider encoding, and applies no size policy.
Callers (provider mappers) decide all policy.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

_DEFAULT_MEDIA_TYPE = "application/octet-stream"


@dataclasses.dataclass(frozen=True)
class MediaSource:
    """The normalized origin of one Anthropic media block.

    ``source_kind`` records which Anthropic source variant produced this:
    ``base64`` (inline bytes in ``data``), ``url`` (fetchable ``url``),
    ``file`` (a Files-API ``file_id`` the bridge cannot resolve — no bytes),
    or ``unknown`` (malformed or out-of-scope source). Exactly one of ``data``
    or ``url`` is populated for ``base64``/``url``; both are ``None`` otherwise.
    """

    kind: Literal["image", "document"]
    media_type: str
    data: str | None
    url: str | None
    filename: str | None
    source_kind: Literal["base64", "url", "file", "unknown"]


def parse_media_source(block: dict) -> MediaSource:
    """Normalize an Anthropic image/document block into a MediaSource. Never raises.

    Reads ``block["type"]`` (image|document), ``block["source"]``, and the
    optional ``block["title"]`` (used as a filename hint for documents). A source
    whose ``type`` is not base64/url/file — or a missing source — normalizes to
    ``source_kind="unknown"`` so callers can degrade observably rather than
    forwarding corrupt bytes.
    """
    kind: Literal["image", "document"] = "document" if block.get("type") == "document" else "image"
    source = block.get("source") or {}
    media_type = source.get("media_type", _DEFAULT_MEDIA_TYPE)
    filename = block.get("title")

    source_type = source.get("type")
    if source_type == "base64":
        return MediaSource(kind, media_type, source.get("data", ""), None, filename, "base64")
    if source_type == "url":
        return MediaSource(kind, media_type, None, source.get("url", ""), filename, "url")
    if source_type == "file":
        return MediaSource(kind, media_type, None, None, filename, "file")
    return MediaSource(kind, media_type, None, None, filename, "unknown")
