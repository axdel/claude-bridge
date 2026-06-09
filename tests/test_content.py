"""Tests for the shared media-source parser (`content.py`).

`parse_media_source` is a pure, policy-free normalizer: it reads an Anthropic
``image``/``document`` block and reports the source shape without emitting
warnings, encoding for any provider, or applying size policy. Every expected
value below is derived from the Anthropic Messages API block schema
(base64 → media_type+data, url → url, file → file_id-only, title → filename),
never from running the parser under test.
"""

from __future__ import annotations

import dataclasses

import pytest

from claude_bridge.content import MediaSource, parse_media_source


class TestParseImageSource:
    """Image blocks: base64 carries data, url carries a fetchable address."""

    def test_base64_image_extracts_media_type_and_data(self):
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        }
        result = parse_media_source(block)
        assert result == MediaSource(
            kind="image",
            media_type="image/png",
            data="AAAA",
            url=None,
            filename=None,
            source_kind="base64",
        )

    def test_url_image_extracts_url_and_leaves_data_none(self):
        block = {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.png"},
        }
        result = parse_media_source(block)
        assert result.kind == "image"
        assert result.source_kind == "url"
        assert result.url == "https://example.com/cat.png"
        assert result.data is None

    def test_image_without_title_has_no_filename(self):
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "ZZZZ"},
        }
        result = parse_media_source(block)
        assert result.filename is None


class TestParseDocumentSource:
    """Document blocks: PDF base64, url, and file_id-only (unresolvable) sources."""

    def test_base64_pdf_extracts_data_and_pdf_media_type(self):
        block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBER"},
        }
        result = parse_media_source(block)
        assert result == MediaSource(
            kind="document",
            media_type="application/pdf",
            data="JVBER",
            url=None,
            filename=None,
            source_kind="base64",
        )

    def test_document_title_becomes_filename(self):
        block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBER"},
            "title": "quarterly-report.pdf",
        }
        result = parse_media_source(block)
        assert result.filename == "quarterly-report.pdf"

    def test_url_document_extracts_url(self):
        block = {
            "type": "document",
            "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        }
        result = parse_media_source(block)
        assert result.kind == "document"
        assert result.source_kind == "url"
        assert result.url == "https://example.com/doc.pdf"
        assert result.data is None

    def test_file_source_has_no_data_or_url(self):
        # file_id is not resolvable by the bridge — no bytes to forward.
        block = {
            "type": "document",
            "source": {"type": "file", "file_id": "file_011CNha8iCJcU1wXNR6q4V8w"},
        }
        result = parse_media_source(block)
        assert result.source_kind == "file"
        assert result.data is None
        assert result.url is None


class TestParseDegradedSource:
    """Malformed or out-of-scope sources normalize to ``unknown`` — never raise."""

    def test_text_source_type_marked_unknown(self):
        # The Anthropic 'text' document source is out of T-002 scope (carries
        # plaintext, not base64) — it normalizes to unknown for observable
        # degradation downstream, not silent corruption.
        block = {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "hi"},
        }
        result = parse_media_source(block)
        assert result.source_kind == "unknown"
        assert result.data is None

    def test_missing_source_key_marked_unknown(self):
        result = parse_media_source({"type": "image"})
        assert result.source_kind == "unknown"
        assert result.data is None
        assert result.url is None

    def test_base64_without_media_type_defaults_to_octet_stream(self):
        # Mirrors the bridge's existing default (openai.py used
        # "application/octet-stream" for media_type-less base64 images).
        block = {"type": "image", "source": {"type": "base64", "data": "AAAA"}}
        result = parse_media_source(block)
        assert result.media_type == "application/octet-stream"


class TestMediaSourceInvariants:
    """MediaSource is an immutable value object."""

    def test_media_source_is_frozen(self):
        result = parse_media_source(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.data = "tampered"  # type: ignore[misc]
