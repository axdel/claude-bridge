"""Google Gemini provider — API key auth + Anthropic/Gemini translation (stdlib only)."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator

from claude_bridge.provider import PROVIDERS

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "gemini-2.5-pro",
    "claude-sonnet-4-6": "gemini-2.5-flash",
    "claude-haiku-4-5-20251001": "gemini-2.5-flash",
}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

_STRIPPED_KEYS = ("output_config",)


# ---------------------------------------------------------------------------
# ID encoding — round-trip Gemini tool call IDs + optional thoughtSignature
# through Anthropic's opaque tool_use ID field.
#
# Format: "call_gemini_<id>" or "call_gemini_<id>:<base64_sig>" when signature present.
# ---------------------------------------------------------------------------

_GEMINI_ID_PREFIX = "call_gemini_"


def _encode_tool_id(gemini_id: str, thought_signature: str | None = None) -> str:
    """Encode a Gemini function call ID + optional thoughtSignature into an Anthropic tool ID."""
    base = f"{_GEMINI_ID_PREFIX}{gemini_id}"
    if thought_signature:
        sig_b64 = base64.urlsafe_b64encode(thought_signature.encode()).decode()
        return f"{base}:{sig_b64}"
    return base


def _decode_tool_id(anthropic_id: str) -> tuple[str, str | None]:
    """Decode an Anthropic tool ID back to (gemini_id, thought_signature | None)."""
    if not anthropic_id.startswith(_GEMINI_ID_PREFIX):
        return anthropic_id, None
    remainder = anthropic_id[len(_GEMINI_ID_PREFIX) :]
    if ":" in remainder:
        gemini_id, sig_b64 = remainder.split(":", 1)
        try:
            sig = base64.urlsafe_b64decode(sig_b64).decode()
        except Exception:
            sig = None
        return gemini_id, sig
    return remainder, None


# ---------------------------------------------------------------------------
# Anthropic → Gemini request translation (pure function, no I/O)
# ---------------------------------------------------------------------------


def _translate_block(
    block: dict,
    tool_id_to_name: dict[str, str],
    warnings: list[str],
) -> dict | None:
    """Translate a single Anthropic content block to a Gemini part. Returns None to skip."""
    block_type = block.get("type")

    if block_type == "text":
        return {"text": block["text"]}

    if block_type == "thinking":
        warnings.append("Stripped thinking block (no Gemini equivalent)")
        return None

    if block_type == "tool_use":
        tool_id_to_name[block["id"]] = block["name"]
        fc: dict = {"name": block["name"], "args": block.get("input", {})}
        _, sig = _decode_tool_id(block["id"])
        if sig:
            return {"functionCall": fc, "thoughtSignature": sig}
        return {"functionCall": fc}

    if block_type == "tool_result":
        return _translate_tool_result(block, tool_id_to_name)

    warnings.append(f"Unknown content block type '{block_type}', converted to text")
    return {"text": str(block)}


def _translate_tool_result(block: dict, tool_id_to_name: dict[str, str]) -> dict:
    """Translate an Anthropic tool_result block to a Gemini functionResponse part."""
    tool_use_id = block.get("tool_use_id", "")
    name = tool_id_to_name.get(tool_use_id, "unknown")

    raw_content = block.get("content", "")
    if isinstance(raw_content, list):
        text_parts = [b.get("text", "") for b in raw_content if b.get("type") == "text"]
        raw_content = "\n".join(text_parts)
    raw_content = str(raw_content) if raw_content else ""

    try:
        response_obj = json.loads(raw_content)
    except (json.JSONDecodeError, ValueError):
        response_obj = {"result": raw_content}

    fr: dict = {"name": name, "response": response_obj}
    gemini_id, _ = _decode_tool_id(tool_use_id)
    if gemini_id != tool_use_id:
        fr["id"] = gemini_id
    return {"functionResponse": fr}


def _translate_messages(messages: list[dict], warnings: list[str]) -> list[dict]:
    """Translate Anthropic messages array to Gemini contents array."""
    tool_id_to_name: dict[str, str] = {}
    contents: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        parts = [
            p
            for block in content
            if (p := _translate_block(block, tool_id_to_name, warnings)) is not None
        ]
        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    return contents


def anthropic_to_gemini(request: dict) -> tuple[dict, list[str]]:
    """Translate an Anthropic Messages API request to a Gemini generateContent request.

    Returns (translated_request, warnings). Pure function — no I/O.
    """
    warnings: list[str] = []

    for key in _STRIPPED_KEYS:
        if key in request:
            warnings.append(f"Stripped unsupported key '{key}' from request")

    if "thinking" in request:
        warnings.append("Stripped 'thinking' config (no Gemini equivalent)")

    model = request.get("model", "")
    result: dict = {"model": MODEL_MAP.get(model, DEFAULT_MODEL)}

    # System prompt → system_instruction
    system = request.get("system")
    if isinstance(system, str):
        result["system_instruction"] = {"parts": [{"text": system}]}
    elif isinstance(system, list):
        joined = "\n".join(block.get("text", "") for block in system)
        result["system_instruction"] = {"parts": [{"text": joined}]}

    # Tools → function_declarations
    if "tools" in request:
        result["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    }
                    for t in request["tools"]
                ]
            }
        ]

    result["contents"] = _translate_messages(request.get("messages", []), warnings)

    if _has_cache_control(request):
        warnings.append(
            "Stripped cache_control hints (no provider equivalent — caching is automatic)"
        )

    return result, warnings


def _has_cache_control(request: dict) -> bool:
    """Return True if any part of the request contains cache_control hints."""
    system = request.get("system")
    if isinstance(system, list) and any("cache_control" in b for b in system):
        return True
    if any("cache_control" in t for t in request.get("tools", [])):
        return True
    for msg in request.get("messages", []):
        content = msg.get("content", [])
        if isinstance(content, list) and any("cache_control" in b for b in content):
            return True
    return False


# ---------------------------------------------------------------------------
# Gemini → Anthropic response translation (pure function, no I/O)
# ---------------------------------------------------------------------------


def gemini_to_anthropic(response: dict) -> dict:
    """Translate a Gemini generateContent response to Anthropic Messages format.

    Pure function — no I/O.
    """
    raise NotImplementedError("T-003")


# ---------------------------------------------------------------------------
# Concrete Provider implementation
# ---------------------------------------------------------------------------


class GeminiProvider:
    """Google Gemini provider implementing the Provider protocol."""

    name = "gemini"

    def __init__(self) -> None:
        model = DEFAULT_MODEL
        self.endpoint = f"{_BASE_URL}/models/{model}"

    async def authenticate(self) -> dict[str, str]:
        """Return Gemini auth header. Requires GEMINI_API_KEY env var."""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            msg = (
                "GEMINI_API_KEY environment variable is required "
                "for the Gemini provider but was not set or is empty."
            )
            raise ValueError(msg)
        return {"x-goog-api-key": api_key}

    def translate_request(self, anthropic_req: dict) -> tuple[dict, list[str]]:
        """Translate Anthropic Messages request to Gemini generateContent request."""
        return anthropic_to_gemini(anthropic_req)

    def translate_response(self, provider_resp: dict) -> dict:
        """Translate Gemini response to Anthropic Messages response."""
        return gemini_to_anthropic(provider_resp)

    def translate_stream(self, raw_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """Translate Gemini SSE stream to Anthropic SSE events."""
        raise NotImplementedError("Gemini stream translation not yet implemented")


PROVIDERS["gemini"] = GeminiProvider
